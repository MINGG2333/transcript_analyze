#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from kb_qa.cli import setup_logger
from kb_qa.qa import VideoKnowledgeQA


def safe_name(value: str, max_len: int = 120) -> str:
    if not value:
        return "unnamed"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return safe[:max_len].rstrip("._-") or "unnamed"


def load_questions(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_qa(args: argparse.Namespace, logger: Any) -> VideoKnowledgeQA:
    return VideoKnowledgeQA(
        records_path=Path(args.records),
        subtitle_root=Path(args.subtitle_root),
        kb_dir=Path(args.kb_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        api_base=args.api_base,
        api_key=args.api_key,
        logger=logger,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="按问题组批量执行问答，为每个访谈生成统一CSV和问题级别的引用JSON")
    parser.add_argument("--input-csv", default="interview_protocol_CN.csv", help="原始问题CSV文件")
    parser.add_argument("--output-base", default="res", help="结果输出根目录")
    parser.add_argument("--records", default="interview_records.json", help="访谈记录JSON路径")
    parser.add_argument("--subtitle-root", default="interview_output", help="字幕输出根目录")
    parser.add_argument("--kb-dir", default="interview_knowledge_db", help="知识库目录")
    parser.add_argument("--embedding-model", default="shibing624/text2vec-base-chinese", help="向量模型")
    parser.add_argument("--llm-model", default="deepseek-v4-flash", help="问答LLM模型名")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="LLM API base url")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"), help="LLM API key")
    parser.add_argument("--vector-top-k", type=int, default=600, help="向量检索top_k")
    parser.add_argument("--bm25-top-k", type=int, default=600, help="BM25检索top_k")
    parser.add_argument("--context-window", type=int, default=6, help="上下文扩展窗口")
    parser.add_argument("--analysis-batch-size", type=int, default=100, help="分析批次大小")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    parser.add_argument("--limit-groups", type=int, default=0, help="只处理前N个问题组，0为全部")
    parser.add_argument("--no-skip-existing", action="store_true", help="不跳过已存在的输出目录")
    args = parser.parse_args()

    logger = setup_logger(debug=args.debug)
    qa = build_qa(args, logger)
    input_path = Path(args.input_csv)
    output_base = Path(args.output_base)

    rows, fieldnames = load_questions(input_path)
    if "answer" not in fieldnames:
        fieldnames.append("answer")
    if "citation_path" not in fieldnames:
        fieldnames.append("citation_path")

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source = row.get("source", "")
        groups.setdefault(source, []).append(row)

    output_base.mkdir(parents=True, exist_ok=True)

    # 累积所有访谈的结果：live_id -> {source -> answers...}
    all_interview_data: dict[str, dict[str, Any]] = {}
    processed_groups = 0
    for source in sorted(groups):
        source_dir = output_base / safe_name(source)
        if not args.no_skip_existing and source_dir.exists():
            logger.info(f"跳过已存在的 source={source}")
            continue
        if args.limit_groups and processed_groups >= args.limit_groups:
            break
        group_rows = sorted(groups[source], key=lambda r: r.get("question_id", ""))
        logger.info(f"开始处理问题组 source={source}，共 {len(group_rows)} 个问题")
        logger.debug(f"读取到的source={source}，共 {len(group_rows)} 个问题:")
        for i, row in enumerate(group_rows, start=1):
            logger.debug(f"  {i}. question_id={row.get('question_id', '')}, question_text={row.get('question_text', '')}")

        questions = [
            {
                "question_id": row.get("question_id", ""),
                "question_text": row.get("question_text", ""),
            }
            for row in group_rows
        ]

        result = qa.ask_group(
            questions=questions,
            source=source,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            context_window=args.context_window,
            vector_score_threshold=0.332,
            bm25_score_threshold=15.0,
            analysis_batch_size=args.analysis_batch_size,
        )

        for interview in result["interview_results"]:
            live_id = interview.get("live_id", "unknown")
            if live_id not in all_interview_data:
                all_interview_data[live_id] = {
                    "live_id": live_id,
                    "video_title": interview.get("video_title", ""),
                    "anchor_name": interview.get("anchor_name", ""),
                    "video_datetime": interview.get("video_datetime", ""),
                    "answers": {},
                }
            all_interview_data[live_id]["answers"][source] = interview.get("answers", [])

        processed_groups += 1

        # 每处理完一个source，就更新所有访谈的CSV
        for live_id, interview_info in sorted(all_interview_data.items(), key=lambda x: x[1].get("video_datetime", "")):
            video_title = interview_info.get("video_title", "")
            interview_name = safe_name(f"{live_id}-{video_title}")
            interview_csv_path = output_base / f"{interview_name}.csv"
            interview_rows = []

            for src in sorted(interview_info["answers"].keys()):
                for answer_item in interview_info["answers"][src]:
                    qid = answer_item.get("question_id", "")
                    answer_text = answer_item.get("answer", "")
                    citations = answer_item.get("citations", []) or []
                    citation_json_path = ""

                    if qid:
                        source_dir = output_base / safe_name(src)
                        question_dir = source_dir / qid
                        question_dir.mkdir(parents=True, exist_ok=True)
                        citation_file = question_dir / f"{interview_name}.json"

                        citation_payload = {
                            "live_id": live_id,
                            "video_title": video_title,
                            "anchor_name": interview_info.get("anchor_name", ""),
                            "video_datetime": interview_info.get("video_datetime", ""),
                            "source": src,
                            "question_id": qid,
                            "question_text": answer_item.get("question_text", ""),
                            "answer": answer_text,
                            "evidence": answer_item.get("evidence", []) or [],
                            "citations": citations,
                            "useful_segment_count": len(citations),
                        }
                        citation_file.write_text(json.dumps(citation_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        citation_json_path = os.path.relpath(citation_file, Path.cwd())

                    interview_rows.append(
                        {
                            "question_id": qid,
                            "question_text": answer_item.get("question_text", ""),
                            "source": src,
                            "answer": answer_text,
                            "citation_path": citation_json_path,
                        }
                    )

            # 合并现有数据
            merged_dict = {}
            if interview_csv_path.exists():
                existing_rows, _ = load_questions(interview_csv_path)
                for row in existing_rows:
                    qid = row.get("question_id", "")
                    if qid:
                        merged_dict[qid] = row
            for row in interview_rows:
                qid = row.get("question_id", "")
                if qid:
                    merged_dict[qid] = row  # 更新或添加

            final_rows = list(merged_dict.values())
            write_csv(interview_csv_path, final_rows, fieldnames)
            logger.info(f"已更新访谈CSV: {interview_csv_path}")

    logger.success(f"批量组问答完成，结果输出到 {output_base}")


if __name__ == "__main__":
    main()
