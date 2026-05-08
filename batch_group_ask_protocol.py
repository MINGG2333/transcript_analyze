#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
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


def _expected_csv_stem(live_id: str, video_title: str) -> str:
    """构造期望的 CSV 文件名（不含扩展名），与 write_csv 中的命名逻辑保持一致。
    
    CSV 命名格式为: safe_name(f"{live_id}-{video_title}") + ".csv"
    """
    return safe_name(f"{live_id}-{video_title}")


def _stem_to_live_id(stem: str, records: dict) -> str | None:
    """从 csv 文件名的 stem（不含扩展名）反向匹配出 live_id。
    
    遍历所有 records 的 live_id，构造其期望的 CSV 文件名，与实际的 stem 做精确比对。
    优先尝试 records 中的 "video_title" 字段，若不存在则使用 "title" 字段。
    """
    for lid, info in records.items():
        for title_field in ("video_title", "title", "video_name"):
            vt = info.get(title_field, "")
            if not vt:
                continue
            expected = _expected_csv_stem(lid, vt)
            if stem == expected:
                return lid
    return None


def get_existing_interviews(res_dir: Path, records_path: Path) -> set[str]:
    """通过构造每个 live_id 的预期 CSV 文件名，与 res 目录下实际 CSV 文件做精确比对，
    找出已处理的访谈 ID。

    使用 records 中的 "video_title"/"title" 字段构造完整文件名，
    避免了前缀碰撞问题（如 访谈16 与 访谈16-15 是不同的访谈）。
    """
    existing = set()
    if not res_dir.exists() or not records_path.exists():
        return existing
    
    with open(records_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    csv_names = {csv_path.name for csv_path in res_dir.glob("*.csv")}
    
    for live_id, info in records.items():
        vt = info.get("video_title") or info.get("title") or ""
        if not vt:
            continue
        expected_filename = _expected_csv_stem(live_id, vt) + ".csv"
        if expected_filename in csv_names:
            existing.add(live_id)
    
    return existing


def get_new_interviews(records_path: Path, existing_interviews: set[str]) -> list[str]:
    """从 interview_records.json 找出新增访谈（不在已有 CSV 中的）"""
    if not records_path.exists():
        return []
    with open(records_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    new_ids = [lid for lid in records if lid not in existing_interviews]
    return new_ids


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
    parser.add_argument("--incremental", action="store_true", help="增量模式：只处理新增访谈，避免重复LLM调用")
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

    # 增量模式：检测新增访谈
    new_interview_ids: list[str] = []
    if args.incremental:
        records_path = Path(args.records)
        existing_interviews = get_existing_interviews(output_base, records_path)
        new_interview_ids = get_new_interviews(records_path, existing_interviews)
        if not new_interview_ids:
            logger.info("增量模式：未检测到新访谈，无需处理。")
            return
        logger.info(f"增量模式：检测到 {len(new_interview_ids)} 个新增访谈: {new_interview_ids}")
        logger.info(f"将只对这些新访谈执行问答（跳过已有 {len(existing_interviews)} 个访谈）")

    # 累积所有访谈的结果：live_id -> {source -> answers...}
    all_interview_data: dict[str, dict[str, Any]] = {}
    # 增量模式下，加载已有 CSV 的回答用于在 codebook 中提供完整数据
    if args.incremental:
        records_for_reverse = json.loads(Path(args.records).read_text('utf-8')) if Path(args.records).exists() else {}
        for csv_path in output_base.glob("*.csv"):
            try:
                interview_rows, _ = load_questions(csv_path)
                if not interview_rows:
                    continue
                stem = csv_path.stem
                live_id = _stem_to_live_id(stem, records_for_reverse)
                if not live_id:
                    continue
                vt = ""
                if live_id in records_for_reverse:
                    vt = records_for_reverse[live_id].get("video_title") or records_for_reverse[live_id].get("title") or ""
                all_interview_data[live_id] = {
                    "live_id": live_id,
                    "video_title": vt,
                    "anchor_name": "",
                    "video_datetime": "",
                    "answers": {},
                }
                for row in interview_rows:
                    src = row.get("source", "")
                    qid = row.get("question_id", "")
                    if src not in all_interview_data[live_id]["answers"]:
                        all_interview_data[live_id]["answers"][src] = []
                    all_interview_data[live_id]["answers"][src].append({
                        "question_id": qid,
                        "question_text": row.get("question_text", ""),
                        "answer": row.get("answer", ""),
                        "citations": [],
                    })
            except Exception as e:
                logger.warning(f"加载已有CSV失败: {csv_path}: {e}")

    processed_groups = 0
    for source in sorted(groups):
        source_dir = output_base / safe_name(source)
        if not args.no_skip_existing and source_dir.exists() and not args.incremental:
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

        # 增量模式：指定只处理新增访谈
        interview_ids = new_interview_ids if args.incremental and new_interview_ids else None

        result = qa.ask_group(
            questions=questions,
            source=source,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            context_window=args.context_window,
            vector_score_threshold=0.332,
            bm25_score_threshold=15.0,
            analysis_batch_size=args.analysis_batch_size,
            interview_ids=interview_ids,
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

        # 每处理完一个source，就更新CSV（增量模式下只更新新增访谈）
        for live_id, interview_info in sorted(all_interview_data.items(), key=lambda x: x[1].get("video_datetime", "")):
            # 增量模式下跳过已有访谈，只写入新增访谈的CSV/JSON
            if args.incremental and live_id not in new_interview_ids:
                continue
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
