#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser(description="从qa_archive元数据更新res目录")
    parser.add_argument("--archive-dir", default="interview_knowledge_db/qa_archive", help="qa_archive目录")
    parser.add_argument("--output-base", default="res", help="结果输出根目录")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # 累积所有访谈的结果：live_id -> {source -> answers...}
    all_interview_data: dict[str, dict[str, Any]] = {}

    for json_file in archive_dir.glob("*.json"):
        print(f"处理 {json_file}")
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        source = data.get("source", "")
        if not source:
            continue

        for interview in data.get("interview_results", []):
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

    # 为每个访谈生成统一CSV和问题级引用JSON
    fieldnames = ["question_id", "question_text", "source", "answer", "citation_path"]
    for live_id, interview_info in sorted(all_interview_data.items(), key=lambda x: x[1].get("video_datetime", "")):
        video_title = interview_info.get("video_title", "")
        interview_name = safe_name(f"{live_id}-{video_title}")
        interview_csv_path = output_base / f"{interview_name}.csv"
        interview_rows = []

        for source in sorted(interview_info["answers"].keys()):
            for answer_item in interview_info["answers"][source]:
                qid = answer_item.get("question_id", "")
                answer_text = answer_item.get("answer", "")
                citations = answer_item.get("citations", []) or []
                citation_json_path = ""

                if qid:
                    source_dir = output_base / safe_name(source)
                    question_dir = source_dir / qid
                    question_dir.mkdir(parents=True, exist_ok=True)
                    citation_file = question_dir / f"{interview_name}.json"

                    citation_payload = {
                        "live_id": live_id,
                        "video_title": video_title,
                        "anchor_name": interview_info.get("anchor_name", ""),
                        "video_datetime": interview_info.get("video_datetime", ""),
                        "source": source,
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
                        "source": source,
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
        print(f"已更新访谈CSV: {interview_csv_path}")

    print(f"从archive更新res完成，结果输出到 {output_base}")


if __name__ == "__main__":
    main()