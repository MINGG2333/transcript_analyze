#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import Counter


def generate_interview_records():
    interview_dir = Path("/mnt/zhitainew/ttt/interview")
    records = {}
    
    for vtt_file in interview_dir.glob("*.vtt"):
        if vtt_file.name.endswith(".vtt"):
            # Extract interview id from filename, e.g., 访谈1.vtt -> 访谈1
            interview_id = vtt_file.stem
            # Extract date if present, e.g., 访谈10-20260412 -> 2026-04-12
            date_match = None
            if "-" in interview_id and len(interview_id.split("-")[-1]) == 8:
                date_str = interview_id.split("-")[-1]
                if date_str.isdigit():
                    date_match = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            
            records[interview_id] = {
                "interview_id": interview_id,
                "vtt_path": str(vtt_file),
                "video_path": str(vtt_file.with_suffix(".mp4")) if vtt_file.with_suffix(".mp4").exists() else "",
                "title": f"{interview_id} 访谈记录",
                "interview_date": date_match or "2026-04-01",  # Default date
                "participants": ["访谈者", "被访者"]  # Placeholder
            }
    
    return records


def validate_records(records: dict) -> None:
    """校验 interview_id 的唯一性和格式一致性。"""
    # 1. interview_id 的 key 唯一性（dict 本身保证，但打印警告以防覆盖）
    keys = list(records.keys())
    assert len(keys) == len(set(keys)), "interview_records.json 中存在重复的 key！"
    
    # 2. 检查每个记录的 interview_id 是否与 key 一致
    mismatches = []
    for key, info in records.items():
        lid = info.get("interview_id", "")
        if lid != key:
            mismatches.append((key, lid))
    if mismatches:
        print(f"[WARNING] 有 {len(mismatches)} 个记录的 interview_id 与 key 不一致: {mismatches}")
    
    # 3. 检查 interview_id 的命名格式是否统一
    ids = list(records.keys())
    prefixes = Counter(i.split("-")[0] for i in ids)
    duplicates = {p: c for p, c in prefixes.items() if c > 1}
    if duplicates:
        print(f"[WARNING] interview_id 前缀存在重复风险: {duplicates}")
        for p, c in duplicates.items():
            colliding = [i for i in ids if i.split("-")[0] == p]
            print(f"   前缀 '{p}' 出现在 {c} 个 ID 中: {colliding}")
            print(f"   ⚠ 在按 '-' 截断取前缀的脚本中，这些 ID 可能被混淆")
    
    print(f"[INFO] 校验通过: {len(ids)} 个 interview_id 均唯一")


if __name__ == "__main__":
    records = generate_interview_records()
    validate_records(records)
    with open("interview_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Generated records for {len(records)} interviews")
