#!/usr/bin/env python3
import json
import os
from pathlib import Path

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

if __name__ == "__main__":
    records = generate_interview_records()
    with open("interview_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Generated records for {len(records)} interviews")