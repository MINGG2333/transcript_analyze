from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Iterable

from .models import Segment, build_segment_id

SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
LRC_TIME_RE = re.compile(r"\[(\d{2}):(\d{2}(?:\.\d{1,3})?)\]\s*(.*)")
LIVEID_RE = re.compile(r"LiveId@(\d+)")
TS_RE = re.compile(r"_(\d{14})(?:_info)?\.")


def safe_name(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*？]', "_", s)


def seconds_from_srt_match(m: re.Match[str], base_idx: int) -> float:
    return (
        int(m.group(base_idx)) * 3600
        + int(m.group(base_idx + 1)) * 60
        + int(m.group(base_idx + 2))
        + int(m.group(base_idx + 3)) / 1000
    )


def extract_live_datetime(metadata_path: Path, video_path: Path) -> datetime:
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            ctime = data.get("ctime")
            if ctime:
                return datetime.fromtimestamp(int(ctime) / 1000)
        except Exception:
            pass

    m = TS_RE.search(video_path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return datetime.min


def infer_subtitle_path(record: dict, output_root: Path) -> Path:
    user = safe_name(record.get("user_name", "unknown"))
    video_stem = safe_name(Path(record["video_path"]).stem)
    return output_root / user / video_stem / f"{video_stem}_subtitles.srt"


def parse_srt(srt_path: Path, record: dict) -> list[Segment]:
    if not srt_path.exists():
        return []
    raw = srt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []

    live_dt = extract_live_datetime(Path(record.get("metadata_path", "")), Path(record["video_path"]))
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [x.strip("\ufeff").strip() for x in block.splitlines() if x.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1] if SRT_TIME_RE.search(lines[1]) else lines[0]
        m = SRT_TIME_RE.search(time_line)
        if not m:
            continue
        text_lines = lines[2:] if time_line == lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        if not text:
            continue
        start = seconds_from_srt_match(m, 1)
        end = seconds_from_srt_match(m, 5)
        sid = build_segment_id(record["live_id"], "speech", start, text)
        segments.append(
            Segment(
                segment_id=sid,
                text=text,
                start_time=start,
                end_time=end,
                source_type="speech",
                file_path=str(srt_path),
                video_path=record["video_path"],
                video_title=record.get("title", ""),
                anchor_name=record.get("user_name", ""),
                live_id=record["live_id"],
                video_datetime=live_dt.isoformat(),
            )
        )
    return segments


def parse_lrc(lrc_path: Path, record: dict) -> list[Segment]:
    if not lrc_path.exists():
        return []
    live_dt = extract_live_datetime(Path(record.get("metadata_path", "")), Path(record["video_path"]))
    segments: list[Segment] = []
    for line in lrc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        m = LRC_TIME_RE.match(line)
        if not m:
            continue
        text = m.group(3).strip()
        if not text:
            continue
        start = int(m.group(1)) * 60 + float(m.group(2))
        sid = build_segment_id(record["live_id"], "danmaku", start, text)
        segments.append(
            Segment(
                segment_id=sid,
                text=text,
                start_time=start,
                end_time=start + 5.0,
                source_type="danmaku",
                file_path=str(lrc_path),
                video_path=record["video_path"],
                video_title=record.get("title", ""),
                anchor_name=record.get("user_name", ""),
                live_id=record["live_id"],
                video_datetime=live_dt.isoformat(),
            )
        )
    return segments


def load_records(records_path: Path) -> dict:
    data = json.loads(records_path.read_text(encoding="utf-8"))
    # 兼容 key=live_id 的结构
    for live_id, rec in data.items():
        rec.setdefault("live_id", live_id)
    return data


def collect_segments(records_path: Path, output_root: Path) -> Iterable[Segment]:
    records = load_records(records_path)
    for rec in records.values():
        if not rec.get("video_path"):
            continue
        srt_path = infer_subtitle_path(rec, output_root)
        lrc_path = Path(rec.get("danmu_path", "")) if rec.get("danmu_path") else None

        for seg in parse_srt(srt_path, rec):
            yield seg
        if lrc_path:
            for seg in parse_lrc(lrc_path, rec):
                yield seg

