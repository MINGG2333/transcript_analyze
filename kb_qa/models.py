from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib


@dataclass
class Segment:
    __slots__ = (
        "segment_id", "text", "start_time", "end_time",
        "source_type", "file_path", "video_path", "video_title",
        "anchor_name", "live_id", "video_datetime",
    )
    segment_id: str
    text: str
    start_time: float
    end_time: float
    source_type: str  # speech | danmaku
    file_path: str
    video_path: str
    video_title: str
    anchor_name: str
    live_id: str
    video_datetime: str  # ISO8601

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Segment":
        return Segment(**data)

    @property
    def video_dt(self) -> datetime:
        return datetime.fromisoformat(self.video_datetime)

    @property
    def source_label(self) -> str:
        return "主播讲话" if self.source_type == "speech" else "观众弹幕"

    @property
    def hhmmss(self) -> str:
        secs = max(0, int(self.start_time))
        return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"

    @property
    def absolute_time(self) -> str:
        return (self.video_dt + timedelta(seconds=self.start_time)).isoformat(timespec="seconds")


def build_segment_id(live_id: str, source_type: str, start_time: float, text: str) -> str:
    digest = hashlib.sha1(f"{live_id}|{source_type}|{start_time:.3f}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{live_id}_{source_type}_{digest}"

