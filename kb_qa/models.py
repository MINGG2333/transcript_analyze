from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib


@dataclass
class Segment:
    segment_id: str
    text: str
    start_time: float
    end_time: float
    source_type: str  # speech | danmaku -> participant
    file_path: str
    video_path: str  # -> vtt_path
    video_title: str  # -> interview_title
    anchor_name: str  # -> participant_name
    live_id: str  # -> interview_id
    video_datetime: str  # -> interview_date

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
        return self.source_type

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

