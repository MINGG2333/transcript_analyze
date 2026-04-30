from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from .models import Segment


class SegmentStore:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.segments: dict[str, Segment] = {}
        self.by_live_source: dict[str, list[str]] = defaultdict(list)

    def load(self) -> None:
        if not self.store_path.exists():
            return
        raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.segments = {k: Segment.from_dict(v) for k, v in raw.get("segments", {}).items()}
        self.by_live_source = defaultdict(list, raw.get("by_live_source", {}))

    def save(self) -> None:
        payload = {
            "segments": {k: v.to_dict() for k, v in self.segments.items()},
            "by_live_source": dict(self.by_live_source),
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert_many(self, segments: Iterable[Segment]) -> list[Segment]:
        changed: list[Segment] = []
        for seg in segments:
            old = self.segments.get(seg.segment_id)
            if old and old.text == seg.text and old.start_time == seg.start_time:
                continue
            self.segments[seg.segment_id] = seg
            changed.append(seg)

        if changed:
            self._rebuild_live_source_index()
        return changed

    def _rebuild_live_source_index(self) -> None:
        idx: dict[str, list[str]] = defaultdict(list)
        for seg in self.segments.values():
            key = f"{seg.live_id}::{seg.source_type}"
            idx[key].append(seg.segment_id)
        for key, ids in idx.items():
            ids.sort(key=lambda sid: self.segments[sid].start_time)
        self.by_live_source = defaultdict(list, idx)

    def expand_context(self, segment_ids: list[str], context_window: int = 3) -> list[Segment]:
        result: dict[str, Segment] = {}
        for sid in segment_ids:
            seg = self.segments.get(sid)
            if not seg:
                continue
            key = f"{seg.live_id}::{seg.source_type}"
            seq = self.by_live_source.get(key, [])
            if not seq:
                result[sid] = seg
                continue
            try:
                pos = seq.index(sid)
            except ValueError:
                result[sid] = seg
                continue
            start = max(0, pos - context_window)
            end = min(len(seq), pos + context_window + 1)
            for near_sid in seq[start:end]:
                near = self.segments.get(near_sid)
                if near:
                    result[near.segment_id] = near
        return sorted(result.values(), key=lambda s: (s.video_datetime, s.start_time))


class VectorIndex:
    def __init__(self, db_dir: Path, embedding_model: str):
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="video_segments",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, segments: list[Segment], batch_size: int = 400, logger=None) -> None:
        total = len(segments)
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            self.collection.upsert(
                ids=[x.segment_id for x in batch],
                documents=[x.text for x in batch],
                metadatas=[
                    {
                        "segment_id": x.segment_id,
                        "start_time": x.start_time,
                        "end_time": x.end_time,
                        "source_type": x.source_type,
                        "anchor_name": x.anchor_name,
                        "video_title": x.video_title,
                        "live_id": x.live_id,
                        "video_datetime": x.video_datetime,
                        "file_path": x.file_path,
                        "video_path": x.video_path,
                    }
                    for x in batch
                ],
            )
            if logger:
                processed = min(i + batch_size, total)
                progress_pct = (processed * 100) // total
                logger.info(f"向量索引更新进度: {processed}/{total} ({progress_pct}%)")

    def retrieve(self, query: str, top_k: int = 30) -> list[str]:
        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas"],
        )
        out: list[str] = []
        for meta in result.get("metadatas", [[]])[0]:
            sid = meta.get("segment_id")
            if sid:
                out.append(sid)
        return out


class BM25Index:
    def __init__(self, segments: dict[str, Segment]):
        self.segment_ids = list(segments.keys())
        self.corpus = [self._tokenize(segments[sid].text) for sid in self.segment_ids]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 中文场景下用字符级分词，避免外部分词依赖
        return [x for x in text.strip() if not x.isspace()]

    def retrieve(self, query: str, top_k: int = 30) -> list[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.segment_ids[i] for i in ranked if scores[i] > 0]

