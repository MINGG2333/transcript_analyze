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

    def expand_context(self, segment_ids: list[str], context_window: int = 3, logger=None) -> list[Segment]:
        result: dict[str, Segment] = {}
        expansion_count = 0
        for i, sid in enumerate(segment_ids):
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
                    if near.segment_id not in result:
                        expansion_count += 1
                    result[near.segment_id] = near
            if logger and (i + 1) % max(1, len(segment_ids) // 5) == 0:
                logger.info(f"  扩展进度: {i + 1}/{len(segment_ids)}, 当前结果集大小: {len(result)}")
        
        if logger:
            logger.info(f"  扩展统计: 基础段数={len(segment_ids)}, 扩展后总段数={len(result)}, 新增段数={expansion_count}")
        
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

    def retrieve(self, query: str, top_k: int = 30) -> tuple[list[str], list[float]]:
        """Retrieve top_k relevant segments with scores.
        Returns: (segment_ids, scores) where scores are cosine similarities [0, 1]
        """
        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        out_ids: list[str] = []
        out_scores: list[float] = []
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for i, meta in enumerate(metadatas):
            sid = meta.get("segment_id")
            if sid:
                # Convert cosine distance to similarity: similarity = 1 - distance
                # cosine distance in [0, 2], similarity in [-1, 1], normalize to [0, 1]
                distance = distances[i] if i < len(distances) else 2.0
                similarity = max(0.0, (1.0 - distance) / 2.0)  # normalize to [0, 1]
                out_ids.append(sid)
                out_scores.append(similarity)
        return out_ids, out_scores


class BM25Index:
    def __init__(self, segments: dict[str, Segment]):
        self.segment_ids = list(segments.keys())
        self.corpus = [self._tokenize(segments[sid].text) for sid in self.segment_ids]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 中文场景下用字符级分词，避免外部分词依赖
        return [x for x in text.strip() if not x.isspace()]

    def retrieve(self, query: str, top_k: int = 30) -> tuple[list[str], list[float]]:
        """Retrieve top_k relevant segments with BM25 scores.
        Returns: (segment_ids, scores)
        """
        if not self.bm25:
            return [], []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out_ids = []
        out_scores = []
        for i in ranked_indices:
            if scores[i] > 0:
                out_ids.append(self.segment_ids[i])
                out_scores.append(scores[i])
        return out_ids, out_scores

