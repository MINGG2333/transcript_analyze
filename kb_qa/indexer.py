"""
Index modules: SegmentStore, BM25Index, VectorIndex.

VectorIndex 的 embedding 模型加载策略：
  1. 先检查本地 HuggingFace 缓存，如果模型已存在，直接走离线模式（不尝试联网下载）
  2. 如果模型不存在，做快速网络连通性检测（3s 超时）：
     - 网络可达：在线加载
     - 网络不可达：立即提示用户手动下载模型，不等默认的重试超时
"""
from __future__ import annotations

import gc
import os
import socket
import warnings
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from .models import Segment

# ── Embedding 模型加载（带智能回退） ─────────────────────────────────────

_MODEL_CACHE_NAME = "models--" + "shibing624--text2vec-base-chinese"
_MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / _MODEL_CACHE_NAME

_MODEL_DOWNLOAD_HELP = f"""
Embedding 模型 'shibing624/text2vec-base-chinese' 未在本地缓存中找到，
且当前服务器无法连接到 huggingface.co（网络不可达）。

请先在**可以访问 huggingface.co 的机器**上下载模型：

    pip install huggingface_hub
    huggingface-cli download shibing624/text2vec-base-chinese

然后将本地缓存中的整个目录复制到服务器的以下位置：

    {_MODEL_CACHE_DIR}/

完成后重新运行即可。

如果模型已存在于缓存中但路径不同，请检查 ~/.cache/huggingface/hub/ 目录。
"""


def _check_hf_cache(model_name: str) -> bool:
    """检查模型是否已在本地 HuggingFace 缓存中。"""
    cache_name = "models--" + model_name.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    return cache_dir.exists()


def _check_network(host: str = "huggingface.co", timeout: int = 3) -> bool:
    """快速检测网络连通性（短超时）。"""
    try:
        socket.create_connection((host, 443), timeout=timeout)
        return True
    except OSError:
        return False


def _create_embedding_fn(model_name: str):
    """加载 embedding 模型，优先使用本地缓存，避免长时间等待联网超时。"""
    # ── 1. 如果模型已在缓存中，直接走离线模式 ──
    if _check_hf_cache(model_name):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print(f"模型 '{model_name}' 已在本地缓存中，以离线模式加载...")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    # ── 2. 模型不在缓存中，先快速检测网络 ──
    if not _check_network():
        print(_MODEL_DOWNLOAD_HELP)
        raise RuntimeError(
            f"Embedding 模型 '{model_name}' 未在本地缓存中找到，且网络不可达。\n"
            f"请按照上方提示手动下载模型到缓存目录后重试。"
        )

    # ── 3. 网络可达，在线加载 ──
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    except Exception as e:
        err_str = str(e).lower()
        is_network_error = (
            "network is unreachable" in err_str
            or "connection" in err_str
            or "max retries" in err_str
            or "cannot connect" in err_str
            or "timeout" in err_str
            or "reset" in err_str
            or "eof" in err_str
        )
        if is_network_error:
            # 在线失败 + 网络错误 → 切离线模式重试一次
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            print(f"在线加载失败（网络错误），切换到离线模式重试...")
            print(f"缓存路径：{_MODEL_CACHE_DIR}")
            try:
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Embedding 模型 '{model_name}' 加载失败。\n"
                    f"在线模式和离线模式均尝试过。请参考上方提示手动下载模型到缓存目录。"
                ) from e2
        else:
            raise  # 非网络错误，直接报错


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
        # 始终从 segments 重建 by_live_source 索引，确保与当前数据一致
        # 不依赖 JSON 中可能过期或为空的历史索引数据
        self._rebuild_live_source_index()

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
            key = seg.live_id
            idx[key].append(seg.segment_id)
        for key, ids in idx.items():
            ids.sort(key=lambda sid: self.segments[sid].start_time)
        self.by_live_source = defaultdict(list, idx)

    def expand_context(self, segment_ids: list[str], context_window: int = 3, logger=None) -> list[Segment]:
        result: dict[str, Segment] = {}
        expansion_count = 0
        seen_live_ids: dict[str, int] = {}
        for i, sid in enumerate(segment_ids):
            seg = self.segments.get(sid)
            if not seg:
                continue
            key = seg.live_id
            seq = self.by_live_source.get(key, [])
            if key not in seen_live_ids:
                seen_live_ids[key] = len(seq)
            if not seq:
                result[sid] = seg
                continue
            try:
                pos = seq.index(sid)
            except ValueError:
                if logger:
                    logger.warning(f"  sid={sid} 在 live_id={key} 的序列中找不到（seq长度={len(seq)}, seq前5={seq[:5]}）")
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
        
        if logger:
            logger.info(f"  扩展统计: 基础段数={len(segment_ids)}, 扩展后总段数={len(result)}, 新增段数={expansion_count}")
            logger.debug("  各live_id在by_live_source中的序列长度:")
            for live_id, seq_len in sorted(seen_live_ids.items()):
                logger.debug(f"    live_id={live_id}: seq_len={seq_len}")

        ordered = sorted(
            result.values(),
            key=lambda s: (s.video_datetime or "", s.video_title or s.live_id, s.live_id, s.start_time),
        )
        if logger:
            logger.debug("  扩展后片段排序前30:")
            for idx, seg in enumerate(ordered[:30], start=1):
                text_preview = seg.text.replace("\n", " ")[:60]
                logger.debug(
                    f"    {idx:02d}. video_title={seg.video_title!r}, live_id={seg.live_id}, source={seg.source_type}, "
                    f"start={seg.start_time:.3f}, hhmmss={seg.hhmmss}, id={seg.segment_id}, text={text_preview!r}"
                )

        return ordered


class VectorIndex:
    def __init__(self, db_dir: Path, embedding_model: str):
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.embedding_fn = _create_embedding_fn(embedding_model)
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
            # 每批处理后释放 batch 对象，防止内存累积
            del batch
            gc.collect()
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

