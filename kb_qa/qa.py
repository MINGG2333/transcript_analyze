from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any
import uuid

from .indexer import BM25Index, SegmentStore, VectorIndex
from .models import Segment
from .parsers import collect_segments


class VideoKnowledgeQA:
    def __init__(
        self,
        records_path: Path,
        subtitle_root: Path,
        kb_dir: Path,
        embedding_model: str,
        llm_model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        logger=None,
    ):
        self.records_path = records_path
        self.subtitle_root = subtitle_root
        self.kb_dir = kb_dir
        self.llm_model = llm_model
        self.logger = logger

        self.store = SegmentStore(kb_dir / "segment_store.json")
        self.store.load()
        self.vector = VectorIndex(kb_dir / "chroma_db", embedding_model=embedding_model)
        self.bm25 = BM25Index(self.store.segments)
        self.api_base = api_base
        self.api_key = api_key
        self.client = None

    def build_or_update(self) -> dict[str, int]:
        if self.logger:
            self.logger.info("开始构建或更新知识库")
            self.logger.info(f"记录文件路径: {self.records_path}")
            self.logger.info(f"字幕根目录: {self.subtitle_root}")
            self.logger.info(f"知识库目录: {self.kb_dir}")
        
        all_segments = list(collect_segments(self.records_path, self.subtitle_root))
        if self.logger:
            self.logger.info(f"解析得到 {len(all_segments)} 个片段")

            # 统计不同类型的片段
            speech_count = sum(1 for seg in all_segments if seg.source_type == "speech")
            danmaku_count = sum(1 for seg in all_segments if seg.source_type == "danmaku")
            self.logger.info(f"片段类型统计: 主播讲话({speech_count}) | 观众弹幕({danmaku_count})")

            self.logger.info("=== 示例片段（随机挑选） ===")
            sample_segments = []
            if speech_count:
                sample_segments.append(random.choice([seg for seg in all_segments if seg.source_type == "speech"]))
            if danmaku_count:
                sample_segments.append(random.choice([seg for seg in all_segments if seg.source_type == "danmaku"]))
            if len(sample_segments) < 2 and all_segments:
                sample_segments.append(random.choice(all_segments))

            for seg in sample_segments:
                self.logger.info(f"[{seg.source_label}] 完整信息:")
                self.logger.info(f"  segment_id: {seg.segment_id}")
                self.logger.info(f"  source_type: {seg.source_type}")
                self.logger.info(f"  file_path: {seg.file_path}")
                self.logger.info(f"  video_path: {seg.video_path}")
                self.logger.info(f"  video_title: {seg.video_title}")
                self.logger.info(f"  anchor_name: {seg.anchor_name}")
                self.logger.info(f"  live_id: {seg.live_id}")
                self.logger.info(f"  video_datetime: {seg.video_datetime}")
                self.logger.info(f"  start_time: {seg.start_time} → end_time: {seg.end_time}")
                self.logger.info(f"  video_offset: {seg.hhmmss}")
                self.logger.info(f"  text: {seg.text!r}")

            if danmaku_count == 0:
                self.logger.warning("未解析到观众弹幕片段，请检查danmu_path和LRC文件格式是否正确")
            
            # 随机选取一些记录，详细统计其中的srt和lrc文件
            self.logger.info("=== 随机文件详细统计 ===")
            from .parsers import load_records, infer_subtitle_path, parse_srt, parse_lrc
            records = load_records(self.records_path)
            valid_records = [rec for rec in records.values() if rec.get("video_path")]
            
            if valid_records:
                sample_records = random.sample(valid_records, min(2, len(valid_records)))
                for rec in sample_records:
                    srt_path = infer_subtitle_path(rec, self.subtitle_root)
                    lrc_path = Path(rec.get("danmu_path", "")) if rec.get("danmu_path") else None
                    
                    self.logger.info(f"\n记录 LiveId={rec['live_id']} ({rec.get('title', 'N/A')})")
                    
                    if srt_path.exists():
                        srt_segs = parse_srt(srt_path, rec)
                        self.logger.info(f"  字幕文件 (SRT): {Path(srt_path)}")
                        self.logger.info(f"    片段数: {len(srt_segs)}")
                        if srt_segs:
                            sample = random.choice(srt_segs)
                            self.logger.info(f"    示例: [{sample.hhmmss}] {sample.anchor_name} | {sample.text[:80]!r}")
                    else:
                        self.logger.info(f"  字幕文件 (SRT): 不存在 ({srt_path})")
                    
                    if lrc_path and lrc_path.exists():
                        lrc_segs = parse_lrc(lrc_path, rec)
                        self.logger.info(f"  弹幕文件 (LRC): {Path(lrc_path)}")
                        self.logger.info(f"    片段数: {len(lrc_segs)}")
                        if lrc_segs:
                            sample = random.choice(lrc_segs)
                            self.logger.info(f"    示例: [{sample.hhmmss}] {sample.anchor_name} | {sample.text[:80]!r}")
                    else:
                        lrc_name = Path(rec.get("danmu_path", "unknown")) if rec.get("danmu_path") else "unknown"
                        self.logger.info(f"  弹幕文件 (LRC): 不存在或未配置 ({lrc_name})")
        
        changed = self.store.upsert_many(all_segments)
        if self.logger:
            self.logger.info(f"检测到 {len(changed)} 个新或更新的片段")
        
        if changed:
            if self.logger:
                self.logger.info("开始更新向量索引（这可能需要几分钟，请耐心等待...）")
            self.vector.upsert(changed, logger=self.logger)
            if self.logger:
                self.logger.info("开始保存片段存储")
            self.store.save()
            if self.logger:
                self.logger.success("片段存储保存完成")
                self.logger.info("重建BM25索引")
            self.bm25 = BM25Index(self.store.segments)
        else:
            if self.logger:
                self.logger.info("没有新的片段，无需更新索引")
        
        stat = {
            "parsed_segments": len(all_segments),
            "updated_segments": len(changed),
            "total_segments": len(self.store.segments),
        }
        if self.logger:
            self.logger.success(f"构建完成: {stat}")
        
        return stat

    def retrieve(self, question: str, vector_top_k: int = 40, bm25_top_k: int = 40, context_window: int = 3) -> list[Segment]:
        vector_ids = self.vector.retrieve(question, top_k=vector_top_k)
        bm25_ids = self.bm25.retrieve(question, top_k=bm25_top_k)
        merged_ids = list(dict.fromkeys(vector_ids + bm25_ids))
        return self.store.expand_context(merged_ids, context_window=context_window)

    def _build_judge_prompt(self, question: str, segments: list[Segment]) -> str:
        lines: list[str] = []
        for s in segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; 直播时间={s.video_datetime}; "
                f"视频内时间={s.hhmmss}; 标题={s.video_title}; 主播={s.anchor_name}; 内容={s.text}"
            )
        context = "\n".join(lines)
        return (
            "你是严谨的证据型问答助手。请根据候选片段回答用户问题，不能臆造。\n"
            f"用户问题：{question}\n\n"
            "候选片段：\n"
            f"{context}\n\n"
            "请输出JSON对象，格式为：\n"
            '{"answer":"...","evidence":[{"segment_id":"...","reason":"..."}]}\n'
            "要求：\n"
            "1) evidence必须只使用给定segment_id；\n"
            "2) 尽量覆盖所有相关证据，按时间顺序选择；\n"
            "3) 如果证据不足，answer里明确说明不确定。\n"
            "仅输出JSON，不要额外文本。"
        )

    def ask(self, question: str, vector_top_k: int = 40, bm25_top_k: int = 40, context_window: int = 3) -> dict[str, Any]:
        self._ensure_client()
        candidates = self.retrieve(question, vector_top_k, bm25_top_k, context_window)
        if not candidates:
            result = {
                "question": question,
                "answer": "未检索到可用片段，无法回答该问题。",
                "citations": [],
                "retrieved_count": 0,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._archive(result)
            return result

        prompt = self._build_judge_prompt(question, candidates)
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)

        evidence = parsed.get("evidence", [])
        selected_ids = [e.get("segment_id") for e in evidence if e.get("segment_id")]
        id_to_seg = {s.segment_id: s for s in candidates}

        citations = []
        for idx, sid in enumerate(selected_ids, start=1):
            seg = id_to_seg.get(sid)
            if not seg:
                continue
            citations.append(
                {
                    "citation_id": f"#{idx}",
                    "segment_id": sid,
                    "source_type": seg.source_label,
                    "quoted_text": seg.text,
                    "video_offset": seg.hhmmss,
                    "absolute_time": seg.absolute_time,
                    "source_file": seg.file_path,
                    "video_path": seg.video_path,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "live_id": seg.live_id,
                }
            )

        result = {
            "question": question,
            "answer": parsed.get("answer", "").strip() or "模型未返回有效答案。",
            "citations": citations,
            "retrieved_count": len(candidates),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._archive(result)
        return result

    def _archive(self, result: dict[str, Any]) -> None:
        archive_dir = self.kb_dir / "qa_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        (archive_dir / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "缺少 openai 依赖，请先执行: python -m pip install -r requirements_kb_qa.txt"
            ) from exc

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base) if self.api_base else OpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI(base_url=self.api_base) if self.api_base else OpenAI()

