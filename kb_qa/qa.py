from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Optional
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
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
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

    def retrieve(
        self,
        question: str,
        vector_top_k: int = 40,
        bm25_top_k: int = 40,
        context_window: int = 3,
        vector_score_threshold: float = 0.3,
        bm25_score_threshold: float = 2.0,
        max_base_segments: Optional[int] = None,
        max_expanded_segments: Optional[int] = None,
    ) -> tuple[list[Segment], dict[str, Any]]:
        if self.logger:
            self.logger.info(f"[1/5] 开始从向量索引检索，top_k={vector_top_k}, score_threshold={vector_score_threshold}")
        vector_ids, vector_scores = self.vector.retrieve(question, top_k=vector_top_k)
        # Filter by score threshold
        vector_filtered = [(sid, score) for sid, score in zip(vector_ids, vector_scores) if score >= vector_score_threshold]
        if self.logger:
            self.logger.info(f"[1/5] 向量检索完成，得到 {len(vector_ids)} 个候选段，过滤后 {len(vector_filtered)} 个")
            self.logger.debug("[1/5] 向量检索按相似度排序的前200条结果：")
            for rank, (sid, score) in enumerate(sorted(vector_filtered, key=lambda x: x[1], reverse=True)[:200], start=1):
                seg = self.store.segments.get(sid)
                text_snippet = seg.text.replace("\n", " ").strip() if seg else "<missing segment>"
                self.logger.debug(
                    f"  {rank:03d}. {sid} score={score:.6f} text={text_snippet}"
                )

        if self.logger:
            self.logger.info("[2/5] 开始BM25查询改写，辅助检索语句更聚焦")
        bm25_query, bm25_refinement = self._refine_bm25_query(question)
        if self.logger:
            self.logger.info(f"[2/5] BM25查询改写完成，使用查询：{bm25_query}")

        if self.logger:
            self.logger.info(f"[3/5] 开始从BM25索引检索，top_k={bm25_top_k}, score_threshold={bm25_score_threshold}")
        bm25_ids, bm25_scores = self.bm25.retrieve(bm25_query, top_k=bm25_top_k)
        # Filter by score threshold
        bm25_filtered = [(sid, score) for sid, score in zip(bm25_ids, bm25_scores) if score >= bm25_score_threshold]
        if self.logger:
            self.logger.info(f"[3/5] BM25检索完成，得到 {len(bm25_ids)} 个候选段，过滤后 {len(bm25_filtered)} 个")
            self.logger.debug("[3/5] BM25检索按分数排序的前200条结果：")
            for rank, (sid, score) in enumerate(sorted(bm25_filtered, key=lambda x: x[1], reverse=True)[:200], start=1):
                seg = self.store.segments.get(sid)
                text_snippet = seg.text.replace("\n", " ").strip() if seg else "<missing segment>"
                self.logger.debug(
                    f"  {rank:03d}. {sid} score={score:.6f} text={text_snippet[:120]}"
                )

        if self.logger:
            self.logger.info(f"[4/5] 合并向量和BM25结果")

        bm25_max_score = max((score for _, score in bm25_filtered), default=0.0)
        merged_dict: dict[str, dict[str, float]] = {}

        for index, (sid, score) in enumerate(vector_filtered):
            merged_dict.setdefault(sid, {"vector_score": 0.0, "bm25_score": 0.0})
            merged_dict[sid]["vector_score"] = score
            merged_dict[sid]["vector_rank"] = index + 1

        for index, (sid, score) in enumerate(bm25_filtered):
            merged_dict.setdefault(sid, {"vector_score": 0.0, "bm25_score": 0.0})
            merged_dict[sid]["bm25_score"] = score
            merged_dict[sid]["bm25_rank"] = index + 1

        for sid, values in merged_dict.items():
            vector_norm = values.get("vector_score", 0.0)
            bm25_norm = values.get("bm25_score", 0.0)
            if bm25_max_score > 0:
                bm25_norm = bm25_norm / bm25_max_score
            values["combined_score"] = max(vector_norm, bm25_norm)

        merged_ids = list(merged_dict.keys())
        raw_merged_count = len(merged_ids)
        if self.logger:
            self.logger.info(f"[4/5] 合并完成，共 {raw_merged_count} 个唯一段")

        if max_base_segments is not None and raw_merged_count > max_base_segments:
            sorted_ids = sorted(
                merged_ids,
                key=lambda x: merged_dict[x]["combined_score"],
                reverse=True,
            )[:max_base_segments]
            merged_ids = sorted_ids
            if self.logger:
                self.logger.info(f"[4/5] 基础段数超出限制，截断至 {max_base_segments}")

        if self.logger:
            self.logger.info(f"[5/5] 开始上下文扩展，context_window={context_window}")
        candidates = self.store.expand_context(merged_ids, context_window=context_window, logger=self.logger)
        if self.logger:
            self.logger.info(f"[5/5] 上下文扩展完成，得到 {len(candidates)} 个扩展后的片段")

        truncated = False
        if max_expanded_segments is not None and len(candidates) > max_expanded_segments:
            candidates = candidates[:max_expanded_segments]
            truncated = True
            if self.logger:
                self.logger.info(f"[5/5] 扩展段数超出限制，截断至 {max_expanded_segments}")

        stats = {
            "vector_hits_raw": len(vector_ids),
            "vector_hits_filtered": len(vector_filtered),
            "vector_score_threshold": vector_score_threshold,
            "bm25_query": bm25_query,
            "bm25_refinement": bm25_refinement,
            "bm25_hits_raw": len(bm25_ids),
            "bm25_hits_filtered": len(bm25_filtered),
            "bm25_score_threshold": bm25_score_threshold,
            "raw_merged_ids": raw_merged_count,
            "used_base_ids": len(merged_ids),
            "candidate_count": len(candidates),
            "context_window": context_window,
            "max_base_segments": max_base_segments,
            "max_expanded_segments": max_expanded_segments,
            "truncated": truncated,
        }
        return candidates, stats

    def _build_judge_prompt(self, question: str, segments: list[Segment]) -> str:
        lines: list[str] = []
        for s in segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; 直播时间={s.video_datetime}; "
                f"视频内时间={s.hhmmss}; 标题={s.video_title}; 用户名={s.anchor_name}; 内容={s.text}"
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
            "1) answer字段中如果列出事实或时间，请尽量用 [#1]、[#2] 这样的引用标记对应 evidence 条目；\n"
            "2) evidence必须只使用给定segment_id；\n"
            "3) evidence列表应按时间顺序排列；\n"
            "4) 尽量覆盖所有相关证据；\n"
            "5) 如果证据不足，answer里明确说明不确定。\n"
            "仅输出JSON，不要额外文本。"
        )

    def _build_bm25_refinement_prompt(self, question: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "你是一个中文检索查询优化助手。请把下面的用户问题改写成一个适用于BM25检索的简洁查询，\n"
                    "去掉“为什么”“什么”等无意义疑问词和口语化表达，保留关键实体，\n"
                    "使检索更聚焦。\n"
                    f"用户问题：{question}\n"
                    "请输出JSON对象：{\"refined_query\":\"...\"}。\n"
                    "仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _refine_bm25_query(self, question: str) -> tuple[str, dict[str, Any]]:
        prompt_messages = self._build_bm25_refinement_prompt(question)
        try:
            parsed, llm_metadata = self._call_llm_json(prompt_messages, "BM25 查询改写")
            refined = (parsed.get("refined_query") or "").strip()
            if not refined:
                raise ValueError("LLM未返回 refined_query")
            return refined, llm_metadata
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"BM25查询改写失败，使用原始问题: {exc}")
            return question, {
                "description": "BM25 查询改写",
                "success": False,
                "error": str(exc),
                "refined_query": question,
                "prompt": prompt_messages[0]["content"],
                "response": "",
            }

    def _call_llm_json(self, messages: list[dict[str, str]], description: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.logger:
            self.logger.info(f"调用LLM: {description}")
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        
        # 记录LLM元数据
        llm_metadata = {
            "model": self.llm_model,
            "description": description,
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens": getattr(resp.usage, "total_tokens", 0),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prompt": messages[0]["content"] if messages and messages[0].get("role") == "user" else "",
            "response": content,
        }
        
        if self.logger:
            self.logger.info(f"  tokens: input={llm_metadata['input_tokens']}, output={llm_metadata['output_tokens']}, total={llm_metadata['total_tokens']}")
        
        try:
            parsed = json.loads(content)
            llm_metadata["success"] = True
            return parsed, llm_metadata
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning(f"LLM返回JSON解析失败，尝试提取内容: {description}")
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(content[start : end + 1])
                    llm_metadata["success"] = True
                    llm_metadata["note"] = "通过字符串提取成功解析"
                    return parsed, llm_metadata
                except json.JSONDecodeError:
                    pass
            llm_metadata["success"] = False
            llm_metadata["error"] = f"无法将LLM响应解析为JSON: {content}"
            raise RuntimeError(llm_metadata["error"])

    def _build_analysis_prompt(
        self,
        question: str,
        segments: list[Segment],
        batch_index: int,
        total_batches: int,
    ) -> list[dict[str, str]]:
        lines: list[str] = []
        for s in segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; 直播时间={s.video_datetime}; "
                f"视频内时间={s.hhmmss}; 标题={s.video_title}; 用户名={s.anchor_name}; 内容={s.text}"
            )
        context = "\n".join(lines)
        return [
            {
                "role": "user",
                "content": (
                    "你是严谨的证据分析助手。\n"
                    f"用户问题：{question}\n"
                    f"这是第 {batch_index}/{total_batches} 批候选片段，请逐条判断是否与问题相关。\n"
                    "请输出JSON对象，格式为：\n"
                    '{"useful":[{"segment_id":"...","reason":"..."}]}\n'
                    "仅将与问题直接相关且可用于回答问题的片段放入 useful。\n"
                    "不要输出多余文本。\n"
                    "候选片段：\n"
                    f"{context}" 
                ),
            }
        ]

    def _analyze_candidates(
        self,
        question: str,
        candidates: list[Segment],
        analysis_batch_size: int = 20,
    ) -> tuple[list[dict[str, Any]], list[Segment], dict[str, Any], list[dict[str, Any]]]:
        analysis: list[dict[str, Any]] = []
        useful_segments: list[Segment] = []
        total_batches = max(1, (len(candidates) + analysis_batch_size - 1) // analysis_batch_size)
        useful_ids: set[str] = set()
        batch_stats: list[dict[str, Any]] = []
        llm_calls: list[dict[str, Any]] = []  # 记录所有LLM调用的元数据

        for batch_index in range(total_batches):
            start = batch_index * analysis_batch_size
            batch = candidates[start : start + analysis_batch_size]
            if self.logger:
                self.logger.info(
                    f"分析候选批次 {batch_index + 1}/{total_batches}，包含 {len(batch)} 个片段"
                )
            
            # 第一批时打印详细日志
            if batch_index == 0 and self.logger:
                self.logger.info("=== 批次1详细信息 ===")
                for i, seg in enumerate(batch[:min(3, len(batch))], 1):
                    self.logger.info(f"  片段{i}: [{seg.segment_id}] {seg.source_label} {seg.hhmmss} {seg.text[:80]}")
                if len(batch) > 3:
                    self.logger.info(f"  ... 还有 {len(batch) - 3} 个片段")
            
            try:
                prompt_messages = self._build_analysis_prompt(question, batch, batch_index + 1, total_batches)
                parsed, llm_metadata = self._call_llm_json(
                    prompt_messages,
                    f"候选分析 {batch_index + 1}/{total_batches}",
                )
                
                # 第一批时打印prompt和response
                if batch_index == 0 and self.logger:
                    self.logger.info("=== 批次1 LLM Prompt ===")
                    self.logger.info(prompt_messages[0]["content"])
                    self.logger.info("=== 批次1 LLM Response ===")
                    self.logger.info(json.dumps(parsed, ensure_ascii=False))
                
                llm_calls.append(llm_metadata)
            except Exception as exc:
                if self.logger:
                    self.logger.error(f"分析候选批次失败，视为全部片段候选: {exc}")
                parsed = {"useful": []}
                llm_calls.append({
                    "description": f"候选分析 {batch_index + 1}/{total_batches}",
                    "success": False,
                    "error": str(exc),
                    "prompt": prompt_messages[0]["content"],
                    "response": "",
                })

            useful_items = parsed.get("useful", []) or []
            useful_batch_ids = {item.get("segment_id") for item in useful_items if item.get("segment_id")}
            useful_count = 0
            for seg in batch:
                item = next(
                    (item for item in useful_items if item.get("segment_id") == seg.segment_id),
                    None,
                )
                is_useful = item is not None
                reason = item.get("reason", "").strip() if item else ""
                analysis.append(
                    {
                        "segment_id": seg.segment_id,
                        "source_label": seg.source_label,
                        "video_title": seg.video_title,
                        "anchor_name": seg.anchor_name,
                        "video_offset": seg.hhmmss,
                        "absolute_time": seg.absolute_time,
                        "text": seg.text,
                        "useful": is_useful,
                        "reason": reason,
                    }
                )
                if is_useful and seg.segment_id not in useful_ids:
                    useful_ids.add(seg.segment_id)
                    useful_segments.append(seg)
                    useful_count += 1

            batch_stats.append(
                {
                    "batch_index": batch_index + 1,
                    "batch_size": len(batch),
                    "useful_count": useful_count,
                    "useful_ids": sorted(useful_batch_ids),
                }
            )
            if self.logger:
                self.logger.info(
                    f"批次 {batch_index + 1} 分析完成: useful={useful_count} / {len(batch)}"
                )

        summary = {
            "total_candidates": len(candidates),
            "useful_segment_count": len(useful_segments),
            "analysis_batches": batch_stats,
            "analysis_batch_size": analysis_batch_size,
        }
        return analysis, useful_segments, summary, llm_calls

    def _build_synthesis_prompt(self, question: str, useful_segments: list[Segment]) -> list[dict[str, str]]:
        lines: list[str] = []
        for s in useful_segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; 直播时间={s.video_datetime}; "
                f"视频内时间={s.hhmmss}; 标题={s.video_title}; 用户名={s.anchor_name}; 内容={s.text}"
            )
        context = "\n".join(lines)
        return [
            {
                "role": "user",
                "content": (
                    "你是严谨的证据型问答助手。请只使用下面列出的有用片段回答问题，不能臆造。\n"
                    f"用户问题：{question}\n\n"
                    "片段列表：\n"
                    f"{context}\n\n"
                    "请输出JSON对象，格式为：\n"
                    '{"answer":"...","evidence":[{"segment_id":"...","reason":"..."}]}\n'
                    "要求：\n"
                    "1) answer字段中必须引用所有有用片段，使用 [#1]、[#2] 等标记对应 evidence 条目；\n"
                    "2) evidence必须包含所有有用segment_id，按时间顺序排列；\n"
                    "3) 每个evidence条目必须说明该片段如何支持答案；\n"
                    "4) 如果片段很多，answer要全面总结所有相关信息；\n"
                    "5) 不要遗漏任何有用片段的引用。\n"
                    "仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def ask(
        self,
        question: str,
        vector_top_k: int = 1000,
        bm25_top_k: int = 1000,
        context_window: int = 3,
        vector_score_threshold: float = 0.3,
        bm25_score_threshold: float = 2.0,
        analysis_batch_size: int = 20,
    ) -> dict[str, Any]:
        self._ensure_client()
        if self.logger:
            self.logger.info(
                f"开始检索: vector_top_k={vector_top_k}, bm25_top_k={bm25_top_k}, context_window={context_window}"
            )
        candidates, stats = self.retrieve(
            question,
            vector_top_k,
            bm25_top_k,
            context_window,
            vector_score_threshold=vector_score_threshold,
            bm25_score_threshold=bm25_score_threshold,
            max_base_segments=None,  # 不限制基础段数
            max_expanded_segments=None,  # 不限制扩展段数
        )
        if self.logger:
            self.logger.info(
                f"检索结果: vector_hits_raw={stats['vector_hits_raw']}, vector_hits_filtered={stats['vector_hits_filtered']}, "
                f"bm25_hits_raw={stats['bm25_hits_raw']}, bm25_hits_filtered={stats['bm25_hits_filtered']}, "
                f"unique_base_ids={stats['raw_merged_ids']}, used_base_ids={stats['used_base_ids']}, "
                f"candidate_count={stats['candidate_count']}, truncated={stats['truncated']}"
            )

        retrieval_segments = [
            {
                "segment_id": seg.segment_id,
                "source_label": seg.source_label,
                "video_title": seg.video_title,
                "anchor_name": seg.anchor_name,
                "video_offset": seg.hhmmss,
                "absolute_time": seg.absolute_time,
                "text": seg.text,
            }
            for seg in candidates
        ]

        if not candidates:
            result = {
                "question": question,
                "answer": "未检索到可用片段，无法回答该问题。",
                "citations": [],
                "retrieved_count": 0,
                "retrieval": stats,
                "analysis_summary": {
                    "total_candidates": 0,
                    "useful_segment_count": 0,
                    "analysis_batches": [],
                    "analysis_batch_size": analysis_batch_size,
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            archive_data = {**result, "retrieval_segments": [], "analysis": [], "useful_segments": []}
            archive_path = self._archive(archive_data)
            result["archive_path"] = str(archive_path)
            return result

        if self.logger:
            self.logger.info(f"候选片段数: {len(candidates)}，开始逐批分析有用性。")
        analysis, useful_segments, analysis_summary, analysis_llm_calls = self._analyze_candidates(
            question, candidates, analysis_batch_size
        )
        if self.logger:
            self.logger.info(
                f"分析完成: useful_segments={analysis_summary['useful_segment_count']} / {analysis_summary['total_candidates']}"
            )

        final_evidence: list[dict[str, Any]] = []
        answer_text = ""
        synthesis_llm_metadata: dict[str, Any] = {}
        if not useful_segments:
            answer_text = "未找到与问题直接相关的片段，无法给出确定答案。"
            synthesis_llm_metadata = {"description": "skip_synthesis_no_useful_segments"}
        else:
            if self.logger:
                self.logger.info(
                    f"准备合成最终回答，使用 {len(useful_segments)} 个有用片段。"
                )
            try:
                parsed, synthesis_llm_metadata = self._call_llm_json(
                    self._build_synthesis_prompt(question, useful_segments),
                    "最终答案合成",
                )
                final_evidence = parsed.get("evidence", []) or []
                answer_text = parsed.get("answer", "").strip() or "模型未返回有效答案。"
            except Exception as exc:
                if self.logger:
                    self.logger.error(f"最终答案合成失败: {exc}")
                answer_text = "模型在生成最终答案时发生错误。"
                synthesis_llm_metadata = {"success": False, "error": str(exc)}

        id_to_seg = {s.segment_id: s for s in useful_segments}
        citations = []
        for idx, item in enumerate(final_evidence, start=1):
            sid = item.get("segment_id")
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
                    "reason": item.get("reason", ""),
                }
            )

        # 确保所有有用段都被引用，如果 LLM 遗漏了某些段
        cited_segment_ids = {item.get("segment_id") for item in final_evidence if item.get("segment_id")}
        missing_segments = [seg for seg in useful_segments if seg.segment_id not in cited_segment_ids]

        if missing_segments:
            if self.logger:
                self.logger.warning(f"LLM 遗漏了 {len(missing_segments)} 个有用段，将自动添加到 evidence")
            for seg in missing_segments:
                final_evidence.append({
                    "segment_id": seg.segment_id,
                    "reason": f"该片段包含与问题相关的有用信息：{seg.text[:100]}..."
                })
                citations.append(
                    {
                        "citation_id": f"#{len(citations) + 1}",
                        "segment_id": seg.segment_id,
                        "source_type": seg.source_label,
                        "quoted_text": seg.text,
                        "video_offset": seg.hhmmss,
                        "absolute_time": seg.absolute_time,
                        "source_file": seg.file_path,
                        "video_path": seg.video_path,
                        "video_title": seg.video_title,
                        "anchor_name": seg.anchor_name,
                        "live_id": seg.live_id,
                        "reason": f"该片段包含与问题相关的有用信息：{seg.text[:100]}...",
                    }
                )

        if citations and "[#" not in answer_text:
            refs = " ".join(f"[#{i}]" for i in range(1, len(citations) + 1))
            answer_text = f"{answer_text} 参考引用：{refs}"

        result = {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(candidates),
            "retrieval": stats,
            "analysis_summary": analysis_summary,
            "useful_segment_count": len(useful_segments),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        archive_data = {
            **result,
            "retrieval_segments": retrieval_segments,
            "analysis": analysis,
            "useful_segments": [
                {
                    "segment_id": seg.segment_id,
                    "source_label": seg.source_label,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "video_offset": seg.hhmmss,
                    "absolute_time": seg.absolute_time,
                    "text": seg.text,
                }
                for seg in useful_segments
            ],
            "llm_calls": {
                "analysis_batches": analysis_llm_calls,
                "synthesis": synthesis_llm_metadata,
            },
        }
        archive_path = self._archive(archive_data)
        result["archive_path"] = str(archive_path)
        return result

    def _archive(self, result: dict[str, Any]) -> Path:
        archive_dir = self.kb_dir / "qa_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        archive_path = archive_dir / name
        archive_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return archive_path

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

