from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
import time
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
            participant_types = {}
            for seg in all_segments:
                participant_types[seg.source_type] = participant_types.get(seg.source_type, 0) + 1
            self.logger.info(f"片段类型统计: {participant_types}")

            self.logger.info("=== 示例片段（随机挑选） ===")
            sample_segments = []
            if all_segments:
                sample_segments = random.sample(all_segments, min(3, len(all_segments)))
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

            if not all_segments:
                self.logger.warning("未解析到任何片段，请检查VTT文件和记录格式是否正确")
        
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
        bm25_score_threshold: float = 15.0,
        max_base_segments: Optional[int] = None,
        max_expanded_segments: Optional[int] = None,
    ) -> tuple[list[Segment], dict[str, Any]]:
        if self.logger:
            self.logger.info(f"[1/5] 开始从向量索引检索，top_k={vector_top_k}, score_threshold={vector_score_threshold}")
            self.logger.debug(f"向量索引检索input: {question}")
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
                    "你是一个中文检索查询优化助手。请把下面的用户问题改写成一个或多个关键词，用于BM25检索。规则：只保留问题中的核心命名实体，去掉疑问词、助词和无关表达。若该名称由重复的单一汉字组成（如“顺顺”），则需同时输出该单字和完整名称。基本原则是用这些关键词搜索到的文本范围内会有问题的答案，关键词越少越好，每增加一个关键词，需要缩小搜索范围而不是扩大。\\n"
                    f"用户问题：{question}\n"
                    "请输出JSON对象：{\"refined_query\":\"...\"}。\n"
                    "仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _refine_bm25_query(self, question: str) -> tuple[str, dict[str, Any]]:
        if self.logger:
            self.logger.debug(f"BM25查询改写input: {question}")
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

    def _call_llm_json(self, messages: list[dict[str, str]], description: str, max_retries: int = 5) -> tuple[dict[str, Any], dict[str, Any]]:
        """调用LLM API 并解析JSON响应，包含重试机制（指数退避）。
        
        Args:
            messages: 聊天消息列表
            description: 本次调用的描述
            max_retries: 最大重试次数，默认5次
            
        Returns:
            (解析后的JSON对象, LLM元数据字典)
            
        Raises:
            RuntimeError: 当所有重试均失败时抛出
        """
        last_raw = None
        
        for attempt in range(max_retries):
            try:
                if self.logger:
                    self.logger.info(f"调用LLM: {description} (尝试 {attempt+1}/{max_retries})")
                
                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                last_raw = content  # 保存原始内容
                
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
                    # JSON解析失败，如果还有重试机会则继续，否则抛出异常
                    if attempt < max_retries - 1:
                        wait_time = 5 ** attempt
                        if self.logger:
                            self.logger.warning(f"JSON解析失败 (第 {attempt+1} 次)，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        llm_metadata["success"] = False
                        llm_metadata["error"] = f"无法将LLM响应解析为JSON: {content}"
                        raise RuntimeError(llm_metadata["error"])
                        
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"LLM请求异常 (第 {attempt+1} 次): {e}")
                
                # 如果还有重试机会则继续
                if attempt < max_retries - 1:
                    wait_time = 5 ** attempt
                    if self.logger:
                        self.logger.warning(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试均失败
                    error_msg = f"LLM调用在 {max_retries} 次重试后仍失败"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

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

    def _format_segment_with_local_context(
        self,
        segment: Segment,
        context_window: int = 6,
    ) -> str:
        """格式化单个片段及其局部上下文，供合成阶段使用。"""
        key = f"{segment.live_id}::{segment.source_type}"
        seq = self.store.by_live_source.get(key, [])
        if not seq:
            return (
                f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
                f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
                f"核心片段内容：{segment.text}"
            )

        try:
            pos = seq.index(segment.segment_id)
        except ValueError:
            return (
                f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
                f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
                f"核心片段内容：{segment.text}"
            )

        start = max(0, pos - context_window)
        end = min(len(seq), pos + context_window + 1)
        local_lines: list[str] = []
        for idx in range(start, end):
            sid = seq[idx]
            local_seg = self.store.segments.get(sid)
            if not local_seg:
                continue
            marker = "核心片段" if sid == segment.segment_id else "上下文片段"
            local_lines.append(
                f"  - [{marker}] ({local_seg.hhmmss}) {local_seg.text}"
            )

        local_context = "\n".join(local_lines)
        return (
            f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
            f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
            f"局部上下文（同一访谈同一来源，窗口={context_window}）：\n{local_context}"
        )

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
            if not isinstance(useful_items, list):
                useful_items = []
            else:
                useful_items = [item for item in useful_items if isinstance(item, dict) and "segment_id" in item]
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

    def _build_synthesis_prompt(
        self,
        question: str,
        useful_segments: list[Segment],
        synthesis_context_window: int,
    ) -> list[dict[str, str]]:
        lines: list[str] = []
        for s in useful_segments:
            lines.append(
                self._format_segment_with_local_context(
                    s,
                    context_window=synthesis_context_window,
                )
            )
        context = "\n".join(lines)
        return [
            {
                "role": "user",
                "content": (
                    "你是严谨的证据型问答助手。请只使用下面列出的有用片段及其局部上下文回答问题，不能臆造。\n"
                    f"用户问题：{question}\n\n"
                    "片段列表（每条含核心片段和局部上下文）：\n"
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

    def _build_batch_synthesis_prompt(
        self,
        question: str,
        segments: list[Segment],
        batch_index: int,
        total_batches: int,
        synthesis_context_window: int,
    ) -> list[dict[str, str]]:
        lines: list[str] = []
        for s in segments:
            lines.append(self._format_segment_with_local_context(s, context_window=synthesis_context_window))
        context = "\n".join(lines)
        return [
            {
                "role": "user",
                "content": (
                    "你是严谨的证据型问答助手。请根据下面的片段及其局部上下文总结与问题相关的关键信息。\n"
                    f"用户问题：{question}\n"
                    f"这是第 {batch_index}/{total_batches} 批片段。\n\n"
                    "片段列表（每条含核心片段和局部上下文）：\n"
                    f"{context}\n\n"
                    "请输出JSON对象，格式为：\n"
                    '{"summary":"...","key_segments":[{"segment_id":"...","reason":"..."}]}\n'
                    "要求：\n"
                    "1) summary中简洁地总结这批片段与问题的相关信息；\n"
                    "2) key_segments列出该批最关键的5-10个segment_id及其原因；\n"
                    "3) 仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _build_group_query(self, questions: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for q in questions:
            qid = q.get("question_id", "")
            text = q.get("question_text", "").strip()
            lines.append(f"[{qid}] {text}")
        return "请根据受访者（采访者是GUO, An）的话回答以下相关子问题：\n" + "\n".join(lines)

    def _build_interview_group_prompt(
        self,
        questions: list[dict[str, str]],
        segments: list[Segment],
        live_id: str,
        meta: dict[str, str],
    ) -> list[dict[str, str]]:
        question_lines: list[str] = []
        for q in questions:
            qid = q.get("question_id", "")
            text = q.get("question_text", "").strip()
            question_lines.append(f"[{qid}] {text}")

        lines: list[str] = []
        for s in segments:
            lines.append(self._format_segment_with_local_context(s))
        context = "\n".join(lines)
        question_block = "\n".join(question_lines)
        return [
            {
                "role": "user",
                "content": (
                    "你是严谨的证据型问答助手。请根据下面给定的片段逐条回答每个子问题，不能臆造。\n"
                    f"当前访谈 live_id={live_id}，标题={meta.get('video_title', '')}，主播={meta.get('anchor_name', '')}。\n\n"
                    "问题列表：\n"
                    f"{question_block}\n\n"
                    "片段列表（每条含核心片段和局部上下文）：\n"
                    f"{context}\n\n"
                    "请输出JSON对象，格式为：\n"
                    '{"answers":[{"question_id":"...","answer":"...","evidence":[{"segment_id":"...","reason":"..."}]}]}\n'
                    "要求：\n"
                    "1) answer必须与该访谈片段内容一致；\n"
                    "2) evidence仅使用给定片段中的segment_id；\n"
                    "3) 对于没有足够证据的问题，answer填写空字符串或“无相关证据”，evidence设置为空数组；\n"
                    "4) 按问题列表顺序输出 answers；\n"
                    "5) 仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _normalize_group_answers(
        self,
        questions: list[dict[str, str]],
        answers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        answers_by_id = {item.get("question_id", ""): item for item in answers if item.get("question_id")}
        normalized: list[dict[str, Any]] = []
        for q in questions:
            qid = q.get("question_id", "")
            item = answers_by_id.get(qid, {})
            normalized.append(
                {
                    "question_id": qid,
                    "question_text": q.get("question_text", ""),
                    "answer": (item.get("answer") or "").strip(),
                    "evidence": item.get("evidence", []) or [],
                }
            )
        return normalized

    def ask_group(
        self,
        questions: list[dict[str, str]],
        source: str = "",
        vector_top_k: int = 1000,
        bm25_top_k: int = 1000,
        context_window: int = 6,
        vector_score_threshold: float = 0.332,
        bm25_score_threshold: float = 15.0,
        analysis_batch_size: int = 20,
    ) -> dict[str, Any]:
        self._ensure_client()
        group_query = self._build_group_query(questions)
        if self.logger:
            self.logger.info(f"开始组问题检索 source={source}，包含 {len(questions)} 个子问题")
            self.logger.debug(f"构建的组查询: {group_query}")
        candidates, stats = self.retrieve(
            group_query,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            context_window=context_window,
            vector_score_threshold=vector_score_threshold,
            bm25_score_threshold=bm25_score_threshold,
            max_base_segments=None,
            max_expanded_segments=None,
        )

        if self.logger:
            self.logger.info(
                f"组问题检索完成: candidate_count={stats['candidate_count']}"
            )

        analysis, useful_segments, analysis_summary, analysis_llm_calls = self._analyze_candidates(
            group_query, candidates, analysis_batch_size
        )

        if self.logger:
            self.logger.info(
                f"组问题分析完成: useful_segments={analysis_summary['useful_segment_count']}"
            )

        interview_meta: dict[str, dict[str, str]] = {}
        for seg in self.store.segments.values():
            if seg.live_id not in interview_meta:
                interview_meta[seg.live_id] = {
                    "live_id": seg.live_id,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "video_datetime": seg.video_datetime,
                }

        interview_results: list[dict[str, Any]] = []
        for live_id, meta in sorted(interview_meta.items(), key=lambda x: (x[1]["video_datetime"], x[0])):
            interview_useful = [seg for seg in useful_segments if seg.live_id == live_id]
            if not interview_useful:
                interview_answers = [
                    {
                        "question_id": q.get("question_id", ""),
                        "question_text": q.get("question_text", ""),
                        "answer": "",
                        "evidence": [],
                        "citations": [],
                    }
                    for q in questions
                ]
            else:
                try:
                    parsed, _ = self._call_llm_json(
                        self._build_interview_group_prompt(questions, interview_useful, live_id, meta),
                        f"组问题 {source} 访谈 {live_id} 回答",
                    )
                    raw_answers = parsed.get("answers", []) or []
                except Exception as exc:
                    if self.logger:
                        self.logger.error(f"访谈 {live_id} 组回答失败: {exc}")
                    raw_answers = []

                interview_answers = []
                for item in self._normalize_group_answers(questions, raw_answers):
                    citations, evidence = self._build_citations_from_evidence(item.get("evidence", []), interview_useful)
                    interview_answers.append(
                        {
                            "question_id": item["question_id"],
                            "question_text": item["question_text"],
                            "answer": item["answer"],
                            "evidence": evidence,
                            "citations": citations,
                        }
                    )

            interview_results.append(
                {
                    **meta,
                    "answers": interview_answers,
                    "useful_segment_count": len(interview_useful),
                }
            )

        result = {
            "source": source,
            "questions": questions,
            "retrieval": stats,
            "analysis_summary": analysis_summary,
            "useful_segment_count": len(useful_segments),
            "interview_results": interview_results,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        archive_data = {
            **result,
            "retrieval_segments": [
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
            ],
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
            },
        }
        result["archive_path"] = str(self._archive(archive_data))
        return result

    def _synthesize_with_batches(
        self,
        question: str,
        useful_segments: list[Segment],
        batch_size: int = 200,
        synthesis_context_window: int = 6,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """用分组合成的方式处理大量有用段。返回 (answer_text, final_evidence, llm_metadata)"""
        total_segments = len(useful_segments)
        total_batches = max(1, (total_segments + batch_size - 1) // batch_size)
        
        if self.logger:
            self.logger.info(
                f"准备分批合成最终回答，共 {total_segments} 个有用片段，分 {total_batches} 批处理"
            )
        
        batch_summaries: list[dict[str, Any]] = []
        key_segment_ids: set[str] = set()
        all_llm_calls: list[dict[str, Any]] = []
        
        # 第一阶段：对每批片段进行合成
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_segments)
            batch = useful_segments[start:end]
            
            if self.logger:
                self.logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch)} 个片段")
            
            try:
                prompt = self._build_batch_synthesis_prompt(
                    question,
                    batch,
                    batch_idx + 1,
                    total_batches,
                    synthesis_context_window=synthesis_context_window,
                )
                parsed, llm_metadata = self._call_llm_json(
                    prompt,
                    f"批次合成 {batch_idx + 1}/{total_batches}",
                )
                summary = parsed.get("summary", "").strip()
                key_segs = parsed.get("key_segments", []) or []
                
                batch_summaries.append({
                    "batch_index": batch_idx + 1,
                    "summary": summary,
                    "segment_count": len(batch),
                })
                
                for seg_info in key_segs:
                    if seg_info.get("segment_id"):
                        key_segment_ids.add(seg_info["segment_id"])
                
                all_llm_calls.append(llm_metadata)
                
                if self.logger:
                    self.logger.info(
                        f"批次 {batch_idx + 1} 合成完成，共提取 {len(key_segs)} 个关键段"
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"批次 {batch_idx + 1} 合成失败: {exc}，使用原始片段")
                batch_summaries.append({
                    "batch_index": batch_idx + 1,
                    "summary": f"该批包含 {len(batch)} 个相关片段，无法生成摘要",
                    "segment_count": len(batch),
                    "error": str(exc),
                })
                all_llm_calls.append({
                    "description": f"批次合成 {batch_idx + 1}/{total_batches}",
                    "success": False,
                    "error": str(exc),
                })
                # 如果合成失败，将此批所有段作为关键段保留
                for seg in batch:
                    key_segment_ids.add(seg.segment_id)
        
        # 第二阶段：基于所有批次的合成，生成最终答案
        if self.logger:
            self.logger.info(f"第一阶段合成完成，提取了 {len(key_segment_ids)} 个关键段，准备生成最终答案")
        
        # 选出关键段及所有段（确保完整性）
        key_segments = [seg for seg in useful_segments if seg.segment_id in key_segment_ids]
        
        final_evidence: list[dict[str, Any]] = []
        final_answer = ""
        final_llm_metadata = {}
        
        try:
            # 构建最终合成 prompt，包含批次摘要和关键段
            batch_summary_text = "\n".join([
                f"[第{s['batch_index']}批] {s['summary']}"
                for s in batch_summaries
            ])
            
            lines: list[str] = []
            for s in key_segments:
                lines.append(
                    self._format_segment_with_local_context(
                        s,
                        context_window=synthesis_context_window,
                    )
                )
            context = "\n".join(lines)
            
            final_prompt = [
                {
                    "role": "user",
                    "content": (
                        "你是严谨的证据型问答助手。基于下面的批次摘要和关键片段，生成一个全面的最终答案。\n"
                        f"用户问题：{question}\n\n"
                        "批次摘要：\n"
                        f"{batch_summary_text}\n\n"
                        "关键片段列表（每条含核心片段和局部上下文）：\n"
                        f"{context}\n\n"
                        "请输出JSON对象，格式为：\n"
                        '{"answer":"...","evidence":[{"segment_id":"...","reason":"..."}]}\n'
                        "要求：\n"
                        "1) answer必须是一个全面的、综合所有批次的答案，涵盖所有重要信息；\n"
                        "2) evidence应包含所有关键segment_id，按时间顺序排列；\n"
                        "3) 每个evidence条目说明该片段如何支持答案；\n"
                        "4) 力求完整性和全面性；\n"
                        "5) 不遗漏任何关键片段。\n"
                        "仅输出JSON，不要额外文本。"
                    ),
                }
            ]
            
            parsed, final_llm_metadata = self._call_llm_json(
                final_prompt,
                "最终答案合成",
            )
            
            final_evidence = parsed.get("evidence", []) or []
            final_answer = parsed.get("answer", "").strip() or "模型未返回有效答案。"
            
            if self.logger:
                self.logger.info(f"最终答案生成成功，包含 {len(final_evidence)} 个引用")
        except Exception as exc:
            if self.logger:
                self.logger.error(f"最终答案合成失败: {exc}")
            final_answer = "模型在生成最终答案时发生错误。"
            final_llm_metadata = {"success": False, "error": str(exc)}
            # 如果最终合成失败，将所有关键段作为 evidence
            for seg in key_segments:
                final_evidence.append({
                    "segment_id": seg.segment_id,
                    "reason": f"该片段包含相关信息"
                })
        
        return final_answer, final_evidence, {
            "batch_synthesis": {
                "total_segments": total_segments,
                "batch_size": batch_size,
                "synthesis_context_window": synthesis_context_window,
                "total_batches": total_batches,
                "batch_summaries": batch_summaries,
                "batch_llm_calls": all_llm_calls,
            },
            "final_synthesis": final_llm_metadata,
        }

    def _build_citations_from_evidence(
        self,
        evidence: list[dict[str, Any]],
        useful_segments: list[Segment],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        id_to_seg = {s.segment_id: s for s in useful_segments}
        normalized_evidence = list(evidence)
        citations: list[dict[str, Any]] = []

        for idx, item in enumerate(normalized_evidence, start=1):
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

        cited_segment_ids = {item.get("segment_id") for item in normalized_evidence if item.get("segment_id")}
        missing_segments = [seg for seg in useful_segments if seg.segment_id not in cited_segment_ids]
        if missing_segments and self.logger:
            self.logger.warning(f"LLM 遗漏了 {len(missing_segments)} 个有用段，将自动添加到 evidence")
        for seg in missing_segments:
            reason = f"该片段包含与问题相关的有用信息：{seg.text[:100]}..."
            normalized_evidence.append(
                {
                    "segment_id": seg.segment_id,
                    "reason": reason,
                }
            )
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
                    "reason": reason,
                }
            )
        return citations, normalized_evidence

    def ask(
        self,
        question: str,
        vector_top_k: int = 1000,
        bm25_top_k: int = 1000,
        context_window: int = 3,
        vector_score_threshold: float = 0.3,
        bm25_score_threshold: float = 15.0,
        analysis_batch_size: int = 20,
        synthesis_context_window: int = 6,
        synthesis_batch_trigger_count: int = 100,
        synthesis_batch_size: int = 50,
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
            
            # 当有用段数超过阈值时，使用分批合成
            if len(useful_segments) > synthesis_batch_trigger_count:
                if self.logger:
                    self.logger.info(
                        f"有用段数较多（>{synthesis_batch_trigger_count}），使用分批合成策略，batch_size={synthesis_batch_size}"
                    )
                answer_text, final_evidence, synthesis_llm_metadata = self._synthesize_with_batches(
                    question,
                    useful_segments,
                    batch_size=synthesis_batch_size,
                    synthesis_context_window=synthesis_context_window,
                )
            else:
                try:
                    parsed, synthesis_llm_metadata = self._call_llm_json(
                        self._build_synthesis_prompt(
                            question,
                            useful_segments,
                            synthesis_context_window=synthesis_context_window,
                        ),
                        "最终答案合成",
                    )
                    final_evidence = parsed.get("evidence", []) or []
                    answer_text = parsed.get("answer", "").strip() or "模型未返回有效答案。"
                except Exception as exc:
                    if self.logger:
                        self.logger.error(f"最终答案合成失败: {exc}")
                    answer_text = "模型在生成最终答案时发生错误。"
                    synthesis_llm_metadata = {"success": False, "error": str(exc)}

        citations, final_evidence = self._build_citations_from_evidence(final_evidence, useful_segments)

        if citations and "[#" not in answer_text:
            refs = " ".join(f"[#{i}]" for i in range(1, len(citations) + 1))
            answer_text = f"{answer_text} 参考引用：{refs}"

        useful_by_interview: dict[str, list[Segment]] = {}
        for seg in useful_segments:
            useful_by_interview.setdefault(seg.live_id, []).append(seg)

        interview_meta: dict[str, dict[str, str]] = {}
        for seg in self.store.segments.values():
            if seg.live_id not in interview_meta:
                interview_meta[seg.live_id] = {
                    "live_id": seg.live_id,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "video_datetime": seg.video_datetime,
                }
        interview_results: list[dict[str, Any]] = []
        for live_id, meta in sorted(interview_meta.items(), key=lambda x: (x[1]["video_datetime"], x[0])):
            interview_useful = useful_by_interview.get(live_id, [])
            if not interview_useful:
                interview_results.append(
                    {
                        **meta,
                        "answer": "",
                        "citations": [],
                        "useful_segment_count": 0,
                    }
                )
                continue
            interview_answer = ""
            interview_evidence: list[dict[str, Any]] = []
            if len(interview_useful) > synthesis_batch_trigger_count:
                interview_answer, interview_evidence, _ = self._synthesize_with_batches(
                    question,
                    interview_useful,
                    batch_size=synthesis_batch_size,
                    synthesis_context_window=synthesis_context_window,
                )
            else:
                try:
                    parsed, _ = self._call_llm_json(
                        self._build_synthesis_prompt(
                            question,
                            interview_useful,
                            synthesis_context_window=synthesis_context_window,
                        ),
                        f"访谈 {live_id} 答案合成",
                    )
                    interview_evidence = parsed.get("evidence", []) or []
                    interview_answer = parsed.get("answer", "").strip() or ""
                except Exception:
                    interview_answer = ""
                    interview_evidence = []
            interview_citations, interview_evidence = self._build_citations_from_evidence(
                interview_evidence,
                interview_useful,
            )
            if interview_citations and "[#" not in interview_answer:
                refs = " ".join(f"[#{i}]" for i in range(1, len(interview_citations) + 1))
                interview_answer = f"{interview_answer} 参考引用：{refs}".strip()
            interview_results.append(
                {
                    **meta,
                    "answer": interview_answer,
                    "citations": interview_citations,
                    "useful_segment_count": len(interview_useful),
                }
            )

        result = {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(candidates),
            "retrieval": stats,
            "analysis_summary": analysis_summary,
            "useful_segment_count": len(useful_segments),
            "interview_results": interview_results,
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

