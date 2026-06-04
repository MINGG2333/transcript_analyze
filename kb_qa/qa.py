"""知识库问答主模块：VideoKnowledgeQA 类定义与主流程编排。

方法从以下子模块导入：
- qa_retrieval.py: 检索逻辑
- qa_analysis.py: 分析与合成
- qa_safety.py: 内容安全审核
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .indexer import BM25Index, SegmentStore, VectorIndex
from .models import RiskLevel

# ── 从子模块导入模块级函数（作为类方法绑定） ──
from .qa_retrieval import (
    load_kb_description,
    build_or_update,
    retrieve,
)
from .qa_analysis import (
    build_kb_background_text,
    build_citations_from_evidence,
    build_merge_prompt,
)
from .qa_safety import check_content_safety, sanitize_content
from .background_knowledge import BackgroundKnowledge
from .qa_utils import (
    call_llm_json,
    validate_answer_citations,
    validate_citations_consistency,
    filter_citations_by_answer,
    renumber_citations,
    archive_result,
    ensure_client,
)
from .qa_prompts import build_judge_prompt


class PrefixedLogger:
    """给日志消息添加统一前缀的轻量包装，适配 loguru / SimpleLogger 等任意 logger。

    用法:
        log = PrefixedLogger(logger, "[前缀]")
        log.info("消息")  # 实际输出: [前缀] 消息
    """
    def __init__(self, logger, prefix: str):
        self._logger = logger
        self._prefix = prefix

    def __getattr__(self, name):
        # 将 info/warning/debug/error 等调用代理到原始 logger，并加上前缀
        attr = getattr(self._logger, name, None)
        if attr is None or not callable(attr):
            raise AttributeError(f"logger 没有方法 {name}")
        def wrapped(msg, *args, **kwargs):
            attr(f"{self._prefix} {msg}", *args, **kwargs)
        return wrapped


class VideoKnowledgeQA:
    """直播视频知识库问答系统的主类。"""

    # ── 从子模块导入的方法（作为类属性绑定） ──
    _load_kb_description = load_kb_description
    build_or_update = build_or_update
    retrieve = retrieve
    _build_kb_background_text = build_kb_background_text
    _build_citations_from_evidence = build_citations_from_evidence
    _check_content_safety = check_content_safety
    _sanitize_content = sanitize_content

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
        background_knowledge_dir: Optional[Path] = None,
    ):
        self.records_path = records_path
        self.subtitle_root = subtitle_root
        self.kb_dir = kb_dir
        self.llm_model = llm_model
        self.logger = logger

        self.store = SegmentStore(kb_dir / "segment_store.json")
        self.store.load()
        self.vector = VectorIndex(kb_dir / "chroma_db", embedding_model=embedding_model)
        # BM25 延迟加载：构建时仅在 upsert 后重建，问答时通过 get_bm25() 惰性获取
        self._bm25: Optional[BM25Index] = None
        self.api_base = api_base
        self.api_key = api_key
        self.client = None
        self.kb_description_path = kb_dir / "kb_description.txt"
        self.kb_description: Optional[str] = None
        self._load_kb_description()

        # ── 加载外部背景知识（成员档案、平台术语等） ──
        self.bg_knowledge = BackgroundKnowledge()
        if background_knowledge_dir is not None:
            bg_dir = Path(background_knowledge_dir)
            if bg_dir.exists() and bg_dir.is_dir():
                loaded = self.bg_knowledge.load_from_dir(bg_dir)
                if self.logger:
                    self.logger.info(
                        f"已加载背景知识: {loaded} 字符, "
                        f"来自 {bg_dir}"
                    )
            elif self.logger:
                self.logger.warning(
                    f"背景知识目录不存在: {bg_dir}"
                )

    def _build_judge_prompt(self, question: str, segments: list) -> str:
        """旧版 judge prompt 构建（保留兼容）。"""
        return build_judge_prompt(question, segments)

    def _archive(self, result: dict[str, Any]) -> Path:
        """归档问答结果。"""
        return archive_result(result, self.kb_dir)

    def _ensure_client(self) -> None:
        """确保 OpenAI 客户端已初始化。"""
        if self.client is not None:
            return
        self.client = ensure_client(self.client, self.api_key, self.api_base, logger=self.logger)

    def _process_video_group(
        self,
        question: str,
        live_id: str,
        segs: list,
        meta: dict,
        video_idx: int,
        total_videos: int,
        per_video_batch_size: int,
    ) -> dict:
        """并发处理单个视频分组。内部串行（含 citation 校验重试），返回局部结果。

        返回格式：
        {
            "meta": {...},             # 视频元信息
            "batches": [               # 每个成功批次的局部结果
                {
                    "answer": str,     # 原始 answer（含原始 LLM 分配的编号）
                    "citations": list, # 原始 citations（含原始 LLM 分配的编号）
                    "evidence": list,  # 原始 evidence
                    "useful_segment_count": int,
                    "batch_count": int,
                },
            ],
            "llm_calls": [...],        # 本组所有 LLM 调用元数据
        }
        """
        from .qa_prompts import build_group_synthesis_prompt

        group_info = (
            f"标题={meta.get('video_title', '')}, "
            f"直播时间={meta.get('video_datetime', '')}, "
            f"主播={meta.get('anchor_name', '')}"
        )
        total = len(segs)
        n_batches = max(1, (total + per_video_batch_size - 1) // per_video_batch_size)

        # 创建带分组前缀的 logger，并发时一眼区分日志属于哪个视频
        group_prefix = (
            f"[{video_idx}/{total_videos} {meta.get('video_title', live_id[-8:])}]"
        )
        log = PrefixedLogger(self.logger, group_prefix) if self.logger else None

        if log:
            log.info(
                f"处理直播 {live_id}，共 {total} 个片段，分 {n_batches} 批（每批最多 {per_video_batch_size} 条）"
            )

        batches_result: list[dict] = []
        all_llm_calls: list[dict] = []

        for batch_idx in range(n_batches):
            start = batch_idx * per_video_batch_size
            batch = segs[start:start + per_video_batch_size]
            batch_label = f"{batch_idx + 1}/{n_batches}" if n_batches > 1 else ""
            is_first_group = (video_idx == 1 and batch_idx == 0)

            try:
                bg_text = self._build_kb_background_text()
                prompt_messages = build_group_synthesis_prompt(
                    question, batch, group_info=group_info, batch_label=batch_label, bg_text=bg_text,
                )
                if is_first_group and log:
                    log.debug("=== 分组处理首个 LLM 调用 - Prompt ===")
                    log.debug(prompt_messages[0].get("content", ""))

                parsed, llm_meta = call_llm_json(
                    self.client, self.llm_model, prompt_messages,
                    f"直播 {live_id} 合成 {batch_idx + 1}/{n_batches}"
                    if n_batches > 1 else f"直播 {live_id} 合成",
                    logger=log,
                )

                if is_first_group and log:
                    log.debug("=== 分组处理首个 LLM 调用 - Response ===")
                    log.debug(json.dumps(parsed, ensure_ascii=False))
                batch_evidence = parsed.get("evidence", []) or []
                batch_answer = parsed.get("answer", "").strip()
                all_llm_calls.append(llm_meta)

                if not batch_answer:
                    if log:
                        log.info(f"批次 {batch_idx + 1}/{n_batches} 无答案，跳过")
                    continue

                batch_citations, _ = self._build_citations_from_evidence(batch_evidence, batch)

                max_retries = 5
                problems: list[str] = []
                final_answer = batch_answer
                final_citations = list(batch_citations)
                final_evidence = list(batch_evidence)

                for retry_attempt in range(max_retries + 1):
                    if not final_citations:
                        final_answer = re.sub(r"\[\#\d+\]", "", batch_answer)
                        final_citations = []
                        if log:
                            log.info(
                                f"批次 {batch_idx + 1}/{n_batches} evidence 为空，已清除引用标记"
                            )
                        break

                    problems = validate_citations_consistency(final_answer, final_citations)
                    if not problems:
                        break

                    if log:
                        log.warning(
                            f"批次 {batch_idx + 1}/{n_batches} 引用一致性校验不通过"
                            f"（第 {retry_attempt + 1}/{max_retries + 1} 次），"
                            f"问题: {'; '.join(problems)}"
                        )

                    if retry_attempt >= max_retries:
                        if log:
                            log.warning(
                                f"批次 {batch_idx + 1}/{n_batches} 重试 {max_retries} 次后仍不通过，跳过"
                            )
                        break

                    retry_prompt = list(prompt_messages)
                    retry_prompt.append({"role": "assistant", "content": json.dumps(parsed)})
                    retry_prompt.append({
                        "role": "user",
                        "content": (
                            f"⚠️ 引用一致性检查发现以下问题：\n"
                            + "\n".join(f"  - {p}" for p in problems)
                            + "\n\n请确保 answer 中使用的每个 [#N] 编号都在 evidence 的 citation_id 中存在，"
                            "且每个 evidence 条目都在 answer 中被引用。"
                            "请修正后重新输出完整的 JSON。"
                        ),
                    })
                    try:
                        parsed, llm_meta = call_llm_json(
                            self.client, self.llm_model, retry_prompt,
                            f"直播 {live_id} 合成 {batch_idx + 1}/{n_batches}（引用修正 第{retry_attempt + 1}次）",
                            logger=log,
                        )
                        all_llm_calls.append(llm_meta)
                        batch_evidence = parsed.get("evidence", []) or []
                        batch_answer = parsed.get("answer", "").strip()
                        if not batch_answer:
                            if log:
                                log.info(f"批次 {batch_idx + 1}/{n_batches} 修正后仍无答案，跳过")
                            break
                        final_citations, _ = self._build_citations_from_evidence(batch_evidence, batch)
                        final_answer = batch_answer
                        final_evidence = batch_evidence
                    except Exception as exc:
                        if log:
                            log.warning(f"批次 {batch_idx + 1}/{n_batches} 修正失败: {exc}")
                        break
                else:
                    if log:
                        log.warning(
                            f"批次 {batch_idx + 1}/{n_batches} 重试 {max_retries} 次后仍不通过，跳过"
                        )
                    continue

                if problems:
                    if log:
                        log.warning(
                            f"批次 {batch_idx + 1}/{n_batches} 引用一致性校验不通过，跳过"
                        )
                    continue

                batches_result.append({
                    "answer": final_answer,
                    "citations": final_citations,
                    "evidence": final_evidence,
                    "useful_segment_count": len(batch),
                    "batch_count": n_batches,
                })

                if log:
                    log.info(
                        f"批次 {batch_idx + 1}/{n_batches} 完成，"
                        f"evidence={len(final_evidence)} 条"
                    )
            except Exception as exc:
                if log:
                    log.warning(f"批次 {batch_idx + 1}/{n_batches} 合成失败: {exc}")

        return {
            "meta": meta,
            "batches": batches_result,
            "llm_calls": all_llm_calls,
        }

    def ask(
        self,
        question: str,
        vector_top_k: int = 40,
        bm25_top_k: int = 50,
        context_window: int = 5,
        vector_score_threshold: float = 0.3,
        bm25_score_threshold: float = 15.0,
        analysis_batch_size: int = 20,
        synthesis_context_window: int = 6,
        synthesis_batch_trigger_count: int = 100,
        synthesis_batch_size: int = 50,
    ) -> dict[str, Any]:
        self._ensure_client()

        # CHANGED: 主题相关性判断——先判断问题是否与陈嘉仪相关
        relevance_messages = [
            {
                "role": "user",
                "content": (
                    "判断以下用户问题是否与SNH48成员陈嘉仪相关。\n\n"
                    "相关范围：她的个人资料（生日/身高/爱好/宠物等）、演出/公演/直播、音乐作品、"
                    "粉丝文化、SNH48团体相关、口袋48平台相关。\n\n"
                    "不相关示例：政治、经济、其他不相关的明星、日常闲聊不涉及陈嘉仪、数学题、代码问题等。\n\n"
                    f"用户问题：{question}\n\n"
                    '请输出JSON：{"relevant": "yes"} 或 {"relevant": "no"}'
                ),
            }
        ]
        try:
            relevance_result, _ = call_llm_json(
                self.client, self.llm_model, relevance_messages, "主题相关性判断",
                max_tokens=30, logger=self.logger, thinking_disabled=True,
            )
            is_relevant = relevance_result.get("relevant", "no") == "yes"
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"主题相关性判断异常: {exc}")
            is_relevant = True  # 异常时默认放行

        if not is_relevant:
            if self.logger:
                self.logger.info(f"❌ 问题与陈嘉仪不相关，已婉拒: {question!r}")
            result = {
                "question": question,
                "answer": "抱歉，我只了解与陈嘉仪相关的信息，请提出与陈嘉仪有关的问题。",
                "citations": [],
                "retrieved_count": 0,
                "retrieval": {
                    "vector_hits_raw": 0,
                    "vector_hits_filtered": 0,
                    "bm25_hits_raw": 0,
                    "bm25_hits_filtered": 0,
                    "raw_merged_ids": 0,
                    "used_base_ids": 0,
                    "candidate_count": 0,
                    "truncated": False,
                    "merged_ids_set": [],
                    "merged_dict_scores": {},
                },
                "analysis_summary": {
                    "total_candidates": 0,
                    "useful_segment_count": 0,
                    "analysis_batches": [],
                    "analysis_batch_size": analysis_batch_size,
                },
                "useful_segment_count": 0,
                "video_results": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "content_safety_flagged": False,
                "risk_level": 0,
                "risk_label": "SAFE",
                "safety_reason": "主题不相关，未进入审核流程",
            }
            archive_data = {
                "question": question,
                "created_at": result["created_at"],
                "retrieval": result["retrieval"],
                "retrieval_segments": [],
                "analysis_summary": result["analysis_summary"],
                "analysis": [],
                "useful_segments": [],
                "video_results": [],
                "llm_calls": {
                    "analysis_batches": [],
                    "synthesis": {"description": "skip_topic_not_relevant"},
                },
                "content_safety": {
                    "flagged": False,
                    "risk_level": 0,
                    "risk_label": "SAFE",
                    "reason": "主题不相关，未进入审核流程",
                    "original_answer": result["answer"],
                    "original_citations": [],
                },
                "answer": result["answer"],
                "citations": [],
                "retrieved_count": 0,
                "useful_segment_count": 0,
            }
            archive_path = self._archive(archive_data)
            result["archive_path"] = str(archive_path)
            return result

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
            max_base_segments=100,
            max_expanded_segments=None,
        )
        if self.logger:
            self.logger.info(
                f"检索结果: vector_hits_raw={stats['vector_hits_raw']}, "
                f"vector_hits_filtered={stats['vector_hits_filtered']}, "
                f"bm25_hits_raw={stats['bm25_hits_raw']}, "
                f"bm25_hits_filtered={stats['bm25_hits_filtered']}, "
                f"unique_base_ids={stats['raw_merged_ids']}, "
                f"used_base_ids={stats['used_base_ids']}, "
                f"candidate_count={stats['candidate_count']}, "
                f"truncated={stats['truncated']}"
            )

        merged_ids_set = set(stats.get("merged_ids_set", []))
        retrieval_segments = [
            {
                "segment_id": seg.segment_id,
                "source_label": seg.source_label,
                "video_title": seg.video_title,
                "anchor_name": seg.anchor_name,
                "video_offset": seg.hhmmss,
                "absolute_time": seg.absolute_time,
                "text": seg.text,
                "source_type": "基段" if seg.segment_id in merged_ids_set else "上下文扩展",
                "vector_score": stats.get("merged_dict_scores", {}).get(seg.segment_id, {}).get("vector_score", None),
                "bm25_score": stats.get("merged_dict_scores", {}).get(seg.segment_id, {}).get("bm25_score", None),
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
            self.logger.info(f"候选片段数: {len(candidates)}，跳过有用性分析，全部视为有用。")
        analysis = [
            {
                "segment_id": seg.segment_id,
                "source_label": seg.source_label,
                "video_title": seg.video_title,
                "anchor_name": seg.anchor_name,
                "video_offset": seg.hhmmss,
                "absolute_time": seg.absolute_time,
                "text": seg.text,
                "useful": True,
                "reason": "跳过分析阶段，全部保留",
            }
            for seg in candidates
        ]
        useful_segments = candidates
        analysis_summary = {
            "total_candidates": len(candidates),
            "useful_segment_count": len(candidates),
            "analysis_batches": [],
            "analysis_batch_size": analysis_batch_size,
            "skipped": True,
        }
        analysis_llm_calls = []
        if self.logger:
            self.logger.info(f"跳过分析，全部 {len(candidates)} 个候选片段均进入合成阶段。")

        if not useful_segments:
            if self.logger:
                self.logger.info("未找到有用片段，快速返回结果。")
            result = {
                "question": question,
                "answer": "未找到与问题直接相关的片段，无法给出确定答案。",
                "citations": [],
                "retrieved_count": len(candidates),
                "retrieval": stats,
                "analysis_summary": analysis_summary,
                "useful_segment_count": 0,
                "video_results": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            archive_data = {
                **result,
                "retrieval_segments": retrieval_segments,
                "analysis": analysis,
                "useful_segments": [],
                "llm_calls": {
                    "analysis_batches": analysis_llm_calls,
                    "synthesis": {"description": "skip_synthesis_no_useful_segments"},
                },
            }
            archive_path = self._archive(archive_data)
            result["archive_path"] = str(archive_path)
            return result

        # ---- 按直播视频分组合成 ----
        video_groups: dict[str, list] = {}
        for seg in useful_segments:
            video_groups.setdefault(seg.live_id, []).append(seg)
        for live_id in video_groups:
            video_groups[live_id].sort(key=lambda s: s.start_time)

        video_meta: dict[str, dict[str, str]] = {}
        for seg in self.store.segments.values():
            if seg.live_id not in video_meta:
                video_meta[seg.live_id] = {
                    "live_id": seg.live_id,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "video_datetime": seg.video_datetime,
                }

        per_video_batch_size = 500
        total_videos = len(video_groups)
        if self.logger:
            self.logger.info(f"开始合成：共 {total_videos} 个视频，{len(useful_segments)} 个片段")
        video_results: list[dict[str, Any]] = []
        synthesis_llm_calls: list[dict[str, Any]] = []

        # ── Phase B: 并发处理各视频分组 ──
        # 所有视频分组相互独立，通过 ThreadPoolExecutor 并发处理
        # 每个视频组内部保持串行（含 citation 校验重试），组间并发
        # DeepSeek v4-flash 支持 2500 并发，视频分组通常 < 50 个，绰绰有余
        MAX_CONCURRENT = 10
        video_group_items = sorted(
            video_groups.items(), key=lambda x: x[1][0].start_time if x[1] else 0
        )

        group_raw_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            future_to_live = {}
            for video_idx, (live_id, segs) in enumerate(video_group_items, start=1):
                meta = video_meta.get(live_id, {})
                future = executor.submit(
                    self._process_video_group,
                    question, live_id, segs, meta, video_idx, total_videos,
                    per_video_batch_size,
                )
                future_to_live[future] = live_id

            for future in as_completed(future_to_live):
                live_id = future_to_live[future]
                try:
                    group_raw_results[live_id] = future.result()
                except Exception as exc:
                    if self.logger:
                        self.logger.error(f"视频组 {live_id} 处理异常: {exc}")
                    group_raw_results[live_id] = None

        # ── 串行后处理：全局引用编码 ──
        # 按原始时间顺序遍历各组，分配全局连续的 evidence 序号和 citation 编号
        all_evidence: list[dict[str, Any]] = []
        global_evidence_offset = 0

        for live_id, _ in video_group_items:
            result = group_raw_results.get(live_id)
            if result is None:
                continue

            meta = result["meta"]
            for batch in result["batches"]:
                local_answer = batch["answer"]
                local_citations = batch["citations"]
                local_evidence = batch.get("evidence", [])

                if not local_citations:
                    continue

                # 构建局部→全局编号映射
                local_to_global: dict[str, str] = {}
                for local_idx, c in enumerate(local_citations):
                    old_id = c["citation_id"]
                    global_idx = global_evidence_offset + local_idx
                    local_to_global[old_id] = f"#{global_idx + 1}"

                # 更新 answer 中的引用编号
                global_answer = local_answer
                for c in local_citations:
                    old_id = c["citation_id"]
                    new_id = local_to_global[old_id]
                    if new_id != old_id:
                        global_answer = re.sub(
                            re.escape(old_id) + r'(?=[^#\d]|$)',
                            new_id, global_answer,
                        )

                # 更新 citations 中的编号
                global_citations = []
                for c in local_citations:
                    gc = dict(c)
                    gc["citation_id"] = local_to_global[c["citation_id"]]
                    global_citations.append(gc)

                all_evidence.extend(local_evidence)
                global_evidence_offset += len(local_evidence)

                video_results.append({
                    **meta,
                    "answer": local_answer,
                    "citations": local_citations,
                    "answer_global": global_answer,
                    "citations_global": global_citations,
                    "useful_segment_count": batch["useful_segment_count"],
                    "batch_count": batch["batch_count"],
                })

            synthesis_llm_calls.extend(result["llm_calls"])

        # ---- 最终答案生成 ----
        answer_videos = [vr for vr in video_results if vr["answer"]]
        if len(answer_videos) == 0:
            answer_text = "未找到与问题直接相关的片段，无法给出确定答案。"
            citations = []
        elif len(answer_videos) == 1:
            answer_text = answer_videos[0]["answer_global"]
            citations = answer_videos[0]["citations_global"]
        else:
            if self.logger:
                self.logger.info(f"开始最终汇总合并，共 {len(answer_videos)} 个视频分组有答案")

            citations = []
            seen_cids: set[str] = set()
            for vr in answer_videos:
                for c in vr["citations_global"]:
                    cid = c["citation_id"]
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        citations.append(c)

            video_summaries = []
            for idx, vr in enumerate(answer_videos, start=1):
                global_answer = vr["answer_global"]
                global_citations = vr["citations_global"]
                evidence_lines = []
                for c in global_citations:
                    cid = c["citation_id"]
                    ctype = c.get("citation_type", "")
                    reason = c.get("reason", "")
                    segments_list = c.get("segments", []) or []
                    seg_lines = []
                    for seg_info in segments_list:
                        stype = seg_info.get("source_type", "")
                        user = seg_info.get("anchor_name", "")
                        offset = seg_info.get("video_offset", "")
                        text = seg_info.get("quoted_text", "")
                        seg_lines.append(f"      [{stype}] 用户名={user} 偏移={offset} \"{text}\"")
                    segs_text = "\n".join(seg_lines)
                    evidence_lines.append(f"  [{cid}] 类型={ctype}\n  ⚠️ 分析：{reason}\n{segs_text}")
                evidence_text = "\n".join(evidence_lines) if evidence_lines else "  无直接引用片段"
                video_summaries.append(
                    f"[视频 {idx}] 标题={vr.get('video_title', '')}, "
                    f"直播时间={vr.get('video_datetime', '')}\n"
                    f"  答案：\n{global_answer}\n"
                    f"  引用片段：\n{evidence_text}"
                )
            summaries_text = "\n\n".join(video_summaries)

            bg_text_for_merge = ""
            if hasattr(self, "bg_knowledge") and self.bg_knowledge and self.bg_knowledge.is_loaded:
                bg_text_for_merge = self.bg_knowledge.to_prompt_block()
            merge_prompt = build_merge_prompt(question, summaries_text, bg_knowledge_text=bg_text_for_merge)
            try:
                if self.logger:
                    self.logger.debug("=== 最终答案合并 LLM 调用 - Prompt ===")
                    self.logger.debug(merge_prompt[0].get("content", ""))
                # CHANGED: 必须串行——依赖所有视频分组合成结果，不能提前并发
                parsed, merge_llm_meta = call_llm_json(
                    self.client, self.llm_model, merge_prompt, "最终答案合并", logger=self.logger
                )
                if self.logger:
                    self.logger.debug("=== 最终答案合并 LLM 调用 - Response ===")
                    self.logger.debug(json.dumps(parsed, ensure_ascii=False))
                answer_text = (parsed.get("answer") or "").strip()
                if not answer_text:
                    raise ValueError("LLM返回空答案")
                synthesis_llm_calls.append(merge_llm_meta)
                if self.logger:
                    self.logger.info(f"最终答案合并完成，答案长度={len(answer_text)}")

                max_merge_retries = 5
                for merge_retry in range(max_merge_retries + 1):
                    invalid_refs = validate_answer_citations(answer_text)
                    if not invalid_refs:
                        break
                    if merge_retry >= max_merge_retries:
                        if self.logger:
                            self.logger.warning(
                                f"最终答案合并重试 {max_merge_retries} 次后仍不通过，使用当前结果"
                            )
                        break
                    correction_msg = (
                        "⚠️ 答案中的以下引用格式不符合要求，请修正：\n"
                        + "\n".join(f"  ❌ {r}" for r in invalid_refs)
                        + "\n\n引用格式必须为 [#N]（如 [#1]、[#5]），一个中括号内只能有一个引用编号。"
                        "禁止使用 [#N-#M]（区间）、[#N, #M]（逗号）等非标准格式。"
                    )
                    if self.logger:
                        self.logger.warning(
                            f"最终答案合并引用格式不通过（第 {merge_retry + 1} 次），重试: {invalid_refs}"
                        )
                    retry_prompt = list(merge_prompt)
                    retry_prompt.append({"role": "assistant", "content": json.dumps({"answer": answer_text})})
                    retry_prompt.append({"role": "user", "content": correction_msg})
                    parsed, merge_llm_meta = call_llm_json(
                        self.client, self.llm_model, retry_prompt,
                        f"最终答案合并（引用格式修正 第{merge_retry + 1}次）",
                        logger=self.logger,
                    )
                    answer_text = (parsed.get("answer") or "").strip()
                    if not answer_text:
                        if self.logger:
                            self.logger.warning("LLM修正后返回空答案，使用修正前结果")
                        break
                    synthesis_llm_calls.append(merge_llm_meta)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"最终答案合并失败，回退到简单拼接: {exc}")
                answer_text = "\n\n---\n\n".join(vr["answer"] for vr in answer_videos)

        synthesis_llm_metadata = {
            "description": "per_video_batch_synthesis",
            "per_video_batch_size": per_video_batch_size,
            "video_count": len(video_results),
            "total_calls": len(synthesis_llm_calls),
            "calls": synthesis_llm_calls,
        }

        try:
            # CHANGED: 安全审核提前到引用重编号之前，便于将token计入汇总
            risk_level, safety_reason, safety_llm_meta = self._check_content_safety(
                question, answer_text, citations
            )
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"内容安全审核异常: {exc}")
            risk_level = RiskLevel.MEDIUM
            safety_reason = f"审核异常: {exc}"
            safety_llm_meta = None

        # CHANGED: 将安全审核的 LLM 调用计入 token 汇总
        if safety_llm_meta:
            synthesis_llm_calls.append(safety_llm_meta)

        # CHANGED: 审核不通过时尝试净化（仅尝试一次）
        sanitize_attempted = False
        if risk_level >= RiskLevel.LOW:
            if self.logger:
                self.logger.info(
                    f"🔄 尝试内容安全净化（原风险={risk_level.name}）: {safety_reason[:80]}"
                )
            new_answer, new_citations, sanitize_llm_meta = self._sanitize_content(
                question, answer_text, citations, safety_reason
            )
            if sanitize_llm_meta:
                synthesis_llm_calls.append(sanitize_llm_meta)

            if new_answer:
                # 重新审核净化后的版本
                try:
                    retry_level, retry_reason, retry_llm_meta = self._check_content_safety(
                        question, new_answer, new_citations
                    )
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(f"安全净化后重审异常: {exc}")
                    retry_level = RiskLevel.MEDIUM
                    retry_reason = f"重审异常: {exc}"
                    retry_llm_meta = None

                if retry_llm_meta:
                    synthesis_llm_calls.append(retry_llm_meta)

                if retry_level == RiskLevel.SAFE:
                    sanitize_attempted = True
                    answer_text = new_answer
                    citations = new_citations
                    risk_level = RiskLevel.SAFE
                    safety_reason = f"已通过净化去除不安全内容（原: {safety_reason}"
                    if self.logger:
                        self.logger.info("✅ 内容安全净化通过，使用净化后版本")
                else:
                    if self.logger:
                        self.logger.warning(
                            f"内容安全净化后仍不通过（{retry_level.name}）: {retry_reason}"
                        )
            else:
                if self.logger:
                    self.logger.info("内容安全净化后无可用内容，保持拦截")

        # CHANGED: Token 用量汇总日志（含安全审核+净化调用）
        if self.logger and synthesis_llm_calls:
            total_prompt = sum(
                c.get("input_tokens", 0) or 0 for c in synthesis_llm_calls
            )
            total_completion = sum(
                c.get("output_tokens", 0) or 0 for c in synthesis_llm_calls
            )
            total_tokens = total_prompt + total_completion
            self.logger.info(
                "━━━ Token 用量汇总 ━━━\n"
                f"  LLM 调用次数: {len(synthesis_llm_calls)}\n"
                f"  输入 tokens : {total_prompt:,}\n"
                f"  输出 tokens : {total_completion:,}\n"
                f"  总 tokens   : {total_tokens:,}\n"
                "━━━━━━━━━━━━━━━━━"
            )

        citations_before = len(citations)
        citations = filter_citations_by_answer(answer_text, citations)
        answer_text, citations = renumber_citations(answer_text, citations)
        if self.logger:
            self.logger.info(f"引用过滤: 保留 {len(citations)}/{citations_before} 个 citations，已重编号")

        original_answer = answer_text
        original_citations = list(citations)

        if risk_level >= RiskLevel.LOW:  # CHANGED: 调严审核——只有 SAFE(0) 才放行
            if self.logger:
                self.logger.warning(
                    f"⚠️ 内容安全审核未通过！风险等级={risk_level.name}({risk_level.value}), "
                    f"问题: {question!r}, 原因: {safety_reason}"
                )
            answer_text = (
                "该回答可能包含需要审核的内容，暂时无法直接显示。\n"
                "由于 AI 输出有不确定性风险，为保证网站合规和内容安全，采取较保守的内容展示策略。\n"
                "如需获取回复，请留下您的邮箱，审核后会通过邮箱发送给您。"
            )
            citations = []
            content_safety_flagged = True
        else:
            content_safety_flagged = False
            if self.logger:
                self.logger.info("✅ 内容安全审核通过（安全）")

        created_at = datetime.now().isoformat(timespec="seconds")
        result = {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(candidates),
            "retrieval": stats,
            "analysis_summary": analysis_summary,
            "useful_segment_count": len(useful_segments),
            "video_results": video_results,
            "created_at": created_at,
            "content_safety_flagged": content_safety_flagged,
            "risk_level": risk_level.value,
            "risk_label": risk_level.name,
            "safety_reason": safety_reason,
        }
        archive_data = {
            "question": question,
            "created_at": created_at,
            "retrieval": stats,
            "retrieval_segments": retrieval_segments,
            "analysis_summary": analysis_summary,
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
            "video_results": video_results,
            "llm_calls": {
                "analysis_batches": analysis_llm_calls,
                "synthesis": synthesis_llm_metadata,
            },
            "content_safety": {
                "flagged": content_safety_flagged,
                "risk_level": risk_level.value,
                "risk_label": risk_level.name,
                "reason": safety_reason,
                "original_answer": original_answer,
                "original_citations": original_citations,
            },
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(candidates),
            "useful_segment_count": len(useful_segments),
        }
        archive_path = self._archive(archive_data)
        result["archive_path"] = str(archive_path)
        return result
