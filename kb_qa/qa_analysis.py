"""分析与合成：候选片段分析、分组/分批合成、引用构建。"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .models import Segment
from .name_normalizer import normalize_text
from .qa_prompts import (
    REASONING_RULES,
    ANSWER_STYLE_RULES,
    CITATION_FORMAT_RULES,
    SOURCE_DISTINCTION_RULES,
    build_analysis_prompt,
    build_group_synthesis_prompt,
    build_batch_synthesis_prompt,
)
from .qa_utils import (
    call_llm_json,
    validate_citations_consistency,
    validate_answer_citations,
)


def build_kb_background_text(self) -> str:
    """构建知识库背景说明，供分析/合成提示使用。"""
    if self.kb_description:
        return f"【数据库背景】{self.kb_description}"
    return ""


def format_segment_with_local_context(
    self,
    segment: Segment,
    context_window: int = 6,
) -> str:
    """格式化单个片段及其局部上下文，供合成阶段使用。"""
    key = segment.live_id
    seq = self.store.by_live_source.get(key, [])
    if not seq:
        return (
            f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
            f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
            f"核心片段内容：{normalize_text(segment.text)}"
        )

    try:
        pos = seq.index(segment.segment_id)
    except ValueError:
        return (
            f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
            f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
            f"核心片段内容：{normalize_text(segment.text)}"
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
            f"  - [{marker}] ({local_seg.hhmmss}) [{local_seg.source_label}] "
            f"用户名={local_seg.anchor_name}; {normalize_text(local_seg.text)}"
        )

    local_context = "\n".join(local_lines)
    return (
        f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
        f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
        f"局部上下文（同一直播同一来源，窗口={context_window}）：\n{local_context}"
    )


def analyze_candidates(
    self,
    question: str,
    candidates: list[Segment],
    analysis_batch_size: int = 20,
    kb_description: str = "",
) -> tuple[list[dict[str, Any]], list[Segment], dict[str, Any], list[dict[str, Any]]]:
    """逐批分析候选片段，筛选有用片段。"""
    analysis: list[dict[str, Any]] = []
    useful_segments: list[Segment] = []
    total_batches = max(1, (len(candidates) + analysis_batch_size - 1) // analysis_batch_size)
    useful_ids: set[str] = set()
    batch_stats: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []

    for batch_index in range(total_batches):
        start = batch_index * analysis_batch_size
        batch = candidates[start: start + analysis_batch_size]
        if self.logger:
            self.logger.info(
                f"分析候选批次 {batch_index + 1}/{total_batches}，包含 {len(batch)} 个片段"
            )

        if batch_index == 0 and self.logger:
            self.logger.info("=== 批次1详细信息 ===")
            for i, seg in enumerate(batch[:min(3, len(batch))], 1):
                self.logger.info(f"  片段{i}: [{seg.segment_id}] {seg.source_label} {seg.hhmmss} {seg.text[:80]}")
            if len(batch) > 3:
                self.logger.info(f"  ... 还有 {len(batch) - 3} 个片段")

        try:
            prompt_messages = build_analysis_prompt(
                question, batch, batch_index + 1, total_batches, kb_description=kb_description
            )
            parsed, llm_metadata = call_llm_json(
                self.client, self.llm_model, prompt_messages,
                f"候选分析 {batch_index + 1}/{total_batches}",
                logger=self.logger,
            )

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


def build_citations_from_evidence(
    self,
    evidence: list[dict[str, Any]],
    useful_segments: list[Segment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """将 evidence 转换为标准化的 citations 格式。"""
    id_to_seg = {s.segment_id: s for s in useful_segments}
    normalized_evidence = list(evidence)
    citations: list[dict[str, Any]] = []

    for idx, item in enumerate(normalized_evidence, start=1):
        sids: list[str] = item.get("segment_ids", []) or []
        if not sids:
            continue

        segs = [id_to_seg.get(sid) for sid in sids if sid]
        segs = [s for s in segs if s is not None]
        if not segs:
            continue

        citation: dict[str, Any] = {
            "citation_id": f"#{idx}",
            "segments": [],
            "citation_type": item.get("citation_type", ""),
            "reason": item.get("reason", ""),
        }
        for s in segs:
            citation["segments"].append(
                {
                    "segment_id": s.segment_id,
                    "source_type": s.source_label,
                    "quoted_text": normalize_text(s.text),
                    "video_offset": s.hhmmss,
                    "absolute_time": s.absolute_time,
                    "source_file": s.file_path,
                    "video_path": s.video_path,
                    "video_title": s.video_title,
                    "anchor_name": s.anchor_name,
                    "live_id": s.live_id,
                }
            )

        citations.append(citation)
    return citations, normalized_evidence


def synthesize_with_batches(
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

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_segments)
        batch = useful_segments[start:end]

        if self.logger:
            self.logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch)} 个片段")

        try:
            bg_text = build_kb_background_text(self)
            prompt = build_batch_synthesis_prompt(
                question,
                batch,
                batch_idx + 1,
                total_batches,
                bg_text=bg_text,
                synthesis_context_window=synthesis_context_window,
                format_segment_fn=lambda s, cw=synthesis_context_window: format_segment_with_local_context(
                    self, s, context_window=cw
                ),
            )
            parsed, llm_metadata = call_llm_json(
                self.client, self.llm_model, prompt,
                f"批次合成 {batch_idx + 1}/{total_batches}",
                logger=self.logger,
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
            for seg in batch:
                key_segment_ids.add(seg.segment_id)

    if self.logger:
        self.logger.info(f"第一阶段合成完成，提取了 {len(key_segment_ids)} 个关键段，准备生成最终答案")

    key_segments = [seg for seg in useful_segments if seg.segment_id in key_segment_ids]

    final_evidence: list[dict[str, Any]] = []
    final_answer = ""
    final_llm_metadata = {}

    try:
        batch_summary_text = "\n".join([
            f"[第{s['batch_index']}批] {s['summary']}"
            for s in batch_summaries
        ])

        lines: list[str] = []
        for s in key_segments:
            lines.append(
                format_segment_with_local_context(
                    self, s, context_window=synthesis_context_window,
                )
            )
        context = "\n".join(lines)

        bg_text = build_kb_background_text(self)
        final_prompt = [
            {
                "role": "user",
                "content": (
                    "你是一名熟知这名主播的粉丝，基于下面的批次摘要和关键片段，生成一个全面的最终答案。\n\n"
                    f"{SOURCE_DISTINCTION_RULES}"
                    f"{bg_text}\n\n"
                    f"{REASONING_RULES}"
                    f"{ANSWER_STYLE_RULES}"
                    f"用户问题：{question}\n\n"
                    "批次摘要：\n"
                    f"{batch_summary_text}\n\n"
                    "关键片段列表（每条含核心片段和局部上下文）：\n"
                    f"{context}\n\n"
                    "请输出JSON对象，格式为：\n"
                    '{"answer":"...","evidence":[{"segment_ids":["..."],"citation_type":"...","reason":"..."}]}\n'
                    "要求：\n"
                    "1) answer必须是一个全面的、综合所有批次的答案，涵盖所有重要信息；\n"
                    "2) evidence应包含所有关键segment_id，按时间顺序排列；\n"
                    "3) 每个evidence条目说明该片段如何支持答案，并注明是「主播自述」还是「主播转述」；\n"
                    "4) 力求完整性和全面性；\n"
                    "5) 不遗漏任何关键片段。\n"
                    "仅输出JSON，不要额外文本。"
                ),
            }
        ]

        parsed, final_llm_metadata = call_llm_json(
            self.client, self.llm_model, final_prompt,
            "最终答案合成",
            logger=self.logger,
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


def build_merge_prompt(
    question: str, video_summaries_text: str
) -> list[dict[str, str]]:
    """构建多视频分组答案合并的 prompt。"""
    return [
        {
            "role": "user",
            "content": (
                "你是一名熟知这名主播的粉丝，现在需要将多个视频分组对同一个问题的分析结果"
                "合并成一个连贯、完整的答案。\n\n"
                "每个视频分组的分析结果包含该组的初步答案和引用的片段（引用编号已经是全局统一的最终编号）。\n"
                "你需要在保持信息完整性的前提下，去除重复内容，组织成一个结构清晰的统一答案。\n"
                "不要在答案中包含组号和分组的标记，而是自然地整合信息。\n"
                "引用编号 [#N] 已在输入中使用最终编号，输出时继续使用这些编号即可。\n\n"
                f"问题：{question}\n\n"
                "各视频分组分析结果：\n"
                f"{video_summaries_text}\n\n"
                "请输出JSON对象，格式为：\n"
                '{"answer":"..."}\n'
                "要求：\n"
                "1) answer 是一个连贯的、完整的最终答案，不要有分组标记；\n"
                "2) 引用编号已在输入中给出，直接沿用即可（无需创建新的引用）；\n"
                "3) 如果某些视频分组的答案只是重复背景常识（如\"SNH48是女子团体\"），"
                "可以合并成一句，不要每个分组都单独说一遍；\n"
                "4) 引用精简：仅当不同引用的论证角度高度相似时才去重保留最具代表性的几个即可；"
                "但如果某个引用提供了独特的信息角度或表达方式，即使都为同一结论服务，也应当保留——独特的表达本身就是亮点。\n"
                "5) 以自然、亲切的口吻回答，就像在跟朋友介绍一样。\n"
                "6) 回答应客观、负责任，避免对主播造成不当误导或负面形象。\n"
                "7) ⚠️ 重要——请仔细阅读每条引用片段的「⚠️ 分析」字段，其中包含了该片段与问题关系的判断以及上下文说明"
                "（如：是否为主播自述/转述、是否确定与问题相关、是否有其他上下文限制等）。"
                "在整合答案时，请尊重这些分析，不要过度解读或脱离片段原有的上下文范围。\n"
                "8) 引用格式必须严格遵守：一个中括号内只能有一个引用编号，格式如 [#1]、[#5]。"
                "禁止使用 [#N-#M]（区间）、[#N, #M]（逗号）、[#N-M]（缺少#号）等非标准格式。\n"
                "8) 引用标记 [#N] 要紧贴所支持的分句之后，不要集中放到句子或段落末尾，更不要统一放到回答最后。\n"
                "仅输出JSON，不要额外文本。"
            ),
        }
    ]
