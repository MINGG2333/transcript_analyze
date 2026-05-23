"""分析与合成：引用构建、多视频分组答案合并。"""

from __future__ import annotations

from typing import Any

from .models import Segment
from .name_normalizer import normalize_text


def build_kb_background_text(self) -> str:
    """构建知识库背景说明，供分析/合成提示使用。"""
    if self.kb_description:
        return f"【数据库背景】{self.kb_description}"
    return ""


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
                "   引用标记 [#N] 要紧贴所支持的分句之后，不要集中放到句子或段落末尾，更不要统一放到回答最后。\n"
                "9) 回答应简洁，控制在 500 字以内。\n"
                "仅输出JSON，不要额外文本。"
            ),
        }
    ]
