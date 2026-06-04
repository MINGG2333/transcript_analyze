"""分析与合成：引用构建、多视频分组答案合并。"""

from __future__ import annotations

from typing import Any, Optional

from .models import Segment
from .name_normalizer import normalize_text


def build_kb_background_text(self) -> str:
    """构建知识库背景说明，供分组合成提示使用。

    背景知识（成员档案、平台术语等）仅在最终合并步骤注入，
    避免每次分组合成 LLM 调用都增加 ~10000 chars 的 token 消耗。
    """
    db_bg = f"【数据库背景】{self.kb_description}" if self.kb_description else ""
    parts = [db_bg] if db_bg else []
    return "\n\n".join(parts) if parts else ""


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
                    "video_datetime": s.video_datetime,
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
    question: str, video_summaries_text: str,
    bg_knowledge_text: str = "",
) -> list[dict[str, str]]:
    """构建多视频分组答案合并的 prompt。"""
    bg_section = f"\n\n{bg_knowledge_text}" if bg_knowledge_text else ""
    return [
        {
            "role": "user",
            "content": (
                "你是一名熟知陈嘉仪的粉丝，现在需要将多个视频分组对同一个问题的分析结果"
                "合并成一个连贯、完整的答案。"
                f"{bg_section}\n\n"
                "每个视频分组的分析结果包含该组的初步答案和引用的片段（引用编号已经是全局统一的最终编号）。\n"
                "不要在答案中包含组号和分组的标记，而是自然地整合信息。\n"
                "引用编号 [#N] 已在输入中使用最终编号，输出时继续使用这些编号即可。\n\n"
                f"问题：{question}\n\n"
                "各视频分组分析结果：\n"
                f"{video_summaries_text}\n\n"
                "请输出JSON对象，格式为：\n"
                '{"answer":"..."}\n'
                "要求：\n"
                "1) answer 是一个连贯、完整的最终答案，不要有分组标记；\n"
                "2) 引用编号已在输入中给出，直接沿用即可（无需创建新的引用）；\n"
                "3) 如果某些视频分组的答案只是重复背景常识（如\"SNH48是女子团体\"），"
                "可以合并成一句，不要每个分组都单独说一遍；\n"
                "4) ⚠️ 引用精简（关键）：合并答案时，必须严格执行以下规则：\n"
                "   - 最终答案中的总引用数控制在 10 条以内。\n"
                "   - 当不同引用的论证角度高度相似时，只保留其中最具代表性的 2-3 条即可，其他的可以去掉。\n"
                "   - 但如果某个引用提供了独特的信息角度或表达方式，即使都为同一结论服务，也应当保留——独特的表达本身就是亮点。\n"
                "5) 以自然、亲切的口吻回答，就像在跟朋友介绍一样。\n"
                "6) 回答应客观、负责任，避免对主播造成不当误导或负面形象。\n"
                "7) ⚠️ 重要——请仔细阅读每条引用片段的「⚠️ 分析」字段，其中包含了该片段与问题关系的判断以及上下文说明"
                "（如：是否为主播自述/转述、是否确定与问题相关、是否有其他上下文限制等）。"
                "在整合答案时，请尊重这些分析，不要过度解读或脱离片段原有的上下文范围。\n"
                "8) 🔥 关键——证据矛盾处理（这是最重要的要求）：\n"
                "   不同视频分组之间可能存在矛盾（有的说\"是\"，有的说\"否\"，有的说\"不确定\"）。"
                "请务必做到：\n"
                "   a) **正视矛盾**：不要选择性忽略与你结论不一致的分组。如果证据相互矛盾，必须承认矛盾的存在。\n"
                "   b) **按证据质量分层**：主播亲口自述 > 主播回应/确认弹幕 > 观众弹幕推测 > 间接线索。"
                "使用更强证据作为主要结论，同时说明弱证据的存在。\n"
                "   c) **考虑时间线**：不同分组来自不同日期的直播，主播的情况可能随时间变化。"
                "在分析时注意直播时间的先后顺序，区分\"曾经/过去\"和\"现在/目前\"的不同状态。\n"
                "   d) **区分细微差异**：有些看似矛盾的说法可能说的是不同的事"
                "（例如\"去过北京\"可能指去旅游/公演，\"没去过北京巡演\"则是特定的巡演经历）。"
                "请仔细辨析不同分组的分析细节，如果矛盾可以调和，请给出更精确的结论。\n"
                "   e) **保留不确定性**：如果正反证据均有且无法完美调和，"
                "应在答案中如实反映这种不确定性（如\"目前没有统一说法\"\"不同时间点说法不一\"），"
                "而非强行选边站。\n"
                "9) 引用格式必须严格遵守：一个中括号内只能有一个引用编号，格式如 [#1]、[#5]。"
                "禁止使用 [#N-#M]（区间）、[#N, #M]（逗号）、[#N-M]（缺少#号）等非标准格式。\n"
                "   引用标记 [#N] 要紧贴所支持的分句之后，不要集中放到句子或段落末尾，更不要统一放到回答最后。\n"
                "仅输出JSON，不要额外文本。"
            ),
        }
    ]
