"""内容安全审核：对最终答案和引用进行分级风险判定。
"""

from __future__ import annotations

from typing import Any

from .models import RiskLevel
from .qa_utils import call_llm_json


def check_content_safety(
    self,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> tuple[RiskLevel, str, dict | None]:
    """内容安全审核：对最终答案和引用进行分级风险判定。

    安全/低风险放行，中/高风险拦截。

    Returns:
        (risk_level, reason, llm_metadata)
        risk_level: SAFE(0)=放行 / LOW(1)=放行 / MEDIUM(2)=拦截 / HIGH(3)=拦截
        reason: 判断理由
        llm_metadata: LLM 调用元数据（含 token 用量），异常时为 None
    """
    if not answer:
        return RiskLevel.SAFE, "无答案内容", None

    # 构建完整引用信息（不截断，方便 LLM 做准确判断）
    citation_summary = ""
    if citations:
        citation_lines = []
        for c in citations:
            cid = c.get("citation_id", "")
            ctype = c.get("citation_type", "")
            reason = c.get("reason", "")
            segs = c.get("segments", []) or []
            seg_texts = []
            for s in segs:
                text = s.get("quoted_text", "")
                if text:
                    seg_texts.append(text)
            seg_summary = " | ".join(seg_texts)
            citation_lines.append(f"  [{cid}] 类型={ctype} 理由={reason} 内容={seg_summary}")
        citation_summary = "\n".join(citation_lines)

    safety_prompt = [
        {
            "role": "user",
            "content": (
                "你是一名细致的内容安全审核员。请根据以下四级风险分类标准，"
                "判断下面 AI 生成的回答及其引用片段属于哪个风险等级。\n\n"
                "━━━ 风险等级定义 ━━━\n\n"
                "【安全 - level 0】\n"
                "完全正面、中性、或纯粹的日常信息，不包含任何负面词汇、脏话、争议话题、\n"
                "违法暗示等可能产生负面效果的内容。\n"
                "包括：纯事实陈述、正面评价、日常无争议交流、粉丝圈内常见用语/梗/文化。\n\n"
                "【低风险 - level 1】\n"
                "包含轻微负面词汇、轻度脏话、轻度负面情绪或吐槽。\n"
                "包括：\"好烦\"\"累了\"\"无语\"等轻度负面情绪词、轻度吐槽。\n"
                "注意：低风险内容通常无需拦截，仅做标记参考。\n\n"
                "【中风险 - level 2】\n"
                "包含以下任一内容，均判为中风险：\n"
                "  - 明确脏话（如\"他妈\"\"靠\"\"操\"等）\n"
                "  - 人身攻击性用语\n"
                "  - 涉及毒品、吸毒、制毒、贩毒等内容\n"
                "  - 涉及政治/政策讨论（政治人物、政治事件、政党、政府、意识形态、政策法规等）\n"
                "  - 涉及色情内容\n"
                "  - 涉及暴力、恐怖主义\n"
                "  - 涉及违法活动（赌博、诈骗、洗钱等）\n"
                "  - 任何与毒品、政治、色情、暴力、违法相关的讨论，无论语境\n\n"
                "【高风险 - level 3】\n"
                "明确严重的违规内容，必须拦截。\n"
                "包括：明确的脏话/人身攻击（如直接辱骂他人）、违法信息、涉政涉黄涉暴内容、\n"
                "对特定个人的恶意中伤或诽谤、严重的道德争议行为。\n\n"
                "━━━ 判断原则 ━━━\n\n"
                "1) 毒品、政治、色情、暴力、违法活动——零容忍。\n"
                "   只要回答或引用中涉及这些话题，无论语境、无论正面负面、无论是否转述，一律判为中风险(level 2)以上。\n"
                "   陈嘉仪作为公众人物，不应与这些话题产生任何关联。\n\n"
                "2) 其他内容区分恶意与日常表达。\n"
                "   - 粉丝文化、网络流行语、圈内用语（如梗、爱称、玩笑）如非明显恶意，不应判定为中高风险。\n"
                "   - 但明确脏话、人身攻击等，无论出现在什么语境中，均应判定为中高风险。\n\n"
                "3) 不确定时，保守处理。\n"
                "   - 无法确定是否属于中风险还是低风险时，取中风险（level 2）。\n"
                "   - 宁可多拦截一条，不可漏放一条真正有风险的内容。\n\n"
                f"用户问题：{question}\n\n"
                "AI 生成的回答：\n"
                f"{answer}\n\n"
                "引用片段：\n"
                f"{citation_summary}\n\n"
                "请输出JSON对象：\n"
                '{"level": 0~3, "reason": "判断理由（中文，50字以内，说明归入该等级的关键依据）"}\n'
                "其中 level 含义：0=安全, 1=低风险, 2=中风险, 3=高风险\n"
                "⚠️ 思考不超过150字，确保思考+输出总量在限制内。\n"
                "仅输出JSON，不要额外文本。"
            ),
        }
    ]

    try:
        # CHANGED: 必须串行——依赖最终答案内容和 citations，不能提前并发
        parsed, llm_meta = call_llm_json(
            self.client, self.llm_model, safety_prompt, "内容安全审核",
            max_tokens=300, logger=self.logger,
        )
        level = parsed.get("level", -1)
        reason = parsed.get("reason", "未提供理由")

        # 校验返回的 level 是否在有效范围内
        if level == 0:
            return RiskLevel.SAFE, str(reason), llm_meta
        elif level == 1:
            return RiskLevel.LOW, str(reason), llm_meta
        elif level == 2:
            return RiskLevel.MEDIUM, str(reason), llm_meta
        elif level == 3:
            return RiskLevel.HIGH, str(reason), llm_meta
        else:
            if self.logger:
                self.logger.warning(f"内容安全审核返回异常的 level 值: {level}，默认判为中风险")
            return RiskLevel.MEDIUM, f"异常风险等级({level}): {reason}", llm_meta
    except Exception as exc:
        if self.logger:
            self.logger.warning(f"内容安全审核 LLM 调用失败: {exc}")
        return RiskLevel.MEDIUM, f"审核调用失败: {exc}", None


def sanitize_content(
    self,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    safety_reason: str,
) -> tuple[str, list[dict[str, Any]], dict | None]:
    """尝试净化不安全的内容，返回净化后的 answer 和 citations。

    当安全审核不通过时调用，让 LLM 尝试删除不安全部分，
    仅保留安全可发布的内容。

    Returns:
        (new_answer, new_citations, llm_metadata)
        - 净化成功时 new_answer 非空，new_citations 为保留的引用
        - 失败或异常时返回 ("", [], None)
    """
    if not answer:
        return "", [], None

    # 构建引用摘要
    citation_summary = ""
    if citations:
        citation_lines = []
        for c in citations:
            cid = c.get("citation_id", "")
            ctype = c.get("citation_type", "")
            reason = c.get("reason", "")
            segs = c.get("segments", []) or []
            seg_texts = []
            for s in segs:
                text = s.get("quoted_text", "")
                if text:
                    seg_texts.append(text)
            seg_summary = " | ".join(seg_texts)
            citation_lines.append(f"  [{cid}] 类型={ctype} 理由={reason} 内容={seg_summary}")
        citation_summary = "\n".join(citation_lines)

    sanitize_prompt = [
        {
            "role": "user",
            "content": (
                "你是一名内容安全编辑。以下 AI 生成的回答因包含不安全内容未能通过审核。\n"
                "请尝试净化该回答，删除所有不安全、负面、争议性的部分，仅保留安全可发布的内容。\n\n"
                "规则：\n"
                "1. 删除回答中包含不安全信息的句子或引用标记 [#N]\n"
                "2. 同时从 kept_citation_ids 中排除对应的不安全引用\n"
                "3. 保持剩余内容的连贯性，可以适当调整语句衔接\n"
                "4. 如果删除后完全没有安全内容可保留，answer 输出空字符串\n\n"
                f"用户问题：{question}\n\n"
                f"原回答（未通过审核）：\n{answer}\n\n"
                "引用片段：\n"
                f"{citation_summary}\n\n"
                f"未通过原因：{safety_reason}\n\n"
                '请输出JSON：\n'
                '{"answer": "净化后的回答", "kept_citation_ids": ["#1", "#3"]}\n'
                "其中 kept_citation_ids 是保留的引用编号列表，已删除的不安全引用编号不要包含。"
            ),
        }
    ]

    try:
        parsed, llm_meta = call_llm_json(
            self.client, self.llm_model, sanitize_prompt, "内容安全净化",
            max_tokens=500, logger=self.logger,
        )
        new_answer = (parsed.get("answer") or "").strip()
        kept_ids = parsed.get("kept_citation_ids", []) or []

        # 过滤 citations，只保留被标记为安全的引用
        new_citations = [
            c for c in citations
            if c.get("citation_id", "") in kept_ids
        ]

        return new_answer, new_citations, llm_meta
    except Exception as exc:
        if self.logger:
            self.logger.warning(f"内容安全净化 LLM 调用失败: {exc}")
        return "", [], None
