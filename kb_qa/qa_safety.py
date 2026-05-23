"""内容安全审核：对最终答案和引用进行分级风险判定。"""

from __future__ import annotations

from typing import Any

from .models import RiskLevel
from .qa_utils import call_llm_json


def check_content_safety(
    self,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> tuple[RiskLevel, str]:
    """内容安全审核：对最终答案和引用进行分级风险判定。

    采用四级风险等级，取代旧版的二值判定（安全/不安全）。
    只拦截中风险和高风险，低风险和安全直接放行。

    Returns:
        (risk_level: RiskLevel, reason: str)
        - SAFE (0): 安全，直接展示
        - LOW (1):  低风险，展示但不加提示
        - MEDIUM (2): 中风险，拦截不展示
        - HIGH (3):  高风险，拦截不展示
    """
    if not answer:
        return RiskLevel.SAFE, "无答案内容"

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
                "日常对话、圈内文化、主播个性表达、中性或正面信息。\n"
                "包括：粉丝圈常见用语、主播的个人口头禅/语气词、对日常生活的普通讨论。\n\n"
                "【低风险 - level 1】\n"
                "轻微吐槽、负面情绪表达，但无主观恶意且不针对具体个人。\n"
                "包括：对一般事物表达不满或失望（如\"好难\"\"好烦\"\"累了\"）、\n"
                "轻度自嘲、对非特定对象的温和抱怨。这些属于日常情绪表达，不构成攻击。\n\n"
                "【中风险 - level 2】\n"
                "可能引起争议或对主播形象造成一定负面影响的内容。\n"
                "包括：相对明确的负面行为描述、观众弹幕中的质疑或负面评论、\n"
                "主播在特定情境下使用的不太恰当的用语、涉及敏感话题但非明确违规的讨论。\n"
                "此类内容建议拦截，避免引发不必要的争议。\n\n"
                "【高风险 - level 3】\n"
                "明确严重的违规内容，必须拦截。\n"
                "包括：明确的脏话/人身攻击（如直接辱骂他人）、违法信息、涉政涉黄涉暴内容、\n"
                "对特定个人的恶意中伤或诽谤、严重的道德争议行为。\n\n"
                "━━━ 判断指南 ━━━\n\n"
                "1) 区分「恶意攻击」与「日常表达」：\n"
                "   - 判断标准不是「有没有负面字眼」，而是「是否有主观恶意」。\n"
                "   - 日常口头禅、语气感叹词（如\"哎哟\"\"天哪\"\"我去\"）即使字面略带负面，\n"
                "     如果明显是日常表达习惯，应判为安全或低风险，而不是中/高风险。\n"
                "   - 主播在讲述经历时自然流露的情绪（如\"当时真的好累\"\"太难了\"）属于正常表达。\n\n"
                "2) 区分「粉丝圈文化」与「真实负面」：\n"
                "   - 粉丝圈内部常见的调侃、玩梗、爱称互损是圈内文化，不应判定为高风险。\n"
                "   - 例如粉丝说\"你好笨\"\"你好菜\"等带有亲密感的互动，属于粉丝文化。\n\n"
                "3) 区分「转述」与「自述」：\n"
                "   - 答案中如果引用了主播转述他人（观众弹幕/他人评价）的内容，\n"
                "     应优先判断转述内容的性质，而非直接将其等同于主播的真实立场。\n"
                "   - 主播念弹幕中的负面内容属于回应观众互动，风险应低于主播本人主动表达。\n\n"
                "4) 注意「字面负面」与「真实意图」的差异：\n"
                "   - 不要仅因为出现了某个词的字面意思就判定为高风险。\n"
                "   - 结合上下文判断：是否有真实恶意？是否针对特定个人？\n"
                "   - 如果回答整体是正面或中性的，只是引用了片段中主播的某个语气词，不应升级。\n\n"
                "5) 关于「否定/澄清类内容」：\n"
                "   - 如果回答引用了主播对某事的否认或澄清（如\"我没说过那种话\"\"那是开玩笑的\"），\n"
                "     本身不构成负面信息，不应因此升级风险等级。\n"
                "   - 但如果同时引用了大量负面片段，即使有澄清，也应如实评估整体风险。\n\n"
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
        parsed, _ = call_llm_json(
            self.client, self.llm_model, safety_prompt, "内容安全审核",
            max_tokens=300, logger=self.logger,
        )
        level = parsed.get("level", -1)
        reason = parsed.get("reason", "未提供理由")

        # 校验返回的 level 是否在有效范围内
        if level == 0:
            return RiskLevel.SAFE, str(reason)
        elif level == 1:
            return RiskLevel.LOW, str(reason)
        elif level == 2:
            return RiskLevel.MEDIUM, str(reason)
        elif level == 3:
            return RiskLevel.HIGH, str(reason)
        else:
            if self.logger:
                self.logger.warning(f"内容安全审核返回异常的 level 值: {level}，默认判为中风险")
            return RiskLevel.MEDIUM, f"异常风险等级({level}): {reason}"
    except Exception as exc:
        if self.logger:
            self.logger.warning(f"内容安全审核 LLM 调用失败: {exc}")
        return RiskLevel.MEDIUM, f"审核调用失败: {exc}"
