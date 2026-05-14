"""
同音异形词归一化模块。

用于处理语音转录文本中常见的同音不同字问题。
将已知的同音变异写法映射为规范名称，避免 LLM 将同一个实体误判为不同的实体。
"""

from __future__ import annotations

import re
from typing import Optional


# ── 规范名称映射表 ──────────────────────────────────────────────────────────
# key: 同音异形（ASR 可能产生的错误写法）
# value: 规范名称
# 请根据数据库实际涉及的人物扩展此表
CANONICAL_NAME_MAP: dict[str, str] = {
    # SNH48-陈嘉仪 的常见同音/近音变异
    "陈佳怡": "陈嘉仪",
    "陈家仪": "陈嘉仪",
    "陈佳仪": "陈嘉仪",
    "晨嘉仪": "陈嘉仪",
    "尘嘉仪": "陈嘉仪",
    "陈嘉怡": "陈嘉仪",
    "程嘉仪": "陈嘉仪",
}

# ── 按长度降序排列，优先匹配更长的（更具体）的变异 ──
_SORTED_VARIANTS = sorted(CANONICAL_NAME_MAP.keys(), key=len, reverse=True)

# 是否启用调试日志
_DEBUG = False


def set_debug(enabled: bool) -> None:
    """设置调试模式。"""
    global _DEBUG
    _DEBUG = enabled


def add_name_mapping(variant: str, canonical: str) -> None:
    """动态添加一个同音映射。"""
    CANONICAL_NAME_MAP[variant] = canonical
    # 重新排序
    global _SORTED_VARIANTS
    _SORTED_VARIANTS = sorted(CANONICAL_NAME_MAP.keys(), key=len, reverse=True)


def normalize_text(text: str, debug: bool = False) -> str:
    """对文本中的同音异形词做归一化替换。
    
    Args:
        text: 原始文本
        debug: 是否打印替换日志
    
    Returns:
        归一化后的文本
    """
    result = text
    for variant in _SORTED_VARIANTS:
        canonical = CANONICAL_NAME_MAP[variant]
        # 用正则做全词/子串替换（不区分边界，因为中文没有空格分词）
        count = 0
        result, count = re.subn(re.escape(variant), canonical, result)
        if count > 0 and (debug or _DEBUG):
            print(f"[NameNormalizer] 替换: '{variant}' → '{canonical}' (共 {count} 处)")
    return result


def normalize_segments_text(segments: list, text_field: str = "text") -> list:
    """对片段列表中的文本字段做批量归一化。
    
    Args:
        segments: 片段对象列表（支持 dict 或 object 类型）
        text_field: 文本字段名（用于 dict），若为 object 则为属性名
    
    Returns:
        归一化后的片段列表（原地修改）
    """
    for seg in segments:
        if isinstance(seg, dict):
            raw = seg.get(text_field, "")
            if raw:
                seg[text_field] = normalize_text(raw)
        else:
            raw = getattr(seg, text_field, "")
            if raw:
                setattr(seg, text_field, normalize_text(raw))
    return segments
