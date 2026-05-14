"""
同音异形词归一化模块。

用于处理语音转录文本中常见的同音不同字问题。
将已知的同音变异写法映射为规范名称，避免 LLM 将同一个实体误判为不同的实体。

三层归一化策略：
  1. **数据层**：入库时对片段文本做归一化（覆写 Segment.text），确保向量/BM25索引基于规范名称
  2. **查询层**：检索时对用户问题做归一化，让「陈佳怡」的查询匹配「陈嘉仪」的片段
  3. **提示层**：拼装 LLM Prompt 时再做一次归一化，兜底防止漏网之鱼

使用方式：
    from .name_normalizer import normalize_text, add_name_mapping
    
    text = normalize_text("陈佳怡是谁")  # → "陈嘉仪是谁"
    add_name_mapping("刘姝娴", "刘姝贤")
    normalize_segments_text(segments)
"""

from __future__ import annotations

import re
from typing import Optional


# ── 规范名称映射表 ──────────────────────────────────────────────────────────
# key:   ASR 可能产生的同音/近音错误写法
# value: 规范名称
#
# ⚠️ 如何高效扩展此表：
#   • 从 qa_archive JSON 中用 grep 找出所有「规范名被误写」的片段
#   • 新增变异时注意优先级：3字词 > 2字词，避免误替换
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_NAME_MAP: dict[str, str] = {
    # ═══════════════════════════════════════════════════════════════
    # SNH48-陈嘉仪 (chén jiā yí)
    # ASR 常见错误模式：
    #   声母: ch- → c-/s-/sh-/zh-/z- (送气清辅音→不送气/平舌/翘舌)
    #   韵母: en → eng/in (前后鼻音不分); ian → iang
    #   声调: 二声→一声/三声/四声
    # ═══════════════════════════════════════════════════════════════
    "陈佳怡": "陈嘉仪",
    "陈家仪": "陈嘉仪",
    "陈佳仪": "陈嘉仪",
    "晨嘉仪": "陈嘉仪",
    "尘嘉仪": "陈嘉仪",
    "陈嘉怡": "陈嘉仪",
    "程嘉仪": "陈嘉仪",
    "辰嘉仪": "陈嘉仪",
    "陈佳宜": "陈嘉仪",
    "陈佳沂": "陈嘉仪",
    "陈佳一": "陈嘉仪",
    "陈嘉一": "陈嘉仪",
    "岑嘉仪": "陈嘉仪",  # cén → chén
    "沈嘉仪": "陈嘉仪",  # shěn → chén
    "曾嘉仪": "陈嘉仪",  # céng/zēng → chén
    "成佳仪": "陈嘉仪",
    "晨佳怡": "陈嘉仪",
    "陈加仪": "陈嘉仪",
    "陈佳亿": "陈嘉仪",
    "陈佳意": "陈嘉仪",
    # ═══════════════════════════════════════════════════════════════
    # SNH48-刘姝贤 (liú shū xián)
    # ASR 常见错误模式：
    #   sh → s (翘→平); ü → u; 贤→娴/含(前后鼻音)
    #   六/留/流 (同音姓氏误写)
    # ═══════════════════════════════════════════════════════════════
    "刘苏贤": "刘姝贤",
    "刘淑贤": "刘姝贤",
    "刘舒贤": "刘姝贤",
    "刘书贤": "刘姝贤",
    "刘姝娴": "刘姝贤",
    "刘殊贤": "刘姝贤",
    "刘姝含": "刘姝贤",
    "刘书娴": "刘姝贤",
    "刘舒娴": "刘姝贤",
    "六姝贤": "刘姝贤",
    "流姝贤": "刘姝贤",
    "留姝贤": "刘姝贤",
    # ═══════════════════════════════════════════════════════════════
    # SNH48-柏欣妤 (bǎi xīn yú)
    # ASR 常见错误模式：
    #   bǎi → bái (上声→阳平); xīn → xīng (前鼻→后鼻)
    #   yú → yǔ/yù (声调)
    # ═══════════════════════════════════════════════════════════════
    "柏昕妤": "柏欣妤",
    "柏新妤": "柏欣妤",
    "白欣妤": "柏欣妤",
    "柏心妤": "柏欣妤",
    "百欣妤": "柏欣妤",
    "柏欣雨": "柏欣妤",
    "白欣雨": "柏欣妤",
    "伯欣妤": "柏欣妤",
    "柏馨予": "柏欣妤",
    "柏新予": "柏欣妤",
    "柏心予": "柏欣妤",
    # ═══════════════════════════════════════════════════════════════
    # SNH48-周诗雨 (zhōu shī yǔ)
    # ASR 常见错误模式：
    #   shī → shí/shǐ/sī/sì (翘舌声调/平翘混淆)
    #   yǔ → yù/yú/yǐ/yi
    # ═══════════════════════════════════════════════════════════════
    "周时雨": "周诗雨",
    "周施雨": "周诗雨",
    "周诗宇": "周诗雨",
    "周诗予": "周诗雨",
    "周诗语": "周诗雨",
    "周思雨": "周诗雨",
    "周丝雨": "周诗雨",
    "舟诗雨": "周诗雨",
    "洲诗雨": "周诗雨",
    "皱诗雨": "周诗雨",
    # ═══════════════════════════════════════════════════════════════
    # SNH48-由淼 (yóu miǎo)
    # ═══════════════════════════════════════════════════════════════
    "由渺": "由淼",
    "由喵": "由淼",
    "由秒": "由淼",
    "尤淼": "由淼",
    "游淼": "由淼",
    "油淼": "由淼",
    # ═══════════════════════════════════════════════════════════════
    # SNH48-王语晨 (wáng yǔ chén)
    # ═══════════════════════════════════════════════════════════════
    "王雨晨": "王语晨",
    "王宇晨": "王语晨",
    "王玉晨": "王语晨",
    "王羽晨": "王语晨",
    "王语辰": "王语晨",
    "王语尘": "王语晨",
    "王语沉": "王语晨",
}

# ── 按长度降序排列，优先匹配更长的（更具体）的变异 ──
# 排除自身到自身的映射（如 "陈嘉仪": "陈嘉仪"），避免无意义替换
_SORTED_VARIANTS = sorted(
    (k for k in CANONICAL_NAME_MAP if k != CANONICAL_NAME_MAP[k]),
    key=len, reverse=True,
)

# 是否启用调试日志
_DEBUG = False


def set_debug(enabled: bool) -> None:
    """设置调试模式。"""
    global _DEBUG
    _DEBUG = enabled


def add_name_mapping(variant: str, canonical: str) -> None:
    """动态添加一个同音映射。
    
    Args:
        variant: ASR 可产生的错误写法
        canonical: 规范名称
    """
    CANONICAL_NAME_MAP[variant] = canonical
    global _SORTED_VARIANTS
    _SORTED_VARIANTS = sorted(
        (k for k in CANONICAL_NAME_MAP if k != CANONICAL_NAME_MAP[k]),
        key=len, reverse=True,
    )


def normalize_text(text: str, debug: bool = False) -> str:
    """对文本中的同音异形词做归一化替换（只在变异与规范名不同时替换）。
    
    Args:
        text: 原始文本
        debug: 是否打印替换日志
    
    Returns:
        归一化后的文本
    """
    result = text
    for variant in _SORTED_VARIANTS:
        canonical = CANONICAL_NAME_MAP[variant]
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


def count_variant_occurrences(segments: list, text_field: str = "text") -> dict[str, int]:
    """统计有多少片段包含哪些同音异形变异（用于诊断/监控）。
    
    Args:
        segments: 片段列表
        text_field: 文本字段名
    
    Returns:
        {变异: 出现次数} 排序字典
    """
    from collections import Counter
    counter: Counter = Counter()
    for seg in segments:
        raw = seg.get(text_field, "") if isinstance(seg, dict) else getattr(seg, text_field, "")
        if not raw:
            continue
        for variant in _SORTED_VARIANTS:
            if variant in raw:
                counter[variant] += 1
    return dict(counter.most_common())
