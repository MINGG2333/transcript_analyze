#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_theme_codebooks.py

调用 LLM（DeepSeek）将多个访谈问题的 codebook 智能合并为一个主题的合并 codebook。

核心思路：
  对于每个主题及其子主题（如 A1 通信信道 + (a) 技术上的防护措施），
  将该主题范围内所有相关问题的原始访谈回答汇总，一次性交给 LLM，
  让 LLM 理解语义后重新编码，输出一个统一的合并 codebook。

  与简单的 code_set 拼接不同，LLM 能识别不同问题中语义相近的 code 并合并，
  从而得出更精简、更高质量的合并结果。

输出格式与原 per_question codebook 完全一致（JSON + Markdown），
便于后续直接使用。

运行方式：
  python3 merge_theme_codebooks.py
  python3 merge_theme_codebooks.py --output-dir codebook/merged --llm-model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ─── 日志 ──────────────────────────────────────────────────────────────────
def setup_logger(debug: bool = False):
    level = "DEBUG" if debug else "INFO"
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level=level)
        return logger
    except ImportError:
        class SimpleLogger:
            def info(self, m): print(f"[INFO] {m}")
            def success(self, m): print(f"[SUCCESS] {m}")
            def warning(self, m): print(f"[WARN] {m}")
            def error(self, m): print(f"[ERROR] {m}")
            def debug(self, m): pass
            def critical(self, m): print(f"[CRIT] {m}")
        return SimpleLogger()


def ensure_client(llm_model: str, api_base: Optional[str], api_key: Optional[str], logger: Any):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("缺少 openai 依赖，请先执行: pip install openai")
    client = OpenAI(api_key=api_key, base_url=api_base) if api_key and api_base else \
             OpenAI(api_key=api_key) if api_key else \
             OpenAI(base_url=api_base) if api_base else OpenAI()
    return client


def call_llm_json(
    client: Any, llm_model: str, messages: list[dict[str, str]],
    description: str, logger: Any, max_retries: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_raw = None
    for attempt in range(max_retries):
        try:
            logger.info(f"  调用LLM: {description} (尝试 {attempt+1}/{max_retries})")
            resp = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            last_raw = content

            llm_meta = {
                "model": llm_model,
                "description": description,
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "output_tokens": getattr(resp.usage, "completion_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "response": content,
            }

            logger.info(f"    tokens: input={llm_meta['input_tokens']}, output={llm_meta['output_tokens']}, total={llm_meta['total_tokens']}")

            try:
                parsed = json.loads(content)
                llm_meta["success"] = True
                return parsed, llm_meta
            except json.JSONDecodeError:
                logger.warning(f"  JSON解析失败: {description}")
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(content[start: end + 1])
                        llm_meta["success"] = True
                        llm_meta["note"] = "通过字符串提取成功解析"
                        return parsed, llm_meta
                    except json.JSONDecodeError:
                        pass
                if attempt < max_retries - 1:
                    wait = 5 ** attempt
                    logger.warning(f"  JSON解析失败，等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                else:
                    llm_meta["success"] = False
                    llm_meta["error"] = f"无法将LLM响应解析为JSON: {content}"
                    raise RuntimeError(llm_meta["error"])

        except Exception as e:
            logger.warning(f"  LLM请求异常 (第 {attempt+1} 次): {e}")
            if attempt < max_retries - 1:
                wait = 5 ** attempt
                logger.warning(f"  等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            else:
                error_msg = f"LLM调用在 {max_retries} 次重试后仍失败: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e


# ─── 主题配置 ──────────────────────────────────────────────────────────────
THEMES = OrderedDict([
    ("A1_通信信道", {
        "title": "A1 通信信道（V2X / 车内总线）",
        "question_ranges": [
            "Q4.1.1", "Q4.1.2", "Q4.1.3", "Q4.1.4", "Q4.1.5",
            "Q4.1.6", "Q4.1.7", "Q4.1.8", "Q4.1.9", "Q4.1.10",
            "Q4.1.11", "Q4.1.12", "Q4.1.13", "Q4.1.14", "Q4.1.15",
            "Q4.1.16", "Q4.1.17", "Q4.1.18", "Q4.1.19", "Q4.1.20",
        ],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("A2_软件更新", {
        "title": "A2 软件更新",
        "question_ranges": ["Q4.2.1", "Q4.2.2", "Q4.2.3", "Q4.2.4", "Q4.2.5"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("A3_外部接口与远程控制", {
        "title": "A3 外部接口与远程控制",
        "question_ranges": ["Q4.4.1", "Q4.4.2", "Q4.4.3", "Q4.4.4", "Q4.4.5", "Q4.4.6", "Q4.4.7", "Q4.8.1"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("A4_后端服务器", {
        "title": "A4 后端服务器",
        "question_ranges": ["Q4.9.1", "Q4.9.2", "Q4.9.3", "Q4.9.4", "Q4.9.5", "Q4.9.6"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("A5_数据与代码完整性", {
        "title": "A5 数据与代码完整性（应用安全与数据安全）",
        "question_ranges": [
            "Q4.5.1", "Q4.5.2", "Q4.5.3", "Q4.5.4", "Q4.5.5",
            "Q4.5.6", "Q4.5.7", "Q4.5.8", "Q4.5.9", "Q4.5.10",
            "Q4.5.11", "Q4.5.12", "Q4.5.13", "Q4.5.14",
            "Q4.7.1",
            "Q4.11.1", "Q4.11.2", "Q4.11.3",
        ],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("A6_系统加固不足", {
        "title": "A6 系统加固不足",
        "question_ranges": [
            "Q4.6.1", "Q4.6.2", "Q4.6.3", "Q4.6.4",
            "Q4.6.5", "Q4.6.6", "Q4.6.7", "Q4.6.8",
        ],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("A7_人为因素与流程", {
        "title": "A7 人为因素与流程",
        "question_ranges": ["Q4.3.1", "Q4.3.2", "Q4.10.1", "Q4.10.2"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 技术上的防护措施", "suffixes": [".2"], "type": "A"}),
            ("b", {"label": "(b) 技术上的挑战",     "suffixes": [".3"], "type": "A"}),
        ]),
    }),
    ("B1_组织与项目网络安全管理", {
        "title": "B1 组织与项目网络安全管理（公司与项目管理体系建立）",
        "question_ranges": ["Q4.13.1", "Q4.13.2", "Q4.13.3", "Q4.13.4"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 组织上的防护措施", "type": "B"}),
            ("b", {"label": "(b) 组织上的挑战",     "type": "B"}),
        ]),
    }),
    ("B2_持续网络安全活动", {
        "title": "B2 持续网络安全活动（持续监测等）",
        "question_ranges": ["Q4.14.1", "Q4.14.2", "Q4.14.3", "Q4.14.4"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 组织上的防护措施", "type": "B"}),
            ("b", {"label": "(b) 组织上的挑战",     "type": "B"}),
        ]),
    }),
    ("B3_分布式网络安全活动", {
        "title": "B3 分布式网络安全活动（供应链安全）",
        "question_ranges": ["Q4.15"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 组织上的防护措施", "type": "B"}),
            ("b", {"label": "(b) 组织上的挑战",     "type": "B"}),
        ]),
    }),
    ("B4_产品生命周期网络安全", {
        "title": "B4 产品生命周期网络安全",
        "question_ranges": [
            "Q4.16.1", "Q4.16.2", "Q4.16.3", "Q4.16.4",
            "Q4.17.1", "Q4.17.2", "Q4.17.3", "Q4.17.4",
            "Q4.17.5", "Q4.17.6", "Q4.17.7",
        ],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 组织上的防护措施", "type": "B"}),
            ("b", {"label": "(b) 组织上的挑战",     "type": "B"}),
        ]),
    }),
    ("B5_风险评估方法论_TARA", {
        "title": "B5 风险评估方法论 / TARA",
        "question_ranges": ["Q4.18.1", "Q4.18.2", "Q4.18.3"],
        "sub_themes": OrderedDict([
            ("a", {"label": "(a) 组织上的防护措施", "type": "B"}),
            ("b", {"label": "(b) 组织上的挑战",     "type": "B"}),
        ]),
    }),
])


# ─── 问题筛选 ──────────────────────────────────────────────────────────────
def load_all_per_question_json(per_question_dir: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for json_path in sorted(per_question_dir.glob("*.json")):
        try:
            entry = json.loads(json_path.read_text(encoding="utf-8"))
            qid = entry.get("question_id", json_path.stem)
            entries[qid] = entry
        except Exception as e:
            print(f"  [WARN] 跳过 {json_path.name}: {e}")
    return entries


def question_matches_range(qid: str, ranges: list[str]) -> bool:
    """判断 question_id 是否匹配某个 range 前缀（精确匹配，防止 Q4.1.1 错误匹配 Q4.10.x）"""
    for r in ranges:
        # 完全相等
        if qid == r:
            return True
        # qid 以 "r." 开头（确保是层级边界，而非字符串前缀）
        if qid.startswith(r + "."):
            return True
    return False


def question_matches_suffix(qid: str, suffixes: list[str]) -> bool:
    for suffix in suffixes:
        if qid.endswith(suffix):
            return True
    return False


def is_b_type_challenge_question(question_text: str) -> bool:
    return "挑战" in question_text or "障碍" in question_text


def filter_questions_for_subtheme(
    theme_config: dict,
    sub_key: str,
    all_entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranges = theme_config["question_ranges"]
    sub_config = theme_config["sub_themes"][sub_key]
    sub_type = sub_config["type"]
    suffixes = sub_config.get("suffixes", [])

    matched: list[dict[str, Any]] = []
    for qid, entry in all_entries.items():
        if not question_matches_range(qid, ranges):
            continue
        if sub_type == "A":
            if question_matches_suffix(qid, suffixes):
                matched.append(entry)
        elif sub_type == "B":
            qtext = entry.get("question_text", "")
            is_challenge = is_b_type_challenge_question(qtext)
            if sub_key == "a" and not is_challenge:
                matched.append(entry)
            elif sub_key == "b" and is_challenge:
                matched.append(entry)
    return matched


# ─── LLM Prompt 构建 ──────────────────────────────────────────────────────
def build_merge_prompt(
    theme_title: str,
    sub_label: str,
    matched_entries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    构建用于合并 codebook 的 LLM prompt。

    将多个问题（每个问题都有自己的问题文本、访谈回答和已有 code_set）
    一次性呈现给 LLM，并要求 LLM ：
    1. 理解所有回答，基于语义对所有回答重新编码
    2. 合并语义相似的 code
    3. 输出统一的一份 code_set 和每访谈的 code 分配
    """
    lines = []
    lines.append(f"主题：{theme_title}")
    lines.append(f"子主题：{sub_label}")
    lines.append("")
    lines.append(f"共包含 {len(matched_entries)} 个访谈问题的回答数据。")
    lines.append("请综合分析所有问题的回答，输出一份统一的、合并后的编码方案。")
    lines.append("")

    for entry in matched_entries:
        qid = entry["question_id"]
        qtext = entry["question_text"]
        i_answers = entry.get("interview_answers", {})

        lines.append(f"=== 问题 {qid}：{qtext} ===")
        for inv_name in sorted(i_answers.keys()):
            answer = i_answers[inv_name] or "（空）"
            lines.append(f"[{inv_name}] {answer}")
        lines.append("")

    answers_block = "\n".join(lines)

    system_prompt = (
        "你是一个严谨的访谈回答分析助手。你的任务是对多个相关问题的所有访谈回答进行统一的编码分析。\n\n"
        "以下是多个访谈问题的回答数据，每个问题有自己的问题文本和各访谈的回答。\n"
        "这些问题的主题相同（均属于同一主题的子主题），因此回答具有可比性和可合并性。\n\n"
        "请遵循以下步骤：\n"
        "1. 仔细阅读所有问题的所有访谈回答\n"
        "2. 跨越问题边界，识别出所有不同的回答模式/类型，将其精炼为简洁的\"code\"\n"
        "3. 对于语义上非常接近的 code（不同问题中可能使用不同表述），应合并为一个统一的 code\n"
        "4. 每个访谈在每道问题下的回答可能对应零个、一个或多个code\n"
        "5. 对于每个访谈，如果其某道题的回答为空，应在那个问题的上下文中包含code \"空\"\n"
        "6. 如果某道题的回答不确定或模棱两可，应在那个问题的上下文中包含code \"未明确回答\"\n\n"
        "重要：所有输出（包括code_set中的code名称）必须全部使用中文，不要使用英文。\n\n"
        "请严格按以下JSON格式输出（统一的合并结果）：\n"
        "{\n"
        '  "code_set": ["code1", "code2", ...],\n'
        '  "interview_codes": {\n'
        '    "访谈名称1": ["code1", "code2"],\n'
        '    "访谈名称2": ["code3", "code4"],\n'
        "    ...\n"
        "  }\n"
        "}\n\n"
        "要求：\n"
        "- code_set 必须覆盖所有回答中出现的所有code，并将语义相近的code合并\n"
        "- code名称必须使用中文\n"
        "- interview_codes 中每个访谈的code必须从code_set中选择\n"
        "- 每个访谈的code应该是其所有问题回答中提及的code的总和（去重）\n"
        "- 如果某个访谈在所有问题中的回答均为空，其code应为[\"空\"]"
    )

    user_prompt = (
        f"{answers_block}\n\n"
        "请综合分析以上所有问题的回答，输出统一合并后的code_set和interview_codes。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ─── 合并输出构建 ─────────────────────────────────────────────────────────
def build_merged_entry(
    theme_title: str,
    sub_label: str,
    matched_entries: list[dict[str, Any]],
    parsed: dict[str, Any],
    llm_meta: dict[str, Any],
) -> dict[str, Any]:
    """将 LLM 返回的合并结果构建为标准格式的 codebook entry"""

    all_qids = sorted(set(e["question_id"] for e in matched_entries))
    code_set = parsed.get("code_set", [])
    interview_codes = parsed.get("interview_codes", {})

    # 收集所有访谈名称和答案
    all_interview_names: set[str] = set()
    interview_answers_combined: dict[str, list[dict]] = {}
    for e in matched_entries:
        i_answers = e.get("interview_answers", {})
        for inv_name, answer in i_answers.items():
            all_interview_names.add(inv_name)
            if inv_name not in interview_answers_combined:
                interview_answers_combined[inv_name] = []
            interview_answers_combined[inv_name].append({
                "question_id": e["question_id"],
                "question_text": e["question_text"],
                "answer": answer,
                "citation_path": e.get("interview_citations", {}).get(inv_name, ""),
            })

    total_interviews = len(all_interview_names)

    # code_counts: 统计每个code在多少访谈中出现
    code_counts: dict[str, int] = {c: 0 for c in code_set}
    for inv_name, codes in interview_codes.items():
        for c in codes:
            if c in code_counts:
                code_counts[c] += 1
            else:
                code_counts[c] = code_counts.get(c, 0) + 1

    code_percentages: dict[str, float] = {
        c: round(cnt / total_interviews * 100, 2) if total_interviews > 0 else 0.0
        for c, cnt in code_counts.items()
    }

    # code_interviews: 每个code关联的访谈详情
    code_interviews: dict[str, list[dict[str, Any]]] = {}
    for c in code_set:
        code_interviews[c] = []
    for inv_name, codes in interview_codes.items():
        for c in codes:
            if c in code_interviews:
                # 找到该访谈在哪些问题中提到了这个code
                context_questions = []
                for qinfo in interview_answers_combined.get(inv_name, []):
                    if qinfo["answer"] and qinfo["answer"] != "（空）":
                        context_questions.append(qinfo["question_id"])
                code_interviews[c].append({
                    "interview_name": inv_name,
                    "answer_context": f"涉及问题: {', '.join(context_questions)}" if context_questions else "",
                    "citation_path": interview_answers_combined.get(inv_name, [{}])[0].get("citation_path", ""),
                })

    return OrderedDict([
        ("theme", theme_title),
        ("sub_theme", sub_label),
        ("total_interviews", total_interviews),
        ("total_questions", len(matched_entries)),
        ("questions_included", all_qids),
        ("code_set", sorted(code_set)),
        ("code_counts", OrderedDict(sorted(code_counts.items()))),
        ("code_percentages", OrderedDict(sorted(
            code_percentages.items(), key=lambda x: x[1], reverse=True,
        ))),
        ("code_interviews", OrderedDict(sorted(code_interviews.items()))),
        ("interview_codes", OrderedDict(sorted(interview_codes.items()))),
        ("llm_metadata", {
            "model": llm_meta.get("model", ""),
            "description": llm_meta.get("description", ""),
            "input_tokens": llm_meta.get("input_tokens", 0),
            "output_tokens": llm_meta.get("output_tokens", 0),
            "total_tokens": llm_meta.get("total_tokens", 0),
            "success": llm_meta.get("success", False),
            "error": llm_meta.get("error", ""),
            "timestamp": llm_meta.get("timestamp", ""),
        }),
        ("generated_at", datetime.now().isoformat(timespec="seconds")),
    ])


# ─── Markdown 输出 ────────────────────────────────────────────────────────
def _md_relative_to(output_dir: Path, citation_path: str) -> str:
    if not citation_path:
        return ""
    citation_abs = Path(citation_path).resolve()
    try:
        return os.path.relpath(citation_abs, output_dir.resolve())
    except ValueError:
        return citation_path


def generate_markdown(merged_entry: dict[str, Any], output_dir: Path) -> str:
    lines = []
    theme = merged_entry["theme"]
    sub = merged_entry["sub_theme"]
    total_interviews = merged_entry["total_interviews"]
    total_questions = merged_entry["total_questions"]
    questions_included = merged_entry["questions_included"]
    code_set = merged_entry["code_set"]
    code_counts = merged_entry["code_counts"]
    code_percentages = merged_entry["code_percentages"]
    code_interviews = merged_entry["code_interviews"]
    interview_codes = merged_entry["interview_codes"]
    llm_meta = merged_entry.get("llm_metadata", {})

    lines.append(f"# {theme}\n")
    lines.append(f"## {sub}\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**访谈总数：** {total_interviews}\n")
    lines.append(f"**合并问题数：** {total_questions}\n")
    lines.append(f"**包含问题：** {', '.join(questions_included)}\n")
    lines.append("---\n")

    # 编码统计
    lines.append("## 编码统计（LLM智能合并）\n")
    lines.append("| Code | 出现次数 | 占比 |")
    lines.append("|------|----------|------|")
    for c in sorted(code_set):
        cnt = code_counts.get(c, 0)
        pct = code_percentages.get(c, 0.0)
        lines.append(f"| {c} | {cnt} | {pct}% |")
    lines.append("")

    # 各 Code 详情
    lines.append("## 各 Code 详情\n")
    for c in sorted(code_set):
        interview_list = code_interviews.get(c, [])
        cnt = code_counts.get(c, 0)
        pct = code_percentages.get(c, 0.0)
        lines.append(f"**{c}**（{cnt}个访谈，{pct}%）")
        if interview_list:
            for item in interview_list:
                inv_name = item.get("interview_name", "")
                context = item.get("answer_context", "")
                citation = item.get("citation_path", "")
                if citation:
                    rel = _md_relative_to(output_dir, citation)
                    lines.append(f"- **{inv_name}**：[详情]({rel}) {context}")
                else:
                    lines.append(f"- **{inv_name}**：{context}")
        else:
            lines.append("- （无）")
        lines.append("")

    # 各访谈编码总览
    lines.append("## 各访谈编码总览\n")
    lines.append("| 访谈 | Code |")
    lines.append("|------|------|")
    for inv_name in sorted(interview_codes.keys()):
        codes = interview_codes.get(inv_name, [])
        codes_str = ", ".join(codes) if codes else "（无）"
        lines.append(f"| {inv_name} | {codes_str} |")
    lines.append("")

    # LLM 元数据
    lines.append("## LLM元数据\n")
    lines.append(f"- **模型：** {llm_meta.get('model', 'N/A')}")
    lines.append(f"- **Token用量：** 输入 {llm_meta.get('input_tokens', 'N/A')} / 输出 {llm_meta.get('output_tokens', 'N/A')} / 总计 {llm_meta.get('total_tokens', 'N/A')}")
    lines.append(f"- **状态：** {'成功' if llm_meta.get('success', False) else '失败'}")
    lines.append(f"- **时间：** {llm_meta.get('timestamp', 'N/A')}")
    if llm_meta.get("error"):
        lines.append(f"- **错误：** {llm_meta['error']}")
    lines.append("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="调用LLM智能合并主题codebook")
    parser.add_argument("--per-question-dir", default="codebook/per_question",
                        help="per_question JSON 目录")
    parser.add_argument("--output-dir", default="codebook/merged",
                        help="合并后输出目录")
    parser.add_argument("--llm-model", default="deepseek-v4-flash",
                        help="LLM 模型名")
    parser.add_argument("--api-base",
                        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                        help="LLM API base url")
    parser.add_argument("--api-key",
                        default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
                        help="LLM API key")
    parser.add_argument("--skip-existing", default=True,
                        help="跳过已存在合并结果的子主题")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    args = parser.parse_args()

    logger = setup_logger(debug=args.debug)
    per_question_dir = Path(args.per_question_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有 per_question codebook
    logger.info(f"加载 per_question codebook 从: {per_question_dir}")
    all_entries = load_all_per_question_json(per_question_dir)
    logger.success(f"共加载 {len(all_entries)} 个问题的 codebook")

    # 初始化 LLM 客户端
    if not args.api_key:
        logger.warning("未设置 API Key，无法调用 LLM")
        return

    client = ensure_client(args.llm_model, args.api_base, args.api_key, logger)

    # 检查已有输出
    existing_outputs: set[str] = set()
    if args.skip_existing:
        for f in output_dir.glob("*.json"):
            existing_outputs.add(f.stem)

    total_tokens = 0
    processed = 0
    skipped = 0

    for theme_key, theme_config in THEMES.items():
        theme_title = theme_config["title"]
        for sub_key, sub_config in theme_config["sub_themes"].items():
            sub_label = sub_config["label"]
            filename_base = f"{theme_key}_{sub_key}"

            # 跳过已存在
            if args.skip_existing and filename_base in existing_outputs:
                logger.info(f"跳过已存在的合并结果: {filename_base}")
                skipped += 1
                continue

            # 筛选问题
            matched_entries = filter_questions_for_subtheme(
                theme_config, sub_key, all_entries)
            logger.info(f"\n{'='*60}")
            logger.info(f"主题: {theme_title} | 子主题: {sub_label}")
            logger.info(f"匹配到 {len(matched_entries)} 个问题: "
                        f"{[e['question_id'] for e in matched_entries]}")

            if not matched_entries:
                logger.warning("  没有匹配到任何问题，跳过")
                continue

            # 构建 LLM prompt
            messages = build_merge_prompt(theme_title, sub_label, matched_entries)

            # 调用 LLM
            try:
                parsed, llm_meta = call_llm_json(
                    client, args.llm_model, messages,
                    f"合并 codebook: {filename_base} ({len(matched_entries)}个问题)",
                    logger,
                )
            except Exception as exc:
                logger.error(f"  LLM调用失败: {exc}")
                continue

            total_tokens += llm_meta.get("total_tokens", 0)

            # 构建合并 entry
            merged = build_merged_entry(
                theme_title, sub_label, matched_entries, parsed, llm_meta)

            # 保存 JSON
            json_path = output_dir / f"{filename_base}.json"
            json_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.success(f"  -> JSON 已保存: {json_path.name}")

            # 保存 Markdown
            md_content = generate_markdown(merged, output_dir)
            md_path = output_dir / f"{filename_base}.md"
            md_path.write_text(md_content, encoding="utf-8")
            logger.success(f"  -> MD 已保存: {md_path.name}")

            processed += 1

    logger.success(f"\n处理完成！成功合并: {processed} 个, 跳过: {skipped} 个")
    logger.info(f"总 Token 用量: {total_tokens}")


if __name__ == "__main__":
    main()
