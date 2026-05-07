#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 res/ 目录下各访谈的 CSV 文件出发，为每个问题生成 codebook。

工作流程：
  1. 读取 res/ 下所有访谈 CSV 文件，按 question_id 聚合各访谈的回答
  2. 对每个问题，调用 LLM 分析所有回答，生成该问题的 code set 以及每个访谈的 code list
  3. 统计数据：每个 code 的出现次数和比例
  4. 输出 codebook（JSON + Markdown 格式）并存档 LLM 元数据

用法：
  python batch_generate_codebook.py
    [--csv-dir res]
    [--output-dir codebook]
    [--llm-model deepseek-v4-flash]
    [--api-base URL]
    [--api-key KEY]
    [--debug]

环境变量：
  DEEPSEEK_API_KEY  /  OPENAI_API_KEY
  DEEPSEEK_BASE_URL  (默认 https://api.deepseek.com)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid


def setup_logger(debug: bool = False):
    """简易日志"""
    class Logger:
        def __init__(self, debug: bool):
            self.debug_mode = debug

        def info(self, msg: str):
            print(f"[INFO] {msg}")

        def debug(self, msg: str):
            if self.debug_mode:
                print(f"[DEBUG] {msg}")

        def warning(self, msg: str):
            print(f"[WARNING] {msg}")

        def error(self, msg: str):
            print(f"[ERROR] {msg}")

        def success(self, msg: str):
            print(f"[SUCCESS] {msg}")

    return Logger(debug)


def ensure_client(llm_model: str, api_base: Optional[str], api_key: Optional[str], logger: Any):
    """初始化 OpenAI 客户端"""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("缺少 openai 依赖，请先执行: pip install openai")

    client = OpenAI(api_key=api_key, base_url=api_base) if api_key and api_base else OpenAI(api_key=api_key) if api_key else OpenAI(base_url=api_base) if api_base else OpenAI()
    return client


def call_llm_json(
    client: Any,
    llm_model: str,
    messages: list[dict[str, str]],
    description: str,
    logger: Any,
    max_retries: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用 LLM 并解析 JSON 响应"""
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

            llm_metadata = {
                "model": llm_model,
                "description": description,
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "output_tokens": getattr(resp.usage, "completion_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt": messages[0]["content"] if messages and messages[0].get("role") == "user" else "",
                "response": content,
            }

            logger.info(f"    tokens: input={llm_metadata['input_tokens']}, output={llm_metadata['output_tokens']}, total={llm_metadata['total_tokens']}")

            try:
                parsed = json.loads(content)
                llm_metadata["success"] = True
                return parsed, llm_metadata
            except json.JSONDecodeError:
                logger.warning(f"  JSON解析失败，尝试提取内容: {description}")
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(content[start: end + 1])
                        llm_metadata["success"] = True
                        llm_metadata["note"] = "通过字符串提取成功解析"
                        return parsed, llm_metadata
                    except json.JSONDecodeError:
                        pass
                if attempt < max_retries - 1:
                    wait = 5 ** attempt
                    logger.warning(f"  JSON解析失败，等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                else:
                    llm_metadata["success"] = False
                    llm_metadata["error"] = f"无法将LLM响应解析为JSON: {content}"
                    raise RuntimeError(llm_metadata["error"])

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


def build_codebook_prompt(
    question_id: str,
    question_text: str,
    interview_answers: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """构建用于 codebook 分析的 prompt

    要求 LLM:
    1. 分析所有回答中有几种针对性的回答，精炼为"code"
    2. 每个回答可能对应多个 code
    3. 返回每个回答对应哪些 code
    """
    lines = []
    for interview_name, answer_text in interview_answers:
        display_text = answer_text if answer_text else "（空）"
        lines.append(f"[{interview_name}] {display_text}")

    answers_block = "\n".join(lines)

    system_prompt = (
        "你是一个严谨的访谈回答分析助手。你的任务是对一个问题的所有访谈回答进行编码分析。\n\n"
        "请遵循以下步骤：\n"
        "1. 仔细阅读所有访谈对该问题的回答\n"
        f"2. 识别出所有不同的回答类型/模式，将其精炼为简洁的\"code\"\n"
        "3. 每个回答可能对应零个、一个或多个code（例如当回答中提及多种措施时，每种措施对应一个code）\n"
        "4. 如果某个回答为空，应包含code \"空\"\n"
        "5. 如果回答不确定或模棱两可，应包含code \"未明确回答\"\n\n"
        "请严格按以下JSON格式输出：\n"
        "{\n"
        '  "code_set": ["code1", "code2", ...],\n'
        '  "interview_codes": {\n'
        '    "访谈名称1": ["code1", "code2"],\n'
        '    "访谈名称2": ["code3"],\n'
        "    ...\n"
        "  }\n"
        "}\n\n"
        "要求：code_set 必须覆盖所有回答中出现的所有code。interview_codes 中每个访谈的code必须从code_set中选择。"
    )

    user_prompt = (
        f"问题ID：{question_id}\n"
        f"问题文本：{question_text}\n\n"
        "各访谈的回答如下（每个回答前标注了访谈名称）：\n"
        f"{answers_block}\n\n"
        "请分析以上回答，输出code_set和interview_codes。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def decode_interview_name(stem: str) -> str:
    """从 CSV 文件 stem 中提取干净的访谈名称。
    
    CSV 文件命名格式举例：
      访谈3-访谈3_访谈记录   ->  访谈3
      访谈1-一汽访谈-访谈1-一汽访谈_访谈记录  ->  访谈1
      访谈13__1_-访谈13__1__访谈记录  ->  访谈13__1_
    """
    name = stem
    # Remove trailing "_访谈记录" or "-访谈记录"
    for suffix in ("_访谈记录", "-访谈记录"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    # Try to extract a clean interview identifier:
    # If name starts with "访谈" followed by alphanumeric/id chars,
    # take the first segment before the first "-" if it looks like an ID
    # (this handles "访谈1-一汽访谈-访谈1-一汽访谈" -> "访谈1")
    if name.startswith("访谈"):
        parts = name.split("-", 1)
        if len(parts) > 1:
            # Check if the first part looks like a pure interview ID
            first = parts[0]
            # It should be "访谈" followed by numbers/underscores
            rest = first[2:]  # after "访谈"
            if rest and all(c.isdigit() or c == "_" for c in rest):
                return first
    return name


def load_all_csvs(csv_dir: Path, logger: Any) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, str]]]:
    """
    加载所有访谈 CSV 文件。

    Returns:
        answers_by_q: {question_id: {interview_name: answer_text}}
        question_texts: {question_id: question_text}
        citations_by_q: {question_id: {interview_name: citation_path}}
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info(f"找到 {len(csv_files)} 个访谈 CSV 文件")

    answers_by_q: dict[str, dict[str, str]] = {}
    citations_by_q: dict[str, dict[str, str]] = {}
    question_texts: dict[str, str] = {}
    interview_names: list[str] = []

    for csv_path in csv_files:
        interview_name = decode_interview_name(csv_path.stem)
        interview_names.append(interview_name)
        logger.info(f"  读取: {csv_path.name} -> {interview_name}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("question_id", "").strip()
                qtext = row.get("question_text", "").strip()
                answer = row.get("answer", "").strip()
                citation_path = row.get("citation_path", "").strip()

                if not qid:
                    continue

                if qid not in question_texts:
                    question_texts[qid] = qtext

                if qid not in answers_by_q:
                    answers_by_q[qid] = {}
                    citations_by_q[qid] = {}
                answers_by_q[qid][interview_name] = answer
                citations_by_q[qid][interview_name] = citation_path

    logger.success(f"共加载 {len(answers_by_q)} 个问题，来自 {len(interview_names)} 个访谈")
    return answers_by_q, question_texts, citations_by_q


def compute_codes(
    client: Any,
    llm_model: str,
    question_id: str,
    question_text: str,
    interview_answers: dict[str, str],
    logger: Any,
    llm_archive_dir: Path,
) -> tuple[list[str], dict[str, list[str]], Optional[dict[str, Any]]]:
    """对单个问题调用 LLM 生成 code set 和每个访谈的 code list"""
    answer_items = sorted(interview_answers.items(), key=lambda x: x[0])
    messages = build_codebook_prompt(question_id, question_text, answer_items)

    try:
        parsed, llm_meta = call_llm_json(
            client, llm_model, messages,
            f"Codebook分析 {question_id}",
            logger,
        )
    except Exception as exc:
        logger.error(f"  {question_id} LLM调用失败: {exc}")
        # Save failed metadata
        llm_meta = {
            "model": llm_model,
            "description": f"Codebook分析 {question_id}",
            "success": False,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prompt": messages[0]["content"] if messages else "",
            "response": "",
        }
        # Return emergency codes
        emergency_codes = ["有", "没有", "未明确回答", "空"]
        emergency_interview_codes = {}
        for interview_name in interview_answers:
            ans = interview_answers[interview_name]
            codes = []
            if not ans:
                codes.append("空")
            elif ans in ("无相关证据", "无相关证据。"):
                codes.append("没有")
            else:
                codes.append("有")
            emergency_interview_codes[interview_name] = codes
        return emergency_codes, emergency_interview_codes, llm_meta

    # Save LLM metadata to archive
    q_archive_dir = llm_archive_dir / safe_name(question_id)
    q_archive_dir.mkdir(parents=True, exist_ok=True)
    archive_file = q_archive_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    archive_file.write_text(
        json.dumps(llm_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    code_set = parsed.get("code_set", [])
    interview_codes = parsed.get("interview_codes", {})

    # Validate: make sure all codes in interview_codes are in code_set
    for inv_name, codes in interview_codes.items():
        for c in codes:
            if c not in code_set:
                logger.warning(f"  {inv_name} 的code '{c}' 不在code_set中，已自动添加")
                code_set.append(c)

    return code_set, interview_codes, llm_meta


def safe_name(value: str, max_len: int = 120) -> str:
    if not value:
        return "unnamed"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return safe[:max_len].rstrip("._-") or "unnamed"


def build_codebook_entry(
    question_id: str,
    question_text: str,
    answer_map: dict[str, str],
    citation_map: dict[str, str],
    code_set: list[str],
    interview_codes: dict[str, list[str]],
    llm_meta: dict[str, Any],
) -> dict[str, Any]:
    """构建单个问题的 codebook 条目，包含统计信息"""
    # Count occurrences for each code
    code_counts: dict[str, int] = {c: 0 for c in code_set}
    for inv_name, codes in interview_codes.items():
        for c in codes:
            if c in code_counts:
                code_counts[c] += 1
            else:
                # Fallback: code not in initial set
                code_counts[c] = code_counts.get(c, 0) + 1

    total_interviews = len(answer_map)
    code_percentages: dict[str, float] = {
        c: round(count / total_interviews * 100, 2) if total_interviews > 0 else 0.0
        for c, count in code_counts.items()
    }

    # For each code, list which interviews have it
    code_interviews: dict[str, list[dict[str, Any]]] = {}
    for c in code_set:
        code_interviews[c] = []
        for inv_name, codes in interview_codes.items():
            if c in codes:
                code_interviews[c].append({
                    "interview_name": inv_name,
                    "answer": answer_map.get(inv_name, ""),
                    "citation_path": citation_map.get(inv_name, ""),
                })

    return OrderedDict([
        ("question_id", question_id),
        ("question_text", question_text),
        ("total_interviews", total_interviews),
        ("code_set", sorted(code_set)),
        ("code_counts", code_counts),
        ("code_percentages", code_percentages),
        ("code_interviews", code_interviews),
        ("interview_codes", interview_codes),
        ("interview_answers", answer_map),
        ("interview_citations", citation_map),
        ("llm_metadata", {
            "input_tokens": llm_meta.get("input_tokens", 0),
            "output_tokens": llm_meta.get("output_tokens", 0),
            "total_tokens": llm_meta.get("total_tokens", 0),
            "success": llm_meta.get("success", False),
            "error": llm_meta.get("error", ""),
            "timestamp": llm_meta.get("timestamp", ""),
        }),
    ])


def _md_relative_to(output_dir: Path, citation_path: str) -> str:
    """将 citation_path（CSV 中的相对路径）转换为相对于 output_dir 的 Markdown 引用路径。"""
    if not citation_path:
        return ""
    # citation_path is relative to script CWD, e.g. "res/Q4.1.10/Q4.1.10.1/访谈3-访谈3_访谈记录.json"
    citation_abs = Path(citation_path).resolve()
    try:
        rel = os.path.relpath(citation_abs, output_dir.resolve())
        return rel
    except ValueError:
        return citation_path


def _format_answer_with_citation(answer: str, citation_path: str, md_dir: Path) -> str:
    """将 answer 和 citation_path 格式化为 Markdown 链接：[answer](citation_path)"""
    if not citation_path:
        return answer if answer else "（空）"
    rel_path = _md_relative_to(md_dir, citation_path)
    if not answer:
        return f"[（空）]({rel_path})"
    # Escape [] in answer text to avoid breaking markdown link syntax
    safe_answer = answer.replace("[", "\\[").replace("]", "\\]")
    return f"[{safe_answer}]({rel_path})"


def generate_markdown_codebook(entries: list[dict[str, Any]], output_path: Path) -> None:
    """生成可读的 Markdown 格式 codebook"""
    md_dir = output_path.parent  # directory containing the .md file
    lines = []
    lines.append("# Codebook - 访谈回答编码手册\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"共 {len(entries)} 个问题\n")
    lines.append("---\n")

    for entry in entries:
        qid = entry["question_id"]
        qtext = entry["question_text"]
        code_set = entry["code_set"]
        code_counts = entry["code_counts"]
        code_percentages = entry["code_percentages"]
        code_interviews = entry["code_interviews"]
        total = entry["total_interviews"]
        interview_codes = entry["interview_codes"]
        interview_answers = entry["interview_answers"]
        interview_citations = entry.get("interview_citations", {})

        lines.append(f"## {qid}\n")
        lines.append(f"**问题：** {qtext}\n")
        lines.append(f"**访谈总数：** {total}\n\n")

        # Summary table
        lines.append("### 编码统计\n")
        lines.append("| Code | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for c in sorted(code_set):
            count = code_counts.get(c, 0)
            pct = code_percentages.get(c, 0.0)
            lines.append(f"| {c} | {count} | {pct}% |")
        lines.append("")

        # Code details with interview names and answers (formatted as markdown links)
        lines.append("### 各Code详情\n")
        for c in sorted(code_set):
            interviews_with_code = code_interviews.get(c, [])
            lines.append(f"**{c}**（{code_counts.get(c, 0)}个访谈，{code_percentages.get(c, 0.0)}%）")
            if interviews_with_code:
                for item in interviews_with_code:
                    inv_name = item["interview_name"]
                    answer = item.get("answer", "")
                    citation = item.get("citation_path", "")
                    link = _format_answer_with_citation(answer, citation, md_dir)
                    lines.append(f"- **{inv_name}**：{link}")
            else:
                lines.append("- （无）")
            lines.append("")

        # Per-interview code assignment
        lines.append("### 各访谈编码\n")
        lines.append("| 访谈 | Code | 原始回答 |")
        lines.append("|------|------|----------|")
        for inv_name in sorted(interview_codes.keys()):
            codes = interview_codes.get(inv_name, [])
            answer = interview_answers.get(inv_name, "")
            citation = interview_citations.get(inv_name, "")
            link = _format_answer_with_citation(answer, citation, md_dir)
            codes_str = ", ".join(codes) if codes else "（无）"
            # For table, show link directly (tables allow inline markdown)
            lines.append(f"| {inv_name} | {codes_str} | {link} |")
        lines.append("")

        # LLM metadata
        llm_meta = entry.get("llm_metadata", {})
        lines.append("**LLM元数据：**")
        lines.append(f"- Token用量：输入 {llm_meta.get('input_tokens', 'N/A')} / 输出 {llm_meta.get('output_tokens', 'N/A')} / 总计 {llm_meta.get('total_tokens', 'N/A')}")
        lines.append(f"- 状态：{'成功' if llm_meta.get('success', False) else '失败'}")
        lines.append(f"- 时间：{llm_meta.get('timestamp', 'N/A')}")
        lines.append("")
        lines.append("---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_markdown_for_single_entry(entry: dict[str, Any], output_path: Path) -> None:
    """生成单个问题的 Markdown 格式 codebook，方便处理完一个问题就检查"""
    md_dir = output_path.parent  # per_question/
    qid = entry["question_id"]
    qtext = entry["question_text"]
    code_set = entry["code_set"]
    code_counts = entry["code_counts"]
    code_percentages = entry["code_percentages"]
    code_interviews = entry["code_interviews"]
    total = entry["total_interviews"]
    interview_codes = entry["interview_codes"]
    interview_answers = entry["interview_answers"]
    interview_citations = entry.get("interview_citations", {})

    lines = []
    lines.append(f"# Codebook - {qid}\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**问题：** {qtext}\n")
    lines.append(f"**访谈总数：** {total}\n")
    lines.append("---\n")

    # Summary table
    lines.append("## 编码统计\n")
    lines.append("| Code | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for c in sorted(code_set):
        count = code_counts.get(c, 0)
        pct = code_percentages.get(c, 0.0)
        lines.append(f"| {c} | {count} | {pct}% |")
    lines.append("")

    # Code details with interview names and answers (formatted as markdown links)
    lines.append("## 各Code详情\n")
    for c in sorted(code_set):
        interviews_with_code = code_interviews.get(c, [])
        lines.append(f"**{c}**（{code_counts.get(c, 0)}个访谈，{code_percentages.get(c, 0.0)}%）")
        if interviews_with_code:
            for item in interviews_with_code:
                inv_name = item["interview_name"]
                answer = item.get("answer", "")
                citation = item.get("citation_path", "")
                link = _format_answer_with_citation(answer, citation, md_dir)
                lines.append(f"- **{inv_name}**：{link}")
        else:
            lines.append("- （无）")
        lines.append("")

    # Per-interview code assignment
    lines.append("## 各访谈编码\n")
    lines.append("| 访谈 | Code | 原始回答 |")
    lines.append("|------|------|----------|")
    for inv_name in sorted(interview_codes.keys()):
        codes = interview_codes.get(inv_name, [])
        answer = interview_answers.get(inv_name, "")
        citation = interview_citations.get(inv_name, "")
        link = _format_answer_with_citation(answer, citation, md_dir)
        codes_str = ", ".join(codes) if codes else "（无）"
        lines.append(f"| {inv_name} | {codes_str} | {link} |")
    lines.append("")

    # LLM metadata
    llm_meta = entry.get("llm_metadata", {})
    lines.append("## LLM元数据\n")
    lines.append(f"- **Token用量：** 输入 {llm_meta.get('input_tokens', 'N/A')} / 输出 {llm_meta.get('output_tokens', 'N/A')} / 总计 {llm_meta.get('total_tokens', 'N/A')}")
    lines.append(f"- **状态：** {'成功' if llm_meta.get('success', False) else '失败'}")
    lines.append(f"- **时间：** {llm_meta.get('timestamp', 'N/A')}")
    if llm_meta.get("error"):
        lines.append(f"- **错误：** {llm_meta['error']}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="基于访谈 CSV 生成 Codebook")
    parser.add_argument("--csv-dir", default="res", help="访谈 CSV 目录")
    parser.add_argument("--output-dir", default="codebook", help="输出目录")
    parser.add_argument("--llm-model", default="deepseek-v4-flash", help="LLM 模型名")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="LLM API base url")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"), help="LLM API key")
    parser.add_argument("--max-questions", type=int, default=0, help="最多处理的问题数，0为全部")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在 codebook 条目的问题")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    args = parser.parse_args()

    logger = setup_logger(debug=args.debug)

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    llm_archive_dir = output_dir / "llm_archive"

    # Load all CSV files
    answers_by_q, question_texts, citations_by_q = load_all_csvs(csv_dir, logger)

    # Deduplicate: keep questions in order of first appearance
    ordered_qids = list(question_texts.keys())

    if args.max_questions > 0:
        ordered_qids = ordered_qids[:args.max_questions]

    # Check existing output
    json_output_path = output_dir / "codebook.json"
    existing_entries: dict[str, dict[str, Any]] = {}
    if args.skip_existing and json_output_path.exists():
        try:
            existing_data = json.loads(json_output_path.read_text(encoding="utf-8"))
            for entry in existing_data:
                existing_entries[entry["question_id"]] = entry
            logger.info(f"已加载 {len(existing_entries)} 个现有 codebook 条目")
        except Exception:
            pass

    # Initialize LLM client
    if not args.api_key:
        logger.warning("未设置 API Key，将使用应急模式（不调用LLM，仅做基础统计）")
        emergency_mode = True
        client = None
    else:
        emergency_mode = False
        client = ensure_client(args.llm_model, args.api_base, args.api_key, logger)

    # Ensure output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_archive_dir.mkdir(parents=True, exist_ok=True)
    per_question_dir = output_dir / "per_question"
    per_question_dir.mkdir(parents=True, exist_ok=True)

    entries = list(existing_entries.values())
    total_tokens = 0
    processed_count = 0
    skipped_count = 0

    logger.info(f"开始处理 {len(ordered_qids)} 个问题...")

    for qid in ordered_qids:
        if qid in existing_entries:
            logger.info(f"  跳过已存在的: {qid}")
            skipped_count += 1
            continue

        qtext = question_texts.get(qid, "")
        answer_map = answers_by_q.get(qid, {})
        citation_map = citations_by_q.get(qid, {})

        logger.info(f"[{processed_count + 1}/{len(ordered_qids)}] 处理问题: {qid} ({len(answer_map)} 个回答)")

        if emergency_mode:
            # Emergency mode: do basic categorization
            code_set = ["有", "没有", "未明确回答", "空"]
            interview_codes = {}
            for inv_name, ans in answer_map.items():
                codes = []
                if not ans:
                    codes.append("空")
                elif ans.strip() in ("无相关证据", "无相关证据。", "无相关证据，"):
                    codes.append("没有")
                elif "是" in ans or "有" in ans or "已" in ans:
                    codes.append("有")
                else:
                    codes.append("未明确回答")
                interview_codes[inv_name] = codes
            llm_meta = {"success": True, "note": "应急模式（未使用LLM）", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "timestamp": datetime.now().isoformat(timespec="seconds")}
        else:
            code_set, interview_codes, llm_meta = compute_codes(
                client, args.llm_model, qid, qtext, answer_map, logger, llm_archive_dir,
            )
            total_tokens += llm_meta.get("total_tokens", 0)

        entry = build_codebook_entry(qid, qtext, answer_map, citation_map, code_set, interview_codes, llm_meta)
        entries.append(entry)
        processed_count += 1

        # Save individual per-question codebook (JSON + Markdown)
        safe_qid = safe_name(qid)
        per_q_json = per_question_dir / f"{safe_qid}.json"
        per_q_md = per_question_dir / f"{safe_qid}.md"
        per_q_json.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        generate_markdown_for_single_entry(entry, per_q_md)
        logger.info(f"  已保存单题codebook: {safe_qid}.json/.md")

        # Save intermediate results periodically
        if processed_count % 10 == 0:
            logger.info(f"  已处理 {processed_count} 个问题，保存中间结果...")
            json_output_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            md_output_path = output_dir / "codebook.md"
            generate_markdown_codebook(entries, md_output_path)
            logger.info(f"  中间结果已保存")

    # Final save
    logger.success(f"\n处理完成！总处理: {processed_count} 个问题, 跳过: {skipped_count} 个")
    logger.info(f"总 Token 用量: {total_tokens}")

    json_output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success(f"JSON codebook 已保存到: {json_output_path}")

    md_output_path = output_dir / "codebook.md"
    generate_markdown_codebook(entries, md_output_path)
    logger.success(f"Markdown codebook 已保存到: {md_output_path}")

    # Save summary
    summary = {
        "total_questions": len(entries),
        "processed_questions": processed_count,
        "skipped_questions": skipped_count,
        "total_interviews": len(set(
            inv for q_entry in entries
            for inv in q_entry.get("interview_answers", {})
        )),
        "total_tokens": total_tokens,
        "llm_model": args.llm_model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"摘要已保存到: {summary_path}")


if __name__ == "__main__":
    main()
