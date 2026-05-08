#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 res/ 目录下各访谈的 CSV 文件出发，为每个问题生成 codebook。

工作流程：
  1. 读取 res/ 下所有访谈 CSV 文件，按 question_id 聚合各访谈的回答
  2. 按 source（问题组）将问题分组，每组调用一次 LLM 分析所有回答，生成该组内每个问题的
     code set 以及每个访谈的 code list（这样可以降低 LLM 调用次数）
  3. 统计数据：每个 code 的出现次数和比例
  4. 输出 codebook（JSON + Markdown 格式）并存档 LLM 元数据

用法：
  python batch_group_generate_codebook.py
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


def setup_logger(debug: bool = False, log_file: str | None = None):
    """设置日志系统，使用loguru或回退到简单日志"""
    level = "DEBUG" if debug else "INFO"
    try:
        from loguru import logger
        # 移除默认的处理器，添加我们自己的格式
        logger.remove()
        # 添加控制台输出
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level=level,
        )
        # 添加文件输出（如果指定了日志文件）
        if log_file:
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level=level,
                rotation="10 MB",
            )
        return logger
    except ImportError:
        # 简单的日志类（回退方案）
        class SimpleLogger:
            def __init__(self):
                self.level_colors = {
                    "INFO": "\033[94m",      # 蓝色
                    "SUCCESS": "\033[92m",   # 绿色
                    "WARNING": "\033[93m",   # 黄色
                    "ERROR": "\033[91m",     # 红色
                    "RESET": "\033[0m",       # 重置
                }

            def _log(self, message, level="INFO"):
                color = self.level_colors.get(level, self.level_colors["RESET"])
                reset = self.level_colors["RESET"]
                print(f"{color}[{level}] {message}{reset}")

            def info(self, message):
                self._log(message, "INFO")

            def success(self, message):
                self._log(message, "SUCCESS")

            def warning(self, message):
                self._log(message, "WARNING")

            def error(self, message):
                self._log(message, "ERROR")

            def debug(self, message):
                self._log(message, "INFO")

            def critical(self, message):
                self._log(message, "ERROR")

        return SimpleLogger()


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


def build_group_codebook_prompt(
    source: str,
    group_questions: list[tuple[str, str, list[tuple[str, str]]]],
) -> list[dict[str, str]]:
    """构建用于按组分析多个问题的 codebook prompt

    每组调用一次 LLM，同时分析多个问题的所有访谈回答。

    Args:
        source: 问题来源组名（如 Q4.1.10）
        group_questions: [(question_id, question_text, [(interview_name, answer_text), ...]), ...]

    Returns:
        messages 列表，用于 LLM 调用
    """
    lines = []
    for qid, qtext, answer_items in group_questions:
        lines.append(f"## 问题 {qid}：{qtext}")
        for inv_name, answer_text in answer_items:
            display_text = answer_text if answer_text else "（空）"
            lines.append(f"[{inv_name}] {display_text}")
        lines.append("")  # blank line between questions

    answers_block = "\n".join(lines)

    system_prompt = (
        "你是一个严谨的访谈回答分析助手。你的任务是对一组问题的所有访谈回答进行编码分析。\n\n"
        "请遵循以下步骤：\n"
        "1. 仔细阅读每个问题下所有访谈的回答\n"
        "2. 对每一个问题，识别出所有不同的回答类型/模式，将其精炼为简洁的\"code\"\n"
        "3. 每个回答可能对应零个、一个或多个code（例如当回答中提及多种措施时，每种措施对应一个code）\n"
        "4. 如果某个回答为空，应包含code \"空\"\n"
        "5. 如果回答不确定或模棱两可，应包含code \"未明确回答\"\n\n"
        "请严格按以下JSON格式输出（每个问题一个独立的条目）：\n"
        "{\n"
        '  "results": [\n'
        "    {\n"
        '      "question_id": "问题1的ID",\n'
        '      "code_set": ["code1", "code2", ...],\n'
        '      "interview_codes": {\n'
        '        "访谈名称1": ["code1", "code2"],\n'
        '        "访谈名称2": ["code3"],\n'
        "        ...\n"
        "      }\n"
        "    },\n"
        "    {\n"
        '      "question_id": "问题2的ID",\n'
        "      ...\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "要求：每个问题的code_set必须覆盖该问题所有回答中出现的所有code。"
        "每个问题的interview_codes中每个访谈的code必须从该问题的code_set中选择。"
        "必须为每个问题都返回一个结果条目，不能遗漏。"
    )

    user_prompt = (
        f"问题组（source）：{source}\n"
        "以下是该组内所有问题及其各访谈的回答：\n\n"
        f"{answers_block}\n\n"
        "请分析以上所有问题的回答，为每个问题分别输出code_set和interview_codes。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def compute_group_codes(
    client: Any,
    llm_model: str,
    source: str,
    group_questions: list[tuple[str, str, dict[str, str]]],
    logger: Any,
    llm_archive_dir: Path,
) -> dict[str, tuple[list[str], dict[str, list[str]], dict[str, Any]]]:
    """对一组问题调用一次 LLM，生成每个问题的 code set 和 interview codes。

    Args:
        group_questions: [(question_id, question_text, {interview_name: answer_text}), ...]

    Returns:
        {question_id: (code_set, interview_codes, llm_meta)}
    """
    # Build prompt with all questions
    question_items = []
    for qid, qtext, answer_map in group_questions:
        answer_items = sorted(answer_map.items(), key=lambda x: x[0])
        question_items.append((qid, qtext, answer_items))

    messages = build_group_codebook_prompt(source, question_items)

    try:
        parsed, llm_meta = call_llm_json(
            client, llm_model, messages,
            f"Codebook分析组 {source}（{len(group_questions)}个问题）",
            logger,
        )
    except Exception as exc:
        logger.error(f"  组 {source} LLM调用失败: {exc}")
        # Save failed metadata
        llm_meta = {
            "model": llm_model,
            "description": f"Codebook分析组 {source}",
            "success": False,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prompt": messages[0]["content"] if messages else "",
            "response": "",
        }
        # Return emergency codes for all questions in the group
        results = {}
        for qid, qtext, answer_map in group_questions:
            emergency_codes = ["有", "没有", "未明确回答", "空"]
            emergency_interview_codes = {}
            for inv_name, ans in answer_map.items():
                codes = []
                if not ans:
                    codes.append("空")
                elif ans in ("无相关证据", "无相关证据。"):
                    codes.append("没有")
                else:
                    codes.append("有")
                emergency_interview_codes[inv_name] = codes
            results[qid] = (emergency_codes, emergency_interview_codes, llm_meta)
        return results

    # Save LLM metadata to archive (group level)
    group_archive_dir = llm_archive_dir / safe_name(f"group_{source}")
    group_archive_dir.mkdir(parents=True, exist_ok=True)
    archive_file = group_archive_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    archive_file.write_text(
        json.dumps(llm_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results_list = parsed.get("results", [])
    # Build a map from question_id to its parsed result
    parsed_by_qid: dict[str, dict] = {}
    for r in results_list:
        qid = r.get("question_id", "")
        if qid:
            parsed_by_qid[qid] = r

    results = {}
    for qid, qtext, answer_map in group_questions:
        if qid in parsed_by_qid:
            r = parsed_by_qid[qid]
            code_set = r.get("code_set", [])
            interview_codes = r.get("interview_codes", {})
            # Validate: make sure all codes in interview_codes are in code_set
            for inv_name, codes in interview_codes.items():
                for c in codes:
                    if c not in code_set:
                        logger.warning(f"  {qid}/{inv_name} 的code '{c}' 不在code_set中，已自动添加")
                        code_set.append(c)
            results[qid] = (code_set, interview_codes, llm_meta)
        else:
            # Question not found in LLM output, use emergency fallback
            logger.warning(f"  {qid} 未在LLM输出中找到，使用应急回退")
            emergency_codes = ["有", "没有", "未明确回答", "空"]
            emergency_interview_codes = {}
            for inv_name, ans in answer_map.items():
                codes = []
                if not ans:
                    codes.append("空")
                elif ans in ("无相关证据", "无相关证据。"):
                    codes.append("没有")
                else:
                    codes.append("有")
                emergency_interview_codes[inv_name] = codes
            results[qid] = (emergency_codes, emergency_interview_codes, llm_meta)

    return results


def _build_stem_to_live_id(records_path: Path) -> dict[str, str]:
    """从 interview_records.json 构建 CSV stem -> live_id 的映射表。

    CSV 文件名是按 safe_name(f"{live_id}-{video_title}").csv 生成的，
    因此通过构造预期文件名与实际 CSV 文件名精确比对，可以获取正确的 live_id。
    """
    stem_to_live_id: dict[str, str] = {}
    try:
        records = json.loads(records_path.read_text("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return stem_to_live_id
    for lid, info in records.items():
        vt = info.get("video_title") or info.get("title") or ""
        if not vt:
            continue
        expected_stem = safe_name(f"{lid}-{vt}")
        stem_to_live_id[expected_stem] = lid
    return stem_to_live_id


def decode_interview_name(stem: str) -> str:
    """从 CSV 文件 stem 中提取干净的访谈名称（回退方案）。

    当 _build_stem_to_live_id 无法匹配时，使用此启发式方法。
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


def load_all_csvs(csv_dir: Path, logger: Any, records_path: str = "interview_records.json") -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, str]], dict[str, str]]:
    """
    加载所有访谈 CSV 文件。

    Returns:
        answers_by_q: {question_id: {interview_name: answer_text}}
        question_texts: {question_id: question_text}
        citations_by_q: {question_id: {interview_name: citation_path}}
        question_sources: {question_id: source_group_name}
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info(f"找到 {len(csv_files)} 个访谈 CSV 文件")

    # Build stem -> live_id mapping from interview_records.json for precise matching
    records_path_obj = csv_dir.parent / records_path if csv_dir.name != records_path and not Path(records_path).exists() else Path(records_path)
    stem_to_live_id = _build_stem_to_live_id(records_path_obj)

    answers_by_q: dict[str, dict[str, str]] = {}
    citations_by_q: dict[str, dict[str, str]] = {}
    question_texts: dict[str, str] = {}
    question_sources: dict[str, str] = {}
    interview_names: list[str] = []

    for csv_path in csv_files:
        # Prefer exact matching via interview_records.json, fall back to heuristic
        interview_name = stem_to_live_id.get(csv_path.stem, decode_interview_name(csv_path.stem))
        interview_names.append(interview_name)
        logger.info(f"  读取: {csv_path.name} -> {interview_name}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("question_id", "").strip()
                qtext = row.get("question_text", "").strip()
                answer = row.get("answer", "").strip()
                citation_path = row.get("citation_path", "").strip()
                source = row.get("source", "").strip()

                if not qid:
                    continue

                if qid not in question_texts:
                    question_texts[qid] = qtext
                    question_sources[qid] = source

                if qid not in answers_by_q:
                    answers_by_q[qid] = {}
                    citations_by_q[qid] = {}
                answers_by_q[qid][interview_name] = answer
                citations_by_q[qid][interview_name] = citation_path

    # Check for duplicate interview names (should not happen with correct mapping)
    unique_interviews = set(interview_names)
    if len(unique_interviews) != len(interview_names):
        from collections import Counter
        dupes = {name: count for name, count in Counter(interview_names).items() if count > 1}
        logger.warning(f"发现 {len(dupes)} 个重复的访谈名称: {dupes}")
        logger.warning("重复的访谈名称会导致数据被静默覆盖，请检查 interview_records.json 和 CSV 文件的一致性")

    logger.success(f"共加载 {len(answers_by_q)} 个问题，来自 {len(interview_names)} 个访谈（{len(unique_interviews)} 个唯一）")
    return answers_by_q, question_texts, citations_by_q, question_sources


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
    parser = argparse.ArgumentParser(description="基于访谈 CSV 生成 Codebook（按问题组分批调用LLM）")
    parser.add_argument("--csv-dir", default="res", help="访谈 CSV 目录")
    parser.add_argument("--output-dir", default="codebook", help="输出目录")
    parser.add_argument("--llm-model", default="deepseek-v4-flash", help="LLM 模型名")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="LLM API base url")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"), help="LLM API key")
    parser.add_argument("--max-questions", type=int, default=0, help="最多处理的问题数，0为全部")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在 codebook 条目的问题组（全量模式）")
    parser.add_argument("--incremental", action="store_true", help="增量模式：以组为单位检测是否有新访谈回答加入，有则重新分析整组（避免完全重跑）")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    args = parser.parse_args()

    logger = setup_logger(debug=args.debug)

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    llm_archive_dir = output_dir / "llm_archive"

    # Load all CSV files (now also returns question_sources)
    answers_by_q, question_texts, citations_by_q, question_sources = load_all_csvs(csv_dir, logger)

    # Build groups by source
    groups: dict[str, list[str]] = {}
    for qid in question_texts:
        src = question_sources.get(qid, "")
        if not src:
            src = "_default_"
        groups.setdefault(src, []).append(qid)

    if not groups:
        logger.warning("未找到任何问题分组，请检查 CSV 文件中是否包含 source 列")
        return

    logger.info(f"共加载 {len(question_texts)} 个问题，分为 {len(groups)} 个组")

    if args.max_questions > 0:
        # Truncate: limit total questions across groups
        remaining = args.max_questions
        truncated_groups: dict[str, list[str]] = {}
        for src in sorted(groups):
            take = min(remaining, len(groups[src]))
            truncated_groups[src] = groups[src][:take]
            remaining -= take
            if remaining <= 0:
                break
        groups = truncated_groups
        logger.info(f"限制问题数，仅处理前 {args.max_questions} 个问题")

    # Check existing output
    json_output_path = output_dir / "codebook.json"
    existing_entries: dict[str, dict[str, Any]] = {}
    if (args.skip_existing or args.incremental) and json_output_path.exists():
        try:
            existing_data = json.loads(json_output_path.read_text(encoding="utf-8"))
            for entry in existing_data:
                existing_entries[entry["question_id"]] = entry
            logger.info(f"已加载 {len(existing_entries)} 个现有 codebook 条目")
        except Exception as e:
            logger.warning(f"加载现有codebook失败，将重新处理全部: {e}")

    # Determine which groups need processing
    # For skip-existing: skip a group if ALL questions in the group already exist
    # For incremental: process a group if ANY question in the group has new answers
    # For full mode: process all groups
    groups_to_process: set[str] = set()
    if args.incremental:
        # Group-level incremental: check if any question in the group has new answers
        for src, qids in groups.items():
            needs_update = False
            for qid in qids:
                current_answer_count = len(answers_by_q.get(qid, {}))
                existing_entry = existing_entries.get(qid)
                if existing_entry is None:
                    needs_update = True
                    break
                existing_answer_count = len(existing_entry.get("interview_answers", {}))
                if current_answer_count > existing_answer_count:
                    needs_update = True
                    break
            if needs_update:
                groups_to_process.add(src)
        logger.info(f"增量模式：{len(groups_to_process)} 个组需要重新分析（共 {len(groups)} 个组）")
    elif args.skip_existing:
        # Skip a group only if ALL its questions exist in existing_entries
        for src, qids in groups.items():
            all_exist = all(qid in existing_entries for qid in qids)
            if not all_exist:
                groups_to_process.add(src)
        logger.info(f"跳过已有模式：处理 {len(groups_to_process)} 个组，跳过 {len(groups) - len(groups_to_process)} 个组")
    else:
        groups_to_process = set(groups.keys())
        logger.info(f"全量模式：处理全部 {len(groups_to_process)} 个组")

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

    # Start with existing entries (all of them)
    entries = list(existing_entries.values())
    total_tokens = 0
    processed_count = 0
    skipped_count = 0
    group_count = 0

    total_groups_to_process = len(groups_to_process)
    total_questions_in_groups = sum(len(qids) for src, qids in groups.items() if src in groups_to_process)
    logger.info(f"开始处理 {total_groups_to_process} 个组（共 {total_questions_in_groups} 个问题）...")

    for src in sorted(groups):
        if src not in groups_to_process:
            skipped_count += len(groups[src])
            continue

        qids_in_group = groups[src]
        group_count += 1

        logger.info(f"\n[{group_count}/{total_groups_to_process}] 处理问题组: {src}（{len(qids_in_group)} 个问题）")

        # Build group data for LLM call
        group_questions: list[tuple[str, str, dict[str, str]]] = []
        for qid in qids_in_group:
            qtext = question_texts.get(qid, "")
            answer_map = answers_by_q.get(qid, {})
            group_questions.append((qid, qtext, answer_map))

        if emergency_mode:
            # Emergency mode: basic categorization for each question
            for qid, qtext, answer_map in group_questions:
                citation_map = citations_by_q.get(qid, {})
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

                entry = build_codebook_entry(qid, qtext, answer_map, citation_map, code_set, interview_codes, llm_meta)
                entries.append(entry)
                processed_count += 1

                # Save individual per-question codebook
                safe_qid = safe_name(qid)
                per_q_json = per_question_dir / f"{safe_qid}.json"
                per_q_md = per_question_dir / f"{safe_qid}.md"
                per_q_json.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
                generate_markdown_for_single_entry(entry, per_q_md)
        else:
            # Call LLM once for the entire group
            group_results = compute_group_codes(
                client, args.llm_model, src, group_questions, logger, llm_archive_dir,
            )

            # Process each question's result
            for qid, qtext, answer_map in group_questions:
                citation_map = citations_by_q.get(qid, {})
                code_set, interview_codes, llm_meta = group_results[qid]
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
                logger.info(f"  ✓ {qid} 已保存单题codebook: {safe_qid}.json/.md")

        # Save intermediate results after each group
        logger.info(f"  组 {src} 处理完成，保存中间结果...")
        json_output_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_output_path = output_dir / "codebook.md"
        generate_markdown_codebook(entries, md_output_path)
        logger.info(f"  中间结果已保存（共 {processed_count} 个问题）")

    # Final save
    logger.success(f"\n处理完成！总处理: {processed_count} 个问题（{group_count} 个组）, 跳过: {skipped_count} 个问题")
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
        "total_groups": len(groups),
        "processed_groups": group_count,
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
