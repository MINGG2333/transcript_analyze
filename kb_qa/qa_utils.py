"""工具函数：LLM 调用、引用校验、重编号、归档等。"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid


def call_llm_json(
    client,
    llm_model: str,
    messages: list[dict[str, str]],
    description: str,
    max_retries: int = 5,
    logger=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用 LLM API 并解析 JSON 响应，包含重试机制（指数退避）。"""
    last_raw = None

    for attempt in range(max_retries):
        try:
            if logger:
                logger.info(f"调用LLM: {description} (尝试 {attempt+1}/{max_retries})")

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

            if logger:
                logger.info(
                    f"  tokens: input={llm_metadata['input_tokens']}, "
                    f"output={llm_metadata['output_tokens']}, total={llm_metadata['total_tokens']}"
                )

            try:
                parsed = json.loads(content)
                llm_metadata["success"] = True
                return parsed, llm_metadata
            except json.JSONDecodeError:
                if logger:
                    logger.warning(f"LLM返回JSON解析失败，尝试提取内容: {description}")
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
                    wait_time = 5 ** attempt
                    if logger:
                        logger.warning(f"JSON解析失败 (第 {attempt+1} 次)，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    llm_metadata["success"] = False
                    llm_metadata["error"] = f"无法将LLM响应解析为JSON: {content}"
                    raise RuntimeError(llm_metadata["error"])

        except Exception as e:
            if logger:
                logger.warning(f"LLM请求异常 (第 {attempt+1} 次): {e}")

            if attempt < max_retries - 1:
                wait_time = 5 ** attempt
                if logger:
                    logger.warning(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"LLM调用在 {max_retries} 次重试后仍失败"
                if logger:
                    logger.error(error_msg)
                raise RuntimeError(error_msg) from e


def validate_answer_citations(answer: str) -> list[str]:
    """校验答案中的引用标记格式是否符合严格格式。

    只允许格式：[#N]（如 [#1]、[#5]），一个中括号内只能有一个引用编号。
    不允许 [#N-#M]（区间写法）、[#N, #M]（逗号）等非标准格式。

    返回：不符合格式的引用字符串列表，为空表示全部合法。
    """
    if not answer:
        return []
    all_brackets = list(re.finditer(r"\[[^\]]*\]", answer))
    allowed = re.compile(r"^\[\#\d+\]$")
    invalid: list[str] = []
    for m in all_brackets:
        text = m.group()
        if "#" in text and not allowed.match(text):
            invalid.append(text)
    return invalid


def validate_citations_consistency(
    answer: str, citations: list[dict[str, Any]]
) -> list[str]:
    """双向校验 answer 和 citations 的引用编号是否完全一致。

    检查：
    1. answer 中引用的 [#N] 是否全部在 citations 中存在（无缺失引用）
    2. citations 中的编号是否全部在 answer 中被引用（无多余引用）

    返回：问题描述列表，为空表示完全一致。
    """
    if not answer and not citations:
        return []

    answer_refs: set[int] = set()
    for m in re.finditer(r"\[#(\d+)\]", answer):
        answer_refs.add(int(m.group(1)))

    citation_ids: set[int] = set()
    for c in citations:
        try:
            citation_ids.add(int(c.get("citation_id", "").lstrip("#")))
        except (ValueError, AttributeError):
            pass

    problems: list[str] = []

    # 方向1：answer 引用了 citations 中不存在的编号
    missing = sorted(answer_refs - citation_ids)
    if missing:
        problems.append(f"answer 引用了不存在的编号: {', '.join(f'#{n}' for n in missing)}")

    # 方向2：citations 中有编号未被 answer 引用
    extra = sorted(citation_ids - answer_refs)
    if extra:
        problems.append(f"citations 中存在未被 answer 引用的编号: {', '.join(f'#{n}' for n in extra)}")

    return problems


def filter_citations_by_answer(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """过滤 citations 列表，只保留答案中通过 [#N] 实际引用的条目。"""
    if not answer or not citations:
        return [] if answer else citations

    referenced: set[int] = set()
    for m in re.finditer(r"\[#(\d+)(?:-#(\d+))?\]", answer):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        referenced.update(range(start, end + 1))

    if not referenced:
        return citations  # 未检测到引用标记，全部保留（兜底）

    def _parse_citation_id(cid: str) -> int:
        try:
            return int(cid.lstrip("#"))
        except (ValueError, AttributeError):
            return -1

    return [c for c in citations if _parse_citation_id(c.get("citation_id", "")) in referenced]


def renumber_citations(
    answer: str, citations: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """过滤后重新从 #1 编号 citations，并更新 answer 中的引用标记。"""
    if not citations:
        return answer, citations

    # 构建旧编号 → 新编号的映射
    old_ids = []
    for c in citations:
        try:
            old_ids.append(int(c.get("citation_id", "").lstrip("#")))
        except (ValueError, AttributeError):
            old_ids.append(-1)

    sorted_old_ids = sorted(old_ids)
    id_mapping: dict[int, int] = {}
    for new_idx, old_id in enumerate(sorted_old_ids, start=1):
        id_mapping[old_id] = new_idx

    # 更新 answer 中的引用标记（从大到小替换以避免冲突）
    def replace_ref(m: re.Match) -> str:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        new_nums = sorted(id_mapping.get(n, n) for n in range(start, end + 1))
        # 压缩连续区间
        ranges = []
        cur_start = cur_end = None
        for n in new_nums:
            if cur_start is None:
                cur_start = cur_end = n
            elif n == cur_end + 1:
                cur_end = n
            else:
                ranges.append((cur_start, cur_end))
                cur_start = cur_end = n
        if cur_start is not None:
            ranges.append((cur_start, cur_end))
        parts = []
        for rs, re_ in ranges:
            if rs == re_:
                parts.append(f"[#{rs}]")
            else:
                parts.append(f"[#{rs}-#{re_}]")
        return "".join(parts)

    answer = re.sub(r"\[#(\d+)(?:-#(\d+))?\]", replace_ref, answer)

    # 更新每个 citation 的 citation_id
    for c in citations:
        try:
            old_id = int(c.get("citation_id", "").lstrip("#"))
            c["citation_id"] = f"#{id_mapping.get(old_id, old_id)}"
        except (ValueError, AttributeError):
            pass

    # 按新编号排序
    def sort_key(c):
        try:
            return int(c.get("citation_id", "").lstrip("#"), 10)
        except (ValueError, AttributeError):
            return 0
    citations.sort(key=sort_key)

    return answer, citations


def archive_result(result: dict[str, Any], kb_dir: Path) -> Path:
    """将问答结果归档到 qa_archive 目录。"""
    archive_dir = kb_dir / "qa_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    archive_path = archive_dir / name
    archive_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return archive_path


def ensure_client(client, api_key: Optional[str], api_base: Optional[str], logger=None):
    """确保 OpenAI 客户端已初始化。"""
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "缺少 openai 依赖，请先执行: python -m pip install -r requirements_kb_qa.txt"
        ) from exc

    if api_key:
        return OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)
    else:
        return OpenAI(base_url=api_base) if api_base else OpenAI()
