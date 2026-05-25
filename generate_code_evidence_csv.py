#!/usr/bin/env python3
"""
从codebook/per_question/Q{question}.md中提取"各访谈编码"表格，
阅读每个访谈的evidence JSON文件，利用LLM（DeepSeek）将每个code与citations匹配，
生成CSV汇总表。

用法: python generate_code_evidence_csv.py --question Q6.2.5

CSV表头：访谈, code, evidence_file, 引用号, source_file, quoted_text, video_offset, source_type, reason
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# 访谈名映射字典（原始名称 -> 标准化名称）
# 此字典优先于两位数转换规则
NAME_MAPPING = {
    "访谈13 (1)": "访谈13 (2)",
    "访谈16-15": "访谈15（2）",
    "访谈17-18": "访谈18（1）",
    "访谈18": "访谈18（2）",
}


def normalize_interview_name(raw_name: str) -> str:
    """
    将原始访谈名标准化为"访谈名"。
    1. 先检查是否在特殊映射字典中
    2. 否则将访谈1-9 -> 访谈01-09（使用负向前瞻避免误改访谈10/11...）
    3. 其余保持原样
    """
    # 1. 特殊映射
    if raw_name in NAME_MAPPING:
        return NAME_MAPPING[raw_name]

    # 2. 两位数转换：将"访谈N"（N为1-9，后不跟数字）改为"访谈0N"
    result = re.sub(r'(访谈)([1-9])(?!\d)', lambda m: f'{m.group(1)}0{m.group(2)}', raw_name)

    return result


# LLM相关
try:
    from openai import OpenAI
except ImportError:
    print("请安装openai: pip install openai")
    sys.exit(1)

# 全局配置
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请设置环境变量 DEEPSEEK_API_KEY")
    sys.exit(1)

API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = "deepseek-chat"
INTERVIEW_DIR = Path("/mnt/zhitainew/ttt/interview")

client = OpenAI(api_key=API_KEY, base_url=API_BASE)


def call_llm(prompt: str, system_prompt: str = "你是一个严谨的数据分析助手。") -> str:
    """调用DeepSeek LLM并返回响应文本"""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM调用失败 (尝试 {attempt+1}/5): {e}")
            if attempt < 4:
                wait = 5 ** attempt
                print(f"  等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                raise


def parse_codebook_table(md_path: str, res_dir: Path) -> list[dict]:
    """
    解析 "## 各访谈编码" 下面的表格
    返回: [{interview, codes_list, evidence_path}, ...]
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 定位 "## 各访谈编码" 部分
    section_match = re.search(r"## 各访谈编码\n+", content)
    if not section_match:
        print("错误: 找不到 '## 各访谈编码' 部分")
        sys.exit(1)

    table_start = section_match.end()
    table_text = content[table_start:]

    # 找到表格行: | 访谈名 | code1, code2... | [描述](filepath) |
    rows = []
    pattern = r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*\[.+?\]\((.+?)\)\s*\|"

    for line in table_text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        # 跳过表头行和分隔行
        if "---" in line or "访谈" in line and "Code" in line and "原始回答" in line:
            continue

        m = re.match(pattern, line)
        if m:
            interview_name = m.group(1).strip()
            codes_str = m.group(2).strip()
            evidence_relpath = m.group(3).strip()

            # 解析codes
            codes = [c.strip() for c in codes_str.split(",") if c.strip()]

            # 构建相对路径：evidence_relpath 格式如 ../../res/Q6.2/Q6.2.5/xxx.json
            # 取最后一部分作为文件名
            evidence_path = res_dir / evidence_relpath.replace("../../res/Q6.2/", "")
            # Fallback: 直接用文件名
            if not evidence_path.exists():
                fname = evidence_relpath.split("/")[-1]
                evidence_path = res_dir / fname

            rows.append({
                "interview": interview_name,
                "codes": codes,
                "evidence_path": evidence_path,
            })

    return rows


def read_evidence_json(evidence_path: Path) -> Optional[dict]:
    """读取evidence JSON文件"""
    if not evidence_path.exists():
        print(f"  警告: 文件不存在 {evidence_path}")
        return None

    with open(evidence_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_codes_to_citations_llm(
    interview_name: str,
    codes: list[str],
    evidence_items: list[dict],
    citations: list[dict],
) -> list[dict]:
    """
    使用LLM将每个code匹配到对应的citations
    返回: [{code, citation_id, reason}, ...]
    """
    if not citations:
        return []

    # 准备LLM输入
    codes_str = "\n".join([f"- {c}" for c in codes])

    evidence_str_lines = []
    for e in evidence_items:
        seg_id = e.get("segment_id", "")
        reason = e.get("reason", "")
        cit = next((c for c in citations if c.get("segment_id") == seg_id), None)
        if cit:
            quoted = cit.get("quoted_text", "")[:150]
            evidence_str_lines.append(
                f"evidence[{cit.get('citation_id', '?')}] reason={reason}\n"
                f"  quoted_text: {quoted}\n"
            )

    evidence_str = "\n".join(evidence_str_lines)

    prompt = f"""请为访谈「{interview_name}」的以下每个code，找到其支持证据所对应的citation编号。

访谈的codes列表：
{codes_str}

证据列表（每个证据包含reason和引用的quoted_text）：
{evidence_str}

任务：对于每个code，判断哪些citation(s)提供了该code的支持依据。
- citation编号格式为 #1, #2, #3 等
- 一个code可能对应多个citation
- 基于reason和quoted_text判断

请严格按照以下JSON格式输出，不要包含其他文本：
{{
  "mappings": [
    {{"code": "组件安全测试", "citation_ids": ["#1"], "reasons": ["提及组件安全和集成测试。"]}},
    {{"code": "集成测试", "citation_ids": ["#1"], "reasons": ["提及组件安全和集成测试。"]}}
  ]
}}
"""

    system_prompt = "你是一个严谨的数据分析助手。请基于给定的证据reason和引文，将每个code匹配到对应的citation。仅输出JSON。"

    try:
        response = call_llm(prompt, system_prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            print(f"  LLM返回无法解析: {response[:200]}")
            return []

        mappings = result.get("mappings", [])

        rows = []
        for mapping in mappings:
            code = mapping.get("code", "")
            citation_ids = mapping.get("citation_ids", [])
            reasons = mapping.get("reasons", [])

            if not code or not citation_ids:
                continue

            for i, cid in enumerate(citation_ids):
                cit = next((c for c in citations if c.get("citation_id") == cid), None)
                if cit:
                    reason = reasons[i] if i < len(reasons) else ""
                    rows.append({
                        "code": code,
                        "citation_id": cid,
                        "source_file": cit.get("source_file", ""),
                        "quoted_text": cit.get("quoted_text", ""),
                        "video_offset": cit.get("video_offset", ""),
                        "source_type": cit.get("source_type", ""),
                        "reason": reason,
                    })

        return rows

    except Exception as e:
        print(f"  LLM匹配失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="生成代码-证据映射CSV")
    parser.add_argument("--question", "-q", default="Q6.2.1",
                        help="问题编号，如 Q6.2.1, Q6.2.5")
    args = parser.parse_args()
    question = args.question

    # 根据问题编号设置路径
    # 例如 Q6.2.5 -> res/Q6.2/Q6.2.5/
    codebook_md = f"codebook/per_question/{question}.md"
    parts = question.rsplit(".", 1)
    res_dir = Path("res") / parts[0] / question
    output_csv = f"{question}_code_evidence_mapping.csv"


    print("=" * 60)
    print(f"开始生成{question}代码-证据映射CSV")
    print("=" * 60)

    # 步骤1: 解析codebook表格
    print(f"\n[1/4] 解析codebook: {codebook_md}")
    rows = parse_codebook_table(codebook_md, res_dir)
    print(f"  解析到 {len(rows)} 个访谈条目")

    # 步骤2: 处理每个访谈
    print(f"\n[2/4] 处理每个访谈的evidence文件")
    all_csv_rows = []

    for i, entry in enumerate(rows):
        interview_name = entry["interview"]
        codes = entry["codes"]
        evidence_path = entry["evidence_path"]

        print(f"\n  [{i+1}/{len(rows)}] {interview_name}")
        print(f"    codes: {codes}")
        print(f"    evidence: {evidence_path}")

        # 读取JSON
        data = read_evidence_json(evidence_path)
        if not data:
            print(f"    跳过: 无法读取evidence文件")
            continue

        evidence_items = data.get("evidence", [])
        citations = data.get("citations", [])
        print(f"    evidence items: {len(evidence_items)}, citations: {len(citations)}")

        if not citations:
            print(f"    无citations，跳过")
            continue

        # 使用LLM匹配codes到citations
        matched_rows = match_codes_to_citations_llm(
            interview_name, codes, evidence_items, citations
        )

        interview_normalized = normalize_interview_name(interview_name)
        for mr in matched_rows:
            all_csv_rows.append({
                "访谈": interview_name,
                "访谈名": interview_normalized,
                "code": mr["code"],
                "evidence_file": str(evidence_path),
                "引用号": mr["citation_id"],
                "source_file": mr["source_file"],
                "quoted_text": mr["quoted_text"],
                "video_offset": mr["video_offset"],
                "source_type": mr["source_type"],
                "reason": mr["reason"],
            })


        print(f"    匹配到 {len(matched_rows)} 条记录")

    # 步骤3: 写入CSV
    print(f"\n[3/4] 写入CSV: {output_csv}")
    fieldnames = ["访谈", "访谈名", "code", "evidence_file", "引用号", "source_file",
                  "quoted_text", "video_offset", "source_type", "reason"]


    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"  共写入 {len(all_csv_rows)} 行")

    # 步骤4: 打印统计
    print(f"\n[4/4] 统计信息")
    code_counts = {}
    for row in all_csv_rows:
        code_counts[row["code"]] = code_counts.get(row["code"], 0) + 1

    print(f"  总记录数: {len(all_csv_rows)}")
    print(f"  涉及code数: {len(code_counts)}")
    print(f"\n  Code分布:")
    for code, count in sorted(code_counts.items(), key=lambda x: -x[1]):
        print(f"    {code}: {count}")

    print(f"\n✅ 完成! 结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()
