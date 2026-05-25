#!/usr/bin/env python3
"""
修复CSV中的"访谈"列，添加标准化的"访谈名"列。
处理规则：
1. 访谈1-9中的数字改为两位数（01-09）
2. 特殊重命名（使用映射字典）：
   - 访谈13 (1) -> 访谈13 (2)
   - 访谈16-15 -> 访谈15（2）
   - 访谈17-18 -> 访谈18（1）
   - 访谈18 -> 访谈18（2）

用法: python fix_interview_names.py <csv_file1> [csv_file2 ...]
"""
import csv
import re
import sys
from pathlib import Path


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


def fix_csv(filepath: str) -> None:
    """处理单个CSV文件"""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ 文件不存在: {filepath}")
        return

    # 读取原CSV
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_fieldnames = reader.fieldnames[:] if reader.fieldnames else []
        rows = list(reader)

    if not all_fieldnames or "访谈" not in all_fieldnames:
        print(f"❌ 无效的CSV格式（缺少'访谈'列）: {filepath}")
        return

    # 移除已有的"访谈名"列（如果有），避免重复
    fieldnames = [fn for fn in all_fieldnames if fn != "访谈名"]

    # 构建新表头：在"访谈"后插入"访谈名"
    new_fieldnames = []
    for fn in fieldnames:
        new_fieldnames.append(fn)
        if fn == "访谈":
            new_fieldnames.append("访谈名")

    new_rows = []
    for row in rows:
        new_row = {}
        for fn in fieldnames:  # 用清理后的fieldnames遍历
            new_row[fn] = row[fn]
            if fn == "访谈":
                new_row["访谈名"] = normalize_interview_name(row["访谈"])
        new_rows.append(new_row)

    # 写回原文件
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    # 显示转换示例
    examples = {}
    for row in new_rows:
        orig = row["访谈"]
        new = row["访谈名"]
        if orig != new and orig not in examples:
            examples[orig] = new

    if examples:
        print(f"  ✓ 转换示例 (已排序):")
        # 按原始名称排序展示
        for orig, new in sorted(examples.items()):
            print(f"    {orig:20s} -> {new}")
    print(f"  ✓ 共处理 {len(new_rows)} 行, 保存至 {path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python fix_interview_names.py <csv_file1> [csv_file2 ...]")
        print("示例: python fix_interview_names.py Q6.2.1_code_evidence_mapping.csv Q6.2.5_code_evidence_mapping.csv")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        print(f"\n📋 处理: {filepath}")
        fix_csv(filepath)

    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
