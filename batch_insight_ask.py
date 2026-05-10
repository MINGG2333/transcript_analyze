#!/usr/bin/env python3
"""
batch_insight_ask.py - 跨访谈Insight分析脚本（基于CSV直读模式）

将25个洞察问题按主题分组，直接从 res/访谈*.csv 中读取各访谈对应协议问题的回答，
由LLM进行跨访谈聚合分析，输出每个主题组的综合洞见报告（含支持依据引用）。

相较基于知识库（kb_qa）的模式，本脚本：
  - 直接读取已生成的CSV回答，无需向量检索
  - 为每个洞察问题（Q1-Q25）明确标注对应的协议 question_id，
    避免无关数据引入LLM上下文，大幅提升效率和准确性
  - 支持分批次处理大量数据

用法：
  python batch_insight_ask.py [options]

示例：
  python batch_insight_ask.py --output-dir insight_res
  python batch_insight_ask.py --output-dir insight_res --limit-groups 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from kb_qa.cli import setup_logger


# ============================================================
# 协议问题映射：洞察问题(Q1-Q25) -> 协议 question_id 前缀/列表
# ============================================================

# 每个洞察问题对应哪些协议问题的question_id前缀
# 使用前缀匹配，例如 "Q4.1" 匹配 Q4.1.1, Q4.1.2, ... Q4.1.20
# 也可以使用完整question_id精确匹配

QUESTION_PROTOCOL_MAP: dict[str, dict[str, Any]] = {
    # === G01: 威胁应对整体情况 ===
    "Q1": {
        "description": "关键威胁应对覆盖率",
        "protocol_include": [
            # 所有威胁类型的 .1（有没有采取措施）和 .2（具体措施）
            # 涵盖通信类(Q4.1.*)、更新类(Q4.2.*)、用户诱导类(Q4.3.*)、
            # 外部连接类(Q4.4.*)、数据/代码操纵类(Q4.5.*)、密码学类(Q4.6.*)、
            # 用户变更类(Q4.7.*)、硬件操纵类(Q4.8.*)、服务器/云类(Q4.9.*)、
            # 用户/流程类(Q4.10.*)、第三方/物理类(Q4.11.*)
            # 每个威胁子类型都有 .1（有无措施）和 .2（具体措施）
            # 同时也包括Q4.12（其他措施）
            "Q4.12",
        ],
        "protocol_include_prefix": [
            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
            "Q4.6", "Q4.7", "Q4.8", "Q4.9",
            "Q4.10", "Q4.11",
        ],
        "suffix_filter": [".1", ".2"],  # 只取.1(有无措施)和.2(具体措施)
    },
    "Q16": {
        "description": "案例威胁类型分布",
        "protocol_include": ["Q6.1.1"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    # === G02: 技术防护措施与挑战 ===
    "Q2": {
        "description": "技术上的防护措施",
        "protocol_include": ["Q4.12"],
        "protocol_include_prefix": [
            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
            "Q4.6", "Q4.7", "Q4.8", "Q4.9",
            "Q4.10", "Q4.11",
        ],
        "suffix_filter": [".2"],  # 具体措施
    },
    "Q3": {
        "description": "技术上的挑战",
        "protocol_include": [],
        "protocol_include_prefix": [
            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
            "Q4.6", "Q4.7", "Q4.8", "Q4.9",
            "Q4.10", "Q4.11",
        ],
        "suffix_filter": [".3"],  # 技术上遇到的障碍
    },
    # === G03: 组织防护措施与挑战 ===
    "Q4": {
        "description": "组织上的防护措施",
        "protocol_include": [
            "Q4.13.1.1", "Q4.13.1.2", "Q4.13.2.1", "Q4.13.2.2",
            "Q4.13.2.4", "Q4.13.2.6", "Q4.13.2.8", "Q4.13.2.10",
            "Q4.13.2.12", "Q4.13.3.1", "Q4.13.3.2", "Q4.13.3.4",
            "Q4.13.3.6", "Q4.13.3.8", "Q4.13.3.10", "Q4.13.3.12",
            "Q4.13.4.1", "Q4.13.4.2", "Q4.13.4.4", "Q4.13.4.6",
            "Q4.14.1.1", "Q4.14.1.2", "Q4.14.1.3", "Q4.14.1.5",
            "Q4.14.1.7",
            "Q4.14.2.1", "Q4.14.2.2",
            "Q4.14.3.1", "Q4.14.3.2",
            "Q4.14.4.1", "Q4.14.4.2",
            "Q4.15", "Q4.16.1", "Q4.16.2", "Q4.16.3", "Q4.16.4",
            "Q4.17.1", "Q4.17.2", "Q4.17.3", "Q4.17.4",
            "Q4.17.5", "Q4.17.6", "Q4.17.7",
            "Q4.18.1", "Q4.18.2", "Q4.18.3",
        ],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q5": {
        "description": "组织上的挑战",
        "protocol_include": [
            "Q4.13.1.3",
            "Q4.13.2.3", "Q4.13.2.5", "Q4.13.2.7", "Q4.13.2.9",
            "Q4.13.2.11", "Q4.13.2.13",
            "Q4.13.3.3", "Q4.13.3.5", "Q4.13.3.7", "Q4.13.3.9",
            "Q4.13.3.11", "Q4.13.3.13",
            "Q4.13.4.3", "Q4.13.4.5", "Q4.13.4.7",
            "Q4.14.1.4", "Q4.14.1.6",
            "Q4.14.2.3",
            "Q4.14.3.3",
            "Q4.14.4.3",
            "Q4.16.1.3", "Q4.16.1.5",
            "Q4.16.2.3",
            "Q4.16.3.3",
            "Q4.17.1.3", "Q4.17.1.5",
            "Q4.17.2.3",
            "Q4.17.3.3",
            "Q4.17.4.3",
            "Q4.17.5.3",
            "Q4.17.6.3",
            "Q4.18.1.3", "Q4.18.1.5", "Q4.18.1.7",
            "Q4.18.2.3", "Q4.18.2.5",
            "Q4.18.3.3",
        ],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    # === G04: 工具使用与工具链 ===
    "Q6": {
        "description": "工具使用普及率",
        "protocol_include": ["Q5.1.1", "Q5.1.2", "Q5.1.3", "Q5.1.4"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q7": {
        "description": "工具链覆盖完整度",
        "protocol_include": ["Q5.1.7", "Q5.1.8", "Q5.1.9", "Q5.1.10", "Q5.1.11"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q8": {
        "description": "缺失工具频次",
        "protocol_include": ["Q5.1.6", "Q5.1.9", "Q5.1.10"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q9": {
        "description": "工具的局限性或不足",
        "protocol_include": ["Q5.1.5", "Q5.1.11"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    # === G05: 知识与技能 ===
    "Q10": {
        "description": "网络安全知识重要度排序",
        "protocol_include": ["Q5.2.1"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q11": {
        "description": "知识获取渠道频次",
        "protocol_include": ["Q5.2.2"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q12": {
        "description": "知识获取挑战",
        "protocol_include": ["Q5.2.3"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q13": {
        "description": "希望市场提供的培训",
        "protocol_include": [],
        "protocol_include_prefix": [],
        "suffix_filter": [],
        "fallback_note": "此问题无直接映射的协议问题，基于Q5.2整体上下文推断",
    },
    # === G06: 基础设施 ===
    "Q14": {
        "description": "基础设施需求频次",
        "protocol_include": ["Q5.3.1", "Q5.3.2"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q15": {
        "description": "基础设施的限制",
        "protocol_include": ["Q5.3.3"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    # === G07: 案例安全分析与处置 ===
    "Q17": {
        "description": "案例威胁触发原因",
        "protocol_include": ["Q6.1.1"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q18": {
        "description": "案例威胁处置手段",
        "protocol_include": ["Q6.1.1"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    # === G08: 测试经验与方法 ===
    "Q19": {
        "description": "测试方法频次",
        "protocol_include": ["Q6.2.1", "Q6.2.2"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q20": {
        "description": "测试方法局限性",
        "protocol_include": ["Q6.2.3"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q21": {
        "description": "测试挑战",
        "protocol_include": ["Q6.2.5"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q22": {
        "description": "期望的测试方法",
        "protocol_include": ["Q6.2.4"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    # === G09: 合规实践 ===
    "Q23": {
        "description": "标准提及频次",
        "protocol_include": ["Q6.3.1", "Q6.3.2"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q24": {
        "description": "标准平均认知广度",
        "protocol_include": ["Q6.3.2"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
    "Q25": {
        "description": "合规挑战",
        "protocol_include": ["Q6.3.3", "Q6.3.4", "Q6.3.5"],
        "protocol_include_prefix": [],
        "suffix_filter": [],
    },
}


# ============================================================
# 问题组定义（含 protocol question_id 映射）
# ============================================================

QUESTION_GROUPS: list[dict[str, Any]] = [
    {
        "group_id": "G01",
        "group_name": "威胁应对整体情况",
        "description": "分析关键威胁应对覆盖率和案例威胁类型分布",
        "insight_questions": ["Q1", "Q16"],
        "combined_question": (
            "请基于所有受访者的访谈内容，综合分析网络安全威胁应对的整体情况：\n\n"
            "1) 关键威胁应对覆盖率：受访方实际部署了应对措施的威胁类型占所有被提及威胁的大致比例如何？"
            "哪些威胁类型（如通信冒充/注入、软件更新篡改、外部接口攻击、数据泄露等）被较多地提及已采取了应对措施？"
            "哪些威胁类型应对不足？\n"
            "2) 案例威胁类型分布：受访者分享的实际网络安全案例中，威胁类型的分布情况如何？"
            "请按通信类攻击、软件更新类、外部连接类、数据/隐私泄露类、加密/算法类、人员/管理类、硬件类、服务器/云端类等维度归类分析。\n\n"
            "请详细列出各项发现，引用受访者的原话作为支持依据，并尽可能提供量化信息（如多少受访者提及了某类威胁或措施）。"
        ),
    },
    {
        "group_id": "G02",
        "group_name": "技术防护措施与挑战",
        "description": "汇总技术层面的安全防护措施和遇到的技术挑战",
        "insight_questions": ["Q2", "Q3"],
        "combined_question": (
            "请基于所有受访者的访谈内容，汇总他们在网络安全方面采取的技术防护措施以及遇到的技术挑战：\n\n"
            "1) 技术防护措施：受访者部署了哪些技术手段来保障网络安全？"
            "例如加密技术、身份认证与PKI、入侵检测系统(IDS/IPS)、安全通信协议（如TLS、IPSec）、代码签名、安全启动(Secure Boot)、"
            "防火墙、网络隔离/分段、HSM安全硬件、TEE可信执行环境、CAN总线保护、V2X安全机制、OTA安全更新机制等。\n"
            "2) 技术挑战：在实施这些技术防护时遇到了哪些困难或局限性？"
            "例如资源限制（算力/存储/带宽）、技术复杂度、兼容性问题、实时性要求、密钥管理困难、供应链安全等。\n\n"
            "请详细列出各项发现并引用相关受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G03",
        "group_name": "组织防护措施与挑战",
        "description": "汇总组织管理层面的安全防护措施和遇到的管理挑战",
        "insight_questions": ["Q4", "Q5"],
        "combined_question": (
            "请基于所有受访者的访谈内容，汇总他们在网络安全方面采取的组织管理措施以及遇到的管理挑战：\n\n"
            "1) 组织防护措施：受访者所在组织采取了哪些非技术性的安全管理措施？"
            "例如制定安全政策和流程、安全培训与意识提升、建立网络安全管理团队和职责分工、供应链安全管理、"
            "第三方安全审计、安全开发生命周期(SDL)管理、威胁分析与风险评估(TARA)、应急响应机制、安全文化建设等。\n"
            "2) 组织挑战：在实施这些组织管理措施时遇到了哪些困难或障碍？"
            "例如管理层支持不足、预算限制、人才短缺、跨部门协作困难、供应商管理复杂、流程执行不到位等。\n\n"
            "请详细列出各项发现并引用相关受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G04",
        "group_name": "工具使用与工具链",
        "description": "分析工具使用普及率、工具链覆盖完整度、缺失工具及局限性",
        "insight_questions": ["Q6", "Q7", "Q8", "Q9"],
        "combined_question": (
            "请基于所有受访者的访谈内容，综合分析网络安全工具使用情况：\n\n"
            "1) 工具使用普及率：受访者使用了哪些网络安全工具？每种工具被多少受访者提到？"
            "哪些工具使用最广泛？哪些工具只有少数受访者使用？\n"
            "2) 工具链覆盖完整度：目前工具链主要覆盖了哪些环节（如代码安全分析、通信安全测试、整车安全测试、云安全测试等）？"
            "哪些环节覆盖较好？哪些环节存在明显缺失？\n"
            "3) 缺失工具：受访者明确提到缺少或不足的工具是什么？哪些需求被多次提及？\n"
            "4) 工具局限性：现有工具有哪些限制或不足（如功能不全、误报率高、集成困难、成本高、操作复杂等）？\n\n"
            "请详细列出各项发现，尽可能提供量化信息，并引用受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G05",
        "group_name": "网络安全知识与技能",
        "description": "分析网络安全技能重要度、知识获取渠道、挑战及期望培训",
        "insight_questions": ["Q10", "Q11", "Q12", "Q13"],
        "combined_question": (
            "请基于所有受访者的访谈内容，综合分析网络安全知识与技能方面的情况：\n\n"
            "1) 技能重要度：受访者认为哪些网络安全知识或专业技能最为重要？"
            "各种技能被提及的重要程度排序如何（如渗透测试、密码学、安全架构、固件安全、合规知识、风险评估等）？\n"
            "2) 知识获取渠道：受访者及其团队通常通过哪些渠道获取或更新网络安全知识？"
            "如内部培训、行业会议/峰会、在线课程、认证培训、学术论文、开源社区、供应商培训等。\n"
            "3) 获取挑战：在获取或更新网络安全知识时遇到了哪些挑战？"
            "如知识更新快、培训资源有限、缺乏实践机会、内容缺乏针对性等。\n"
            "4) 期望培训：受访者希望市场提供哪些课程或训练来培养网络安全人才？\n\n"
            "请详细列出各项发现，尽可能提供量化信息，并引用受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G06",
        "group_name": "基础设施需求与限制",
        "description": "分析网络安全基础设施需求和存在的限制",
        "insight_questions": ["Q14", "Q15"],
        "combined_question": (
            "请基于所有受访者的访谈内容，综合分析网络安全基础设施方面的情况：\n\n"
            "1) 基础设施需求：受访者提到了哪些基础设施或组织支持可以促进网络安全活动？"
            "团队目前需要哪些基础设施支持来开展网络安全工作？哪些需求被多次提及？\n"
            "2) 基础设施限制：存在哪些限制有效安全实施的基础设施约束或不足？"
            "例如测试实验室/环境不足、云平台限制、硬件资源不足、网络带宽限制、安全工具平台缺乏等。\n\n"
            "请详细列出各项发现，尽可能提供量化信息，并引用受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G07",
        "group_name": "案例安全分析与处置",
        "description": "分析实际案例中威胁触发原因和处置手段",
        "insight_questions": ["Q17", "Q18"],
        "combined_question": (
            "请基于所有受访者的访谈内容，分析他们分享的实际网络安全案例：\n\n"
            "1) 威胁触发原因：在案例中，网络安全威胁的触发原因或根本原因有哪些？"
            "例如软件漏洞、配置错误、人为失误、供应链问题、缺乏安全测试、外部攻击等。\n"
            "2) 处置手段：针对这些威胁案例，受访者采取了哪些处置或缓解手段？"
            "例如补丁修复、系统隔离、应急响应、流程改进、安全加固、密钥轮换、策略调整等。\n\n"
            "请详细列出各项发现，并引用相关受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G08",
        "group_name": "测试经验与方法",
        "description": "分析测试方法频次、局限性、挑战和期望方法",
        "insight_questions": ["Q19", "Q20", "Q21", "Q22"],
        "combined_question": (
            "请基于所有受访者的访谈内容，综合分析网络安全测试方面的经验：\n\n"
            "1) 测试方法频次：受访者在车辆系统开发或运行期间使用了哪些网络安全测试或验证方法？"
            "各种方法被提及的频次如何（如渗透测试、模糊测试、代码审计、漏洞扫描、合规检查、红队演练等）？\n"
            "2) 测试局限性：这些测试方法在哪些情况下效果较差或存在哪些局限性？"
            "例如覆盖不全、无法模拟真实场景、耗时耗力、自动化程度低等。\n"
            "3) 测试挑战：在测试过程中遇到了哪些困难或挑战？"
            "如缺乏测试环境、测试工具不足、测试人员技能不足、时间预算压力、无法覆盖所有攻击面等。\n"
            "4) 期望方法：受访者希望拥有或看到更成熟的哪些网络安全测试方法或工具？\n\n"
            "请详细列出各项发现，尽可能提供量化信息，并引用受访者的原话作为支持依据。"
        ),
    },
    {
        "group_id": "G09",
        "group_name": "合规实践",
        "description": "分析标准提及频次、标准认知广度和合规挑战",
        "insight_questions": ["Q23", "Q24", "Q25"],
        "combined_question": (
            "请基于所有受访者的访谈内容，综合分析网络安全合规方面的情况：\n\n"
            "1) 标准提及频次：受访者提到了哪些网络安全相关法规或标准？"
            "每条标准（如ISO 21434、UN R155、R156、GDPR、WP.29、国标GB等）被多少受访者提及？\n"
            "2) 标准认知广度：受访者平均了解或提及多少条不同的标准？"
            "不同受访者之间的标准认知范围差异如何？\n"
            "3) 合规挑战：在网络安全合规过程中遇到了哪些最具挑战性的方面？"
            "如标准理解困难、合规成本高、标准与具体实践脱节、跨地区标准差异、合规流程复杂等。\n\n"
            "请详细列出各项发现，尽可能提供量化信息，并引用受访者的原话作为支持依据。"
        ),
    },
]


# ============================================================
# CSV 数据加载
# ============================================================

class InterviewAnswerStore:
    """加载和管理所有 res/访谈*.csv 文件的回答数据"""

    def __init__(self, res_dir: Path):
        self.res_dir = res_dir
        # interviews[interviewee_name][question_id] = {"answer": str, "citation_path": str}
        self.interviews: dict[str, dict[str, dict[str, str]]] = {}
        self._load_all()

    def _load_all(self) -> None:
        csv_files = sorted(self.res_dir.glob("访谈*.csv"))
        for csv_path in csv_files:
            interviewee = self._extract_interviewee_name(csv_path)
            if not interviewee:
                continue
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                answers: dict[str, dict[str, str]] = {}
                for row in reader:
                    qid = row.get("question_id", "").strip()
                    if not qid:
                        continue
                    answer_text = row.get("answer", "").strip()
                    citation_path = row.get("citation_path", "").strip()
                    if answer_text:
                        answers[qid] = {
                            "answer": answer_text,
                            "citation_path": citation_path,
                        }
                self.interviews[interviewee] = answers

    def _extract_interviewee_name(self, csv_path: Path) -> str:
        """从CSV文件名提取受访者名称前缀，如 '访谈1', '访谈2' 等"""
        name = csv_path.stem
        # 截取 "访谈N" 部分，如 "访谈1-一汽访谈" -> "访谈1"
        match = re.match(r"(访谈\d+)", name)
        if match:
            return match.group(1)
        return name

    def get_all_interviewees(self) -> list[str]:
        """返回所有受访者名称列表（按访谈编号排序）"""
        def sort_key(name: str) -> int:
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else 0
        return sorted(self.interviews.keys(), key=sort_key)

    def match_question_ids(self, qid: str, config: dict) -> bool:
        """判断一个question_id是否匹配某个洞察问题的配置"""
        # 精确匹配
        if qid in config.get("protocol_include", []):
            return True

        # 前缀匹配
        for prefix in config.get("protocol_include_prefix", []):
            if qid.startswith(prefix):
                # 如果指定了suffix_filter，只保留匹配后缀的问题
                suffixes = config.get("suffix_filter", [])
                if suffixes:
                    # 检查最后一个数字后缀
                    last_part = qid.rsplit(".", 1)[-1] if "." in qid else ""
                    if f".{last_part}" in suffixes or last_part in suffixes:
                        return True
                else:
                    return True
        return False

    def get_answers_for_insight(
        self,
        insight_qid: str,
    ) -> dict[str, list[dict[str, str]]]:
        """获取某个洞察问题对应的所有访谈回答
        
        返回: {interviewee: [{question_id, question_text, answer}, ...]}
        """
        config = QUESTION_PROTOCOL_MAP.get(insight_qid)
        if not config:
            return {}

        # 加载协议CSV以获取question_text
        protocol_qid_to_text: dict[str, str] = {}
        proto_path = self.res_dir.parent / "interview_protocol_CN.csv"
        if proto_path.exists():
            with proto_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    qid = row.get("question_id", "").strip()
                    qtext = row.get("question_text", "").strip()
                    if qid:
                        protocol_qid_to_text[qid] = qtext

        result: dict[str, list[dict[str, str]]] = {}
        for interviewee in self.get_all_interviewees():
            interviewee_answers = self.interviews.get(interviewee, {})
            matched: list[dict[str, str]] = []
            for qid, data in interviewee_answers.items():
                if self.match_question_ids(qid, config):
                    matched.append({
                        "question_id": qid,
                        "question_text": protocol_qid_to_text.get(qid, ""),
                        "answer": data["answer"],
                        "citation_path": data["citation_path"],
                    })
            if matched:
                result[interviewee] = matched
        return result


# ============================================================
# Prompt 构建与 LLM 调用
# ============================================================

def _build_synthesis_prompt(
    combined_question: str,
    group_name: str,
    interviewee_data: dict[str, list[dict[str, str]]],
    batch_info: str = "",
) -> list[dict[str, str]]:
    """构建LLM综合分析的prompt"""
    sections: list[str] = []
    for interviewee in sorted(interviewee_data.keys()):
        answers = interviewee_data[interviewee]
        section = f"【{interviewee}】"
        for item in answers:
            qid = item["question_id"]
            qtext = item["question_text"]
            answer = item["answer"]
            # 截取过长的question_text
            qtext_short = qtext[:60] + "..." if len(qtext) > 60 else qtext
            section += f"\n  [{qid}] {qtext_short}"
            section += f"\n    回答: {answer}"
        sections.append(section)

    context = "\n\n".join(sections)
    total_qa_pairs = sum(len(v) for v in interviewee_data.values())
    total_interviewees = len(interviewee_data)

    user_prompt = (
        "你是严谨的证据型数据分析助手。请基于所有受访者的问答内容，对指定的分析问题进行综合分析和回答。\n\n"
        f"分析主题：{group_name}\n"
        f"分析问题：{combined_question}\n\n"
        f"{batch_info}"
        f"共 {total_interviewees} 位受访者，{total_qa_pairs} 条问答记录：\n\n"
        f"{context}\n\n"
        "请输出JSON对象，格式为：\n"
        '{"answer":"...","evidence":[{"interviewee":"受访者名","question_id":"...","reason":"..."}]}\n'
        "要求：\n"
        "1) answer中必须全面分析所有受访者的回答，给出具体的发现和结论；\n"
        "2) 在列出重要发现时，应注明信息来源的受访者（如【访谈1】）；\n"
        "3) 如果发现具有量化特征（如多少受访者提及某项措施），应明确写出量化统计；\n"
        "4) evidence应列出最关键的支持证据，每条注明对应的受访者和问题；\n"
        "5) 仅输出JSON，不要额外文本。"
    )

    return [{"role": "user", "content": user_prompt}]


def _call_llm_json(
    messages: list[dict[str, str]],
    description: str,
    llm_model: str,
    api_base: str,
    api_key: str,
    logger: Any,
    max_retries: int = 3,
) -> dict[str, Any]:
    """调用LLM API并解析JSON响应"""
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    last_raw = None
    for attempt in range(max_retries):
        try:
            logger.info(f"  LLM调用: {description} (尝试 {attempt+1}/{max_retries})")
            resp = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            last_raw = content

            logger.info(f"  tokens: input={resp.usage.prompt_tokens}, output={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"JSON解析失败，尝试提取: {description}")
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(content[start:end+1])
                    except json.JSONDecodeError:
                        pass

            if attempt < max_retries - 1:
                import time
                wait = 5 ** attempt
                logger.warning(f"JSON解析失败，{wait}秒后重试...")
                time.sleep(wait)
                continue
            else:
                raise RuntimeError(f"LLM响应无法解析为JSON: {content[:200]}")

        except Exception as e:
            import time
            if attempt < max_retries - 1:
                wait = 5 ** attempt
                logger.warning(f"LLM异常: {e}, {wait}秒后重试...")
                time.sleep(wait)
                continue
            else:
                raise RuntimeError(f"LLM调用在{max_retries}次重试后仍失败: {e}")

    raise RuntimeError(f"LLM调用失败，last_raw={last_raw[:200] if last_raw else 'None'}")


# ============================================================
# 后处理：引用内联
# ============================================================

def _insert_inline_citations(answer: str, citations: list[dict]) -> str:
    """在答案文本中为自然语言引用（如【访谈1】）插入编号标记"""
    if not citations:
        answer = re.sub(r'\s*参考引用：\s*(\[#\d+\]\s*)+$', '', answer, flags=re.DOTALL)
        return answer.rstrip()

    # 去除末尾堆积
    answer = re.sub(r'\s*参考引用：\s*(\[#\d+\]\s*)+$', '', answer, flags=re.DOTALL)
    answer = answer.rstrip()

    # 构建引用查找表: interviewee_name -> [citation_index]
    cite_lookup: dict[str, list[int]] = {}
    for i, cite in enumerate(citations):
        interviewee = cite.get("interviewee", "")
        if interviewee:
            if interviewee not in cite_lookup:
                cite_lookup[interviewee] = []
            cite_lookup[interviewee].append(i + 1)

    # 在【访谈N】后插入 [引用标记]
    def _inject(match):
        name = match.group(0)
        idxs = cite_lookup.get(name, [])
        if idxs:
            tags = " ".join(f"[#{i}]" for i in sorted(idxs))
            return f"{name} {tags}"
        return name

    answer = re.sub(r'【(访谈\d+)】', _inject, answer)
    return answer


# ============================================================
# 保存结果
# ============================================================

def save_group_result(
    group: dict[str, Any],
    answer_text: str,
    citations: list[dict[str, Any]],
    store: InterviewAnswerStore,
    output_dir: Path,
) -> Path:
    """保存单个组的问答结果"""
    group_id = group["group_id"]
    group_dir = output_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=True)

    # 后处理引用
    cleaned_answer = _insert_inline_citations(answer_text, citations)

    # 保存 answer.md
    metadata = {
        "group_id": group["group_id"],
        "group_name": group["group_name"],
        "insight_questions": group["insight_questions"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "citation_count": len(citations),
        "answered_interviewees": [
            {
                "interviewee": c.get("interviewee", ""),
                "question_id": c.get("question_id", ""),
                "reason": c.get("reason", ""),
            }
            for c in citations
        ],
    }

    md_lines = [
        f"# {group_id}: {group['group_name']}",
        f"",
        f"**描述**: {group.get('description', '')}",
        f"",
        f"**分析时间**: {metadata['timestamp']}",
        f"",
        f"**统计**:",
        f"- 引用数: {len(citations)}",
        f"- 涉及访谈数: {len(set(c.get('interviewee','') for c in citations))}",
        f"",
        f"**映射的洞察问题**: " + ", ".join(group["insight_questions"]),
        f"",
        f"---",
        f"",
        f"## 分析结果",
        f"",
        cleaned_answer,
        f"",
    ]

    if citations:
        md_lines.append(f"---")
        md_lines.append(f"## 支持依据")
        md_lines.append(f"")
        for i, c in enumerate(citations, 1):
            interviewee = c.get("interviewee", "")
            qid = c.get("question_id", "")
            reason = c.get("reason", "")
            md_lines.append(f"### [#{i}] {interviewee} - {qid}")
            md_lines.append(f"")
            md_lines.append(f"- **理由**: {reason}")
            md_lines.append(f"")

    answer_path = group_dir / "answer.md"
    answer_path.write_text("\n".join(md_lines), encoding="utf-8")

    # 保存完整结果JSON
    result = {
        "group_id": group_id,
        "group_name": group["group_name"],
        "insight_questions": group["insight_questions"],
        "question": group["combined_question"],
        "answer": cleaned_answer,
        "citations": citations,
        "metadata": metadata,
    }
    result_path = group_dir / "full_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return group_dir


def print_group_summary(group: dict, answer_text: str, citations: list, logger: Any) -> None:
    preview = answer_text[:150].replace("\n", " ").strip()
    if len(answer_text) > 150:
        preview += "..."
    interviewees = set(c.get("interviewee", "") for c in citations)
    logger.info(f"  [结果] 引用={len(citations)}, 涉及访谈={len(interviewees)}")
    logger.info(f"  [摘要] {preview}")


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="跨访谈Insight分析（CSV直读模式）- 从res/访谈*.csv直接读取回答进行分析"
    )
    parser.add_argument("--res-dir", default="res", help="访谈回答CSV目录（含访谈*.csv文件）")
    parser.add_argument("--output-dir", default="insight_res", help="洞察结果输出根目录")
    parser.add_argument("--llm-model", default="deepseek-v4-flash", help="问答LLM模型名")
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="LLM API base url")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"), help="LLM API key")
    parser.add_argument("--limit-groups", type=int, default=0, help="只处理前N个组，0为全部")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有结果的组")
    parser.add_argument("--debug", default=True, help="启用调试日志")
    args = parser.parse_args()

    logger = setup_logger(debug=args.debug)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # 加载所有CSV数据
    res_dir = Path(args.res_dir)
    logger.info(f"加载访谈回答数据: {res_dir}")
    store = InterviewAnswerStore(res_dir)
    interviewees = store.get_all_interviewees()
    logger.info(f"加载完成: {len(interviewees)} 位受访者")

    # 打印每个洞察问题的数据统计
    total_qa = 0
    for qid, config in sorted(QUESTION_PROTOCOL_MAP.items()):
        data = store.get_answers_for_insight(qid)
        qa_count = sum(len(v) for v in data.values())
        total_qa += qa_count
        logger.debug(f"  {qid} ({config['description']}): {qa_count} 条回答, {len(data)} 位受访者")
    logger.info(f"总匹配问答对: {total_qa}")

    # 保存映射定义
    mapping_path = output_base / "protocol_mapping.json"
    mapping_data = {}
    for g in QUESTION_GROUPS:
        for iq in g["insight_questions"]:
            mapping_data[iq] = {
                "description": QUESTION_PROTOCOL_MAP[iq]["description"],
                "include_qids": QUESTION_PROTOCOL_MAP[iq].get("protocol_include", []),
                "include_prefixes": QUESTION_PROTOCOL_MAP[iq].get("protocol_include_prefix", []),
                "suffix_filters": QUESTION_PROTOCOL_MAP[iq].get("suffix_filter", []),
            }
    mapping_path.write_text(
        json.dumps(mapping_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"协议映射定义已保存至: {mapping_path}")

    # 处理每个组
    groups_to_process = QUESTION_GROUPS
    if args.limit_groups > 0:
        groups_to_process = QUESTION_GROUPS[:args.limit_groups]
        logger.info(f"限制处理前 {args.limit_groups} 个组")

    total_groups = len(groups_to_process)
    summary_results: list[dict[str, Any]] = []

    for idx, group in enumerate(groups_to_process, 1):
        group_id = group["group_id"]
        group_name = group["group_name"]
        insight_qs = group["insight_questions"]

        # 跳过已有结果
        group_dir = output_base / group_id
        if args.skip_existing and (group_dir / "full_result.json").exists():
            logger.info(f"[{idx}/{total_groups}] 跳过 {group_id}: {group_name}（已有结果）")
            try:
                existing = json.loads((group_dir / "full_result.json").read_text("utf-8"))
                summary_results.append(existing)
            except Exception:
                pass
            continue

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"[{idx}/{total_groups}] 开始处理组 {group_id}: {group_name}")
        logger.info(f"  洞察问题: {insight_qs}")
        for iq in insight_qs:
            config = QUESTION_PROTOCOL_MAP[iq]
            data = store.get_answers_for_insight(iq)
            qa_count = sum(len(v) for v in data.values())
            logger.info(f"    - {iq} ({config['description']}): {qa_count} 条回答")
        logger.info(f"{'='*60}")

        # 收集该组所有洞察问题的数据
        group_data: dict[str, list[dict[str, str]]] = {}
        for iq in insight_qs:
            iq_data = store.get_answers_for_insight(iq)
            for interviewee, answers in iq_data.items():
                if interviewee not in group_data:
                    group_data[interviewee] = []
                group_data[interviewee].extend(answers)

        if not group_data:
            logger.warning(f"  组 {group_id} 无匹配数据，跳过")
            continue

        total_answers = sum(len(v) for v in group_data.values())
        logger.info(f"  总数据: {len(group_data)} 位受访者, {total_answers} 条回答")

        # 判断是否需要分批处理（阈值：总字符数超过50000）
        total_chars = sum(
            len(item["answer"]) + len(item["question_text"])
            for answers in group_data.values()
            for item in answers
        )
        logger.info(f"  总数据字符数: {total_chars}")

        # 调用LLM
        try:
            if total_chars > 50000:
                logger.info(f"  数据量大，采用分批合成模式")

                # 分批：每位受访者先独立分析
                batch_summaries: list[dict[str, Any]] = []
                all_citations: list[dict[str, Any]] = []

                for batch_idx, interviewee in enumerate(sorted(group_data.keys()), 1):
                    batch_data = {interviewee: group_data[interviewee]}
                    prompt_messages = _build_synthesis_prompt(
                        combined_question=f"这是第 {batch_idx}/{len(group_data)} 批分析。\n请仅分析{interviewee}的回答，提炼出与上述分析主题相关的关键信息。不要跨受访者总结。",
                        group_name=f"{group_name} (分批-{interviewee})",
                        interviewee_data=batch_data,
                        batch_info=f"这是第 {batch_idx}/{len(group_data)} 批，独立分析单名受访者。\n",
                    )
                    try:
                        parsed = _call_llm_json(
                            prompt_messages,
                            f"{group_id} 分批 {batch_idx}/{len(group_data)} {interviewee}",
                            args.llm_model,
                            args.api_base,
                            args.api_key,
                            logger,
                        )
                        summary = parsed.get("answer", "")
                        evidence = parsed.get("evidence", []) or []
                        batch_summaries.append({
                            "interviewee": interviewee,
                            "summary": summary,
                            "evidence_count": len(evidence),
                        })
                        for ev in evidence:
                            ev["interviewee"] = interviewee
                            all_citations.append(ev)
                    except Exception as exc:
                        logger.error(f"  分批 {interviewee} 失败: {exc}")
                        batch_summaries.append({
                            "interviewee": interviewee,
                            "summary": f"（处理失败）",
                            "evidence_count": 0,
                        })

                # 第二阶段：综合所有批次的摘要，生成最终答案
                batch_context = "\n\n".join(
                    f"【{b['interviewee']}】分析摘要:\n{b['summary']}"
                    for b in batch_summaries
                )
                final_prompt = [
                    {
                        "role": "user",
                        "content": (
                            "你是严谨的证据型数据分析助手。请综合以下各受访者的分析摘要，生成一份全面的最终分析报告。\n\n"
                            f"分析主题：{group_name}\n"
                            f"分析问题：{group['combined_question']}\n\n"
                            f"共有 {len(batch_summaries)} 位受访者的分摘要：\n\n"
                            f"{batch_context}\n\n"
                            "请输出JSON对象，格式为：\n"
                            '{"answer":"...","evidence":[{"interviewee":"受访者名","question_id":"...","reason":"..."}]}\n'
                            "要求：\n"
                            "1) answer必须是全面的、综合所有受访者的最终分析报告；\n"
                            "2) 重要发现必须注明来自哪位受访者；\n"
                            "3) 尽可能提供量化统计信息；\n"
                            "4) evidence列出所有关键支持证据；\n"
                            "5) 仅输出JSON，不要额外文本。"
                        ),
                    }
                ]
                logger.info(f"  综合阶段: 发送最终合成请求")
                final_parsed = _call_llm_json(
                    final_prompt,
                    f"{group_id} 最终合成",
                    args.llm_model,
                    args.api_base,
                    args.api_key,
                    logger,
                )
                answer_text = final_parsed.get("answer", "（合成失败）")
                citations = final_parsed.get("evidence", []) or []
                # 如果没有citations，使用分批收集的
                if not citations:
                    citations = all_citations

            else:
                # 单次合成
                prompt_messages = _build_synthesis_prompt(
                    combined_question=group["combined_question"],
                    group_name=group_name,
                    interviewee_data=group_data,
                )
                parsed = _call_llm_json(
                    prompt_messages,
                    f"{group_id} 合成",
                    args.llm_model,
                    args.api_base,
                    args.api_key,
                    logger,
                )
                answer_text = parsed.get("answer", "（无答案）")
                citations = parsed.get("evidence", []) or []

            # 保存结果
            result_dir = save_group_result(group, answer_text, citations, store, output_base)
            summary_results.append({
                "group_id": group_id,
                "group_name": group_name,
                "answer": answer_text,
                "citations": citations,
            })

            logger.info(f"  ✓ 完成 {group_id}")
            print_group_summary(group, answer_text, citations, logger)
            logger.info(f"  结果保存至: {result_dir}")

        except Exception as exc:
            logger.error(f"  ✗ 组 {group_id} 处理失败: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            summary_results.append({
                "group_id": group_id,
                "group_name": group_name,
                "answer": f"（处理失败: {exc}）",
                "citations": [],
            })

    # 生成汇总报告
    summary_path = output_base / "SUMMARY.md"
    summary_lines = [
        "# 跨访谈洞察分析汇总报告",
        f"",
        f"**生成时间**: {datetime.now().isoformat(timespec='seconds')}",
        f"**分析组数**: {len(summary_results)}",
        f"",
        f"---",
        f"",
    ]
    for sr in summary_results:
        gid = sr["group_id"]
        gname = sr["group_name"]
        answer = sr.get("answer", "")
        citations = sr.get("citations", [])
        preview = answer[:300].replace("\n", " ").strip()
        if len(answer) > 300:
            preview += "..."

        summary_lines.extend([
            f"## {gid}: {gname}",
            f"",
            f"- **引用数**: {len(citations)}",
            f"- **详细报告**: [{gid}/answer.md]({gid}/answer.md)",
            f"",
            f"**分析摘要**:",
            f"",
            f"> {preview}",
            f"",
            f"---",
            f"",
        ])

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"所有洞察分析完成！")
    logger.info(f"  处理组数: {len(summary_results)}/{total_groups}")
    logger.info(f"  详细结果: {output_base}/G*/answer.md")
    logger.info(f"  汇总报告: {summary_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
