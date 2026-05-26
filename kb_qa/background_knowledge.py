"""背景知识管理：加载并格式化成员档案、平台术语等外部知识，供 LLM prompt 注入。

典型用法：
    bk = BackgroundKnowledge()
    bk.load_from_dir(Path("docs/Background"))
    prompt_block = bk.to_prompt_block()
    # → "【背景知识】...（压缩后的表格和列表）"
"""

from __future__ import annotations

import re
from pathlib import Path


# ── 全局最大注入字符数 ──
# 所有背景知识合并后超过此长度将被截断，防止 prompt 过长
# 两个知识库文件合计约 ~8500 压缩后字符（保留所有表格列）
_MAX_BG_CHARS = 10000


class BackgroundKnowledge:
    """管理背景知识文档，支持从 markdown 文件加载并格式化为 LLM prompt 片段。"""

    def __init__(self):
        self._sections: list[dict[str, str]] = []

    # ── 加载 ──────────────────────────────────────────────────────────

    def load_from_file(self, path: Path) -> int:
        """从单个 markdown 文件加载背景知识。返回加载的字符数。"""
        path = Path(path)
        if not path.exists():
            return 0
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return 0

        # 提取文件标题（第一个 # 标题）
        title = path.stem
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        compressed = self._compress(text)
        self._sections.append({
            "title": title,
            "source": str(path),
            "text": compressed,
        })
        return len(compressed)

    def load_from_dir(self, path: Path, glob_pattern: str = "*.md") -> int:
        """从目录加载所有匹配的 markdown 文件。返回总加载字符数。

        加载顺序：优先加载成员档案类文件（体积小、高频使用），
        再加载平台术语类文件（体积大、低频使用）。
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            return 0
        total = 0
        files = sorted(path.glob(glob_pattern))

        # 按文件名排序，但让「个人档案」类的文件排在前面
        # 策略：包含"档案""个人""成员""背景"等词的先加载
        priority_keywords = ["档案", "个人", "成员"]
        priority_files = []
        normal_files = []
        for f in files:
            name = f.stem
            if any(kw in name for kw in priority_keywords):
                priority_files.append(f)
            else:
                normal_files.append(f)

        for md_file in priority_files + normal_files:
            total += self.load_from_file(md_file)
        return total

    def clear(self) -> None:
        self._sections.clear()

    @property
    def is_loaded(self) -> bool:
        return len(self._sections) > 0

    @property
    def total_chars(self) -> int:
        return sum(len(s["text"]) for s in self._sections)

    # ── 格式化输出 ────────────────────────────────────────────────────

    def to_compact_block(self, max_chars: int = 7000) -> str:
        """生成紧凑版背景知识，供检索改写等需要简短上下文的阶段使用。

        策略：从每个章节的每个子节中取前 N 行关键信息（表格行/列表项），
        保证覆盖更多子节而非只填满第一个大节。
        大表格（如基本档案）取 14 行，小表格取 5 行。
        """
        if not self._sections:
            return ""

        parts: list[str] = []
        total = 0

        for section in self._sections:
            remaining = max_chars - total
            if remaining <= 100:
                break

            # 按子节（## 标题）分组
            lines = section["text"].split("\n")
            sub_sections: list[list[str]] = []
            current_sub: list[str] = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("##"):
                    if current_sub:
                        sub_sections.append(current_sub)
                    current_sub = [stripped]
                elif stripped.startswith("|") or stripped.startswith("- ") or stripped.startswith("* "):
                    current_sub.append(stripped)

            if current_sub:
                sub_sections.append(current_sub)

            # 从每个子节取前 N 行
            compact_lines: list[str] = []
            for sub in sub_sections:
                compact_lines.append(sub[0])  # 子节标题
                rows = sub[1:]
                # 大表格（>10行）取前 12 行，小表格取前 3 行
                max_rows = 12 if len(rows) > 10 else 3
                for row in rows[:max_rows]:
                    compact_lines.append(row)

            compact = "\n".join(compact_lines)

            header = f"\n📋 {section['title']}"
            needed = len(header) + len(compact) + 2
            if total + needed > max_chars:
                avail = max_chars - total - len(header) - 5
                if avail > 50:
                    compact = compact[:avail] + "\n..."
                else:
                    break

            block = f"{header}\n{compact}"
            parts.append(block)
            total += len(block)

        if not parts:
            return ""

        return (
            "【补充参考信息】\n"
            "以下信息可供检索改写时参考，用于辅助联想和关键词扩展：\n\n"
            + "\n\n".join(parts)
        )

    def to_prompt_block(self, max_chars: int = _MAX_BG_CHARS) -> str:
        """将所有加载的背景知识格式化为 LLM prompt 可用的文本块。

        Args:
            max_chars: 最大字符数（超过的部分被截断）。
        """
        if not self._sections:
            return ""

        parts: list[str] = []
        total = 0

        for section in self._sections:
            remaining = max_chars - total
            if remaining <= 100:
                break

            header = f"\n📋 {section['title']}"
            content = section["text"]

            needed = len(header) + len(content) + 2  # +2 for \n\n
            if total + needed > max_chars:
                # 截断当前章节
                avail = max_chars - total - len(header) - 5
                if avail > 50:
                    content = content[:avail] + "\n..."
                else:
                    break

            block = f"{header}\n\n{content}"
            parts.append(block)
            total += len(block)

        if not parts:
            return ""

        return (
            "【补充参考信息】\n"
            "以下信息可供你在回答问题时参考使用，就像使用你训练数据中的常识一样——"
            "它们来自公开档案和平台资料，是可靠的背景知识。"
            "你可以自然而然地用它们来丰富你的回答，无需特殊标记或引用编号。"
            "\n\n" + "\n\n".join(parts)
        )

    # ── 内部: Markdown 压缩 ──────────────────────────────────────────

    @classmethod
    def _compress(cls, text: str) -> str:
        """压缩 markdown 文本：去 frontmatter、空行、注释、装饰线，精简表格。"""
        lines = text.split("\n")
        result: list[str] = []
        in_frontmatter = False
        skip_section = False

        # frontmatter 仅出现在文件最开头（第一行是 ---）
        # 后面出现的 --- 是 markdown 分隔线，不能当作 frontmatter
        first_real_line = ""
        for line in lines:
            s = line.strip()
            if s:
                first_real_line = s
                break

        if first_real_line == "---":
            in_frontmatter = True

        for line in lines:
            stripped = line.strip()

            # ── frontmatter（仅文件开头） ──
            if in_frontmatter:
                if stripped == "---":
                    in_frontmatter = False
                continue

            # ── 跳过 HTML 注释 ──
            if stripped.startswith("<!--"):
                if "-->" in stripped:
                    continue
                skip_section = True
                continue
            if skip_section:
                if "-->" in stripped:
                    skip_section = False
                continue
            if stripped.startswith("-->") and skip_section:
                skip_section = False
                continue

            # ── 跳过分隔线 ──
            if re.match(r"^[-=]{3,}$", stripped):
                continue

            # ── 跳过纯空行 ──
            if not stripped:
                continue

            # ── 跳过分隔线（|---| 格式的表格分隔行） ──
            if stripped.startswith("|") and all(c in "| :-" for c in stripped):
                continue

            result.append(stripped)

        return "\n".join(result)
