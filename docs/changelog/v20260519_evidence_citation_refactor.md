# 证据结构重构与引用格式规范化

**日期：** 2026-05-19
**最后更新：** 2026-05-20
**目标：** 解决多段组合引用问题、LLM 输出非标准引用格式导致过滤/重排出错的问题

---

## 改动总览

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `kb_qa/qa.py` | 重构 + 新增 | evidence 结构变更、新增校验方法、prompt 升级、重试逻辑 |

---

### Prompt 优化：引用标记紧跟所支持的分句

**问题：** LLM 有时将 `[#N]` 统一放在句子或段落末尾，导致引用标记离其所支持的内容过远，读者难以对应。

**优化：** prompt 中新增详细说明和正反示例：

> 引用标记 `[#N]` 要紧贴所支持的分句/段落之后，不要集中放到句子或段落末尾，更不要统一放到回答最后。
> 例如：
>   ✅ "她很喜欢广州粉丝 [#1]，多次询问粉丝是否会去广州看她 [#2][#3]。"
>   ❌ "她很喜欢广州粉丝，多次询问粉丝是否会去广州看她 [#1][#2][#3]。"
>   ❌ "她很喜欢广州粉丝。多次询问粉丝是否会去广州看她。[#1][#2][#3]"

---

### 修复：引用编号歧义 -> evidence 自包含 `citation_id`

**问题：** LLM 将 answer 中的 `[#N]` 理解为**输入片段列表中的位置**（如第 9 条片段 → `[#9]`），而非 evidence 条目编号，导致引用编号与 evidence 不匹配。

**修复：** evidence 输出格式新增 `citation_id` 字段，LLM 在构建 JSON 时直接标注 #1、#2，answer 直接引用这些编号即可。prompt 说明改为：

> 每个 evidence 条目的 `citation_id` 字段已标注编号（#1、#2...），你只需在 answer 中使用对应编号的引用标记即可。

输出格式示例更新为包含 `citation_id`：
```json
{"answer":"...","evidence":[{"citation_id":"#1","segment_ids":["...","..."],"citation_type":"...","reason":"..."}]}
```

从此不需要再解释"编号与输入片段列表中的位置无关"——因为编号是自包含在 JSON 输出中的。

---

## 1. Evidence 输出结构变化

### 背景

原来每个 evidence 条目只支持单条片段引用（`segment_id` 为字符串），LLM 无法将多个相关片段组合为一条引用，导致同一组片段分散在不同 evidence 条目中，缺乏整体性。

### 变化

**旧格式（已移除，不再兼容）：**
```json
{
  "evidence": [
    {"segment_id": "xxx", "reason": "..."}
  ]
}
```

**新格式（唯一支持）：**
```json
{
  "evidence": [
    {
      "segment_ids": ["xxx", "yyy"],
      "citation_type": "主播讲话|观众弹幕|互动对话",
      "reason": "..."
    }
  ]
}
```

- `segment_ids` (array)：可以包含一条或多条片段 ID
- `citation_type` (string)：该组证据的类型
  - 「主播讲话」：仅包含主播讲话片段
  - 「观众弹幕」：仅包含观众弹幕片段
  - 「互动对话」：同时包含主播讲话和观众弹幕（显示对话互动关系）

### 涉及的方法

**`_build_group_synthesis_prompt()`** — Prompt 更新
- 输出格式说明改为 `segment_ids` 数组 + `citation_type`
- evidence 可引用多条片段的组合

**`_build_citations_from_evidence()`** — 重写
- 只读取 `segment_ids` 数组，不再兼容 `segment_id` 单字段
- 生成的 citation 包含 `segments` 数组字段，包含每条片段的完整元信息
- **不再保留旧的顶层字段**（`segment_id`、`quoted_text` 等已移除）

**多视频合并路径（`ask()` 方法）**
- `seg_to_final_citation` 映射支持多段：遍历 `segments` 数组中的每个 segment_id 建立映射
- 重写答案引用编号时，优先用 `segments` 数组查找新编号
- **移除了所有旧格式兼容代码**（`c.get("segment_id")` 回退等）

---

## 2. 引用格式规范化：`[#N]` 严格格式

### 背景

LLM 输出的引用格式不统一，常见问题：
- `[#3, #34]`（逗号分隔多个引用）
- `[#18-20]`（缺少第二个 # 号）
- `[#2-#5]`（区间写法）

这些非标准格式导致 `_filter_citations_by_answer()` 解析遗漏，最终引用列表额外显示未使用的条目。

### 解决方案：三层机制

#### 第一层：Prompt 预防

**分组合成 prompt（`_build_group_synthesis_prompt`）** 和 **最终合并 prompt** 均添加了引用格式要求：

> ⚠️ 引用格式要求（重要）：
> 一个中括号内**只能有一个引用编号**，格式如 [#1]、[#5] 等。
> 禁止使用任何其他格式，包括但不限于：
>   - ❌ [#N-#M]（不允许区间写法，需要逐一列出 [#N][#M]）
>   - ❌ [#N, #M]（逗号分隔）
>   - ❌ [#N, #M, #K]（多个逗号分隔）
>   - ❌ [#N-M]（缺少#号）

#### 第二层：校验拦截

**新增 `_validate_answer_citations(answer: str) -> list[str]`：**
- 扫描答案中所有 `[...]` 括号
- 对含 `#` 的括号匹配正则 `^\[\#\d+\]$`
- 返回不符合格式的引用字符串列表

测试结果：
| 格式 | 结果 |
|------|------|
| `[#1]` | ✅ 合法 |
| `[#10]` | ✅ 合法 |
| `[#2-#5]` | ❌ 检出 |
| `[#3, #34]` | ❌ 检出 |
| `[#18-20]` | ❌ 检出 |
| `[#1][#2]` | ✅ 合法（两个独立引用） |

#### 第三层：自动重试

在最终答案合并后添加校验：
1. 若发现非法格式，拼接纠正消息告知 LLM 具体哪个引用格式有误
2. 将 LLM 之前的回答和纠正消息一起发回重试
3. 修正后的答案替换原答案继续流程

```python
invalid_refs = self._validate_answer_citations(answer_text)
if invalid_refs:
    correction_msg = (
        "⚠️ 答案中的以下引用格式不符合要求，请修正：\n"
        + "\n".join(f"  ❌ {r}" for r in invalid_refs)
        + "\n\n引用格式必须为 [#N]（如 [#1]、[#5]），..."
    )
    retry_prompt = list(merge_prompt)
    retry_prompt.append({"role": "assistant", "content": json.dumps({"answer": answer_text})})
    retry_prompt.append({"role": "user", "content": correction_msg})
    parsed, merge_llm_meta = self._call_llm_json(retry_prompt, "最终答案合并（引用格式修正）")
```

---

## 3. 引用过滤与重编号

### `_filter_citations_by_answer()` — 更新

保持不变，正则 `\[#(\d+)(?:-#(\d+))?\]` 仍兼容旧数据中遗留的 `[#N-#M]` 格式（展开为 `range(start, end+1)` 处理）。

### `_renumber_citations()` — 重写

过滤后重新从 #1 编号，同步更新答案中的引用标记：

- 支持连续区间压缩：`[#1][#2][#3]` → `[#1-#3]`
- 有 gap 的场景：`[#1][#2][#3][#5]` → `[#1-#3][#5]`
- 单个引用不压缩：`[#1][#4]` → `[#1][#4]`

示例：
```
Before filter: 根据[#2]和[#3]和[#4]的内容
Citations:      [#1, #2, #3, #4]
After filter:   保留 [#2, #3, #4]
After renumber: 根据[#1]和[#2]和[#3]的内容
Citations:      [#1, #2, #3]
```

---

## 4. Debug 日志新增

在两处关键 LLM 调用点添加了 debug 级别日志：

### 分组合成首个 LLM 调用
```python
# 触发条件：video_idx == 1 and batch_idx == 0
logger.debug("=== 分组处理首个 LLM 调用 - Prompt ===")
logger.debug(prompt_messages[0].get("content", ""))
logger.debug("=== 分组处理首个 LLM 调用 - Response ===")
logger.debug(json.dumps(parsed, ensure_ascii=False))
```

### 最终答案合并 LLM 调用
```python
logger.debug("=== 最终答案合并 LLM 调用 - Prompt ===")
logger.debug(merge_prompt[0].get("content", ""))
logger.debug("=== 最终答案合并 LLM 调用 - Response ===")
logger.debug(json.dumps(parsed, ensure_ascii=False))
```

### 修正重试
```python
logger.debug("=== 最终答案合并 LLM 调用 - 修正后 Response ===")
logger.debug(json.dumps(parsed, ensure_ascii=False))
```

---

## 5. 回答 JSON 结构变化

最终返回结果的 `citations` 列表中，每条 citation 现在包含 `segments` 数组和 `citation_type` 字段：

```json
{
  "citations": [
    {
      "citation_id": "#1",
      "segments": [
        {
          "segment_id": "xxx",
          "source_type": "speech",
          "quoted_text": "...",
          "video_offset": "00:01:23",
          "video_title": "...",
          "anchor_name": "SNH48-陈嘉仪",
          "live_id": "..."
        },
        {
          "segment_id": "yyy",
          "source_type": "danmaku",
          "quoted_text": "...",
          // ...
        }
      ],
      "citation_type": "互动对话",
      "reason": "该段中主播回应了观众的弹幕"
    }
  ]
}
```

> **注意：** 旧兼容字段（`segment_id`、`source_type`、`quoted_text` 等顶层字段）已移除，不再保留。

---

## 6. 测试验证

| 场景 | 结果 |
|------|------|
| 严格格式 `[#N]` 校验通过 | ✅ |
| `[#N-#M]` 被拒 | ✅ |
| `[#N, #M]` 被拒 | ✅ |
| 过滤 + 重编号全流程 | ✅ |
| 连续区间压缩 | ✅ |
| 有 gap 的区间压缩 | ✅ |
| 旧格式 `[#N-#M]` 向后兼容 | ✅ |
| 语法验证 | ✅ |

---

## 7. 2026-05-20 优化：`_renumber_citations` 区间压缩格式

`_renumber_citations` 会将连续区间压缩为 `[#1-#3]` 格式。`_filter_citations_by_answer` 的正则 `\[#(\d+)(?:-#(\d+))?\]` 兼容这种格式，`_validate_answer_citations` 只校验 LLM 输出（不校验重编号后的结果），所以区间压缩没有问题，保持原样。

---

## 8. 2026-05-20 修复：多视频合并时 `local_to_global` 映射错误

### 问题

多视频合并路径中，`local_to_global` 映射构建逻辑有严重 bug。代码遍历所有全局 citations，对每个全局 citation 的 segment，再遍历**所有视频分组**的 local citations 来匹配。由于 `break` 只跳出最内层 `for old_seg` 循环，一个 segment 可能匹配到**其他视频分组**的 local citation，导致 `local_to_global` 映射混乱。

例如：视频 1 的 local `[#1]` 包含 segment `A`，视频 2 的 local `[#5]` 也包含 segment `A`（因为全局去重后同一个 segment 只出现一次），结果 `local_to_global["#1"]` 和 `local_to_global["#5"]` 都被设为同一个全局编号，导致视频 1 的 answer 中 `[#1]` 被错误替换。

### 修复（两次迭代）

**第一次修复（错误方案）：** 遍历 `all_evidence`，对每个 local citation 的 segment_ids 集合与每个 evidence 条目的 segment_ids 集合做匹配。这种方法虽然正确，但效率低且逻辑复杂。

**第二次修复（正确方案）：** 在构建 `all_evidence` 时，记录每个视频分组的 evidence 在 `all_evidence` 中的起始索引（`_evidence_range`）。由于每个视频分组的 local citations 是 `all_evidence` 的一个连续子集且顺序不变，`local_to_global` 映射可以直接通过 `all_evidence_start_idx + local_idx` 计算得出，无需事后匹配。

具体改动：
1. 在 `video_results` 中新增 `_evidence_range` 字段，记录该分组 evidence 在 `all_evidence` 中的 `(start, end)` 范围
2. `local_to_global` 映射直接通过 `ev_start + local_idx` 计算全局索引
3. 移除了之前复杂的 `for ev_idx, ev in enumerate(all_evidence)` 匹配循环

这样视频 1 的 `[#1]` 会正确映射到全局 `[#1]`（如果它在 all_evidence 中是第 1 条），视频 2 的 `[#1]` 会映射到全局 `[#4]`（如果它在 all_evidence 中是第 4 条），以此类推。

### 补充修复：在 per-video 循环中立即计算全局编号

**问题：** 之前 `local_to_global` 映射和 `remapped_answer` 的计算分散在 per-video 循环和 merge 阶段两处，逻辑复杂且容易出错。`evidence_lines` 使用 `seg_to_final_citation` 查找全局编号，但同一个 segment 可能被多个视频分组的 LLM 引用，导致编号错位。

**修复：** 在 per-video 循环中，得到 `_evidence_range` 后**立即**计算全局编号：
1. 根据 `ev_start + local_idx` 计算每个 local citation 对应的全局 citation_id
2. 替换 answer 中的 local `[#N]` 为全局 `[#N]`，存入 `answer_global`
3. 生成全局编号的 citations 副本，存入 `citations_global`
4. merge 阶段直接使用 `answer_global` 和 `citations_global`，无需任何映射

这样彻底消除了 merge 阶段的 `local_to_global` 映射和 `seg_to_final_citation` 映射，逻辑更清晰、更可靠。

---

## 9. 2026-05-20 彻底移除旧格式兼容代码

### 背景

之前 `_build_citations_from_evidence()` 保留了旧格式 `segment_id` 单字段的向后兼容逻辑，以及 `seg_to_final_citation`、`local_to_global` 映射中的旧格式回退代码。这些兼容代码增加了维护成本，且可能导致 `TypeError: unhashable type: 'dict'`（`segments` 是 dict 列表不能直接 `set()`）。

### 变更

以下旧格式兼容代码已全部移除：

| 位置 | 移除内容 |
|------|---------|
| `_build_citations_from_evidence()` | `item.get("segment_id")` 单字段回退；顶层旧字段（`segment_id`、`source_type`、`quoted_text` 等） |
| `seg_to_final_citation` 构建 | `if c.get("segment_id"): seg_to_final_citation.setdefault(c["segment_id"], c_id)` |
| `local_to_global` 映射 | `elif ev.get("segment_id"): ev_sids = {ev["segment_id"]}` |
| `video_summaries` evidence 行 | `if not final_id and c.get("segment_id"): final_id = seg_to_final_citation.get(c["segment_id"], c["citation_id"])` |
| Prompt 示例 | `_build_judge_prompt` 和 `_synthesize_with_batches` 中的 JSON 格式示例从 `{"segment_id":"..."}` 改为 `{"segment_ids":["..."],"citation_type":"...","reason":"..."}` |

### 修复的 Bug

`TypeError: unhashable type: 'dict'` — 在 `local_to_global` 映射中，`set(old_c.get("segments", []) or [])` 试图对 dict 列表做 set 操作，因为 `segments` 是 `[{"segment_id": "xxx", ...}, ...]` 格式，dict 不可哈希。修复为遍历提取 `segment_id` 后再做 set 比较。

### 影响

- 所有 LLM 输出的 evidence 必须使用新格式 `{"segment_ids": [...], "citation_type": "...", "reason": "..."}`
- 旧格式 `{"segment_id": "...", "reason": "..."}` 不再被识别
- 最终返回的 `citations` 列表中不再包含旧兼容顶层字段
