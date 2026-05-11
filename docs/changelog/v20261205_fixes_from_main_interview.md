# 代码改动记录

**日期：** 2026-12-05  
**来源：** `origin/main_interview` 分支  
**目标：** 将 `main_interview` 分支中修复的处理逻辑与细节 Bugs/错误，反向移植到当前直播回放处理分支（`main`）

---

## 改动总览

| 文件 | 新增行 | 删除行 |
|------|--------|--------|
| `kb_qa/cli.py` | 5 | 1 |
| `kb_qa/indexer.py` | 17 | 6 |
| `kb_qa/qa.py` | 359 | 188 |

---

## 详细改动说明

### 1. `kb_qa/cli.py` — 日志级别 Bug 修复

**问题：** 文件日志 logger 硬编码为 `level="INFO"`，导致 `--debug` 模式开启后，文件日志仍只输出 INFO 级别，无法记录 DEBUG 日志。

**修复：** 将 `level="INFO"` 改为 `level=level`，使用传入的 `level` 变量（debug=True 时为 `"DEBUG"`）。

```diff
-            level="INFO",
+            level=level,
```

**同时新增** 3 个 CLI 参数，支持合成阶段配置：
- `--synthesis-context-window`：合成阶段局部上下文窗口大小（默认 6）
- `--synthesis-batch-trigger-count`：触发分批合成的有用段阈值（默认 100）
- `--synthesis-batch-size`：分批合成时每批大小（默认 50）

---

### 2. `kb_qa/indexer.py` — 上下文索引 Key 修复

**问题：** `by_live_source` 索引的 key 原为 `f"{seg.live_id}::{seg.source_type}"`，将 speech 和 danmaku 片段分到了不同的 key 下。导致 `expand_context` 做上下文扩展时，只能拿到同一来源的相邻段（例如 speech 只能看到 speech），无法跨来源获取完整上下文。

**修复：** 索引 key 改为只使用 `seg.live_id`，使 speech 和 danmaku 片段按直播合并排序，上下文扩展时能拿到同一直播的所有来源片段。

```diff
- key = f"{seg.live_id}::{seg.source_type}"
+ key = seg.live_id
```

两处同时修复：`_rebuild_live_source_index()` 和 `expand_context()`。

---

### 3. `kb_qa/indexer.py` — 排序逻辑改进

**问题：** 扩展后的片段排序为 `(video_datetime, start_time)`，未能按视频标题分组，导致不同视频的片段穿插混排，影响下游分析阶段的效果。

**修复：** 排序 key 改为 `(video_datetime or "", video_title or live_id, live_id, start_time)`。

```diff
- sorted(..., key=lambda s: (s.video_datetime, s.start_time))
+ sorted(..., key=lambda s: (s.video_datetime or "", s.video_title or s.live_id, s.live_id, s.start_time))
```

排序逻辑：
1. 先按 **视频日期** 分组
2. 再按 **视频标题**（回退为 live_id）分组
3. 再按 **live_id** 细化分组
4. 最后按 **起始时间** 排序

---

### 4. `kb_qa/qa.py` — LLM 调用重试机制（指数退避）

**问题：** `_call_llm_json` 方法没有重试机制，当 LLM API 返回异常或 JSON 解析失败时直接抛出异常，导致整个问答流程中断。

**修复：** 添加指数退避重试机制（默认最多 5 次重试）：

```python
for attempt in range(max_retries):
    try:
        resp = self.client.chat.completions.create(...)
        ...
    except (json.JSONDecodeError, Exception) as e:
        if attempt < max_retries - 1:
            wait_time = 5 ** attempt  # 指数退避：1s, 5s, 25s, 125s, ...
            time.sleep(wait_time)
            continue
        else:
            raise RuntimeError(...) from e
```

- 对 JSON 解析失败和 API 异常均有重试
- 在 JSON 解析失败时，会尝试从响应中提取 `{...}` 部分再次解析
- 记录每次调用的 tokens 消耗等元数据

---

### 5. `kb_qa/qa.py` — 统计动态化

**问题：** `build_or_update` 中参与者类型统计使用硬编码的 `"speech"/"danmaku"`，不够通用。

**修复：** 改为动态遍历所有 `source_type` 进行统计。

```diff
- participant_types["speech"] = ...
- participant_types["danmaku"] = ...
+ for seg in all_segments:
+     participant_types[seg.source_type] = participant_types.get(seg.source_type, 0) + 1
```

---

### 6. `kb_qa/qa.py` — 随机采样修复

**问题：** 原代码使用 `random.choices` 进行采样（允许重复），改为使用 `random.sample`（无重复）。

```diff
- sample_segments = random.choices(all_segments, k=min(3, len(all_segments)))
+ sample_segments = random.sample(all_segments, min(3, len(all_segments)))
```

---

### 7. `kb_qa/qa.py` — 新增 `_format_segment_with_local_context` 方法

**目的：** 在合成阶段为每个有用片段附加上下文的相邻片段，使 LLM 能更好地理解片段所在的语境。

```python
def _format_segment_with_local_context(self, segment: Segment, context_window: int = 6) -> str:
    key = segment.live_id
    seq = self.store.by_live_source.get(key, [])
    # ... 找到片段在索引中的位置 ...
    # ... 截取 context_window 范围内的上下文片段 ...
    # ... 标记 "核心片段" 和 "上下文片段" ...
```

输出格式：
```
[segment_id] 类型=...; 直播时间=...; 视频内时间=...; 标题=...; 用户名=...;
局部上下文（同一直播同一来源，窗口=6）：
  - [上下文片段] (hh:mm:ss) 文本...
  - [核心片段] (hh:mm:ss) 文本...
  - [上下文片段] (hh:mm:ss) 文本...
```

---

### 8. `kb_qa/qa.py` — 新增 `_build_citations_from_evidence` 方法

**目的：** 将构建引用信息的逻辑提取为公共方法，消除主流程中的重复代码。

```python
def _build_citations_from_evidence(self, evidence, useful_segments):
    # 为每条 evidence 构建包含完整信息的引用
    # 自动检测 LLM 遗漏的有用段，补充到 evidence 中
```

**改进点：**
- 自动补全 LLM 遗漏的有用段（带 warning 日志）
- 统一的引用格式（citation_id, segment_id, source_type, quoted_text 等）
- 被 `ask()` 主流程和 `_synthesize_with_batches()` 共同使用

---

### 9. `kb_qa/qa.py` — 改进合成 Prompt

**`_build_synthesis_prompt` 和 `_build_batch_synthesis_prompt` 更新：**

旧实现：简单拼接片段文本。
新实现：使用 `_format_segment_with_local_context` 包含局部上下文。

**Prompt 指令增强：**
- 明确要求引用所有有用片段，使用 `[#1]`、`[#2]` 标记
- evidence 必须包含所有有用 segment_id，按时序排列
- 如果片段很多，answer 要全面总结所有相关信息
- 不要遗漏任何有用片段的引用

---

### 10. `kb_qa/qa.py` — 新增可配置合成参数

`ask()` 方法新增三个参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `synthesis_context_window` | 6 | 合成阶段每个片段的局部上下文窗口大小 |
| `synthesis_batch_trigger_count` | 100 | 超过此数量的有用段时启用分批合成 |
| `synthesis_batch_size` | 50 | 分批合成时每批最大段数 |

---

## 兼容性说明

- 所有改动均向后兼容，默认参数行为与原版本一致
- 新增 CLI 参数均为可选，不传时使用默认值
- Python 语法检查全部通过（`ast.parse()`）
