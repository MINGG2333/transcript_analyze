# 增量更新方案：新增访谈时的最佳实践

## 工作流回顾

当前完整流程（4个步骤）：
1. **`generate_records.py`** → 扫描 `/mnt/zhitainew/ttt/interview/*.vtt`，生成 `interview_records.json`
2. **`run_kb_qa.py build`** → 解析 VTT 文件，构建向量索引 + BM25 索引 + `segment_store.json`
3. **`batch_group_ask_protocol.py`** → 对每个问题组，用 LLM 回答所有访谈的问题，结果保存到 `res/` 文件夹
4. **`batch_generate_codebook.py`** → 读取 `res/` 下的 CSV，对每个问题调用 LLM 生成 codebook，保存到 `codebook/` 文件夹

---

## 各步骤的增量支持现状分析

### 步骤1: `generate_records.py` — ✅ 无需改，直接重跑
- 该脚本仅扫描 `.vtt` 文件生成 JSON 元数据，**无 LLM 调用，开销极小**
- **建议：** 每次新增访谈后直接重新运行即可

### 步骤2: `run_kb_qa.py build` — ✅ 已有增量支持
- `SegmentStore.upsert_many()` 会按 `segment_id` 对比文本和时间戳，**只处理新变更的片段**
- Chroma 向量索引只添加新片段，不影响已有索引
- BM25 索引在每次 build 时重建（但重建开销与片段数相关，可接受）
- **建议：** 直接重新运行 `python run_kb_qa.py build`

### 步骤3: `batch_group_ask_protocol.py` — ❌ 需改进
- `qa.ask_group()` 在内部对所有 `live_id`（所有访谈）逐个调用 LLM 生成回答
- 默认 `--no-skip-existing` 只跳过已处理过的 source（问题组），但**每个 source 都会重新为所有访谈生成回答**
- 已有 `res/` 目录中的访谈 CSV 虽然会做 `merged_dict` 合并去重，但**LLM 调用仍在所有访谈上发生**

### 步骤4: `batch_generate_codebook.py` — ❌ 需改进
- `--skip-existing` 通过 `question_id` 跳过已有 codebook 条目的问题
- 问题是：当新增访谈后，**已有 codebook 条目的问题需要重新分析（加入新访谈的回答）**，而 `--skip-existing` 会错误地跳过它们
- 若不使用 `--skip-existing`，则所有 480+ 个问题都会被重新调用 LLM，浪费大量 tokens

---

## 推荐的增量更新方案

### 核心策略：只处理新增访谈，合并结果

```
新增访谈.vtt  →  只需为这些新访谈运行
                 步骤3（问答）和步骤4（codebook 合并）
```

### 具体实施方案

#### 改进步骤3：`batch_group_ask_protocol.py` — 只处理新访谈

**修改思路：**
1. 运行前检查 `res/` 目录中已有的 CSV 文件，解析出**已处理的访谈列表**
2. 与 `interview_records.json` 对比，找出**新增的访谈**
3. 只对新增的访谈调用 LLM（通过 `ask_group()` 或新增的 `ask_group_for_interviews()` 方法）
4. 将新结果合并写入已有的 `res/` CSV

**关键代码修改示例：**

```python
# 在 batch_group_ask_protocol.py 的 main() 开头添加
import re

def get_existing_interviews(res_dir: Path) -> set[str]:
    """从 res/ 目录的 CSV 文件名中提取已处理的访谈 ID"""
    existing = set()
    for csv_path in res_dir.glob("*.csv"):
        # 文件名格式: "访谈N-访谈N_访谈记录.csv" -> 提取 "访谈N"
        match = re.match(r'(访谈\d+(?:[_-]\d+)?(?:[_-][^_]+)?)', csv_path.stem)
        if match:
            existing.add(match.group(1))
    return existing

def get_new_interviews(records_path: Path, existing_interviews: set[str]) -> list[str]:
    """从 interview_records.json 找出新增访谈"""
    with open(records_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    # records 的 key 就是 interview_id
    return [lid for lid in records if lid not in existing_interviews]
```

**然后修改 `qa.ask_group()` 的调用，或新增 `qa.ask_group_for_interviews()` 方法：**

核心思路：在 `VideoKnowledgeQA` 中添加一个参数 `interview_ids: Optional[list[str]] = None`，指定只对特定访谈生成回答：

```python
def ask_group(
    self,
    questions: list[dict[str, str]],
    source: str = "",
    interview_ids: Optional[list[str]] = None,  # 新增
    ...
) -> dict[str, Any]:
    ...
    # 在处理 interview_results 时，只处理指定的 interview_ids
    for live_id, meta in sorted(interview_meta.items(), ...):
        if interview_ids is not None and live_id not in interview_ids:
            continue  # 跳过不在白名单中的访谈
        ...
```

#### 改进步骤4：`batch_generate_codebook.py` — 增量合并

**修改思路：**
1. 检查 `codebook/codebook.json` 中已有的各问题 codebook 条目
2. 只针对**有新访谈回答的问题**重新调用 LLM
3. 重新分析时，传入**全部回答（旧回答+新回答）**，确保 code_set 覆盖所有访谈
4. 这样可以保证 codebook 一致性，同时只对有变化的问答

```python
# 在 batch_generate_codebook.py 中
def get_questions_with_new_answers(
    csv_dir: Path,
    existing_entries: dict[str, dict],
    logger
) -> list[str]:
    """找出有新访谈回答的问题（与已有 codebook 中的回答数对比）"""
    # 加载当前所有 CSV 的最新回答
    answers_by_q, _, _ = load_all_csvs(csv_dir, logger)
    
    # 对比现有 codebook 条目中的 interview_answers 数量
    questions_to_update = []
    for qid, all_answers in answers_by_q.items():
        existing = existing_entries.get(qid)
        if existing is None:
            questions_to_update.append(qid)  # 全新问题
        else:
            existing_count = len(existing.get("interview_answers", {}))
            current_count = len(all_answers)
            if current_count > existing_count:
                questions_to_update.append(qid)  # 有新回答
    
    return questions_to_update
```

---

## 完整操作流程（新增访谈时）

### 方法一：手动执行（最灵活，推荐）

```bash
cd /mnt/zhitainew/ttt/interview_transcript/transcript_analyze

# 步骤1: 更新记录文件
python generate_records.py

# 步骤2: 更新知识库（已有增量支持，自动只处理新片段）
python run_kb_qa.py build

# 步骤3: 增量问答——只处理新访谈（需要修改脚本）
python batch_group_ask_protocol.py --incremental  # 新增 --incremental 参数

# 步骤4: 增量 codebook——只重跑有新回答的问题（需要修改脚本）
python batch_generate_codebook.py --incremental  # 新增 --incremental 参数
```

### 方法二：完全重跑（简单但浪费 tokens）

如果新增访谈不多，且希望完全不需要改代码：

```bash
# 步骤1-2 不变
python generate_records.py
python run_kb_qa.py build

# 步骤3: 利用 --no-skip-existing 处理所有（会浪费 tokens 重跑已有访谈）
python batch_group_ask_protocol.py --no-skip-existing

# 步骤4: 不使用 --skip-existing，全部重跑（最浪费）
python batch_generate_codebook.py
```

### 方法三：写一个一键增量更新脚本

将以上逻辑封装为一个 Python 脚本 `incremental_update.py`，自动完成所有检测和增量处理。

---

## 各方案的成本对比

| 方案 | LLM 调用次数 | 优点 | 缺点 |
|:---|:---:|:---|:---|
| **完全重跑** | (问题组数×访谈数) + 问题数 | 无需改代码 | 大量浪费 tokens |
| **方法一：增量** | 新增问题组×新增访谈 + 新增问题 | 只处理新增，无浪费 | 需要修改 2 个脚本 |
| **方法三：一键脚本** | 同上 | 自动化，一键完成 | 需要开发脚本 |

---

## 建议的优先级

1. **立即可以做**：保持现有脚本不变，新增访谈后用 `--no-skip-existing` + 无 `--skip-existing` 全量重跑（浪费 tokens 但正确）
2. **短期优化**：修改 `batch_group_ask_protocol.py` 和 `batch_generate_codebook.py` 增加 `--incremental` 模式
3. **长期优化**：编写 `incremental_update.py` 一键增量更新脚本

---

## 关于知识库（步骤2）的额外说明

当前 `run_kb_qa.py build` 的 `SegmentStore.upsert_many()` 已经通过 `segment_id`（包含 `live_id`+`source_type`+`start_time`+`text` 的 SHA1 哈希）来判断是否为新片段。这意味着：

- 如果新访谈的 VTT 文件是全新的（新的 `live_id`），**所有片段都会被视为新增** ✅
- 如果同名文件被替换（相同的 `live_id` 但文本不同），**旧片段会被覆盖** ✅
- **无需任何修改**，直接重跑即可
