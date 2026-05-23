# ask 任务完整工作流及LLM参与环节

## 整体流程概览

```
ask(question)
  │
  ├─ Phase A: 检索（串行，但改写查询可后续优化为并行）
  │   ├─ a. 加载知识库描述
  │   │     [无LLM调用]
  │   │
  │   ├─ b. 向量查询改写 (_refine_vector_query)
  │   │     [LLM调用 #1]
  │   │
  │   ├─ c. 向量索引检索 (vector.retrieve)
  │   │     [无LLM调用] 用改写后的查询做语义搜索
  │   │
  │   ├─ d. BM25查询改写 (_refine_bm25_query)
  │   │     [LLM调用 #2]
  │   │
  │   ├─ e. BM25索引检索 (bm25.retrieve)
  │   │     [无LLM调用] 用关键词做词频搜索
  │   │
  │   ├─ f. 合并向量+BM25结果
  │   │     [无LLM调用] 向量全保留，BM25补充
  │   │
  │   └─ g. 上下文扩展 (expand_context)
  │         [无LLM调用] 给基段加同场前后片段
  │
  ├─ [跳过] 候选片段有用性分析
  │     [已被跳过，方案C]
  │
  ├─ Phase B: 分组合成（并发 ⚡）
  │   │   ThreadPoolExecutor(max_workers=10)
  │   │   各视频分组互不依赖，并发处理
  │   │
  │   ├─ 视频组1 ──→ _process_video_group()
  │   │                ├─ batch1: [LLM调用 #3] 
  │   │                ├─ batch2: [LLM调用 #4] (如有)
  │   │                └─ ... (含 citation 校验重试)
  │   ├─ 视频组2 ──→ _process_video_group()
  │   ├─ ...
  │   └─ 视频组N ──→ _process_video_group()
  │
  ├─ Phase C: 串行后处理（全局引用编码）
  │   │   [无LLM调用] 按原始时间顺序遍历各组结果，
  │   │   分配全局连续的 citation 编号
  │
  ├─ Phase D: 最终答案合并
  │     [LLM调用 #N+1] 仅当多个视频分组都有答案时触发
  │
  └─ Phase E: 内容安全审核 (check_content_safety)
        [LLM调用 #N+2] 对最终答案做四级风险判定
  
  LLM总调用次数 ≈ 2（改写）+ 视频分组数 + 0~1（合并）+ 1（安全审核）
  并发阶段：ThreadPoolExecutor(max_workers=10)，组间并发

  max_tokens 策略：
  - 查询改写：50（简短JSON，配合prompt“控制在50字以内”）
  - 分组合成/合并：不限制（需输出完整 answer + evidence）
  - 安全审核/KB描述：300（简单JSON）
  思考模式：关闭 `extra_body={"thinking": {"type": "disabled"}}`
```

---

## 各环节详解

### 环节0：加载知识库描述
- **函数**：`_load_kb_description()` / `_build_kb_background_text()`
- **LLM**：❌ 无
- **输入**：`video_knowledge_db/kb_description.txt`（前期 build 时由 LLM 生成并存为文件）
- **输出**：一段文本描述，如 `"包含SNH48-陈嘉仪107场直播的主播讲话（71.7%）和观众弹幕（28.3%）共292700条片段。"`
- **存档位置**：仅作为 prompt 的一部分嵌入后续 LLM 调用的 prompt 中，不单独存档

---

### 环节1：向量查询改写
- **函数**：`_refine_vector_query(question)`
- **LLM**：✅ 是
- **调用**：`_call_llm_json(_build_vector_refinement_prompt(...), "向量查询改写")`
- **输入**：用户原始问题（如 "陈嘉仪是什么性别"）
- **输出**：改写后的查询（如 "这个人的性别是什么？"）
- **存档状态**：✅ `retrieval.vector_query` + `retrieval.vector_refinement`（含 prompt/response/tokens）
- **作用**：弱化高频词（如主播名），让向量搜索更关注语义区分度

---

### 环节2：向量索引检索
- **函数**：`vector.retrieve(vector_query, top_k=1000)`
- **LLM**：❌ 无
- **输入**：改写后的查询文本
- **输出**：`(segment_ids[], scores[])` — 1000 条候选
- **后处理**：按 `score_threshold=0.31` 过滤
- **存档状态**：
  - `retrieval.vector_hits_raw` ✅ — 原始命中数
  - `retrieval.vector_hits_filtered` ✅ — 过滤后数
  - `retrieval.vector_score_threshold` ✅
  - `retrieval.vector_hits_all[]` ✅ — 全部命中的 top-N（含分数和片段摘要）

---

### 环节3：BM25 查询改写
- **函数**：`_refine_bm25_query(question)`
- **LLM**：✅ 是
- **调用**：`_call_llm_json(_build_bm25_refinement_prompt(...), "BM25 查询改写")`
- **输入**：用户原始问题
- **输出**：关键词（如 "性别"、"祖籍"）
- **存档状态**：✅ `retrieval.bm25_query` + `retrieval.bm25_refinement`

---

### 环节4：BM25 索引检索
- **函数**：`bm25.retrieve(bm25_query, top_k=1000)`
- **LLM**：❌ 无
- **输入**：改写后的关键词
- **输出**：`(segment_ids[], scores[])`
- **存档状态**：
  - `retrieval.bm25_hits_raw` ✅
  - `retrieval.bm25_hits_filtered` ✅
  - `retrieval.bm25_score_threshold` ✅
  - `retrieval.bm25_hits_all[]` ✅ — 全部命中 top-N（含分数和片段摘要）

---

### 环节5：合并结果
- **函数**：`retrieve() 内部合并逻辑`
- **LLM**：❌ 无
- **策略**：向量结果全保留 + BM25 适量补充
  - 向量有结果时 BM25 最多补 30 条
  - 向量无结果时 BM25 最多补 80 条
- **存档状态**：
  - `retrieval.raw_merged_ids` ✅ — 合并前唯一ID数
  - `retrieval.used_base_ids` ✅ — 合并后使用的基段数
  - `retrieval.merged_ids_set[]` ✅ — 所有基段 ID 列表
  - `retrieval.merged_dict_scores{}` ✅ — 每条基段的分数来源明细

---

### 环节6：上下文扩展
- **函数**：`store.expand_context(merged_ids, context_window=10)`
- **LLM**：❌ 无
- **操作**：对每个基段，取同一直播同一来源的前后 N 条片段
- **存档状态**：
  - `retrieval.context_window` ✅
  - `retrieval.truncated` ✅（当前恒为 false）
  - `retrieval_segments[]` ✅ — 所有扩展后的候选片段，每条标注了 `source_type`（"基段"/"上下文扩展"）以及检索分数（`vector_score`/`bm25_score`）

---

### 环节7：分组合成（核心 LLM 环节）⚡
- **函数**：`_process_video_group()` + `ThreadPoolExecutor(max_workers=10)`
- **LLM**：✅ 是（可能多次）
- **执行模式**：**组间并发**（所有视频分组通过 `ThreadPoolExecutor` 同时提交），**组内串行**（每个视频分组内部的批处理 + citation 校验重试保持顺序）
- **触发条件**：每个直播视频的候选片段，每 500 条为一批，每批一次 LLM 调用
- **进度日志**：`[1/20] 处理直播 1213958151848398848（想泥萌）...`（并发下日志交错输出）
- **输入**：
  - 用户问题
  - 一批候选片段（按视频内时间排序，简单列出，不加局部上下文）
  - 视频元信息（标题、时间、主播）
- **输出（局部结果）**：
  ```
  {
    "meta": {...},               # 视频元信息
    "batches": [                 # 每个成功批次的局部结果
      {
        "answer": str,           # answer（含原始 LLM 分配的 citation 编号）
        "citations": list,       # citations（含原始 LLM 分配的编号）
        "evidence": list,        # 原始 evidence
        "useful_segment_count": int,
        "batch_count": int,
      },
    ],
    "llm_calls": [...],          # 本组所有 LLM 调用元数据
  }
  ```
- **并发控制**：`max_workers=10`，DeepSeek v4-flash 支持 2500 并发，10 路并发远低于限值，无需担心 429
- **注意**：并行阶段**不做全局引用编码**，全局编号在 Phase C 串行后处理中完成

---

### 环节8：最终答案合并
- **函数**：`ask()` 内的 merge 逻辑
- **LLM**：✅ 是（仅当多个视频分组都有答案时触发；仅一个分组有答案时直接使用）
- **触发条件**：`len(answer_videos) > 1`

#### Phase C：串行后处理（全局引用编码）
在并发合成的所有视频组都完成后，按**原始视频时间顺序**依次处理每组结果：

```
Step 1: 遍历各组（按时间顺序）
Step 2: 对每组每批，构建局部→全局编号映射
        local_to_global = { "#1" → "#5", "#2" → "#6", ... }
        其中全局偏移量 = 已处理的 evidence 总数
Step 3: 替换 answer 中的局部编号为全局编号
Step 4: 更新 citations 中的 citation_id 为全局编号
Step 5: 累加 evidence 到全局列表，更新偏移量
Step 6: 构建 video_results（格式与改造前完全一致）
```

#### Phase D：合并 LLM
```
Step 7: 从所有分组的 global citations 去重构建最终 citations 表
Step 8: 构建每个分组的完整答案 + 引用列表（最终编号），输入给 merge LLM
Step 9: LLM 合并各分组答案，去重后输出一条连贯答案
Step 10: 如果引用格式不通过，最多重试 5 次修正
```
- **回退策略**：合并 LLM 失败时回退到 `\n\n---\n\n` 简单拼接
- **存档状态**：
  - `synthesis_llm_calls` 中新增一次 merge LLM 调用元数据
  - `answer` — 合并后的最终答案
  - `citations` — 全局统一编号的引用列表

---

### 环节9：内容安全审核（新增）
- **函数**：`_check_content_safety(question, answer, citations)`
- **LLM**：✅ 是
- **触发条件**：始终执行（在最终答案生成之后）
- **风险等级**：四级风险（SAFE/LOW/MEDIUM/HIGH），仅拦截 MEDIUM 和 HIGH
- **回退策略**：审核调用失败时默认判为 MEDIUM（保守拦截）

---

## LLM推理与回答要求满足情况

> 对照 `docs/llm_reasoning_requirements.md` 逐项检查

### 一、推理层面

| 要求 | 分组阶段 (环节7) | 合并阶段 (环节8) | 说明 |
|---|---|---|---|
| 1. 充分利用背景常识 | ✅ Prompt含"背景常识"推理原则 | ✅ Merge prompt含"粉丝立场" | 分组和合并两阶段都保留此要求 |
| 2. 检测常识矛盾 | ✅ Prompt含"明显矛盾可能是转述/玩梗"通用原则 | ❌ **未包含** | Merge阶段只做文本合并，不改推理，但保留该要求可以让LLM在合并时重新审视矛盾内容 |
| 3. 结合上下文推测真实含义 | ✅ Prompt含"结合同一直播相邻片段上下文" | ❌ **未包含** | 同上，merge时可能丢失这段约束 |

**差距**：merge prompt 缺少"常识矛盾检测"和"结合上下文推测"的要求。不过 merge 阶段的任务是合并已推理好的答案（引用的片段摘要也在输入中），矛盾检测已在分组阶段完成。如果要最大化稳健性，可以在 merge prompt 中添加通用原则。

### 二、立场层面

| 要求 | 分组阶段 | 合并阶段 | 说明 |
|---|---|---|---|
| 1. 粉丝立场 | ✅ "你是一名熟知这名主播的粉丝" | ✅ 已更新为"你是一名熟知这名主播的粉丝" | 一致 |
| 2. 避免负面形象 | ✅ "避免对主播造成不当误导或负面形象" | ✅ 已添加"避免对主播造成不当误导或负面形象" | 一致 |

### 三、Prompt 设计层面

| 要求 | 分组阶段 | 合并阶段 | 说明 |
|---|---|---|---|
| 1. 避免具体例子 | ✅ 使用"例如一个公开身份为女性的人说'我是男生'"作为示例 | ✅ 无具体例子 | 符合要求 |
| 2. 使用通用原则 | ✅ 已改为通用原则 | ✅ 使用通用原则 | 符合 |

### 四、回答风格层面

| 要求 | 分组阶段 | 合并阶段 | 说明 |
|---|---|---|---|
| 1. 自然亲切口吻 | ✅ "用自然、亲切的口吻回答，就像在跟朋友介绍一样" | ✅ "以自然、亲切的口吻回答，就像在跟朋友介绍一样" | 一致 |
| 2. 引用标记 [#N] | ✅ "在引用证据时插入 [#N] 标记" | ✅ "引用编号已在输入中给出，直接沿用即可" | 输入层保证编号正确性 |
| 3. 引用精简原则 | ✅ "evidence仅包含那些确实为答案提供了独特信息的片段" | ✅ "引用精简：每个引用应为回答提供独特的、补充性的信息。如果多个分组引用了同一片段或提供重复信息，只保留最具代表性的几个" | 一致 |
| 4. 清晰的判断说明 | ✅ "写明判断依据"（在 evidence.reason 中） | ❌ **未要求** | Merge阶段只输出 answer，不输出 evidence |

**差距**：merge prompt 不输出 evidence（只输出 answer），因此"清晰的判断说明"在 merge 阶段不适用。分组阶段每个 evidence 的 reason 字段已在 `video_results` 中保留。

### 五、元数据利用

| 要求 | 分组阶段 | 合并阶段 | 说明 |
|---|---|---|---|
| 利用片段元数据 | ✅ Prompt含"用户名中的前缀格式也是有效证据信息" | ❌ **未包含** | Merge阶段输入已包含每个分组引用的引用片段详情（含元数据），但未在 prompt 中强调 |

---

### 总结：差距与改进建议

| 未满足项 | 严重程度 | 建议 |
|---|---|---|
| Merge阶段缺少"常识矛盾检测" | 🔷 低 — 已在分组阶段完成 | 可在 merge prompt 中添加通用原则作为兜底 |
| Merge阶段缺少"结合上下文推测" | 🔷 低 — 分组阶段已完成 | 同上 |
| Merge阶段不要求输出 evidence 的 reason | 🔵 设计如此 — 无需修改 | merge 只合并 answer，evidence 已在分组阶段保留 |

**结论**：当前 pipeline 基本满足所有要求。merge prompt 更新后已补充了"粉丝立场""引用精简""避免负面形象"，唯一未覆盖的是"常识矛盾检测"和"结合上下文推测"——但这属于低风险项，因为分组阶段已完成推理。

---

### 六、内容安全审核（新增）

| 要求 | 审核阶段 (环节9) | 说明 |
|---|---|---|
| 四级风险分类 | ✅ SAFE/LOW/MEDIUM/HIGH | 取代旧版二值判定 |
| 拦截策略 | ✅ 仅拦截 MEDIUM 和 HIGH | LOW 和 SAFE 放行 |
| 判断指南 | ✅ 含5条详细指南 | 区分恶意攻击/日常表达、粉丝文化/真实负面、转述/自述、字面负面/真实意图、否定澄清类内容 |
| 失败处理 | ✅ 默认判为 MEDIUM | 保守拦截，避免漏判 |
| 存档状态 | ✅ content_safety 字段保留原始答案和引用 | 方便事后复审 |

---

## 存档文件完整结构（当前）

```json
{
  // ── 0. 元信息 ──
  "question": "陈嘉仪是什么性别",
  "created_at": "2026-05-14T18:13:17",

  // ── 1. 检索阶段 ──
  "retrieval": {
    "vector_query": "这个人的性别是什么？",
    "vector_refinement": { "prompt": "...", "response": "...", "tokens": {...} },
    "vector_hits_raw": 1000,
    "vector_hits_filtered": 12,
    "vector_hits_all": [ {"rank": 1, "segment_id": "...", "score": 0.399, ...}, ... ],
    "bm25_query": "性别",
    "bm25_refinement": { "prompt": "...", "response": "...", "tokens": {...} },
    "bm25_hits_raw": 1000,
    "bm25_hits_filtered": 14,
    "bm25_hits_all": [ {"rank": 1, "segment_id": "...", "score": 14.539, ...}, ... ],
    "raw_merged_ids": 18,
    "used_base_ids": 18,
    "merged_ids_set": [...],
    "merged_dict_scores": { "segment_id": {"vector_score": ..., "bm25_score": ..., "source": "..."} },
    "candidate_count": 300,
    "context_window": 10,
    "max_base_segments": 200,
    "max_expanded_segments": null,
    "truncated": false
  },

  // ── 2. 扩展后的候选片段列表 ──
  "retrieval_segments": [
    {
      "segment_id": "...",
      "source_label": "主播讲话",
      "video_title": "...",
      "anchor_name": "SNH48-陈嘉仪",
      "video_offset": "00:18:14",
      "absolute_time": "2025-10-20T17:34:19",
      "text": "...",
      "source_type": "上下文扩展",       // "基段" 或 "上下文扩展"
      "vector_score": null,
      "bm25_score": null
    },
    ...
  ],

  // ── 3. 分析摘要 ──
  "analysis_summary": {"skipped": true, ...},

  // ── 4. 分析结果 ──
  "analysis": [...],

  // ── 5. 有用片段列表 ──
  "useful_segments": [...],

  // ── 6. 按直播分组合成结果 ──
  "video_results": [
    {
      "live_id": "...",
      "video_title": "...",
      "anchor_name": "SNH48-陈嘉仪",
      "video_datetime": "...",
      "answer": "根据片段中的内容...",
      "citations": [{"citation_id": "#1", "segment_id": "...", "reason": "..."}],
      "useful_segment_count": 21,
      "batch_count": 1
    },
    ...
  ],

  // ── 7. 所有LLM调用的元数据 ──
  "llm_calls": {
    "analysis_batches": [],
    "synthesis": {
      "description": "per_video_batch_synthesis",
      "per_video_batch_size": 500,
      "video_count": 20,
      "total_calls": 21,  // 20个视频分组 + 1次最终合并
      "calls": [
        { "prompt": "...", "response": "...", "input_tokens": 2763, ... },
        ...
      ]
    }
  },

  // ── 8. 最终答案 ──
  "answer": "陈嘉仪是SNH48的成员，从片段中可以看到...",
  "citations": [...],
  "retrieved_count": 629,
  "useful_segment_count": 629
}
```

---

## log 的优势 vs 存档JSON

存档 JSON 已经包含了调试所需的绝大部分信息，但 log 仍有以下独特优势：

| 信息类型 | 存档JSON | log | 说明 |
|---|---|---|---|
| 向量改写+检索的实时日志 | ✅ 有改写结果 | ✅ 有改写过程 | 重叠 |
| BM25改写+检索的实时日志 | ✅ 同上 | ✅ 同上 | 重叠 |
| **LLM调用耗时** | ❌ 无 | ✅ 可通过前后时间戳推算 | 存档只有最终时间戳，无每个LLM调用的耗时 |
| **向量检索top-200快速浏览** | ✅ 有全部命中 | ✅ 有前200条 | 存档更全（所有命中），log可快速查看 |
| **BM25检索top-200快速浏览** | ✅ 有全部命中 | ✅ 有前200条 | 同上 |
| **扩展后片段排序前30示例** | ❌ 无 | ✅ log中有 | 存档JSON中没有扩展后的排序示例 |
| **各live_id序列长度** | ❌ 无 | ✅ log中有 | 用于调试上下文扩展覆盖范围 |
| **合并策略详细描述** | ✅ 有统计信息 | ✅ 有策略说明 | 存档更详细（具体条数） |
| **"川渝娘"等具体片段追踪** | ✅ 能在citations中找到 | ✅ 实时显示 | 存档更适合事后分析，log更适合实时debug |
| **实时进度（当前处理到第几个视频）** | ❌ 不适用 | ✅ log中有 `[1/20]` 等 | 存档是最终结果，log是实时过程 |
| **LLM调用token消耗** | ✅ 有具体数值 | ✅ 有具体数值 | 两者均有 |

**总结**：如果你需要 **事后深入分析某次执行**（比如查为什么某条片段没被检索到），存档 JSON 已经足够——它包含了按环节组织的数据、所有 LLM 调用记录、检索超清单、基段/扩展段标记等。
如果你需要 **实时监控执行进度** 或 **快速定位问题**（比如看某次改写用了什么关键词），log 会更方便——按时间线顺序呈现，不需要打开 JSON 文件翻阅。

---

## Debug 建议（当前状态）

| 原缺少项 | 当前状态 | 说明 |
|---|---|---|
| 向量查询改写的 LLM 元数据 | ✅ | `retrieval.vector_refinement` |
| 向量检索超清单 | ✅ | `retrieval.vector_hits_all` |
| BM25 检索超清单 | ✅ | `retrieval.bm25_hits_all` |
| 基段 vs 扩展段标记 | ✅ | `retrieval_segments[].source_type` |
| 候选片段检索分数 | ✅ | `retrieval_segments[].vector_score` / `bm25_score` |
| 按处理环节顺序存档 | ✅ | 存档 JSON 按 0~8 环节组织 |
| 多个分组答案合并策略 | ✅ | 新增 merge LLM 调用 + 引用序号映射 |
| 处理进度日志 | ✅ | `[1/20]` 格式的 log |
| 截断已移除 | ✅ | `max_expanded_segments=None`，截断代码已删除 |

**结论**：存档文件 + log 均可独立还原一次 ask 执行过程的全部信息。
