# 2026-05-13 内存优化

## 背景

服务器仅 1.9GB 内存且无 swap，运行 KB_QA 系统时频繁被 OOM Killer 杀死进程。

**服务器现状：**
- 物理内存：1.9GB
- Swap：0
- 数据库量：107 个视频，21万条片段
- segment_store.json：265MB
- chroma_db 向量库：2.8GB
- embedding 模型：shibing624/text2vec-base-chinese（~500MB）

---

## 根因

| 内存消耗项 | 约占用 |
|-----------|-------|
| OS + 其他进程 | 1.1GB |
| segment_store.json 加载 | ~300MB |
| embedding 模型加载 | ~500MB |
| ChromaDB 内部开销 | ~200MB |
| build: 20万条解析结果清单 | ~200MB |
| Python 自身 | ~100MB |
| **合计需求** | **~2.4GB > 1.9GB → OOM Kill** |

---

## 修改的文件

### 1. `kb_qa/models.py` — Segment 类添加 `__slots__`

- 为 Segment 数据类添加 `__slots__` 声明
- 每个对象节省约 40% 内存（21万段预期节省 80~100MB）
- 依赖 Python 3.10+ 的 `@dataclass` + `__slots__` 兼容支持

### 2. `kb_qa/indexer.py` — VectorIndex 逐批回收内存

- 添加 `import gc`
- `VectorIndex.upsert()` 每批处理后执行 `del batch + gc.collect()`
- 防止 embedding 向量化过程中 batch 对象在内存中累积

### 3. `kb_qa/qa.py` — 多个内存优化

- **BM25 惰性加载**：`self.bm25` → `self._bm25`，通过 `get_bm25()` 惰性获取
  - `__init__` 不再立即构建 BM25 索引（减少了初始化内存）
  - BM25 仅在 `retrieve()` 首次调用需要时才构建
- **build 流程内存释放**：
  - 向量化之前释放解析结果：`del all_segments + gc.collect()`
  - 修复 `len(all_segments)` 提前删除导致的引用问题，改用 `parsed_count` 变量保存
  - BM25 重建后再执行一次 `gc.collect()`
- 添加 `import gc`

### 4. `kb_qa/cli.py` — 降低默认检索参数

- `--vector-top-k` 默认值：10000 → **1000**（减少 90% 向量检索内存）
- `--bm25-top-k` 默认值：10000 → **1000**
- 更新 help 文本提示低内存服务器建议 500~1000

---

## 建议服务器操作

### 1. 创建 2GB swap 文件（最有效）
```bash
# 创建 2GB swap 文件
dd if=/dev/zero of=/swapfile bs=1M count=2048 status=progress
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
# 验证是否生效
free -h
swapon --show
# 持久化到 /etc/fstab（可选，重启后也生效）
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
```

### 2. 测试命令
```bash
# 测试 ask（正常情况应能运行）
python run_kb_qa.py --debug ask --question "陈嘉仪和北舞的关联是什么？"

# 测试 build
python run_kb_qa.py --debug build
```

### 3. 如果 build 仍被 Kill
```bash
# 清理旧的 chroma_db 减少碎片
rm -rf video_knowledge_db/chroma_db/
python run_kb_qa.py --debug build
```
