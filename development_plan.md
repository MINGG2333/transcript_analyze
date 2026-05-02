我们来实现一个完整的 **视频片段智能问答系统**，它能够从数百个直播录屏的弹幕和字幕中，根据用户提问进行语义检索、上下文扩展，并由LLM生成带引用的答案，所有数据全部可溯源。

---

## 系统架构概览

1. **预处理**：解析 `download_records.json`、`firered_output_batch` 下的 SRT 字幕与 LRC 弹幕，结合元数据 JSON 提取真实直播时间，统一为 `TextSegment` 对象。
2. **构建向量索引**：将所有片段嵌入后存入 **Chroma**，同时构建一个内存片段索引（用于上下文扩展）并持久化。
3. **检索 + 上下文扩展**：对于每个问题，先用 Chroma 向量检索召回 Top‑K 片段，再根据每个命中的片段，从索引中取前后各 N 条相邻字幕/弹幕，形成上下文窗口。(TBD: 对齐弹幕和视频/字幕的时间戳, 建立包括弹幕、字幕的综合上下文)
4. **LLM 推理**：将扩展后的候选片段（按直播时间排序）和精心设计的 Prompt 发送给 LLM，要求它：
   - 理解语义，找出所有真正相关的片段（不仅依赖关键词）。
   - 按时间顺序整理。
   - 生成引用列表。
   - 输出带有引用标号的最终答案。
5. **存档**：将每次问答的原始问题、最终答案、引用列表以及检索到的片段全量保存为 JSON 文件。

---

## 环境准备

```bash
pip install chromadb sentence-transformers openai rank_bm25 tqdm
```

推荐嵌入模型：`shing624/text2vec-base-chinese`（轻量、效果好）。
LLM 可用 OpenAI GPT‑4 或任何兼容 API 的模型。

---

## 第一步：定义统一数据结构

```python
# schema.py
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

@dataclass
class TextSegment:
    text: str
    start_time: float       # 片段开始秒数（视频内偏移）
    end_time: float         # 片段结束秒数
    source_type: str        # "speech" 或 "danmaku"

    # 溯源信息
    file_path: str          # 该片段所在文件的绝对路径（srt 或 lrc）
    video_path: str         # 原始视频 TS 文件路径
    video_title: str
    anchor_name: str
    live_id: str
    video_datetime: datetime # 真实直播发生时间（从元数据/文件名提取）

    def unique_id(self) -> str:
        return f"{self.live_id}_{self.start_time}_{self.source_type}"
```

---

## 第二步：解析原始数据，生成所有 TextSegment

### 1. 提取视频发生时间

```python
def extract_video_datetime(metadata_path: str, filename: str) -> datetime:
    """优先使用 info.json 中的 ctime，否则从文件名解析"""
    try:
        if Path(metadata_path).exists():
            with open(metadata_path, encoding='utf-8') as f:
                meta = json.load(f)
            ms = int(meta.get('ctime', 0))
            if ms:
                return datetime.fromtimestamp(ms / 1000)
    except Exception:
        pass
    # 回退：从文件名提取 _YYYYMMDDHHmmss
    import re
    match = re.search(r'_(\d{14})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
    return datetime.min
```

### 2. 解析 SRT 字幕

```python
import re

def parse_srt(srt_path: Path, video_meta: dict) -> list[TextSegment]:
    segments = []
    if not srt_path.exists():
        return segments
    with open(srt_path, encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return segments
    for block in content.split('\n\n'):
        lines = block.split('\n')
        if len(lines) < 3:
            continue
        m = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
        if not m:
            continue
        start = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/1000
        end = int(m.group(5))*3600 + int(m.group(6))*60 + int(m.group(7)) + int(m.group(8))/1000
        text = ' '.join(lines[2:]).strip()
        if not text:
            continue
        segments.append(TextSegment(
            text=text,
            start_time=start,
            end_time=end,
            source_type="speech",
            file_path=str(srt_path),
            video_path=video_meta['video_path'],
            video_title=video_meta.get('title', ''),
            anchor_name=video_meta.get('user_name', ''),
            live_id=video_meta['live_id'],
            video_datetime=extract_video_datetime(
                video_meta.get('metadata_path', ''),
                Path(video_meta['video_path']).name
            )
        ))
    return segments
```

### 3. 解析 LRC 弹幕

```python
def parse_lrc(lrc_path: Path, video_meta: dict) -> list[TextSegment]:
    segments = []
    if not lrc_path or not lrc_path.exists():
        return segments
    with open(lrc_path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        # 匹配 [mm:ss.xx] 或 [mm:ss.xxx]
        m = re.match(r'\[(\d{2}):(\d{2}\.\d{2,3})\](.*)', line)
        if m:
            minutes = int(m.group(1))
            seconds = float(m.group(2))
            start = minutes * 60 + seconds
            text = m.group(3).strip()
            if not text:
                continue
            segments.append(TextSegment(
                text=text,
                start_time=start,
                end_time=start + 5.0,  # 弹幕默认持续5秒
                source_type="danmaku",
                file_path=str(lrc_path),
                video_path=video_meta['video_path'],
                video_title=video_meta.get('title', ''),
                anchor_name=video_meta.get('user_name', ''),
                live_id=video_meta['live_id'],
                video_datetime=extract_video_datetime(
                    video_meta.get('metadata_path', ''),
                    Path(video_meta['video_path']).name
                )
            ))
    return segments
```

### 4. 遍历所有视频，收集片段

```python
import json

def load_all_segments(records_file: str = 'download_records.json',
                      output_base: str = 'firered_output_batch') -> list[TextSegment]:
    with open(records_file, encoding='utf-8') as f:
        records = json.load(f)

    all_segments = []
    for live_id, rec in records.items():
        # 构造处理后的字幕路径
        safe_user = regex_safe(rec.get('user_name', 'unknown'))
        safe_video = regex_safe(Path(rec['video_path']).stem)
        srt_path = Path(output_base) / safe_user / safe_video / f"{safe_video}_subtitles.srt"
        lrc_path = Path(rec.get('danmu_path', '')) if rec.get('danmu_path') else None

        # 解析字幕
        all_segments.extend(parse_srt(srt_path, rec))
        # 解析弹幕
        if lrc_path:
            all_segments.extend(parse_lrc(lrc_path, rec))

    return all_segments

def regex_safe(s: str) -> str:
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', s)
```

---

## 第三步：构建向量数据库与片段索引

### 1. 创建 Chroma 集合

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="video_knowledge_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="shing624/text2vec-base-chinese"
)
collection = client.get_or_create_collection(
    name="video_segments",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)
```

### 2. 批量添加片段

```python
from tqdm import tqdm

def add_segments_to_db(segments: list[TextSegment], batch_size=500):
    for i in tqdm(range(0, len(segments), batch_size)):
        batch = segments[i:i+batch_size]
        ids = [seg.unique_id() for seg in batch]
        texts = [seg.text for seg in batch]
        metadatas = [{
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "source_type": seg.source_type,
            "anchor_name": seg.anchor_name,
            "video_title": seg.video_title,
            "live_id": seg.live_id,
            "video_datetime": seg.video_datetime.isoformat(),
            "file_path": seg.file_path,
            "video_path": seg.video_path
        } for seg in batch]
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
```

### 3. 构建全局片段索引（用于上下文扩展）

```python
import bisect

# 全局字典：key = f"{live_id}_{source_type}"，value = TextSegment 列表（按 start_time 排序）
segment_index: dict[str, list[TextSegment]] = {}

def build_memory_index(all_segments: list[TextSegment]):
    index = {}
    for seg in all_segments:
        key = f"{seg.live_id}_{seg.source_type}"
        index.setdefault(key, []).append(seg)
    for k in index:
        index[k].sort(key=lambda s: s.start_time)
    return index

# 保存到磁盘，下次直接加载
import pickle

def save_index(index, path='segment_index.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(index, f)

def load_index(path='segment_index.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
```

---

## 第四步：检索与上下文扩展

### 1. 向量检索

```python
def retrieve_from_db(query: str, top_k=50) -> list[TextSegment]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas']
    )
    hits = []
    for text, meta in zip(results['documents'][0], results['metadatas'][0]):
        seg = TextSegment(
            text=text,
            start_time=meta['start_time'],
            end_time=meta['end_time'],
            source_type=meta['source_type'],
            file_path=meta['file_path'],
            video_path=meta['video_path'],
            video_title=meta['video_title'],
            anchor_name=meta['anchor_name'],
            live_id=meta['live_id'],
            video_datetime=datetime.fromisoformat(meta['video_datetime'])
        )
        hits.append(seg)
    return hits
```

### 2. 上下文扩展

```python
def expand_context(hits: list[TextSegment], context_size=5, index: dict = None) -> list[TextSegment]:
    if index is None:
        index = segment_index
    expanded_ids = set()
    for seg in hits:
        key = f"{seg.live_id}_{seg.source_type}"
        seq = index.get(key, [])
        if not seq:
            expanded_ids.add(seg.unique_id())
            continue
        # 二分查找当前片段在序列中的位置
        pos = bisect.bisect_left([s.start_time for s in seq], seg.start_time)
        start_idx = max(0, pos - context_size)
        end_idx = min(len(seq), pos + context_size + 1)
        for s in seq[start_idx:end_idx]:
            expanded_ids.add(s.unique_id())
    # 去重后按视频时间、内部时间排序
    all_segs = [s for s in hits if s.unique_id() in expanded_ids]
    # 还要从 index 中获取实际对象（因为可能只保留了 id）
    # 更简单的方法：在 expand 时直接收集对象
    # 这里给出改进版：返回实际对象列表
    # 实际实现请参考前面的 expand_context 返回 set of TextSegment
    # 由于我们只存储了 unique_id，这里需要保留对象，所以改良一下：
    pass
```

**优化后的 expand_context**（直接返回对象）：
```python
def expand_context(hits: list[TextSegment], context_size=5) -> list[TextSegment]:
    result = []
    seen = set()
    for seg in hits:
        key = f"{seg.live_id}_{seg.source_type}"
        seq = segment_index.get(key, [])
        if not seq:
            if seg.unique_id() not in seen:
                result.append(seg)
                seen.add(seg.unique_id())
            continue
        pos = bisect.bisect_left([s.start_time for s in seq], seg.start_time)
        start = max(0, pos - context_size)
        end = min(len(seq), pos + context_size + 1)
        for s in seq[start:end]:
            uid = s.unique_id()
            if uid not in seen:
                result.append(s)
                seen.add(uid)
    # 按视频时间、视频内时间排序
    result.sort(key=lambda s: (s.video_datetime, s.start_time))
    return result
```

---

## 第五步：LLM 问答与引用生成

### 1. 构建 Prompt

```python
def build_prompt(question: str, segments: list[TextSegment]) -> str:
    # 将片段格式化为带编号的上下文
    context_lines = []
    for idx, seg in enumerate(segments):
        vid_ts = seg.video_datetime.strftime("%Y-%m-%d %H:%M:%S")  # 直播发生时间
        internal_ts = f"{int(seg.start_time//3600):02d}:{int((seg.start_time%3600)//60):02d}:{int(seg.start_time%60):02d}"
        source_label = "主播" if seg.source_type == "speech" else "弹幕"
        line = (f"[{idx}] {source_label} | 直播时间: {vid_ts} | 视频内: {internal_ts} | "
                f"视频: {seg.video_title} (主播: {seg.anchor_name})\n"
                f"内容: {seg.text}")
        context_lines.append(line)
    context_block = "\n\n".join(context_lines)

    prompt = f"""你是一个视频内容分析助手。你有一系列从直播视频中提取的字幕和弹幕片段，每个片段都带有编号和溯源信息。

用户问题：{question}

所有候选片段如下：
{context_block}

请完成以下任务：
1. 仔细阅读所有片段，找出所有与问题**真正相关**的内容。注意语义理解，不能仅仅依赖关键词。例如，如果问题是关于宠物狗“顺顺”，那么“它把我的鞋叼走了”这样的互动描述也应该视为相关。
2. 将所有相关片段按照**直播发生时间顺序**排列（如果同一直播，再按视频内时间排序）。
3. 为每一个相关片段生成一条引用条目，格式如下（在大括号内换行，使其易于解析，但实际输出请用英文大括号和逗号分隔）：
   - 引用编号：#1, #2...
   - 类型：主播讲话 / 观众弹幕
   - 时间戳：YYYY-MM-DD HH:MM:SS / 视频内 HH:MM:SS
   - 内容：原片段文本
   - 来源文件：{文件路径}
   - 视频：{视频标题} (主播: {主播})
4. 基于这些相关片段，用简洁的语言回答用户问题，并在回答中标注引用编号，例如“...顺顺在直播中多次出现[#1][#3]...”。
5. 最后单独输出“## 引用列表”，按编号列出所有引用条目。

请确保回答完整、清晰，引用列表包含所有相关片段。"""
    return prompt
```

### 2. 调用 LLM

```python
import openai

openai.api_key = 'YOUR_KEY'

def ask_llm(question: str, segments: list[TextSegment], model="gpt-4") -> str:
    prompt = build_prompt(question, segments)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

### 3. 存档

```python
import json, uuid
from datetime import datetime as dt

def save_archive(question: str, answer: str, segments_used: list[TextSegment],
                 archive_dir="qa_archive"):
    Path(archive_dir).mkdir(exist_ok=True)
    rec = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": dt.now().isoformat(),
        "question": question,
        "answer": answer,
        "segments": [
            {
                "text": s.text,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "source_type": s.source_type,
                "file_path": s.file_path,
                "video_path": s.video_path,
                "video_title": s.video_title,
                "anchor": s.anchor_name,
                "live_id": s.live_id,
                "video_datetime": s.video_datetime.isoformat()
            } for s in segments_used
        ]
    }
    with open(f"{archive_dir}/{rec['id']}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"存档完成: {rec['id']}")
```

---

## 第六步：组装主流程

```python
def main():
    # 1. 加载所有片段（如果首次运行，需扫描所有目录）
    all_segments = load_all_segments()  # 可缓存到 pickle，增加判断

    # 2. 构建向量库（增量：只添加新 ID，或重建）
    # 这里简单重建，生产环境可检查 collection.count()
    global collection
    add_segments_to_db(all_segments, batch_size=500)

    # 3. 构建内存索引（用于上下文扩展）
    global segment_index
    segment_index = build_memory_index(all_segments)
    save_index(segment_index)  # 下次启动直接 load_index

    # 4. 问答循环
    while True:
        question = input("请输入问题（输入 exit 退出）: ")
        if question.strip().lower() == 'exit':
            break

        # 4.1 检索 + 扩展
        hits = retrieve_from_db(question, top_k=50)
        expanded = expand_context(hits, context_size=5)

        # 4.2 LLM 回答
        answer = ask_llm(question, expanded)

        # 4.3 显示并保存
        print("\n" + answer + "\n")
        save_archive(question, answer, expanded)

if __name__ == "__main__":
    main()
```

---

## 进阶优化与注意事项

- **增量更新**：扫描 `download_records.json` 中的新视频，只解析尚未入库的片段（通过检查 Chroma 集合的 ID 或本地记录）。
- **性能**：数十万片段，Chroma 检索在毫秒级；上下文扩展若每次读取全量索引（驻留内存）也无压力。
- **混合检索**：如果需要提高召回，可以结合 BM25 全文检索（使用 `rank_bm25`），对 `all_segments` 文本建索引，取 Top‑20，与向量结果合并去重。
- **弹幕清洗**：可过滤纯符号、过短的无意义弹幕，提高信噪比。
- **LLM 上下文长度**：如果扩展后片段太多（如超过模型 token 限制），可先让 LLM 筛选一遍，或截断到合理数量（例如按相关性分数取前30条）。
- **引用解析**：LLM 输出的引用格式可通过正则提取时间戳和文件路径，用于后续点击跳转（可集成到前端）。

这样，你就拥有了一个**完全可溯源的、能理解语义的直播视频问答系统**，无论问题是关于宠物狗、特定事件还是其他话题，它都能给出有序且带证据的答案。