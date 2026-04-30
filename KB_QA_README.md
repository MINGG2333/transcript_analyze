# 视频知识库问答系统

## 1) 安装依赖

```bash
python -m pip install -r requirements_kb_qa.txt
```

## 2) 构建知识库（可重复执行，支持增量）

```bash
python run_kb_qa.py build ^
  --records download_records.json ^
  --subtitle-root firered_output_batch ^
  --kb-dir video_knowledge_db
```

## 3) 进行问答

```bash
python run_kb_qa.py ask ^
  --question "我想知道视频中的人在哪些时候提到了养的宠物狗顺顺，请按时间顺序列出" ^
  --records download_records.json ^
  --subtitle-root firered_output_batch ^
  --kb-dir video_knowledge_db ^
  --llm-model gpt-4o-mini
```

## 4) 环境变量

- `OPENAI_API_KEY`：LLM API Key
- `OPENAI_BASE_URL`：兼容 OpenAI 接口的自定义地址（可选）

## 5) 输出说明

- `video_knowledge_db/chroma_db`：向量数据库
- `video_knowledge_db/segment_store.json`：完整片段及上下文索引
- `video_knowledge_db/qa_archive/*.json`：每次问答归档（问题、答案、引用列表）

