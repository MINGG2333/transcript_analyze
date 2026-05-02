# 视频知识库问答系统

## 1) 安装依赖

```bash
python -m pip install -r requirements_kb_qa.txt
```

## 2) 构建知识库（可重复执行，支持增量）

```bash
python run_kb_qa.py \
  --records download_records.json \
  --subtitle-root firered_output_batch \
  --kb-dir video_knowledge_db \
  build
```
```bash
# for the first run
~/miniconda3/envs/koudai48/lib/python3.9/site-packages/requests/__init__.py:86: RequestsDependencyWarning: Unable to find acceptable character detection dependency (chardet or charset_normalizer).
  warnings.warn(
modules.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:00<00:00, 37.2kB/s]
README.md: 13.7kB [00:00, 18.2MB/s]
sentence_bert_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54.0/54.0 [00:00<00:00, 10.7kB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 856/856 [00:00<00:00, 1.50MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 409M/409M [00:26<00:00, 15.7MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 319/319 [00:00<00:00, 102kB/s]
vocab.txt: 110kB [00:00, 8.04MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 94.2kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 74.0/74.0 [00:00<00:00, 52.8kB/s]
{
  "parsed_segments": 209990,
  "updated_segments": 209990,
  "total_segments": 209990
}
```

## 3) 进行问答

```bash
python run_kb_qa.py \
  --records download_records.json \
  --subtitle-root firered_output_batch \
  --kb-dir video_knowledge_db \
  ask \
  --question "我想知道视频中的人在哪些时候提到了养的宠物狗顺顺，请按时间顺序列出"
```

## 4) 环境变量

- `OPENAI_API_KEY`：LLM API Key
- `OPENAI_BASE_URL`：兼容 OpenAI 接口的自定义地址（可选）

## 5) 输出说明

- `video_knowledge_db/chroma_db`：向量数据库
- `video_knowledge_db/segment_store.json`：完整片段及上下文索引
- `video_knowledge_db/qa_archive/*.json`：每次问答归档（问题、答案、引用列表）

