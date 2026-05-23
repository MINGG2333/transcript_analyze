# DeepSeek API 优化指南

> 用于改进基于 DeepSeek API 的网站后端服务（Python，OpenAI 兼容格式）
> 最后更新：2026-05-23
> 参考来源：
> - [DeepSeek API Docs（首页）](https://api-docs.deepseek.com/zh-cn/)
> - [思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode) — 关闭后输出 token 减半
> - [JSON 输出](https://api-docs.deepseek.com/zh-cn/guides/json_mode) — 结构化输出
> - [多轮对话](https://api-docs.deepseek.com/zh-cn/guides/multi_round_chat) — 无状态 API
> - [工具调用](https://api-docs.deepseek.com/zh-cn/guides/tool_calls) — 扩展能力
> - [限速与隔离](https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit) — v4-pro 500 并发
> - [错误码](https://api-docs.deepseek.com/zh-cn/quick_start/error_codes) — 7 种错误码处理
> - [查询余额](https://api-docs.deepseek.com/zh-cn/api/get-user-balance) — API 状态监控
> - [Token 用量计算](https://api-docs.deepseek.com/zh-cn/quick_start/token_usage) — 计费用 tokenizer
> - [OpenAI Chat API 参数（max_tokens 依据）](https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_tokens) — DeepSeek 兼容

---

你的网站目前使用 **DeepSeek API**（OpenAI 兼容格式），以下从**官方文档**出发，整合了所有 API 调用相关的最佳实践。

---

## 一、思考模式开关（影响成本和速度的关键 🔑）

### 默认行为

DeepSeek 模型的思考模式**默认开启**（`thinking: {"type": "enabled"}`），模型在回答前会输出一段思维链（reasoning_content），提升复杂问题的准确性。

### 建议：**默认关闭思考模式**

对于直播转录 Q&A 场景（事实性问答），关闭思考模式可以：

| 维度 | 开启思考模式 | 关闭思考模式 |
|------|-------------|-------------|
| 输出 token 量 | content + reasoning_content | 仅 content |
| 响应时间 | 慢（需先思考再输出） | **快 2-5x** |
| 成本 | 按输出 token 计费，**翻倍** | **降低 50%+** |
| 复杂推理 | 准确 ✅ | 可能略差 |
| 事实性问答 | — | **无影响** ✅ |

**实现方式：**

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "提问内容"}],
    extra_body={"thinking": {"type": "disabled"}}
)
```

### 如果需要保留思考模式

仅对**复杂任务**（多步推理、综合多段直播内容）开启：

```python
payload = {}
if task_type == "complex_reasoning":
    payload["extra_body"] = {"thinking": {"type": "enabled"}}
    payload["reasoning_effort"] = "low"
```

**注意：** 思考模式下不支持 `temperature`、`top_p`、`presence_penalty`、`frequency_penalty` 参数（设置不会报错但不生效）。关闭思考模式后这些参数恢复正常。

---

## 1.5 限制输出 Token 长度（max_tokens，减少浪费 🔑）

### 依据

DeepSeek API 兼容 OpenAI 格式，`max_tokens` 是 OpenAI Chat Completions API 的标准参数（[OpenAI 文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_tokens)），DeepSeek 完全支持。

### 现状

当前 QA 后端未设置 `max_tokens`，模型可能回复过长。一个 QA 回答通常只需 100-300 token 就能完成（回答 + 引用标记），但无限制时可能输出 500-1000+ token 的扩展内容。

### 建议

在 API 调用中加入 `max_tokens=300`：

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "提问内容"}],
    max_tokens=300,  # ← 限制输出长度
    extra_body={"thinking": {"type": "disabled"}}
)
```

**预期收益：** 输出 token 减少 50-70%，每问成本进一步降低。

**注意：** `max_tokens` 限制的是输出 token 数（不含输入），设为 300 表示即使模型想写更多也会在 300 token 处截断。如果回答被截断了，可以在系统 prompt 中提示「请简明扼要地回答，不超过 300 token」来缓解。

---

## 二、JSON 输出模式（结构化响应 🔑）

### 适用场景

网站中有 **QA 问答解析**、**视频摘要结构化输出**等需求，需要 LLM 返回格式化的 JSON 数据。

### 官方用法

```python
import json
from openai import OpenAI

client = OpenAI(api_key="<your api key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "请输出 JSON 格式"},
        {"role": "user", "content": "提问内容"}
    ],
    response_format={'type': 'json_object'}
)

print(json.loads(response.choices[0].message.content))
```

### 应用示例

**视频摘要结构化输出：**

```python
system_prompt = """
你是一个直播内容分析助手。请分析输入的直播文本，输出 JSON 格式的结果。

输出格式：
{
    "summary": "直播内容概要（50字以内）",
    "key_points": ["要点1", "要点2", ...],
    "mentioned_topics": ["话题1", "话题2", ...],
    "mood": "positive/neutral/negative"
}
"""
```

**QA 问答结构化输出：**

```python
system_prompt = """
你是一个 idol 知识问答助手。请根据问题和上下文知识，输出 JSON 格式。

输出格式：
{
    "answer": "回答内容",
    "confidence": "high/medium/low",
    "sources": ["信息来源1", "信息来源2"]
}
"""
```

---

## 三、多轮对话（后端无状态）

DeepSeek 的 `/chat/completions` API 是**无状态 API**，服务端不记录上下文。每次请求都需要**客户端传入完整的对话历史**。

```python
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "世界最高的山是什么？"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages
)
messages.append(response.choices[0].message)

# Round 2 — 必须拼上历史消息
messages.append({"role": "user", "content": "第二高呢？"})
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages
)
```

**注意事项：**
- 如果 QA 界面需要**上下文连续对话**，每条新问题都要携带之前的对话历史
- 建议设置**最大上下文轮数**（如 10 轮），超出后丢弃最早的消息，避免 token 过长
- 如果开启思考模式，未调用工具的轮次，前一轮的 reasoning_content **不需要**拼入上下文；调用了工具的轮次则**必须**拼入（否则报 400 错误）

---

## 四、并发限速（Rate Limit）与账号隔离

### 官方限速表

| 模型 | 并发限制 |
|------|----------|
| deepseek-v4-pro | 500 |
| deepseek-v4-flash | 2500 |

⚠️ 注意：
- 一个请求从发出到模型响应完成**记为一个并发**
- 并发限制以**账号粒度**计，与 API Key 无关
- 超过并发时返回 **HTTP 429** 错误码

### user_id 隔离机制

DeepSeek 支持传入 `user_id` 参数，实现同一账号下的**用户级细粒度管理**：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "Hello!"}],
  "user_id": "your_user_id"
}
```

**user_id 的作用：**
1. **内容安全隔离** — 区分不同用户身份，进行内容安全处理
2. **KVCache 隔离** — 不同用户 KVCache 不共享，隐私隔离
3. **调度隔离** — 高并发下每个 user_id 也有独立限制（v4-pro: 500，v4-flash: 2500）

**使用 OpenAI SDK：**

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[...],
    extra_body={"user_id": "jia_yi_live_room"}
)
```

---

## 五、请求保活机制（Keep-Alive）

### 现象

请求发出后，有时需要排队等待服务器空闲。在等待期间：
- **非流式请求：** 持续返回空行（`\n`）
- **流式请求：** 持续返回 SSE keep-alive 注释（`: keep-alive`）

### ⚠️ 必须处理

如果后端代码**自己解析 HTTP 响应**（而不是用 OpenAI SDK），必须**忽略这些空行和注释**，否则会解析失败。

```python
# 流式响应处理时忽略 keep-alive 行
for line in response.iter_lines():
    if not line:
        continue  # 忽略空行
    decoded = line.decode('utf-8')
    if decoded.startswith(': keep-alive'):
        continue  # 忽略保活注释
    # 正常解析 SSE 数据...
```

如果请求发出后 **10 分钟**仍未开始推理，服务器将关闭连接。

---

## 六、错误码完整对照表

| 错误码 | 含义 | 原因 | 建议处理方式 |
|--------|------|------|-------------|
| **400** | 格式错误 | 请求体格式错误 | 根据错误信息修改 payload |
| **401** | 认证失败 | API Key 错误或过期 | 检查 API Key |
| **402** | 余额不足 | 账户余额不足 | 前往平台充值 |
| **422** | 参数错误 | 请求体参数错误 | 根据错误信息修改参数 |
| **429** | 速率限制 | 并发超过限额 | **指数退避重试 + 控制并发数** |
| **500** | 服务器故障 | 服务器内部故障 | 等待后重试；持续则联系支持 |
| **503** | 服务器繁忙 | 负载过高 | 稍后重试 |

### 429 错误的标准处理代码

```python
import time
import random
from requests.exceptions import HTTPError

MAX_RETRIES = 5

def call_deepseek_with_retry(prompt, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            if e.response.status_code == 429:
                # 指数退避 + 随机抖动
                wait = (2 ** attempt) + random.uniform(0, 1)
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                time.sleep(wait)
                continue
            elif e.response.status_code in (500, 503):
                time.sleep(2 ** attempt)
                continue
            else:
                raise  # 400/401/402/422 直接失败
    raise Exception(f"超过最大重试次数 {max_retries}")
```

---

## 七、批处理改造方案（核心）

### 方案一：`asyncio` 异步并发请求 🥇 推荐优先

```python
import asyncio
import aiohttp

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_CONCURRENT = 10

async def fetch_one(session, payload):
    async with session.post(DEEPSEEK_URL, headers=headers, json=payload) as resp:
        if resp.status == 429:
            retry_after = float(resp.headers.get("Retry-After", 2))
            await asyncio.sleep(retry_after)
            return await fetch_one(session, payload)
        return await resp.json()

async def batch_process(video_groups, user_id=None):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async def limited_fetch(payload):
        async with sem:
            return await fetch_one(session, payload)
    async with aiohttp.ClientSession() as session:
        tasks = [
            limited_fetch(build_payload(group, user_id=user_id))
            for group in video_groups
        ]
        return await asyncio.gather(*tasks)
```

- 串行 N×T → 并发 ≈ T（最慢一个），**成本不变**（按 token 计费）

### 方案二：`ThreadPoolExecutor` 多线程并发（备选）

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_process(prompts, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(call_deepseek_api, p): p for p in prompts
        }
        results = {}
        for future in as_completed(future_map):
            prompt = future_map[future]
            try:
                results[prompt] = future.result()
            except Exception as e:
                results[prompt] = {"error": str(e)}
        return results
```

### 方案三：硅基流动批量推理

- 支持 DeepSeek V3/R1 批量推理 API，不受实时速率限制
- V3 批量推理比实时调用**低 50%**
- 适合离线批量处理

---

## 八、本地缓存策略

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=500)
def get_cache_key(prompt_text, model="deepseek-v4-pro"):
    raw = f"{model}:{prompt_text}"
    return hashlib.md5(raw.encode()).hexdigest()
```

**适合做缓存：** 同一场直播的多个重复提问、同一视频摘要请求

---

## 九、工具调用（Tool Calls，记录备用）

> 当前不需要，但可扩展性强。

Tool Calls 让模型能够调用外部工具增强自身能力。**模型本身不执行函数，只返回调用请求，由你的后端代码执行真正的函数逻辑。**

### 示例

```python
from openai import OpenAI

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Shanghai"
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "杭州天气怎么样？"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    tools=tools
)
message = response.choices[0].message
# message.tool_calls[0] → 模型请求调用 get_weather({location: 'Hangzhou'})
# 你的代码执行真实函数，把结果传回模型
```

**可能的用途：** 查公演排期、查口袋48消息、RAG 检索增强、多步推理

### strict 模式（Beta）

设置 `base_url="https://api.deepseek.com/beta"`，每个 function 加 `"strict": true`，严格遵循 JSON Schema。

---

## 十、Token 用量计算

> 官方文档：[Token 用量计算](https://api-docs.deepseek.com/zh-cn/quick_start/token_usage)

### 为什么需要计算 Token 用量

DeepSeek API 按 token 计费（输入 + 输出），了解 Token 用量可以帮助：
- **估算成本** — 提前算出一次查询花了多少钱
- **优化 prompt** — 发现哪些轮次输入的 prompt 过长
- **监控异常** — 突然暴增的 token 用量可能意味着 bug

### Token 计数通用方法

OpenAI 提供 `tiktoken` 库，但 DeepSeek 的 tokenizer 与 OpenAI 不完全相同，更准确的方式是使用 DeepSeek 官方 API 或自行预估。

#### 方法一：API 返回中的 usage 字段

DeepSeek 每次 API 调用都会在响应中返回 `usage` 信息，直接读取即可：

```python
import openai

response = openai.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "你好"}],
    extra_body={"thinking": {"type": "disabled"}}
)

# 从 API 返回值中直接查看用量
usage = response.usage
print(f"输入 tokens: {usage.prompt_tokens}")
print(f"输出 tokens: {usage.completion_tokens}")
print(f"总 tokens: {usage.total_tokens}")
```

#### 方法二：使用 DeepSeek 官方 Tokenizer（推荐 ✅ 最准确）

> 官方 tokenizer 已保存在 `dev-notes/deepseek_v3_tokenizer/` 目录下

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "dev-notes/deepseek_v3_tokenizer",  # 本地路径
    trust_remote_code=True
)

# 编码文本
tokens = tokenizer.encode("你好世界 Hello World")
print(f"Token 数量: {len(tokens)}")

# 编码 messages（更准确的成本估算）
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "陈嘉仪今天直播了吗？"},
]

# 使用 chat_template 计算完整输入 token
full_text = tokenizer.apply_chat_template(messages, tokenize=False)
input_tokens = tokenizer.encode(full_text)
print(f"输入 tokens: {len(input_tokens)}")
```

> 💡 **安装依赖：** `pip install transformers`

#### 方法三：使用 tiktoken 本地测算（无需额外文件，中英文场景可用）

```python
import tiktoken

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """估算 token 数，适合中英文混合场景"""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # 回退公式：中文 ≈ 1.5 tokens/字，英文 ≈ 0.3 tokens/字母
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.3)
```

#### 方法四：从 API 响应直接读取

```python
# 每次 API 调用的响应中已经包含 token 用量，这是最准确的
usage = response.usage
print(f"输入 tokens: {usage.prompt_tokens}")  # 实际计费的输入
print(f"输出 tokens: {usage.completion_tokens}")  # 实际计费的输出
print(f"总 tokens: {usage.total_tokens}")
```

### Token 与成本对照

> ⚠️ 具体价格请以 DeepSeek 官网最新定价为准。以下仅为参考示例。

| 模型 | 输入价格 | 输出价格 | 备注 |
|------|---------|---------|------|
| deepseek-v4-pro | 约 ¥X/1K tokens | 约 ¥X/1K tokens | 实时推理 |
| deepseek-v4-flash | 约 ¥X/1K tokens | 约 ¥X/1K tokens | 快速、便宜 |

**实用公式：**
```python
def estimate_cost(prompt_tokens, completion_tokens, input_price=0.002, output_price=0.008):
    """估算一次 API 调用的成本（单位：人民币）"""
    cost = (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
    return cost  # 返回元
```

---

## 十一、查询余额 —— 用于网站服务状态监控（Python）

> 官方文档：[查询余额 API](https://api-docs.deepseek.com/zh-cn/api/get-user-balance)

| 项目 | 内容 |
|------|------|
| 请求方式 | GET |
| 接口地址 | `https://api.deepseek.com/user/balance` |
| 鉴权 | `Authorization: Bearer <API_KEY>` |
| 返回字段 | `currency`(CNY/USD)、`total_balance`(总余额)、`granted_balance`(赠金)、`topped_up_balance`(充值) |

### Python 实现

```python
import requests

DEEPSEEK_API_KEY = "your_api_key_here"
BALANCE_URL = "https://api.deepseek.com/user/balance"

def check_balance(api_key=DEEPSEEK_API_KEY):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    try:
        resp = requests.get(BALANCE_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        total = float(data["total_balance"])
        if total <= 0:
            status = "empty"
        elif total < 10:
            status = "low"
        else:
            status = "ok"
        return {
            "currency": data["currency"],
            "total_balance": data["total_balance"],
            "granted_balance": data["granted_balance"],
            "topped_up_balance": data["topped_up_balance"],
            "is_low": total <= 10,
            "status": status
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}

# 带 10 分钟缓存的查询
API_STATUS_CACHE = {"balance": None, "last_check": 0}

def get_api_status():
    import time
    now = time.time()
    if now - API_STATUS_CACHE["last_check"] < 600:
        return API_STATUS_CACHE["balance"]
    result = check_balance()
    API_STATUS_CACHE["balance"] = result
    API_STATUS_CACHE["last_check"] = now
    return result
```

### 集成到 FastAPI / Flask

```python
@app.route("/api/status")
def api_status():
    balance_info = get_api_status()
    return {
        "deepseek_api": {
            "status": balance_info.get("status", "unknown"),
            "balance": balance_info.get("total_balance", "N/A"),
            "currency": balance_info.get("currency", "CNY"),
        }
    }
```

**前端展示：** 正常 → 绿色 ✅ | 低于阈值 → 黄色 ⚠️ | 余额 0 → 红色 ❌

---

## 推荐实施优先级

| 阶段 | 措施 | 预期效果 | 工作量 |
|------|------|----------|--------|
| **Phase 1 🔥** | 默认关闭思考模式 | 成本降 **50%+**，响应快 2-5x | 🔧 极小 |
| **Phase 1 🔥** | 接入 JSON 输出模式 | 结构化响应，解析稳定 | 🔧 极小 |
| **Phase 1.5** | 串行→异步并发（asyncio） | 批处理 3-10x 提速 | 🔧 小 |
| **Phase 1.5** | 加入完整错误码处理 + 指数退避 | 稳定性提升 | 🔧 小 |
| **Phase 1.5** | 接入请求保活处理 | 避免解析异常 | 🔧 极小 |
| **Phase 1.5** | 接入查询余额接口 | 网站状态监控 | 🔧 极小 |
| **Phase 2** | 上传 user_id（直播间ID） | 多主播隔离 | 🔧 极小 |
| **Phase 3** | 本地 LRU 缓存 | 减少重复调用 | 🔧 中 |
| **Phase 4** | 消息队列 + Redis | 生产级高吞吐 | 🏗️ 大 |

---

## 总结：改造前后对比

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| 思考模式 | 默认开启 | **默认关闭**，仅复杂任务开启 |
| 输出格式 | 纯文本，手动解析 | **JSON 模式**，直接反序列化 |
| 响应速度 | 慢（思考+输出） | 快 2-5x |
| 成本 | 输出 token 翻倍 | **降低 50%+** ✅ |
| 批处理 | 串行逐个 | 异步并发 |
| 耗时 (N个视频) | N × T | ≈ T（最慢一个） |
| 错误处理 | 可能崩溃 | 自动重试+退避 |
| 限流处理 | 无 | 429→指数退避 |
| 保活机制 | 可能解析失败 | 忽略空行/注释 |
| 多主播隔离 | 无 | user_id 隔离 |
| 上下文管理 | 未知 | 无状态，客户端维护历史 |
| 缓存复用 | 无 | LRU 缓存 |
| 服务状态监控 | 无 | 余额查询接口 |
