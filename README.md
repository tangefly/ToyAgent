# ToyAgent: main agent + sub agent（vLLM 后端）

一个简单的 tool-calling agent 示例：**main agent** 负责问题分析与规划，通过
`call_sub_agent` 工具把子任务分派给 **sub agent**，形成链式调用：

```
main -> sub -> main -> sub -> ... -> main -> out
```

每一步都以 `[agent名]` 前缀实时打印，结束时输出整条链的 trace 摘要。

## 架构

```
┌─────────────────────┐  tool: call_sub_agent   ┌──────────────────────┐
│     main agent      │ ──────────────────────▶ │  sub agents           │
│  分析 · 规划 · 汇总   │ ◀────────────────────── │  researcher / writer  │
└─────────────────────┘       返回结果文本       │  reviewer (可扩展)     │
         │                                        └──────────────────────┘
         ▼
   OpenAI 兼容 HTTP (vLLM /v1/chat/completions)
```

| 文件 | 说明 |
| --- | --- |
| `llm.py` | OpenAI 兼容客户端，仅依赖 `requests`，直连 vLLM `/v1/chat/completions` |
| `agent.py` | `Trace`（链式 trace）、`Tool`、`Agent`（tool-calling 循环，支持两种工具模式） |
| `main.py` | 示例：1 个 main agent + 3 个 sub agent（researcher/writer/reviewer）+ CLI |
| `smoke_test.py` | 离线冒烟测试：验证 `main -> sub -> main -> sub -> main -> out` 链路（两种模式） |

## 依赖

- Python 3.10+
- `pip install -r requirements.txt`（仅 `requests`）

## 1. 启动 vLLM 服务

模型**不需要**原生支持 tool calling（见下方「工具调用模式」）：

```bash
vllm serve /path/to/Mistral-7B-Instruct-v0.2 \
    --served-model-name Mistral-7B-Instruct \
    --max-model-len 20480 \
    --port 8000
```

若模型原生支持 OpenAI tool calling（如 Qwen2.5-Instruct），想走原生模式，可加：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

## 2. 运行

```bash
# 方式一：使用配置文件（推荐，见 example/ 目录）
python main.py --config example/config-mistral.yaml

# 方式二：命令行参数
python main.py \
    --base-url http://localhost:8000/v1 \
    --model Mistral-7B-Instruct \
    --task "请写一个 Python 函数 is_prime(n) 判断整数 n 是否为素数，然后分别判断 97 和 91，最后给出函数代码和两个数的判断结果。"
```

**配置优先级**：命令行参数 > 配置文件 > 环境变量（`VLLM_BASE_URL` / `VLLM_MODEL`）> 内置默认值。
即：配置里的值可被对应命令行参数覆盖；两者都没给时回落到环境变量/默认值。

### 配置文件（`--config`）

YAML 格式，可写注释。示例见 `example/`：

| 文件 | 场景 |
| --- | --- |
| `example/config-mistral.yaml` | Mistral-7B-Instruct（文本协议自动降级） |
| `example/config-qwen3.yaml` | Qwen3-8B（当前部署，含 vLLM 原生 tool 参数说明） |
| `example/config-qwen-native.yaml` | Qwen2.5-Instruct（原生 tool calling） |
| `example/config-custom-sub-agents.yaml` | 自定义 sub agents（翻译+校对），演示不改代码扩展 |

主要字段：

| 字段 | 说明 |
| --- | --- |
| `base_url` / `model` / `api_key` | vLLM 服务连接 |
| `temperature` / `max_tokens` / `max_iters` / `sub_max_iters` / `tool_mode` / `quiet` | 对应同名命令行参数 |
| `task` | 交给 main agent 的任务 |
| `sub_agents` | 映射：子 agent 名称 -> 系统提示词；**完全由配置定义**，名称自动进入 `call_sub_agent` 的枚举 |
| `main_system_prompt` | 可选，覆盖 main agent 的默认系统提示词 |

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | 无 | YAML 配置文件路径（见 `example/` 目录） |
| `--base-url` | `http://localhost:8000/v1`（env: `VLLM_BASE_URL`） | vLLM 服务地址，可带可不带 `/v1` |
| `--model` | env: `VLLM_MODEL`（必填） | 模型名称，需与 vLLM 服务一致 |
| `--api-key` | `EMPTY` | vLLM 默认不校验 |
| `--task` | 内置示例任务 | 交给 main agent 的任务 |
| `--temperature` | `0.7` | 采样温度 |
| `--max-iters` | `10` | main agent 最大循环轮数（防止死循环） |
| `--sub-max-iters` | `1` | sub agent 最大轮数（默认 1：直接作答，不再调用工具） |
| `--tool-mode` | `auto` | `auto` / `native` / `text`，见下 |
| `--quiet` | 关 | 不实时打印，只输出最终答案与链式 trace 摘要 |

### 工具调用模式（`--tool-mode`）

| 模式 | 说明 |
| --- | --- |
| `auto`（默认） | 先尝试 OpenAI 原生 `tool_calls`；若服务返回 HTTP 400（模型不支持），**自动降级**为文本协议 |
| `native` | 强制原生 `tool_calls`，需模型与 vLLM 支持 |
| `text` | 强制文本协议：工具列表写进首条消息，模型输出 `TOOL_CALL: {"tool": ..., "arguments": {...}}`（支持一次多个），代码解析后执行。任何模型可用（Mistral、Llama 2 等） |

> **兼容性说明**：实现刻意不使用 `role: "system"`（部分 vLLM 构建会拒绝任何含 system
> 角色的消息，报 "Conversation roles must alternate..."），系统设定并入首条 user 消息。
> 小模型可能输出格式不规范的调用（如臆造子 agent 名字、漏掉 `task` 字段），此时工具
> 返回明确错误并回传给模型，由模型自行纠正。Qwen3 等模型输出的 `<think>` 推理块会在
> 解析工具调用前剥离，不会污染最终答案。

### 离线测试（无需服务）

```bash
python smoke_test.py
```

用脚本化的假 LLM 覆盖两种场景（原生模式 / 400 自动降级文本协议），
断言链路顺序 `main -> researcher -> main -> writer -> main -> out` 正确。

## 工作原理

1. **main agent** 收到任务后，LLM 返回工具调用（原生 `tool_calls` 或文本协议
   `TOOL_CALL: {json}`，均指 `call_sub_agent(name, task)`；文本协议一次可返回多个调用，
   会按顺序逐个执行）；
2. 主循环执行工具：找到对应 sub agent，调用其 `run(task)`；
3. **sub agent** 独立完成自己的对话轮次（无工具，直接作答），结果文本返回给 main；
4. 结果回填给 main 的对话历史（原生为 `role: "tool"` 消息，文本协议为 `[工具结果]` 用户消息），
   main 继续推理；
5. 重复直到 main 的回复不含工具调用 → 该回复即为最终答案（`main -> out`）。

## 扩展

- 新增 sub agent：在 `main.py` 的 `SUB_SYSTEM_PROMPTS` 里加一项即可，名字会自动
  出现在 `call_sub_agent` 工具的 `enum` 中。
- sub agent 也可自带工具（嵌套调用）：给 `Agent` 传 `tools=[...]` 即可，
  `sub-max-iters` 相应调大。
