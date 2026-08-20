# ToyAgent: main agent + sub agent（LMInfer 后端）

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
   OpenAI 兼容 HTTP (LMInfer /v1/chat/completions, agent 模式)
```

| 文件 | 说明 |
| --- | --- |
| `llm.py` | OpenAI 兼容客户端，仅依赖 `requests`，直连 LMInfer `/v1/chat/completions`（agent 模式） |
| `agent.py` | `Trace`（链式 trace + agent 调用链）、`Tool`、`Agent`（tool-calling 循环，支持两种工具模式） |
| `main.py` | 示例：1 个 main agent + 3 个 sub agent（researcher/writer/reviewer）+ CLI |
| `smoke_test.py` | 离线冒烟测试：验证 `main -> sub -> main -> sub -> main -> out` 链路（两种模式）+ agent trace 断言 |

## 依赖

- Python 3.10+
- `pip install -r requirements.txt`（仅 `requests`）

## 1. 启动 LMInfer 服务

默认以 **LMInfer 的 agent 模式**运行：请求携带 `mode: "agent"`、调用链 `trace` 和
`session_id`，LMInfer 在服务端把主/子 agent 的多次调用关联到同一个会话并累计消耗
（见下方「Agent 模式」）。

```bash
# 启动 LMInfer（本项目后端；Qwen3 原生支持 tool calling）
python -m lminfer serve /home/tanger/workspace/models/Qwen3-0.6B \
    --served-model-name Qwen3-0.6B \
    --port 8000
```

若模型**不**原生支持 tool calling（如 Mistral-7B-Instruct-v0.2），无需任何参数——
`tool_mode: auto` 会在服务返回 400 时自动降级为文本协议。想改用 vLLM 等普通
OpenAI 兼容服务时，加 `--no-agent-mode` 关闭 agent 模式即可。

## 2. 运行

```bash
# 方式一：使用配置文件（推荐，见 example/ 目录）
python main.py --config example/config-lminfer.yaml

# 方式二：命令行参数
python main.py \
    --base-url http://localhost:8000/v1 \
    --model Qwen3-0.6B \
    --task "请先调用 call_sub_agent 工具，把「快速排序的平均时间复杂度是多少？」交给 researcher 子代理调研，然后原样复述它的答案。"
```

**配置优先级**：命令行参数 > 配置文件 > 环境变量（`VLLM_BASE_URL` / `VLLM_MODEL`）> 内置默认值。
即：配置里的值可被对应命令行参数覆盖；两者都没给时回落到环境变量/默认值。

### 配置文件（`--config`）

YAML 格式，可写注释。示例见 `example/`：

| 文件 | 场景 |
| --- | --- |
| `example/config-lminfer.yaml` | LMInfer + Qwen3（agent 模式，推荐） |
| `example/config-mistral.yaml` | Mistral-7B-Instruct（文本协议自动降级） |
| `example/config-qwen3.yaml` | Qwen3-8B（含 vLLM 原生 tool 参数说明） |
| `example/config-qwen-native.yaml` | Qwen2.5-Instruct（原生 tool calling） |
| `example/config-custom-sub-agents.yaml` | 自定义 sub agents（翻译+校对），演示不改代码扩展 |

主要字段：

| 字段 | 说明 |
| --- | --- |
| `base_url` / `model` / `api_key` | LMInfer 服务连接 |
| `temperature` / `max_tokens` / `max_iters` / `sub_max_iters` / `tool_mode` / `quiet` | 对应同名命令行参数 |
| `force_tool_call` | 第一轮强制工具调用（默认 true；见下） |
| `agent_mode` | LMInfer agent 模式开关（默认 true；设 false 兼容 vLLM） |
| `task` | 交给 main agent 的任务 |
| `sub_agents` | 映射：子 agent 名称 -> 系统提示词；**完全由配置定义**，名称自动进入 `call_sub_agent` 的枚举（并成为其别名，见下） |
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
| `--force-tool-call` / `--no-force-tool-call` | 开 | 第一轮强制工具调用：原生模式首轮 `tool_choice="required"`，文本模式首条回复必须输出 `TOOL_CALL`，保证 main 先分派子任务而不是直接作答 |
| `--agent-mode` / `--no-agent-mode` | 开 | LMInfer agent 模式（默认开；`--no-agent-mode` 兼容 vLLM 等普通 OpenAI 服务） |
| `--quiet` | 关 | 不实时打印，只输出最终答案与链式 trace 摘要 |

### 工具调用模式（`--tool-mode`）

| 模式 | 说明 |
| --- | --- |
| `auto`（默认） | 先尝试 OpenAI 原生 `tool_calls`；若服务返回 HTTP 400（模型不支持），**自动降级**为文本协议 |
| `native` | 强制原生 `tool_calls`，需模型与 vLLM 支持 |
| `text` | 强制文本协议：工具列表写进首条消息，模型输出 `TOOL_CALL: {"tool": ..., "arguments": {...}}`（支持一次多个），代码解析后执行。任何模型可用（Mistral、Llama 2 等） |

> **兼容性说明**：实现刻意不使用 `role: "system"`（部分 vLLM 构建会拒绝任何含 system
> 角色的消息，报 "Conversation roles must alternate..."），系统设定并入首条 user 消息。
> Qwen3 等模型输出的 `<think>` 推理块会在解析工具调用前剥离，不会污染最终答案。

### 工具调用可靠性（模型总是跳过或不规范调用工具？）

三个机制保证 main agent 稳定调用 sub agent：

1. **首轮强制调用**（`--force-tool-call`，默认开）：第一轮就发 `tool_choice="required"`
   （原生模式）或要求首条回复必须输出 `TOOL_CALL`（文本模式），模型必须先分派子任务
   才能作答，无法跳过工具直接回答。
2. **子代理名即别名**：Qwen3 等小模型经常直接用子代理名调用（`writer({"task": ...})`
   而不是规范的 `call_sub_agent({"name": "writer", "task": ...})`）。所有 sub agent 名
   自动成为 `call_sub_agent` 的别名，此类调用会被翻译并补上 `name` 参数后照常执行，
   不再报「未知工具」。trace 中会打印 `别名 'writer' -> call_sub_agent({...})` 便于观察。
3. **提示词明示**：main 的系统提示词明确列出可用的 sub agent 名单与精确调用格式，
   并规定「第一步必须调用工具、拿到结果前不得输出最终回答」。

另外两个实践建议：把 `temperature` 调到 **0.3 左右**（0.7 容易让模型跳过工具直接
作答）；示例任务应**明说必须先调用工具、并把结果复述出来**，而不是给一个模型能
直接答完的任务（如内置示例的快速排序调研任务——纯推理分派，不涉及代码执行）。

### Agent 模式（LMInfer 会话追踪）

agent 模式（默认开启）下，每次模型请求都会携带：

- `mode: "agent"`：告诉 LMInfer 这是 agent 事务内的请求；
- `trace`：调用链，如 `["main", "researcher", "main", "writer", "main"]`，最后一个
  元素是当前 agent。由 `Trace.agent_trace()` 在每次请求前维护（连续同名去重），
  主/子 agent 的调用关系一目了然；
- `session_id`：由 `LLMClient` 自动维护——首次请求不带，LMInfer 生成后随响应返回，
  客户端记住并自动用于后续请求，从而在服务端把同一个任务的全部调用关联到一个会话。

运行结束会打印本次的 `LMInfer agent 会话: <id>`，在服务端
`GET /v1/agent/sessions` 可以看到该会话的请求数、累计 token 和 agent 名单。
若 LMInfer 重启导致内存会话丢失（请求返回 404），客户端会自动清掉 `session_id`
并以新会话重试一次，agent 无需感知。想关闭 agent 模式（如改用 vLLM），用
`--no-agent-mode` 或配置 `agent_mode: false`。

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

- 新增 sub agent：在 `main.py` 的 `SUB_SYSTEM_PROMPTS`（或配置文件的 `sub_agents`）
  里加一项即可，名字会自动出现在 `call_sub_agent` 工具的 `enum` 中，并自动成为该
  工具的别名（可直接用子代理名调用）。
- sub agent 也可自带工具（嵌套调用）：给 `Agent` 传 `tools=[...]` 即可，
  `sub-max-iters` 相应调大。
