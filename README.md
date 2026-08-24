# ToyAgent: main agent + sub agent（OpenAI 兼容后端，LMInfer/vLLM）

一个干净、可扩展的 tool-calling agent 示例：**main agent** 收到任务后**自主决定**
如何执行——直接回答、用文件工具（`read_file` 等）、还是通过 `call_sub_agent`
把子任务分派给 **sub agent**；**sub agent 同样可以调用文件工具**，只是不能再继续
向下分派子任务。形成典型的智能体调用链：

```
main -> (read_file | call_sub_agent) -> sub -> main -> ... -> main -> out
```

每一步都以 `[agent名]` 前缀实时打印，结束时输出整条链的 trace 摘要。

## 架构

```
┌─────────────────────┐  tool: call_sub_agent   ┌──────────────────────┐
│     main agent      │ ──────────────────────▶ │  sub agents           │
│  分析 · 规划 · 汇总   │ ◀────────────────────── │  researcher/writer/   │
│  + 文件工具(通用)     │       返回结果文本       │  reviewer (可扩展)     │
└─────────────────────┘                         │  + 文件工具(通用)      │
         │                                      └──────────────────────┘
         ▼
   OpenAI 兼容 HTTP (LMInfer /v1/chat/completions, agent 模式)
```

**主/子 agent 共用同一个 `Agent` 循环，区别只在工具集**：
main 拥有 `call_sub_agent`（root_only）+ 通用文件工具；sub agent 只有通用文件工具，
构造时自动剥离 root_only 工具（`sub -> sub` 最多两级，避免无限递归）。

| 文件 | 说明 |
| --- | --- |
| `llm.py` | OpenAI 兼容客户端，仅依赖 `requests`，直连 LMInfer `/v1/chat/completions`（agent 模式：mode/trace/session_id） |
| `tools.py` | `Tool`（工具 = 可调用对象 + OpenAI schema）+ 内置文件工具（read_file / write_file / list_directory / search_files），主/子 agent 通用 |
| `agent.py` | `Trace`（链式日志 + agent 调用链）、`ChatBackend`（工具协议适配层）、`Agent`（ReAct 风格 tool-calling 循环） |
| `main.py` | 示例：1 个 main agent + N 个 sub agent（researcher/writer/reviewer）+ CLI + 配置加载 |
| `smoke_test.py` | 离线冒烟测试：6 个场景（native / forced / sub-tools / direct / text-fallback / alias）+ 解析器与文件工具单测 |
| `gaia_test.py` | GAIA_Text 基准测试运行器（真实模型，结果落盘 `gaia_results/`，数据在 `data/`，配置 `example/config-gaia.yaml`） |

## 依赖

- Python 3.10+
- `pip install -r requirements.txt`（仅 `requests` + `PyYAML`）

## 1. 启动 LMInfer 服务

默认以 **LMInfer 的 agent 模式**运行：请求携带 `mode: "agent"`、调用链 `trace` 和
`session_id`，LMInfer 在服务端把主/子 agent 的多次调用关联到同一个会话并累计消耗。

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

### 配置文件（`--config`）

YAML 格式，可写注释。示例见 `example/`：

| 文件 | 场景 |
| --- | --- |
| `example/config-lminfer.yaml` | LMInfer + Qwen3（agent 模式，推荐） |
| `example/config-mistral.yaml` | Mistral-7B-Instruct（文本协议自动降级） |
| `example/config-qwen3.yaml` | Qwen3-8B（含 vLLM 原生 tool 参数说明） |
| `example/config-qwen-native.yaml` | Qwen2.5-Instruct（原生 tool calling） |
| `example/config-custom-sub-agents.yaml` | 自定义 sub agents（翻译+校对），演示不改代码扩展 |
| `example/config-gaia.yaml` | GAIA 基准测试（vLLM + Qwen3-8B，配合 `gaia_test.py`） |

主要字段：

| 字段 | 说明 |
| --- | --- |
| `base_url` / `model` / `api_key` | LLM 服务连接 |
| `temperature` / `max_tokens` / `max_iters` / `sub_max_iters` / `tool_mode` / `quiet` | 对应同名命令行参数 |
| `force_tool_call` | 第一轮强制工具调用（**默认 false**；见下） |
| `agent_mode` | LMInfer agent 模式开关（默认 true；设 false 兼容 vLLM） |
| `task` | 交给 main agent 的任务 |
| `sub_agents` | 映射：子 agent 名称 -> 系统提示词；**完全由配置定义**，名称自动进入 `call_sub_agent` 的枚举（并成为其别名） |
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
| `--sub-max-iters` | `4` | sub agent 最大轮数（默认 4：留出「调用文件工具后再作答」的轮数） |
| `--tool-mode` | `auto` | `auto` / `native` / `text`，见下 |
| `--force-tool-call` / `--no-force-tool-call` | 关 | 首轮强制工具调用：原生模式首轮 `tool_choice="required"`，文本模式首条回复必须输出 `TOOL_CALL`。**默认关**——主 agent 自主决定是否调用工具 |
| `--agent-mode` / `--no-agent-mode` | 开 | LMInfer agent 模式（默认开；`--no-agent-mode` 兼容 vLLM 等普通 OpenAI 服务） |
| `--quiet` | 关 | 不实时打印，只输出最终答案与链式 trace 摘要 |

### 工具调用模式（`--tool-mode`）

三种模式的全部差异封装在 `ChatBackend` 里，主循环只认 OpenAI 原生 `tool_calls` 形状：

| 模式 | 说明 |
| --- | --- |
| `auto`（默认） | 先尝试 OpenAI 原生 `tool_calls`；若服务返回 HTTP 400（模型不支持），**自动降级**为文本协议 |
| `native` | 强制原生 `tool_calls`，需模型与服务支持 |
| `text` | 强制文本协议：工具列表写进首条消息，模型输出 `TOOL_CALL: {"tool": ..., "arguments": {...}}`（支持一次多个），后端解析后归一化为原生形状。任何模型可用（Mistral、Llama 2 等） |

> **兼容性说明**：实现刻意不使用 `role: "system"`（部分 vLLM 构建会拒绝任何含 system
> 角色的消息，报 "Conversation roles must alternate..."），系统设定并入首条 user 消息。
> Qwen3 等模型输出的 `<think>` 推理块会在解析工具调用前剥离，不会污染最终答案。

### 主 agent 自主决定（`force_tool_call` 默认关）

默认情况下**不强制**主 agent 首轮调用工具：模型根据任务自行判断——简单问题直接回答、
需要看文件就用 `read_file`、需要专项能力就调用 `call_sub_agent`。`main` 的系统提示词
明确列出两类能力并规定「是否调用、调用哪个、调用几次由你自主决定」。

`force_tool_call: true` 作为兜底开关保留：小模型（如 Qwen3-0.6B）默认倾向跳过工具
直接作答，工具调用不稳定时打开它，第一轮就发 `tool_choice="required"`（原生模式）
或要求首条回复必须输出 `TOOL_CALL`（文本模式），保证 main 先分派子任务。

**工具调用可靠性**的另外三个机制（与之前相同）：

1. **子代理名即别名**：所有 sub agent 名自动成为 `call_sub_agent` 的别名，模型直接写
   `writer({"task": ...})` 会被翻译并补上 `name` 参数后照常执行，不再报「未知工具」。
2. **提示词明示**：main 的系统提示词明确列出可用的 sub agent 名单与精确调用格式。
3. **每轮显式 `tool_choice="auto"`（原生模式）**：LMInfer 默认
   `enable_auto_tool_choice=false`，带 tools 但省略 tool_choice 的请求会被按
   `none` 处理、渲染 prompt 时不输出 tools 系统段——首轮（required）与后续轮
   prompt 前缀不一致，agent 模式的跨请求前缀 KV 复用（`--reuse-agent-kv` /
   `--reuse-agent-kv-append`）无法命中。首轮 `required`、后续轮显式 `auto`，
   保证每轮都渲染相同的 tools 段，各轮 prompt 逐字延续上一轮完整序列。

另外两个实践建议：把 `temperature` 调到 **0.3 左右**（0.7 容易让模型跳过工具直接
作答）；示例任务应**明说需要用到哪些工具/子代理**，而不是给一个模型能直接答完的任务。

### Agent 模式（LMInfer 会话追踪）

agent 模式（默认开启）下，每次模型请求都会携带：

- `mode: "agent"`：告诉 LMInfer 这是 agent 事务内的请求；
- `trace`：调用链，如 `["main", "researcher", "main", "writer", "main"]`，最后一个
  元素是当前 agent。由 `Trace.agent_trace()` 在每次请求前维护（连续同名去重）；
- `session_id`：由 `LLMClient` 自动维护——首次请求不带，LMInfer 生成后随响应返回，
  客户端记住并自动用于后续请求，从而在服务端把同一个任务的全部调用关联到一个会话。

运行结束会打印本次的 `LMInfer agent 会话: <id>`。若 LMInfer 重启导致内存会话丢失
（请求返回 404），客户端会自动清掉 `session_id` 并以新会话重试一次，agent 无需感知。
想关闭 agent 模式（如改用 vLLM），用 `--no-agent-mode` 或配置 `agent_mode: false`。

### GAIA 基准测试（`gaia_test.py`）

用真实模型跑 GAIA_Text 数据集（数据在 `data/`，配置在 `example/config-gaia.yaml`）：

```bash
# 1. 启动 vLLM（Qwen3-8B，原生 tool calling）
python -m vllm.entrypoints.openai.api_server \
    --model /home/tanger/workspace/models/Qwen3-8B \
    --served-model-name Qwen3-8B --max-model-len 20480 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes

# 2. 跑第一条 GAIA 任务
cd /home/tanger/workspace/ToyAgent
python3 gaia_test.py --config example/config-gaia.yaml --limit 1

# 3. 跑全量 23 条
python3 gaia_test.py --config example/config-gaia.yaml \
    --data data/gaia_document_only.jsonl \
    --out gaia_results/toyagent_gaia_all.jsonl
```

- `data/task_1.jsonl`：GAIA 第一条（spreadsheet 库存，gold answer 为
  `Time-Parking 2: Parallel Universe`）；全量数据集为 `data/gaia_document_only.jsonl`，
  附件树在 `data/documents/`（与数据集内 `file_path` 的相对路径一致，read_file 从
  ToyAgent 目录解析）。数据来源见 `data/README.md`；
- 附件由模型自己用 `read_file` 读取（main 自己读，或分派给 researcher 读，自主决定）；
- 结果落盘到 `gaia_results/*.jsonl`（含调用链、token 消耗、GAIA 风格 exact-match 判分）；
- 单条失败不会中断整批（逐条捕获，`error` 字段记录）。
- 实测（Qwen3-8B，thinking 开，force_tool_call 关）：
  - 第 1 条自动决策模式：main 自己 `read_file` 读完直接作答，`main -> out`，✓；
  - 第 1 条分派模式（`--delegate-first`）：`main -> researcher(read_file) -> main -> out`，✓。

### 离线测试（无需服务）

```bash
python smoke_test.py
```

用脚本化的假 LLM 覆盖 6 个场景：

| 场景 | 验证点 |
| --- | --- |
| `native` | 原生 tool_calls；主 agent 自主决定分派；**sub agent 请求也携带文件工具 schema** |
| `forced` | `force_tool_call=True` 时首轮 `tool_choice="required"` |
| `sub-tools` | **sub agent 自己调用 `read_file`** 后再作答（主/子 agent 都能调用工具） |
| `direct` | **主 agent 自主决定不调用任何工具**，第一轮直接回答 |
| `text-fallback` | 400 自动降级文本协议（Mistral 场景） |
| `alias` | 模型直接用子代理名调用，别名层自动翻译（native + text） |

并断言链路顺序 `main -> researcher -> main -> writer -> main -> out`、LMInfer
agent 模式 trace 链逐次增长、每轮显式 `tool_choice`。

## 工作原理

1. **main agent** 收到任务后，模型自主决定下一步：返回工具调用（原生 `tool_calls`
   或文本协议 `TOOL_CALL: {json}`，均可指 `call_sub_agent` 或 `read_file` 等文件工具；
   文本协议一次可返回多个调用，会按顺序逐个执行）；
2. 主循环执行工具：`call_sub_agent` 找到对应 sub agent 并调用其 `run(task)`；
   `read_file` 等文件工具直接操作本地文件；
3. **sub agent** 独立完成自己的对话轮次（同样可以调用文件工具），结果文本返回给 main；
4. 结果回填给 main 的对话历史（原生为 `role: "tool"` 消息，文本协议为 `[工具结果]` 用户消息），
   main 继续推理；
5. 重复直到 main 的回复不含工具调用 → 该回复即为最终答案（`main -> out`）。

## 扩展

- **新增工具**：在 `tools.py` 里加一个 `Tool`（name/description/parameters/func），
  需要让主/子 agent 都用就加进 `build_file_tools()`；只想给某个 agent 用，就在
  构造该 agent 时传 `tools=[...]`。工具返回字符串给模型，抛异常会被捕获并转成
  `ERROR: ...` 文本回填，模型自行调整。
- **新增 sub agent**：在 `main.py` 的 `SUB_SYSTEM_PROMPTS`（或配置文件的 `sub_agents`）
  里加一项即可，名字会自动出现在 `call_sub_agent` 工具的 `enum` 中，并自动成为该
  工具的别名（可直接用子代理名调用）。
- **sub agent 也带工具**：默认已带文件工具；`--sub-max-iters` 相应调大（默认 4）。
- **换后端**：只要实现 OpenAI 兼容 `/v1/chat/completions` 即可（vLLM、LMInfer、llama.cpp 等）。
