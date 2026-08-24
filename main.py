"""示例入口：main agent + 若干 sub agent，模型服务为 OpenAI 兼容接口（LMInfer/vLLM）。

用法：
    python main.py --config example/config-lminfer.yaml
    python main.py --base-url http://localhost:8000/v1 --model Qwen3-0.6B

架构（主/子 agent 共用同一个 Agent 循环，区别只在工具集）：
- main agent:   通用文件工具（read_file/write_file 等）+ call_sub_agent；
                收到任务后自主决定：直接回答 / 用文件工具 / 调用子 agent。
- sub agent:    只有通用文件工具（能自己读文件、写文件），不能继续向下分派子任务
                （call_sub_agent 是 root_only 工具，构造时自动剥离，最多两级）。

默认以 LMInfer 的 agent 模式请求（mode/trace/session_id，服务端按会话统计），
用 --no-agent-mode 关闭后可兼容 vLLM 等普通 OpenAI 服务。

配置优先级: 命令行参数 > 配置文件 > 环境变量(VLLM_BASE_URL/VLLM_MODEL) > 内置默认
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from agent import Agent, Trace
from llm import LLMClient
from tools import Tool, build_file_tools

MAIN_SYSTEM_PROMPT = (
    "你是 main agent，负责接收用户任务、分析问题、规划并执行步骤，最后汇总输出。\n"
    "你拥有两类能力:\n"
    "1. 调用子 agent: call_sub_agent 工具可以把需要专项能力（调研/实现/审查等）的"
    "子任务分派给 sub agent 执行；只有当任务确实需要专项处理时才调用它。\n"
    "2. 使用文件工具: read_file / write_file / list_directory / search_files 可以直接"
    "读写文件、查看目录结构，用于需要读取本地文件或产出文件的任务。\n"
    "执行规则:\n"
    "1. 收到任务后先分析: 简单问题可直接回答；需要看文件就用文件工具；需要专项能力"
    "（如调研、写代码、审查）就调用 call_sub_agent 分派给合适的 sub agent——"
    "是否调用、调用哪个、调用几次，都由你根据任务自主决定，不要为了调用而调用。\n"
    "2. call_sub_agent 的参数 name 必须是已列出的 sub agent 名之一，task 写清楚交给"
    "该 sub agent 的具体问题（也可以直接用子代理名作为工具名调用，如 writer({\"task\": ...})）。\n"
    "3. 可以按需多次、串行调用不同的 sub agent 和文件工具，每一步都基于已获得的结果继续推进。\n"
    "4. 所有子任务完成后，综合所有结果给出最终答案；给出最终回答时不要再调用任何工具。\n"
)

SUB_SYSTEM_PROMPTS: Dict[str, str] = {
    "researcher": (
        "你是 researcher sub agent，负责快速给出准确的事实、定义与实现思路调研结果。"
        "你可以使用文件工具（read_file/list_directory/search_files 等）查看本地文件来辅助"
        "调研，需要时再调用；直接输出调研结论（要点即可），不要寒暄。"
    ),
    "writer": (
        "你是 writer sub agent，负责根据任务要求完成具体产出（如编写代码、撰写文本）。"
        "你可以使用文件工具：需要参考本地文件时用 read_file，需要产出文件时用 write_file。"
        "直接输出完整产出，不要寒暄。"
    ),
    "reviewer": (
        "你是 reviewer sub agent，负责审查他人给出的实现或结论：找出错误、遗漏与可改进点，"
        "输出明确的审查意见。需要查看被审查的文件时可以使用 read_file 等文件工具。"
    ),
}

# 示例任务（文件操作 + 子代理分派）：同时演示「主 agent 用 read_file 读文件」与
# 「主 agent 自主决定调用子 agent」，不涉及代码执行。
DEFAULT_TASK = (
    "请完成下面这个「调研 + 文件操作」任务:\n"
    "1. 用 read_file 读取本仓库的 requirements.txt，了解项目依赖；\n"
    "2. 把「如何用最少的第三方依赖实现一个 OpenAI 兼容的 LLM 客户端？」交给 researcher 子代理调研；\n"
    "3. 综合两者结果输出：第一行「依赖清单：」+ requirements.txt 中的依赖；"
    "第二行「调研结论：」+ researcher 结论的要点。\n"
    "请自主决定调用哪些工具、以什么顺序完成；第 1、2 步不要跳过。"
)


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件，返回配置字典。"""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("缺少 PyYAML，请先执行: pip install -r requirements.txt") from exc
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise SystemExit(f"配置文件不存在: {path}")
    if not isinstance(cfg, dict):
        raise SystemExit(f"配置文件格式错误（应为 YAML 映射）: {path}")
    return cfg


def resolve(args: argparse.Namespace, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """取值顺序: 命令行参数 > 配置文件 > 默认值。"""
    v = getattr(args, key)
    if v is not None:
        return v
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def build_sub_agents(
    llm: LLMClient,
    trace: Trace,
    temperature: float = 0.7,
    max_iters: int = 4,
    tool_mode: str = "auto",
    max_tokens: int = 2048,
    prompts: Optional[Dict[str, str]] = None,
    tools: Optional[list] = None,
) -> Dict[str, Agent]:
    """按提示词字典构建 sub agent。

    sub agent 拥有通用文件工具（read_file/write_file 等），可以自己查文件、写文件；
    但 call_sub_agent 是 root_only 工具，Agent 构造时会被自动剥离——sub agent 不能
    再向下分派子任务（最多两级），避免无限递归。
    """
    prompts = prompts or SUB_SYSTEM_PROMPTS
    tools = tools if tools is not None else build_file_tools()
    return {
        name: Agent(
            name=name,
            system_prompt=prompt,
            llm=llm,
            tools=list(tools),
            trace=trace,
            max_iters=max_iters,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_mode=tool_mode,
            is_root=False,  # sub agent: 剥离 call_sub_agent，禁止再向下分派（最多两级）
        )
        for name, prompt in prompts.items()
    }


def make_call_sub_agent(sub_agents: Dict[str, Agent]) -> Tool:
    """构造 main agent 调用 sub agent 的工具。"""
    names = sorted(sub_agents)

    def call_sub_agent(name: str, task: str) -> str:
        agent = sub_agents.get(name)
        if agent is None:
            return f"ERROR: 未知子 agent {name!r}，可用: {names}"
        return agent.run(task)

    return Tool(
        name="call_sub_agent",
        description=(
            "把子任务分派给指定的 sub agent 执行，并返回它的结果文本。"
            "当任务需要专项能力（调研/实现/审查等）时调用它，而不是自己直接回答。"
            "name 取 sub agent 名（researcher/writer/reviewer）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": names,
                    "description": "要调用的 sub agent 名称",
                },
                "task": {
                    "type": "string",
                    "description": "交给该 sub agent 的具体任务描述",
                },
            },
            "required": ["name", "task"],
        },
        func=call_sub_agent,
        aliases=names,  # 子代理名可直接当工具名调用（模型常见行为），自动翻译
        root_only=True,  # 只允许 main agent 调用；sub agent 构造时会被自动剥离（最多两级）
    )


def build_main_agent(
    llm: LLMClient,
    sub_agents: Dict[str, Agent],
    trace: Trace,
    temperature: float = 0.7,
    max_iters: int = 10,
    tool_mode: str = "auto",
    system_prompt: Optional[str] = None,
    force_tool_call: bool = False,
    max_tokens: int = 2048,
    tools: Optional[list] = None,
) -> Agent:
    """main agent: 通用文件工具 + call_sub_agent（root_only）。"""
    tools = tools if tools is not None else build_file_tools()
    return Agent(
        name="main",
        system_prompt=system_prompt or MAIN_SYSTEM_PROMPT,
        llm=llm,
        tools=[*tools, make_call_sub_agent(sub_agents)],
        max_iters=max_iters,
        temperature=temperature,
        max_tokens=max_tokens,
        trace=trace,
        tool_mode=tool_mode,
        force_tool_call=force_tool_call,
        is_root=True,  # main agent: 唯一允许调用 call_sub_agent 的 root
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="main agent + sub agent，模型服务为 OpenAI 兼容接口（LMInfer/vLLM）"
    )
    parser.add_argument("--config", default=None, help="YAML 配置文件路径（见 example/ 目录）")
    parser.add_argument(
        "--base-url", default=None,
        help="vLLM 服务地址，默认 http://localhost:8000/v1（env: VLLM_BASE_URL）",
    )
    parser.add_argument("--model", default=None, help="模型名称，需与 vLLM 服务一致（env: VLLM_MODEL）")
    parser.add_argument("--api-key", default=None, help="API key，vLLM 默认不校验")
    parser.add_argument("--task", default=None, help="交给 main agent 的任务")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度")
    parser.add_argument("--max-iters", type=int, default=None, help="main agent 最大循环轮数（防止死循环）")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="单次模型请求的最大生成 token 数（默认 2048；Qwen3 带 think 时需留足空间）",
    )
    parser.add_argument(
        "--sub-max-iters", type=int, default=None,
        help="sub agent 最大循环轮数（默认 4：留出「调用文件工具后再作答」的轮数）",
    )
    parser.add_argument(
        "--tool-mode",
        choices=["auto", "native", "text"],
        default=None,
        help="工具调用模式: auto 先试原生 tool_calls, 400 时自动降级为文本协议; "
             "native 强制原生; text 强制文本协议（Mistral 等不支持原生 tool calling 的模型用）",
    )
    parser.add_argument(
        "--force-tool-call", action=argparse.BooleanOptionalAction, default=None,
        help="第一轮强制工具调用（默认关，主 agent 自主决定是否调用工具）: 开启后首轮 "
             "tool_choice=\"required\"(原生模式) / 首条回复必须输出 TOOL_CALL(文本模式), "
             "保证 main 先分派子任务而不是直接作答; --no-force-tool-call 关闭",
    )
    parser.add_argument(
        "--agent-mode", action=argparse.BooleanOptionalAction, default=None,
        help="LMInfer agent 模式: 请求携带 mode/trace/session_id, 服务端把主/子 agent 的"
             "多次调用关联到同一个会话并统计消耗(默认开; --no-agent-mode 关闭以兼容 vLLM)",
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="Qwen3 等模型的 thinking 开关(随请求体发送): --enable-thinking 开, "
             "--no-enable-thinking 关; 都不传则用服务端默认",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="不实时打印 trace，只输出最终答案与链式摘要",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    base_url = resolve(args, cfg, "base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    model = resolve(args, cfg, "model", os.environ.get("VLLM_MODEL"))
    api_key = resolve(args, cfg, "api_key", "EMPTY")
    task = resolve(args, cfg, "task", DEFAULT_TASK)
    temperature = resolve(args, cfg, "temperature", 0.7)
    max_iters = resolve(args, cfg, "max_iters", 10)
    max_tokens = resolve(args, cfg, "max_tokens", 2048)
    sub_max_iters = resolve(args, cfg, "sub_max_iters", 4)
    tool_mode = resolve(args, cfg, "tool_mode", "auto")
    force_tool_call = resolve(args, cfg, "force_tool_call", False)
    agent_mode = resolve(args, cfg, "agent_mode", True)
    enable_thinking = resolve(args, cfg, "enable_thinking", None)
    quiet = args.quiet or bool(cfg.get("quiet", False))

    if not model:
        parser.error("未指定模型: 用 --model、配置文件的 model 字段或环境变量 VLLM_MODEL")

    # sub agents 与 main 提示词均可由配置文件覆盖（无需改代码）
    sub_prompts = cfg.get("sub_agents") if isinstance(cfg.get("sub_agents"), dict) else SUB_SYSTEM_PROMPTS
    main_prompt = cfg.get("main_system_prompt") or MAIN_SYSTEM_PROMPT

    llm = LLMClient(
        base_url=base_url, api_key=api_key, model=model,
        agent_mode=agent_mode, enable_thinking=enable_thinking,
    )
    trace = Trace(verbose=not quiet)
    sub_agents = build_sub_agents(
        llm, trace, temperature, sub_max_iters, tool_mode,
        max_tokens=max_tokens, prompts=sub_prompts,
    )
    main_agent = build_main_agent(
        llm, sub_agents, trace, temperature, max_iters, tool_mode,
        system_prompt=main_prompt, force_tool_call=force_tool_call,
        max_tokens=max_tokens,
    )

    src = f" (配置文件: {args.config})" if args.config else ""
    print(f"模型: {model} @ {base_url}{src}，agent 模式: {'开' if agent_mode else '关'}，"
          f"强制首轮工具调用: {'开' if force_tool_call else '关'}")
    answer = main_agent.run(task)

    print("\n========== 最终答案 ==========")
    print(answer)


if __name__ == "__main__":
    main()
