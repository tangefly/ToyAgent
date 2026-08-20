"""示例入口：1 个 main agent + 若干 sub agent，模型服务为 LMInfer（OpenAI 兼容接口）。

用法：
    python main.py --config example/config-lminfer.yaml
    python main.py --base-url http://localhost:8000/v1 --model Qwen3-0.6B

默认以 LMInfer 的 agent 模式请求（mode/trace/session_id，服务端按会话统计），
用 --no-agent-mode 关闭后可兼容 vLLM 等普通 OpenAI 服务。

配置优先级: 命令行参数 > 配置文件 > 环境变量(VLLM_BASE_URL/VLLM_MODEL) > 内置默认
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from agent import Agent, Tool, Trace
from llm import LLMClient

MAIN_SYSTEM_PROMPT = (
    "你是 main agent，负责接收用户任务、分析问题并规划执行步骤。\n"
    "可用的 sub agent 只有三个: researcher（调研）、writer（实现/写作）、reviewer（审查）。\n"
    "执行规则:\n"
    "1. 收到任务后，第一步必须先调用 call_sub_agent 工具把任务分派给合适的 sub agent，"
    "禁止跳过工具直接回答；在拿到工具结果之前不得输出最终回答。\n"
    "2. call_sub_agent 的参数 name 必须精确取 researcher/writer/reviewer 之一，"
    "task 写清楚交给该 sub agent 的具体问题。"
    "（也可以直接用子代理名作为工具名调用，如 writer({\"task\": ...})，效果相同。）\n"
    "3. 可以按需多次、串行调用不同的 sub agent，每一步都基于已获得的结果继续推进。\n"
    "4. 所有子任务完成后，综合所有 sub agent 的输出，给出最终答案；给出最终回答时不要调用任何工具。\n"
)

SUB_SYSTEM_PROMPTS: Dict[str, str] = {
    "researcher": (
        "你是 researcher sub agent，负责快速给出准确的事实、定义与实现思路调研结果。"
        "直接输出调研结论（要点即可），不要调用任何工具，不要寒暄。"
    ),
    "writer": (
        "你是 writer sub agent，负责根据任务要求完成具体产出（如编写代码、撰写文本）。"
        "直接输出完整产出，不要调用任何工具。"
    ),
    "reviewer": (
        "你是 reviewer sub agent，负责审查他人给出的实现或结论：找出错误、遗漏与可改进点，"
        "输出明确的审查意见。不要调用任何工具。"
    ),
}

# 示例任务（纯推理分派）：模型必须调用 call_sub_agent 让 researcher 完成推理，
# 再复述其结果——不涉及代码执行，重点演示 tool call 链路本身。
DEFAULT_TASK = (
    "请先调用 call_sub_agent 工具，把下面这个问题交给 researcher 子代理调研：\n"
    "「快速排序的平均时间复杂度是多少？用一两句话说明原因。」\n"
    "拿到 researcher 的结果后，先输出一行「researcher 调研结果：」，"
    "再原样复述它的答案，除此之外不要输出任何其他内容。"
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
    max_iters: int = 1,
    prompts: Optional[Dict[str, str]] = None,
) -> Dict[str, Agent]:
    """按提示词字典构建 sub agent（默认用内置的 SUB_SYSTEM_PROMPTS）。"""
    prompts = prompts or SUB_SYSTEM_PROMPTS
    return {
        name: Agent(
            name=name,
            system_prompt=prompt,
            llm=llm,
            trace=trace,
            max_iters=max_iters,
            temperature=temperature,
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
    )


def build_main_agent(
    llm: LLMClient,
    sub_agents: Dict[str, Agent],
    trace: Trace,
    temperature: float = 0.7,
    max_iters: int = 10,
    tool_mode: str = "auto",
    system_prompt: Optional[str] = None,
    force_tool_call: bool = True,
) -> Agent:
    return Agent(
        name="main",
        system_prompt=system_prompt or MAIN_SYSTEM_PROMPT,
        llm=llm,
        tools=[make_call_sub_agent(sub_agents)],
        max_iters=max_iters,
        temperature=temperature,
        trace=trace,
        tool_mode=tool_mode,
        force_tool_call=force_tool_call,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="main agent + sub agent，模型服务为 LMInfer（OpenAI 兼容接口）"
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
        "--sub-max-iters", type=int, default=None,
        help="sub agent 最大循环轮数（默认 1：直接作答）",
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
        help="第一轮强制工具调用(默认开): 首轮 tool_choice=\"required\"(原生模式) / "
             "首条回复必须输出 TOOL_CALL(文本模式), 保证 main 先分派子任务而不是直接作答; "
             "--no-force-tool-call 关闭",
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
    sub_max_iters = resolve(args, cfg, "sub_max_iters", 1)
    tool_mode = resolve(args, cfg, "tool_mode", "auto")
    force_tool_call = resolve(args, cfg, "force_tool_call", True)
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
    sub_agents = build_sub_agents(llm, trace, temperature, sub_max_iters, prompts=sub_prompts)
    main_agent = build_main_agent(
        llm, sub_agents, trace, temperature, max_iters, tool_mode,
        system_prompt=main_prompt, force_tool_call=force_tool_call,
    )

    src = f" (配置文件: {args.config})" if args.config else ""
    print(f"模型: {model} @ {base_url}{src}，agent 模式: {'开' if agent_mode else '关'}，"
          f"强制首轮工具调用: {'开' if force_tool_call else '关'}")
    answer = main_agent.run(task)

    print("\n========== 最终答案 ==========")
    print(answer)
    trace.dump()
    if agent_mode and llm.session_id:
        print(f"\nLMInfer agent 会话: {llm.session_id}"
              "（服务端 GET /v1/agent/sessions 可查该会话的请求数与 token 消耗）")


if __name__ == "__main__":
    main()
