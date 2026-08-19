"""示例入口：1 个 main agent + 若干 sub agent，模型服务为 vLLM（OpenAI 兼容接口）。

用法：
    python main.py --config example/config-mistral.yaml
    python main.py --base-url http://localhost:8000/v1 --model Mistral-7B-Instruct

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
    "执行规则:\n"
    "1. 先分析任务目标，判断需要拆解出哪些子任务。\n"
    "2. 需要专项能力（调研、实现、审查等）时，调用 call_sub_agent 工具把子任务分派给对应的 sub agent，等待其返回结果。\n"
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

DEFAULT_TASK = (
    "请写一个 Python 函数 is_prime(n) 判断整数 n 是否为素数，"
    "然后分别判断 97 和 91，最后给出函数代码和两个数的判断结果。"
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
    )


def build_main_agent(
    llm: LLMClient,
    sub_agents: Dict[str, Agent],
    trace: Trace,
    temperature: float = 0.7,
    max_iters: int = 10,
    tool_mode: str = "auto",
    system_prompt: Optional[str] = None,
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="main agent + sub agent，模型服务为 vLLM（OpenAI 兼容接口）"
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
    quiet = args.quiet or bool(cfg.get("quiet", False))

    if not model:
        parser.error("未指定模型: 用 --model、配置文件的 model 字段或环境变量 VLLM_MODEL")

    # sub agents 与 main 提示词均可由配置文件覆盖（无需改代码）
    sub_prompts = cfg.get("sub_agents") if isinstance(cfg.get("sub_agents"), dict) else SUB_SYSTEM_PROMPTS
    main_prompt = cfg.get("main_system_prompt") or MAIN_SYSTEM_PROMPT

    llm = LLMClient(base_url=base_url, api_key=api_key, model=model)
    trace = Trace(verbose=not quiet)
    sub_agents = build_sub_agents(llm, trace, temperature, sub_max_iters, prompts=sub_prompts)
    main_agent = build_main_agent(
        llm, sub_agents, trace, temperature, max_iters, tool_mode, system_prompt=main_prompt
    )

    src = f" (配置文件: {args.config})" if args.config else ""
    print(f"模型: {model} @ {base_url}{src}")
    answer = main_agent.run(task)

    print("\n========== 最终答案 ==========")
    print(answer)
    trace.dump()


if __name__ == "__main__":
    main()
