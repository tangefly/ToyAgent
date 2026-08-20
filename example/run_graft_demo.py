#!/usr/bin/env python3
"""--reuse-agent-kv-append 效果验证脚本: 运行多级 agent 链并汇总 KV 复用统计.

与 example/config-qwen3-graft-demo.yaml 配套: main -> researcher -> main
-> writer -> main -> reviewer -> main, 三级子代理流水线, 每次 main 返回
都会触发"子 agent 输出 KV 拼接"。

用法(先启动带 --reuse-agent-kv-append 的 LMInfer 服务):
    python example/run_graft_demo.py --config example/config-qwen3-graft-demo.yaml \
        --no-enable-thinking
    (不带 --config 也可, 用内置的同一任务与默认子代理提示词)

输出:
    1. 每次模型请求的 trace / prompt tokens / 复用 token / 跳过 prefill 比例 ——
       复用 token 来自请求响应中的实验观测字段 reused_prompt_tokens(LMInfer
       agent 模式返回, 含"拼接的子 agent 输出 KV");
    2. 服务端会话统计(总复用 token 与比例, GET /v1/agent/sessions)。

判断标准:
    - 子 agent 请求(researcher/writer/reviewer 自身): 复用 ~0, 全量 prefill;
    - main 返回请求(trace 以 main 结尾): 复用 ≈ main 历史 KV + 上个子 agent
      输出正文 KV, 剩余 prefill 只有 role 标记等十几个 token —— 这就是
      --reuse-agent-kv-append 的效果; 只开 --reuse-agent-kv 时, 这些请求
      只能复用 main 历史, 剩余 prefill 会多出约一个子输出的长度。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from llm import LLMClient
from main import (
    SUB_SYSTEM_PROMPTS,
    build_main_agent,
    build_sub_agents,
    load_config,
    resolve,
    Trace,
)

DEMO_TASK = (
    "请完成「Python 单词频统计工具」的完整设计与实现。必须严格按顺序完成以下 4 步,"
    "任何一步都不能跳过, 每一步都必须先拿到上一步的结果再继续:\n"
    "第 1 步:调用 call_sub_agent(name=\"researcher\"),让它调研:用 Python 统计一段英文"
    "文本中每个单词出现次数, 最高效的做法是什么(collections.Counter / 普通 dict / 其他),"
    "以及处理大小写、标点符号时需要注意什么。\n"
    "第 2 步:拿到 researcher 的结果后, 调用 call_sub_agent(name=\"writer\"),把调研结论"
    "概要连同任务「根据以上结论, 编写一个完整的 Python 函数 "
    "word_frequency(text: str) -> list[tuple[str, int]], 返回按频次降序排列的 "
    "(单词, 次数) 列表, 要求统一小写、去除标点, 包含必要的注释」一起交给 writer 实现。\n"
    "第 3 步:拿到 writer 的结果后, 调用 call_sub_agent(name=\"reviewer\"),把 writer "
    "输出的完整代码原文(一字不改)连同任务「审查下面这份代码的正确性、健壮性与可改进点,"
    "输出明确的审查意见」一起交给 reviewer 审查。\n"
    "第 4 步:拿到 reviewer 的结果后, 综合三步结果输出最终答案, 格式必须为:\n"
    "调研结论: ...(researcher 结论要点)\n"
    "最终代码: ...(writer 的完整代码)\n"
    "审查意见: ...(reviewer 意见要点)\n"
    "注意: 禁止跳过任何一步, 禁止在拿到 reviewer 结果之前输出最终答案。"
)


class RecordingLLMClient(LLMClient):
    """在每次模型请求后记录 usage 与 KV 复用统计(列表顺序即请求发生顺序)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.request_stats: list[dict] = []

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.7,
             max_tokens=2048, trace=None) -> dict:
        message = super().chat(
            messages, tools=tools, tool_choice=tool_choice,
            temperature=temperature, max_tokens=max_tokens, trace=trace,
        )
        self.request_stats.append({
            "trace": list(trace or []),
            "prompt_tokens": int(self.last_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(self.last_usage.get("completion_tokens", 0)),
            "reused_tokens": int(self.last_reused_tokens or 0),
        })
        return message


def main() -> None:
    parser = argparse.ArgumentParser(
        description="--reuse-agent-kv-append 效果验证(多级 agent 链: researcher -> writer -> reviewer)"
    )
    parser.add_argument("--config", default=None, help="YAML 配置文件路径(见 example/ 目录)")
    parser.add_argument("--base-url", default=None, help="LMInfer 服务地址, 默认 http://localhost:8000/v1")
    parser.add_argument("--model", default=None, help="模型名称, 需与 --served-model-name 一致")
    parser.add_argument("--api-key", default=None, help="API key, LMInfer 默认不校验")
    parser.add_argument("--task", default=None, help="覆盖任务(默认用内置三级子代理任务)")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度(默认 0.3)")
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
        help="Qwen3 等模型的 thinking 开关: --enable-thinking 开, --no-enable-thinking 关; "
             "都不传则用服务端默认",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    base_url = resolve(args, cfg, "base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    model = resolve(args, cfg, "model", os.environ.get("VLLM_MODEL"))
    if not model:
        parser.error("未指定模型: 用 --model、配置文件的 model 字段或环境变量 VLLM_MODEL")
    api_key = resolve(args, cfg, "api_key", "EMPTY")
    temperature = resolve(args, cfg, "temperature", 0.3)
    task = args.task or cfg.get("task") or DEMO_TASK

    llm = RecordingLLMClient(
        base_url=base_url, api_key=api_key, model=model,
        agent_mode=True, enable_thinking=args.enable_thinking,
    )
    trace = Trace(verbose=True)
    sub_prompts = (cfg.get("sub_agents")
                   if isinstance(cfg.get("sub_agents"), dict) else SUB_SYSTEM_PROMPTS)
    sub_agents = build_sub_agents(llm, trace, temperature, max_iters=1, prompts=sub_prompts)
    main_agent = build_main_agent(llm, sub_agents, trace, temperature, max_iters=10)

    print(f"模型: {model} @ {base_url}，任务: 三级子代理流水线 "
          f"(researcher -> writer -> reviewer)\n")
    answer = main_agent.run(task)
    print("\n========== 最终答案 ==========")
    print(answer)
    trace.dump()

    # ---- KV 复用验证: 每次请求的复用情况 ----
    print("\n========== KV 复用验证 (reused_prompt_tokens = 跳过 prefill 的 token 数) ==========")
    print(f"{'#':<3}{'trace':<30}{'prompt':>7}{'复用':>7}{'跳过%':>7}{'输出':>7}")
    print("-" * 61)
    for i, s in enumerate(llm.request_stats, 1):
        p, r = s["prompt_tokens"], s["reused_tokens"]
        pct = 100.0 * r / p if p else 0.0
        print(f"{i:<3}{str(s['trace']):<30}{p:>7}{r:>7}{pct:>6.0f}%{s['completion_tokens']:>7}")
    total_p = sum(s["prompt_tokens"] for s in llm.request_stats)
    total_r = sum(s["reused_tokens"] for s in llm.request_stats)
    print("-" * 61)
    print(f"合计: prompt {total_p} tok, 复用 {total_r} tok, "
          f"跳过 {100.0 * total_r / max(total_p, 1):.0f}% prefill")

    if llm.session_id:
        try:
            data = requests.get(f"{base_url}/agent/sessions", timeout=10).json()
        except requests.ConnectionError:
            data = {"data": []}
        row = next((s for s in data.get("data", [])
                    if s["session_id"] == llm.session_id), None)
        if row:
            print(f"服务端会话 {llm.session_id}: {row['request_count']} 次请求, "
                  f"KV 复用 {row['kv_reuse_count']} 次 / {row['kv_reuse_tokens']} tok")

    print("\n判断标准:")
    print("  子 agent 请求(researcher/writer/reviewer): 复用 ~0, 全量 prefill;")
    print("  main 返回请求(trace 以 main 结尾): 复用 ≈ main 历史 + 上个子 agent 输出正文,")
    print("  剩余 prefill 只有 role 标记等十几个 token —— 这就是 --reuse-agent-kv-append")
    print("  的效果(只开 --reuse-agent-kv 时这些请求会多 prefill 一个子输出的长度);")
    print("  子输出 KV 拼接的直接证据在服务端日志: 每次 main 返回时打印")
    print("  「拼接子 agent 输出 KV N tok(位置 a..b) + 复用 main 历史 KV M tok」。")


if __name__ == "__main__":
    main()
