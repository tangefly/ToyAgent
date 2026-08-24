#!/usr/bin/env python3
"""用重构后的 ToyAgent 跑 GAIA_Text 数据集（默认只跑 data/task_1.jsonl 第一条）。

适配重构后的 API 要点:
- Tool 类已从 agent.py 移到 tools.py（from tools import Tool）;
- 主/子 agent 都自带文件工具（read_file/write_file/list_directory/search_files）:
  附件由模型自己调用 read_file 读取——main 可以自己读，也可以分派给 researcher 读，
  由模型自主决定（force_tool_call 默认关，主 agent 自主决定是否调用子代理/工具）;
- build_main_agent / build_sub_agents 现在接受 max_tokens 参数（注意: 重构后
  max_tokens 在构造 Agent 时就被 ChatBackend 捕获，事后改 main_agent.max_tokens 无效）。

数据布局（ToyAgent 目录下）:
- data/task_1.jsonl              第一条任务（默认）
- data/gaia_document_only.jsonl  全量 23 条
- data/documents/                附件树（与数据集内 file_path 相对路径一致）

用法:
    cd /home/tanger/workspace/ToyAgent
    python3 gaia_test.py --model Qwen3-8B --no-agent-mode --limit 1
    python3 gaia_test.py --config example/config-gaia.yaml --limit 1
    python3 gaia_test.py --config example/config-gaia.yaml \
        --data data/gaia_document_only.jsonl --out gaia_results/toyagent_gaia_all.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from agent import Trace
from llm import LLMClient
from main import (
    SUB_SYSTEM_PROMPTS,
    build_main_agent,
    build_sub_agents,
    load_config,
)

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "task_1.jsonl"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "gaia_results"

# GAIA 任务全英文：sub agent 提示词统一英文，避免中英混杂导致回答失真。
BENCH_SUB_SYSTEM_PROMPTS = dict(SUB_SYSTEM_PROMPTS)
BENCH_SUB_SYSTEM_PROMPTS["researcher"] = (
    "You are the researcher sub-agent. Give accurate, concise research results for the task. "
    "If the task specifies attachment file path(s), call the read_file tool with exactly "
    "those paths and answer strictly based on the file content; never guess file content or "
    "invent paths. If the task mentions no file, answer directly from the task text. "
    "After reading, output your conclusion directly (key points). Do not call other tools. "
    "Do not echo the tool result text — output only your own answer. No small talk."
)
BENCH_SUB_SYSTEM_PROMPTS["writer"] = (
    "You are the writer sub-agent. Produce the concrete deliverable requested "
    "(code, text, or a direct answer). Output the complete result directly. "
    "Do not call tools. No small talk."
)
BENCH_SUB_SYSTEM_PROMPTS["reviewer"] = (
    "You are the reviewer sub-agent. Review the given work for errors, omissions and "
    "improvements; output clear review comments. Do not call tools."
)

# main agent 系统提示词：自主决定——自己用 read_file 读附件，或分派给 researcher。
MAIN_SYSTEM_PROMPT = (
    "You are the main agent. You receive a user task (a question, possibly with attachment "
    "files) and you must plan and execute steps.\n"
    "You have file tools: read_file / write_file / list_directory / search_files.\n"
    "Available sub agents: researcher (research / read documents / answer), "
    "writer (implementation / writing), reviewer (review).\n"
    "Rules:\n"
    "1. Analyze the task first. If it involves attachment files, you MUST read them before "
    "answering — you may read them yourself with read_file, or delegate the reading to the "
    "researcher sub-agent via call_sub_agent. You decide which is more appropriate; do not "
    "answer about file content you have not read.\n"
    "2. The name parameter must be exactly one of researcher/writer/reviewer, and task must "
    "state clearly what that sub agent should do. "
    "(You may also call a sub agent directly by its name, e.g. researcher({\"task\": ...}) — same effect.)\n"
    "3. You may call tools and sub agents several times, serially, building on previous results.\n"
    "4. If a sub agent fails or returns an error, stop re-delegating and answer directly "
    "from what you already know.\n"
    "5. After all sub-tasks complete, give the final answer. The final reply must contain "
    "ONLY the bare answer: a number, a name, or a short phrase. Absolutely NO lead-in "
    "sentence (no 'The answer is', no 'The odds are', no 'The type is'), no quotes, no "
    "explanation, no markdown. Do not call any tool when giving the final answer.\n"
)

# --delegate-first 时使用：首步必须分派给 researcher，演示「子 agent 自己读文件」。
DELEGATE_MAIN_SYSTEM_PROMPT = (
    "You are the main agent. You receive a user task (a question, possibly with attachment "
    "files) and you must plan and execute steps.\n"
    "Rules:\n"
    "1. On the first step you MUST call the call_sub_agent tool to delegate the task to a "
    "suitable sub agent. Never answer directly without a tool call; never output a final "
    "answer before you have tool results.\n"
    "2. The name parameter must be exactly one of researcher/writer/reviewer, and task must "
    "state clearly what that sub agent should do. "
    "(You may also call a sub agent directly by its name, e.g. researcher({\"task\": ...}) — same effect.)\n"
    "3. If the task has attachment files, delegate the reading to the researcher: the "
    "attachment paths are included in the delegated task, and the researcher will call "
    "read_file itself. Do NOT read files yourself.\n"
    "4. You may call sub agents several times, serially, building on previous results.\n"
    "5. After all sub-tasks complete, give the final answer. The final reply must contain "
    "ONLY the bare answer: a number, a name, or a short phrase. Absolutely NO lead-in "
    "sentence, no quotes, no explanation, no markdown. Do not call any tool when giving the "
    "final answer.\n"
)


class _AbortTask(BaseException):
    """重复分派打断信号：逃过 Tool/Agent 层的 except Exception 兜底，直接终止该任务。"""


class UsageTrackingClient(LLMClient):
    """在 LLMClient 基础上累计每次请求的 token 用量（任务级总消耗）。

    关闭 thinking 时（vLLM, agent_mode=False）走 chat_template_kwargs 注入;
    LMInfer（agent_mode=True）沿用顶层 enable_thinking 字段（LLMClient 已发送）。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.total_usage: Dict[str, int] = {}

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if (
            not self.agent_mode
            and self.enable_thinking is False
            and "chat_template_kwargs" not in payload
        ):
            payload = dict(payload)
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return super()._post(payload)

    def chat(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        msg = super().chat(*args, **kwargs)
        for k, v in (self.last_usage or {}).items():
            if isinstance(v, (int, float)):
                self.total_usage[k] = self.total_usage.get(k, 0) + int(v)
        return msg


# ---------- 判分（GAIA 风格） ----------


def normalize_answer(s: str) -> str:
    """答案归一化: 小写、去冠词、去标点、压空白（中文按字保留，只去标点空白）。"""
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w一-鿿\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def cleanup_answer(answer: str) -> str:
    """清理模型原始输出: 剥掉未闭合 think 块、[工具结果] 残留与答案外包裹。"""
    if "<think>" in answer and "</think>" not in answer:
        answer = answer.split("<think>", 1)[0]
    while answer.startswith("[工具结果]"):
        idx = answer.find("]:")
        if idx == -1:
            break
        answer = answer[idx + 2:].strip()
    answer = re.sub(r"^[*_ ]*(final answer|answer)[:*_ ]*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^\*\*(.+)\*\*$", r"\1", answer).strip()
    return answer


def exact_match(answer: str, gold: Any) -> bool:
    """与 gold_answer 精确匹配（支持多候选，用竖线分隔或列表）。"""
    if isinstance(gold, list):
        candidates = [str(g) for g in gold]
    else:
        candidates = [c for c in str(gold).split("|")]
    a = normalize_answer(answer)
    if not a:
        return False
    return any(a == normalize_answer(c) for c in candidates)


def relaxed_match(answer: str, gold: Any) -> bool:
    """宽松判分: 严格匹配失败后，再尝试剥 markdown / 取最后一行 / 取引号内值。"""
    if exact_match(answer, gold):
        return True
    a = cleanup_answer(answer)
    cands = [a]
    a2 = re.sub(r"\*\*(.+?)\*\*", r"\1", a)
    cands.append(a2)
    lines = [ln.strip() for ln in a2.splitlines() if ln.strip()]
    if lines:
        cands.append(lines[-1])
    quoted = re.findall(r"“([^”]+)”|“([^”]+)”|\"([^\"]+)\"|\\boxed\{([^}]+)\}", a2)
    for grp in quoted:
        for frag in grp:
            if frag:
                cands.append(frag)
    return any(exact_match(c, gold) for c in cands if c)


def build_task_prompt(rec: Dict[str, Any]) -> str:
    """任务 prompt: 问题 + 附件文件清单（内容由模型自己用 read_file 读取）。"""
    parts = [f"【问题】\n{rec['question']}"]
    paths = rec.get("file_path") or []
    if paths:
        parts.append(
            "【附件】\n本题附带以下文件，回答前必须先调用 read_file 工具读取内容"
            "（path 参数原样传下面的路径）:\n" + "\n".join(f"- {p}" for p in paths)
        )
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用重构后的 ToyAgent 跑 GAIA_Text 数据集"
    )
    parser.add_argument("--config", default=None, help="ToyAgent 风格 YAML 配置")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="数据集 JSONL 路径")
    parser.add_argument("--base-url", default=None, help="服务地址（默认 http://localhost:8000/v1）")
    parser.add_argument("--model", default=None, help="模型名称（env: VLLM_MODEL，必填）")
    parser.add_argument("--api-key", default=None, help="API key，默认 EMPTY")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="单次生成上限（默认 4096；Qwen3 带 think 时需留足空间）")
    parser.add_argument("--max-iters", type=int, default=None, help="main agent 最大循环轮数")
    parser.add_argument("--sub-max-iters", type=int, default=None, help="sub agent 最大循环轮数")
    parser.add_argument("--tool-mode", choices=["auto", "native", "text"], default=None)
    parser.add_argument("--force-tool-call", action=argparse.BooleanOptionalAction, default=None,
                        help="首轮强制工具调用（默认关：主 agent 自主决定是否调用子代理/工具）")
    parser.add_argument("--delegate-first", action="store_true",
                        help="主提示词改为「首步必须分派给子代理」，演示子 agent 自己调用 read_file")
    parser.add_argument("--agent-mode", action=argparse.BooleanOptionalAction, default=None,
                        help="LMInfer agent 模式（默认开；vLLM 用 --no-agent-mode）")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="Qwen3 thinking 开关（默认关；--enable-thinking 打开）")
    parser.add_argument("--limit", type=int, default=None, help="最多跑多少条")
    parser.add_argument("--offset", type=int, default=0, help="从第几条开始")
    parser.add_argument("--only-ids", default=None, help="只跑指定 task_id（逗号分隔）")
    parser.add_argument("--out", default=None, help="结果 JSONL 路径（默认 gaia_results/ 下）")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件里已存在的任务")
    parser.add_argument("--quiet", action="store_true", help="不实时打印每步 trace")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    def resolve(key: str, default: Any) -> Any:
        v = getattr(args, key)
        if v is not None:
            return v
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return default

    base_url = resolve("base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    model = resolve("model", os.environ.get("VLLM_MODEL"))
    api_key = resolve("api_key", "EMPTY")
    temperature = resolve("temperature", 0.3)
    max_tokens = resolve("max_tokens", 4096)
    max_iters = resolve("max_iters", 10)
    sub_max_iters = resolve("sub_max_iters", 3)  # researcher: 1 轮 read_file + 1 轮作答
    tool_mode = resolve("tool_mode", "auto")
    force_tool_call = resolve("force_tool_call", False)  # 默认关: 主 agent 自主决定
    agent_mode = resolve("agent_mode", True)
    enable_thinking = resolve("enable_thinking", False)
    if not model:
        parser.error("未指定模型: 用 --model、配置文件 model 字段或环境变量 VLLM_MODEL")

    data_path = Path(args.data)
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_ids:
        wanted = {x.strip() for x in args.only_ids.split(",") if x.strip()}
        records = [r for r in records if r["task_id"] in wanted]

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"toyagent_gaia_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set = set()
    if args.resume and out_path.exists():
        done_ids = {json.loads(l)["task_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"resume: 跳过 {len(done_ids)} 个已完成任务 -> {out_path}")

    print(f"模型: {model} @ {base_url} | tool_mode: {tool_mode} | agent_mode: {agent_mode} | "
          f"force_tool_call: {force_tool_call} | 任务总数: {len(records)} | 输出: {out_path}")

    results: List[Dict[str, Any]] = []
    selected = records[args.offset: args.limit if args.limit is None else args.offset + args.limit]
    for i, rec in enumerate(selected):
        if rec["task_id"] in done_ids:
            continue
        # 每个任务独立 LLMClient + Trace: agent 模式下独占一个会话，互不污染
        llm = UsageTrackingClient(
            base_url=base_url, api_key=api_key, model=model,
            agent_mode=agent_mode, enable_thinking=enable_thinking,
        )
        trace = Trace(verbose=not args.quiet)
        # 重构后: sub agent 默认自带文件工具（read_file 等），无需额外注入
        sub_agents = build_sub_agents(
            llm, trace, temperature, sub_max_iters, tool_mode,
            max_tokens=max_tokens, prompts=BENCH_SUB_SYSTEM_PROMPTS,
        )
        system_prompt = (
            DELEGATE_MAIN_SYSTEM_PROMPT if args.delegate_first else MAIN_SYSTEM_PROMPT
        )
        main_agent = build_main_agent(
            llm, sub_agents, trace, temperature, max_iters, tool_mode,
            system_prompt=system_prompt, force_tool_call=force_tool_call,
            max_tokens=max_tokens,
        )
        # call_sub_agent 包装: ① 问题原文+附件路径机械注入（模型分派时常丢上下文）;
        # ② 防重复分派
        _question = rec["question"]
        _att_paths = rec.get("file_path") or []
        _call_sub = main_agent.tools["call_sub_agent"]
        _orig_call = _call_sub.func
        _seen: Dict[tuple, int] = {}

        def _guarded_call(name: str, task: str) -> str:
            if _question not in task and _question[:40] not in task:
                task = f"Question: {_question}\n\nSub-task: {task}"
            if _att_paths and not any(p in task for p in _att_paths):
                task = (f"First call the read_file tool to read the attachment file(s) "
                        f"(use exactly these paths): {', '.join(_att_paths)}.\n\n{task}")
            key = (name, task)
            _seen[key] = _seen.get(key, 0) + 1
            if _seen[key] >= 3:
                raise _AbortTask(f"same sub-task delegated {_seen[key]} times")
            if _seen[key] >= 2:
                return ("ERROR: This sub-task was already attempted without progress. "
                        "Stop re-delegating and give the final answer directly from what "
                        "you already know (including any attachment content you read).")
            return _orig_call(name, task)

        _call_sub.func = _guarded_call
        task_prompt = build_task_prompt(rec)
        print(f"\n[{i + 1}/{len(selected)}] task_id={rec['task_id'][:8]} level={rec['level']} "
              f"附件: {len(_att_paths)} 个")

        start = time.time()
        try:
            answer = main_agent.run(task_prompt)
            error = None
        except _AbortTask as exc:
            answer, error = "", f"aborted: {exc}"
        except Exception as exc:
            answer, error = "", repr(exc)
            print(f"  !! 任务失败: {error}")
        elapsed = time.time() - start
        answer = cleanup_answer(answer)

        hit = exact_match(answer, rec["gold_answer"])
        hit_relaxed = relaxed_match(answer, rec["gold_answer"])
        result = {
            "task_id": rec["task_id"],
            "level": rec["level"],
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
            "answer": answer,
            "exact_match": hit,
            "exact_match_relaxed": hit_relaxed,
            "error": error,
            "elapsed_s": round(elapsed, 1),
            "session_id": llm.session_id,
            "total_usage": llm.total_usage,
            "trace_steps": trace.steps,
            "chain": trace.chain,
        }
        results.append(result)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"  -> {'✓' if hit else '✗'} exact_match: {hit} | {elapsed:.1f}s | "
              f"调用链: {' -> '.join(trace.chain) if trace.chain else 'main'}")
        print(f"  gold:   {rec['gold_answer']!r}")
        print(f"  answer: {answer[:300]!r}")

    if not results:
        print("没有新任务执行。")
        return
    total = len(results)
    correct = sum(1 for r in results if r["exact_match"])
    correct_rx = sum(1 for r in results if r.get("exact_match_relaxed"))
    print(f"\n========== 摘要 ({out_path}) ==========")
    print(f"任务数: {total} | exact_match: {correct}/{total} ({correct / total:.1%}) | "
          f"relaxed: {correct_rx}/{total} ({correct_rx / total:.1%})")
    delegated = sum(1 for r in results if len(r.get("chain") or []) > 1)
    print(f"分派子代理: {delegated}/{total} | 平均链长: "
          f"{sum(len(r.get('chain') or []) for r in results) / total:.1f}")
    tok = sum(r["total_usage"].get("total_tokens", 0) for r in results)
    print(f"总 token 消耗: {tok} | 平均耗时: {sum(r['elapsed_s'] for r in results) / total:.1f}s")


if __name__ == "__main__":
    main()
