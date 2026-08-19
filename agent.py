"""核心 agent：基于 OpenAI 兼容接口（如 vLLM）的 tool-calling agent 循环。

链式结构：main agent 通过 call_sub_agent 工具调用 sub agent，
形成 main -> sub -> main -> sub -> ... -> main -> out 的调用链，
每一步都以 [agent名] 前缀写入 Trace，实时可见整条链路。

两种工具调用模式：
- native: OpenAI 原生 tool_calls 字段（需要模型与 vLLM 支持，如 Qwen2.5 等）；
- text:   文本协议（回复中输出 TOOL_CALL: {json}，可一次多个），任何模型可用
          （如 Mistral-7B-Instruct-v0.2，原生不支持 tool calling）；
- auto:   默认。先尝试 native，若服务返回 400 则自动切换到 text。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import requests

from llm import LLMClient

TOOL_CALL_MARKER = "TOOL_CALL:"


class Trace:
    """记录整条 agent 链上每一步日志：可实时打印，也可在结束时输出摘要。"""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.steps: List[Dict[str, str]] = []

    def log(self, agent: str, msg: str) -> None:
        self.steps.append({"agent": agent, "msg": msg})
        if self.verbose:
            print(f"[{agent}] {msg}")

    def dump(self) -> None:
        """输出压缩版的链式 trace 摘要（main -> sub -> ... -> main -> out）。"""
        print("\n===== 链式 trace (main -> sub -> ... -> main -> out) =====")
        for s in self.steps:
            print(f"  {s['agent']}: {s['msg']}")
        print("============================================================")


class Tool:
    """一个可被 LLM 调用的工具：可调用对象 + OpenAI 函数 schema。"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: Dict[str, Any]) -> Any:
        return self.func(**arguments)


class Agent:
    """tool-calling agent 循环。

    流程：向 LLM 发消息 -> 若返回工具调用则执行并把结果回填 -> 重复，
    直到 LLM 给出不含工具调用的最终回答。
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        tools: Optional[List[Tool]] = None,
        max_iters: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        trace: Optional[Trace] = None,
        tool_mode: str = "auto",
    ) -> None:
        if tool_mode not in ("auto", "native", "text"):
            raise ValueError(f"tool_mode 必须为 auto/native/text, 收到 {tool_mode!r}")
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iters = max_iters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.trace = trace or Trace(verbose=False)
        self.tool_mode = tool_mode
        self.text_mode = tool_mode == "text"  # auto 模式下运行时可能切换为 True

    # ---------- 文本协议（text mode） ----------

    def _first_message(self, task: str) -> Dict[str, Any]:
        """构造首条 user 消息：系统设定 + 任务。

        刻意不使用 role=system 开头：部分 vLLM 构建会拒绝任何含 system 角色的
        消息（报 "Conversation roles must alternate..."），把系统设定并入首条
        user 消息对任何 OpenAI 兼容服务都通用。
        """
        sys_text = self._text_mode_system_prompt() if self.text_mode else self.system_prompt
        return {"role": "user", "content": f"【系统设定】\n{sys_text}\n\n【任务】\n{task}"}

    def _text_mode_system_prompt(self) -> str:
        """把工具列表和输出协议写进首条消息（文本协议模式下使用）。"""
        if not self.tools:
            return self.system_prompt
        tools_doc = "\n".join(
            f"- {t.name}: {t.description}"
            + (
                f"（必填参数: {', '.join(t.parameters.get('required', []))}）"
                if t.parameters.get("required")
                else ""
            )
            for t in self.tools.values()
        )
        # 用第一个工具拼一个具体的调用示例
        first = next(iter(self.tools.values()))
        req = first.parameters.get("required") or list(first.parameters.get("properties", {}))[:1]
        example_args = ", ".join(f'"{p}": "..."' for p in req)
        example = f'{TOOL_CALL_MARKER} {{"tool": "{first.name}", "arguments": {{{example_args}}}}}'
        return (
            self.system_prompt + "\n\n"
            "【工具调用协议】\n"
            "你只能调用以下工具（名字必须完全一致，不要臆造其他工具名）:\n"
            f"{tools_doc}\n\n"
            "需要调用工具时，在回复中输出一行（一次可输出多行）:\n"
            f"{example}\n"
            "arguments 必须包含该工具的全部必填参数。\n"
            "工具执行结果会以「[工具结果] 工具名: ...」的形式作为下一条用户消息返回给你。\n"
            "不需要调用工具时，直接输出最终回答即可。"
        )

    @staticmethod
    def _strip_think(text: str) -> str:
        """去掉 <think>...</think> 推理块（Qwen3 等模型会输出）。"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
        """扫描文本协议回复中的所有 TOOL_CALL: {json}，返回调用列表（按出现顺序）。

        容错：一次回复可含多个 TOOL_CALL、JSON 前后可带说明文字、arguments 可为
        JSON 字符串、字符串内可含花括号等模型常见输出偏差。
        """
        calls: List[Dict[str, Any]] = []
        idx = 0
        while True:
            marker_pos = content.find(TOOL_CALL_MARKER, idx)
            if marker_pos == -1:
                break
            brace_start = content.find("{", marker_pos)
            if brace_start == -1:
                break
            # 字符串感知的花括号深度匹配，定位该 JSON 的收尾 '}'
            depth = 0
            brace_end = -1
            in_string = False
            escaped = False
            for i in range(brace_start, len(content)):
                ch = content[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break
            if brace_end == -1:
                break
            try:
                data = json.loads(content[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                idx = brace_start + 1
                continue
            if isinstance(data, dict) and isinstance(data.get("tool"), str):
                args = data.get("arguments") or {}
                if isinstance(args, str):  # 容忍 arguments 是 JSON 字符串
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append({"tool": data["tool"], "arguments": args})
            idx = brace_end + 1
        return calls

    # ---------- 主循环 ----------

    def run(self, task: str) -> str:
        """执行一个任务，返回该 agent 的最终输出文本。"""
        self.trace.log(self.name, f"收到任务: {task}")
        messages: List[Dict[str, Any]] = [self._first_message(task)]

        step = 0
        while step < self.max_iters:
            step += 1
            message = self._chat(messages)
            messages.append(message)

            if self.text_mode:
                content = self._strip_think(message.get("content") or "")
                calls = self._parse_text_tool_calls(content)
                if not calls:
                    self.trace.log(self.name, f"最终输出 (第 {step} 轮): {content}")
                    return content
                results = []
                for call in calls:
                    name, arguments = call["tool"], call["arguments"]
                    self.trace.log(
                        self.name,
                        f"调用工具: {name}({json.dumps(arguments, ensure_ascii=False)})",
                    )
                    result = self._run_tool(name, arguments)
                    self.trace.log(self.name, f"工具 {name} 返回: {result}")
                    results.append(f"[工具结果] {name}: {result}")
                # 一次回复可能含多个调用；合并为一条 user 消息，保持 user/assistant 交替
                messages.append({"role": "user", "content": "\n\n".join(results)})
                continue

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final = self._strip_think(message.get("content") or "")
                self.trace.log(self.name, f"最终输出 (第 {step} 轮): {final}")
                return final
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        arguments = {}
                except json.JSONDecodeError:
                    arguments = {}
                self.trace.log(
                    self.name,
                    f"调用工具: {name}({json.dumps(arguments, ensure_ascii=False)})",
                )
                result = self._run_tool(name, arguments)
                self.trace.log(self.name, f"工具 {name} 返回: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        raise RuntimeError(f"[{self.name}] 达到最大迭代次数 {self.max_iters}，任务未完成")

    # ---------- 内部辅助 ----------

    def _chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """发一次 LLM 请求；auto 模式下遇到 400（模型不支持原生 tool calling）自动降级。"""
        tools = [t.schema() for t in self.tools.values()] if (self.tools and not self.text_mode) else None
        try:
            return self.llm.chat(
                messages,
                tools=tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except requests.HTTPError as exc:
            unsupported = (
                exc.response is not None
                and exc.response.status_code == 400
                and self.tools
                and not self.text_mode
                and self.tool_mode == "auto"
            )
            if not unsupported:
                raise
            self.text_mode = True
            # 把文本协议说明追加进首条消息（由 _first_message 构造，内含原始系统设定）
            messages[0]["content"] = messages[0]["content"].replace(
                self.system_prompt, self._text_mode_system_prompt()
            )
            self.trace.log(self.name, "模型不支持原生 tool calling (HTTP 400)，自动切换到文本协议模式")
            return self.llm.chat(
                messages,
                tools=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def _run_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """执行工具并返回结果文本；报错也返回给 LLM，让它自行调整。"""
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: 未知工具 {name!r}，可用工具: {sorted(self.tools)}"
        try:
            result = tool.call(arguments)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            return result
        except Exception as exc:
            return f"ERROR: 工具 {name} 执行失败: {exc!r}"
