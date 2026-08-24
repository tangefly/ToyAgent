"""核心 agent：ReAct 风格 tool-calling 循环（OpenAI 兼容接口）。

模块结构：
- strip_think / parse_text_tool_calls —— 文本协议的文本处理工具
- Trace        —— 链式日志 + agent 调用链（LMInfer agent 模式用）
- ChatBackend  —— 工具协议适配层：把 native / text / auto 三种模式归一化为
                  OpenAI 原生 tool_calls 形状；主循环只认这一种形状
- Agent        —— 主循环：发消息 -> 有 tool_calls 就执行并回填 -> 直到无调用输出最终答案

设计要点：
- 主 agent 与子 agent 共用同一个 Agent 类与同一个循环，区别只在工具集：
  main 拥有 call_sub_agent（root_only）+ 通用文件工具；sub agent 只有通用文件
  工具（read_file/write_file 等），构造时自动剥离 root_only 工具、不能再向下
  分派（最多两级，避免无限递归）。因此主 agent 可以自主决定是否调用子 agent，
  而主/子 agent 都能调用 read_file 等普通工具。
- 每轮是否调用工具由模型自己决定（force_tool_call 默认关闭）：模型认为需要
  专项能力时调用 call_sub_agent、需要看文件时调用 read_file，否则直接回答。
- 两种工具协议的全部差异（tool_calls 字段 vs TOOL_CALL: 文本、工具说明注入、
  400 自动降级、结果回填格式）都封装在 ChatBackend 里，Agent.run 不再出现
  `if text_mode` 之类的分支。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests

from llm import LLMClient

TOOL_CALL_MARKER = "TOOL_CALL:"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """去掉 <think>...</think> 推理块（Qwen3 等模型会输出）。"""
    return _THINK_RE.sub("", text).strip()


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
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


class Trace:
    """记录整条 agent 链上每一步日志：可实时打印，也可在结束时输出摘要。

    同时维护 agent 调用链 chain：每个 agent 每次发起模型请求时把自己的名字
    追加进链尾（连续同名不重复），整条链即 LMInfer agent 模式的 trace 字段，
    用于在服务端把同一个 agent 任务的多次模型请求关联到同一个会话。
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.steps: List[Dict[str, str]] = []
        self.chain: List[str] = []  # agent 调用链: main -> sub -> ...（发给 LMInfer 的 trace）

    def agent_trace(self, name: str) -> List[str]:
        """当前 agent 进入调用链（连续同名不重复），返回整条链的副本。

        例如 main 调 researcher 再回到 main：["main"] -> ["main","researcher"]
        -> ["main","researcher","main"]，最后一个元素始终是当前 agent。
        """
        if not self.chain or self.chain[-1] != name:
            self.chain.append(name)
        return list(self.chain)

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


# ---------- 工具协议适配层 ----------


def parse_json_arguments(raw: str) -> Dict[str, Any]:
    """解析模型返回的工具参数 JSON 字符串（解析失败或非对象时返回空 dict）。"""
    try:
        arguments = json.loads(raw or "{}")
        return arguments if isinstance(arguments, dict) else {}
    except json.JSONDecodeError:
        return {}


class ChatBackend:
    """统一 chat 后端：把不同工具协议归一化为 OpenAI 原生 tool_calls 形状。

    - NativeBackend: 直接走原生 tool_calls 字段（Qwen2.5/Qwen3 等原生支持）。
    - TextBackend:   文本协议（回复中输出 TOOL_CALL: {json}），任何模型可用
                     （如 Mistral-7B-Instruct-v0.2，原生不支持 tool calling）。
    - AutoBackend:   先 NativeBackend，服务返回 HTTP 400（模型不支持原生 tool
                     calling）时自动降级为 TextBackend，对主循环透明。

    每个后端返回的 assistant message 形状一致：有调用时 content=None +
    tool_calls=[{id, type, function:{name, arguments}}]，无调用时直接给 content。
    """

    # 当前是否处于文本协议模式（NativeBackend 恒 False，TextBackend 恒 True，
    # AutoBackend 在 400 降级后置 True）
    text_mode: bool = False

    def __init__(
        self,
        llm: LLMClient,
        temperature: float,
        max_tokens: int,
        force_tool_call: bool = False,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.force_tool_call = force_tool_call

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Dict[str, Any],
        tool_choice: Optional[str],
        trace: List[str],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class NativeBackend(ChatBackend):
    """OpenAI 原生 tool_calls 协议：tools/tool_choice 原样透传给服务端。"""

    def chat(self, messages, tools, tool_choice, trace):
        return self.llm.chat(
            messages,
            tools=[t.schema() for t in tools.values()] if tools else None,
            tool_choice=tool_choice if tools else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            trace=trace,
        )


class TextBackend(ChatBackend):
    """文本协议后端：把工具列表与输出协议写进首条用户消息，模型输出
    TOOL_CALL: {json} 表示调用；返回前统一归一化为原生 tool_calls 形状。
    """

    text_mode = True
    _DOC_MARKER = "【工具调用协议】"

    def __init__(self, llm, temperature, max_tokens, system_prompt: str,
                 force_tool_call: bool = False) -> None:
        super().__init__(llm, temperature, max_tokens, force_tool_call)
        self.system_prompt = system_prompt
        self._call_seq = 0  # 文本协议没有服务端生成的 call id，本地自增生成

    # ---------- 协议说明（注入首条用户消息） ----------

    def _protocol_doc(self, tools: Dict[str, Any]) -> str:
        """把工具列表、调用示例与输出协议拼成一段说明文字。"""
        if not tools:
            return self.system_prompt
        tools_doc = "\n".join(
            f"- {t.name}: {t.description}"
            + (
                f"（必填参数: {', '.join(t.parameters.get('required', []))}）"
                if t.parameters.get("required")
                else ""
            )
            for t in tools.values()
        )
        # 用第一个工具拼一个具体的调用示例
        first = next(iter(tools.values()))
        req = first.parameters.get("required") or list(first.parameters.get("properties", {}))[:1]
        example_args = ", ".join(f'"{p}": "..."' for p in req)
        example = f'{TOOL_CALL_MARKER} {{"tool": "{first.name}", "arguments": {{{example_args}}}}}'
        # 别名提示：子代理名可直接当工具名调用（如 writer({...})），代码会自动翻译
        alias_names = sorted(a for t in tools.values() for a in t.aliases)
        alias_doc = (
            f"\n提示: 也可以直接用子代理名 {', '.join(alias_names)} 作为工具名调用"
            f"（如 {TOOL_CALL_MARKER} {{\"tool\": \"{alias_names[0]}\", \"arguments\": {{\"task\": \"...\"}}}}），"
            "效果与调用 call_sub_agent 相同。"
            if alias_names
            else ""
        )
        force_doc = (
            "你的第一条回复必须输出 TOOL_CALL 调用工具把任务分派出去，"
            "禁止跳过工具直接回答；拿到工具结果后再输出最终回答。\n"
            if self.force_tool_call
            else ""
        )
        return (
            self.system_prompt + "\n\n"
            f"{self._DOC_MARKER}\n"
            "你只能调用以下工具（名字必须完全一致，不要臆造其他工具名）:\n"
            f"{tools_doc}{alias_doc}\n\n"
            f"{force_doc}"
            "需要调用工具时，在回复中输出一行（一次可输出多行）:\n"
            f"{example}\n"
            "arguments 必须包含该工具的全部必填参数。\n"
            "工具执行结果会以「[工具结果] 工具名: ...」的形式作为下一条用户消息返回给你；"
            "那是工具的返回，不是你自己的回答，不要复述它。\n"
            "不需要调用工具时，直接输出最终回答即可。"
        )

    # ---------- 消息转换 ----------

    def _to_text_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把主循环维护的「原生形状」消息转成文本协议能吃的纯文本消息。

        - role=tool 的消息转成 user（文本协议没有 tool 角色，很多服务端会拒绝），
          格式为「[工具结果] 工具名: ...」并显式声明这是工具返回、不是自己的回答；
        - 带 tool_calls 的 assistant 消息保留原 content（含 TOOL_CALL 文本）。
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            m = dict(m)
            if m["role"] == "tool":
                name = m.pop("name", "工具")
                out.append({
                    "role": "user",
                    "content": (
                        f"[工具结果] {name} 返回: {m.get('content', '')}"
                        "（以上是工具的返回内容，不是你自己的回答，"
                        "请基于它继续作答，不要再复述本段）"
                    ),
                })
            elif m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or "（已调用工具，见上方 TOOL_CALL 行）",
                })
            else:
                out.append(m)
        return out

    def _inject_protocol_doc(self, messages: List[Dict[str, Any]], tools: Dict[str, Any]) -> None:
        """把协议说明注入首条用户消息（只注入一次，用 marker 判重）。"""
        if not messages or messages[0].get("role") != "user":
            return
        first = dict(messages[0])
        content = first.get("content", "")
        if self._DOC_MARKER in content:
            return
        doc = self._protocol_doc(tools)
        if self.system_prompt in content:
            content = content.replace(self.system_prompt, doc)
        else:
            content = content + "\n\n" + doc
        first["content"] = content
        messages[0] = first

    # ---------- chat ----------

    def chat(self, messages, tools, tool_choice, trace):
        msgs = self._to_text_messages(list(messages))
        self._inject_protocol_doc(msgs, tools)
        raw = self.llm.chat(
            msgs,
            tools=None,  # 文本协议不发送 tools 字段
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            trace=trace,
        )
        content = strip_think(raw.get("content") or "")
        calls = parse_text_tool_calls(content)
        if not calls:
            return {"role": "assistant", "content": content}
        # 归一化为原生 tool_calls 形状（本地生成 call id）
        tool_calls = []
        for call in calls:
            self._call_seq += 1
            tool_calls.append({
                "id": f"call_text_{self._call_seq}",
                "type": "function",
                "function": {
                    "name": call["tool"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            })
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}


class AutoBackend(ChatBackend):
    """先按原生 tool_calls 请求；服务返回 HTTP 400（模型不支持原生 tool calling）
    时自动降级为文本协议，并对主循环透明。"""

    def __init__(self, llm, temperature, max_tokens, system_prompt: str,
                 force_tool_call: bool = False) -> None:
        super().__init__(llm, temperature, max_tokens, force_tool_call)
        self._native = NativeBackend(llm, temperature, max_tokens, force_tool_call)
        self._text = TextBackend(llm, temperature, max_tokens, system_prompt, force_tool_call)
        self.text_mode = False  # 降级后置 True（Agent 据此记录一次日志）

    def chat(self, messages, tools, tool_choice, trace):
        backend = self._text if self.text_mode else self._native
        try:
            return backend.chat(messages, tools, tool_choice, trace)
        except requests.HTTPError as exc:
            unsupported = (
                not self.text_mode
                and tools
                and exc.response is not None
                and exc.response.status_code == 400
            )
            if not unsupported:
                raise
            self.text_mode = True
            # 文本协议说明由 TextBackend 在首条消息中注入，这里直接重试
            return self._text.chat(messages, tools, tool_choice, trace)


def make_backend(
    tool_mode: str,
    llm: LLMClient,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    force_tool_call: bool = False,
) -> ChatBackend:
    """按 tool_mode 构造对应协议后端。"""
    if tool_mode == "native":
        return NativeBackend(llm, temperature, max_tokens, force_tool_call)
    if tool_mode == "text":
        return TextBackend(llm, temperature, max_tokens, system_prompt, force_tool_call)
    return AutoBackend(llm, temperature, max_tokens, system_prompt, force_tool_call)


# ---------- Agent 主循环 ----------


class Agent:
    """tool-calling agent 循环（主/子 agent 共用）。

    流程：向 LLM 发消息 -> 若返回工具调用则逐个执行并把结果回填 -> 重复，
    直到 LLM 给出不含工具调用的最终回答。
    每轮是否调用工具、调用哪个工具，完全由模型自主决定。
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        tools: Optional[List[Any]] = None,
        max_iters: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        trace: Optional[Trace] = None,
        tool_mode: str = "auto",
        force_tool_call: bool = False,
        is_root: bool = True,
    ) -> None:
        if tool_mode not in ("auto", "native", "text"):
            raise ValueError(f"tool_mode 必须为 auto/native/text, 收到 {tool_mode!r}")
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.trace = trace or Trace(verbose=False)
        self.tools: Dict[str, Any] = {t.name: t for t in (tools or [])}
        self.is_root = is_root
        if not is_root:
            # 两级 agent 限制：只有 root(main) agent 可以向下分派。sub agent 构造时
            # 自动剥离 root_only 工具（如 call_sub_agent）——请求里根本不含其 schema，
            # 模型无从调用，从源头杜绝 sub -> sub 嵌套；_run_tool 里另有运行时兜底。
            stripped = [n for n, t in self.tools.items() if t.root_only]
            for n in stripped:
                del self.tools[n]
            if stripped:
                self.trace.log(
                    name, f"sub agent 剥离 root_only 工具（禁止继续分派）: {sorted(stripped)}"
                )
        self.max_iters = max_iters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.force_tool_call = force_tool_call  # 第一轮强制要求工具调用（见 run）
        self.backend = make_backend(
            tool_mode, llm, temperature, max_tokens, system_prompt, force_tool_call
        )

    # ---------- 主循环 ----------

    def run(self, task: str) -> str:
        """执行一个任务，返回该 agent 的最终输出文本。"""
        self.trace.log(self.name, f"{task}")
        messages: List[Dict[str, Any]] = [self._first_message(task)]

        for step in range(1, self.max_iters + 1):
            assistant = self._chat(messages, step)
            messages.append(assistant)

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                final = strip_think(assistant.get("content") or "")
                self.trace.log(self.name, f"{final}")
                return final

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                arguments = parse_json_arguments(fn.get("arguments"))
                self.trace.log(
                    self.name,
                    f"{name}({json.dumps(arguments, ensure_ascii=False)})",
                )
                result = self._run_tool(name, arguments)
                self.trace.log(self.name, f"工具 {name} 返回: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,  # 文本协议后端用它格式化「[工具结果] 工具名: ...」
                    "content": result,
                })

        raise RuntimeError(f"[{self.name}] 达到最大迭代次数 {self.max_iters}，任务未完成")

    # ---------- 内部辅助 ----------

    def _first_message(self, task: str) -> Dict[str, Any]:
        """构造首条 user 消息：系统设定 + 任务。

        刻意不使用 role=system 开头：部分 vLLM 构建会拒绝任何含 system 角色的
        消息（报 "Conversation roles must alternate..."），把系统设定并入首条
        user 消息对任何 OpenAI 兼容服务都通用。
        """
        return {"role": "user", "content": f"【系统设定】\n{self.system_prompt}\n\n【任务】\n{task}"}

    def _chat(self, messages: List[Dict[str, Any]], step: int) -> Dict[str, Any]:
        """发一次模型请求，返回 assistant message（形状与 OpenAI 一致）。

        force_tool_call: 第一轮强制要求工具调用（tool_choice="required"），保证
        main 先分派子任务而不是直接作答——小模型（如 Qwen3-8B）默认倾向跳过工具
        直接回答。原生模式的每一轮都显式传 tool_choice（首轮 required / 后续 auto）：
        LMInfer 的 enable_auto_tool_choice 默认关闭，请求带 tools 但省略 tool_choice
        时服务端按 none 处理、渲染 prompt 不输出 tools 系统段，首轮(required)与
        后续轮的 prompt 前缀就会不一致，agent 模式的跨请求前缀 KV 复用永远无法命中。
        显式 auto 保证每轮渲染相同的 tools 段、prompt 逐字延续。
        """
        text_mode = self.backend.text_mode
        tool_choice = (
            "required"
            if self.force_tool_call and self.tools and step == 1 and not text_mode
            else "auto" if self.tools and not text_mode else None
        )
        assistant = self.backend.chat(
            messages,
            tools=self.tools or None,
            tool_choice=tool_choice,
            trace=self.trace.agent_trace(self.name),  # 本 agent 进入调用链（LMInfer agent 模式用）
        )
        if self.backend.text_mode and not text_mode:
            # AutoBackend 本轮发生了 400 降级（text_mode 从 False 变 True），记录一次
            self.trace.log(
                self.name, "模型不支持原生 tool calling (HTTP 400)，自动切换到文本协议模式"
            )
        return assistant

    def _resolve_tool(self, name: str):
        """按名字或别名解析工具。返回 (工具, 命中的别名)；按名字命中时第二个值为 None。

        别名即 sub agent 名（如 researcher/writer/reviewer）：模型经常直接写
        writer({...}) 而不用规范形式 call_sub_agent({name: "writer", ...})，
        此时解析到 call_sub_agent 并记录命中的别名。
        """
        tool = self.tools.get(name)
        if tool is not None:
            return tool, None
        for t in self.tools.values():
            if name in t.aliases:
                return t, name
        return None, None

    def _run_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """执行工具并返回结果文本；报错也返回给 LLM，让它自行调整。

        别名容错：模型用子代理名调用时（如 writer({"task": ...})），自动翻译为
        call_sub_agent({"name": "writer", "task": ...})，不再报「未知工具」。
        """
        tool, alias_for = self._resolve_tool(name)
        if tool is None:
            alias_names = sorted(a for t in self.tools.values() for a in t.aliases)
            hint = (
                f"；另外可以直接用子代理名 {alias_names} 调用（效果等同 call_sub_agent）"
                if alias_names else ""
            )
            return f"ERROR: 未知工具 {name!r}，可用工具: {sorted(self.tools)}{hint}"
        if alias_for is not None:
            # 把别名补进缺省的 name 参数: writer({"task": ...}) -> call_sub_agent({"name": "writer", "task": ...})
            if "name" in (tool.parameters.get("properties") or {}) and "name" not in arguments:
                arguments = dict(arguments)
                arguments["name"] = alias_for
            self.trace.log(
                self.name,
                f"别名 {name!r} -> {tool.name}({json.dumps(arguments, ensure_ascii=False)})",
            )
        # 两级 agent 运行时兜底: 理论上非 root agent 构造时已剥离 root_only 工具，
        # 这里再拦一道（覆盖调用方事后往 sub agent 塞工具等场景）
        if tool.root_only and not self.is_root:
            return (
                f"ERROR: 工具 {name!r} 只能由 main agent 调用，sub agent 禁止继续向下"
                f"分派（最多两级 agent）；请直接作答，或改用你自带的普通工具。"
            )
        try:
            result = tool.call(arguments)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            return result
        except Exception as exc:
            return f"ERROR: 工具 {name} 执行失败: {exc!r}"
