"""离线冒烟测试：用假 LLM 模拟主/子 agent 的 tool-calling 链路。

覆盖场景：
1) native:        模型支持原生 tool_calls；主 agent 自主决定分派（force_tool_call 默认关）；
2) forced:        force_tool_call=True 时首轮 tool_choice="required"；
3) sub-tools:     sub agent 自己调用文件工具（read_file）后再作答（主/子 agent 都能调用工具）；
4) direct:        主 agent 自主决定不调用任何工具，第一轮直接给出最终答案；
5) text-fallback: 模型不支持原生 tool calling（如 Mistral-7B-Instruct-v0.2），
                  服务返回 400 时自动降级为文本协议（TOOL_CALL: {json}）；
6) alias:         Qwen3 常见失败模式——模型直接用子代理名调用（writer({...}) 而非
                  call_sub_agent({name: "writer", ...})），别名层自动翻译，链路不受影响。

同时断言：原生模式每轮显式传 tool_choice（保证 LMInfer 每轮渲染 tools 系统段、
agent 模式跨请求 KV 复用才能命中）；sub agent 的请求里也携带文件工具 schema
（read_file 等，验证「子 agent 也能调用工具」）；LMInfer agent 模式调用链 trace
逐次增长，最终为 ["main","researcher","main","writer","main"]。

运行: python smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import Trace, parse_text_tool_calls, strip_think  # noqa: E402
from main import build_main_agent, build_sub_agents  # noqa: E402


def native_tool_call(ident: str, name: str, arguments: dict) -> dict:
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def text_tool_call(name: str, arguments: dict) -> str:
    return f'TOOL_CALL: {json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False)}'


REPO = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(REPO, "requirements.txt")


class FakeLLM:
    """脚本化假 LLM：main 先调 researcher，再调 writer，最后直接作答。

    native 模式（tools 非空）返回 tool_calls 字段；text 模式（tools 为空）返回 TOOL_CALL 文本。
    按 agent 各自的调用次数返回脚本，与消息历史的具体格式无关。
    """

    def __init__(self) -> None:
        self.counts: dict = {}
        self.traces: List[List[str]] = []  # 每次 chat 收到的 trace 调用链（断言用）
        self.tool_choices: List[Optional[str]] = []  # 每次 chat 收到的 tool_choice（断言用）
        self.tool_names: List[List[str]] = []  # 每次 chat 收到的工具名列表（断言子 agent 也有文件工具）

    def _next(self, key: str) -> int:
        i = self.counts.get(key, 0)
        self.counts[key] = i + 1
        return i

    def chat(self, messages, tools=None, tool_choice=None, trace=None, **kwargs):
        self.tool_choices.append(tool_choice)
        self.traces.append(list(trace) if trace else [])
        self.tool_names.append(sorted(t["function"]["name"] for t in (tools or [])))
        return self._chat(messages, tools)

    def _chat(self, messages, tools):  # 由 chat 记录 trace 后进入真正的脚本逻辑
        self._validate_roles(messages)
        system = messages[0]["content"]
        if "main agent" in system:
            i = self._next("main")
            if i == 0:
                args = {"name": "researcher", "task": "调研 is_prime 的实现思路"}
            elif i == 1:
                args = {"name": "writer", "task": "实现 is_prime 并判断 97 和 91"}
            else:
                return {"role": "assistant",
                        "content": "最终答案: is_prime 实现完成，97 是素数，91 不是素数。"}
            if tools:  # native 模式
                return {"role": "assistant", "content": None,
                        "tool_calls": [native_tool_call(f"call_{i + 1}", "call_sub_agent", args)]}
            return {"role": "assistant", "content": text_tool_call("call_sub_agent", args)}
        if "researcher sub agent" in system:
            self._next("researcher")
            return {"role": "assistant", "content": "调研结果: 用试除法判断素数即可。"}
        if "writer sub agent" in system:
            self._next("writer")
            return {"role": "assistant", "content": "实现: def is_prime(n): ... 97 是素数，91 不是素数。"}
        raise AssertionError(f"未知 agent 的 system prompt: {system[:50]}")

    @staticmethod
    def _http400(msg: str) -> requests.exceptions.HTTPError:
        resp = requests.models.Response()
        resp.status_code = 400
        return requests.exceptions.HTTPError(msg, response=resp)

    @staticmethod
    def _validate_roles(messages) -> None:
        """模拟真实 vLLM 的行为：拒绝 system 角色、user/assistant 必须交替。

        tool 消息在原生 tool-calling 协议中打断交替链（assistant(tool_calls) -> tool
        -> assistant 是标准序列），因此 tool 角色把 prev 重置为 None。
        """
        prev = None
        for m in messages:
            role = m["role"]
            if role == "system":
                raise FakeLLM._http400("vLLM: 拒绝 system 角色 (Conversation roles must alternate)")
            if role == "tool":
                prev = None
            elif role in ("user", "assistant"):
                if prev == role:
                    raise FakeLLM._http400("vLLM: roles must alternate user/assistant")
                prev = role


class ToolsUnsupportedLLM(FakeLLM):
    """模拟不支持原生 tool calling 的模型：请求带 tools 时返回 HTTP 400。"""

    def chat(self, messages, tools=None, tool_choice=None, trace=None, **kwargs):
        if tools:
            self.tool_choices.append(tool_choice)
            self.traces.append(list(trace) if trace else [])  # 被拒的请求也记录其 trace
            resp = requests.models.Response()
            resp.status_code = 400
            raise requests.exceptions.HTTPError("Bad Request", response=resp)
        return super().chat(messages, tools=None, trace=trace, **kwargs)


class FileToolSubLLM(FakeLLM):
    """模拟「子 agent 也调用工具」：researcher 第一轮调用 read_file 读文件，第二轮再作答。"""

    def _chat(self, messages, tools):
        system = messages[0]["content"]
        if "researcher sub agent" in system and self.counts.get("researcher", 0) == 0:
            self._next("researcher")
            args = {"path": REQUIREMENTS}
            if tools:  # native 模式
                return {"role": "assistant", "content": None,
                        "tool_calls": [native_tool_call("call_r1", "read_file", args)]}
            return {"role": "assistant", "content": text_tool_call("read_file", args)}
        return super()._chat(messages, tools)


class AliasLLM(FakeLLM):
    """模拟 Qwen3 的典型行为：用子代理名直接调用（researcher({...}) 而非
    call_sub_agent({name: "researcher", ...})），应由别名层自动翻译。"""

    def _chat(self, messages, tools):
        system = messages[0]["content"]
        if "main agent" in system:
            i = self._next("main")
            if i == 0:
                name, args = "researcher", {"task": "调研 is_prime 的实现思路"}
            elif i == 1:
                name, args = "writer", {"task": "实现 is_prime 并判断 97 和 91"}
            else:
                return {"role": "assistant",
                        "content": "最终答案: is_prime 实现完成，97 是素数，91 不是素数。"}
            if tools:  # native 模式
                return {"role": "assistant", "content": None,
                        "tool_calls": [native_tool_call(f"call_{i + 1}", name, args)]}
            return {"role": "assistant", "content": text_tool_call(name, args)}
        return super()._chat(messages, tools)


class DirectLLM(FakeLLM):
    """模拟「主 agent 自主决定不调用工具」：第一轮直接给出最终答案（无强制分派）。"""

    def _chat(self, messages, tools):
        self._validate_roles(messages)
        system = messages[0]["content"]
        if "main agent" in system:
            self._next("main")
            return {"role": "assistant", "content": "直接回答: 97 是素数，91 不是素数。"}
        return super()._chat(messages, tools)


# ---------- 单元测试 ----------


def test_parser() -> None:
    """文本协议解析器单元测试。"""
    p = parse_text_tool_calls
    # 1. 单个调用
    assert p('TOOL_CALL: {"tool": "writer", "arguments": {"task": "写代码"}}') == \
        [{"tool": "writer", "arguments": {"task": "写代码"}}]
    # 2. 多个调用 + 前后说明文字
    text = ('我先调研。\nTOOL_CALL: {"tool": "researcher", "arguments": {"task": "查资料"}}\n'
            '然后实现。\nTOOL_CALL: {"tool": "writer", "arguments": {"task": "写代码"}}\n完毕。')
    calls = p(text)
    assert [c["tool"] for c in calls] == ["researcher", "writer"]
    # 3. arguments 为 JSON 字符串
    assert p(r'TOOL_CALL: {"tool": "writer", "arguments": "{\"task\": \"写代码\"}"}') == \
        [{"tool": "writer", "arguments": {"task": "写代码"}}]
    # 4. 字符串里的花括号不影响深度匹配
    assert p('TOOL_CALL: {"tool": "writer", "arguments": {"task": "输出 {x}"}}') == \
        [{"tool": "writer", "arguments": {"task": "输出 {x}"}}]
    # 5. 无调用 -> 空列表
    assert p("直接回答即可，不需要工具") == []
    # 6. think 推理块剥离
    assert strip_think("前<think>内部思考</think>后") == "前后"
    assert strip_think("无推理块") == "无推理块"
    # 7. think 块内的 TOOL_CALL 不应被当作工具调用
    assert p("TOOL_CALL: {\"tool\": \"writer\", \"arguments\": {\"task\": \"x\"}}") != []
    assert parse_text_tool_calls(strip_think("<think>TOOL_CALL: {\"tool\": \"writer\", \"arguments\": {\"task\": \"x\"}}</think>")) == []
    print("✅ [parser] 文本协议解析器用例通过")


class StubLLM:
    """记录请求的最小 LLM 桩，返回固定 content。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.sent: List[dict] = []

    def chat(self, messages, **kwargs):
        self.sent.append({"messages": messages, **kwargs})
        return {"role": "assistant", "content": self.content}


def test_text_backend_normalization() -> None:
    """TextBackend 把 TOOL_CALL 文本归一化为原生 tool_calls 形状，并在首条消息注入协议说明。"""
    from agent import TextBackend
    from tools import build_file_tools

    stub = StubLLM('TOOL_CALL: {"tool": "read_file", "arguments": {"path": "a.txt"}}')
    backend = TextBackend(stub, temperature=0.7, max_tokens=64, system_prompt="你是测试 agent")
    tools = {t.name: t for t in build_file_tools()}
    msg = backend.chat(
        [{"role": "user", "content": "【系统设定】\n你是测试 agent\n\n【任务】\n读文件"}],
        tools=tools, tool_choice=None, trace=["main"],
    )
    assert msg["content"] is None
    calls = msg["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["id"].startswith("call_text_")
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}
    # 协议说明注入到首条用户消息
    assert "【工具调用协议】" in stub.sent[0]["messages"][0]["content"]
    # 没有 TOOL_CALL 时直接返回 content
    stub2 = StubLLM("直接回答。")
    backend2 = TextBackend(stub2, temperature=0.7, max_tokens=64, system_prompt="你是测试 agent")
    msg2 = backend2.chat([{"role": "user", "content": "【系统设定】\n你是测试 agent\n\n【任务】\nhi"}],
                         tools=tools, tool_choice=None, trace=["main"])
    assert msg2.get("tool_calls") is None and msg2["content"] == "直接回答。"
    print("✅ [text-backend] TOOL_CALL 归一化为原生 tool_calls 形状")


def test_file_tools() -> None:
    """内置文件工具的基本行为。"""
    from tools import _tool_list_directory, _tool_read_file, _tool_search_files

    content = _tool_read_file(REQUIREMENTS)
    assert "requests" in content and "PyYAML" in content and "ERROR" not in content
    assert _tool_read_file(os.path.join(REPO, "no_such_file_xyz")).startswith("ERROR:")
    listing = _tool_list_directory(REPO)
    assert "requirements.txt" in listing and "dir  example" in listing
    hits = _tool_search_files("*.py", REPO)
    assert "agent.py" in hits and "main.py" in hits
    print("✅ [file-tools] read_file / list_directory / search_files 基本行为正确")


# ---------- 链路断言 ----------

EXPECTED_CHAIN = ["main", "researcher", "main", "writer", "main"]


def check_chain(trace: Trace, tag: str) -> None:
    steps = trace.steps
    # 日志消息无固定前缀: main 的工具调用行形如 {工具名}({"name": ...}),
    # 别名翻译行以 "别名" 开头需排除; 最终输出按 FakeLLM 脚本内容的开头关键词识别。
    main_tool_calls = [i for i, s in enumerate(steps)
                       if s["agent"] == "main" and "({\"" in s["msg"]
                       and not s["msg"].startswith("别名")]
    researcher_done = next(i for i, s in enumerate(steps)
                           if s["agent"] == "researcher" and s["msg"].startswith("调研结果:"))
    writer_done = next(i for i, s in enumerate(steps)
                       if s["agent"] == "writer" and s["msg"].startswith("实现:"))
    main_final = next(i for i, s in enumerate(steps)
                      if s["agent"] == "main" and s["msg"].startswith("最终答案:"))

    assert len(main_tool_calls) == 2, f"[{tag}] main 应恰好调用 2 次 sub agent"
    assert main_tool_calls[0] < researcher_done < main_tool_calls[1] < writer_done < main_final, \
        f"[{tag}] 链路顺序应为 main -> researcher -> main -> writer -> main -> out"
    assert all(s["agent"] in ("main", "researcher", "writer") for s in steps)
    print(f"✅ [{tag}] 链路正确: main -> researcher -> main -> writer -> main -> out")


def check_agent_trace(llm, tag: str) -> None:
    """断言 LMInfer agent 模式的调用链：最终应为 main -> researcher -> main -> writer -> main。"""
    assert llm.traces, f"[{tag}] 应有至少一次模型请求"
    assert llm.traces[-1] == EXPECTED_CHAIN, \
        f"[{tag}] 最终调用链应为 {EXPECTED_CHAIN}，实际 {llm.traces[-1]}"
    assert all(t and t[-1] in ("main", "researcher", "writer") for t in llm.traces), \
        f"[{tag}] trace 末位必须是对应当前 agent 的名字: {llm.traces}"
    print(f"✅ [{tag}] agent trace 链正确: {' -> '.join(EXPECTED_CHAIN)}")


FILE_TOOL_NAMES = {"read_file", "write_file", "list_directory", "search_files"}


def check_sub_agents_have_file_tools(llm, tag: str) -> None:
    """断言 sub agent 的模型请求也携带文件工具 schema（子 agent 也能调用 read_file 等）。"""
    researcher_idx = llm.traces.index(["main", "researcher"])
    writer_idx = llm.traces.index(["main", "researcher", "main", "writer"])
    # main 的工具集 = 文件工具 + call_sub_agent
    assert {"call_sub_agent"} | FILE_TOOL_NAMES <= set(llm.tool_names[0]), \
        f"[{tag}] main 首轮应同时携带 call_sub_agent 与文件工具: {llm.tool_names[0]}"
    # sub agent 只有文件工具，没有 call_sub_agent（不能继续分派）
    assert "call_sub_agent" not in llm.tool_names[researcher_idx], \
        f"[{tag}] sub agent 不应携带 call_sub_agent: {llm.tool_names[researcher_idx]}"
    assert FILE_TOOL_NAMES <= set(llm.tool_names[researcher_idx]), \
        f"[{tag}] researcher 应携带文件工具: {llm.tool_names[researcher_idx]}"
    assert FILE_TOOL_NAMES <= set(llm.tool_names[writer_idx]), \
        f"[{tag}] writer 应携带文件工具: {llm.tool_names[writer_idx]}"
    print(f"✅ [{tag}] 主/子 agent 都能调用文件工具，sub agent 无 call_sub_agent")


# ---------- 场景 ----------


def scenario_native() -> None:
    """主 agent 自主决定分派（force_tool_call 默认关），sub agent 也带文件工具。"""
    trace = Trace(verbose=True)
    llm = FakeLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace, max_iters=4), trace)
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    assert not any("切换到文本协议" in s["msg"] for s in trace.steps), "native 场景不应发生降级"
    check_chain(trace, "native")
    check_agent_trace(llm, "native")
    check_sub_agents_have_file_tools(llm, "native")
    # force_tool_call 默认关：首轮也显式 "auto"；后续 native 轮显式 "auto"（保证
    # LMInfer 每轮都渲染 tools 系统段、prompt 前缀链不断，agent 模式跨请求 KV 复用
    # 才能命中）；sub agent 带工具时同样每轮显式 "auto"。
    assert llm.tool_choices == ["auto"] * 5, \
        f"tool_choice 序列应为 5 个 auto, 实际: {llm.tool_choices}"
    # 每次请求的 trace 应精确等于逐步增长序列
    assert llm.traces == [
        ["main"],
        ["main", "researcher"],
        ["main", "researcher", "main"],
        ["main", "researcher", "main", "writer"],
        EXPECTED_CHAIN,
    ], f"native 场景 trace 序列不符: {llm.traces}"


def scenario_forced() -> None:
    """force_tool_call=True 时首轮 tool_choice="required"。"""
    trace = Trace(verbose=True)
    llm = FakeLLM()
    main_agent = build_main_agent(
        llm, build_sub_agents(llm, trace, max_iters=4), trace, force_tool_call=True
    )
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    check_chain(trace, "forced")
    assert llm.tool_choices == ["required", "auto", "auto", "auto", "auto"], \
        f"force 场景 tool_choice 序列应为 [required, auto, auto, auto, auto], 实际: {llm.tool_choices}"
    print("✅ [forced] 首轮 tool_choice=required，后续轮显式 auto")


def scenario_sub_tools() -> None:
    """sub agent 自己调用文件工具（read_file）后再作答。"""
    trace = Trace(verbose=True)
    llm = FileToolSubLLM()
    main_agent = build_main_agent(
        llm, build_sub_agents(llm, trace, max_iters=2), trace
    )
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    check_chain(trace, "sub-tools")
    check_agent_trace(llm, "sub-tools")
    # researcher 自己调用了一次 read_file，并拿到了文件内容
    assert any(s["agent"] == "researcher" and s["msg"].startswith("read_file(")
               for s in trace.steps), f"[sub-tools] researcher 应调用 read_file: {trace.steps}"
    assert any(s["agent"] == "researcher" and s["msg"].startswith("工具 read_file 返回")
               for s in trace.steps), "[sub-tools] researcher 应收到 read_file 的结果"
    # researcher 多了一轮（read_file -> 作答），共 6 次请求
    assert llm.traces == [
        ["main"],
        ["main", "researcher"],
        ["main", "researcher"],  # researcher 连续两轮，同名不重复
        ["main", "researcher", "main"],
        ["main", "researcher", "main", "writer"],
        EXPECTED_CHAIN,
    ], f"[sub-tools] trace 序列不符: {llm.traces}"
    assert llm.tool_choices == ["auto"] * 6, f"[sub-tools] tool_choice 应为 6 个 auto: {llm.tool_choices}"
    print("✅ [sub-tools] 子 agent 也能调用 read_file 等文件工具")


def scenario_direct() -> None:
    """主 agent 自主决定不调用任何工具，第一轮直接作答（无强制分派）。"""
    trace = Trace(verbose=True)
    llm = DirectLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace), trace)
    answer = main_agent.run("97 是素数吗？")
    trace.dump()
    assert "直接回答" in answer
    assert llm.traces == [["main"]], f"[direct] 应只有一次 main 请求: {llm.traces}"
    assert llm.tool_choices == ["auto"], f"[direct] tool_choice 应为 [auto]: {llm.tool_choices}"
    assert llm.counts == {"main": 1}, f"[direct] 不应有任何子 agent 被调用: {llm.counts}"
    assert "call_sub_agent" in llm.tool_names[0], "工具仍在（模型只是选择不用）"
    print("✅ [direct] 主 agent 自主决定不调用工具，直接回答")


def scenario_text_fallback() -> None:
    """模型不支持原生 tool calling：400 后自动降级为文本协议。"""
    trace = Trace(verbose=True)
    llm = ToolsUnsupportedLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace, max_iters=4), trace)
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    assert any("切换到文本协议" in s["msg"] for s in trace.steps), "应发生文本协议降级"
    check_chain(trace, "text-fallback")
    # 首次请求被 400 拒后重试, trace 首项重复, 但最终链相同
    assert llm.traces[0] == ["main"] and llm.traces[-1] == EXPECTED_CHAIN, \
        f"text-fallback 场景 trace 不符: {llm.traces}"
    check_agent_trace(llm, "text-fallback")


def scenario_alias() -> None:
    """Qwen3 常见失败模式：模型直接用子代理名调用（writer({...}) 而非
    call_sub_agent({name: "writer", ...})）。别名层应自动翻译为 call_sub_agent，
    并把 name 补进参数，子代理照常执行、链路不受影响。native 与 text 两种模式都验证。"""
    for tool_mode in ("auto", "text"):
        trace = Trace(verbose=True)
        llm = AliasLLM()
        main_agent = build_main_agent(
            llm, build_sub_agents(llm, trace, max_iters=4), trace, tool_mode=tool_mode
        )
        answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
        trace.dump()
        assert "最终答案" in answer, f"[{tool_mode}] 应得到最终答案"
        # 两次调用都命中别名并被翻译为 call_sub_agent（name 自动补进参数）
        alias_lines = [s["msg"] for s in trace.steps
                       if s["agent"] == "main" and "别名" in s["msg"]]
        assert len(alias_lines) == 2, f"[{tool_mode}] 应有 2 次别名翻译: {trace.steps}"
        assert "researcher" in alias_lines[0] and '"name": "researcher"' in alias_lines[0], \
            f"[{tool_mode}] 首次别名翻译异常: {alias_lines[0]}"
        assert "writer" in alias_lines[1] and '"name": "writer"' in alias_lines[1], \
            f"[{tool_mode}] 二次别名翻译异常: {alias_lines[1]}"
        check_chain(trace, f"alias-{tool_mode}")
    print("✅ [alias] 子代理名直接调用自动翻译为 call_sub_agent (native + text)")


def main() -> None:
    test_parser()
    test_text_backend_normalization()
    test_file_tools()
    scenario_native()
    scenario_forced()
    scenario_sub_tools()
    scenario_direct()
    scenario_text_fallback()
    scenario_alias()
    print("\n✅ 全部冒烟测试通过")


if __name__ == "__main__":
    main()
