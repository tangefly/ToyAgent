"""离线冒烟测试：用假 LLM 模拟 main -> sub -> main -> sub -> main -> out 调用链。

覆盖三种场景：
1) native:        模型支持原生 tool_calls（如 Qwen2.5/Qwen3）；
2) text-fallback: 模型不支持原生 tool calling（如 Mistral-7B-Instruct-v0.2），
                  服务返回 400 时自动降级为文本协议（TOOL_CALL: {json}）；
3) alias:         Qwen3 常见失败模式——模型直接用子代理名调用（writer({...}) 而非
                  call_sub_agent({name: "writer", ...})），别名层自动翻译，链路不受影响。

同时断言：首轮强制工具调用（tool_choice="required"，force_tool_call 默认开）与
LMInfer agent 模式的调用链 trace 逐次增长，最终为 ["main","researcher","main","writer","main"]。

运行: python smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import Agent, Trace  # noqa: E402
from main import build_main_agent, build_sub_agents  # noqa: E402


def native_tool_call(ident: str, name: str, arguments: dict) -> dict:
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def text_tool_call(name: str, arguments: dict) -> str:
    return f'TOOL_CALL: {json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False)}'


class FakeLLM:
    """脚本化假 LLM：main 先调 researcher，再调 writer，最后直接作答。

    native 模式（tools 非空）返回 tool_calls 字段；text 模式（tools 为空）返回 TOOL_CALL 文本。
    按 agent 各自的调用次数返回脚本，与消息历史的具体格式无关。
    """

    def __init__(self) -> None:
        self.counts: dict = {}
        self.traces: List[List[str]] = []  # 每次 chat 收到的 trace 调用链（断言用）
        self.tool_choices: List[Optional[str]] = []  # 每次 chat 收到的 tool_choice（断言用）

    def _next(self, key: str) -> int:
        i = self.counts.get(key, 0)
        self.counts[key] = i + 1
        return i

    def chat(self, messages, tools=None, tool_choice=None, trace=None, **kwargs):
        self.tool_choices.append(tool_choice)
        self.traces.append(list(trace) if trace else [])
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
        if "researcher" in system:
            self._next("researcher")
            return {"role": "assistant", "content": "调研结果: 用试除法判断素数即可。"}
        if "writer" in system:
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


def test_parser() -> None:
    """文本协议解析器单元测试。"""
    p = Agent._parse_text_tool_calls
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
    assert Agent._strip_think("前<think>内部思考</think>后") == "前后"
    assert Agent._strip_think("无推理块") == "无推理块"
    # 7. think 块内的 TOOL_CALL 不应被当作工具调用
    assert p("TOOL_CALL: {\"tool\": \"writer\", \"arguments\": {\"task\": \"x\"}}") != []
    assert Agent._parse_text_tool_calls(Agent._strip_think("<think>TOOL_CALL: {\"tool\": \"writer\", \"arguments\": {\"task\": \"x\"}}</think>")) == []
    print("✅ [parser] 文本协议解析器用例通过")


def check_chain(trace: Trace, tag: str) -> None:
    steps = trace.steps
    main_tool_calls = [i for i, s in enumerate(steps)
                       if s["agent"] == "main" and "调用工具" in s["msg"]]
    researcher_done = next(i for i, s in enumerate(steps)
                           if s["agent"] == "researcher" and "最终输出" in s["msg"])
    writer_done = next(i for i, s in enumerate(steps)
                       if s["agent"] == "writer" and "最终输出" in s["msg"])
    main_final = next(i for i, s in enumerate(steps)
                      if s["agent"] == "main" and "最终输出" in s["msg"])

    assert len(main_tool_calls) == 2, f"[{tag}] main 应恰好调用 2 次 sub agent"
    assert main_tool_calls[0] < researcher_done < main_tool_calls[1] < writer_done < main_final, \
        f"[{tag}] 链路顺序应为 main -> researcher -> main -> writer -> main -> out"
    assert all(s["agent"] in ("main", "researcher", "writer") for s in steps)
    print(f"✅ [{tag}] 链路正确: main -> researcher -> main -> writer -> main -> out")


EXPECTED_CHAIN = ["main", "researcher", "main", "writer", "main"]


def check_agent_trace(llm, tag: str) -> None:
    """断言 LMInfer agent 模式的调用链：最终应为 main -> researcher -> main -> writer -> main。"""
    assert llm.traces, f"[{tag}] 应有至少一次模型请求"
    assert llm.traces[-1] == EXPECTED_CHAIN, \
        f"[{tag}] 最终调用链应为 {EXPECTED_CHAIN}，实际 {llm.traces[-1]}"
    assert all(t and t[-1] in ("main", "researcher", "writer") for t in llm.traces), \
        f"[{tag}] trace 末位必须是对应当前 agent 的名字: {llm.traces}"
    print(f"✅ [{tag}] agent trace 链正确: {' -> '.join(EXPECTED_CHAIN)}")


def scenario_native() -> None:
    trace = Trace(verbose=True)
    llm = FakeLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace), trace)
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    assert not any("切换到文本协议" in s["msg"] for s in trace.steps), "native 场景不应发生降级"
    check_chain(trace, "native")
    # force_tool_call 默认开: 首轮必须 tool_choice="required", 之后放开让模型自由作答
    assert llm.tool_choices == ["required", None, None, None, None], \
        f"首轮应强制工具调用, 实际: {llm.tool_choices}"
    # native 场景无 400 重试, 每次请求的 trace 应精确等于逐步增长序列
    assert llm.traces == [
        ["main"],
        ["main", "researcher"],
        ["main", "researcher", "main"],
        ["main", "researcher", "main", "writer"],
        EXPECTED_CHAIN,
    ], f"native 场景 trace 序列不符: {llm.traces}"
    check_agent_trace(llm, "native")


def scenario_text_fallback() -> None:
    trace = Trace(verbose=True)
    llm = ToolsUnsupportedLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace), trace)
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    assert any("切换到文本协议" in s["msg"] for s in trace.steps), "应发生文本协议降级"
    check_chain(trace, "text-fallback")
    # 文本降级场景: 首次请求被 400 拒后重试, trace 首项重复, 但最终链相同
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
            llm, build_sub_agents(llm, trace), trace, tool_mode=tool_mode
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
    scenario_native()
    scenario_text_fallback()
    scenario_alias()
    print("\n✅ 全部冒烟测试通过")


if __name__ == "__main__":
    main()
