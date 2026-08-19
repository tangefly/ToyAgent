"""离线冒烟测试：用假 LLM 模拟 main -> sub -> main -> sub -> main -> out 调用链。

覆盖两种工具调用模式：
1) native:        模型支持原生 tool_calls（如 Qwen2.5）；
2) text-fallback: 模型不支持原生 tool calling（如 Mistral-7B-Instruct-v0.2），
                  服务返回 400 时自动降级为文本协议（TOOL_CALL: {json}）。

运行: python smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys

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

    def _next(self, key: str) -> int:
        i = self.counts.get(key, 0)
        self.counts[key] = i + 1
        return i

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

    def chat(self, messages, tools=None, tool_choice=None, **kwargs):
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


class ToolsUnsupportedLLM(FakeLLM):
    """模拟不支持原生 tool calling 的模型：请求带 tools 时返回 HTTP 400。"""

    def chat(self, messages, tools=None, tool_choice=None, **kwargs):
        if tools:
            resp = requests.models.Response()
            resp.status_code = 400
            raise requests.exceptions.HTTPError("Bad Request", response=resp)
        return super().chat(messages, tools=None, **kwargs)


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


def scenario_native() -> None:
    trace = Trace(verbose=True)
    llm = FakeLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace), trace)
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    assert not any("切换到文本协议" in s["msg"] for s in trace.steps), "native 场景不应发生降级"
    check_chain(trace, "native")


def scenario_text_fallback() -> None:
    trace = Trace(verbose=True)
    llm = ToolsUnsupportedLLM()
    main_agent = build_main_agent(llm, build_sub_agents(llm, trace), trace)
    answer = main_agent.run("写一个 is_prime 函数并判断 97 和 91")
    trace.dump()
    assert "最终答案" in answer
    assert any("切换到文本协议" in s["msg"] for s in trace.steps), "应发生文本协议降级"
    check_chain(trace, "text-fallback")


def main() -> None:
    test_parser()
    scenario_native()
    scenario_text_fallback()
    print("\n✅ 全部冒烟测试通过")


if __name__ == "__main__":
    main()
