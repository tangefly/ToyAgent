"""OpenAI 兼容 LLM 客户端（专为 LMInfer 设计，兼容 vLLM）。

只依赖 requests，直接请求 LMInfer 的 OpenAI 兼容端点 /v1/chat/completions，
支持可选的 tool calling（工具调用）与 agent 模式（mode/trace/session_id，
对应 LMInfer 的 agent 会话追踪接口）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """极简 OpenAI 兼容 chat 客户端。

    Args:
        base_url: LMInfer 服务地址，如 http://localhost:8000/v1（未带 /v1 会自动补上）。
        api_key: LMInfer 默认不校验，填 "EMPTY" 即可。
        model: 模型名称，必须与 LMInfer 启动时 --served-model-name 一致（或直接用模型名）。
        timeout: 单次请求超时（秒）。
        agent_mode: 是否以 LMInfer agent 模式请求（携带 mode/trace/session_id，
            让服务端把主/子 agent 的多次调用关联到同一个会话）。
            agent 模式相关的会话状态全部由本客户端维护，对调用方透明；
            agent_mode=False 时退化为普通 OpenAI 请求，可兼容 vLLM 等服务。
        enable_thinking: Qwen3 等模型的 thinking 开关（随请求体发送，
            服务端按请求级参数处理；None 表示不传，用服务端默认）。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: Optional[str] = None,
        timeout: float = 300.0,
        agent_mode: bool = True,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.agent_mode = agent_mode
        self.enable_thinking = enable_thinking
        # agent 模式会话 id: 首次请求不带, 由 LMInfer 生成并在响应中带回, 之后自动复用
        self.session_id: Optional[str] = None
        # 最近一次响应的观测信息(LMInfer agent 模式专用, 验证 KV 复用用, 不改变返回结构):
        # last_usage = OpenAI usage 字典; last_reused_tokens = 本次请求跳过 prefill 的
        # token 数(含拼接的子 agent 输出 KV); vLLM 等普通服务无该字段, 保持 0
        self.last_usage: Dict[str, int] = {}
        self.last_reused_tokens: int = 0

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发一次请求并解析 JSON 响应。

        agent 模式遇到 404（会话不存在）自动恢复：LMInfer 的会话保存在内存，
        服务重启后 session_id 会失效，此时清掉它、以新会话重试一次，
        而不是把错误抛给 agent 循环。
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"无法连接 LLM 服务 {self.base_url}，请确认服务已启动"
            ) from exc
        except requests.HTTPError as exc:
            if (
                self.agent_mode
                and self.session_id is not None
                and exc.response is not None
                and exc.response.status_code == 404
            ):
                self.session_id = None
                payload.pop("session_id", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
            else:
                raise
        return resp.json()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        trace: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """一次非流式 chat 补全，返回 assistant 的 message dict。

        返回的 message 可能带 content（最终回答）或 tool_calls（工具调用），
        字段格式与 OpenAI API 一致，可直接追加回 messages。

        trace: agent 调用链（如 ["main","sub1","main"]），由 Agent 层在每次
        请求前维护；agent 模式下作为 trace 字段发给 LMInfer。
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if self.enable_thinking is not None:  # 请求级 thinking 开关(Qwen3 等)
            payload["enable_thinking"] = self.enable_thinking
        if self.agent_mode:
            payload["mode"] = "agent"
            if self.session_id is not None:
                payload["session_id"] = self.session_id
            if trace:
                payload["trace"] = trace

        data = self._post(payload)
        # 记录本次响应的用量与 KV 复用统计(agent 模式观测用, 普通服务无字段时为 0)
        self.last_usage = data.get("usage") or {}
        self.last_reused_tokens = data.get("reused_prompt_tokens") or 0
        if self.agent_mode and data.get("session_id"):
            self.session_id = data["session_id"]  # 首次请求后记住会话 id, 供后续请求复用
        return data["choices"][0]["message"]
