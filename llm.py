"""OpenAI 兼容 LLM 客户端（专为 vLLM 设计）。

只依赖 requests，直接请求 vLLM 的 OpenAI 兼容端点 /v1/chat/completions，
支持可选的 tool calling（工具调用）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """极简 OpenAI 兼容 chat 客户端。

    Args:
        base_url: vLLM 服务地址，如 http://localhost:8000/v1（未带 /v1 会自动补上）。
        api_key: vLLM 默认不校验，填 "EMPTY" 即可。
        model: 模型名称，必须与 vLLM 启动时 --served-model-name 一致（或直接用模型名）。
        timeout: 单次请求超时（秒）。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """一次非流式 chat 补全，返回 assistant 的 message dict。

        返回的 message 可能带 content（最终回答）或 tool_calls（工具调用），
        字段格式与 OpenAI API 一致，可直接追加回 messages。
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

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"无法连接 vLLM 服务 {self.base_url}，请确认服务已启动"
            ) from exc

        return resp.json()["choices"][0]["message"]
