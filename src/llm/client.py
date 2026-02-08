"""LLM Client using OpenRouter API (requests) - no litellm."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

os.environ["OPENROUTER_API_KEY"] = ""

class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""
    def __init__(self, message: str, used: float = 0, limit: float = 0):
        super().__init__(message)
        self.used = used
        self.limit = limit


class LLMError(Exception):
    """LLM API error."""
    def __init__(self, message: str, code: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class FunctionCall:
    """Represents a function/tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_openai(cls, call: Dict[str, Any]) -> "FunctionCall":
        """Parse from OpenAI tool_calls format."""
        func = call.get("function", {})
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}

        return cls(
            id=call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )


@dataclass
class LLMResponse:
    """Response from the LLM."""
    text: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    tokens: Optional[Dict[str, int]] = None
    model: str = ""
    finish_reason: str = ""
    raw: Optional[Dict[str, Any]] = None
    cost: float = 0.0
    # For reasoning models (e.g. gpt-5.2-codex): pass back so next request can continue reasoning
    reasoning_details: Optional[Dict[str, Any]] = None

    def has_function_calls(self) -> bool:
        """Check if response contains function calls."""
        return len(self.function_calls) > 0


def _normalize_openrouter_model(model: str) -> str:
    """Strip openrouter/ prefix so we send 'openai/gpt-5.2-codex' to OpenRouter."""
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


def _build_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Build tools in OpenAI format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return result


class OpenRouterClient:
    """LLM Client using OpenRouter API via requests (no litellm)."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 16384,
        cost_limit: Optional[float] = None,
        timeout: Optional[int] = None,
        api_key: Optional[str] = None,
        # Kept for API compatibility; not used by OpenRouter client
        cache_extended_retention: bool = True,
        cache_key: Optional[str] = None,
    ):
        self.model = _normalize_openrouter_model(model)
        self.temperature = temperature if temperature is not None else 0.0
        self.max_tokens = max_tokens
        self.cost_limit = cost_limit or float(os.environ.get("LLM_COST_LIMIT", "100.0"))
        self.timeout = timeout if timeout is not None else int(os.environ.get("LLM_TIMEOUT", "300"))
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

        self._total_cost = 0.0
        self._total_tokens = 0
        self._request_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_tokens = 0

    def _supports_temperature(self, model: str) -> bool:
        """Reasoning/codex models often use fixed temperature"""
        model_lower = model.lower()
        if any(x in model_lower for x in ["o1", "o3", "deepseek-r1"]):
            return False
        return True

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a chat request to OpenRouter."""
        if self._total_cost >= self.cost_limit:
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit:.4f}",
                used=self._total_cost,
                limit=self.cost_limit,
            )

        if not self._api_key:
            raise LLMError("OPENROUTER_API_KEY is not set", code="authentication_error")

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if self._supports_temperature(self.model):
            body["temperature"] = temperature

        if tools:
            body["tools"] = _build_tools(tools)
            body["tool_choice"] = "auto"

        if extra_body:
            body.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                self.OPENROUTER_URL,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as e:
            raise LLMError(str(e), code="timeout")
        except requests.exceptions.RequestException as e:
            raise LLMError(str(e), code="network_error")

        if resp.status_code == 401:
            raise LLMError("Invalid or missing API key", code="authentication_error")
        if resp.status_code == 429:
            raise LLMError("Rate limit exceeded", code="rate_limit")

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON: {e}", code="api_error")

        if resp.status_code != 200:
            err_msg = data.get("error", {}).get("message", data.get("message", resp.text))
            raise LLMError(err_msg, code="api_error")

        # Parse OpenAI-format response
        choices = data.get("choices") or []
        if not choices:
            return LLMResponse(model=self.model, raw=data)

        choice = choices[0]
        msg = choice.get("message") or {}
        result = LLMResponse(
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "") or "",
            text=(msg.get("content") or "") or "",
            raw=data,
        )

        # reasoning_details: preserve for next turn (gpt-5.2-codex multi-turn reasoning)
        if "reasoning_details" in msg:
            result.reasoning_details = msg["reasoning_details"]

        # Tool calls
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            result.function_calls.append(FunctionCall.from_openai(call))

        # Usage
        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        cached_tokens = 0
        if "prompt_tokens_details" in usage:
            details = usage["prompt_tokens_details"]
            if isinstance(details, dict):
                cached_tokens = details.get("cached_tokens", 0) or 0

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cached_tokens += cached_tokens
        self._total_tokens += input_tokens + output_tokens
        result.tokens = {
            "input": input_tokens,
            "output": output_tokens,
            "cached": cached_tokens,
        }

        # Cost if OpenRouter returns it
        result.cost = float(data.get("usage", {}).get("total_cost", 0) or 0)
        self._total_cost += result.cost
        self._request_count += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self._total_tokens,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cached_tokens": self._cached_tokens,
            "total_cost": self._total_cost,
            "request_count": self._request_count,
        }

    def close(self) -> None:
        """No-op for requests client."""
        pass


# Alias for drop-in replacement
LiteLLMClient = OpenRouterClient
