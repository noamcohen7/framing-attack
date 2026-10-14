"""
OpenAI-compatible chat-completions client (OpenAI, OpenRouter, DeepInfra).

Talks to the /chat/completions endpoint directly over httpx, so one class
serves every provider that speaks the OpenAI wire format.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from src.config import get_api_key
from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)

PROVIDERS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "max_tokens_param": "max_completion_tokens",
        "include_usage": False,
        "default_model": "gpt-4.1-mini",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "max_tokens_param": "max_tokens",
        "include_usage": True,  # OpenRouter reports cost in usage when asked
        "default_model": "google/gemini-3.5-flash-lite",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "env_var": "DEEPINFRA_API_KEY",
        "max_tokens_param": "max_tokens",
        "include_usage": False,  # reports usage.estimated_cost on every response
        "default_model": "openai/gpt-oss-120b",
    },
}


def _build_ssl_context():
    try:
        import truststore
        return truststore.SSLContext()
    except ImportError:
        return True


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        provider: str,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        requests_per_minute: Optional[float] = None,
        concurrency: int = 1,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        reasoning_effort: Optional[str] = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown OpenAI-compatible provider {provider!r}. "
                f"Supported: {', '.join(PROVIDERS)}"
            )
        self._provider = provider
        self._spec = PROVIDERS[provider]
        super().__init__(
            model_name or self._spec["default_model"],
            temperature,
            max_tokens,
            requests_per_minute,
            concurrency,
        )
        self.reasoning_effort = reasoning_effort
        self._api_key = api_key or get_api_key(provider)
        if http_client is None:
            proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            http_client = httpx.Client(
                verify=_build_ssl_context(), proxy=proxy, timeout=httpx.Timeout(60.0)
            )
        self._http = http_client

    @property
    def provider_name(self) -> str:
        return self._provider

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature
        body: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            self._spec["max_tokens_param"]: self.max_tokens,
        }
        if self._spec["include_usage"]:
            body["usage"] = {"include": True}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._provider == "openrouter":
            headers["X-Title"] = "framing-attack"

        resp = self._http.post(
            f"{self._spec['base_url']}/chat/completions", json=body, headers=headers
        )
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            hint = f" Please retry in {retry_after}s." if retry_after else ""
            raise httpx.HTTPStatusError(
                f"429 rate limited by {self._provider}.{hint} Body: {resp.text[:300]}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text = choice.get("message", {}).get("content") or ""

        usage_raw = data.get("usage") or {}
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }
        # Provider-reported cost: OpenRouter "cost", DeepInfra "estimated_cost".
        for key in ("cost", "estimated_cost"):
            if usage_raw.get(key) is not None:
                usage["cost_usd"] = float(usage_raw[key])
                break

        return LLMResponse(
            text=text.strip(),
            model=data.get("model", self.model_name),
            provider=self.provider_name,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            raw=data,
        )
