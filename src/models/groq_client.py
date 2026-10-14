"""
Groq API client.

Provides fast inference for open models (Llama, Mixtral, Gemma).
Free tier: 30 RPM on most models.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from groq import Groq

from src.config import get_api_key
from src.models.base import BaseLLM, LLMResponse
from src.models.openai_compat import _build_ssl_context

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """Groq API wrapper for open-source models."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        max_tokens: int = 256,
        requests_per_minute: Optional[float] = None,
        concurrency: int = 1,
        reasoning_effort: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens, requests_per_minute, concurrency)
        self.reasoning_effort = reasoning_effort
        api_key = get_api_key("groq")
        self._client = Groq(api_key=api_key, http_client=httpx.Client(verify=_build_ssl_context()))

    @property
    def provider_name(self) -> str:
        return "groq"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature

        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=self.max_tokens,
            **extra,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=text.strip(),
            model=self.model_name,
            provider=self.provider_name,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw=response,
        )
