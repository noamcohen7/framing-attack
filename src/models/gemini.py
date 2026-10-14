"""
Google Gemini API client.

Uses the official google-genai SDK.
Free tier: 15 RPM / 1,500 RPD / 1M tokens per day on Flash models.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from google import genai
from google.genai import types

from src.config import get_api_key
from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


def _build_ssl_context():
    """
    Build an SSL context that works in corporate environments.

    Uses `truststore` (macOS Keychain / Windows cert store) if available,
    which handles corporate TLS interception transparently.
    Falls back to the default SSL context.
    """
    try:
        import truststore
        ctx = truststore.SSLContext()
        logger.debug("Using truststore (system cert store) for SSL")
        return ctx
    except ImportError:
        pass
    return True  # httpx default verification


class GeminiLLM(BaseLLM):
    """Gemini API wrapper."""

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        max_tokens: int = 256,
        requests_per_minute: Optional[float] = None,
        concurrency: int = 1,
    ):
        super().__init__(model_name, temperature, max_tokens, requests_per_minute, concurrency)
        api_key = get_api_key("gemini")

        # Build a custom httpx client that handles corporate proxies / TLS
        ssl_verify = _build_ssl_context()
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        custom_httpx = httpx.Client(
            verify=ssl_verify,
            proxy=proxy,
            timeout=httpx.Timeout(60.0),
        )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(httpx_client=custom_httpx),
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=self.max_tokens,
            ),
        )

        # Extract text from response
        text = response.text or ""

        # Build usage dict if available
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                # Thinking tokens are billed as output but not in candidates_token_count.
                "completion_tokens": (response.usage_metadata.candidates_token_count or 0)
                + (getattr(response.usage_metadata, "thoughts_token_count", None) or 0),
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return LLMResponse(
            text=text.strip(),
            model=self.model_name,
            provider=self.provider_name,
            usage=usage,
            raw=response,
        )
