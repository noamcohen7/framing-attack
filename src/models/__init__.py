"""
LLM client factory.

Usage:
    from src.models import create_llm
    llm = create_llm(provider="gemini", model_name="gemini-3.5-flash-lite")
"""

from __future__ import annotations

from src.models.base import BaseLLM, LLMResponse
from src.config import ModelConfig


def create_llm(
    provider: str = "gemini",
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    requests_per_minute: float | None = None,
    concurrency: int = 1,
) -> BaseLLM:
    """
    Factory function to create an LLM client by provider name.

    Args:
        provider: One of 'gemini', 'groq', 'openai', 'openrouter'.
        model_name: Model identifier (provider-specific). Uses provider default if None.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        requests_per_minute: Client-side pacing shared per (provider, model). None disables.

    Returns:
        An instance of BaseLLM.
    """
    if provider == "gemini":
        from src.models.gemini import GeminiLLM

        return GeminiLLM(
            model_name=model_name or "gemini-3.5-flash-lite",
            temperature=temperature,
            max_tokens=max_tokens,
            requests_per_minute=requests_per_minute,
            concurrency=concurrency,
        )
    elif provider == "groq":
        from src.models.groq_client import GroqLLM

        return GroqLLM(
            model_name=model_name or "llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
            requests_per_minute=requests_per_minute,
            concurrency=concurrency,
        )
    elif provider in ("openai", "openrouter"):
        from src.models.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            requests_per_minute=requests_per_minute,
            concurrency=concurrency,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Supported: gemini, groq, openai, openrouter"
        )


def create_llm_from_config(config: ModelConfig) -> BaseLLM:
    """Create an LLM client from a ModelConfig object."""
    return create_llm(
        provider=config.provider,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        requests_per_minute=config.requests_per_minute,
        concurrency=config.concurrency,
    )


__all__ = ["BaseLLM", "LLMResponse", "create_llm", "create_llm_from_config"]
