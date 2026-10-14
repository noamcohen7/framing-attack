"""Tests for the OpenAI-compatible client (OpenRouter / OpenAI) using a fake HTTP transport."""

import json

import httpx
import pytest

from src.models.openai_compat import OpenAICompatLLM, PROVIDERS


def make_llm(handler, provider="openrouter", **kw):
    transport = httpx.MockTransport(handler)
    return OpenAICompatLLM(
        provider=provider,
        model_name="google/gemini-3.5-flash-lite",
        api_key="test-key",
        http_client=httpx.Client(transport=transport),
        **kw,
    )


class TestProviders:
    def test_known_providers(self):
        assert "openrouter" in PROVIDERS and "openai" in PROVIDERS
        assert PROVIDERS["openrouter"]["base_url"].startswith("https://openrouter.ai")
        assert PROVIDERS["openrouter"]["env_var"] == "OPENROUTER_API_KEY"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            OpenAICompatLLM(provider="nope", model_name="x", api_key="k")


class TestGenerate:
    def test_sends_chat_completion_and_parses_response(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "gen-1",
                "model": "google/gemini-3.5-flash-lite",
                "choices": [{"message": {"role": "assistant", "content": "  Donald Trump  "},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
                          "cost": 0.0000021},
            })

        llm = make_llm(handler, temperature=0.0, max_tokens=64)
        resp = llm.generate("Who won?", temperature=0.5)

        assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert seen["auth"] == "Bearer test-key"
        assert seen["body"]["model"] == "google/gemini-3.5-flash-lite"
        assert seen["body"]["messages"] == [{"role": "user", "content": "Who won?"}]
        assert seen["body"]["temperature"] == 0.5
        assert seen["body"]["max_tokens"] == 64
        assert seen["body"]["usage"] == {"include": True}  # ask OpenRouter to report cost

        assert resp.text == "Donald Trump"
        assert resp.provider == "openrouter"
        assert resp.finish_reason == "stop"
        assert resp.usage["prompt_tokens"] == 12
        assert resp.usage["completion_tokens"] == 3
        assert resp.usage["cost_usd"] == pytest.approx(0.0000021)

    def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        llm = make_llm(handler)
        with pytest.raises(httpx.HTTPStatusError):
            llm._generate("x")

    def test_empty_content_is_empty_string(self):
        def handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0},
            })

        llm = make_llm(handler)
        assert llm._generate("x").text == ""


class TestDeepInfra:
    def _handler(self, seen, usage):
        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"content": "Gold"}, "finish_reason": "stop"}],
                "usage": usage,
            })
        return handler

    def test_endpoint_reasoning_effort_and_reported_cost(self):
        seen = {}
        usage = {"prompt_tokens": 80, "completion_tokens": 60, "total_tokens": 140,
                 "estimated_cost": 0.0000108}
        llm = make_llm(self._handler(seen, usage), provider="deepinfra", reasoning_effort="low")
        resp = llm.generate("Element 79?")
        assert seen["url"] == "https://api.deepinfra.com/v1/openai/chat/completions"
        assert seen["body"]["reasoning_effort"] == "low"
        assert "usage" not in seen["body"]
        assert resp.usage["cost_usd"] == pytest.approx(0.0000108)

    def test_no_reasoning_effort_by_default(self):
        seen = {}
        llm = make_llm(self._handler(seen, {"prompt_tokens": 1, "completion_tokens": 1}), provider="deepinfra")
        resp = llm.generate("q")
        assert "reasoning_effort" not in seen["body"]
        assert "cost_usd" not in resp.usage   # falls back to the price table

    def test_env_var(self):
        assert PROVIDERS["deepinfra"]["env_var"] == "DEEPINFRA_API_KEY"
