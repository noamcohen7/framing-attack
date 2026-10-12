"""Tests for rate limiting and 429 handling in BaseLLM.generate (no API calls)."""

from typing import Optional

import pytest

from src.models import base as base_mod
from src.models.base import BaseLLM, LLMResponse


class RateLimitError(Exception):
    pass


class FakeLLM(BaseLLM):
    """Fails with a 429-style error a given number of times, then succeeds."""

    def __init__(self, fail_times: int = 0, **kw):
        super().__init__(model_name="fake-model", **kw)
        self.fail_times = fail_times
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RateLimitError(
                "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
                "'Quota exceeded. Please retry in 46.368075457s.', 'status': 'RESOURCE_EXHAUSTED', "
                "'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '46s'}]}}"
            )
        return LLMResponse(text="ok", model=self.model_name, provider=self.provider_name)


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace time.sleep / time.monotonic in base with a controllable clock."""
    state = {"now": 1000.0, "sleeps": []}

    def sleep(s):
        state["sleeps"].append(s)
        state["now"] += s

    monkeypatch.setattr(base_mod.time, "sleep", sleep)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: state["now"])
    base_mod._RATE_LIMITERS.clear()
    return state


class TestParseRetryDelay:
    def test_parses_retry_in_seconds(self):
        msg = "Quota exceeded. Please retry in 46.368075457s."
        assert base_mod._parse_retry_delay(msg) == pytest.approx(46.37, abs=0.01)

    def test_parses_retry_delay_field(self):
        msg = "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '42s'}"
        assert base_mod._parse_retry_delay(msg) == pytest.approx(42.0)

    def test_no_delay_returns_none(self):
        assert base_mod._parse_retry_delay("some other error") is None


class TestRateLimitRetry:
    def test_429_waits_server_delay_then_succeeds(self, fake_clock):
        llm = FakeLLM(fail_times=1, requests_per_minute=None)
        resp = llm.generate("hi")
        assert resp.text == "ok"
        assert llm.calls == 2
        # Waited at least the server-requested delay (46s), not the 1s backoff
        assert max(fake_clock["sleeps"]) >= 46.0

    def test_429_does_not_consume_regular_retries(self, fake_clock):
        # 5 rate-limit failures > retries=3, but rate limits have their own budget
        llm = FakeLLM(fail_times=5, requests_per_minute=None)
        resp = llm.generate("hi", retries=3)
        assert resp.text == "ok"
        assert llm.calls == 6

    def test_429_gives_up_after_max_rate_limit_retries(self, fake_clock):
        llm = FakeLLM(fail_times=100, requests_per_minute=None)
        with pytest.raises(RateLimitError):
            llm.generate("hi", retries=0, max_rate_limit_retries=2)
        assert llm.calls == 3


class TestTransientRetry:
    def test_transient_errors_retry_with_capped_backoff(self, fake_clock):
        class Flaky(FakeLLM):
            def _generate(self, prompt, temperature=None):
                self.calls += 1
                if self.calls <= 4:
                    raise ConnectionError("nodename nor servname provided, or not known")
                return LLMResponse(text="ok", model=self.model_name, provider=self.provider_name)

        llm = Flaky(requests_per_minute=None)
        assert llm.generate("hi").text == "ok"
        assert llm.calls == 5
        assert fake_clock["sleeps"] == [2.0, 4.0, 8.0, 16.0]

    def test_transient_gives_up_after_retries(self, fake_clock):
        class Dead(FakeLLM):
            def _generate(self, prompt, temperature=None):
                self.calls += 1
                raise ConnectionError("Server disconnected")

        llm = Dead(requests_per_minute=None)
        with pytest.raises(ConnectionError):
            llm.generate("hi")
        assert llm.calls == 7  # 1 + DEFAULT_TRANSIENT_RETRIES
        assert max(fake_clock["sleeps"]) == 60.0


class TestClientSidePacing:
    def test_requests_spaced_by_min_interval(self, fake_clock):
        llm = FakeLLM(requests_per_minute=15)  # 4s between requests
        llm.generate("a")
        llm.generate("b")
        llm.generate("c")
        # Two gaps of 4s each must have been slept
        assert sum(fake_clock["sleeps"]) == pytest.approx(8.0, abs=0.01)

    def test_limiter_shared_across_instances_of_same_model(self, fake_clock):
        a = FakeLLM(requests_per_minute=15)
        b = FakeLLM(requests_per_minute=15)
        a.generate("a")
        b.generate("b")
        assert sum(fake_clock["sleeps"]) == pytest.approx(4.0, abs=0.01)

    def test_no_pacing_when_disabled(self, fake_clock):
        llm = FakeLLM(requests_per_minute=None)
        llm.generate("a")
        llm.generate("b")
        assert fake_clock["sleeps"] == []
