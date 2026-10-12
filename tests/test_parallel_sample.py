"""Tests for concurrent sampling in BaseLLM.sample."""

import threading
import time
from typing import Optional

from src.models.base import BaseLLM, LLMResponse


class SlowLLM(BaseLLM):
    def __init__(self, **kw):
        super().__init__(model_name="slow", **kw)
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        return "fake"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return LLMResponse(text=prompt, model=self.model_name, provider=self.provider_name)


def test_sample_runs_concurrently_when_configured():
    llm = SlowLLM(concurrency=4)
    out = llm.sample("p", n=8, temperature=0.7)
    assert [r.text for r in out] == ["p"] * 8
    assert llm.max_active > 1


def test_sample_is_sequential_by_default():
    llm = SlowLLM()
    out = llm.sample("p", n=3)
    assert len(out) == 3
    assert llm.max_active == 1


def test_generate_batch_preserves_order_when_concurrent():
    llm = SlowLLM(concurrency=4)
    out = llm.generate_batch([str(i) for i in range(10)])
    assert [r.text for r in out] == [str(i) for i in range(10)]
