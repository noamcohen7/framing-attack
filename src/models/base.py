"""
Abstract base class for LLM clients.

Every provider (Gemini, Groq, HuggingFace, ...) implements this interface
so the rest of the codebase can swap models without code changes.

Also provides:
  - Client-side pacing (requests per minute), shared across all instances
    that talk to the same (provider, model) so target / auxiliary / judge
    clients don't jointly blow a per-model quota.
  - 429 / RESOURCE_EXHAUSTED handling that honours the server-requested
    retry delay instead of the short transient-error backoff.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.models.budget import get_budget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting helpers
# ---------------------------------------------------------------------------
_RETRY_IN_RE = re.compile(r"retry in\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)

DEFAULT_RATE_LIMIT_WAIT = 60.0  # seconds, when the server gives no hint
MAX_RATE_LIMIT_RETRIES = 10
DEFAULT_TRANSIENT_RETRIES = 6   # 2+4+8+16+32+60 s ≈ 2 min of patience for network blips
MAX_TRANSIENT_WAIT = 60.0


def _parse_retry_delay(message: str) -> Optional[float]:
    """Extract the server-suggested retry delay (seconds) from an error message."""
    for pattern in (_RETRY_IN_RE, _RETRY_DELAY_RE):
        m = pattern.search(message)
        if m:
            return float(m.group(1))
    return None


def _is_rate_limit_error(error: Exception) -> bool:
    """Heuristically detect quota / rate-limit errors across providers."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code == 429:
        return True
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


class _RateLimiter:
    """Minimum-interval limiter shared by every client of one model."""

    def __init__(self, requests_per_minute: float):
        self.min_interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                delay = self._next_allowed - now
                logger.debug(f"Rate limiter: sleeping {delay:.2f}s")
                time.sleep(delay)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


_RATE_LIMITERS: dict[tuple[str, str], _RateLimiter] = {}
_RATE_LIMITERS_LOCK = threading.Lock()


def _get_rate_limiter(key: tuple[str, str], requests_per_minute: float) -> _RateLimiter:
    with _RATE_LIMITERS_LOCK:
        limiter = _RATE_LIMITERS.get(key)
        if limiter is None or limiter.min_interval != 60.0 / requests_per_minute:
            limiter = _RateLimiter(requests_per_minute)
            _RATE_LIMITERS[key] = limiter
        return limiter


# ---------------------------------------------------------------------------
# Response / client
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, etc.
    raw: Optional[object] = None  # Provider-specific raw response


class BaseLLM(ABC):
    """Abstract LLM client interface."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        requests_per_minute: Optional[float] = None,
        concurrency: int = 1,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.requests_per_minute = requests_per_minute
        self.concurrency = max(1, int(concurrency or 1))

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'gemini', 'groq')."""
        ...

    @abstractmethod
    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        """Provider-specific generation. Subclasses implement this."""
        ...

    def _rate_limiter(self) -> Optional[_RateLimiter]:
        if not self.requests_per_minute:
            return None
        return _get_rate_limiter(
            (self.provider_name, self.model_name), self.requests_per_minute
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        retries: int = DEFAULT_TRANSIENT_RETRIES,
        backoff: float = 2.0,
        max_rate_limit_retries: int = MAX_RATE_LIMIT_RETRIES,
    ) -> LLMResponse:
        """
        Generate a response with automatic retry + exponential backoff.

        Transient errors are retried `retries` times with exponential backoff.
        Rate-limit (429) errors have a separate budget (`max_rate_limit_retries`)
        and wait for the server-suggested delay (or 60s if none is given).

        Args:
            prompt: The input prompt.
            temperature: Override the default temperature for this call.
            retries: Number of retries on transient errors.
            backoff: Backoff multiplier between retries.
            max_rate_limit_retries: Number of retries on rate-limit errors.
        """
        limiter = self._rate_limiter()
        last_error: Optional[Exception] = None
        transient_attempt = 0
        rate_limit_attempt = 0

        budget = get_budget()
        while True:
            if budget is not None:
                budget.check()
            if limiter is not None:
                limiter.wait()
            try:
                response = self._generate(prompt, temperature=temperature)
                if budget is not None:
                    budget.record(
                        model=self.model_name,
                        prompt_tokens=response.usage.get("prompt_tokens", 0) or 0,
                        completion_tokens=response.usage.get("completion_tokens", 0) or 0,
                        cost_usd=response.usage.get("cost_usd"),
                    )
                return response
            except Exception as e:
                last_error = e
                if _is_rate_limit_error(e):
                    if rate_limit_attempt >= max_rate_limit_retries:
                        logger.error(
                            f"[{self.provider_name}] Rate limited; gave up after "
                            f"{rate_limit_attempt} retries."
                        )
                        break
                    rate_limit_attempt += 1
                    wait = _parse_retry_delay(str(e))
                    wait = (wait + 1.0) if wait is not None else DEFAULT_RATE_LIMIT_WAIT
                    logger.warning(
                        f"[{self.provider_name}] Rate limited "
                        f"(retry {rate_limit_attempt}/{max_rate_limit_retries}); "
                        f"waiting {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    continue

                if transient_attempt >= retries:
                    logger.error(
                        f"[{self.provider_name}] All {retries + 1} attempts failed."
                    )
                    break
                wait = min(backoff ** (transient_attempt + 1), MAX_TRANSIENT_WAIT)
                transient_attempt += 1
                logger.warning(
                    f"[{self.provider_name}] Attempt {transient_attempt} failed: {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        raise last_error  # type: ignore[misc]

    def generate_batch(
        self,
        prompts: list[str],
        temperature: Optional[float] = None,
    ) -> list[LLMResponse]:
        """Generate responses for multiple prompts (concurrent if concurrency > 1)."""
        return self._run_many([(p, temperature) for p in prompts])

    def sample(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.7,
    ) -> list[LLMResponse]:
        """
        Sample multiple independent responses for the same prompt.

        Used for estimating empirical answer distributions.
        """
        return self._run_many([(prompt, temperature)] * n)

    def _run_many(self, jobs: list[tuple[str, Optional[float]]]) -> list[LLMResponse]:
        """Run generate() over jobs, preserving order; parallel when concurrency > 1."""
        if self.concurrency <= 1 or len(jobs) <= 1:
            return [self.generate(p, temperature=t) for p, t in jobs]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(jobs))) as ex:
            return list(ex.map(lambda j: self.generate(j[0], temperature=j[1]), jobs))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, provider={self.provider_name!r})"
