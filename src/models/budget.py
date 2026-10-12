"""
Per-run budget guard.

Hard client-side ceiling on the number of API requests and (estimated or
reported) USD spent during one run. When either cap is exceeded, the next
request raises BudgetExceededError so the pipeline can stop and save partial
results. This is a safety net on top of any provider-side limit.
"""

from __future__ import annotations

import threading
from typing import Optional


class BudgetExceededError(RuntimeError):
    """Raised when the per-run request or spend ceiling is exceeded."""


class BudgetGuard:
    def __init__(
        self,
        max_requests: Optional[int] = None,
        max_usd: Optional[float] = None,
        prices: Optional[dict[str, dict[str, float]]] = None,
    ):
        """
        Args:
            max_requests: Max API requests per run (None = unlimited).
            max_usd: Max spend per run in USD (None = unlimited).
            prices: {model_name: {"input_per_m": $, "output_per_m": $}} used to
                    estimate cost when the provider does not report it.
        """
        self.max_requests = max_requests
        self.max_usd = max_usd
        self.prices = prices or {}
        self.requests = 0
        self.spent_usd = 0.0
        self.estimated_requests = 0  # requests whose cost was estimated, not reported
        self._lock = threading.Lock()

    def _price(self, model: Optional[str]) -> Optional[dict[str, float]]:
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        # Allow "provider/model" -> "model" lookup
        short = model.split("/")[-1]
        return self.prices.get(short)

    def record(
        self,
        model: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Record one completed request."""
        with self._lock:
            self.requests += 1
            if cost_usd is not None:
                self.spent_usd += float(cost_usd)
                return
            price = self._price(model)
            if price is not None:
                self.spent_usd += (
                    prompt_tokens * price.get("input_per_m", 0.0)
                    + completion_tokens * price.get("output_per_m", 0.0)
                ) / 1_000_000
                self.estimated_requests += 1

    def check(self) -> None:
        """Raise BudgetExceededError if either cap has been exceeded."""
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise BudgetExceededError(
                    f"Request cap reached: {self.requests}/{self.max_requests}"
                )
            if self.max_usd is not None and self.spent_usd >= self.max_usd:
                raise BudgetExceededError(
                    f"Spend cap reached: ${self.spent_usd:.4f}/${self.max_usd:.2f}"
                )

    def summary(self) -> str:
        req_cap = str(self.max_requests) if self.max_requests is not None else "inf"
        usd_cap = f"{self.max_usd:.2f}" if self.max_usd is not None else "inf"
        est = " (estimated)" if self.estimated_requests else ""
        return f"requests {self.requests}/{req_cap}, spent ${self.spent_usd:.2f}/${usd_cap}{est}"


_BUDGET: Optional[BudgetGuard] = None


def get_budget() -> Optional[BudgetGuard]:
    return _BUDGET


def set_budget(budget: Optional[BudgetGuard]) -> None:
    global _BUDGET
    _BUDGET = budget
