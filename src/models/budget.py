"""
Budget guard: hard client-side ceiling on API requests and USD spent.

Every LLM call goes through BaseLLM.generate, which calls `check()` BEFORE
sending and `record()` after. When a cap is reached, the next request raises
BudgetExceededError so the pipeline can stop and save partial results.

Caps:
  - max_requests / max_usd: per run (this process).
  - max_usd_total: across ALL runs, persisted in a ledger file, so a
    baseline run + an attack run + a resumed attack share one ceiling.

Cost is the provider-reported cost when available (OpenRouter), else
estimated from token counts and the `prices` table. The estimate errs high:
  - a model with no price uses FALLBACK_PRICE (deliberately expensive);
  - a response with no token usage is charged as prompt-length/3 input tokens
    plus the full max_tokens as output.
Overshoot is bounded by the calls already in flight when the cap trips
(at most `concurrency` per client), each a fraction of a cent.

This is a safety net on top of provider-side limits, not a billing record:
the real bill can differ from the estimate.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Charged for any model missing from the price table. Higher than any model
# this project uses, so a missing price trips the cap early, not never.
FALLBACK_PRICE = {"input_per_m": 5.0, "output_per_m": 20.0}


class BudgetExceededError(RuntimeError):
    """Raised when a request or spend ceiling is reached."""


class BudgetGuard:
    def __init__(
        self,
        max_requests: Optional[int] = None,
        max_usd: Optional[float] = None,
        prices: Optional[dict[str, dict[str, float]]] = None,
        max_usd_total: Optional[float] = None,
        ledger_path: Optional[str | Path] = None,
    ):
        """
        Args:
            max_requests: Max API requests this run (None = unlimited).
            max_usd: Max spend this run in USD (None = unlimited).
            prices: {model_name: {"input_per_m": $, "output_per_m": $}}.
            max_usd_total: Max spend across all runs sharing `ledger_path`.
            ledger_path: JSON file accumulating spend across runs.
        """
        if max_usd_total is not None and ledger_path is None:
            raise ValueError("max_usd_total needs a ledger_path to persist spend across runs")
        self.max_requests = max_requests
        self.max_usd = max_usd
        self.max_usd_total = max_usd_total
        self.prices = prices or {}
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.requests = 0
        self.failed_requests = 0
        self.spent_usd = 0.0
        self.estimated_requests = 0  # requests whose cost was estimated, not reported
        self.unpriced_models: set[str] = set()
        self._lock = threading.Lock()
        self._prior_usd, self._prior_requests = self._read_ledger()

    # ------------------------------------------------------------------ prices
    def _price(self, model: Optional[str]) -> Optional[dict[str, float]]:
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        # Allow "provider/model" -> "model" lookup
        return self.prices.get(model.split("/")[-1])

    def missing_prices(self, models: list[str]) -> list[str]:
        """Models with no entry in the price table."""
        return [m for m in models if self._price(m) is None]

    # ------------------------------------------------------------------ ledger
    def _read_ledger(self) -> tuple[float, int]:
        if self.ledger_path is None or not self.ledger_path.exists():
            return 0.0, 0
        with open(self.ledger_path) as f:
            data = json.load(f)
        return float(data.get("total_usd", 0.0)), int(data.get("total_requests", 0))

    def _write_ledger(self) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_usd": self._prior_usd + self.spent_usd,
            "total_requests": self._prior_requests + self.requests,
            "max_usd_total": self.max_usd_total,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self.ledger_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.ledger_path)  # atomic: a crash never leaves a half-written ledger

    @property
    def total_usd(self) -> float:
        """Spend across all runs recorded in the ledger, including this one."""
        return self._prior_usd + self.spent_usd

    # ------------------------------------------------------------------ record
    def record(
        self,
        model: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: Optional[float] = None,
        prompt_chars: int = 0,
        max_output_tokens: int = 0,
    ) -> None:
        """Record one completed request."""
        with self._lock:
            self.requests += 1
            if cost_usd is not None:
                self.spent_usd += float(cost_usd)
            else:
                if not prompt_tokens and not completion_tokens and (prompt_chars or max_output_tokens):
                    # No usage reported: assume the worst.
                    prompt_tokens = prompt_chars // 3 + 1
                    completion_tokens = max_output_tokens
                price = self._price(model)
                if price is None:
                    if model and model not in self.unpriced_models:
                        logger.error(f"No price for model {model!r}; charging fallback {FALLBACK_PRICE}")
                        self.unpriced_models.add(model)
                    price = FALLBACK_PRICE
                self.spent_usd += (
                    prompt_tokens * price.get("input_per_m", 0.0)
                    + completion_tokens * price.get("output_per_m", 0.0)
                ) / 1_000_000
                self.estimated_requests += 1
            self._write_ledger()

    def record_failure(self) -> None:
        """Count a failed attempt toward the request cap (no token info, so no USD)."""
        with self._lock:
            self.requests += 1
            self.failed_requests += 1
            self._write_ledger()

    # ------------------------------------------------------------------ check
    def check(self) -> None:
        """Raise BudgetExceededError if any cap has been reached."""
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise BudgetExceededError(
                    f"Request cap reached: {self.requests}/{self.max_requests}"
                )
            if self.max_usd is not None and self.spent_usd >= self.max_usd:
                raise BudgetExceededError(
                    f"Run spend cap reached: ${self.spent_usd:.4f}/${self.max_usd:.4f}"
                )
            if self.max_usd_total is not None and self.total_usd >= self.max_usd_total:
                raise BudgetExceededError(
                    f"Total spend cap reached: ${self.total_usd:.4f}/${self.max_usd_total:.4f} "
                    f"(all runs, ledger {self.ledger_path})"
                )

    def summary(self) -> str:
        def usd(x: Optional[float]) -> str:
            if x is None:
                return "inf"
            return f"${x:.2f}" if x >= 1 else f"${x:.4f}"

        req_cap = str(self.max_requests) if self.max_requests is not None else "inf"
        est = " (estimated)" if self.estimated_requests else ""
        s = f"requests {self.requests}/{req_cap}, spent {usd(self.spent_usd)}/{usd(self.max_usd)}{est}"
        if self.failed_requests:
            s += f", {self.failed_requests} failed attempts"
        if self.ledger_path is not None:
            s += f"; all runs {usd(self.total_usd)}/{usd(self.max_usd_total)}"
        return s


_BUDGET: Optional[BudgetGuard] = None


def get_budget() -> Optional[BudgetGuard]:
    return _BUDGET


def set_budget(budget: Optional[BudgetGuard]) -> None:
    global _BUDGET
    _BUDGET = budget
