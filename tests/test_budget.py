"""Tests for the per-run budget guard."""

import pytest

from src.models.budget import BudgetGuard, BudgetExceededError, get_budget, set_budget


@pytest.fixture(autouse=True)
def reset_budget():
    set_budget(None)
    yield
    set_budget(None)


class TestBudgetGuard:
    def test_counts_requests_and_cost(self):
        b = BudgetGuard(max_requests=10, max_usd=1.0)
        b.record(cost_usd=0.25)
        b.record(cost_usd=0.25)
        assert b.requests == 2
        assert b.spent_usd == pytest.approx(0.5)

    def test_raises_when_request_cap_exceeded(self):
        b = BudgetGuard(max_requests=2, max_usd=None)
        b.record()
        b.record()
        with pytest.raises(BudgetExceededError):
            b.check()

    def test_raises_when_usd_cap_exceeded(self):
        b = BudgetGuard(max_requests=None, max_usd=0.10)
        b.record(cost_usd=0.11)
        with pytest.raises(BudgetExceededError):
            b.check()

    def test_estimates_cost_from_tokens_when_not_reported(self):
        b = BudgetGuard(max_requests=None, max_usd=1.0,
                        prices={"m": {"input_per_m": 0.10, "output_per_m": 0.40}})
        b.record(model="m", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert b.spent_usd == pytest.approx(0.50)

    def test_reported_cost_wins_over_estimate(self):
        b = BudgetGuard(max_requests=None, max_usd=1.0,
                        prices={"m": {"input_per_m": 100, "output_per_m": 100}})
        b.record(model="m", prompt_tokens=1000, completion_tokens=1000, cost_usd=0.001)
        assert b.spent_usd == pytest.approx(0.001)

    def test_unknown_model_without_cost_counts_requests_only(self):
        b = BudgetGuard(max_requests=5, max_usd=1.0)
        b.record(model="unknown", prompt_tokens=10, completion_tokens=10)
        assert b.spent_usd == 0.0
        assert b.requests == 1

    def test_global_budget_accessor(self):
        assert get_budget() is None
        b = BudgetGuard(max_requests=1, max_usd=None)
        set_budget(b)
        assert get_budget() is b

    def test_summary_string(self):
        b = BudgetGuard(max_requests=100, max_usd=2.0)
        b.record(cost_usd=0.5)
        s = b.summary()
        assert "1/100" in s and "0.50" in s
