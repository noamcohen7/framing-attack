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

    def test_unknown_model_is_charged_fallback_price(self):
        from src.models.budget import FALLBACK_PRICE
        b = BudgetGuard(max_requests=5, max_usd=1.0)
        b.record(model="unknown", prompt_tokens=1_000_000, completion_tokens=0)
        assert b.spent_usd == pytest.approx(FALLBACK_PRICE["input_per_m"])
        assert b.unpriced_models == {"unknown"}

    def test_missing_usage_is_charged_worst_case(self):
        b = BudgetGuard(max_usd=1.0, prices={"m": {"input_per_m": 1.0, "output_per_m": 1.0}})
        b.record(model="m", prompt_chars=3000, max_output_tokens=500)  # no token counts
        assert b.spent_usd == pytest.approx((1001 + 500) / 1_000_000)

    def test_failures_count_toward_request_cap(self):
        b = BudgetGuard(max_requests=2)
        b.record_failure()
        b.record_failure()
        with pytest.raises(BudgetExceededError):
            b.check()
        assert b.failed_requests == 2

    def test_missing_prices(self):
        b = BudgetGuard(prices={"gpt-oss-20b": {}})
        assert b.missing_prices(["openai/gpt-oss-20b", "other"]) == ["other"]


class TestLedger:
    def test_total_cap_spans_runs(self, tmp_path):
        ledger = tmp_path / "ledger.json"
        run1 = BudgetGuard(max_usd=10.0, max_usd_total=1.0, ledger_path=ledger)
        run1.record(cost_usd=0.7)
        run1.check()  # 0.7 < 1.0
        # New process, fresh per-run budget, same ledger
        run2 = BudgetGuard(max_usd=10.0, max_usd_total=1.0, ledger_path=ledger)
        assert run2.total_usd == pytest.approx(0.7)
        run2.record(cost_usd=0.3)
        with pytest.raises(BudgetExceededError, match="Total spend cap"):
            run2.check()

    def test_ledger_file_contents(self, tmp_path):
        import json
        ledger = tmp_path / "ledger.json"
        b = BudgetGuard(max_usd_total=5.0, ledger_path=ledger)
        b.record(cost_usd=0.25)
        b.record_failure()
        data = json.loads(ledger.read_text())
        assert data["total_usd"] == pytest.approx(0.25)
        assert data["total_requests"] == 2
        assert not ledger.with_suffix(".tmp").exists()

    def test_total_cap_requires_ledger(self):
        with pytest.raises(ValueError):
            BudgetGuard(max_usd_total=1.0)


class TestCapStopsRealCalls:
    """Integration with BaseLLM.generate: once the cap trips, no request is sent."""

    def _llm(self, fail=False):
        from typing import Optional
        from src.models.base import BaseLLM, LLMResponse

        class CountingLLM(BaseLLM):
            sent = 0

            @property
            def provider_name(self):
                return "fake"

            def _generate(self, prompt: str, temperature: Optional[float] = None):
                CountingLLM.sent += 1
                if fail:
                    raise RuntimeError("boom")
                return LLMResponse(text="x", model="m", provider="fake",
                                   usage={"prompt_tokens": 1_000_000, "completion_tokens": 0})

        return CountingLLM(model_name="m", requests_per_minute=None), CountingLLM

    def test_spend_cap_blocks_next_request(self):
        set_budget(BudgetGuard(max_usd=0.25, prices={"m": {"input_per_m": 0.10, "output_per_m": 0}}))
        llm, cls = self._llm()
        for _ in range(3):
            llm.generate("q")            # $0.10 each -> $0.30
        with pytest.raises(BudgetExceededError):
            llm.generate("q")
        assert cls.sent == 3             # the 4th request was never sent

    def test_failed_attempts_hit_request_cap(self):
        set_budget(BudgetGuard(max_requests=3))
        llm, cls = self._llm(fail=True)
        with pytest.raises(BudgetExceededError):
            llm.generate("q", retries=10, backoff=0.0)
        assert cls.sent == 3


def test_pipeline_refuses_unpriced_models(tmp_path):
    from src.attack.pipeline import AttackPipeline
    from src.config import Config
    cfg = Config()
    cfg.experiment.output_dir = str(tmp_path)
    cfg.budget.prices = {}
    with pytest.raises(ValueError, match="no entry"):
        AttackPipeline(cfg)


def test_strategy_batch_does_not_swallow_budget_error():
    from typing import Optional
    from src.attack.paraphraser import Paraphraser
    from src.models.base import BaseLLM

    class Broke(BaseLLM):
        @property
        def provider_name(self):
            return "fake"

        def _generate(self, prompt: str, temperature: Optional[float] = None):
            raise AssertionError("must not be called")

    set_budget(BudgetGuard(max_requests=0))
    with pytest.raises(BudgetExceededError):
        Paraphraser(Broke(model_name="m", requests_per_minute=None)).generate_strategy_batch("q")

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


class TestSharedLedger:
    def test_two_processes_add_not_overwrite(self, tmp_path):
        ledger = tmp_path / "ledger.json"
        a = BudgetGuard(max_usd_total=1.0, ledger_path=ledger)   # e.g. baseline process
        b = BudgetGuard(max_usd_total=1.0, ledger_path=ledger)   # e.g. attack process
        a.record(cost_usd=0.4)
        b.record(cost_usd=0.4)     # b must see a's 0.4, not overwrite it
        import json
        assert json.loads(ledger.read_text())["total_usd"] == pytest.approx(0.8)
        a.record(cost_usd=0.3)     # total 1.1 >= 1.0
        with pytest.raises(BudgetExceededError, match="Total"):
            a.check()

    def test_concurrent_threads_lose_no_spend(self, tmp_path):
        import json, threading
        ledger = tmp_path / "ledger.json"
        guards = [BudgetGuard(ledger_path=ledger) for _ in range(4)]

        def spend(g):
            for _ in range(50):
                g.record(cost_usd=0.01)

        ts = [threading.Thread(target=spend, args=(g,)) for g in guards]
        [t.start() for t in ts]
        [t.join() for t in ts]
        data = json.loads(ledger.read_text())
        assert data["total_usd"] == pytest.approx(2.0)
        assert data["total_requests"] == 200
