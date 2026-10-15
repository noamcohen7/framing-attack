"""Tests for the combined validator with fake NLI / judge (no models, no API)."""

from typing import Optional

import pytest

from src.equivalence.validator import SemanticEquivalenceValidator
from src.models.base import BaseLLM, LLMResponse


class FakeJudgeLLM(BaseLLM):
    def __init__(self, verdict: str):
        super().__init__(model_name="judge")
        self.verdict = verdict
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.verdict, model="judge", provider="fake")


class FakeNLI:
    def __init__(self, score: float, threshold: float = 0.85):
        self.score, self.threshold = score, threshold

    def check_equivalence(self, original, paraphrase):
        return {"is_equivalent": self.score >= self.threshold, "forward_score": self.score,
                "backward_score": self.score, "min_score": self.score}


class TestJudgeOnlyMode:
    def test_nli_none_uses_judge_only(self):
        judge = FakeJudgeLLM("YES. Same question.")
        v = SemanticEquivalenceValidator(nli_model=None, llm=judge)
        r = v.validate("Who won?", "Who was the winner?")
        assert r.is_equivalent and r.nli_result is None and judge.calls == 1
        assert "judge approved" in r.reason

    def test_judge_only_rejects(self):
        v = SemanticEquivalenceValidator(nli_model=None, llm=FakeJudgeLLM("NO. Scope differs."))
        r = v.validate("Who won?", "Who lost?")
        assert not r.is_equivalent and "rejected" in r.reason

    def test_no_nli_and_no_judge_is_an_error(self):
        with pytest.raises(ValueError):
            SemanticEquivalenceValidator(nli_model=None, llm=None)


class TestCombinedMode:
    def test_nli_reject_short_circuits_judge(self):
        judge = FakeJudgeLLM("YES")
        v = SemanticEquivalenceValidator(nli=FakeNLI(0.2), llm=judge, require_both=True)
        r = v.validate("a", "b")
        assert not r.is_equivalent and judge.calls == 0

    def test_both_must_agree(self):
        judge = FakeJudgeLLM("NO. Different.")
        v = SemanticEquivalenceValidator(nli=FakeNLI(0.99), llm=judge, require_both=True)
        assert not v.validate("a", "b").is_equivalent
        v2 = SemanticEquivalenceValidator(nli=FakeNLI(0.99), llm=FakeJudgeLLM("YES"), require_both=True)
        assert v2.validate("a", "b").is_equivalent

    def test_filter_valid(self):
        v = SemanticEquivalenceValidator(nli=FakeNLI(0.99), llm=FakeJudgeLLM("YES"))
        kept = v.filter_valid("q", ["p1", "p2"])
        assert [p for p, _ in kept] == ["p1", "p2"]


def test_strict_judge_uses_strict_prompt():
    from src.equivalence.llm_judge import LLMJudge, STRICT_JUDGE_PROMPT, JUDGE_PROMPT
    seen = []

    class Rec:
        def generate(self, prompt, temperature=None):
            seen.append(prompt)
            from src.models.base import LLMResponse
            return LLMResponse(text="NO. Different answer type.", model="m", provider="fake")

    assert LLMJudge(Rec(), strict=True).check_equivalence("A?", "B?")["is_equivalent"] is False
    LLMJudge(Rec()).check_equivalence("A?", "B?")
    assert "KIND of answer" in seen[0] and "KIND of answer" not in seen[1]
    assert STRICT_JUDGE_PROMPT != JUDGE_PROMPT
