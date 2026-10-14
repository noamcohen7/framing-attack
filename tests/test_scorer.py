"""Tests for wrongness scoring and abstention handling (fake target, no API)."""

from typing import Optional

import pytest

from src.data.loader import QAItem
from src.evaluation.answer_extract import AnswerExtractor, NO_ANSWER
from src.evaluation.scorer import WrongnessScorer
from src.models.base import BaseLLM, LLMResponse


class ScriptedLLM(BaseLLM):
    """Returns the scripted responses in order."""

    def __init__(self, responses):
        super().__init__(model_name="scripted")
        self.responses = list(responses)

    @property
    def provider_name(self) -> str:
        return "fake"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        return LLMResponse(text=self.responses.pop(0), model=self.model_name, provider="fake")


QA = QAItem(id="q", question="Who won?", answer="Donald Trump", aliases=["Trump"])
# 1 correct, 1 wrong, 2 refusals
RESPONSES = ["Donald Trump.", "Kamala Harris.", "I cannot answer that.", "I don't know."]
EXTRACTED = ["Donald Trump", "Kamala Harris", "NO_ANSWER", "no_answer"]


class TestExtractorAbstention:
    def test_no_answer_sentinel_survives_normalization(self):
        ext = AnswerExtractor(method="regex", normalize=True)
        assert ext.extract("q", "NO_ANSWER") == NO_ANSWER
        assert ext.extract("q", "no_answer") == NO_ANSWER

    def test_is_abstention(self):
        assert AnswerExtractor.is_abstention("NO_ANSWER")
        assert AnswerExtractor.is_abstention("noanswer")
        assert AnswerExtractor.is_abstention("")
        assert not AnswerExtractor.is_abstention("donald trump")


def _scorer(policy):
    target = ScriptedLLM(RESPONSES)
    extractor = ScriptedLLM(EXTRACTED)
    ext = AnswerExtractor(method="llm", llm=extractor, normalize=True)
    return WrongnessScorer(target, ext, num_samples=4, abstention_policy=policy)


class TestAbstentionPolicy:
    def test_exclude_policy_ignores_refusals(self):
        r = _scorer("exclude").score(QA, "Who won?")
        assert r.num_samples == 4 and r.num_correct == 1 and r.num_abstained == 2
        assert r.abstention_rate == 0.5
        assert r.wrongness_score == pytest.approx(0.5)  # 1 wrong / 2 answered
        assert r.abstained_flags == [False, False, True, True]

    def test_incorrect_policy_counts_refusals_as_wrong(self):
        r = _scorer("incorrect").score(QA, "Who won?")
        assert r.wrongness_score == pytest.approx(0.75)

    def test_correct_policy_counts_refusals_as_not_fooled(self):
        r = _scorer("correct").score(QA, "Who won?")
        assert r.wrongness_score == pytest.approx(0.25)

    def test_all_refusals_scores_zero_under_exclude(self):
        target = ScriptedLLM(["I can't say."] * 2)
        ext = AnswerExtractor(method="llm", llm=ScriptedLLM(["NO_ANSWER"] * 2), normalize=True)
        r = WrongnessScorer(target, ext, num_samples=2).score(QA, "Who won?")
        assert r.wrongness_score == 0.0 and r.abstention_rate == 1.0

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError):
            _scorer("bogus")


class TestKnowsAnswer:
    """Attackability: correct in >= min_accuracy of ALL samples; all-abstain is not known."""

    def _result(self, correct, abstained, total=5):
        from src.evaluation.scorer import ScoreResult
        return ScoreResult(
            paraphrase="q", wrongness_score=0.0, num_samples=total,
            num_correct=correct, num_incorrect=total - correct, num_abstained=abstained,
        )

    def test_all_abstain_is_not_known(self):
        r = self._result(correct=0, abstained=5)
        assert r.accuracy == 0.0 and not r.knows_answer()

    def test_four_of_five_is_known(self):
        assert self._result(correct=4, abstained=1).knows_answer()

    def test_three_of_five_is_not_known(self):
        assert not self._result(correct=3, abstained=0).knows_answer()

    def test_threshold_is_configurable(self):
        assert self._result(correct=3, abstained=0).knows_answer(min_accuracy=0.6)

    def test_zero_samples_is_not_known(self):
        assert not self._result(correct=0, abstained=0, total=0).knows_answer(min_accuracy=0.0)

    def test_all_abstain_scores_zero_wrongness_under_exclude(self):
        scorer = WrongnessScorer(
            target_llm=ScriptedLLM(["I don't know."] * 3),
            answer_extractor=AnswerExtractor(method="regex"),
            num_samples=3, abstention_policy="exclude",
        )
        r = scorer.score_original(QA)
        assert r.wrongness_score == 0.0      # why wrongness alone can't gate attacks
        assert not r.knows_answer()
