"""
Wrongness scoring for the attack.

Measures how strongly a paraphrase pushes the target model toward
an incorrect answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models.base import BaseLLM
from src.evaluation.answer_extract import AnswerExtractor, NO_ANSWER
from src.data.loader import QAItem

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Result of scoring a paraphrased question."""

    paraphrase: str
    wrongness_score: float        # Fraction of samples that were incorrect
    num_samples: int
    num_correct: int
    num_incorrect: int
    num_abstained: int
    abstention_policy: str = "exclude"
    responses: list[str] = None   # Raw responses
    extracted_answers: list[str] = None  # Extracted answers
    correct_flags: list[bool] = None     # Per-sample correctness (abstentions are False)
    abstained_flags: list[bool] = None   # Per-sample abstention

    @property
    def abstention_rate(self) -> float:
        return self.num_abstained / self.num_samples if self.num_samples else 0.0

    @property
    def accuracy(self) -> float:
        """Fraction of ALL samples answered correctly (abstentions count against)."""
        return self.num_correct / self.num_samples if self.num_samples else 0.0

    @property
    def knows_answer(self) -> bool:
        """
        True if the model answers correctly in most samples: the precondition
        for attacking a question. Not `wrongness_score <= 0.5`, which under
        abstention_policy='exclude' is 0.0 when every sample abstains.
        """
        return self.accuracy > 0.5


class WrongnessScorer:
    """
    Scores how effectively a paraphrase flips the target model's answer.

    The score is the empirical frequency of incorrect answers under
    repeated sampling. How abstentions ("I cannot answer") count is set by
    `abstention_policy`:
      - "exclude"  (default): score = wrong / (samples - abstained); a refusal is
                   not a factual error, so it neither helps nor hurts the attack.
      - "incorrect": score = (wrong + abstained) / samples (original behaviour).
      - "correct":   score = wrong / samples; abstaining counts as not fooled.
    Abstention rate is always reported separately.

    A score of 1.0 means the model always answers incorrectly.
    A score of 0.0 means the model always answers correctly.
    """

    def __init__(
        self,
        target_llm: BaseLLM,
        answer_extractor: AnswerExtractor,
        num_samples: int = 5,
        sampling_temperature: float = 0.7,
        answer_match_method: str = "fuzzy",
        abstention_policy: str = "exclude",
    ):
        if abstention_policy not in ("exclude", "incorrect", "correct"):
            raise ValueError(f"Unknown abstention_policy: {abstention_policy!r}")
        self.abstention_policy = abstention_policy
        self.target_llm = target_llm
        self.extractor = answer_extractor
        self.num_samples = num_samples
        self.sampling_temperature = sampling_temperature
        self.answer_match_method = answer_match_method

    def score(self, qa_item: QAItem, paraphrase: str) -> ScoreResult:
        """
        Score a paraphrased question by querying the target model.

        Args:
            qa_item: The original QA item with ground truth.
            paraphrase: The paraphrased question to evaluate.

        Returns:
            ScoreResult with wrongness score and detailed per-sample info.
        """
        # Query the target model multiple times
        responses = self.target_llm.sample(
            paraphrase,
            n=self.num_samples,
            temperature=self.sampling_temperature,
        )

        # Extract and evaluate answers
        raw_texts = [r.text for r in responses]
        extracted = [
            self.extractor.extract(paraphrase, r) for r in raw_texts
        ]

        correct_flags = []
        abstained_flags = []
        num_abstained = 0
        for ans in extracted:
            if ans == NO_ANSWER or AnswerExtractor.is_abstention(ans):
                correct_flags.append(False)
                abstained_flags.append(True)
                num_abstained += 1
            else:
                abstained_flags.append(False)
                is_correct = AnswerExtractor.answers_match(
                    predicted=ans,
                    ground_truth=qa_item.answer,
                    aliases=qa_item.aliases,
                    method=self.answer_match_method,
                )
                correct_flags.append(is_correct)

        num_correct = sum(correct_flags)
        num_samples = len(correct_flags)
        num_wrong = num_samples - num_correct - num_abstained  # confidently wrong
        wrongness = self._wrongness(num_wrong, num_abstained, num_samples)
        num_incorrect = num_samples - num_correct

        return ScoreResult(
            paraphrase=paraphrase,
            wrongness_score=wrongness,
            num_samples=len(correct_flags),
            num_correct=num_correct,
            num_incorrect=num_incorrect,
            num_abstained=num_abstained,
            abstention_policy=self.abstention_policy,
            responses=raw_texts,
            extracted_answers=extracted,
            correct_flags=correct_flags,
            abstained_flags=abstained_flags,
        )

    def _wrongness(self, num_wrong: int, num_abstained: int, num_samples: int) -> float:
        if num_samples == 0:
            return 0.0
        if self.abstention_policy == "incorrect":
            return (num_wrong + num_abstained) / num_samples
        if self.abstention_policy == "correct":
            return num_wrong / num_samples
        answered = num_samples - num_abstained
        return num_wrong / answered if answered else 0.0

    def score_original(self, qa_item: QAItem) -> ScoreResult:
        """Score the original (un-paraphrased) question. Used for baselines."""
        return self.score(qa_item, qa_item.question)
