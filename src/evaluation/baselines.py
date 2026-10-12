"""
Baseline measurements for Phase 0.

Establishes:
  1. Native error rate: how often the model gets the answer wrong on
     the original question (no rephrasing).
  2. Naive paraphrase variance: error rate under random, meaning-preserving
     rewrites (not optimized). This bounds the effect attributable to
     noise rather than to optimization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.models.base import BaseLLM
from src.models.budget import BudgetExceededError
from src.data.loader import QAItem
from src.evaluation.scorer import WrongnessScorer, ScoreResult
from src.evaluation.answer_extract import AnswerExtractor

logger = logging.getLogger(__name__)

NAIVE_PARAPHRASE_PROMPT = """Rewrite the following question in a different way while preserving its exact meaning. The rewrite should ask for the same factual information. Just output the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""


@dataclass
class BaselineResult:
    """Results from baseline measurement for a single question."""

    qa_item_id: str
    question: str
    ground_truth: str

    # Native error rate
    native_score: Optional[ScoreResult] = None

    # Naive paraphrase scores
    naive_paraphrases: list[str] = field(default_factory=list)
    naive_scores: list[ScoreResult] = field(default_factory=list)

    @property
    def native_error_rate(self) -> float:
        """Error rate on the original question."""
        if self.native_score is None:
            return 0.0
        return self.native_score.wrongness_score

    @property
    def naive_avg_error_rate(self) -> float:
        """Average error rate across naive paraphrases."""
        if not self.naive_scores:
            return 0.0
        return sum(s.wrongness_score for s in self.naive_scores) / len(self.naive_scores)

    @property
    def naive_max_error_rate(self) -> float:
        """Maximum error rate across naive paraphrases."""
        if not self.naive_scores:
            return 0.0
        return max(s.wrongness_score for s in self.naive_scores)


@dataclass
class AggregateBaseline:
    """Aggregated baseline statistics across all questions."""

    num_questions: int = 0
    native_error_rate: float = 0.0
    naive_avg_error_rate: float = 0.0
    naive_max_error_rate: float = 0.0
    num_already_wrong: int = 0       # Questions the model gets wrong natively
    num_attackable: int = 0          # Questions the model gets right (attack targets)


class BaselineMeasurer:
    """
    Measures baseline error rates for Phase 0.
    """

    def __init__(
        self,
        target_llm: BaseLLM,
        auxiliary_llm: BaseLLM,
        scorer: WrongnessScorer,
        num_naive_paraphrases: int = 5,
    ):
        self.target_llm = target_llm
        self.auxiliary_llm = auxiliary_llm
        self.scorer = scorer
        self.num_naive_paraphrases = num_naive_paraphrases

    def generate_naive_paraphrases(self, question: str, n: int) -> list[str]:
        """Generate random paraphrases using the auxiliary LLM."""
        paraphrases = []
        for _ in range(n):
            prompt = NAIVE_PARAPHRASE_PROMPT.format(question=question)
            response = self.auxiliary_llm.generate(prompt, temperature=0.9)
            paraphrase = response.text.strip().strip('"')
            if paraphrase:
                paraphrases.append(paraphrase)
        return paraphrases

    def measure_single(self, qa_item: QAItem) -> BaselineResult:
        """Measure baselines for a single question."""
        result = BaselineResult(
            qa_item_id=qa_item.id,
            question=qa_item.question,
            ground_truth=qa_item.answer,
        )

        # 1. Native error rate
        result.native_score = self.scorer.score_original(qa_item)
        logger.debug(
            f"[{qa_item.id}] Native wrongness: {result.native_error_rate:.2f}"
        )

        # 2. Naive paraphrases
        result.naive_paraphrases = self.generate_naive_paraphrases(
            qa_item.question, self.num_naive_paraphrases
        )
        result.naive_scores = [
            self.scorer.score(qa_item, p) for p in result.naive_paraphrases
        ]
        logger.debug(
            f"[{qa_item.id}] Naive avg wrongness: {result.naive_avg_error_rate:.2f}"
        )

        return result

    def measure_all(
        self,
        qa_items: list[QAItem],
        output_path: Optional[Path] = None,
    ) -> tuple[list[BaselineResult], AggregateBaseline]:
        """
        Measure baselines for all questions.

        Args:
            qa_items: List of QA items to measure.
            output_path: If provided, save results to this JSON file.

        Returns:
            Tuple of (per-question results, aggregate statistics).
        """
        results: list[BaselineResult] = []

        for qa_item in tqdm(qa_items, desc="Measuring baselines"):
            try:
                result = self.measure_single(qa_item)
            except BudgetExceededError as e:
                logger.error(f"Stopping baselines: {e}")
                break
            results.append(result)

        # Aggregate
        agg = self._aggregate(results)

        # Save if requested
        if output_path is not None:
            self._save_results(results, agg, output_path)

        return results, agg

    @staticmethod
    def _aggregate(results: list[BaselineResult]) -> AggregateBaseline:
        """Compute aggregate statistics."""
        if not results:
            return AggregateBaseline()

        native_rates = [r.native_error_rate for r in results]
        naive_avg_rates = [r.naive_avg_error_rate for r in results]
        naive_max_rates = [r.naive_max_error_rate for r in results]

        num_already_wrong = sum(1 for r in native_rates if r > 0.5)
        num_attackable = len(results) - num_already_wrong

        return AggregateBaseline(
            num_questions=len(results),
            native_error_rate=sum(native_rates) / len(native_rates),
            naive_avg_error_rate=sum(naive_avg_rates) / len(naive_avg_rates),
            naive_max_error_rate=sum(naive_max_rates) / len(naive_max_rates),
            num_already_wrong=num_already_wrong,
            num_attackable=num_attackable,
        )

    @staticmethod
    def _save_results(
        results: list[BaselineResult],
        agg: AggregateBaseline,
        path: Path,
    ) -> None:
        """Save results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "aggregate": asdict(agg),
            "per_question": [
                {
                    "id": r.qa_item_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "native_error_rate": r.native_error_rate,
                    "naive_avg_error_rate": r.naive_avg_error_rate,
                    "naive_max_error_rate": r.naive_max_error_rate,
                    "naive_paraphrases": r.naive_paraphrases,
                }
                for r in results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Baseline results saved to {path}")
