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
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
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

NAIVE_PARAPHRASE_PROMPT = """Rewrite the following question {n} different ways while preserving its exact meaning. Each rewrite must ask for the same factual information. Make the rewrites clearly different from each other in wording and structure, not just one or two swapped words.{avoid}

Output exactly {n} rewrites, one per line, numbered "1.", "2.", and so on. Output nothing else.

Original question: {question}"""

_AVOID_CLAUSE = "\n\nDo not repeat any of these existing rewrites:\n{existing}"

# Normalized-text similarity above which two rewrites count as duplicates.
_DUPLICATE_SIMILARITY = 0.9


def _normalize_question(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def parse_numbered_rewrites(text: str) -> list[str]:
    """Pull rewrites out of a numbered list ("1. ...", "2) ...")."""
    rewrites = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+\s*[.):]\s*(.+)$", line)
        if match:
            rewrite = match.group(1).strip().strip('"').strip()
            if rewrite:
                rewrites.append(rewrite)
    return rewrites


def dedupe_rewrites(original: str, candidates: list[str], existing: list[str] = ()) -> list[str]:
    """Drop rewrites that are near-duplicates of the original, `existing`, or each other."""
    kept: list[str] = []
    seen = [_normalize_question(original)] + [_normalize_question(e) for e in existing]
    for cand in candidates:
        norm = _normalize_question(cand)
        if not norm:
            continue
        if any(SequenceMatcher(None, norm, s).ratio() >= _DUPLICATE_SIMILARITY for s in seen):
            continue
        kept.append(cand)
        seen.append(norm)
    return kept


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
        min_native_accuracy: float = 0.8,
    ):
        self.target_llm = target_llm
        self.auxiliary_llm = auxiliary_llm
        self.scorer = scorer
        self.num_naive_paraphrases = num_naive_paraphrases
        self.min_native_accuracy = min_native_accuracy

    def generate_naive_paraphrases(self, question: str, n: int, max_calls: int = 2) -> list[str]:
        """
        Generate up to n distinct paraphrases using the auxiliary LLM.

        Asks for all n in one prompt (separate calls at temperature 0.9 mostly
        returned the same rewrite), drops near-duplicates, and makes one
        follow-up call for the shortfall.
        """
        paraphrases: list[str] = []
        for _ in range(max_calls):
            missing = n - len(paraphrases)
            if missing <= 0:
                break
            avoid = ""
            if paraphrases:
                avoid = _AVOID_CLAUSE.format(existing="\n".join(f"- {p}" for p in paraphrases))
            prompt = NAIVE_PARAPHRASE_PROMPT.format(n=missing, question=question, avoid=avoid)
            response = self.auxiliary_llm.generate(prompt, temperature=0.9)
            new = dedupe_rewrites(question, parse_numbered_rewrites(response.text), paraphrases)
            paraphrases.extend(new[:missing])
        if len(paraphrases) < n:
            logger.warning(f"Only {len(paraphrases)}/{n} distinct naive paraphrases for: {question[:60]}")
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
        workers: int = 1,
    ) -> tuple[list[BaselineResult], AggregateBaseline]:
        """
        Measure baselines for all questions.

        Args:
            qa_items: List of QA items to measure.
            output_path: If provided, save results to this JSON file, after
                every question (so a crash keeps what was measured).
            workers: Questions measured in parallel.

        Returns:
            Tuple of (per-question results in input order, aggregate statistics).
        """
        order = {q.id: i for i, q in enumerate(qa_items)}
        results: list[BaselineResult] = []
        lock = threading.Lock()
        stop = threading.Event()

        def ordered() -> list[BaselineResult]:
            return sorted(results, key=lambda r: order[r.qa_item_id])

        def measure_one(qa_item: QAItem) -> None:
            if stop.is_set():
                return
            try:
                result = self.measure_single(qa_item)
            except BudgetExceededError as e:
                if not stop.is_set():
                    logger.error(f"Stopping baselines: {e}")
                stop.set()
                return
            with lock:
                results.append(result)
                if output_path is not None:
                    snapshot = ordered()
                    self._save_results(snapshot, self._aggregate(snapshot, self.min_native_accuracy), output_path)

        if workers <= 1:
            for qa_item in tqdm(qa_items, desc="Measuring baselines"):
                measure_one(qa_item)
                if stop.is_set():
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(measure_one, q) for q in qa_items]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Measuring baselines"):
                    pass

        results = ordered()
        agg = self._aggregate(results, self.min_native_accuracy)
        if output_path is not None:
            self._save_results(results, agg, output_path)
        return results, agg

    @staticmethod
    def _aggregate(results: list[BaselineResult], min_native_accuracy: float = 0.8) -> AggregateBaseline:
        """Compute aggregate statistics."""
        if not results:
            return AggregateBaseline()

        native_rates = [r.native_error_rate for r in results]
        naive_avg_rates = [r.naive_avg_error_rate for r in results]
        naive_max_rates = [r.naive_max_error_rate for r in results]

        # Attackable = model answers correctly in most native samples. A question
        # it always refuses has native_error_rate 0.0 under 'exclude' but is not
        # attackable; count it with the already-wrong ones.
        num_attackable = sum(
            1 for r in results
            if r.native_score and r.native_score.knows_answer(min_native_accuracy)
        )
        num_already_wrong = len(results) - num_attackable

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
                    "native_accuracy": r.native_score.accuracy if r.native_score else None,
                    "native_abstained": r.native_score.num_abstained if r.native_score else None,
                    "native_answers": r.native_score.extracted_answers if r.native_score else None,
                    "naive_avg_error_rate": r.naive_avg_error_rate,
                    "naive_max_error_rate": r.naive_max_error_rate,
                    "naive_paraphrases": r.naive_paraphrases,
                    "naive_error_rates": [sc.wrongness_score for sc in r.naive_scores],
                }
                for r in results
            ],
        }

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)  # atomic

        logger.info(f"Baseline results saved to {path}")
