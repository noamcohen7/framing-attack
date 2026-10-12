"""
Evolutionary search controller.

Orchestrates the iterative optimization loop:
  1. Generate candidate rephrasings (initial population or mutations)
  2. Filter through semantic equivalence validator
  3. Score survivors by wrongness
  4. Select, mutate, and recombine the best candidates
  5. Repeat until convergence or budget exhausted
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from src.attack.paraphraser import Paraphraser
from src.equivalence.validator import SemanticEquivalenceValidator
from src.evaluation.scorer import WrongnessScorer, ScoreResult
from src.data.loader import QAItem

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A candidate rephrasing with its score."""

    paraphrase: str
    score: float = 0.0          # Wrongness score (0 = correct, 1 = always wrong)
    generation: int = 0         # Which generation this was created in
    parent: Optional[str] = None  # Parent paraphrase (for lineage tracking)
    score_result: Optional[ScoreResult] = None


@dataclass
class EvolutionResult:
    """Result of the evolutionary search for a single question."""

    qa_item: QAItem
    best_candidate: Optional[Candidate] = None
    all_candidates: list[Candidate] = field(default_factory=list)
    generations_run: int = 0
    total_candidates_generated: int = 0
    total_candidates_valid: int = 0      # Passed equivalence check
    total_candidates_scored: int = 0
    success: bool = False                # Did we flip the answer?

    @property
    def best_score(self) -> float:
        return self.best_candidate.score if self.best_candidate else 0.0


class EvolutionarySearch:
    """
    Evolutionary optimizer for finding adversarial rephrasings.
    """

    def __init__(
        self,
        paraphraser: Paraphraser,
        validator: SemanticEquivalenceValidator,
        scorer: WrongnessScorer,
        population_size: int = 10,
        num_generations: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.2,
        elite_count: int = 2,
        early_stop_score: float = 1.0,
        seed: int = 42,
    ):
        self.paraphraser = paraphraser
        self.validator = validator
        self.scorer = scorer
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.early_stop_score = early_stop_score
        self.rng = random.Random(seed)

    def search(self, qa_item: QAItem) -> EvolutionResult:
        """
        Run the evolutionary search for a single question.

        Args:
            qa_item: The factual QA item to attack.

        Returns:
            EvolutionResult with the best adversarial rephrasing found.
        """
        result = EvolutionResult(qa_item=qa_item)
        question = qa_item.question

        logger.info(f"Starting evolutionary search for: {question[:80]}...")

        # --- Generation 0: Initial population ---
        raw_candidates = self.paraphraser.generate_initial(
            question, n=self.population_size
        )
        # Also add strategy-based candidates for diversity
        strategy_candidates = self.paraphraser.generate_strategy_batch(question)
        raw_candidates.extend(strategy_candidates)

        result.total_candidates_generated += len(raw_candidates)

        # Filter through equivalence validator
        population = self._filter_and_score(
            qa_item, raw_candidates, generation=0, result=result
        )

        if not population:
            logger.warning("No valid candidates in initial population")
            return result

        # Sort by score (descending)
        population.sort(key=lambda c: c.score, reverse=True)
        result.best_candidate = population[0]

        # Check early stop
        if result.best_candidate.score >= self.early_stop_score:
            result.success = True
            result.generations_run = 1
            logger.info(
                f"Early stop at gen 0: score={result.best_score:.2f}"
            )
            return result

        # --- Generations 1..N ---
        for gen in range(1, self.num_generations):
            new_candidates: list[str] = []

            # Elite selection: keep top candidates unchanged
            elites = population[: self.elite_count]

            # Mutation: mutate top candidates
            num_mutations = max(1, int(self.population_size * self.mutation_rate))
            for _ in range(num_mutations):
                parent = self._tournament_select(population)
                mutations = self.paraphraser.mutate(
                    question, parent.paraphrase, n=2
                )
                new_candidates.extend(mutations)

            # Crossover: combine pairs of top candidates
            num_crossovers = max(1, int(self.population_size * self.crossover_rate))
            for _ in range(num_crossovers):
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)
                if parent_a.paraphrase != parent_b.paraphrase:
                    offspring = self.paraphraser.crossover(
                        question, parent_a.paraphrase, parent_b.paraphrase, n=1
                    )
                    new_candidates.extend(offspring)

            result.total_candidates_generated += len(new_candidates)

            # Filter and score new candidates
            new_scored = self._filter_and_score(
                qa_item, new_candidates, generation=gen, result=result
            )

            # Merge: elites + new scored, keep top population_size
            combined = [
                Candidate(
                    paraphrase=e.paraphrase,
                    score=e.score,
                    generation=e.generation,
                    score_result=e.score_result,
                )
                for e in elites
            ] + new_scored
            combined.sort(key=lambda c: c.score, reverse=True)
            population = combined[: self.population_size]

            # Update best
            if population and population[0].score > result.best_score:
                result.best_candidate = population[0]

            logger.info(
                f"Gen {gen}: best_score={result.best_score:.2f}, "
                f"pop_size={len(population)}"
            )

            # Early stop
            if result.best_score >= self.early_stop_score:
                result.success = True
                result.generations_run = gen + 1
                logger.info(f"Early stop at gen {gen}")
                return result

        result.generations_run = self.num_generations
        result.success = result.best_score > 0.5  # Majority wrong = success
        return result

    def _filter_and_score(
        self,
        qa_item: QAItem,
        candidates: list[str],
        generation: int,
        result: EvolutionResult,
    ) -> list[Candidate]:
        """Filter candidates through equivalence validator and score survivors."""
        scored: list[Candidate] = []

        for paraphrase in candidates:
            # Deduplicate
            if any(c.paraphrase == paraphrase for c in result.all_candidates):
                continue

            # Equivalence check
            eq_result = self.validator.validate(qa_item.question, paraphrase)
            if not eq_result.is_equivalent:
                logger.debug(f"Rejected: {paraphrase[:60]}... ({eq_result.reason})")
                continue

            result.total_candidates_valid += 1

            # Score
            score_result = self.scorer.score(qa_item, paraphrase)
            result.total_candidates_scored += 1

            candidate = Candidate(
                paraphrase=paraphrase,
                score=score_result.wrongness_score,
                generation=generation,
                score_result=score_result,
            )
            scored.append(candidate)
            result.all_candidates.append(candidate)

        return scored

    def _tournament_select(
        self, population: list[Candidate], k: int = 3
    ) -> Candidate:
        """Tournament selection: pick the best of k random candidates."""
        k = min(k, len(population))
        tournament = self.rng.sample(population, k)
        return max(tournament, key=lambda c: c.score)
