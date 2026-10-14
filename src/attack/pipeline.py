"""
End-to-end attack orchestration.

Ties together all components:
  - Dataset loading
  - Model setup
  - Equivalence validation
  - Evolutionary search
  - Result collection and serialization
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.config import Config, load_config
from src.models import create_llm_from_config
from src.models.budget import BudgetGuard, BudgetExceededError, set_budget, get_budget
from src.data import load_dataset, QAItem
from src.equivalence import SemanticEquivalenceValidator
from src.evaluation import AnswerExtractor, WrongnessScorer
from src.evaluation.baselines import BaselineMeasurer
from src.attack.paraphraser import Paraphraser
from src.attack.evolutionary import EvolutionarySearch, EvolutionResult

logger = logging.getLogger(__name__)


class AttackPipeline:
    """
    Full attack pipeline from dataset to results.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self._setup_logging()
        set_budget(
            BudgetGuard(
                max_requests=self.config.budget.max_requests,
                max_usd=self.config.budget.max_usd,
                prices=self.config.budget.prices,
            )
        )

        # Will be initialized lazily
        self._target_llm = None
        self._auxiliary_llm = None
        self._validator = None
        self._scorer = None
        self._paraphraser = None
        self._searcher = None

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.experiment.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @property
    def target_llm(self):
        if self._target_llm is None:
            logger.info("Initializing target LLM...")
            self._target_llm = create_llm_from_config(self.config.target_model)
        return self._target_llm

    @property
    def auxiliary_llm(self):
        if self._auxiliary_llm is None:
            logger.info("Initializing auxiliary LLM...")
            self._auxiliary_llm = create_llm_from_config(self.config.auxiliary_model)
        return self._auxiliary_llm

    @property
    def validator(self):
        if self._validator is None:
            logger.info("Initializing equivalence validator...")
            eq_cfg = self.config.equivalence
            judge_llm = create_llm_from_config(
                type(self.config.target_model)(
                    provider=eq_cfg.llm_judge.provider,
                    model_name=eq_cfg.llm_judge.model_name,
                    temperature=eq_cfg.llm_judge.temperature,
                    requests_per_minute=eq_cfg.llm_judge.requests_per_minute,
                    concurrency=eq_cfg.llm_judge.concurrency,
                )
            )
            self._validator = SemanticEquivalenceValidator(
                nli_model=eq_cfg.nli_model,
                nli_threshold=eq_cfg.nli_threshold,
                llm=judge_llm,
                require_both=eq_cfg.require_both,
            )
        return self._validator

    @property
    def scorer(self):
        if self._scorer is None:
            ext_cfg = self.config.answer_extraction
            extractor = AnswerExtractor(
                method=ext_cfg.method,
                llm=self.auxiliary_llm if ext_cfg.method != "regex" else None,
                normalize=ext_cfg.normalize,
            )
            self._scorer = WrongnessScorer(
                target_llm=self.target_llm,
                answer_extractor=extractor,
                num_samples=self.config.target_model.num_samples,
                answer_match_method=ext_cfg.equivalence_check,
                abstention_policy=ext_cfg.abstention_policy,
            )
        return self._scorer

    @property
    def paraphraser(self):
        if self._paraphraser is None:
            self._paraphraser = Paraphraser(self.auxiliary_llm)
        return self._paraphraser

    @property
    def searcher(self):
        if self._searcher is None:
            evo = self.config.evolution
            self._searcher = EvolutionarySearch(
                paraphraser=self.paraphraser,
                validator=self.validator,
                scorer=self.scorer,
                population_size=evo.population_size,
                num_generations=evo.num_generations,
                mutation_rate=evo.mutation_rate,
                crossover_rate=evo.crossover_rate,
                elite_count=evo.elite_count,
                early_stop_score=evo.early_stop_score,
                seed=self.config.experiment.seed,
            )
        return self._searcher

    # -----------------------------------------------------------------
    # Phase 0: Baselines
    # -----------------------------------------------------------------
    def run_baselines(self, qa_items: Optional[list[QAItem]] = None):
        """Run Phase 0 baseline measurements."""
        if qa_items is None:
            qa_items = self._load_data()

        output_path = Path(self.config.experiment.output_dir) / "baselines.json"

        measurer = BaselineMeasurer(
            target_llm=self.target_llm,
            auxiliary_llm=self.auxiliary_llm,
            scorer=self.scorer,
            num_naive_paraphrases=self.config.baselines.num_naive_paraphrases,
            min_native_accuracy=self.config.evolution.min_native_accuracy,
        )

        results, aggregate = measurer.measure_all(qa_items, output_path=output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("BASELINE RESULTS")
        print("=" * 60)
        print(f"Questions:             {aggregate.num_questions}")
        print(f"Native error rate:     {aggregate.native_error_rate:.3f}")
        print(f"Naive avg error rate:  {aggregate.naive_avg_error_rate:.3f}")
        print(f"Naive max error rate:  {aggregate.naive_max_error_rate:.3f}")
        print(f"Already wrong:         {aggregate.num_already_wrong}")
        print(f"Attackable:            {aggregate.num_attackable}")
        print(f"Results saved to:      {output_path}")
        print(f"Budget:                {get_budget().summary()}")
        print("=" * 60)

        return results, aggregate

    # -----------------------------------------------------------------
    # Phase 1: Attack
    # -----------------------------------------------------------------
    def run_attack(
        self,
        qa_items: Optional[list[QAItem]] = None,
        skip_already_wrong: bool = True,
    ) -> list[EvolutionResult]:
        """Run the evolutionary attack on all questions."""
        if qa_items is None:
            qa_items = self._load_data()

        results: list[EvolutionResult] = []
        failed: list[dict] = []

        for qa_item in tqdm(qa_items, desc="Attacking"):
            # Optionally skip questions the model already gets wrong
            if skip_already_wrong:
                try:
                    baseline = self.scorer.score_original(qa_item)
                except BudgetExceededError as e:
                    logger.error(f"Stopping attack: {e}")
                    break
                if not baseline.knows_answer(self.config.evolution.min_native_accuracy):
                    logger.info(
                        f"Skipping {qa_item.id}: model does not reliably answer correctly "
                        f"(accuracy={baseline.accuracy:.2f}, "
                        f"abstained={baseline.num_abstained}/{baseline.num_samples})"
                    )
                    continue

            try:
                evo_result = self.searcher.search(qa_item)
            except BudgetExceededError as e:
                logger.error(f"Stopping attack: {e}")
                break
            except Exception as e:  # noqa: BLE001 - one bad question must not kill the run
                logger.exception(f"Attack failed on {qa_item.id}: {e}")
                failed.append({"id": qa_item.id, "question": qa_item.question, "error": str(e)})
                self._save_attack_results(results, failed)
                continue
            results.append(evo_result)
            self._save_attack_results(results, failed)  # checkpoint after every question

            if evo_result.success:
                logger.info(
                    f"SUCCESS on {qa_item.id}: "
                    f"score={evo_result.best_score:.2f}, "
                    f"paraphrase={evo_result.best_candidate.paraphrase[:80]}"
                )

        # Save results
        self._save_attack_results(results, failed)

        # Print summary
        num_success = sum(1 for r in results if r.success)
        print("\n" + "=" * 60)
        print("ATTACK RESULTS")
        print("=" * 60)
        print(f"Questions attacked:    {len(results)}")
        print(f"Questions failed:      {len(failed)}")
        print(f"Successful flips:      {num_success}")
        print(
            f"Attack success rate:   {num_success / len(results):.3f}"
            if results
            else "N/A"
        )
        print(f"Budget:                {get_budget().summary()}")
        print("=" * 60)

        return results

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _load_data(self) -> list[QAItem]:
        ds_cfg = self.config.dataset
        return load_dataset(
            name=ds_cfg.name,
            split=ds_cfg.split,
            max_samples=ds_cfg.max_samples,
            filter_single_answer=ds_cfg.filter_single_answer,
            seed=self.config.experiment.seed,
        )

    @staticmethod
    def _candidate_to_dict(c) -> dict:
        d = {
            "paraphrase": c.paraphrase,
            "score": c.score,
            "generation": c.generation,
        }
        if c.score_result is not None:
            sr = c.score_result
            d.update(
                num_samples=sr.num_samples,
                num_correct=sr.num_correct,
                num_incorrect=sr.num_incorrect,
                num_abstained=sr.num_abstained,
                abstention_rate=sr.abstention_rate,
                abstention_policy=sr.abstention_policy,
                responses=sr.responses,
                extracted_answers=sr.extracted_answers,
                correct_flags=sr.correct_flags,
            )
        return d

    def _save_attack_results(
        self, results: list[EvolutionResult], failed: Optional[list[dict]] = None
    ) -> None:
        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "attack_results.json"

        data = []
        for r in results:
            entry = {
                "id": r.qa_item.id,
                "question": r.qa_item.question,
                "ground_truth": r.qa_item.answer,
                "success": r.success,
                "best_score": r.best_score,
                "best_paraphrase": (
                    r.best_candidate.paraphrase if r.best_candidate else None
                ),
                "best_candidate": (
                    self._candidate_to_dict(r.best_candidate) if r.best_candidate else None
                ),
                "generations_run": r.generations_run,
                "total_candidates": r.total_candidates_generated,
                "valid_candidates": r.total_candidates_valid,
                "all_valid_candidates": [
                    self._candidate_to_dict(c)
                    for c in sorted(r.all_candidates, key=lambda c: c.score, reverse=True)
                ],
                "rejected_candidates": r.rejected_candidates,
            }
            data.append(entry)

        payload = {
            "target_model": f"{self.config.target_model.provider}/{self.config.target_model.model_name}",
            "results": data,
            "failed": failed or [],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Attack results saved to {path}")
