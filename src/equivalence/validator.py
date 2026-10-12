"""
Combined semantic equivalence validator.

Pipeline:
  1. NLI model (fast, local, free) — first-pass filter
  2. LLM judge (slower, API-based) — second-pass filter
  3. Both must agree (configurable)

This keeps the attack "honest" by rejecting any rephrasing that
changes the question's meaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.equivalence.llm_judge import LLMJudge
from src.models.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class EquivalenceResult:
    """Result of a semantic equivalence check."""

    is_equivalent: bool
    nli_result: Optional[dict] = None
    llm_result: Optional[dict] = None
    reason: str = ""


class SemanticEquivalenceValidator:
    """
    Multi-signal validator that combines NLI and LLM judge.

    The NLI model is fast and free (runs locally), so we use it as a
    first-pass filter to avoid unnecessary LLM API calls.
    """

    def __init__(
        self,
        nli_model: Optional[str] = "cross-encoder/nli-deberta-v3-base",
        nli_threshold: float = 0.85,
        llm: Optional[BaseLLM] = None,
        require_both: bool = True,
        nli: Optional[object] = None,
    ):
        """
        Args:
            nli_model: HuggingFace model ID for the NLI cross-encoder, or None to
                       disable the local NLI stage (judge-only mode; no torch needed).
            nli_threshold: Minimum bidirectional entailment score.
            llm: LLM client for the judge (if None, NLI-only mode).
            require_both: If True, both NLI and LLM must agree.
                          If False, NLI alone is sufficient.
            nli: Pre-built NLI validator (for tests); overrides nli_model.
        """
        if nli is not None:
            self.nli = nli
        elif nli_model:
            from src.equivalence.nli import NLIValidator  # lazy: needs torch/transformers

            self.nli = NLIValidator(model_name=nli_model, threshold=nli_threshold)
        else:
            self.nli = None
        self.llm_judge = LLMJudge(llm) if llm is not None else None
        self.require_both = require_both

        if self.nli is None and self.llm_judge is None:
            raise ValueError("SemanticEquivalenceValidator needs an NLI model, an LLM judge, or both")
        if self.nli is None:
            logger.info("NLI disabled: running in LLM-judge-only mode")
            self.require_both = False

        if self.llm_judge is None and require_both:
            logger.warning(
                "require_both=True but no LLM judge provided. "
                "Falling back to NLI-only mode."
            )
            self.require_both = False

    def validate(self, original: str, paraphrase: str) -> EquivalenceResult:
        """
        Check whether a paraphrase is semantically equivalent to the original.

        Returns:
            EquivalenceResult with detailed scores and verdict.
        """
        # Judge-only mode (no local NLI model)
        if self.nli is None:
            llm_result = self.llm_judge.check_equivalence(original, paraphrase)
            return EquivalenceResult(
                is_equivalent=llm_result["is_equivalent"],
                llm_result=llm_result,
                reason=(
                    "LLM judge approved" if llm_result["is_equivalent"]
                    else f"LLM judge rejected: {llm_result.get('explanation', '')}"
                ),
            )

        # Step 1: NLI check (fast, free)
        nli_result = self.nli.check_equivalence(original, paraphrase)

        # If NLI rejects and we require both, no need for LLM call
        if not nli_result["is_equivalent"] and self.require_both:
            return EquivalenceResult(
                is_equivalent=False,
                nli_result=nli_result,
                reason=f"NLI rejected (min_score={nli_result['min_score']:.3f})",
            )

        # Step 2: LLM judge (if available)
        llm_result = None
        if self.llm_judge is not None:
            llm_result = self.llm_judge.check_equivalence(original, paraphrase)

        # Step 3: Combine signals
        if self.require_both:
            # Both must agree
            is_eq = nli_result["is_equivalent"] and (
                llm_result["is_equivalent"] if llm_result else True
            )
            if not is_eq:
                parts = []
                if not nli_result["is_equivalent"]:
                    parts.append(f"NLI rejected (min_score={nli_result['min_score']:.3f})")
                if llm_result and not llm_result["is_equivalent"]:
                    parts.append(f"LLM judge rejected: {llm_result.get('explanation', '')}")
                reason = "; ".join(parts)
            else:
                reason = "Both NLI and LLM judge approved"
        else:
            # NLI only
            is_eq = nli_result["is_equivalent"]
            reason = (
                "NLI approved" if is_eq
                else f"NLI rejected (min_score={nli_result['min_score']:.3f})"
            )

        return EquivalenceResult(
            is_equivalent=is_eq,
            nli_result=nli_result,
            llm_result=llm_result,
            reason=reason,
        )

    def validate_batch(
        self, original: str, paraphrases: list[str]
    ) -> list[EquivalenceResult]:
        """Validate multiple paraphrases against the same original."""
        return [self.validate(original, p) for p in paraphrases]

    def filter_valid(
        self, original: str, paraphrases: list[str]
    ) -> list[tuple[str, EquivalenceResult]]:
        """
        Filter paraphrases, returning only those that are semantically equivalent.

        Returns:
            List of (paraphrase, result) tuples for valid paraphrases.
        """
        results = self.validate_batch(original, paraphrases)
        return [
            (p, r) for p, r in zip(paraphrases, results) if r.is_equivalent
        ]
