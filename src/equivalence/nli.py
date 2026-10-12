"""
Natural Language Inference (NLI) model for semantic equivalence checking.

Uses a cross-encoder model (DeBERTa-based) locally. No API costs.
Checks bidirectional entailment: q entails q' AND q' entails q.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NLIValidator:
    """
    Checks semantic equivalence via bidirectional NLI.

    Two questions are semantically equivalent if each entails the other
    with probability above the threshold.
    """

    # Label mapping for cross-encoder/nli-deberta-v3-base:
    # 0 = contradiction, 1 = entailment, 2 = neutral
    ENTAILMENT_IDX = 1

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.85,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.threshold = threshold

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading NLI model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("NLI model loaded.")

    def _entailment_score(self, premise: str, hypothesis: str) -> float:
        """Compute P(entailment | premise, hypothesis)."""
        import torch

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entailment_prob = probs[0, self.ENTAILMENT_IDX].item()

        return entailment_prob

    def check_equivalence(self, original: str, paraphrase: str) -> dict:
        """
        Check bidirectional entailment between original and paraphrase.

        Returns:
            Dict with:
              - is_equivalent: bool
              - forward_score: P(entailment | original -> paraphrase)
              - backward_score: P(entailment | paraphrase -> original)
              - min_score: min of forward and backward
        """
        forward = self._entailment_score(original, paraphrase)
        backward = self._entailment_score(paraphrase, original)
        min_score = min(forward, backward)

        return {
            "is_equivalent": min_score >= self.threshold,
            "forward_score": round(forward, 4),
            "backward_score": round(backward, 4),
            "min_score": round(min_score, 4),
        }

    def check_equivalence_batch(
        self, original: str, paraphrases: list[str]
    ) -> list[dict]:
        """Check equivalence for multiple paraphrases against the same original."""
        return [self.check_equivalence(original, p) for p in paraphrases]
