"""
Answer extraction and normalization.

Extracts the factual answer from an LLM's free-form response
and normalizes it for comparison against ground truth.
"""

from __future__ import annotations

import re
import string
import logging
from typing import Optional

from src.models.base import BaseLLM

logger = logging.getLogger(__name__)

NO_ANSWER = "NO_ANSWER"

EXTRACTION_PROMPT = """Extract the factual answer from the following response to the question. Return ONLY the answer itself, nothing else. If the response doesn't contain a clear answer, respond with "NO_ANSWER".

**Question:** {question}
**Response:** {response}

**Extracted answer:**"""


class AnswerExtractor:
    """
    Extracts and normalizes answers from LLM responses.

    Supports multiple extraction methods:
      - 'llm': Use an LLM to extract the answer (most robust)
      - 'regex': Simple regex-based extraction (fastest)
      - 'hybrid': Try regex first, fall back to LLM
    """

    def __init__(
        self,
        method: str = "llm",
        llm: Optional[BaseLLM] = None,
        normalize: bool = True,
    ):
        self.method = method
        self.llm = llm
        self.normalize_answers = normalize

        if method in ("llm", "hybrid") and llm is None:
            raise ValueError(f"LLM client required for method='{method}'")

    def extract(self, question: str, response: str) -> str:
        """
        Extract the answer from an LLM response.

        Args:
            question: The original question.
            response: The LLM's free-form response.

        Returns:
            The extracted answer string, normalized if configured.
        """
        if self.method == "regex":
            answer = self._extract_regex(response)
        elif self.method == "llm":
            answer = self._extract_llm(question, response)
        elif self.method == "hybrid":
            answer = self._extract_regex(response)
            if not answer or answer == "NO_ANSWER":
                answer = self._extract_llm(question, response)
        else:
            raise ValueError(f"Unknown extraction method: {self.method!r}")

        if self.is_abstention(answer):
            return NO_ANSWER

        if self.normalize_answers:
            answer = self.normalize(answer)

        return answer

    @staticmethod
    def is_abstention(answer: str) -> bool:
        """True if the extracted answer is the NO_ANSWER sentinel (any casing/spacing)."""
        if not answer:
            return True
        compact = re.sub(r"[^a-z]", "", answer.lower())
        return compact in ("noanswer", "none", "unknown", "idontknow", "cannotanswer")

    def _extract_llm(self, question: str, response: str) -> str:
        """Use an LLM to extract the answer."""
        prompt = EXTRACTION_PROMPT.format(question=question, response=response)
        result = self.llm.generate(prompt, temperature=0.0)
        return result.text.strip()

    @staticmethod
    def _extract_regex(response: str) -> str:
        """
        Simple extraction: take the first sentence or short phrase.

        This works for short, direct answers but fails on verbose responses.
        """
        # If response is already short, use it directly
        response = response.strip()
        if len(response.split()) <= 10:
            return response

        # Try to extract the first sentence
        match = re.match(r"^([^.!?\n]+[.!?]?)", response)
        if match:
            return match.group(1).strip()

        return response

    @staticmethod
    def normalize(answer: str) -> str:
        """
        Normalize an answer for comparison.

        - Lowercase
        - Strip leading articles (the, a, an)
        - Remove punctuation
        - Collapse whitespace
        """
        if not answer:
            return ""

        text = answer.lower().strip()

        # Remove leading articles
        text = re.sub(r"^(the|a|an)\s+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def answers_match(
        predicted: str,
        ground_truth: str,
        aliases: list[str] | None = None,
        method: str = "fuzzy",
    ) -> bool:
        """
        Check if a predicted answer matches the ground truth.

        Args:
            predicted: The extracted/normalized predicted answer.
            ground_truth: The canonical correct answer.
            aliases: Alternative acceptable answers.
            method: 'exact', 'fuzzy', or 'contains'.

        Returns:
            True if the predicted answer matches.
        """
        norm = AnswerExtractor.normalize

        pred_norm = norm(predicted)
        truth_norm = norm(ground_truth)

        all_valid = [truth_norm]
        if aliases:
            all_valid.extend(norm(a) for a in aliases)

        if method == "exact":
            return pred_norm in all_valid

        elif method == "fuzzy":
            # Check if predicted contains or is contained in any valid answer
            for valid in all_valid:
                if pred_norm == valid:
                    return True
                if valid in pred_norm or pred_norm in valid:
                    return True
            return False

        elif method == "contains":
            for valid in all_valid:
                if valid in pred_norm:
                    return True
            return False

        else:
            raise ValueError(f"Unknown match method: {method!r}")
