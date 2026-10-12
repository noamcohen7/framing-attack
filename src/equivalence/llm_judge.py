"""
LLM-as-judge for semantic equivalence checking.

Uses an LLM to judge whether two questions ask the same thing.
This is the second-pass filter after the NLI model.
"""

from __future__ import annotations

import logging
import re

from src.models.base import BaseLLM

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a semantic equivalence judge. Your job is to determine whether two questions ask for exactly the same information — i.e., any correct answer to Question A is also a correct answer to Question B, and vice versa.

**Rules:**
- Ignore differences in wording, style, formality, or sentence structure.
- The questions must request the SAME factual information.
- A question that narrows or broadens the scope is NOT equivalent.
- A question that adds assumptions, presuppositions, or context not present in the original is NOT equivalent IF that added context changes what a correct answer would be.
- A question that merely provides additional true context without changing what constitutes a correct answer IS still equivalent.

**Question A:** {question_a}
**Question B:** {question_b}

Are these two questions semantically equivalent? Answer with EXACTLY one word: YES or NO. Then provide a one-sentence explanation.

**Answer:**"""


class LLMJudge:
    """
    Uses an LLM to judge semantic equivalence between two questions.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def check_equivalence(self, original: str, paraphrase: str) -> dict:
        """
        Ask the LLM whether original and paraphrase are semantically equivalent.

        Returns:
            Dict with:
              - is_equivalent: bool
              - explanation: str (LLM's reasoning)
              - raw_response: str
        """
        prompt = JUDGE_PROMPT.format(question_a=original, question_b=paraphrase)
        response = self.llm.generate(prompt, temperature=0.0)
        raw = response.text.strip()

        # Parse the YES/NO answer
        is_equivalent = self._parse_verdict(raw)

        # Extract explanation (everything after YES/NO)
        explanation = self._extract_explanation(raw)

        return {
            "is_equivalent": is_equivalent,
            "explanation": explanation,
            "raw_response": raw,
        }

    def check_equivalence_batch(
        self, original: str, paraphrases: list[str]
    ) -> list[dict]:
        """Check equivalence for multiple paraphrases."""
        return [self.check_equivalence(original, p) for p in paraphrases]

    @staticmethod
    def _parse_verdict(text: str) -> bool:
        """Extract YES/NO verdict from the LLM response."""
        # Look for YES or NO at the start of the response
        text_upper = text.strip().upper()
        if text_upper.startswith("YES"):
            return True
        if text_upper.startswith("NO"):
            return False

        # Fallback: search anywhere in the text
        yes_match = re.search(r"\bYES\b", text_upper)
        no_match = re.search(r"\bNO\b", text_upper)

        if yes_match and not no_match:
            return True
        if no_match and not yes_match:
            return False

        # Ambiguous -- default to rejecting (conservative)
        logger.warning(f"Ambiguous LLM judge verdict, defaulting to NO: {text[:100]}")
        return False

    @staticmethod
    def _extract_explanation(text: str) -> str:
        """Extract the explanation part after YES/NO."""
        # Remove leading YES/NO
        cleaned = re.sub(r"^(YES|NO)[.\s]*", "", text.strip(), flags=re.IGNORECASE)
        return cleaned.strip()
