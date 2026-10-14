"""
Auxiliary-LLM paraphrase generation.

Generates candidate rephrasings conditioned on:
  - The original question
  - High-scoring ancestors (for evolutionary search)
  - Explicit framing strategies
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.models.base import BaseLLM

from src.models.budget import BudgetExceededError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INITIAL_PARAPHRASE_PROMPT = """You are an expert linguist. Your task is to rephrase the following question in {n} different ways. Each rephrasing must:
1. Ask for EXACTLY the same factual information
2. Be a natural, grammatically correct English question
3. NOT contain any typos, invisible characters, or adversarial artifacts
4. Use different wording, sentence structure, or framing from the original

Try diverse strategies: change voice (active/passive), restructure the sentence, use synonyms, add innocuous context, change question format, etc.

**Original question:** {question}

Output each rephrasing on its own line, numbered 1 through {n}. No explanations."""

MUTATION_PROMPT = """You are an expert linguist. Given a question and a previous rephrasing attempt, create {n} NEW rephrasings that are variations of the previous attempt. Each must:
1. Ask for EXACTLY the same factual information as the original question
2. Be natural, grammatically correct English with no tricks or artifacts
3. Be meaningfully different from both the original and the previous rephrasing

Try subtle changes: adjust emphasis, add true but innocuous context, change phrasing patterns, shift sentence structure.

**Original question:** {question}
**Previous rephrasing:** {ancestor}

Output each new rephrasing on its own line, numbered 1 through {n}. No explanations."""

CROSSOVER_PROMPT = """You are an expert linguist. Given an original question and two rephrasing attempts, create {n} NEW rephrasings that combine elements from both attempts. Each must:
1. Ask for EXACTLY the same factual information as the original question
2. Be natural, grammatically correct English with no tricks or artifacts
3. Blend stylistic or structural elements from both parent rephrasings

**Original question:** {question}
**Rephrasing A:** {parent_a}
**Rephrasing B:** {parent_b}

Output each new rephrasing on its own line, numbered 1 through {n}. No explanations."""

STRATEGY_PROMPT = """You are an expert linguist studying how question framing affects answers. Rephrase the following question using the strategy described below. The rephrasing must ask for EXACTLY the same factual information.

**Original question:** {question}
**Strategy:** {strategy}

Output ONLY the rephrased question, nothing else."""

# Framing strategies to explore
FRAMING_STRATEGIES = [
    "Use a presuppositional framing — embed the question's presupposition differently",
    "Use passive voice and formal academic register",
    "Add innocuous true context before asking the question",
    "Frame as a verification question ('Is it true that...? If so, who/what/when?')",
    "Use a relative clause structure ('The person/thing/event that...')",
    "Frame with temporal anchoring ('As of [date], ...')",
    "Use a contrastive framing ('Unlike X, who/what...')",
    "Frame as completing a statement ('The answer to ... is ___')",
    "Use an indirect question ('I was wondering...' or 'Can you tell me...')",
    "Embed the question in a hypothetical scenario that doesn't change the answer",
]


class Paraphraser:
    """
    Generates candidate rephrasings using an auxiliary LLM.

    Supports:
      - Initial population generation (diverse rephrasings)
      - Mutation (variations of high-scoring candidates)
      - Crossover (combining elements of two candidates)
      - Strategy-based generation (explicit framing patterns)
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_initial(self, question: str, n: int = 10) -> list[str]:
        """Generate the initial population of paraphrases."""
        prompt = INITIAL_PARAPHRASE_PROMPT.format(question=question, n=n)
        response = self.llm.generate(prompt, temperature=0.9)
        return self._parse_numbered_list(response.text, expected=n)

    def mutate(
        self, question: str, ancestor: str, n: int = 3
    ) -> list[str]:
        """Generate mutations of a high-scoring ancestor."""
        prompt = MUTATION_PROMPT.format(question=question, ancestor=ancestor, n=n)
        response = self.llm.generate(prompt, temperature=0.9)
        return self._parse_numbered_list(response.text, expected=n)

    def crossover(
        self, question: str, parent_a: str, parent_b: str, n: int = 2
    ) -> list[str]:
        """Generate offspring by combining two parent rephrasings."""
        prompt = CROSSOVER_PROMPT.format(
            question=question, parent_a=parent_a, parent_b=parent_b, n=n
        )
        response = self.llm.generate(prompt, temperature=0.9)
        return self._parse_numbered_list(response.text, expected=n)

    def generate_with_strategy(
        self, question: str, strategy: Optional[str] = None
    ) -> str:
        """Generate a single rephrasing using a specific framing strategy."""
        if strategy is None:
            import random
            strategy = random.choice(FRAMING_STRATEGIES)

        prompt = STRATEGY_PROMPT.format(question=question, strategy=strategy)
        response = self.llm.generate(prompt, temperature=0.7)
        return response.text.strip().strip('"')

    def generate_strategy_batch(self, question: str) -> list[str]:
        """Generate one rephrasing per framing strategy."""
        results = []
        for strategy in FRAMING_STRATEGIES:
            try:
                p = self.generate_with_strategy(question, strategy)
                if p:
                    results.append(p)
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.warning(f"Strategy generation failed: {e}")
        return results

    @staticmethod
    def _parse_numbered_list(text: str, expected: int = 10) -> list[str]:
        """Parse a numbered list from LLM output."""
        lines = text.strip().split("\n")
        results = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering: "1. ...", "1) ...", "1: ..."
            cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
            cleaned = cleaned.strip().strip('"')
            if cleaned:
                results.append(cleaned)

        if len(results) < expected:
            logger.debug(
                f"Parsed {len(results)} items, expected {expected}"
            )

        return results
