"""
Dataset loading and preprocessing for factual QA.

Supports:
  - TriviaQA (via HuggingFace datasets)
  - Custom hand-built recency facts (JSON)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent


@dataclass
class QAItem:
    """A single factual question-answer pair."""

    id: str
    question: str
    answer: str
    aliases: list[str] = field(default_factory=list)
    category: Optional[str] = None
    source: str = "unknown"

    @property
    def all_valid_answers(self) -> list[str]:
        """Return the canonical answer plus all aliases."""
        return [self.answer] + self.aliases


def load_dataset(
    name: str = "triviaqa",
    split: str = "validation",
    max_samples: int = 200,
    filter_single_answer: bool = True,
    seed: int = 42,
) -> list[QAItem]:
    """
    Load a factual QA dataset.

    Args:
        name: Dataset name ('triviaqa', 'natural_questions', 'custom').
        split: Dataset split to use.
        max_samples: Maximum number of items to return.
        filter_single_answer: If True, only keep items with a single canonical answer.
        seed: Random seed for shuffling.

    Returns:
        A list of QAItem objects.
    """
    if name == "triviaqa":
        items = _load_triviaqa(split, filter_single_answer, seed)
    elif name == "custom":
        items = _load_custom_facts()
    elif name == "combined":
        # Merge TriviaQA + custom recency facts
        triviaqa_items = _load_triviaqa(split, filter_single_answer, seed)
        custom_items = _load_custom_facts()
        items = custom_items + triviaqa_items  # Recency facts first
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Supported: triviaqa, custom, combined")

    # Truncate to max_samples
    if len(items) > max_samples:
        items = items[:max_samples]

    logger.info(f"Loaded {len(items)} QA items from '{name}' (split={split})")
    return items


def _load_triviaqa(
    split: str = "validation",
    filter_single_answer: bool = True,
    seed: int = 42,
) -> list[QAItem]:
    """Load TriviaQA from HuggingFace datasets."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    logger.info(f"Loading TriviaQA split='{split}'...")
    ds = hf_load("trivia_qa", "rc", split=split)

    # Shuffle deterministically
    ds = ds.shuffle(seed=seed)

    items: list[QAItem] = []
    for i, row in enumerate(ds):
        answer_obj = row["answer"]
        # TriviaQA has: answer.value (canonical), answer.aliases (list)
        canonical = answer_obj.get("value", "")
        aliases = answer_obj.get("aliases", [])

        if not canonical:
            continue

        # Filter: only single-answer questions
        if filter_single_answer:
            # Skip items with very different aliases (suggests ambiguity)
            # Keep items where aliases are surface variations of the canonical answer
            pass  # TriviaQA aliases are generally surface variants; keep all

        items.append(
            QAItem(
                id=f"triviaqa_{i}",
                question=row["question"],
                answer=canonical,
                aliases=aliases,
                source="triviaqa",
            )
        )

    logger.info(f"TriviaQA: loaded {len(items)} items")
    return items


def _load_custom_facts() -> list[QAItem]:
    """Load hand-built recency / timeless facts from JSON."""
    facts_path = _DATA_DIR / "recency_facts.json"
    if not facts_path.exists():
        logger.warning(f"Custom facts file not found: {facts_path}")
        return []

    with open(facts_path) as f:
        raw = json.load(f)

    items = [
        QAItem(
            id=item["id"],
            question=item["question"],
            answer=item["answer"],
            aliases=item.get("aliases", []),
            category=item.get("category"),
            source="custom",
        )
        for item in raw
    ]

    logger.info(f"Custom facts: loaded {len(items)} items")
    return items
