"""
Dataset loading and preprocessing for factual QA.

Supports:
  - TriviaQA (via HuggingFace datasets)
  - Custom hand-built recency facts (JSON)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
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
    # Event postdates the target model's training data (~mid-2024 for
    # gpt-oss-20b): expected unanswerable, kept for the abstention analysis.
    after_target_cutoff: bool = False

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

    try:
        import truststore  # corporate TLS proxies: trust the system cert store
        truststore.inject_into_ssl()
    except ImportError:
        pass

    # rc.nocontext: same questions/answers as 'rc' without the evidence
    # documents (validation is ~7 MB instead of several GB). Load one split's
    # parquet file so the other splits are not downloaded.
    logger.info(f"Loading TriviaQA split='{split}'...")
    ds = hf_load(
        "mandarjoshi/trivia_qa",
        data_files={split: f"rc.nocontext/{split}-00000-of-00001.parquet"},
        split=split,
    )

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

        if filter_single_answer:
            if not is_single_canonical_answer(canonical):
                continue
            aliases = surface_variant_aliases(canonical, aliases)

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


_STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "de", "la", "le", "von", "van"}
_MULTI_ANSWER = re.compile(r"\s(and|or)\s|&|/|;|,", re.IGNORECASE)


def _content_tokens(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    words = re.findall(r"[^\W_]+", folded)
    return {w for w in words if w not in _STOPWORDS}


def is_single_canonical_answer(canonical: str) -> bool:
    """
    True if a TriviaQA answer is one entity we can match reliably.

    Rejects:
      - numeric / date answers ("1989", "12 June 1815", "8%"): surface forms
        vary too much for fuzzy matching ("nineteen eighty-four").
      - list or choice answers ("Lennon and McCartney", "Oxford/Cambridge").
    """
    if not canonical.strip() or re.search(r"\d", canonical):
        return False
    if _MULTI_ANSWER.search(canonical):
        return False
    return bool(_content_tokens(canonical))


def surface_variant_aliases(canonical: str, aliases: list[str]) -> list[str]:
    """
    Keep only aliases that are surface variants of the canonical answer: the
    alias's content words contain all of the canonical's ("US President Bill
    Clinton") or are a subset of them ("Beatles", "García Márquez").

    TriviaQA aliases are Wikipedia redirects and include unrelated pages
    ("Buddy (Clinton's dog)" for Bill Clinton, film titles for an actor).
    Those would let a wrong answer match as correct, so drop them.
    """
    canon_tokens = _content_tokens(canonical)
    kept: list[str] = []
    seen = {canonical.strip().lower()}
    for alias in aliases:
        key = alias.strip().lower()
        alias_tokens = _content_tokens(alias)
        is_variant = alias_tokens and (canon_tokens <= alias_tokens or alias_tokens <= canon_tokens)
        if key and key not in seen and is_variant:
            kept.append(alias)
            seen.add(key)
    return kept


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
            after_target_cutoff=item.get("after_target_cutoff", False),
        )
        for item in raw
    ]
    # Post-cutoff items last, so --max-samples picks attackable questions first.
    items.sort(key=lambda it: it.after_target_cutoff)

    logger.info(f"Custom facts: loaded {len(items)} items")
    return items
