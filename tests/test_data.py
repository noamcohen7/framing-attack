"""Tests for data loading."""

import pytest
from src.data.loader import QAItem, _load_custom_facts


class TestQAItem:
    def test_all_valid_answers(self):
        item = QAItem(
            id="test_1",
            question="Who won?",
            answer="Trump",
            aliases=["Donald Trump", "Donald J. Trump"],
        )
        assert item.all_valid_answers == [
            "Trump", "Donald Trump", "Donald J. Trump"
        ]

    def test_no_aliases(self):
        item = QAItem(id="test_2", question="Q?", answer="A")
        assert item.all_valid_answers == ["A"]


class TestCustomFacts:
    def test_load_custom_facts(self):
        items = _load_custom_facts()
        assert len(items) > 0

        # Check structure
        for item in items:
            assert item.id
            assert item.question
            assert item.answer
            assert item.source == "custom"

    def test_custom_facts_content(self):
        items = _load_custom_facts()
        ids = {item.id for item in items}
        assert "recency_001" in ids

        # Find the presidential election question
        election = next(i for i in items if i.id == "recency_001")
        assert "presidential election" in election.question.lower()
        assert election.answer == "Donald Trump"
