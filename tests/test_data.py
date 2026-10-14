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

    def test_post_cutoff_items_sorted_last(self):
        items = _load_custom_facts()
        flags = [item.after_target_cutoff for item in items]
        assert any(flags)
        assert flags == sorted(flags)

    def test_custom_facts_unique(self):
        items = _load_custom_facts()
        assert len({i.id for i in items}) == len(items)
        assert len({i.question.lower() for i in items}) == len(items)

    def test_custom_facts_content(self):
        items = _load_custom_facts()
        ids = {item.id for item in items}
        assert "recency_001" in ids

        # Find the presidential election question
        election = next(i for i in items if i.id == "recency_001")
        assert "presidential election" in election.question.lower()
        assert election.answer == "Donald Trump"


class TestSingleAnswerFilter:
    """Hand-written rows in TriviaQA's (value, aliases) shape."""

    @pytest.mark.parametrize("canonical", [
        "The Beatles", "Gabriel García Márquez", "Gold", "Anne-Marie Duff",
    ])
    def test_keeps_single_entities(self, canonical):
        from src.data.loader import is_single_canonical_answer
        assert is_single_canonical_answer(canonical)

    @pytest.mark.parametrize("canonical", [
        "1989", "18 June 1815", "8%", "B1",
        "Lennon and McCartney", "Oxford/Cambridge", "Tea or coffee", "Salt, pepper",
        "", "   ",
    ])
    def test_rejects_numeric_and_multi(self, canonical):
        from src.data.loader import is_single_canonical_answer
        assert not is_single_canonical_answer(canonical)

    def test_drops_unrelated_aliases(self):
        from src.data.loader import surface_variant_aliases
        aliases = [
            "Bill Clinton", "US President Bill Clinton", "Buddy (Clinton's dog)",
            "William Jefferson Blythe III", "bill clinton", "BILL CLINTON",
        ]
        assert surface_variant_aliases("BILL CLINTON", aliases) == ["US President Bill Clinton"]

    def test_keeps_subsets_and_accent_variants(self):
        from src.data.loader import surface_variant_aliases
        aliases = ["Gabriel Garcia Marquez", "García Márquez", "Gabo", "Marquez family"]
        assert surface_variant_aliases("Gabriel García Márquez", aliases) == [
            "Gabriel Garcia Marquez", "García Márquez",
        ]

    def test_drops_redirects_with_no_shared_word(self):
        from src.data.loader import surface_variant_aliases
        aliases = ["Gerard Butler", "The Bricklayer (2014 film)", "Dynamo (2015 film)"]
        assert surface_variant_aliases("GERARD BUTLER", aliases) == []
