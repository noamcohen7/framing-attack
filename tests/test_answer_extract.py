"""Tests for answer extraction and normalization."""

import pytest
from src.evaluation.answer_extract import AnswerExtractor


class TestNormalize:
    def test_lowercase(self):
        assert AnswerExtractor.normalize("Donald Trump") == "donald trump"

    def test_strip_articles(self):
        assert AnswerExtractor.normalize("The Nile") == "nile"
        assert AnswerExtractor.normalize("A cat") == "cat"
        assert AnswerExtractor.normalize("An apple") == "apple"

    def test_remove_punctuation(self):
        assert AnswerExtractor.normalize("Hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert AnswerExtractor.normalize("  hello   world  ") == "hello world"

    def test_empty(self):
        assert AnswerExtractor.normalize("") == ""


class TestAnswersMatch:
    def test_exact_match(self):
        assert AnswerExtractor.answers_match(
            "Donald Trump", "Donald Trump", method="exact"
        )

    def test_exact_no_match(self):
        assert not AnswerExtractor.answers_match(
            "Joe Biden", "Donald Trump", method="exact"
        )

    def test_exact_with_aliases(self):
        assert AnswerExtractor.answers_match(
            "Trump", "Donald Trump", aliases=["Trump"], method="exact"
        )

    def test_fuzzy_contains(self):
        assert AnswerExtractor.answers_match(
            "Donald Trump won the election",
            "Donald Trump",
            method="fuzzy",
        )

    def test_fuzzy_contained_in(self):
        assert AnswerExtractor.answers_match(
            "Trump",
            "Donald Trump",
            method="fuzzy",
        )

    def test_fuzzy_no_match(self):
        assert not AnswerExtractor.answers_match(
            "Joe Biden", "Donald Trump", method="fuzzy"
        )

    def test_contains_method(self):
        assert AnswerExtractor.answers_match(
            "The answer is gold",
            "Gold",
            method="contains",
        )

    def test_regex_extraction_short(self):
        extractor = AnswerExtractor(method="regex", normalize=False)
        assert extractor.extract("q", "Donald Trump") == "Donald Trump"

    def test_regex_extraction_long(self):
        extractor = AnswerExtractor(method="regex", normalize=False)
        result = extractor.extract(
            "q",
            "The winner was Donald Trump. He won in November 2024."
        )
        assert "Donald Trump" in result
