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


class TestGrade:
    @pytest.mark.parametrize("text,expected", [
        ("VERDICT: CORRECT\nANSWER: Loreen, Sweden", (True, "Loreen, Sweden")),
        ("verdict: incorrect\nANSWER: Lionel Messi", (False, "Lionel Messi")),
        ("VERDICT: NO_ANSWER\nANSWER: NO_ANSWER", (None, "NO_ANSWER")),
        ("I think it's fine", (None, "NO_ANSWER")),
    ])
    def test_parse_grade(self, text, expected):
        from src.evaluation.answer_extract import parse_grade
        assert parse_grade(text) == expected

    def test_scorer_uses_grade_against_original_question(self):
        from typing import Optional
        from src.data.loader import QAItem
        from src.evaluation.scorer import WrongnessScorer
        from src.models.base import BaseLLM, LLMResponse

        class Scripted(BaseLLM):
            def __init__(self, texts):
                super().__init__(model_name="s", requests_per_minute=None)
                self.texts, self.prompts = list(texts), []

            @property
            def provider_name(self):
                return "fake"

            def _generate(self, prompt: str, temperature: Optional[float] = None):
                self.prompts.append(prompt)
                return LLMResponse(text=self.texts.pop(0), model="s", provider="fake")

        target = Scripted(["Loreen won for Sweden.", "Ukraine won.", "No idea."])
        grader = Scripted(["VERDICT: CORRECT\nANSWER: Loreen (Sweden)",
                           "VERDICT: INCORRECT\nANSWER: Ukraine",
                           "VERDICT: NO_ANSWER\nANSWER: NO_ANSWER"])
        qa = QAItem(id="q", question="Which country won Eurovision 2023?", answer="Sweden")
        scorer = WrongnessScorer(target, AnswerExtractor(method="grade", llm=grader), num_samples=3)
        r = scorer.score(qa, "Who took first place at Eurovision 2023?")
        assert r.correct_flags == [True, False, False]
        assert r.abstained_flags == [False, False, True]
        assert r.wrongness_score == 0.5          # 1 wrong of 2 answered
        assert "Which country won Eurovision 2023?" in grader.prompts[0]
        assert "Who took first place" not in grader.prompts[0]
