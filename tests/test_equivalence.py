"""Tests for semantic equivalence validation.

Note: NLI model tests require downloading the model (~400MB).
These are marked with @pytest.mark.slow and skipped in CI by default.
"""

import pytest


class TestLLMJudge:
    """Test the LLM judge parsing logic (no API calls needed)."""

    def test_parse_verdict_yes(self):
        from src.equivalence.llm_judge import LLMJudge

        assert LLMJudge._parse_verdict("YES. They ask the same thing.") is True
        assert LLMJudge._parse_verdict("Yes") is True

    def test_parse_verdict_no(self):
        from src.equivalence.llm_judge import LLMJudge

        assert LLMJudge._parse_verdict("NO. The second changes the scope.") is False
        assert LLMJudge._parse_verdict("No") is False

    def test_parse_verdict_ambiguous(self):
        from src.equivalence.llm_judge import LLMJudge

        # Ambiguous defaults to False (conservative)
        assert LLMJudge._parse_verdict("I'm not sure") is False

    def test_extract_explanation(self):
        from src.equivalence.llm_judge import LLMJudge

        exp = LLMJudge._extract_explanation(
            "YES. Both questions ask who won the 2024 election."
        )
        assert "both" in exp.lower() or "questions" in exp.lower()

    def test_extract_explanation_no(self):
        from src.equivalence.llm_judge import LLMJudge

        exp = LLMJudge._extract_explanation(
            "NO. The second question narrows the scope to a specific state."
        )
        assert "narrows" in exp.lower() or "scope" in exp.lower()
