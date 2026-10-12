"""Tests for configuration loading."""

import pytest
from src.config import load_config, Config, _deep_merge


class TestConfig:
    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, Config)
        assert config.target_model.provider == "gemini"
        assert config.evolution.population_size == 10

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}

    def test_config_defaults(self):
        config = Config()
        assert config.experiment.seed == 42
        assert config.equivalence.nli_threshold == 0.85

    def test_requests_per_minute_loaded(self):
        config = load_config()
        assert config.target_model.requests_per_minute == 15
        assert config.auxiliary_model.requests_per_minute == 15
        assert config.equivalence.llm_judge.requests_per_minute == 15

    def test_requests_per_minute_can_be_disabled(self, tmp_path):
        override = tmp_path / "o.yaml"
        override.write_text("target_model:\n  requests_per_minute: null\n")
        config = load_config(override)
        assert config.target_model.requests_per_minute is None
