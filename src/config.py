"""
Configuration management for framing-attack.

Loads settings from:
  1. config/default.yaml (defaults)
  2. Environment variables via .env
  3. Optional override YAML passed at runtime
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load .env as early as possible
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------
class ModelConfig(BaseModel):
    provider: str = "gemini"
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 256
    num_samples: int = 5
    requests_per_minute: Optional[float] = 15  # Free-tier Gemini limit; None disables pacing
    concurrency: int = 1  # Parallel requests for sample()/generate_batch()
    reasoning_effort: Optional[str] = None  # Groq reasoning models (gpt-oss): low/medium/high


class LLMJudgeConfig(BaseModel):
    provider: str = "gemini"
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    requests_per_minute: Optional[float] = 15
    concurrency: int = 1


class EquivalenceConfig(BaseModel):
    nli_model: Optional[str] = "cross-encoder/nli-deberta-v3-base"  # null = judge-only, no torch
    nli_threshold: float = 0.85
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)
    require_both: bool = True


class DatasetConfig(BaseModel):
    name: str = "triviaqa"
    split: str = "validation"
    max_samples: int = 200
    filter_single_answer: bool = True


class EvolutionConfig(BaseModel):
    population_size: int = 10
    num_generations: int = 20
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_count: int = 2
    early_stop_score: float = 1.0
    # Attack a question only if the model answers it correctly in at least this
    # fraction of native samples (abstentions count against). 0.8 = 4 of 5.
    min_native_accuracy: float = 0.8


class AnswerExtractionConfig(BaseModel):
    method: str = "llm"
    normalize: bool = True
    equivalence_check: str = "fuzzy"
    abstention_policy: str = "exclude"  # exclude | incorrect | correct


class BaselinesConfig(BaseModel):
    num_naive_paraphrases: int = 5
    num_samples_per_paraphrase: int = 5


class BudgetConfig(BaseModel):
    """Hard per-run ceiling. Either cap None = unlimited."""

    max_requests: Optional[int] = 3000
    max_usd: Optional[float] = 2.0
    # USD per million tokens, used to estimate cost when the API does not report it.
    prices: dict[str, dict[str, float]] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    seed: int = 42
    output_dir: str = "results"
    log_level: str = "INFO"


class Config(BaseModel):
    """Top-level configuration."""

    target_model: ModelConfig = Field(default_factory=ModelConfig)
    auxiliary_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(temperature=0.9, max_tokens=512)
    )
    equivalence: EquivalenceConfig = Field(default_factory=EquivalenceConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    answer_extraction: AnswerExtractionConfig = Field(default_factory=AnswerExtractionConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_config(override_path: Optional[str | Path] = None) -> Config:
    """Load configuration, merging default.yaml with an optional override."""
    default_path = _PROJECT_ROOT / "config" / "default.yaml"

    merged: dict = {}
    if default_path.exists():
        with open(default_path) as f:
            merged = yaml.safe_load(f) or {}

    if override_path is not None:
        with open(override_path) as f:
            overrides = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, overrides)

    return Config(**merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_api_key(provider: str) -> str:
    """Retrieve the API key for a given provider from the environment."""
    env_map = {
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HF_TOKEN",
    }
    env_var = env_map.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    if not key or key.startswith("your-"):
        raise ValueError(
            f"API key for '{provider}' not set. "
            f"Set {env_var} in your .env file."
        )
    return key
