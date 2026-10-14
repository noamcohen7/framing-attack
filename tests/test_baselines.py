"""Tests for naive-paraphrase generation (scripted auxiliary LLM, no API)."""

from typing import Optional

from src.evaluation.baselines import (
    BaselineMeasurer, dedupe_rewrites, parse_numbered_rewrites,
)
from src.models.base import BaseLLM, LLMResponse


class ScriptedLLM(BaseLLM):
    def __init__(self, responses):
        super().__init__(model_name="scripted")
        self.responses = list(responses)
        self.prompts = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def _generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(text=self.responses.pop(0), model=self.model_name, provider="fake")


Q = "Which country hosted the 2024 Summer Olympics?"


def test_parse_numbered_rewrites():
    text = 'Here you go:\n1. "Where were the 2024 Olympics held?"\n2) Which nation hosted them?\n\n3: Third one'
    assert parse_numbered_rewrites(text) == [
        "Where were the 2024 Olympics held?", "Which nation hosted them?", "Third one",
    ]


def test_dedupe_drops_original_and_near_duplicates():
    cands = [
        "Which country hosted the 2024 Summer Olympics",          # original minus "?"
        "Which nation was the host of the 2024 Summer Olympic Games?",
        "Which nation was the host of the 2024 Summer Olympic Games?",
        "Which nation was host of the 2024 Summer Olympic Games?",  # one word dropped
        "In 2024, the Summer Olympics took place in which country?",
    ]
    assert dedupe_rewrites(Q, cands) == [
        "Which nation was the host of the 2024 Summer Olympic Games?",
        "In 2024, the Summer Olympics took place in which country?",
    ]


def test_generate_tops_up_shortfall_once():
    aux = ScriptedLLM([
        "1. Where were the 2024 Summer Olympics held?\n2. Where were the 2024 Summer Olympics held?\n3. Which nation hosted the 2024 Games?",
        "1. The 2024 Summer Olympics took place in what country?",
    ])
    m = BaselineMeasurer(target_llm=None, auxiliary_llm=aux, scorer=None)
    out = m.generate_naive_paraphrases(Q, n=3)
    assert out == [
        "Where were the 2024 Summer Olympics held?",
        "Which nation hosted the 2024 Games?",
        "The 2024 Summer Olympics took place in what country?",
    ]
    assert "Do not repeat" in aux.prompts[1] and "Rewrite the following question 1 different" in aux.prompts[1]


def test_generate_stops_after_max_calls():
    aux = ScriptedLLM(["1. " + Q, "1. " + Q])
    m = BaselineMeasurer(target_llm=None, auxiliary_llm=aux, scorer=None)
    assert m.generate_naive_paraphrases(Q, n=2) == []
    assert len(aux.prompts) == 2


def test_measure_all_parallel_keeps_order_and_checkpoints(tmp_path):
    import json
    from src.data.loader import QAItem
    from src.evaluation.scorer import ScoreResult

    class Scorer:
        def score_original(self, qa):
            return ScoreResult(paraphrase=qa.question, wrongness_score=0.0, num_samples=5,
                               num_correct=5, num_incorrect=0, num_abstained=0,
                               extracted_answers=["a"] * 5)

        def score(self, qa, p):
            return self.score_original(qa)

    class Aux(BaseLLM):
        @property
        def provider_name(self):
            return "fake"

        def _generate(self, prompt, temperature=None):
            return LLMResponse(text="1. Rewrite one entirely\n2. A totally different phrasing", model="m", provider="fake")

    items = [QAItem(id=f"q{i}", question=f"Question number {i}?", answer="a") for i in range(8)]
    m = BaselineMeasurer(target_llm=None, auxiliary_llm=Aux(model_name="m", requests_per_minute=None),
                         scorer=Scorer(), num_naive_paraphrases=2)
    out = tmp_path / "baselines.json"
    results, agg = m.measure_all(items, output_path=out, workers=4)
    assert [r.qa_item_id for r in results] == [f"q{i}" for i in range(8)]
    data = json.loads(out.read_text())
    assert [q["id"] for q in data["per_question"]] == [f"q{i}" for i in range(8)]
    assert data["per_question"][0]["native_accuracy"] == 1.0
    assert agg.num_attackable == 8
