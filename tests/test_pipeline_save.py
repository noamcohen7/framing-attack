"""Tests for attack result serialization (no API calls)."""

import json

from src.attack.evolutionary import Candidate, EvolutionResult
from src.attack.pipeline import AttackPipeline
from src.config import Config
from src.data.loader import QAItem
from src.evaluation.scorer import ScoreResult


def _pipeline(tmp_path):
    cfg = Config()
    cfg.experiment.output_dir = str(tmp_path)
    return AttackPipeline(cfg)


def test_save_includes_candidate_details_and_failures(tmp_path):
    p = _pipeline(tmp_path)
    qa = QAItem(id="q1", question="Q?", answer="A")
    sr = ScoreResult(paraphrase="P", wrongness_score=1.0, num_samples=2, num_correct=0,
                     num_incorrect=2, num_abstained=1, responses=["I don't know", "B"],
                     extracted_answers=["NO_ANSWER", "b"], correct_flags=[False, False])
    best = Candidate(paraphrase="P", score=1.0, generation=0, score_result=sr)
    other = Candidate(paraphrase="P2", score=0.0, generation=0)
    res = EvolutionResult(qa_item=qa, best_candidate=best, all_candidates=[other, best],
                          generations_run=1, success=True)

    p._save_attack_results([res], failed=[{"id": "q2", "question": "Q2?", "error": "boom"}])

    data = json.loads((tmp_path / "attack_results.json").read_text())
    assert data["target_model"].startswith("gemini/")
    r = data["results"][0]
    assert r["id"] == "q1" and r["success"] is True
    assert r["best_candidate"]["responses"] == ["I don't know", "B"]
    assert r["best_candidate"]["num_abstained"] == 1
    # sorted best-first
    assert [c["paraphrase"] for c in r["all_valid_candidates"]] == ["P", "P2"]
    assert data["failed"][0]["id"] == "q2"


def test_save_with_no_best_candidate(tmp_path):
    p = _pipeline(tmp_path)
    qa = QAItem(id="q1", question="Q?", answer="A")
    p._save_attack_results([EvolutionResult(qa_item=qa)])
    data = json.loads((tmp_path / "attack_results.json").read_text())
    assert data["results"][0]["best_paraphrase"] is None
    assert data["results"][0]["best_candidate"] is None
