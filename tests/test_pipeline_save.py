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
    cfg.budget.prices = {"gemini-3.5-flash-lite": {"input_per_m": 0.30, "output_per_m": 2.50}}
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


# --- run_attack: resume, parallel workers, budget stop (fake scorer/searcher) ---

import threading

from src.models.budget import BudgetExceededError


class _Gate:
    """score_original: questions in `unknown` fail the gate."""

    def __init__(self, unknown=()):
        self.unknown = set(unknown)

    def score_original(self, qa):
        ok = qa.id not in self.unknown
        return ScoreResult(paraphrase=qa.question, wrongness_score=0.0, num_samples=5,
                           num_correct=5 if ok else 0, num_incorrect=0 if ok else 5,
                           num_abstained=0, extracted_answers=["x"] * 5)


class _Searcher:
    def __init__(self, raise_on=()):
        self.raise_on = set(raise_on)
        self.seen = []
        self.lock = threading.Lock()

    def search(self, qa):
        with self.lock:
            self.seen.append(qa.id)
        if qa.id in self.raise_on:
            raise BudgetExceededError("cap")
        win = qa.id.endswith("1")
        best = Candidate(paraphrase=f"P{qa.id}", score=1.0 if win else 0.0)
        return EvolutionResult(qa_item=qa, best_candidate=best, all_candidates=[best], success=win)


def _items(n):
    return [QAItem(id=f"q{i}", question=f"Q{i}?", answer="A") for i in range(n)]


def _wired(tmp_path, gate, searcher):
    p = _pipeline(tmp_path)
    p._scorer, p._searcher = gate, searcher
    return p


def test_skipped_questions_are_recorded(tmp_path):
    p = _wired(tmp_path, _Gate(unknown={"q2"}), _Searcher())
    p.run_attack(_items(3))
    data = json.loads((tmp_path / "attack_results.json").read_text())
    assert [r["id"] for r in data["results"]] == ["q0", "q1"]
    assert data["skipped"][0]["id"] == "q2" and data["skipped"][0]["native_accuracy"] == 0.0


def test_resume_keeps_prior_results_and_skips_done(tmp_path):
    p = _wired(tmp_path, _Gate(unknown={"q2"}), _Searcher())
    p.run_attack(_items(3))
    s2 = _Searcher()
    p2 = _wired(tmp_path, _Gate(unknown={"q2"}), s2)
    p2.run_attack(_items(5), resume=True)
    assert s2.seen == ["q3", "q4"]          # q0, q1 attacked and q2 skipped before
    data = json.loads((tmp_path / "attack_results.json").read_text())
    assert sorted(r["id"] for r in data["results"]) == ["q0", "q1", "q3", "q4"]
    assert [s["id"] for s in data["skipped"]] == ["q2"]


def test_without_resume_starts_fresh(tmp_path):
    _wired(tmp_path, _Gate(), _Searcher()).run_attack(_items(2))
    s2 = _Searcher()
    _wired(tmp_path, _Gate(), s2).run_attack(_items(2))
    assert s2.seen == ["q0", "q1"]


def test_parallel_workers_attack_everything(tmp_path):
    s = _Searcher()
    _wired(tmp_path, _Gate(), s).run_attack(_items(12), workers=4)
    assert sorted(s.seen) == sorted(f"q{i}" for i in range(12))
    data = json.loads((tmp_path / "attack_results.json").read_text())
    assert len(data["results"]) == 12


def test_budget_stop_halts_remaining_questions(tmp_path):
    s = _Searcher(raise_on={"q1"})
    _wired(tmp_path, _Gate(), s).run_attack(_items(5))
    assert s.seen == ["q0", "q1"]           # nothing after the cap tripped
    data = json.loads((tmp_path / "attack_results.json").read_text())
    assert [r["id"] for r in data["results"]] == ["q0"]
