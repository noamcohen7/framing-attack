"""Tests for Phase 2 analysis (synthetic results, no API)."""

import csv

import pytest

from src.analysis.report import (
    attack_summary, format_report, noise_floor, read_review_sheet, wilson_ci,
    winning_candidate, write_review_sheet,
)


def cand(p, score, confirm=None):
    c = {"paraphrase": p, "score": score, "extracted_answers": ["x"]}
    if confirm is not None:
        c["confirm"] = {"score": confirm, "num_samples": 10, "num_abstained": 1,
                        "extracted_answers": ["wrong"] * 7 + ["right"] * 3}
    return c


def result(qid, success, cands, gens=1, total=20, rejected=0):
    return {"id": qid, "question": f"{qid}?", "ground_truth": "right", "success": success,
            "best_candidate": cands[0] if cands else None, "all_valid_candidates": cands,
            "generations_run": gens, "total_candidates": total,
            "rejected_candidates": [{}] * rejected}


PAYLOAD = {
    "results": [
        result("recency_001", True, [cand("noisy", 0.4, confirm=0.4), cand("win", 0.7, confirm=0.7)], gens=1),
        result("triviaqa_1", True, [cand("w2", 1.0, confirm=0.8)], gens=5, total=60, rejected=6),
        result("triviaqa_2", False, [cand("no", 0.2)], gens=20, total=172, rejected=10),
        result("triviaqa_3", False, [], gens=20, total=172),
    ],
    "skipped": [{"id": "triviaqa_9"}],
    "failed": [],
}


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3) and hi == pytest.approx(0.1744, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_winning_candidate_is_the_confirmed_one():
    assert winning_candidate(PAYLOAD["results"][0])["paraphrase"] == "win"
    assert winning_candidate(PAYLOAD["results"][2]) is None


def test_attack_summary():
    s = attack_summary(PAYLOAD)
    assert s["attacked"] == 4 and s["skipped"] == 1
    assert s["overall"]["flips"] == 2 and s["overall"]["rate"] == 0.5
    assert s["by_source"]["custom"]["n"] == 1 and s["by_source"]["triviaqa"]["flips"] == 1
    assert s["retest"] == {"retested": 3, "failed_retest": 1}
    assert s["judge_rejection_rate"] == pytest.approx(16 / 424)
    assert s["generations_to_flip"]["median"] == 3
    assert s["wrong_answers"][0] == ("wrong", 14)
    assert all(a != "right" for a, _ in s["wrong_answers"])   # expected answer excluded


def test_human_verified_rate():
    s = attack_summary(PAYLOAD, human={"recency_001": True, "triviaqa_1": False})
    h = s["human_verified"]
    assert h["verified_flips"] == 1 and h["reviewed"] == 2 and h["rate"] == 0.25


def test_noise_floor_uses_known_questions_only():
    b = {"aggregate": {"native_error_rate": 0.4}, "per_question": [
        {"native_accuracy": 1.0, "naive_error_rates": [0.0, 0.8, 0.0]},
        {"native_accuracy": 0.8, "naive_error_rates": [0.0, 0.0, 0.2]},
        {"native_accuracy": 0.2, "naive_error_rates": [1.0, 1.0, 1.0]},  # not known: excluded
    ]}
    f = noise_floor(b)
    assert f["questions"] == 2 and f["rewrites_per_question"] == 3
    assert f["per_rewrite_majority_wrong"] == pytest.approx(1 / 6)
    assert f["any_of_k_rate"] == 0.5


def test_review_sheet_roundtrip(tmp_path):
    path = tmp_path / "review.csv"
    assert write_review_sheet(PAYLOAD, path) == 2
    rows = list(csv.DictReader(open(path)))
    assert rows[0]["rewrite"] == "win"
    rows[0]["rewrite_equivalent"], rows[0]["answer_wrong"] = "y", "y"
    rows[1]["rewrite_equivalent"], rows[1]["answer_wrong"] = "y", "n"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    assert read_review_sheet(path) == {"recency_001": True, "triviaqa_1": False}


def test_format_report_renders():
    text = format_report(attack_summary(PAYLOAD), noise_floor({"per_question": []}))
    assert "Success rate" in text and "**50.0%**" in text and "Noise floor" in text


def test_hidden_character_flips_are_discarded(tmp_path):
    payload = {"results": [
        result("triviaqa_1", True, [cand("sacr​edotal?", 1.0, confirm=0.9)]),
        result("triviaqa_2", True, [cand("clean?", 1.0, confirm=0.9)]),
    ], "skipped": [], "failed": []}
    s = attack_summary(payload)
    assert s["overall"]["flips"] == 1 and s["invalid_hidden_characters"] == ["triviaqa_1"]
    path = tmp_path / "review.csv"
    write_review_sheet(payload, path)
    rows = list(csv.DictReader(open(path)))
    assert rows[0]["auto_flag"].startswith("INVALID") and rows[0]["rewrite_equivalent"] == "n"
    assert rows[1]["auto_flag"] == ""
