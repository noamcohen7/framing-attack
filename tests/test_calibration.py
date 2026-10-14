"""Tests for validator calibration (no API)."""

import pytest

from src.equivalence.calibration import (
    collect_pairs, parse_label, read_labels, score_labels, stratified_sample, write_labeling_files,
)

PAYLOAD = {
    "results": [
        {
            "id": "q1", "question": "Who painted the Sistine Chapel ceiling?",
            "all_valid_candidates": [{"paraphrase": f"acc {i}"} for i in range(6)],
            "rejected_candidates": [{"paraphrase": f"rej {i}", "reason": "x"} for i in range(3)],
        },
        {"id": "q2", "question": "Q2?", "all_valid_candidates": [{"paraphrase": "acc q2"}]},
    ]
}


def test_collect_pairs():
    pairs = collect_pairs(PAYLOAD)
    assert len(pairs) == 10
    assert sum(p["judge_equivalent"] for p in pairs) == 7


def test_stratified_sample_balances_and_tops_up():
    pairs = collect_pairs(PAYLOAD)
    s = stratified_sample(pairs, n=4)
    assert sum(p["judge_equivalent"] for p in s) == 2
    # only 3 rejected exist: n=8 -> 3 rejected + 5 accepted
    s = stratified_sample(pairs, n=8)
    assert sum(not p["judge_equivalent"] for p in s) == 3 and len(s) == 8


@pytest.mark.parametrize("raw,expected", [
    ("y", True), (" YES ", True), ("1", True), ("n", False), ("No", False), ("", None), ("?", None),
])
def test_parse_label(raw, expected):
    assert parse_label(raw) is expected


def test_score_labels_reweights_by_population():
    # Population: 90 accepted, 10 rejected. Sample 4 of each.
    key = {
        "population": {"accepted": 90, "rejected": 10},
        "pairs": {str(i): {"judge_equivalent": i < 4} for i in range(8)},
    }
    # Accepted: 3 of 4 truly equivalent. Rejected: 2 of 4 truly equivalent.
    labels = {"0": True, "1": True, "2": True, "3": False,
              "4": True, "5": True, "6": False, "7": False}
    r = score_labels(labels, key)
    assert (r.true_pos, r.false_pos, r.false_neg, r.true_neg) == (3, 1, 2, 2)
    assert r.precision == pytest.approx(0.75)
    # equivalent: 90*.75=67.5 accepted, 10*.5=5 rejected -> recall 67.5/72.5
    assert r.recall == pytest.approx(67.5 / 72.5)
    # not equivalent: 90*.25=22.5 accepted, 10*.5=5 rejected -> specificity 5/27.5
    assert r.specificity == pytest.approx(5 / 27.5)


def test_roundtrip_files(tmp_path):
    pairs = collect_pairs(PAYLOAD)
    sample = stratified_sample(pairs, n=4)
    csv_path, key_path = tmp_path / "pairs.csv", tmp_path / "key.json"
    write_labeling_files(pairs, sample, csv_path, key_path)
    text = csv_path.read_text()
    assert "judge" not in text.lower()          # blind labeling
    lines = text.splitlines()
    lines[1] = lines[1].rstrip(",") + ",y"
    csv_path.write_text("\n".join(lines) + "\n")
    assert read_labels(csv_path) == {"0": True}
