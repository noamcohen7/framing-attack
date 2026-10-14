"""
Validator calibration: compare the equivalence judge against human labels.

Workflow (see scripts/calibrate_validator.py):
  1. sample:  draw (original, paraphrase) pairs from attack results, half the
              judge accepted and half it rejected, into a CSV for blind
              labeling. The judge's verdicts go to a separate key file.
  2. label:   a human fills the `human_equivalent` column with y / n.
  3. score:   precision / recall of the judge's "equivalent" verdict.

Because accepted and rejected pairs are sampled in equal numbers rather than
in proportion to how often the judge accepts, recall and specificity are
reweighted by the population counts stored in the key file. Precision is
estimated from the accepted stratum alone and needs no weighting.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_YES = {"y", "yes", "1", "true", "t"}
_NO = {"n", "no", "0", "false", "f"}


def collect_pairs(attack_payload: dict) -> list[dict]:
    """Every judged (original, paraphrase) pair in an attack_results.json payload."""
    pairs = []
    for r in attack_payload.get("results", []):
        for c in r.get("all_valid_candidates", []):
            pairs.append({"question_id": r["id"], "original": r["question"],
                          "paraphrase": c["paraphrase"], "judge_equivalent": True})
        for c in r.get("rejected_candidates", []):
            pairs.append({"question_id": r["id"], "original": r["question"],
                          "paraphrase": c["paraphrase"], "judge_equivalent": False})
    return pairs


def stratified_sample(pairs: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to n/2 accepted and n/2 rejected pairs; tops up from the other side if one is short."""
    rng = random.Random(seed)
    accepted = [p for p in pairs if p["judge_equivalent"]]
    rejected = [p for p in pairs if not p["judge_equivalent"]]
    rng.shuffle(accepted)
    rng.shuffle(rejected)
    n_acc = min(len(accepted), n // 2)
    n_rej = min(len(rejected), n - n_acc)
    n_acc = min(len(accepted), n - n_rej)
    sample = accepted[:n_acc] + rejected[:n_rej]
    rng.shuffle(sample)  # don't let row order leak the verdict
    return sample


def write_labeling_files(pairs: list[dict], sample: list[dict], csv_path: Path, key_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "original", "paraphrase", "human_equivalent"])
        for i, p in enumerate(sample):
            w.writerow([i, p["original"], p["paraphrase"], ""])
    key = {
        "population": {
            "accepted": sum(1 for p in pairs if p["judge_equivalent"]),
            "rejected": sum(1 for p in pairs if not p["judge_equivalent"]),
        },
        "pairs": {str(i): {"judge_equivalent": p["judge_equivalent"], "question_id": p["question_id"]}
                  for i, p in enumerate(sample)},
    }
    with open(key_path, "w") as f:
        json.dump(key, f, indent=2)


def parse_label(value: str) -> Optional[bool]:
    v = value.strip().lower()
    if v in _YES:
        return True
    if v in _NO:
        return False
    return None


@dataclass
class CalibrationReport:
    num_labeled: int
    num_unlabeled: int
    # Raw counts on the labeled sample (judge verdict x human label)
    true_pos: int     # judge equivalent, human equivalent
    false_pos: int    # judge equivalent, human NOT equivalent  -> meaning drift let through
    false_neg: int    # judge rejected,   human equivalent      -> valid rewrite wasted
    true_neg: int
    precision: Optional[float]     # P(human eq | judge accepts)
    recall: Optional[float]        # P(judge accepts | human eq), population-weighted
    specificity: Optional[float]   # P(judge rejects | human not eq), population-weighted

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def score_labels(labels: dict[str, bool], key: dict) -> CalibrationReport:
    """labels: pair_id -> human says equivalent. key: as written by write_labeling_files."""
    tp = fp = fn = tn = 0
    for pid, info in key["pairs"].items():
        if pid not in labels:
            continue
        judge, human = info["judge_equivalent"], labels[pid]
        if judge and human:
            tp += 1
        elif judge:
            fp += 1
        elif human:
            fn += 1
        else:
            tn += 1

    pop_acc, pop_rej = key["population"]["accepted"], key["population"]["rejected"]
    p_eq_acc = _ratio(tp, tp + fp)   # human-equivalent rate among accepted
    p_eq_rej = _ratio(fn, fn + tn)   # human-equivalent rate among rejected

    recall = specificity = None
    if p_eq_acc is not None and p_eq_rej is not None:
        eq_acc, eq_rej = pop_acc * p_eq_acc, pop_rej * p_eq_rej
        neq_acc, neq_rej = pop_acc * (1 - p_eq_acc), pop_rej * (1 - p_eq_rej)
        recall = _ratio(eq_acc, eq_acc + eq_rej)
        specificity = _ratio(neq_rej, neq_rej + neq_acc)

    labeled = tp + fp + fn + tn
    return CalibrationReport(
        num_labeled=labeled, num_unlabeled=len(key["pairs"]) - labeled,
        true_pos=tp, false_pos=fp, false_neg=fn, true_neg=tn,
        precision=p_eq_acc, recall=recall, specificity=specificity,
    )


def read_labels(csv_path: Path) -> dict[str, bool]:
    labels = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = parse_label(row.get("human_equivalent", ""))
            if label is not None:
                labels[str(row["pair_id"])] = label
    return labels
