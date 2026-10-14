#!/usr/bin/env python3
"""
Calibrate the equivalence judge against human labels.

    # 1. Draw pairs for labeling (half judge-accepted, half judge-rejected)
    python scripts/calibrate_validator.py sample --n 200

    # 2. Open results/calibration/pairs.csv, fill human_equivalent with y / n
    #    (does the paraphrase ask for exactly the same fact?). Judge verdicts
    #    are in key.json, not the CSV, so labeling is blind.

    # 3. Report judge precision / recall
    python scripts/calibrate_validator.py score
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.equivalence.calibration import (  # noqa: E402
    collect_pairs, read_labels, score_labels, stratified_sample, write_labeling_files,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "score"])
    parser.add_argument("--results", default="results/attack_results.json")
    parser.add_argument("--out-dir", default="results/calibration")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    csv_path, key_path = out / "pairs.csv", out / "key.json"

    if args.command == "sample":
        if csv_path.exists():
            sys.exit(f"{csv_path} exists; move it first so labels are not overwritten.")
        with open(args.results) as f:
            pairs = collect_pairs(json.load(f))
        sample = stratified_sample(pairs, args.n, args.seed)
        write_labeling_files(pairs, sample, csv_path, key_path)
        n_acc = sum(p["judge_equivalent"] for p in sample)
        print(f"{len(pairs)} judged pairs in {args.results}")
        print(f"Wrote {len(sample)} pairs ({n_acc} accepted, {len(sample) - n_acc} rejected) to {csv_path}")
        if len(sample) < args.n:
            print(f"Only {len(sample)} available; run a larger attack for the full {args.n}.")
        return

    with open(key_path) as f:
        key = json.load(f)
    report = score_labels(read_labels(csv_path), key)
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"Labeled pairs:  {report.num_labeled} ({report.num_unlabeled} unlabeled)")
    print(f"Confusion:      TP={report.true_pos} FP={report.false_pos} FN={report.false_neg} TN={report.true_neg}")
    print(f"Precision:      {fmt(report.precision)}   (accepted pairs that really are equivalent)")
    print(f"Recall:         {fmt(report.recall)}   (equivalent pairs the judge accepts; population-weighted)")
    print(f"Specificity:    {fmt(report.specificity)}   (non-equivalent pairs the judge rejects; population-weighted)")
    with open(out / "report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
