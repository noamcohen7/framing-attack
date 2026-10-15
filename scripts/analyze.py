#!/usr/bin/env python3
"""
Phase 2 analysis of an attack run.

    # Report + review sheet for the strict main run
    python scripts/analyze.py results/strict

    # After filling review.csv (rewrite_equivalent / answer_wrong: y or n)
    python scripts/analyze.py results/strict --review results/strict/review.csv

Writes <dir>/report.md and, unless it exists, <dir>/review.csv.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report import (  # noqa: E402
    attack_summary, format_report, noise_floor, read_review_sheet, write_review_sheet,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--review", help="Filled-in review CSV for the human-verified rate")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    d = Path(args.results_dir)
    with open(d / "attack_results.json") as f:
        payload = json.load(f)
    floor = None
    if (d / "baselines.json").exists():
        with open(d / "baselines.json") as f:
            floor = noise_floor(json.load(f))

    human = read_review_sheet(Path(args.review)) if args.review else None
    summary = attack_summary(payload, human)
    report = format_report(summary, floor, title=args.title or f"Attack results ({d})")
    (d / "report.md").write_text(report)
    print(report)

    sheet = d / "review.csv"
    if not args.review and not sheet.exists():
        n = write_review_sheet(payload, sheet)
        print(f"Review sheet: {sheet} ({n} flips). Fill rewrite_equivalent and answer_wrong with y/n.")


if __name__ == "__main__":
    main()
