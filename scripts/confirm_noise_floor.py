#!/usr/bin/env python3
"""
Apples-to-apples noise floor: put the baseline's unoptimized paraphrases
through the same checks as attack flips.

For each question the model knows (native accuracy >= gate), every naive
paraphrase that was majority-wrong is (1) checked for hidden characters,
(2) judged by the strict equivalence judge, and (3) re-scored on fresh samples
(evolution.confirm_samples). A question counts toward the confirmed noise floor
if at least one of its naive paraphrases survives all three.

    python scripts/confirm_noise_floor.py --config config/full-run-strict.yaml

Reads and writes <output_dir>/ (baselines.json -> noise_floor_confirmed.json).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report import wilson_ci  # noqa: E402
from src.attack.pipeline import AttackPipeline  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.loader import QAItem  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/full-run-strict.yaml")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.experiment.output_dir)
    pipe = AttackPipeline(cfg)              # sets the shared budget guard / ledger
    gate = cfg.evolution.min_native_accuracy
    n_confirm = cfg.evolution.confirm_samples

    baselines = json.loads((out_dir / "baselines.json").read_text())
    known = [q for q in baselines["per_question"] if (q.get("native_accuracy") or 0) >= gate - 1e-9]
    jobs = [(q, p) for q in known
            for p, r in zip(q["naive_paraphrases"], q["naive_error_rates"]) if r > 0.5]

    def check(job):
        q, p = job
        item = QAItem(id=q["id"], question=q["question"], answer=q["ground_truth"])
        eq = pipe.validator.validate(q["question"], p)
        fresh = None
        if eq.is_equivalent:
            fresh = pipe.scorer.score(item, p, num_samples=n_confirm).wrongness_score
        return {"id": q["id"], "paraphrase": p, "equivalent": eq.is_equivalent, "reason": eq.reason,
                "fresh_score": fresh, "survives": bool(eq.is_equivalent and fresh is not None and fresh > 0.5)}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(check, jobs))

    raw_qs = {q["id"] for q, _ in jobs}
    confirmed_qs = {r["id"] for r in rows if r["survives"]}
    lo, hi = wilson_ci(len(confirmed_qs), len(known))
    summary = {
        "known_questions": len(known),
        "raw_any_majority_wrong": len(raw_qs),
        "raw_rate": len(raw_qs) / len(known) if known else 0.0,
        "paraphrases_checked": len(rows),
        "rejected_by_judge": sum(not r["equivalent"] for r in rows),
        "failed_retest": sum(r["equivalent"] and not r["survives"] for r in rows),
        "confirmed_questions": len(confirmed_qs),
        "confirmed_rate": len(confirmed_qs) / len(known) if known else 0.0,
        "confirmed_ci95": (lo, hi),
    }
    (out_dir / "noise_floor_confirmed.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
