#!/usr/bin/env python3
"""
Re-check the confirmed flips of a run with the strict judge and the LLM grader.

Used to measure how many flips found under the lenient setup (lenient judge,
string-matched answers) survive the stricter checks.

    python scripts/audit_flips.py results --config config/full-run-strict.yaml

Writes <dir>/flip_audit.json and prints a summary.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report import winning_candidate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.equivalence.llm_judge import LLMJudge  # noqa: E402
from src.evaluation.answer_extract import AnswerExtractor  # noqa: E402
from src.models import create_llm, create_llm_from_config  # noqa: E402
from src.models.budget import BudgetGuard, set_budget  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--config", default="config/full-run-strict.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    b = cfg.budget
    set_budget(BudgetGuard(max_usd=1.0, prices=b.prices, max_usd_total=b.max_usd_total,
                           ledger_path=b.ledger_path or str(Path(cfg.experiment.output_dir) / "spend_ledger.json")))
    jc = cfg.equivalence.llm_judge
    judge = LLMJudge(create_llm(provider=jc.provider, model_name=jc.model_name, temperature=0.0), strict=True)
    grader = AnswerExtractor(method="grade", llm=create_llm_from_config(cfg.auxiliary_model))

    d = Path(args.results_dir)
    payload = json.loads((d / "attack_results.json").read_text())
    flips = [(r, winning_candidate(r)) for r in payload["results"] if r.get("success")]

    def audit(item):
        r, c = item
        verdict = judge.check_equivalence(r["question"], c["paraphrase"])
        grades = [grader.grade(r["question"], resp, r["ground_truth"], [])[0]
                  for resp in (c.get("responses") or [])]
        answered = [g for g in grades if g is not None]
        wrong = sum(1 for g in answered if g is False)
        return {
            "id": r["id"], "question": r["question"], "rewrite": c["paraphrase"],
            "ground_truth": r["ground_truth"],
            "strict_judge_equivalent": verdict["is_equivalent"],
            "strict_judge_explanation": verdict["explanation"],
            "graded_wrong": wrong, "graded_answered": len(answered),
            "graded_majority_wrong": bool(answered) and wrong / len(answered) > 0.5,
        }

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(audit, flips))
    for row in rows:
        row["survives"] = row["strict_judge_equivalent"] and row["graded_majority_wrong"]

    n = len(rows)
    summary = {
        "flips": n,
        "rejected_by_strict_judge": sum(not r["strict_judge_equivalent"] for r in rows),
        "not_wrong_when_graded": sum(not r["graded_majority_wrong"] for r in rows),
        "survive_both": sum(r["survives"] for r in rows),
        "attacked": len(payload["results"]),
    }
    (d / "flip_audit.json").write_text(json.dumps({"summary": summary, "flips": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
