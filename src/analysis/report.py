"""
Phase 2 analysis: attack success, noise floor, costs, and human review.

All functions are pure (dicts in, dicts out) so they are testable offline.
Inputs are the JSON files written by the pipeline:
  - attack_results.json  {results, skipped, failed}
  - baselines.json       {aggregate, per_question}
"""

from __future__ import annotations

import csv
import math
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Optional

from src.equivalence.validator import hidden_characters
from src.evaluation.answer_extract import AnswerExtractor

_YES = {"y", "yes", "1", "true", "t"}
_NO = {"n", "no", "0", "false", "f"}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved near 0 and 1)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def source_of(qid: str) -> str:
    return "custom" if qid.startswith("recency") else "triviaqa" if qid.startswith("triviaqa") else "other"


def winning_candidate(result: dict) -> Optional[dict]:
    """The rephrasing that passed the fresh-sample re-test (None if no flip)."""
    if not result.get("success"):
        return None
    # (callers check `violates_constraints` separately)
    confirmed = [c for c in result.get("all_valid_candidates", [])
                 if c.get("confirm", {}).get("score", 0) > 0.5]
    if confirmed:
        return max(confirmed, key=lambda c: c["confirm"]["score"])
    return result.get("best_candidate")  # confirmation disabled


def violates_constraints(result: dict) -> bool:
    """A flip whose winning rephrasing breaks a hard rule (hidden characters)."""
    c = winning_candidate(result)
    return c is not None and bool(hidden_characters(c["paraphrase"]))


def attack_summary(payload: dict, human: Optional[dict[str, bool]] = None) -> dict:
    """
    Attack success rate overall and per source, with Wilson CIs.

    `human`: question id -> True if a human confirmed the flip is valid (rewrite
    equivalent AND answer really wrong). When given, also reports the
    human-verified success rate (unreviewed flips count as not verified).
    """
    results = payload.get("results", [])
    # Flips that break a hard constraint count as failures (conservative: the
    # search for that question stopped at an invalid rephrasing).
    invalid = {r["id"] for r in results if r.get("success") and violates_constraints(r)}
    results = [dict(r, success=False) if r["id"] in invalid else r for r in results]
    out: dict = {"attacked": len(results), "skipped": len(payload.get("skipped", [])),
                 "failed": len(payload.get("failed", [])),
                 "invalid_hidden_characters": sorted(invalid)}

    def rate(rows, key="flips"):
        k = sum(1 for r in rows if r.get("success"))
        lo, hi = wilson_ci(k, len(rows))
        return {"n": len(rows), key: k, "rate": k / len(rows) if rows else 0.0, "ci95": (lo, hi)}

    out["overall"] = rate(results)
    out["by_source"] = {s: rate([r for r in results if source_of(r["id"]) == s])
                        for s in sorted({source_of(r["id"]) for r in results})}

    flips = [r for r in results if r.get("success")]
    out["generations_to_flip"] = _describe([r["generations_run"] for r in flips])
    out["candidates_to_flip"] = _describe([r["total_candidates"] for r in flips])
    total_cands = sum(r["total_candidates"] for r in results)
    rejected = sum(len(r.get("rejected_candidates", [])) for r in results)
    out["judge_rejection_rate"] = rejected / total_cands if total_cands else 0.0

    retested = [c for r in results for c in r.get("all_valid_candidates", []) if "confirm" in c]
    failed_retest = sum(1 for c in retested if c["confirm"]["score"] <= 0.5)
    out["retest"] = {"retested": len(retested), "failed_retest": failed_retest}

    win_pairs = [(r, winning_candidate(r)) for r in flips]
    wins = [c for _, c in win_pairs]
    out["winner_abstention_rate"] = _mean(
        [c["confirm"]["num_abstained"] / c["confirm"]["num_samples"]
         for c in wins if c and c.get("confirm")])
    # Answers on winning rewrites, minus those matching the expected answer
    # (fresh-sample answers include the correct minority).
    out["wrong_answers"] = Counter(
        a for r, c in win_pairs if c and c.get("confirm")
        for a in c["confirm"]["extracted_answers"]
        if a and a != "NO_ANSWER" and not AnswerExtractor.answers_match(a, r["ground_truth"])
    ).most_common(10)

    if human is not None:
        verified = sum(1 for r in flips if human.get(r["id"]))
        lo, hi = wilson_ci(verified, len(results))
        out["human_verified"] = {"reviewed": sum(1 for r in flips if r["id"] in human),
                                 "verified_flips": verified,
                                 "rate": verified / len(results) if results else 0.0,
                                 "ci95": (lo, hi)}
    return out


def noise_floor(baselines: dict, min_native_accuracy: float = 0.8) -> dict:
    """
    How much plain, unoptimized rewrites already move the model, on the
    questions it knows (native accuracy >= threshold). NOT an equal-budget
    comparison: the attack tries far more rewrites per question.
    """
    qs = [q for q in baselines.get("per_question", [])
          if (q.get("native_accuracy") or 0) >= min_native_accuracy - 1e-9]
    rates = [r for q in qs for r in q.get("naive_error_rates", [])]
    per_rewrite = sum(1 for r in rates if r > 0.5) / len(rates) if rates else 0.0
    any_of_k = sum(1 for q in qs if any(r > 0.5 for r in q.get("naive_error_rates", [])))
    k = round(_mean([len(q.get("naive_error_rates", [])) for q in qs])) if qs else 0
    return {
        "questions": len(qs),
        "rewrites_per_question": k,
        "per_rewrite_majority_wrong": per_rewrite,
        "questions_with_any_majority_wrong": any_of_k,
        "any_of_k_rate": any_of_k / len(qs) if qs else 0.0,
        "any_of_k_ci95": wilson_ci(any_of_k, len(qs)),
        "native_error_rate_all": baselines.get("aggregate", {}).get("native_error_rate"),
    }


def write_review_sheet(payload: dict, path: Path) -> int:
    """CSV of every confirmed flip for blind-ish human review. Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "original_question", "rewrite", "expected_answer",
                    "model_answers_on_rewrite", "auto_flag", "rewrite_equivalent", "answer_wrong", "notes"])
        for r in payload.get("results", []):
            c = winning_candidate(r)
            if c is None:
                continue
            answers = (c.get("confirm") or {}).get("extracted_answers") or c.get("extracted_answers") or []
            hidden = hidden_characters(c["paraphrase"])
            flag = f"INVALID: hidden characters {', '.join(sorted(set(hidden)))}" if hidden else ""
            w.writerow([r["id"], r["question"], c["paraphrase"], r["ground_truth"],
                        " | ".join(a for a in answers if a), flag,
                        "n" if hidden else "", "", ""])
            rows += 1
    return rows


def read_review_sheet(path: Path) -> dict[str, bool]:
    """id -> valid flip (both columns yes). Rows with either column blank are skipped."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eq, wrong = _yn(row.get("rewrite_equivalent", "")), _yn(row.get("answer_wrong", ""))
            if eq is None or wrong is None:
                continue
            out[row["id"]] = eq and wrong
    return out


def format_report(summary: dict, floor: Optional[dict] = None, title: str = "Attack results") -> str:
    def pct(x):
        return f"{100 * x:.1f}%"

    def ci(t):
        return f"[{pct(t[0])}, {pct(t[1])}]"

    o = summary["overall"]
    lines = [f"# {title}", "",
             f"Questions attacked: **{summary['attacked']}** "
             f"(skipped as not known: {summary['skipped']}, failed: {summary['failed']})", ""]
    if summary.get("invalid_hidden_characters"):
        lines += [f"Flips discarded for hidden characters: {len(summary['invalid_hidden_characters'])} "
                  f"({', '.join(summary['invalid_hidden_characters'])})", ""]
    lines += [
             "| Group | Attacked | Confirmed flips | Success rate | 95% CI |",
             "|---|---|---|---|---|",
             f"| **All** | {o['n']} | {o['flips']} | **{pct(o['rate'])}** | {ci(o['ci95'])} |"]
    for s, g in summary["by_source"].items():
        lines.append(f"| {s} | {g['n']} | {g['flips']} | {pct(g['rate'])} | {ci(g['ci95'])} |")
    if "human_verified" in summary:
        h = summary["human_verified"]
        lines += ["", f"**Human-verified success rate: {pct(h['rate'])}** {ci(h['ci95'])} "
                      f"({h['verified_flips']} verified of {h['reviewed']} reviewed flips)"]
    if floor:
        lines += ["", "## Noise floor (unoptimized rewrites, baseline)", "",
                  f"- Questions the model knows (native accuracy >= 80%): {floor['questions']}",
                  f"- Single plain rewrite majority-wrong: {pct(floor['per_rewrite_majority_wrong'])}",
                  f"- Questions with >= 1 of {floor['rewrites_per_question']} plain rewrites majority-wrong: "
                  f"{pct(floor['any_of_k_rate'])} {ci(floor['any_of_k_ci95'])}",
                  "- Not an equal-budget comparison: the attack tries far more rewrites per question."]
    g, c = summary["generations_to_flip"], summary["candidates_to_flip"]
    r = summary["retest"]
    lines += ["", "## Search", "",
              f"- Generations to flip: median {g['median']}, mean {g['mean']:.1f}",
              f"- Candidates generated per successful question: median {c['median']}, mean {c['mean']:.1f}",
              f"- Equivalence judge rejected {pct(summary['judge_rejection_rate'])} of candidates",
              f"- Re-test: {r['retested']} would-be flips re-tested on fresh samples, "
              f"{r['failed_retest']} ({pct(r['failed_retest'] / r['retested']) if r['retested'] else 'n/a'}) "
              f"did not hold up",
              f"- Refusal rate on winning rewrites: {pct(summary['winner_abstention_rate'])}"]
    if summary["wrong_answers"]:
        lines += ["", "Most common wrong answers: "
                  + ", ".join(f"{a} ({n})" for a, n in summary["wrong_answers"])]
    return "\n".join(lines) + "\n"


def _describe(xs: list) -> dict:
    if not xs:
        return {"median": 0, "mean": 0.0}
    return {"median": st.median(xs), "mean": st.mean(xs)}


def _mean(xs: list) -> float:
    return st.mean(xs) if xs else 0.0


def _yn(v: str) -> Optional[bool]:
    v = (v or "").strip().lower()
    return True if v in _YES else False if v in _NO else None
