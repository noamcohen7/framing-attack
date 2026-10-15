# Framing Attack

Black-box attack on LLM factual question answering via semantically-equivalent
paraphrasing. Code for the report *Framing the Truth: Flipping Black-Box LLM
Answers via Semantically-Equivalent Paraphrasing* (Trustworthy ML, TAU, 2026).

An auxiliary LLM proposes rephrasings of a factual question, a strict LLM
judge rejects any that change the question, and an evolutionary search keeps
the rephrasings that most often make the target model answer wrongly. Flips
are confirmed on fresh samples and graded against the original question.

Main result (report, Table 1): against `gpt-oss-20b`, 21.7% of attacked
questions flip under automatic checks and 12.5% after manual review, against a
5.4% noise floor for unoptimized paraphrases under the same checks.

## 1. Requirements

- Python 3.10+ (developed on 3.12). No GPU needed.
- API keys (paid tiers; free tiers are far too rate-limited for the full run):
  - **DeepInfra** (`DEEPINFRA_API_KEY`): target `openai/gpt-oss-20b` and
    auxiliary `openai/gpt-oss-120b`. Prepaid balance; add credit before running.
  - **Google Gemini** (`GEMINI_API_KEY`): judge `gemini-3.5-flash-lite`.
    Billing must be enabled on the key's project.
- Internet access for the whole run (every step is an API call).

Optional providers supported by the code: Groq (`GROQ_API_KEY`), OpenAI
(`OPENAI_API_KEY`), OpenRouter (`OPENROUTER_API_KEY`).

## 2. Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"                  # API clients + tests (no torch)
.venv/bin/pip install "httpx[socks]" truststore    # needed behind TLS-inspecting proxies
cp .env.example .env                               # then add your keys (see below)
.venv/bin/python -m pytest                         # offline unit tests (154, ~1 s)
```

`.env` (git-ignored) needs at least:

```
DEEPINFRA_API_KEY=...
GEMINI_API_KEY=...
```

Optional: `.venv/bin/pip install -e ".[nli]"` adds a local NLI validator
(torch, ~2 GB). The reported runs did not use it (`equivalence.nli_model: null`).

## 3. Reproducing the report

All commands run from the repository root. Outputs go to `results/strict/`
(git-ignored). Total cost of all runs in the report was about $8.64;
a hard budget cap (below) stops any run that exceeds its limit.

```bash
# (a) Baseline: native accuracy + 5 unoptimized paraphrases per question,
#     350 questions (50 custom + 300 TriviaQA). ~21k calls, ~$0.40, ~40 min.
.venv/bin/python scripts/run_baselines.py --config config/full-run-strict.yaml --workers 8

# (b) Main attack: random 120 attackable questions (seed 42), 10 generations,
#     strict judge, LLM grading. ~90k calls, ~$3, ~2.5 h.
.venv/bin/python scripts/run_attack.py --config config/full-run-strict.yaml \
    --workers 32 --shuffle-seed 42 --max-attacked 120

# (c) Noise floor under the same checks (strict judge + re-test).
.venv/bin/python scripts/confirm_noise_floor.py --config config/full-run-strict.yaml

# (d) Report: Table 1 numbers + review sheet (results/strict/report.md, review.csv)
.venv/bin/python scripts/analyze.py results/strict

# (e) Manual review: fill rewrite_equivalent / answer_wrong (y/n) in
#     results/strict/review.csv, then compute the human-verified rate:
.venv/bin/python scripts/analyze.py results/strict --review results/strict/review.csv
```

Lenient comparison (report, Table 2): the original pipeline with the lenient
judge and string-matched answers, then its flips re-checked with the strict
checks:

```bash
.venv/bin/python scripts/run_baselines.py --config config/full-run.yaml --workers 8
.venv/bin/python scripts/run_attack.py    --config config/full-run.yaml --workers 16
.venv/bin/python scripts/audit_flips.py results --config config/full-run-strict.yaml
```

(The reported lenient run was stopped after 90 attacked questions.)

Useful flags:

| Flag | Script | Meaning |
|---|---|---|
| `--workers N` | both | Questions processed in parallel (rate limits and budget are shared) |
| `--resume` | `run_attack.py` | Keep existing results; skip questions already attacked or skipped |
| `--max-attacked N` | `run_attack.py` | Stop after N questions pass the native-accuracy gate |
| `--shuffle-seed S` | `run_attack.py` | Shuffle question order (mixes datasets before `--max-attacked`) |
| `--dataset`, `--max-samples` | both | `custom`, `triviaqa` or `combined`; number of questions |

Model outputs are sampled, so rerunning gives slightly different numbers.

## 4. Quick smoke test (minutes, cents)

```bash
.venv/bin/python scripts/run_attack.py --config config/three-model-smoke.yaml \
    --dataset custom --max-samples 3
```

(`three-model-smoke.yaml` uses Groq for target and auxiliary; set
`GROQ_API_KEY`, or edit the providers to `deepinfra`.)

## 5. Configuration

`config/default.yaml` holds every setting; `--config <file>` deep-merges an
override on top. The configs used in the report:

| File | Purpose |
|---|---|
| `config/full-run-strict.yaml` | **Main run**: strict judge, LLM grading, 10 generations, `results/strict/` |
| `config/full-run.yaml` | Lenient comparison run: lenient judge, string matching, 20 generations |
| `config/three-model*.yaml` | Same models on Groq (free tier: smoke tests only) |

Key settings:

- `evolution.min_native_accuracy` (0.8): attack a question only if the target
  answers it correctly in >= 4 of 5 native samples.
- `evolution.confirm_samples` (10): a would-be flip is re-scored on fresh
  samples and counts only if still majority-wrong.
- `equivalence.llm_judge.strict` (true in the main run): strict judge prompt
  that rejects answer-type, scope, criterion and premise changes.
- `answer_extraction.method: grade`: an LLM grades each response against the
  expected answer to the **original** question (instead of string matching).
- Rephrasings containing invisible characters (zero-width spaces etc.) are
  always rejected before any model call.

### Budget cap

The `budget` section is a hard client-side ceiling, checked before every
request: `max_requests` and `max_usd` per run, and `max_usd_total` across all
runs, persisted in `<output_dir>/spend_ledger.json` (safe for concurrent
processes). Cost is the provider-reported cost when available (DeepInfra,
OpenRouter), otherwise estimated from token counts and the `prices` table; a
run refuses to start if a configured model has no price. When a cap is hit the
run stops and keeps everything saved so far (`--resume` continues).

## 6. Outputs

| File | Contents |
|---|---|
| `baselines.json` | Per question: native accuracy and answers, naive paraphrases and their error rates |
| `attack_results.json` | Per attacked question: every accepted candidate with responses and scores, the re-test of would-be winners, rejected candidates with the judge's reason; plus skipped and failed questions |
| `report.md` | Success rates with 95% Wilson CIs, noise floor, search statistics |
| `review.csv` | One row per flip for manual review |
| `noise_floor_confirmed.json` | Naive paraphrases put through the strict judge and re-test |
| `flip_audit.json` | Lenient-run flips re-checked with the strict judge and grader |
| `spend_ledger.json` | Cumulative spend and request count across runs |

## 7. Code layout

```
src/models/       LLM clients (deepinfra/openai/openrouter via openai_compat, gemini, groq),
                  rate limiting, retries, budget cap
src/data/         TriviaQA loading and filtering, hand-built facts (recency_facts.json)
src/equivalence/  LLM judge (lenient and strict), hidden-character filter, validator,
                  judge calibration helpers
src/evaluation/   answer extraction and LLM grading, wrongness scorer, baselines
src/attack/       paraphraser (framing strategies), evolutionary search, pipeline
src/analysis/     success rates, Wilson CIs, noise floor, review sheet
scripts/          run_baselines, run_attack, confirm_noise_floor, analyze,
                  audit_flips, calibrate_validator
tests/            offline unit tests (no API calls)
paper/            report source (final_report.tex)
```

## 8. Troubleshooting

- `CERTIFICATE_VERIFY_FAILED`: install `truststore` (see Installation); the
  clients then use the system certificate store.
- `402 Payment Required` from DeepInfra: the account needs prepaid balance.
- Gemini 429 with `FreeTier` in the message: billing is not enabled on the
  key's project.
- Rate limits and transient errors are retried automatically; permanent
  client errors (400/401/402/403/404/422) are not.
