# Framing Attack

Black-box attack on LLM factual question answering via semantically-equivalent
paraphrasing. Implements the plan in *Framing the Truth: Flipping Black-Box LLM
Answers via Semantically-Equivalent Paraphrasing* (Noam Cohen, TAU).

An auxiliary LLM proposes rephrasings of a factual question, a validator
(local NLI cross-encoder + LLM judge) rejects any that change the meaning,
and an evolutionary search keeps the rephrasings that push the target model
toward a wrong answer.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "httpx[socks]" truststore   # corporate TLS / proxies
cp .env.example .env                                # then fill in a key
.venv/bin/python -m pytest                          # offline unit tests
```

The NLI model (`cross-encoder/nli-deberta-v3-base`, ~700 MB) downloads on first
use and runs locally (Apple GPU via MPS, CUDA, or CPU).

## Providers and keys

| Provider    | `.env` variable      | Notes |
|-------------|----------------------|-------|
| `gemini`    | `GEMINI_API_KEY`     | AI Studio key (`AIza...` or newer `AQ....`). Free tier: 15 requests/min per model. |
| `groq`      | `GROQ_API_KEY`       | Open models (Llama etc.). |
| `openai`    | `OPENAI_API_KEY`     | Set a hard monthly budget in Settings > Limits. |
| `openrouter`| `OPENROUTER_API_KEY` | Prepaid credits act as a hard cap; reports cost per call. |

## Running

```bash
# Phase 0: native error rate + naive-paraphrase variance
.venv/bin/python scripts/run_baselines.py --dataset custom --max-samples 3

# Phase 1: evolutionary attack
.venv/bin/python scripts/run_attack.py --dataset custom --max-samples 3

# Small search budget (finishes in minutes on a free tier)
.venv/bin/python scripts/run_attack.py --config config/smoke.yaml --dataset custom --max-samples 3

# Paid OpenAI, parallel requests, no pacing
.venv/bin/python scripts/run_attack.py --config config/openai.yaml --dataset custom --max-samples 3
```

Results land in `results/baselines.json` and `results/attack_results.json`.

## Configuration

`config/default.yaml` holds all settings; `--config <file>` deep-merges an
override on top. Per model you can set `requests_per_minute` (client-side
pacing shared across every client of the same model; `null` disables) and
`concurrency` (parallel requests when sampling).

The `budget` section is a hard per-run ceiling (`max_requests`, `max_usd`).
Cost is taken from the API when reported (OpenRouter) or estimated from the
`prices` table. When a cap is hit the run stops and saves partial results.

## Cost guide (gpt-4.1-mini, ~250 tokens per call)

| Run | Calls | Approx. cost |
|---|---|---|
| Smoke tests, 3 questions | ~500 | < $0.10 |
| 50 questions, 10 generations | ~100k | ~$15 |
| Default config, 200 questions, worst case | ~400k | ~$60 |

## Changes from the original handoff

- `pyproject.toml`: build backend fixed to `setuptools.build_meta`.
- Default model `gemini-2.0-flash` (shut down June 2026) replaced by
  `gemini-3.5-flash-lite` (fast, no thinking tokens). Newer Flash models
  think by default and can exhaust `max_tokens` before answering.
- `src/models/base.py`: shared per-model rate limiter; 429 handling that
  honours the server's retry delay; optional concurrency; budget hooks.
- New `src/models/budget.py` (per-run spend/request cap) and
  `src/models/openai_compat.py` (OpenAI / OpenRouter client).
- New `config/smoke.yaml` and `config/openai.yaml` overrides.

## Layout

```
src/models/       LLM clients (gemini, groq, openai_compat), factory, budget
src/data/         QA dataset loading, hand-built recency facts
src/equivalence/  NLI validator, LLM judge, combined validator
src/evaluation/   answer extraction, wrongness scorer, baselines
src/attack/       paraphraser, evolutionary search, pipeline
scripts/          run_baselines.py, run_attack.py
tests/            offline unit tests (no API calls)
```
