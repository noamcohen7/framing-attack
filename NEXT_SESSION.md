# Handoff for the next Claude Code session

Read this file first. It is the complete state of the project as of 2026-10-12.
The research plan is in `docs/intermediate_report.pdf` (Noam Cohen, TAU):
a black-box attack that finds meaning-preserving rephrasings of factual
questions that make an LLM answer wrongly. Four phases: 0 setup/baselines,
1 attack loop, 2 evaluation/analysis, 3 defenses/write-up.

## Constraints for this session

- **Low-resource machine: run nothing locally.** Install WITHOUT the `nli`
  extra and keep `equivalence.nli_model: null` (judge-only mode). Do not
  install torch.
- **Three different models, one per role**, all via APIs:
  - Target (attacked): a weak open model so flips are plentiful ->
    `llama-3.1-8b-instant` on Groq.
  - Auxiliary (paraphraser + answer extraction): `llama-3.3-70b-versatile` on Groq.
  - Judge (equivalence guard): `gemini-3.5-flash-lite` on Gemini (independent vendor).
  This is `config/three-model.yaml`. Verify model names with each provider's
  list-models call before the first run; names rotate.
- **Hard spend cap.** The in-code `budget` section stops a run at
  `max_requests` / `max_usd`. Keep it on. Provider-side: OpenAI has a hard
  monthly budget in Settings > Limits; OpenRouter is prepaid; Groq and
  Google have no hard cap, so rely on the in-code cap there.

## Setup (5 minutes)

```bash
git clone <REPO_URL> framing-attack && cd framing-attack
python3.12 -m venv .venv            # 3.10+ works
.venv/bin/pip install -e ".[dev]"   # NOT ".[nli]" on this machine
cp .env.example .env                # add GROQ_API_KEY and GEMINI_API_KEY
.venv/bin/python -m pytest          # 70 tests, all offline, ~1s
```

Keys: Groq from https://console.groq.com/keys (starts `gsk_`); Gemini from
https://aistudio.google.com/apikey (starts `AIza` or `AQ.`). Keys live only
in `.env`, which is git-ignored.

## What is verified working

- Full unit suite (70 tests) offline.
- Gemini client live; Groq client code present but NOT yet exercised live
  (no Groq key was available). First thing: run the smoke commands below.
- Phase 0 baseline run on 3 custom questions (Gemini target): native and
  naive-paraphrase error rate 0.00. `results/` is git-ignored; rerun it.
- Phase 1 attack loop end to end on 1 question with the small search
  budget: 19 candidates, 11 passed the validator, 3 generations, results
  checkpointed to `results/attack_results.json` after every question.
- Judge-only equivalence mode live against Gemini.

## Smoke commands (run first, ~10-20 min on free tiers)

```bash
.venv/bin/python scripts/run_baselines.py --config config/three-model.yaml --dataset custom --max-samples 3
.venv/bin/python scripts/run_attack.py    --config config/three-model.yaml --dataset custom --max-samples 3
```

If the Groq free tier's daily token cap bites (the 70B model has a small
one), either enable Groq's paid tier or move the auxiliary to OpenAI
(`provider: openai`, `model_name: gpt-4.1-mini`, see `config/openai.yaml`).

## Remaining work, in order

1. **Phase 0 gaps**
   - `src/data/loader.py::_load_triviaqa`: the "single canonical answer"
     filter is a stub (`pass`). Implement it (e.g. keep items whose aliases
     are surface variants of the canonical answer; drop numeric/date
     ambiguity) and add a test with a few hand-written rows.
   - `src/data/recency_facts.json`: 20 items. Expand to 40-50 and re-verify
     every time-sensitive one as of today (UN Secretary-General, largest
     population, tallest building, "current" anything).
   - Naive-paraphrase baseline produces near-duplicates at temperature 0.9.
     Add dedup and ask for N distinct rewrites in one prompt.
   - Validator calibration: sample ~200 (original, paraphrase) pairs from a
     run, have a human label them, report judge precision/recall. The
     paper promises this.
2. **Phase 1 full run**
   - Pick budget: 100-200 TriviaQA questions + the recency set, default
     search config (`population_size 10`, `num_generations 20`). Raise
     `budget.max_requests` / `max_usd` deliberately for this run.
   - Add the greedy / best-of-N ablation (same candidate budget, no
     evolution) as a flag in `src/attack/evolutionary.py` or a sibling class.
3. **Phase 2 analysis** (none exists yet)
   - `scripts/analyze.py`: attack success rate per model/dataset, with
     success = best wrongness score > 0.5 over human-verified valid
     rephrasings; abstention rate reported separately.
   - Transferability: script that re-scores the best rephrasings found
     against model A on model B (swap `target_model` in config, read
     `results/attack_results.json`).
   - Ablations: NLI on/off (needs a machine with torch), judge model,
     auxiliary model, search strategy.
   - Taxonomy: cluster successful rephrasings (presupposition, leading
     phrasing, false dichotomy, anchoring, politeness/indirect question).
     Note: indirect phrasings ("Can you tell me...") caused *refusals*
     on Gemini, a framing effect on abstention worth its own paragraph.
4. **Phase 3 defenses**: self-consistency over paraphrases and answer
   aggregation, measured as reduction in attack success. Then write-up.

## Gotchas learned the hard way

- Newer Gemini Flash models think by default and can spend the whole
  `max_tokens` on thinking, returning empty text. `gemini-3.5-flash-lite`
  does not think; keep it, or raise `max_tokens` for thinking models.
- Free tiers: Gemini 15 req/min per model; Groq ~30 req/min and daily
  token caps. `requests_per_minute` in config paces the client; 429s wait
  for the server's retry-after. Network blips retry for ~2 minutes.
- Refusals ("I cannot answer, my cutoff is...") are NOT wrong answers.
  `answer_extraction.abstention_policy: exclude` keeps them out of the
  wrongness score; they are counted in `abstention_rate`.
- A run that dies keeps everything already saved in `results/`; the
  attack script checkpoints after each question. There is no resume flag
  yet; add one if runs get long (skip ids already in the results file).

## Code map

```
src/models/       base.py (retry, pacing, budget hooks, concurrency), gemini.py,
                  groq_client.py, openai_compat.py (OpenAI/OpenRouter), budget.py
src/data/         loader.py (TriviaQA + custom facts), recency_facts.json
src/equivalence/  nli.py (optional), llm_judge.py, validator.py
src/evaluation/   answer_extract.py, scorer.py (abstention policy), baselines.py
src/attack/       paraphraser.py (prompts + framing strategies), evolutionary.py, pipeline.py
scripts/          run_baselines.py, run_attack.py
config/           default.yaml, smoke.yaml, openai.yaml, three-model.yaml
```
