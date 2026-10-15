# Handoff for the next Claude Code session

Read this file first. It is the complete state of the project as of 2026-10-14.
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
    `openai/gpt-oss-20b` on Groq (reasoning_effort low).
  - Auxiliary (paraphraser + answer extraction): `openai/gpt-oss-120b` on Groq.
  (Groq retired llama-3.1-8b-instant and llama-3.3-70b-versatile by 2026-10.)
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

## Status 2026-10-15: final report drafted (paper/final_report.tex)

Main run (results/strict/, config/full-run-strict.yaml, --shuffle-seed 42
--max-attacked 120, 10 generations): 120 attacked, 26 confirmed flips
(21.7%), 19 after review (15.8%, CI 10.4-23.4). Confirmed noise floor 5.4%
(scripts/confirm_noise_floor.py). Lenient run (results/, paused at 90
attacked) audited with scripts/audit_flips.py: 15 of 36 flips survive.
Review labels: LLM-proposed (results/strict/review_proposed.csv), approved
by the author -> results/strict/review.csv. Total API spend ~$8.65.

Not done (future work in the paper): transferability, equal-budget
best-of-N baseline, ablations, defenses, ~200-pair judge calibration.
Hidden-character filter and parallel grading postdate the main run's start
(filter applied in analysis).

## Session 2026-10-14: what changed

- Groq models swapped to gpt-oss-20b / gpt-oss-120b; new `reasoning_effort`
  field in model config (passed to Groq). `config/three-model-smoke.yaml` =
  three-model + small search, for live checks.
- Groq client now uses truststore (corporate TLS proxy on this machine).
- BUG FIXED: under `abstention_policy: exclude` a question the model always
  refuses has wrongness 0.0, so it was treated as "known" and attacked, and
  any wrong guess counted as a flip. Attackability now uses
  `ScoreResult.knows_answer` (correct in >50% of ALL samples), in both the
  attack skip and the baseline "attackable" count.
- TriviaQA: loads the 7 MB `rc.nocontext` parquet instead of multi-GB `rc`.
  Filter drops numeric/date and list answers (~6%) and strips unrelated
  Wikipedia-redirect aliases (avg 15.6 -> 4.6 aliases).
- Naive paraphrases: one prompt for N numbered rewrites + near-dup removal.
- Recency set: 50 items. Post-cutoff items tagged `after_target_cutoff`
  and sorted last; time-sensitive ones carry `verified_as_of` (UN SG item
  has `valid_until: 2026-12-31`).
- Attack results now save `rejected_candidates` (with judge reason).
  `scripts/calibrate_validator.py sample|score` builds the blind human
  labeling CSV and reports judge precision / population-weighted recall.
- Smoke attack, 10 custom questions: 7 attacked, 2 "flips", both noise
  (borderline/post-cutoff questions, mild rewrites); 5 timeless questions
  best score 0.00 after 3 generations. 986 requests, ~$0.05.

## Ready for the Phase 1 full run (2026-10-14, evening)

- Spend cap hardened and tested live: checks before every request; unknown
  model prices refuse to start; missing usage charged worst case; Gemini
  thinking tokens counted; failed attempts count toward the request cap;
  `max_usd_total` caps ALL runs via `results/spend_ledger.json`.
  Caps: $30 per run, $30 total.
- Attack gate 4-of-5 (`evolution.min_native_accuracy: 0.8`); winning
  rephrasings re-tested on 10 fresh samples (`evolution.confirm_samples`).
- `run_attack.py --resume --workers N`, `run_baselines.py --workers N`;
  skipped questions logged in attack_results.json; baselines checkpoint
  after every question and save native accuracy + answers.
- `config/full-run.yaml`: 50 custom + 300 TriviaQA, paid-tier pacing.
- Groq Developer upgrades were "temporarily unavailable" (2026-10-14), and
  the corporate proxy blocks openrouter.ai. full-run.yaml therefore serves
  target + auxiliary from DeepInfra (same gpt-oss models, cheaper, uses
  DeepInfra's reported per-call cost). Full run est. ~$8.
- BLOCKED ON: DEEPINFRA_API_KEY in .env and Gemini billing enabled. Then:
    python scripts/run_baselines.py --config config/full-run.yaml --workers 8
    python scripts/run_attack.py    --config config/full-run.yaml --workers 8

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
   - DONE 2026-10-14: TriviaQA filter, recency set (50), naive-paraphrase
     dedup, calibration tooling.
   - Validator calibration still needs the HUMAN labels: after a run with
     rejected-candidate logging, `calibrate_validator.py sample --n 200`,
     label `results/calibration/pairs.csv`, then `score`.
   - Native accuracy with 3 samples is noisy: borderline questions pass the
     attackability gate and then "flip" on trivial rewrites. Consider
     num_samples 5+ for the gate, or requiring native accuracy >= 0.8.
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
