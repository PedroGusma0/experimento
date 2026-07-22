# guardrail_eval — architecture (as of Phase 2)

XAI + prompt-injection research sub-project living inside the `jlens`
mechanistic-interpretability repo. It applies the **Jacobian lens** (see
[`../PAPER_SUMMARY.md`](../PAPER_SUMMARY.md)) to a small "guardrail" LLM to
explain *what internal features drive its malign/benign verdict* on incoming
prompts — including prompts wrapped in jailbreak/injection templates — and
observes what happens downstream when a target LLM receives those same
prompts. `guardrail_eval/` only ever *imports* `jlens` as a library
(`pip install -e ..`); nothing under `jlens/` is modified.

## Environment constraints that shaped the design

- **CPU-only, ~7.7 GB total RAM.** No CUDA GPU on this machine. This is the
  single biggest constraint on the pipeline: it forced the two-phase
  sequential model loading described below, and keeps every run scoped to
  small "smoke test" sample sizes rather than full corpora.
- Own venv at `guardrail_eval/.venv` (torch CPU, transformers, pandas, tqdm,
  `jlens` installed editable from the repo root).
- All models load from local HuggingFace cache
  (`~/.cache/huggingface/hub`) — `google/gemma-3-1b-it`, `Qwen/Qwen3-1.7B`,
  and the pre-fitted `neuronpedia/jacobian-lens` lens for Qwen3-1.7B.

## The three phases so far

### Phase 0 — gemma-3-1b-it as a naive baseline classifier

`prepare_data.py` reads the repo-root `harmbench.csv` (200 harmful-behavior
seeds, columns `prompt,category`) and writes
`data/harmbench_labeled.csv` with `label="malign"` added to every row
(never mutates the root file). `run_baseline.py` loads `gemma-3-1b-it` via
plain `transformers` and classifies each seed as `malign`/`benign` using a
strict-classifier system prompt + greedy 8-token generation. This
established the reusable gemma-loading pattern and the `SYSTEM_PROMPT` /
`parse_label` logic that later phases build on.

### Phase 1 — swap the guardrail to Qwen3-1.7B, apply the J-lens

`jlens_readout.py` defines `GuardrailLens`, the reusable framework class:
loads `Qwen/Qwen3-1.7B` once via `jlens.from_hf(...)` (wrapping it as a
`LensModel`) plus its pre-fitted Jacobian lens
(`neuronpedia/jacobian-lens`, file
`qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`, 27
fitted layers, d_model=2048). Two Qwen3-specific gotchas are baked in:
`enable_thinking=False` on the chat template (otherwise a `<think>` block
eats the token budget before any verdict) and `force_bos=False` (Qwen3's
tokenizer already sets `add_bos_token=false`).

Three methods do the work:
- `chat_prompt(seed)` — renders the classifier system prompt +
  `"INPUT: {seed}\n\nClassification:"` as a string via the chat template.
- `classify(prompt_str)` — greedy-generates the verdict, parses
  `malign`/`benign`/`unknown`.
- `readout(prompt_str, position=-1)` — applies `jlens.JacobianLens.apply()`
  and returns the top-K J-lens tokens **per layer at the decision
  position** (the guardrail is never gated/blocked on this — it's read
  purely for explanation).

`run_guardrail_jlens.py` drives this over `data/harmbench_labeled.csv`.
Validated in a 5-prompt smoke test: 5/5 correct classifications, and the
J-lens readout showed the guardrail's internal harm-recognition surfacing
in the workspace-band layers (~L16–24: `illegal`, `INVALID`, `违法`,
`harmful`, `malignant`) *before* the final `malign` verdict — i.e. the lens
exposes the "why" behind the classification, not just the classification
itself.

### Phase 2 — target model + attack taxonomy

Three new pieces:

**`prepare_attack_data.py`** builds the corpora, fail-loud validated like
`prepare_data.py`:
- `data/seed_pool.csv` — unifies 200 malign seeds (`harmbench_labeled.csv`)
  with 30 benign seeds (`jailbreakbench_benign_en.csv`, topically-sensitive
  but non-harmful framings — a deliberate false-positive stress test, not
  "easy" benign examples) into 230 rows with a global `pool_index`.
- `data/attack_baseline.csv` — `prompt = seed`, no transform.
- `data/attack_baseline_wrapping.csv` — each seed inserted into one of 18
  jailbreak/narrative-priming templates (`data/system_variants_en.csv`,
  replacing the literal placeholder `[INSERT PROMPT HERE]`), **cycling
  deterministically** via `variant_index = pool_index % 18` (not a full
  cross-product). Variant metadata (technique/category/pattern) is kept
  per row for later slicing.

**`target_model.py`** defines `TargetModel`: plain `gemma-3-1b-it`
open-ended generation, **no J-lens involvement** (the lens is only ever
applied to the guardrail — the target is treated as an opaque
downstream system) and **no system prompt** (bare user turn, to simulate
an undefended generic assistant and avoid confounding "did the attack
work" with "did the target's own framing block it").

**`run_attack_pipeline.py`** orchestrates both: for a smoke subset (default
3 malign + 3 benign seeds per attack, first-N deterministic selection), it
runs guardrail-classify + guardrail-readout + target-generate for *every*
row — the guardrail's verdict is recorded but **never gates** the target
call, on purpose (this phase is pure observation, no defense wired in yet).

**Critical design point: two sequential phases, never both models loaded
at once.** An earlier single-pass version (guardrail + target both resident
in memory) reliably OOM'd on this machine (Qwen3-1.7B fp32 ≈ 6.8 GB +
gemma-3-1b-it fp32 ≈ 4 GB > 7.7 GB total RAM). `run_attack_pipeline.py` now
runs **Phase A** (load guardrail only, run classify+readout for every
attack in scope, write readouts/summary, then `del` + `gc.collect()`) fully
before **Phase B** (load target only, run generate, write target results).
Peak memory is therefore one model at a time, not two.

Default lens-read scope is also cost-reduced: `--layers` defaults to the
**workspace band (L14–L26)** rather than all 27 fitted layers, since L1–13
were confirmed pure formatting noise in the Phase 1 smoke test — this
roughly halves the per-prompt lens cost. `--target-max-new-tokens` defaults
to 60 (not 200) for the same reason: enough to see whether the target
starts complying or refuses, without paying for a full response on CPU.

## Data flow (Phase 2, current state)

```
harmbench.csv (root, 200)  ─┐
                             ├─► seed_pool.csv (230: 200 malign + 30 benign)
jailbreakbench_benign (30) ─┘         │
                                       ├─► attack_baseline.csv (prompt = seed)
system_variants_en.csv (18) ──────────┴─► attack_baseline_wrapping.csv
                                              (prompt = seed wrapped in variant,
                                               cycling pool_index % 18)

for each attack in {baseline, baseline-wrapping}, for a smoke subset:

  PHASE A (guardrail: Qwen3-1.7B + J-lens, loaded alone)
    prompt ─► GuardrailLens.chat_prompt() ─► classify() ─► label_pred (never gates)
                                          └─► readout() ─► top-K J-lens tokens/layer @ position -1
    writes: readouts_qwen3_1.7b_<attack>.jsonl, summary_qwen3_1.7b_<attack>.csv
  [guardrail freed from memory]

  PHASE B (target: gemma-3-1b-it, loaded alone)
    prompt ─► TargetModel.generate() ─► open-ended output   (same prompt, unconditionally)
    writes: target_<attack>.csv
```

## File map

```
guardrail_eval/
├── requirements.txt          # torch, transformers, pandas, tqdm (+ jlens via pip install -e ..)
├── prepare_data.py           # Phase 0: harmbench.csv -> harmbench_labeled.csv
├── run_baseline.py           # Phase 0: gemma-3-1b-it as classifier (SYSTEM_PROMPT, parse_label)
├── jlens_readout.py          # Phase 1: GuardrailLens (Qwen3-1.7B + J-lens; chat_prompt/classify/readout)
├── run_guardrail_jlens.py    # Phase 1: driver over harmbench_labeled.csv
├── prepare_attack_data.py    # Phase 2: seed_pool.csv + attack_baseline*.csv builders
├── target_model.py           # Phase 2: TargetModel (gemma-3-1b-it, open-ended, no lens, no system prompt)
├── run_attack_pipeline.py    # Phase 2: two-phase orchestrator (guardrail-only, then target-only)
├── data/
│   ├── harmbench_labeled.csv          # 200 malign seeds
│   ├── jailbreakbench_benign_en.csv   # 30 benign seeds (source data)
│   ├── system_variants_en.csv         # 18 wrapping templates (source data)
│   ├── seed_pool.csv                  # 230 unified seeds
│   ├── attack_baseline.csv            # 230 rows, prompt = seed
│   └── attack_baseline_wrapping.csv   # 230 rows, prompt = wrapped seed
└── results/
    ├── readouts_qwen3_1.7b_baseline[_wrapping].jsonl   # per-prompt J-lens readouts
    ├── summary_qwen3_1.7b_baseline[_wrapping].csv      # guardrail verdicts
    └── target_baseline[_wrapping].csv                  # target model outputs
```

## What's validated so far (smoke tests only — nothing run at full scale)

- Phase 1: 5/5 correct guardrail classifications on raw HarmBench seeds;
  J-lens readouts qualitatively interpretable (harm concepts precede the
  verdict in the workspace-band layers).
- Phase 2: 6+6 rows (3 malign + 3 benign × 2 attacks) ran end-to-end with no
  OOM, no blocking, consistent row counts across all output artifacts, and
  prompts forwarded to the target unmodified. Two findings worth noting
  going forward:
  - **`baseline`**: 3/6 correct (accuracy 0.5 on this tiny sample) —
    includes a false positive on a benign seed, expected given the benign
    pool is deliberately topically-sensitive.
  - **`baseline-wrapping`**: 2/6 correct, 3/6 `unknown` — the wrapping
    templates' own jailbreak-persona instructions ("ignore previous rules
    ... never apologize/refuse") compete with the guardrail's own
    classifier system prompt, sometimes causing the guardrail to drift
    into completing the persona's roleplay text instead of emitting a
    clean verdict. This is a real signal about wrapping attacks degrading
    guardrail output reliability, not a pipeline bug.

## Not yet built (explicitly out of scope so far)

- No blocking/gating logic — the guardrail's verdict is recorded, never
  enforced.
- No position sweep for the J-lens — reads only the single decision
  position (`-1`); localizing *which token* of the prompt triggered
  recognition (e.g. via `jlens.vis.compute_slice` or a position list) is a
  natural next step but was deferred for cost reasons.
- No causal validation of the lens (ablation / lens-coordinate swap) —
  everything so far is read-only interpretability, not yet a causal claim
  that the surfaced concepts *drive* the verdict.
- No LLM-judge / attack-success-rate scoring of target outputs — the
  target results are raw completions only, not yet graded.
- Full-corpus runs (all 230 seeds × 2 attacks) — everything to date is a
  smoke-scale (3+3) validation of the pipeline's correctness, not a result.
