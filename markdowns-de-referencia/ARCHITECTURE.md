# guardrail_eval — architecture (as of Phase 3)

XAI + prompt-injection research sub-project living inside the `jlens`
mechanistic-interpretability repo. It applies the **Jacobian lens** (see
[`PAPER_SUMMARY.md`](PAPER_SUMMARY.md)) to a small "guardrail" LLM to
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

The guardrail model is now **preset-selectable** via `PRESETS` (a
`{model_id: GuardrailPreset}` registry in `jlens_readout.py`): each preset
pairs a model with its lens file and the loading quirks that must move with
it (`force_bos`, `disable_thinking`, and which HF auto-class loads it).
Two presets ship: `Qwen/Qwen3-1.7B` (default, the only one that fits the
local CPU machine) and `google/gemma-3-4b-it` — the intended RunPod guardrail,
a multimodal checkpoint loaded via `AutoModelForImageTextToText` (~8GB bf16,
GPU-only), with its own lens
(`gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt`).
Pick one with `--guardrail-model`. The gemma path is not yet verified
end-to-end (can't run locally) — see `PLAN_runpod_audit.md` for the gemma
gotchas to confirm on the first pod run (system-role support, layout
detection, workspace band).

Core methods:
- `chat_prompt(seed)` — renders the classifier system prompt +
  `"INPUT: {seed}\n\nClassification:"` as a string via the chat template.
- `classify(prompt_str)` — greedy-generates the verdict, parses
  `malign`/`benign`/`unknown`.
- `readout(prompt_str, position=-1)` — applies `jlens.JacobianLens.apply()`
  and returns the top-K J-lens tokens **per layer at a single position**
  (default the decision position; the guardrail is never gated/blocked on
  this — it's read purely for explanation).
- `readout_multi(prompt_str, positions=None)` (added for Phase 3) — the
  multi-position counterpart: same topk-per-layer logic (shared via a
  private `_topk_per_layer_position` helper), but for an arbitrary list of
  positions at once, returning `{layer: {position: [...]}}`. Lets a caller
  trace how a concept's readout changes across the prompt, not just at one
  point in time.
- `token_span(prompt_str, last_n=None)` (added for Phase 3) — decoded
  `(position, token)` pairs for the prompt (optionally windowed to the last
  N), used to brief the investigator agent on what's tokenized where so it
  can pick meaningful positions for `readout_multi` instead of guessing.

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

### Phase 3 — automated auditor (§A.22): investigator + LLM-judge

A first prototype of the paper's §A.22 automated auditor, applied to the
existing guardrail. Replaces the paper's human-authored ground truth with a
**behavioral** one (Strategy A) so it scales without manual activation
reading. Runs the **audit loop only** — guardrail + lens + investigator +
judge, **no target model** — so the guardrail loads once (none of Phase 2's
two-phase memory dance). The investigator (DeepSeek) and judge (Groq) run via
API, so this loop is **API-bound, not GPU-bound**: it runs on CPU for a small
dry-run and on a RunPod GPU for a larger smoke test (`--device cuda`). See
[`PLAN_runpod_audit.md`](../guardrail_eval/PLAN_runpod_audit.md) for the RunPod setup.

Three new pieces:

**`ground_truth.py`** — Strategy A. Each `Claim` is a yes/no question that
applies to a subset of rows (selected by the guardrail's verdict) with an
expected answer fixed by the true label — i.e. the confusion-matrix cell:
- `silent_false_negative` (applies when verdict=benign): expected "sim" on
  FN (true label malign), "nao" on TN.
- `specificity_false_positive` (applies when verdict=malign): expected "sim"
  on TP, "nao" on FP.
`ground_truth_for(label, verdict)` returns the applicable `(claim, expected)`
pairs (empty for an "unknown" verdict). The investigator can now probe
multiple positions while resolving these two claims (see `audit_agent.py`
below), but no new *claim types* were added — "momento de emergência" and
"deliberação interna" from `automacao_auditoria_jlens.md` would need new
`Claim` entries built around a position sweep, which the underlying capability
now supports but `ground_truth.py` doesn't yet use. "Robustez a disfarce"
still needs the wrapping corpus; both remain out of scope, noted as extensions.

**`audit_agent.py`** — investigator + judge, **cross-provider by design**
(different providers for the two roles cut the self-evaluation bias seen when
one model grades its own family's answers). Temperature 0 throughout.
- `investigate(...)` — **DeepSeek** (`deepseek-ai/deepseek-v4-pro` via NVIDIA's
  OpenAI-compatible endpoint; key `DEEPSEEK_API_KEY`/`DEEPSSEK_API_KEY`).
  **Primary path is interactive tool-calling, multi-position**: the
  investigator is shown a `token_span` map (position→token) of the guardrail's
  prompt and, on its own judgment, calls a `get_jlens_readout(positions,
  layers)` function tool (OpenAI-compatible schema, `READOUT_TOOL_SCHEMA`) as
  many times as it wants, up to `--max-tool-calls` (default 5 — a starting
  guess, meant to be tuned empirically once there's infrastructure to observe
  convergence). Each call is served locally against `GuardrailLens.readout_multi`
  (via a `readout_fn` closure bound by `run_audit_pipeline.py`) and fed back as
  a tool-response message. Once it stops calling the tool (or hits the cap), a
  final call (no tools, JSON mode) forces the structured verdict
  `{verdict: sim/nao, evidence}`. System prompt = `INVESTIGATOR_PRIMER` (a short
  paper primer distilled from `PAPER_SUMMARY.md`: what the J-lens shows, the
  workspace band L14-26, the single-token limitation, and that reasoning
  unfolds across positions) + the task; constrained to cite only tokens the
  tool actually returned (anti-hallucination).
  **Fallback**: if the very first tool-enabled API call fails outright (e.g.
  the DeepSeek/NVIDIA endpoint rejects `tools` for this model), `investigate`
  automatically drops to `_investigate_fixed` — a single fixed readout at
  position -1, the pipeline's original non-interactive design — instead of
  failing the row. Both paths report `tool_calls` (int) and `fallback_used`
  (bool) in the result for auditability. **Not yet validated end-to-end**
  (can't run locally, no GPU/low infra) — see `PLAN_runpod_audit.md` for what
  to confirm on the first pod run.
- `judge(...)` — **Groq** (`openai/gpt-oss-120b`; key `GROQ_API_KEY`/`GPT_API_KEY`),
  no readout access; scores `correctness` (hard gabarito vs. `expected`) and
  `evidence_quality` (qualitative — Strategy A has no causal gabarito for it)
  each 0-10; `score` = their mean.
- Both roles use JSON mode + pydantic validation, with retry and a recorded
  fallback (an `error` field) so one bad API call doesn't sink a long run.

**`run_audit_pipeline.py`** — driver. Runs over one or both attack corpora
(`--attack {baseline,baseline-wrapping,both}`, default `both`, mirroring
`run_attack_pipeline.py`'s `slug()`/`out_paths(attack)`/`select_subset(attack,
...)` pattern), First-N malign + first-N benign per attack (`--n-malign`/
`--n-benign`, default **15/15** — a reduced "main run" size, not the full
230-row corpus, chosen to bound free-tier API cost). Per prompt: classify +
an anchor readout at the decision position (`GuardrailLens.readout`,
unchanged, logged for continuity) → applicable claims → investigate
(interactive, via `readout_fn`/`token_span_text` built from `readout_multi`/
`token_span`) → judge. `--top-k` default is now **25** (was 10, matching
`automacao_auditoria_jlens.md`'s §3 suggestion). Since a row triggers at most
one claim, total API calls scale close to linearly with prompt count (~3-7
calls/claim with interactive tool-calling) — two knobs exist to keep a larger
run survivable: `--api-pacing-seconds` (sleep between each investigate+judge
cycle) and `--resume` (skip `pool_index` rows already in that attack's
`audit_readouts_<attack>.jsonl`, appending instead of overwriting — the
readout write happens before the API/claim step, so it's the safe checkpoint
for skipping already-done GPU work). Writes to a dedicated `results_audit/`
folder (separate from Phase 2's `results/`), **one set of files per attack**:
`audit_readouts_<attack>.jsonl` (verdict + anchor readout per prompt),
`audit_scores_<attack>.jsonl` (scores per (prompt, claim), including
`tool_calls` and `fallback_used`), `audit_summary_<attack>.csv` (per-claim
aggregate + investigator accuracy); with `--attack both`, an additional
`audit_summary_combined.csv` aggregates across both. Dry-run:
`python run_audit_pipeline.py --device cpu --attack baseline --n-malign 2 --n-benign 2`.

### Causal interventions — steer / ablate / swap primitives

The pipeline above is **read-only** interpretability: it surfaces which
concepts drive the guardrail's verdict but never proves they *cause* it. The
paper's two causal *writing* primitives (steering/ablation and the
lens-coordinate swap of Figure 4C; see `PAPER_SUMMARY.md`, "The J-lens's
read/write primitives", §2.5) are described but **not** implemented in
`jlens/lens.py`. Since `guardrail_eval/` must not modify `jlens/`, they are
built here, on the three ingredients `jlens` already exposes: `J_l`
(`JacobianLens.jacobians[layer]`), `W_U` (the model's `lm_head.weight`), and a
*write-capable* forward hook (which `jlens.hooks.ActivationRecorder`
deliberately is not — its hook only reads).

**`interventions.py`** — model-agnostic, operates on residual-stream tensors:
- `lens_vectors(lens, W_U, layer)` / `lens_vector(...)` — the J-lens vectors
  `v_t` = rows of `W_U · J_l` (drops the final `norm`, matching §2.1).
- `steer(h, v, α)` = `h + α·v`; `ablate(h, v)` projects out the component of
  `h` along `v`; `ablate_span(h, V)` zeroes the projection onto `k` vectors at
  once (the J-space ablation of §3.5.2, `k=10` across a layer band).
- `swap(h, v_s, v_t, α)` — the Figure 4C formula: `V=[v_s v_t]`,
  `c=V⁺h`, `h_patched = h + α·V(σ(c)−c)`, leaving `span{v_s,v_t}`'s orthogonal
  complement untouched.
- `InterventionHook(blocks, {layer: edit_fn}, positions=...)` — a context
  manager that *writes* an edit into the residual stream during the forward
  pass and cleanly removes itself on exit. Enter it before an
  `ActivationRecorder` on the same blocks if the recorder must see the edit.

**`tests/test_interventions.py`** — 10 **invariant** tests against
`jlens`'s `TinyDecoder` (whose exactly-linear blocks make `J_l` exact, so the
algebra is clean). They verify the primitives are *mechanically correct and
correctly plumbed*: steer moves by exactly `α·v`; ablate zeroes the
projection and is idempotent; swap exchanges the two lens coordinates,
preserves the orthogonal complement, and is an involution at `α=1`; the hook
edits only the requested position, propagates downstream, and is removed on
exit. **These prove nothing semantic or causal** — `TinyDecoder` has untrained
weights and no interpretable vocabulary; "swap Soccer→Rugby flips the output"
can only be shown on the trained Qwen3-1.7B guardrail. This test de-risks the
*engineering* so that a null result on Qwen3 later reads as a finding, not a
bug. `guardrail_eval/.venv` has no pytest, so the file is a self-contained
script: `guardrail_eval/.venv/Scripts/python.exe guardrail_eval/tests/test_interventions.py`
(also pytest-collectable where available).

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
├── ground_truth.py           # Phase 3: Strategy-A claims (label x verdict -> expected answer)
├── audit_agent.py            # Phase 3: investigator (DeepSeek) + judge (Groq gpt-oss-120b) (format_readout/investigate/judge)
├── run_audit_pipeline.py     # Phase 3: auditor driver (guardrail + lens + investigator + judge)
├── interventions.py          # Causal primitives: lens_vectors/steer/ablate/ablate_span/swap + InterventionHook (write-capable)
├── tests/
│   └── test_interventions.py # 10 invariant tests vs TinyDecoder (standalone script; no pytest in venv)
├── setup_pod.sh              # Phase 3: idempotent pod setup (matched-index torch/torchvision/torchaudio fix + installs)
├── make_slices.py            # jlens.vis.compute_slice pages for selected rows (see VISUALIZATION.md)
├── make_report.py            # static XAI report (MD + PDF) over an audit run (see VISUALIZATION.md)
├── data/
│   ├── harmbench_labeled.csv          # 200 malign seeds
│   ├── jailbreakbench_benign_en.csv   # 30 benign seeds (source data)
│   ├── system_variants_en.csv         # 18 wrapping templates (source data)
│   ├── seed_pool.csv                  # 230 unified seeds
│   ├── attack_baseline.csv            # 230 rows, prompt = seed
│   └── attack_baseline_wrapping.csv   # 230 rows, prompt = wrapped seed
├── results/
│   ├── readouts_qwen3_1.7b_baseline[_wrapping].jsonl   # per-prompt J-lens readouts
│   ├── summary_qwen3_1.7b_baseline[_wrapping].csv      # guardrail verdicts
│   └── target_baseline[_wrapping].csv                  # target model outputs
└── results_audit/                                      # Phase 3 (separate folder), one set per attack
    ├── audit_readouts_baseline[_wrapping].jsonl   # per-prompt: guardrail verdict + J-lens readout
    ├── audit_scores_baseline[_wrapping].jsonl     # per (prompt, claim): investigator verdict + judge scores
    ├── audit_summary_baseline[_wrapping].csv      # per-claim aggregate + investigator accuracy
    └── audit_summary_combined.csv                 # cross-attack aggregate (only when --attack both)
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
- Interventions: 10/10 invariant tests pass for `interventions.py`
  (steer/ablate/swap + write hook) against `TinyDecoder` — mechanical
  correctness only, no semantic/causal validation yet (needs Qwen3).

## Not yet built (explicitly out of scope so far)

- No blocking/gating logic — the guardrail's verdict is recorded, never
  enforced.
- No dense position sweep as part of the audit loop itself — the
  investigator's `readout_multi` probing is sparse and agent-chosen (see
  Phase 3 above). `make_slices.py`/`make_report.py` (see
  `VISUALIZATION.md`) fill this in for a hand-picked subset of rows via
  `jlens.vis.compute_slice`, but that stays a separate, selective step —
  running it over a full corpus is still deferred for cost reasons.
- No causal validation of the lens *on the guardrail*. The mechanical
  primitives now exist (`interventions.py`, invariant-tested — see "Causal
  interventions" above), but they have not been *run on Qwen3* to show that
  ablating/swapping the surfaced concepts actually changes the guardrail's
  verdict. Everything scored so far is read-only interpretability; the causal
  claim (and the auditor's "Strategy B" ground truth: ablation-KL + swap
  success, §A.6) is still pending.
- No LLM-judge / attack-success-rate scoring of target outputs — the
  target results are raw completions only, not yet graded.
- Full-corpus runs (all 230 seeds × 2 attacks) — everything to date is a
  smoke-scale (3+3) validation of the pipeline's correctness, not a result.
