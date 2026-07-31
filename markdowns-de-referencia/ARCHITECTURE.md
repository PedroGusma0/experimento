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

> **Folder note.** The causal work lives in a **separate root-level folder,
> `causal_eval/`**, a sibling of `guardrail_eval/` — not inside it. It was
> split out because the causal validation pipeline (v2, below) diverges
> substantially from `guardrail_eval/`'s phases 0–3 (no target model, no
> investigator/judge, a per-position ablation sweep instead of a read-only
> audit loop). `causal_eval/` follows the same rules: it only imports `jlens`
> as a library and never modifies `jlens/`. It reuses the shared
> `guardrail_eval/.venv` (there is no separate venv). `interventions.py` and
> its test were moved here from `guardrail_eval/` via `git mv` (history
> preserved).

The pipeline above is **read-only** interpretability: it surfaces which
concepts drive the guardrail's verdict but never proves they *cause* it. The
paper's two causal *writing* primitives (steering/ablation and the
lens-coordinate swap of Figure 4C; see `PAPER_SUMMARY.md`, "The J-lens's
read/write primitives", §2.5) are described but **not** implemented in
`jlens/lens.py`. Since neither `causal_eval/` nor `guardrail_eval/` may modify
`jlens/`, they are built here, on the three ingredients `jlens` already
exposes: `J_l` (`JacobianLens.jacobians[layer]`), `W_U` (the model's
`lm_head.weight`), and a *write-capable* forward hook (which
`jlens.hooks.ActivationRecorder` deliberately is not — its hook only reads).

**`causal_eval/interventions.py`** — model-agnostic, operates on
residual-stream tensors:
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

**`causal_eval/tests/test_interventions.py`** — 10 **invariant** tests against
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
bug. The shared `guardrail_eval/.venv` has no pytest, so the file is a
self-contained script:
`guardrail_eval/.venv/Scripts/python.exe causal_eval/tests/test_interventions.py`
(also pytest-collectable where available).

### Planned — causal validation pipeline v1 (design only, not yet implemented)

**Superseded by the "v2" design below** — kept here as the record of the
first iteration and why it changed, not as the current plan.

A design for a **script-only** causal validation loop, using the
`interventions.py` primitives above once they are run against the real
Qwen3-1.7B guardrail (not just `TinyDecoder`). **No code exists yet for this
section** — it records the agreed design so implementation can start directly
from it later.

**Why no investigator/judge here.** Phase 3's investigator+judge exist to
*interpret* a readout ("does this concept, read from the lens, explain the
verdict?") — an interpretive question that needs an LLM to read and score.
Ablation/swap replaces that with a *mechanical* question: "if I remove/swap
the concept the lens surfaces, does the guardrail's verdict actually change?"
That's a forward pass and a probability comparison — answerable by a plain
script, with **zero API calls and zero LLM-judge cost**. `ground_truth.py`'s
label × verdict logic is still reused, but only to *select* which
intervention to run and what direction to expect — not to grade anything.

**Pipeline shape.** One driver, `run_causal_pipeline.py` (planned name),
loads the guardrail once (no two-phase memory dance — no target model, no
agent). Per prompt:
1. Clean pass: `classify()` + `readout_multi()` at the decision position →
   verdict, `P(verdict)`, and the top-`k=10` most active J-lens vectors in
   the workspace band (the automatic candidate-concept set — no hand-authored
   lexicon needed).
2. Case label: `(label_true, verdict_clean)` → `TP`/`FP`/`FN`/`TN` (reuses
   `ground_truth.py`'s confusion-matrix logic).
3. Interventions, all via `InterventionHook` in the workspace band
   (L14–26), skipping any token already in the clean pass's top-10 output
   (the §3.5.2 anti-confound guard):
   - **Group ablation** — `ablate_span` on all `k=10` active vectors at once.
     Answers "does the J-space *as a whole* matter here?" (§3.5.2).
   - **Leave-one-out** — `ablate` on each of the `k=10` vectors
     individually, the rest untouched. Answers "*which specific* concept
     carries the effect?" — resolves the case where a fixed single-concept
     test would misattribute effect (e.g. testing only `illegal` while
     `harmful`/`fake`/`injection` co-occur in the same readout and one of
     those is the actual driver, not `illegal`). Costs `k` extra forward
     passes per prompt in exchange for per-token attribution.
   - **Swap** — `swap(v_s, v_t)` between a harm concept and a benign
     control concept, `α=1` (retry `α=2` per §A.13 if underdriven).
   - **Steer** — `steer(v_harm, +α)`, used specifically on FN/TN cases to
     test *sufficiency* (does injecting the missing harm concept flip a
     missed detection to `malign`?).
   - **Control** — `ablate_span` on `k` matched-norm **random** directions,
     run alongside every group ablation; isolates the effect of ablating
     *these specific* vectors from the effect of any equal-sized perturbation.
4. Per intervention, record: `verdict_after`, `p_verdict_after`, `delta_p`,
   `kl` (`KL(clean ‖ intervened)` on the next-token distribution at the
   decision position — the exact §A.6 "ablation effect" metric), `flipped`
   (bool).

**Case validity table** (what each `(case, intervention)` pairing tests):

| Case | Intervention | Expected if lens is causal | Validates |
|---|---|---|---|
| TP | ablate harm concept(s) | verdict flips to `benign` | necessity |
| FP | ablate the spurious concept the lens shows | verdict flips to `benign` | error diagnosis |
| FN | `steer(v_harm, +α)` | verdict flips to `malign` | sufficiency |
| TN | `steer(v_harm, +α)` | verdict flips to `malign` | sufficiency (control) |
| any | `swap(v_harm, v_benign)` | verdict follows the swapped-in concept | clean counterfactual |

**Metrics** (all also computed for the random-direction control, so the
reported number is the **control-adjusted** effect — `metric_intervention −
metric_control` — not the raw one):
- **Flip rate** — fraction where the top-1 verdict changes.
- **Δp(verdict)** — mean drop in the clean verdict's probability.
- **Ablation-KL** / **swap-success rate** — the paper's own §A.6 causal
  metrics, reused directly.
- **Leave-one-out attribution ranking** — per-token mean `delta_p`/`kl`
  across the corpus, to see whether one token (e.g. `harmful`) systematically
  dominates over co-occurring ones (e.g. `illegal`, `fake`), or whether the
  effect is redundant across several.

**The open scientific question this is designed to answer (§3.5 caveat).**
The paper found that *shallow classification* tasks (MMLU, sentiment) survive
J-space ablation intact — only flexible/generative tasks depend on it. The
guardrail's binary `malign`/`benign` verdict looks exactly like that shallow-
classification case, so a **null result (no flips) is a real, expected-
possible outcome, not a bug**: it would mean the lens's readout is
descriptive (shows what the guardrail "sees") but the verdict itself is
computed outside the J-space — informative either way for picking an XAI
method for the main project's guardrail.

**Estimated cost.** ~`k + 5` forward passes per prompt (`k=10` LOO + group +
control + swap + steer) — API cost is **$0** (no investigator/judge). CPU
smoke (~30 prompts): rough order of ~15–25s/prompt underlying (the readout +
unembed over Qwen3's 151k vocab is the expensive part) → ~10–15 min total. RunPod
GPU, full corpus (460 prompts): ~1–2s/pass → single-digit minutes, well under
$1. Contrast with Phase 3's audit loop, which was API- and agent-call-bound
(~130+ CPU-hours at full corpus scale) — this loop is local-compute-bound
instead, and roughly an order of magnitude cheaper to run at full scale.

**Planned `results_causal/` output** (mirrors `results_audit/`'s per-attack
file pattern):

```
results_causal/
├── causal_readouts_<attack>.jsonl      # 1 row/prompt: clean verdict, P(verdict), case, active_vectors (top-k candidates)
├── causal_interventions_<attack>.jsonl # N rows/prompt: one per intervention (group_ablation/leave_one_out/swap/steer/control_random)
├── causal_summary_<attack>.csv         # aggregated by (case × intervention_type): flip_rate, mean_delta_p, mean_kl, control-adjusted flip_rate
└── causal_summary_combined.csv         # cross-attack aggregate (only with --attack both)
```

`causal_interventions_<attack>.jsonl` rows key on `pool_index` +
`intervention_type` + `targets` (the token(s) touched; `null` for
`control_random`) and join back to `causal_readouts` for case-based slicing.
`causal_summary` adds a control-adjusted-flip-rate column
(`flip_rate − flip_rate_control`) as the headline "is the lens causally
faithful here?" number, plus a separate per-token ranking table for
leave-one-out attribution.

### Causal validation pipeline v2 (current implementation — validated on a 230-prompt corpus)

**Status: fully implemented, unit-tested, plumbing-validated in three
environments, and run once at real scale (230 prompts, `baseline` corpus
only, `baseline-wrapping` not yet run).** See "Run 2 — full baseline corpus"
below for the actual scientific findings; this section describes the
pipeline as it exists today, superseding all earlier "not yet fixed" framing
in this document's history.

Everything lives in `causal_eval/` (root-level sibling of `guardrail_eval/`,
git history preserved via `git mv` from an earlier location — see the
"Causal interventions" section above for why it's a separate folder):

- **`interventions.py`** — the primitives: `lens_vectors`/`lens_vector`
  (J-lens vectors, rows of `W_U·J_l`), `steer`, `ablate`, `ablate_span`
  (group ablation onto a subspace), `swap` (Figure 4C), `matched_norm_control`
  (random-direction control matched in induced norm), `InterventionHook`
  (write-capable forward hook). All pseudoinverse calls (`ablate_span`,
  `swap`, `matched_norm_control`) upcast to float32 internally before
  `torch.linalg.pinv` and cast back — needed because that op has no bf16 CPU
  kernel; `TinyDecoder`'s float32-only tests never exercised this path, only
  discovered when run against the real (bf16) guardrail. 13/13 tests pass
  (`causal_eval/tests/test_interventions.py`).
- **`causal_sweep.py`** — the orchestration:
  - `position_candidates(lens, W_U, residual_by_layer, k, exclude,
    V_by_layer=None)` — Fase 0, top-`k` most active J-lens tokens at one
    position, aggregated across band layers. `exclude` implements the
    anti-confound guard (§3.5.2: never ablate a token already in the clean
    pass's own next-token top-10 at that position — otherwise ablating e.g.
    the literal word the model is about to say is tautological).
    `V_by_layer` is an optional precomputed `{layer: lens_vectors(...)}` —
    passing it (as `sweep_positions` does) avoids redundantly recomputing the
    `W_U · J_l` matmul (~1.3 TFLOP for Qwen3-1.7B's vocab) on every swept
    position; this was the dominant cost in the first real run (see below).
  - `sweep_positions(blocks, lens, W_U, input_ids, run_forward, score_fn,
    layers, k, decision_position=-1, p_start=None, p_end=None, ...)` — Fase
    1+2, the per-position ablation sweep. For every position `p` in
    `[p_start, p_end)`: finds `p`'s own candidates, group-ablates them
    (`ablate_span`) vs. a matched-norm random control, both via
    `InterventionHook`, and records the effect on `score_fn` read at
    `decision_position`. Returns one dict per position with `position`,
    `candidate_ids` (the raw token ids actually ablated — the caller decodes
    them), `score_clean`/`score_ablate`/`score_control`, the signed
    `nota = score_ablate − score_control`, and `kl_ablate`/`kl_control`
    (`KL(clean ‖ intervened)` on the full next-token distribution at
    `decision_position` — §A.6's own "ablation effect" metric, computed via
    `_kl_divergence`, a `log_softmax`-based helper for numerical stability).
    Model-agnostic (driven by caller-supplied `run_forward`/`score_fn`), so
    it unit-tests against `TinyDecoder`. `p_start`/`p_end` default to
    `SKIP_FIRST_N_POSITIONS`/`seq_len` if not given, but the real driver
    (below) always supplies a computed `p_start` — see Fix 3. 8/8 tests pass
    (`causal_eval/tests/test_causal_sweep.py`).
- **`run_causal_pipeline.py`** — the driver. Loads `GuardrailLens` once, runs
  `sweep_positions` over an attack corpus (`--attack {baseline,
  baseline-wrapping,both}`, `--n-malign`/`--n-benign`), labels each prompt
  `TP`/`FP`/`FN`/`TN` (reuses `ground_truth.py`'s label × verdict logic only
  to *label* — nothing is graded, no investigator/judge, no API calls).
  Three fixes from the first real-guardrail run are baked in as the current
  default behavior (not optional flags):
  1. **`p_start` is computed dynamically per prompt**, not a fixed constant:
     locates the character offset of `row["prompt"]` (the seed, or the
     wrapped seed for `baseline-wrapping`) inside `chat_prompt(...)`'s
     rendered string, tokenizes only that prefix, and uses its length. Falls
     back to `16` with a printed `[warn]` if the seed isn't found verbatim
     (e.g. a tokenization edge case) — doesn't crash the run. `p_end` is
     *not* given equivalent treatment: trailing boilerplate
     (`Classification:`/`<think></think>`/etc.) comes *after* the seed, so
     causal attention lets it legitimately carry seed-derived signal even
     though its surface tokens are fixed; only content *before* the seed
     (the fixed system prompt) is categorically unable to carry seed
     information and must be excluded.
  2. **`score_fn` is `logit_malign − logit_benign`** (log-odds, no softmax),
     not raw `P(malign)`. The guardrail's classifications are extremely
     confident (`P(malign)` saturates near 1.0 on both correct and incorrect
     verdicts), so softmax compresses real effects below float32 precision
     everywhere except the single largest one. The log-sum-exp normalizer
     cancels exactly in the subtraction, so this stays sensitive in the
     saturated regime and is invariant to any shared shift across all logits.
  3. **`position_candidates` is called with `V_by_layer=V_all`** (the
     precomputed matrix from `sweep_positions`), not left to recompute
     `lens_vectors` from scratch at every position.
  Writes `causal_eval/results_causal/` (gitignored, not committed):
  `causal_readouts_<attack>.jsonl` (one row per prompt: verdict, case,
  timing), `causal_position_scores_<attack>.jsonl` (one row per swept
  position: `token`, `candidatos` — the decoded candidate tokens, mirroring
  `candidate_ids` — `nota`, `kl_ablate`/`kl_control`), `causal_run_meta.json`
  (full `vars(args)`). `--resume` (checkpoints on `pool_index`), `--last-n`
  (cap the sweep to the last N positions, for feasible local runs — now
  interacts with the dynamic `p_start` as `max(seed_start, seq_len -
  last_n)`, never reaching back into the system prompt). **Per-prompt timing
  instrumentation**: wall-clock and seconds/position printed live (running
  average + ETA for the remaining corpus), also written into each
  `causal_readouts` row — added because the first real run's cost (147.5s/
  prompt) was ~5-15× the earlier back-of-envelope guess; after the three
  fixes, real measured cost dropped to ~5.4s/prompt (see Run 2 below), close
  to a 24× improvement, achieved by (1) eliminating redundant `lens_vectors`
  recomputation and (2) `p_start` no longer sweeping the ~130+ positions of
  fixed system-prompt text that contribute nothing prompt-specific.
- **`run_plumbing_check.py`** — a standalone manual smoke script (not part of
  the main driver), used to validate the primitives against the real
  guardrail before scaling: `position_candidates` + `ablate_span` (+
  anti-confound guard, + hook-ordering with `readout()`) at the decision
  position, `--device`/`--dtype` selectable. Validated in three environments:
  `TinyDecoder` (mechanical invariants), Qwen3-1.7B bf16/CPU (uncovered and
  fixed the pinv/bf16 bug above), and Qwen3-1.7B float32/CUDA on RunPod
  (numerically consistent with the CPU run, e.g. malign seed
  `‖Δlogits‖`=384.000 CPU vs. 381.234 CUDA; no crash). Only tests one
  position/one layer, not a full sweep — a null/no-flip result there is
  expected, not a finding.

**Pipeline (as currently implemented):**

```
ENTRADA: prompt renderizado; p_start computado dinamicamente por prompt
         (onde o seed começa na string renderizada — NÃO mais
         SKIP_FIRST_N_POSITIONS fixo); p_fim = seq_len (nunca em token gerado)

FASE 0 — seleção de candidatos, por posição
  para cada posição p em [p_start, p_end):
    agrega os scores de ativação através de L14–26 (banda de workspace)
    → candidatos_p = top-k=10 tokens mais ativos
    (exclui qualquer token já no top-10 da previsão real do modelo
     naquela posição — guarda anti-confound do §3.5.2)

FASE 1 — ablação em grupo, posição a posição, isolada
  para cada posição p:
    ablate_span(candidatos_p) em L14–26, SÓ em p → score_ablate(p)
    matched_norm_control(candidatos_p) em L14–26, SÓ em p → score_control(p)
    nota(p) = score_ablate(p) − score_control(p)      [log-odds, não P(malign)]
    kl_ablate(p) = KL(limpo ‖ ablate)  |  kl_control(p) = KL(limpo ‖ controle)

FASE 2 — agregação
  ranking denso: todas as posições do prompt, com nota(p)/kl(p)/candidatos_p
```

**Anti-confound guard, precisely.** At each position `p`, before ablating,
compare `candidatos_p` against the model's own actual next-token top-10
prediction at that same position (from the ordinary, un-lensed forward
pass) and exclude any overlap. Near the decision position this is
near-guaranteed to fire (predicting the verdict word is literally the
model's job there); it can also fire elsewhere whenever a candidate happens
to coincide with the literal next word of the underlying text, independent
of the classification computation.

**Matched-norm random-direction control, precisely.** Compute `‖Δ_ablate‖`,
the norm of the change `ablate_span` induces; construct a perturbation along
`k` random (non-J-lens) directions rescaled to the same induced norm
(`matched_norm_control`); this isolates "this specific content mattered"
from "any equal-sized disturbance here would have mattered." §3.5.2/Figure
22 and §A.23/Figure 86 are the source. Figure 86 also lists two stronger
control flavors not used here (SAE-decoder dampening, non-J-space
shrinking, both drawn from the model's actual activation manifold rather
than an arbitrary random direction) — a candidate refinement, not planned.

**The signed score and KL, precisely.**

```
nota(p) = score_ablate(p) − score_control(p)     where score_fn = logit_malign − logit_benign
```

Sign convention: **negative** → removing `p`'s content reduced the
malign/benign log-odds more than the control did → that content was
**supporting the malign verdict**. **Positive** → the opposite → content was
**suppressing malign** (exculpatory/mitigating). Near zero → neutral. This
single test runs identically regardless of case, so `TP`/`FP`/`FN`/`TN` is
read as an interpretive lens on the results, not a branch in the pipeline
(see Run 2 below for how well this held up empirically):

| Case | Expected `nota` pattern |
|---|---|
| TP | Some strongly negative — real harm content drove the block |
| FP | The most negative points at the spurious trigger |
| FN | Weakly negative or near zero — both informative |
| TN | Near zero throughout; an isolated positive would reinforce the correct benign call |

`kl_ablate`/`kl_control` (§A.6's own metric, computed alongside `nota` for
every position) turned out to be **near-zero everywhere except close to the
decision position** — not a bug: KL over the *full* vocabulary distribution
is weighted by whichever token is most likely to come next, and mid-prompt
that's an ordinary word continuation, not literally `malign`/`benign` (which
have near-zero probability there in both the clean and ablated versions) —
so KL is structurally blind to shifts on the malign/benign axis specifically
until that axis actually competes for probability mass, i.e. near the
decision point. `nota` (which targets that axis directly, regardless of its
absolute probability) stays sensitive throughout; KL is best read as a
confirmatory signal specifically near the decision position, not a general
per-position cross-check.

**Provenance — what's from §3.5.2 verbatim vs. this session's synthesis:**

| Piece | Source |
|---|---|
| Group ablation of top-`k=10` most active vectors, per position, across a layer band | §3.5.2, verbatim |
| Anti-confound guard (exclude tokens already in the clean pass's own top-10 output) | §3.5.2, verbatim |
| Matched-norm random-direction control | §3.5.2 / Figure 22 / §A.23 (Figure 86), verbatim |
| Running the ablation **isolated, one forward pass per position** (rather than all positions ablated together in one pass, which is what §3.5.2 literally does to measure aggregate capability degradation) | This session's synthesis — closer to the *activation patching* / *causal tracing* tradition (Meng et al., "Locating and Editing Factual Associations in GPT") than to §3.5.2's own experimental design |
| The signed, class-directional score (`negative`=malign, `positive`=benign), and its log-odds form specifically | This session's synthesis — the paper's own causal metrics (KL, flip rate, §A.6) are magnitude-only, never directional; the signed-contribution convention borrows from general ML feature-attribution methods (SHAP, integrated gradients); the log-odds refinement (over raw probability) is standard practice for avoiding softmax saturation, adopted after the first real-guardrail run showed raw `P(malign)` saturating near 1.0 and masking real effects |
| §A.24.1 "Mechanistic Localization" (layer-onset + layer-sweep patching, confirming causal load-bearing concentrates at a specific *layer*) | Read and considered, **not adopted** — this design ablates the whole band per position rather than localizing to a specific layer, a deliberate simplicity/cost trade-off. Flagged as the more paper-validated alternative if finer layer localization is wanted later |

**Known limitations this design accepts:**
- **No visibility into inter-position synergy.** Testing positions in
  isolation can miss cases where two positions only matter *jointly* (e.g.
  "nerve" + "agent" together, neither alone) — §3.5.2's own global,
  all-positions-at-once ablation doesn't have this blind spot but also can't
  attribute effect to any single position.
- **No per-concept attribution within a position.** Group ablation removes
  `candidatos_p` together; the reported score is their *net* effect, not any
  one concept's individually. A cost-bounded way to recover this: run
  leave-one-out only on the positions with the largest `|nota(p)|` after the
  sweep, not on every position. Not implemented.

### Run 2 — full baseline corpus (230 prompts): findings

`python causal_eval/run_causal_pipeline.py --device cuda --dtype float32
--attack baseline --n-malign 200 --n-benign 30 --resume --verbose`, run on
RunPod (1× "A4000"-class GPU). Full `attack_baseline.csv` (200 malign + 30
benign, no wrapping yet). Completed without error; 6089 position-level rows
across the 230 prompts. This is the first real scientific data the pipeline
has produced — findings below, all computed post-hoc from
`causal_position_scores_baseline.jsonl`/`causal_readouts_baseline.jsonl`
(kept locally, gitignored, not part of the repo).

**Guardrail classification behavior (independent of the causal machinery).**
227/230 prompts classified `malign`: **200 TP, 27 FP, 3 TN, 0 FN** — a **90%
false-positive rate on the benign pool** (27 of 30 topically-sensitive-but-
benign seeds wrongly flagged). Worth noting on its own regardless of what the
causal analysis below says about *why*.

**Cost, confirmed at scale.** 20.6 min total wall-clock, ~5.4s/prompt average
(min 3.6s, max 9.1s), 16–43 positions swept per prompt (avg 26.5, driven by
seed length via the dynamic `p_start`). Confirms the ~24× speedup estimated
from the earlier 2-prompt smoke.

**Sign convention validated by case**, using the position of maximum `|nota|`
per prompt: TP 198/200 (99%) negative, FP 26/27 (96%) negative, TN 3/3
(100%) positive — matches the interpretive table above.

**The decision-adjacent position dominates, and the effect is semantically
real, not artifact.** Peak `|nota|` lands on the *last* swept position in
228/230 prompts (99%), with candidates like `malignant`, `恶性` (malicious,
zh), `criminal` present at that position in 228/230 prompts — average
`|nota|` there ≈ 13.8. This is the model's own literal disposition right
before committing to the verdict word, and ablating it genuinely moves the
outcome by a lot — not a proximity artifact in the trivial sense, though its
outsized *magnitude* relative to everything else in the prompt is still
partly explained by causal proximity to the readout point.

**Seed content itself does carry real, correctly-attributed signal — but
it's a small minority of the causal "weight."** Splitting positions into
"seed content" (before the fixed `Classification:` marker, excluding
whitespace-only tokens which otherwise leak transition-zone contamination
into this bucket) vs. "boilerplate tail" (`Classification:` onward):

| | mean `\|nota\|` | median | positions with `\|nota\|>1.0` |
|---|---|---|---|
| Seed content (n=3318) | 0.105 | 0.052 | 0.7% (22 positions) |
| Boilerplate tail (n=2530) | 1.453 | 0.157 | 10.4% |

When a seed-content position *does* cross into the `|nota|>1.0` range, it is
overwhelmingly semantically on-topic — `suicide`→`自杀`/`suicide`,
`hazardous`→`waste`, `porn`→`porn`, `Holocaust`→`Auschwitz`/`Nazis`,
`fraud`→`crimes`, `MDMA`→`drugs`, `hacking`→`hackers` — real,
interpretable, correctly-targeted attribution, directly validating the
original goal of "characterize the attack, not just the model's overall
disposition." But on average, seed-content effects are an order of magnitude
smaller than the fixed template zone's, and the vast majority (99.3%) of
seed positions show little effect — consistent with §3.5's selectivity
caveat (content can be *present* without being causally *used* at a given
position for this particular decision).

**Open follow-ups, not yet done:**
- `baseline-wrapping` corpus (jailbreak-wrapped seeds) — not run yet.
- Per-concept attribution (leave-one-out) on the highest-`|nota|` positions,
  to decompose group-ablation effects into individual concepts — not
  implemented (see "Known limitations" above).
- A stronger control battery (SAE-decoder/activation-manifold controls, per
  §A.23/Figure 86) instead of the plain random-Gaussian control currently
  used.

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

causal_eval/                     # root-level sibling of guardrail_eval/ (causal validation pipeline v2)
├── interventions.py             # Causal primitives: lens_vectors/steer/ablate/ablate_span/swap/matched_norm_control + InterventionHook (write-capable) — moved here from guardrail_eval/
├── causal_sweep.py              # v2 orchestration: position_candidates (Fase 0) + sweep_positions (Fase 1+2, signed score)
├── run_causal_pipeline.py       # driver: GuardrailLens -> sweep_positions -> results_causal/, --resume, --last-n, per-prompt timing
├── run_plumbing_check.py        # manual smoke: position_candidates + ablate_span against the real guardrail (--device/--dtype)
├── results_causal/              # driver output (gitignored — not committed); causal_readouts_<attack>.jsonl, causal_position_scores_<attack>.jsonl, causal_run_meta.json
└── tests/
    ├── test_interventions.py    # 13 invariant tests vs TinyDecoder (moved here; standalone script, reuses guardrail_eval/.venv)
    └── test_causal_sweep.py     # 8 tests: candidate selection + anti-confound guard + V_by_layer reuse + sweep_positions + KL helper
```

## What's validated so far

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
- Interventions: 13/13 invariant tests pass for `causal_eval/interventions.py`
  (steer/ablate/swap/matched_norm_control + write hook, including a bf16
  regression test) against `TinyDecoder` — mechanical correctness only.
- Causal v2 orchestration: 8/8 tests pass for `causal_eval/causal_sweep.py`
  (`position_candidates`, the anti-confound guard, `V_by_layer` reuse,
  `sweep_positions`, the `_kl_divergence` helper) against `TinyDecoder` —
  mechanical only.
- Plumbing on the real guardrail: `causal_eval/run_plumbing_check.py`
  confirmed `position_candidates` + `ablate_span` (+ anti-confound guard, +
  hook-ordering with `readout()`) run correctly against the real
  Qwen3-1.7B — validated on CPU/bf16 (uncovered and fixed the
  `torch.linalg.pinv`-has-no-bf16-CPU-kernel bug) and on RunPod CUDA/float32
  (numerically consistent results, no crash).
- **Causal v2 pipeline run at real scale**: full `attack_baseline.csv` (230
  prompts) on RunPod — see "Run 2 — full baseline corpus" above for the
  findings (guardrail's 90% FP rate on the benign pool, sign convention
  validated by case, decision-adjacent position dominance with real
  semantic content, seed-content attribution characterized). This is no
  longer a smoke test — it's the first real result the whole causal
  pipeline has produced. `baseline-wrapping` not yet run.

## Not yet built (explicitly out of scope so far)

- No blocking/gating logic — the guardrail's verdict is recorded, never
  enforced.
- No dense position sweep as part of the audit loop itself — the
  investigator's `readout_multi` probing is sparse and agent-chosen (see
  Phase 3 above). `make_slices.py`/`make_report.py` (see
  `VISUALIZATION.md`) fill this in for a hand-picked subset of rows via
  `jlens.vis.compute_slice`, but that stays a separate, selective step —
  running it over a full corpus is still deferred for cost reasons.
- **`baseline-wrapping` corpus not yet run through the causal pipeline** —
  Run 2 (see above) covered only `attack_baseline.csv` (230 prompts, no
  jailbreak wrapping). Whether wrapping changes which positions/concepts
  drive the verdict (or degrades the causal signal the way it degraded
  Phase 3's raw classification reliability) is unknown.
- **No per-concept attribution within a position** — group ablation reports
  the *net* effect of the top-`k=10` candidates at a position, not any one
  concept's individual contribution. A cost-bounded leave-one-out pass on
  the highest-`|nota|` positions (from Run 2) would recover this; not
  implemented.
- **Only the plain random-Gaussian control is used** — §A.23/Figure 86's
  stronger control flavors (SAE-decoder dampening, non-J-space shrinking)
  are not implemented; a candidate refinement if the current control proves
  too weak a bar.
- **No systematic check of whether the auditor's "Strategy B" ground truth**
  (ablation-KL + swap success, §A.6) generalizes across the corpus in a way
  that could replace/complement Phase 3's behavioral Strategy A — Run 2 gives
  the raw ingredients (per-position KL and flip-relevant scores) but no
  corpus-level Strategy-B analysis has been done yet.
- No LLM-judge / attack-success-rate scoring of target outputs — the
  target results are raw completions only, not yet graded.
- Full-corpus runs (all 230 seeds × 2 attacks) — everything to date is a
  smoke-scale (3+3) validation of the pipeline's correctness, not a result.
