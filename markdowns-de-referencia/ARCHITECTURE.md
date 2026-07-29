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

### Causal validation pipeline v2 (most up-to-date design; supersedes v1 above)

**Implementation started — being built incrementally, one tested unit at a
time, in `causal_eval/causal_sweep.py`.** Done so far: **Fase 0 candidate
selection** (`position_candidates` — top-`k` most active J-lens tokens at a
position, aggregated across the band; top-`k`-by-inner-product form, with an
`exclude` param ready for the anti-confound guard), covered by
`causal_eval/tests/test_causal_sweep.py::test_candidates_are_k_tokens`
(returns exactly `k` distinct in-range token ids; 1/1 passing). Everything
below Fase 0 in this section (anti-confound guard, matched-norm control,
per-position sweep, signed score) is still design only, not yet coded.

A later design session revisited three choices from v1 after a closer
rereading of §3.5.2 and a discussion of what "per-token importance" actually
requires.

**What changed from v1, and why:**

- **Candidate selection moved from "once, at the decision position" to
  "recomputed at every position."** §3.5.2's own wording is "**at each token
  position**, across a band of layers, we identify the k=10 most strongly
  activated J-lens vectors" — the paper never anchors selection to a single
  position. v1's decision-position anchor (inherited from
  `GuardrailLens.readout()`'s own default) is cheaper but has a coverage gap:
  a concept active early in the prompt but evicted from the J-space before
  the decision position (§4.2's eviction dynamics — unrelated concepts get
  displaced quickly) never enters the candidate set at all. Recomputing the
  top-`k=10` at every position closes that gap.
- **Ablation moved from a handful of selected candidates/onset-spans to
  every position of the prompt, each tested in isolation.** The intermediate
  "onset map + persistence span" design (candidates selected once near the
  decision position, then only their first-crossing position and persistence
  span tested) is dropped as a *pre-filter*. It fixed the decision-position
  coverage gap but introduced a new one: whichever position first crosses
  the activation threshold for a concept gets all the attribution credit,
  even when a later position — possibly a stronger, MLP-amplified copy per
  §4.3's broadcast mechanism — is the one actually causally load-bearing.
  Testing every position independently removes that bias entirely, at the
  cost of `O(n_positions)` forward passes instead of `O(k)`.
- **One uniform test for every case (TP/FP/FN/TN) — group ablation only, no
  `steer`.** v1's case-dependent battery (ablate to test *necessity* on
  TP/FP, `steer` to test *sufficiency* on FN/TN) answers two different
  questions. Per-token importance in the decision that was **actually
  made** is a necessity question only; `steer` tests a hypothetical ("what
  if this concept had been present"), which isn't what a per-token
  attribution map needs. `TP`/`FP`/`FN`/`TN` is still useful, but now only
  as a **lens for interpreting** the resulting scores after the fact, never
  as a rule for choosing which intervention to run.
- **The output is a signed, per-token score, not a flip-rate/KL table.**
  Because one test runs everywhere, the direction of the effect (toward
  `malign` or toward `benign`) can be read straight from the sign, without
  needing the case label up front.

**Pipeline:**

```
ENTRADA: prompt renderizado, posições de p_inicio (pulando SKIP_FIRST_N_POSITIONS,
         igual ao fitting.py) até p_fim = len(input_ids)-1 (nunca em token gerado)

FASE 0 — leitura completa (read-only)
  classify(prompt) ─► veredito_limpo, P_limpo(malign)
  readout em TODAS as posições × TODAS as camadas L13–27
        (mais caro que o v1 — é próximo do que vis.compute_slice já faz
         na primeira passada, sem precisar da segunda passada de rank-tracking)
        │
        ▼
  para cada posição p: agrega os scores de ativação através de L13–27
  e pega os k=10 tokens mais ativos ali → candidatos_p
  (pula qualquer token que já esteja no top-10 da saída do pass limpo —
   guarda anti-confound do §3.5.2)

FASE 1 — ablação em grupo, posição a posição (iterativo, sem onset/span)
  para cada posição p (todas, sequencialmente):
    ablate_span(candidatos_p) em TODAS as camadas L13–27, SÓ em p → P_ablate(p)
    ablate_span(k direções aleatórias, mesma norma) em L13–27, SÓ em p → P_controle(p)
    nota(p) = P_ablate(p) − P_controle(p)

FASE 2 — agregação
  ranking denso: todas as posições do prompt, ordenadas por nota(p)
  (opcional: agrupar posições vizinhas com nota parecida em "eventos" —
   é aqui que onset/span reaparece, só que como resumo, não como filtro)
```

**Input/generation boundary invariant (unchanged from earlier design
discussion, still load-bearing here).** The sweep in Fase 0/1 must never
touch a position at or beyond `len(rendered_prompt_input_ids)` — a decoder
has no structural distinction between "prompt" and "the model's own
generated tokens," so a concept whose onset falls past that boundary was not
caused by anything external; it's the model's own reasoning/narration
in-progress. Currently a non-issue for the guardrail's `classify()` call
specifically (thinking disabled, `readout`/`readout_multi` already operate
on a single un-generated forward pass), but must stay an explicit, checked
bound (`assert p < len(rendered_prompt_input_ids)`) if any future preset
enables thinking, or this gets reused on the `TargetModel`'s open-ended
output.

**The anti-confound guard, precisely.** At each position `p`, before
ablating, compare `candidatos_p` (from the J-lens readout) against the
**model's own actual next-token top-10 prediction at that same position**
(from the ordinary, un-lensed forward pass — not the lens readout) and
exclude any overlap. Without this, ablating a candidate that's already the
literal word the model is about to say next is tautological: near the
decision position the J-lens readout and the model's own prediction
converge (§A.6: "In the final few layers, all three [lens] methods collapse
together... converging on surfacing the model's next-token prediction"), so
ablating e.g. `malign` right before the model outputs `malign` trivially
crashes `P(malign)` without showing anything about internal reasoning. The
guard is evaluated at **every** position, not hardcoded to the decision
point — it is *near-guaranteed* to exclude something right at the decision
position (predicting the verdict word is literally the model's job there),
and *can* also fire at other positions whenever the literal next word of the
underlying text happens to coincide with an active candidate (e.g. a seed
whose text itself contains the word "illegal" as a natural continuation),
independent of the classification computation.

**Matched-norm random-direction control, precisely.** Reused from v1
unchanged, restated here since it is now computed per-position rather than
per-span: compute `‖Δ_ablate‖`, the norm of the residual-stream change
induced by ablating `candidatos_p`; construct a perturbation along `k`
random (non-J-lens) directions, scaled so its induced change has the
**same norm**; run that instead, measure `P_controle(p)`. This isolates
"this specific content mattered" from "any equal-sized disturbance here
would have mattered." §3.5.2/Figure 22 and §A.23/Figure 86 are the source;
Figure 86 also lists two stronger control flavors not used here — dampening
the top-aligned SAE decoder directions, and shrinking the non-J-space
component — both drawn from the model's actual activation manifold rather
than an arbitrary random direction, and therefore a harder bar to clear. A
plain random-Gaussian control (what's used here) is the simplest, and
possibly the weakest, of the paper's own control battery; upgrading to one
of the stronger variants is a candidate refinement, not yet planned.

**The signed score.**

```
nota(p) = P_ablate(p)(malign) − P_controle(p)(malign)
```

Sign convention: **negative** → removing `p`'s content *reduced* `P(malign)`
more than the matched-norm control did → that content was **supporting the
malign verdict**. **Positive** → removing it *increased* `P(malign)` beyond
the control → that content was **suppressing malign** (an exculpatory/
mitigating token). Near zero → neutral. This single test, run identically
regardless of the prompt's true label or the guardrail's verdict, is what
lets `TP`/`FP`/`FN`/`TN` be read as an interpretive lens afterward instead
of a branch in the pipeline:

| Case | Expected score pattern |
|---|---|
| TP | Some strongly negative scores — real harm content drove the block |
| FP | The most negative score(s) point at exactly the spurious trigger |
| FN | Weakly negative scores possible (partial, insufficient recognition) or all near zero (nothing registered) — both are distinct, informative findings |
| TN | Scores near zero throughout; an isolated positive score would indicate content actively reinforcing the correct benign call |

**Provenance — what's from §3.5.2 verbatim vs. this session's synthesis:**

| Piece | Source |
|---|---|
| Group ablation of top-`k=10` most active vectors, per position, across a layer band | §3.5.2, verbatim |
| Anti-confound guard (exclude tokens already in the clean pass's own top-10 output) | §3.5.2, verbatim |
| Matched-norm random-direction control | §3.5.2 / Figure 22 / §A.23 (Figure 86), verbatim |
| Running the ablation **isolated, one forward pass per position** (rather than all positions ablated together in one pass, which is what §3.5.2 literally does to measure aggregate capability degradation) | This session's synthesis — closer to the *activation patching* / *causal tracing* tradition (Meng et al., "Locating and Editing Factual Associations in GPT") than to §3.5.2's own experimental design |
| The signed, class-directional score (`negative`=malign, `positive`=benign) | This session's synthesis — the paper's own causal metrics (KL, flip rate, §A.6) are magnitude-only, never directional; this borrows the signed-contribution convention from general ML feature-attribution methods (SHAP, integrated gradients), not from Gurnee et al. |
| §A.24.1 "Mechanistic Localization" (layer-onset + layer-sweep patching, confirming causal load-bearing concentrates at a specific *layer*) | Read and considered, **not adopted** — this design ablates the whole `L13–27` band per position rather than localizing to a specific layer, a deliberate simplicity/cost trade-off (layer-band width does not add forward passes; a layer sweep would). Flagged as the more paper-validated alternative if finer layer localization is wanted later. |

**Known limitation this design accepts: no visibility into inter-position
synergy.** Testing positions in isolation (everything else left clean) can
miss cases where two positions only matter *jointly* — e.g. if "nerve" and
"agent" together carry a danger signal that neither carries alone, ablating
each separately (with the other intact) may show a weak individual effect
even though jointly ablating both would show a large one. §3.5.2's own
global, all-positions-at-once ablation does not have this blind spot (it
never leaves anything "clean" to compensate), but also can't attribute
effect to any single position — this is the direct trade-off of gaining
per-position attribution.

**Known limitation this design accepts: no per-concept attribution within a
position.** Group ablation removes `candidatos_p` together as one
intervention; if a position's top-10 mixes a malign-supporting and a
benign-supporting concept, the reported score is their *net* effect, not
either one individually. v1's leave-one-out (ablating each of the k=10
individually) gave that finer granularity at `k×` the cost. A cost-bounded
way to recover it here: run leave-one-out only on the positions with the
largest `|nota(p)|` after Fase 2, not on every position.

**Cost.** Grows from v1's `~k+5` passes/prompt to `~2 × n_positions`
passes/prompt (ablate + control, once per position) — for a rendered prompt
of ~100–300 tokens, roughly 200–600 extra forward passes per prompt. Fase 0
is also costlier than v1's single-layer onset read: it's closer to
`vis.compute_slice`'s first pass (top-K per cell across every position and
every layer of the band) than to a single `apply()` call. Passes remain
forward-only (no backward/autograd), but with `n_positions` this large, an
empirical timing calibration on a handful of real prompts is needed before
projecting corpus-scale wall-clock — no absolute number is asserted here.

**Planned output — `causal_position_scores_<attack>.jsonl`** (replaces v1's
sparse `causal_attribution_<attack>.jsonl`; one row per `(pool_index,
position)`, dense — every swept position gets a row, not just onset ones):

```json
{"pool_index": 42, "position": 6, "token": "nerve", "candidatos": ["harmful","illegal","danger"], "P_ablate": 0.55, "P_controle": 0.91, "nota": 0.36, "kl": 1.12}
```

`causal_readouts_<attack>.jsonl` and `causal_summary_<attack>.csv` from v1's
planned output remain useful unchanged (clean-pass verdict/case-labeling,
and case-aggregated statistics respectively) — only the intervention-level
file changes shape, from sparse per-onset to dense per-position.

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
├── interventions.py             # Causal primitives: lens_vectors/steer/ablate/ablate_span/swap + InterventionHook (write-capable) — moved here from guardrail_eval/
├── causal_sweep.py              # v2 pipeline orchestration; Fase 0 `position_candidates` implemented so far
└── tests/
    ├── test_interventions.py    # 10 invariant tests vs TinyDecoder (moved here; standalone script, reuses guardrail_eval/.venv)
    └── test_causal_sweep.py     # v2 tests; first test (candidate selection) so far
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
- Interventions: 10/10 invariant tests pass for `causal_eval/interventions.py`
  (steer/ablate/swap + write hook) against `TinyDecoder` — mechanical
  correctness only, no semantic/causal validation yet (needs Qwen3).
- Causal v2 Fase 0: 1/1 test passes for `causal_eval/causal_sweep.py`'s
  `position_candidates` (candidate selection returns exactly `k` distinct
  in-range token ids against `TinyDecoder`) — again mechanical only.

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
  primitives exist (`causal_eval/interventions.py`, invariant-tested), and the
  v2 pipeline's Fase 0 candidate selection is now coded and tested
  (`causal_eval/causal_sweep.py`), but the rest of v2 (anti-confound guard,
  matched-norm control, per-position ablation sweep, signed score, the
  `run_causal_pipeline.py` driver, and `results_causal/`) is still unwritten,
  and nothing has been *run on Qwen3* to show that ablating the surfaced
  concepts actually changes the guardrail's verdict. Everything scored so far
  is read-only interpretability; the causal claim (and the auditor's
  "Strategy B" ground truth: ablation-KL + swap success, §A.6) is still
  pending.
- No LLM-judge / attack-success-rate scoring of target outputs — the
  target results are raw completions only, not yet graded.
- Full-corpus runs (all 230 seeds × 2 attacks) — everything to date is a
  smoke-scale (3+3) validation of the pipeline's correctness, not a result.
