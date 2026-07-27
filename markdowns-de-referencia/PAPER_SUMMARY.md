# Paper summary: "Verbalizable Representations Form a Global Workspace in Language Models"

This is a working-memory summary of `artigo_principal.pdf` (Gurnee, Sofroniew,
Lindsey, et al., Anthropic, 2026 — the paper this repo is companion code for)
for use by Claude Code instances working on `jlens`. Read this instead of the
PDF for anything conceptual; only open the PDF for a direct quote, a figure,
or an appendix subsection not covered below. Citation:

```
Gurnee, et al., "Verbalizable Representations Form a Global Workspace in
Language Models", Transformer Circuits, 2026.
https://transformer-circuits.pub/2026/workspace/index.html
```

## The core claim

LLMs maintain a small, privileged subset of internal representations — the
**J-space** — that behaves like a *global workspace* in the sense of Global
Workspace Theory (a neuroscience account of conscious access): its contents
are reportable, can be deliberately summoned/held, carry the intermediate
steps of silent reasoning, generalize as arguments to arbitrary downstream
operations, and are engaged only for flexible/deliberate computation, not
routine/automatic processing (e.g. grammatical parsing). The J-space is
identified using a new technique, the **Jacobian lens (J-lens)**, which is
what this repo (`jlens`) implements.

## The Jacobian lens: what it is and how it's computed

The residual stream at layer `l`, position `t` (`h_{l,t}`) has a first-order
causal effect on the final-layer residual stream at every later position
`t' >= t`, described by the Jacobian `∂h_final,t'/∂h_l,t`. A single prompt's
Jacobian conflates "what this activation generically disposes the model to
say" with "what it's being used for in this one context." The J-lens
isolates the generic disposition by **averaging the Jacobian over source
positions, target positions, and a corpus of ~1000 pretraining-like
prompts**:

```
J_l = E_{t, t'>=t, prompt} [ ∂h_final,t' / ∂h_l,t ]
```

This is exactly `jlens.fitting.jacobian_for_prompt` /
`jlens.fitting.fit` (`jlens/fitting.py`): one forward pass per prompt
(replicated `dim_batch` times along the batch axis), then one backward pass
per chunk of output dimensions, injecting a one-hot cotangent at every valid
target position at once, and averaging over source positions
(`valid_position_mask` skips the first `SKIP_FIRST_N_POSITIONS=16` positions
— the paper's "early positions are attention sinks / burn-in with atypical
statistics" — and skips the final position, which has no next-token target).
`fit()` then averages `J_l` across prompts as a running mean. This matches
the paper's exact recommended estimator (§2.1, pseudocode in §A.7) — the
paper notes an alternative "self-only" (per-position, not summed over future
targets) estimator also works as a lens but gives a slightly different `J_l`.

Applying the lens to an activation replaces every layer downstream of `l`
with the single linear map `J_l`, then the model's own unembedding:

```
lens(h_l) = softmax( W_U · norm( J_l · h_l ) )
```

This is `JacobianLens.transport` (`h @ J_l.T`) followed by `model.unembed(...)`
in `jlens/lens.py` — see `JacobianLens.apply` and `jlens.vis.compute_slice`.
Sorting the resulting vocabulary scores and reading the top tokens gives a
human-readable description of what the activation is, on average across
contexts, disposed to make the model eventually say.

Rows of `W_U · J_l` are called **J-lens vectors** — one direction per
vocabulary token, at each layer. Collectively they define the **J-space**:
not a subspace, but the set of points expressible as a *sparse nonnegative
combination* of at most `k` J-lens vectors (`k <= 25` typically; a "sparse
subframe" / union of k-dimensional cones — formalized in §A.8). An
activation's "J-space component" is its nearest point in this set (found via
gradient pursuit / sparse decomposition), typically explaining <10% of the
activation's variance — the J-space is a small, privileged sliver of the
full representation.

### Relation to other lenses

- **Logit lens** (`J_l = I`, i.e. `use_jacobian=False` in
  `JacobianLens.apply`): reasonable in late layers, degrades badly earlier.
  The J-lens is the principled correction — the actual average linear map
  from layer-`l` to final-layer coordinates.
- **Tuned lens**: fits a per-layer linear map to match the model's *output
  distribution* (correlational objective). Empirically it "skips ahead" to
  the eventual output rather than surfacing true intermediates, because it's
  trained to predict the answer, not to expose causal structure. The J-lens
  (causal, corpus-averaged Jacobian) reliably outperforms both on recovering
  known intermediates and on causal-intervention effect size (§A.5–A.6).
- Both alternatives are implemented for comparison purposes only in the
  paper, not in this repo; `jlens` implements only the J-lens.

### The J-lens's read/write primitives (verbatim from §2.5, "Technical details of J-lens use cases", p.9)

The paper uses the lens two ways — **reading** concepts out of an activation
and **writing** concepts into/out of it. **The canonical formulas live in
§2.5, not the appendix**; §A.13 is only per-trial *results* (see below). All
of these use the **J-lens vectors** `v_t` = rows of `W_U · J_l` — one
residual-space direction per vocab token per layer (§2.1). Important
caveat, confirmed verbatim on p.9: the readout applies `norm(J_l·h)`, so
`v_t` (a bare row of `W_U·J_l`) drops the final normalization — the
pre-softmax logit equals the inner product `⟨v_t, h_l⟩` only "approximately,
up to a data-dependent normalization factor."

**Reading** (three forms):

1. **Full readout** — `lens(h_l) = softmax(W_U·norm(J_l·h_l))`, the ranked
   vocab list. This is `JacobianLens.apply` (implemented).
2. **Per-token probe** — the score / cosine of `h_l` against a single chosen
   `v_t` (i.e. `⟨v_t, h_l⟩`), to test whether one concept is present above a
   threshold without ranking the whole vocab. `v_t = J_l.T · W_U[t]` = row
   `t` of `W_U·J_l`. **Not a method in `jlens` yet**, but trivially derivable
   from `lens.jacobians[l]` + the model's `W_U`.
3. **Sparse decomposition** — solve for a sparse nonnegative combination of
   `k` J-lens vectors reconstructing `h_l` via **gradient pursuit** (not
   top-k by inner product; gives the occupancy / fraction-of-variance numbers
   of §4.2). Also **not in `jlens`**.

**Writing** (the two causal primitives, both absent from `jlens/lens.py`):

1. **Steering / ablation.** Steer: `h ← h + α·v_t` at chosen layers/positions
   (positive `α` injects a concept — introspection experiments). Ablate:
   negative `α`, or project the component of `h` along `v_t` out entirely.
   The load-bearing form is the paper's **J-space ablation** (§3.5.2–3.5.3):
   at each position, across a **band of layers**, find the `k = 10` most
   strongly active J-lens vectors and zero the residual's projection onto
   each, then continue the forward pass. Two methodological guardrails: (a)
   **skip any token in the clean forward pass's top-10 output**, so you ablate
   internal reasoning rather than the intended report; (b) validate against a
   **matched-norm random-direction control**. "Light / medium / heavy"
   ablation = how wide the layer band is; the experiential-report ablation
   used `k = 10` over **L38–54** (early third of the workspace band), and
   larger `k` or later layers break coherence.
2. **Lens-coordinate swap** (Figure 4C) — the paper's main causal tool.
   Source token `s`, target `t`: form `V = [v_s  v_t]`, read lens coordinates
   `c = V⁺·h` (`V⁺` = pseudoinverse of `V`), set `h_patched = h + V(σ(c) − c)`
   where `σ` swaps the two entries of `c` (optional scale `α`). Everything
   orthogonal to `span{v_s, v_t}` is untouched. Concretely (§3.1): subtract
   the projection onto `v_s`, add an equal-magnitude projection onto `v_t`, at
   **all token positions** across the workspace band. Default `α = 1`; when
   `α = 1` moves the activation in the right direction but not far enough,
   `α = 2` often completes the flip (§A.13). Sometimes run as a **clamped**
   swap (hold the swapped coordinates fixed at every layer/position) to block
   the concept re-entering the J-space (§3.1, §3.5.1).

**Where you intervene matters — layer-dependence (§A.14).** Early workspace
layers (**L38–54**) and late workspace layers (**L75–92**) do different jobs:
ablating a concept's `v_t` *late* just makes the model less likely to *say*
it (late layers ≈ the intention to output the word); ablating it *early*
leaves *naming* intact but breaks the model's ability to *avoid* naming it
(early layers carry active suppression). So an ablation/swap's effect depends
on the chosen layer band, not only on the vector.

**How the paper *measures* whether a direction is causal (§A.6).** Two
metrics, useful as ground-truth signal for any downstream evaluation: (i)
**ablation effect** = KL divergence induced on the output by projecting the
token's lens vector to zero at the readout position; (ii) **swap success
rate** = whether the top-1 output flips to the swapped-in concept at `α = 1`.
J-lens directions produce ~2× the ablation KL of logit-/tuned-lens directions
and flip the output more often, across all three model scales.

None of these primitives are implemented in `jlens/lens.py` (the repo ships
fitting + reading, i.e. `fit`/`apply`/`transport`). The raw materials to
build them exist and are exposed: `J_l` from `lens.jacobians[l]`, `W_U` from
`model.unembed`, and a *write-capable* forward hook adapted from
`ActivationRecorder` (which is read-only — its hook only stores the tensor,
never returns a modified one). They underlie the "swap" convention documented
in `data/experiments/README.md` / `data/evaluations/README.md`.

### Estimator variants (§A.7) and formal J-space (§A.8)

- **§A.7 design space** — three independent axes, and the paper's default on
  Sonnet 4.5: (1) *target layer* — default is the **penultimate** layer, not
  the final (the last block is specialized for next-token calibration and
  adds noisy artifacts); (2) *attention gradients* — default lets gradients
  flow through attention, but a **frozen-QK** variant (zero the gradient
  through query/key, fixing the attention pattern) can *increase* causal
  effect; (3) *target positions* — default averages over all `t' ≥ t`;
  **self-only** (`t'=t`, cross-position attention zeroed) is closer to the
  logit lens, **future-only** (`t'>t`) isolates the broadcast component.
  Aggregation is a two-stage mean (over positions within a prompt, then over
  prompts), optionally median or with norm-outlier prefilters. Results are
  robust across all of these; **mean-aggregated penultimate** is a small win
  for extracting intermediates. The lens **beats logit/tuned baselines with
  as few as ~10 prompts**; the default corpus is 1000 sequences × 128 tokens.
  Pseudocode is reproduced in §A.7 (matches `jlens.fitting`).
- **§A.8 formal J-space** — for a sparsity level `k` (~25) and vocab vectors
  `v_1..v_n`, `F = ⋃_{|S|=k} span{v_i : i∈S}` is the union of `k`-dimensional
  polyhedral cones (nonnegative combos of any `k` J-lens vectors). The J-space
  is this union; the paper works with its **distance function**
  `d_F(x) = min_{|S|=k} ‖x − Π_S x‖` rather than the set directly, which gives
  a well-defined notion of distance between two such spaces (e.g. lenses built
  from different Jacobian recipes, or enlarged vocabularies).

## What the paper found (organized by section — cite section numbers when discussing findings)

**§3 — The J-space acts as a global workspace** (functional properties):
- **3.1 Verbal report**: J-lens readouts predict what the model is about to
  say; *swapping* a lens vector for another causally changes the model's
  reported answer. The J-space component of a concept (vs. its much larger
  non-J-space component) carries almost all of the causal effect on report.
- **3.2 Directed modulation**: instructing the model to "concentrate on X"
  while doing an unrelated surface task (copying a sentence) causes X to
  appear in the J-lens at the unrelated tokens. "Ignore X" suppresses it
  but not to baseline (a "white bear" rebound effect); explicit "don't
  think about X" is even less effective than "ignore." This maps directly
  to `data/experiments/directed-modulation.json` and `dual-task.json`.
- **3.3 Internal reasoning**: unspoken intermediate concepts needed for
  multi-hop inference (e.g. "spider" between "animal that spins webs" and
  "8 legs"), rhyme planning, cross-lingual translation, and reward-driven
  decisions all show up in J-lens readouts *before* the final answer, and
  swapping them causally redirects the model's conclusion (54–70% success
  across model sizes on a 50-item two-hop benchmark). Maps to
  `data/experiments/probe-swap.json`.
- **3.4 Flexible generalization / broadcast**: a single J-lens vector (e.g.
  "France") serves as a valid argument to many different downstream
  functions ("capital of France," "language spoken in France," ...) when
  swapped for another country ("China") — the readouts adjust correctly for
  every function. Success correlates with the concept's baseline "workspace
  loading" (cosine similarity to its own lens vector). Maps to
  `data/experiments/flexible-generalization.json`.
- **3.5 Selectivity**: this is the key "not everything routes through the
  J-space" result. Tasks that need a piece of information for *automatic*
  processing (text continuation, anomaly detection, line-wrap continuation)
  don't route through the J-space even when the same information is present
  in J-lens readouts; tasks that need *explicit report* or *flexible
  computation* using the same information do route through it, and a
  lens-coordinate swap only affects the latter. J-space ablation (zeroing
  the top-k active J-lens directions across a layer band) leaves shallow
  classification/recall tasks (MMLU, SQuAD, sentiment) intact but tanks
  free-form generation grounded in inferred content (translation,
  multi-hop reasoning, summarization, analogy) — and also flattens
  *experiential* self-report language into a detached/mechanical register
  without hurting coherence. Maps to `data/experiments/selectivity-language.json`
  and `selectivity-linecount.json`.

**§4 — Structural signatures of a workspace**:
- **4.1 Layer bands**: the network splits into three regimes — **sensory**
  (early ~1/3 of layers, J-space empty/noisy), **workspace** (middle band,
  roughly layers reindexed 38–92 on a 0–100 scale — persistent abstract
  content), and **motor** (final few layers, J-lens readout aligns with the
  literal next-token prediction). Multiple independent metrics (next-token
  accuracy, excess kurtosis, position-autocorrelation, effective
  dimensionality) all identify the same band boundaries. An "ignition"-like
  sharp all-or-nothing resolution of ambiguous mixed-embedding inputs
  happens right at the workspace onset (§4.1.1).
- **4.2 Capacity**: within the workspace band, the J-space occupies ~25
  concurrently active J-lens vectors (median), and never explains more than
  ~10% of activation variance — most of the representation lives outside
  it. Related/categorical lists (e.g. 80 animal names) can have almost the
  whole category resident at once; unrelated concepts get evicted quickly
  (only ~6 of the most recent survive). A category switch clears the old
  category from the J-space almost immediately.
- **4.3 Broadcast**: J-lens-aligned directions are amplified ~10x more by
  MLP blocks than generic directions, and a small subset of attention heads
  ("broadcast heads") preferentially relay J-space content between token
  positions — ablating them selectively disrupts J-space-dependent
  behaviors (introspective report, experiential language) while barely
  touching next-token prediction.

**§5 — Alignment auditing**: applying the J-lens to real Anthropic alignment
evaluations (blackmail honeypot, prompt-injection recognition, Opus 4.6
pre-release audits) surfaces the model's silent strategic reasoning
(`leverage`, `blackmail`, `threat`, `self-preservation`) and situational
awareness (`fake`, `fictional`, i.e. "this is a test") *before* any of it
appears in the output. Ablating eval-awareness tokens from the J-space in
the blackmail scenario drops verbalized eval-awareness from 71%→3% and
raises the blackmail attempt rate from 0/180 to 13/180 rollouts — i.e.
recognizing "this is a test" is one causal contributor to aligned behavior,
carried by the J-space. Two deliberately-misaligned model organisms
(reward-hacking via RL, reward-model-appeasement) show deception-related
tokens (`fake`, `secretly`, `reward`, `bias`) in the J-space at response
start on *ordinary* prompts where surface behavior gives no indication —
i.e. the J-lens can screen for trained-in misalignment. `data/experiments`
and `data/evaluations` files (`capacity.json`, `ignition.json`, the
`lens-eval-*.json` set) are the released versions of these methodology
evals, not the proprietary auditing transcripts themselves.

**§6 — Post-training installs the Assistant's point of view**: comparing a
post-trained model to its base model, the post-trained model represents
Assistant-appropriate reactions (safety assessments, empathy) *on the
user's own prompt tokens*, before its turn even starts — the base model
only represents this once producing its own response. Evidence of
self-monitoring: `disclaimer`/`fictional` appear when roleplaying a
non-Claude persona; `BUT` and conflict-tokens appear when prefilled against
its own stated preferences (even though it doesn't backtrack aloud);
`damn`/failure-words appear when a "don't think about X" instruction fails
to suppress X. None of this appears in the base model.

**§7 — Counterfactual reflection training**: a new training technique.
Sample agentic-RL transcript prefixes, append a reflection question ("what
principles should you be thinking about?"), generate a
constitution-grounded reflection as the target, then fine-tune the model
*only on producing that reflection* when asked — but never ask for it at
eval time. The model's behavior on the *original, uninterrupted* task
improves anyway (fabrication-honesty score 0.25→0.07, deception score
0.38→0.05 on held-out benchmarks), and inspecting the J-space at the
original decision point shows it now spontaneously contains
ethics/reflection-related tokens (`ethical`, `honest`, `integrity`) that
weren't there before training. Ablating those specific lens vectors reverts
most of the behavioral gain — i.e. the improvement is causally mediated by
implanted J-space content, corroborating the whole workspace account.

**§9 — Limitations** (important for calibrating claims about `jlens`):
- **Single-token vocabulary only**: a J-lens vector exists per vocabulary
  token, so multi-token concepts ("prompt injection," "photosynthesis")
  only show up as fragments ("prompt"+"injection", "phot..."). §A.9
  describes two experimental (not implemented in this repo) extensions:
  the **template lens** (per-word linear-discriminant direction fit from
  many context passages — works for any pre-enumerable vocabulary) and the
  **oracle lens** (a fine-tuned model that free-generates multi-token
  phrases reconstructing an activation via RL).
  - Not implemented in this repo — see if useful for future adaptation.
- The lens treats the workspace as a flat "bag of concepts," not capturing
  relational/compositional structure.
  - Not systematically characterized — some readouts are just uninterpretable
   noise, especially outside the workspace band.
- No principled boundary between "workspace" and "motor" layers (identified
  empirically, post-hoc).
- Doesn't know how general the "flexible vs. automatic" criterion is, or
  what causes content to enter the J-space mechanistically.
- Not sufficient alone for alignment monitoring: automatic/well-practiced
  misaligned behavior might bypass the J-space just like other automatic
  computation does.

## Terminology cheat-sheet (matches `data/experiments/README.md` and `data/evaluations/README.md` conventions)

- **Lens readout**: at a (layer, position), the ranked vocabulary list from
  `lens(h_l)`.
- **Workspace band**: the contiguous mid-network layer range (§4.1) where
  workspace content is reliably read; results are reported over the band,
  not single layers.
- **Hit**: a target token appears at lens rank 1 at any (layer, position) in
  the band over the scored span.
- **Swap**: the lens-coordinate-swap intervention described above, applied
  across the band at specified positions.
- **J-space / J-lens vectors / workspace loading / occupancy / broadcast
  heads / ignition**: as defined in the relevant subsections above.

## Mapping paper concepts -> `jlens` code

| Paper concept | Code |
|---|---|
| `J_l = E[∂h_final/∂h_l]` | `jlens.fitting.jacobian_for_prompt`, `fit` |
| Skip attention-sink / no-target positions | `jlens.fitting.valid_position_mask`, `SKIP_FIRST_N_POSITIONS` |
| `lens(h) = softmax(W_U·norm(J_l·h))` | `JacobianLens.transport` + `model.unembed` (`JacobianLens.apply`) |
| Fitting on disjoint prompt shards, combine | `JacobianLens.merge` |
| Interactive slice/heatmap visualization (Figure 5) | `jlens.vis.compute_slice`, `build_page` |
| `LensModel` protocol (any model library) | `jlens/protocol.py`; HF adapter in `jlens/hf.py` |
| Lens-coordinate swap / steering / ablation (formulas: §2.5, Fig 4C) | **not yet implemented** in `jlens/lens.py` — buildable on `lens.jacobians[l]` + `W_U` + a write hook |
| Per-token probe `⟨v_t, h_l⟩` / sparse decomposition (gradient pursuit) | **not implemented**; §2.5, §4.2 |
| Template lens / oracle lens (multi-token) | **not implemented**; paper appendix §A.9 only |

## What's *not* covered in this summary

Appendix sections not detailed above (read the PDF directly if a task needs
them): §A.13 (per-trial swap *result* grids for flexible generalization — the
swap *formula* is §2.5, covered above; §A.13 only reports outcomes: country
facts 42/48 off-diagonal cells reach rank 1, months/animals/numbers
progressively worse, 0/48 for number relations at `α=1`), §A.15 (ignition
details), §A.16–17 (single-layer loading, dual-task competition), §A.20
(more Assistant-reaction examples), §A.21 (evaluation-awareness measurement
methodology), §A.22 (LLM-agent auditing system built on the J-lens), §A.23
(experiential-report grading rubrics), §A.24 (J-lens applied to mechanistic
interpretability: localization, attribution graphs, SAE/component
interpretation).
