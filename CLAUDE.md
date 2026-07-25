# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`jlens` is the reference implementation for the Jacobian lens, a technique for
reading out what an internal residual-stream activation of a decoder
transformer is disposed to make the model say. It linearly transports a
residual vector at layer `l` into the final-layer basis using the average
input-output Jacobian `J_l = E[∂h_final / ∂h_l]`, then decodes it with the
model's own unembedding. Companion code for the paper *Verbalizable
Representations Form a Global Workspace in Language Models*. Not accepting
contributions.

**Before doing any conceptual/research task involving the J-lens or the
paper's findings, read [`PAPER_SUMMARY.md`](PAPER_SUMMARY.md) instead of
`artigo_principal.pdf`.** It's a full section-by-section summary (methods,
formula, pseudocode, every experimental result, terminology, limitations,
and a table mapping paper concepts to exact code locations) distilled from
the 117-page PDF, which is too large to re-read on every task. Only open the
PDF directly for a verbatim quote, a figure, or an appendix subsection the
summary explicitly flags as uncovered.

This repo also hosts an applied experiment, `guardrail_eval/`, that is not
part of the `jlens` library or the paper — see
[`guardrail_eval/ (experiment)`](#guardrail_eval-experiment) below.

## Commands

Install (editable, with dev deps):

```bash
pip install -e ".[dev]"
```

Run tests (CPU-only; no GPU/model download needed — tests use the
`TinyDecoder` toy model in `tests/tiny.py`):

```bash
pytest
pytest tests/test_fitting.py::test_fit_and_apply_tiny  # single test
```

Lint:

```bash
ruff check .
```

The project uses `uv.lock`; if using `uv`, prefer `uv run pytest` / `uv sync`
over invoking `pip`/`pytest` directly.

## Architecture

The package is small and each module has one job; read `jlens/protocol.py`
first since everything else is written against it.

- **`protocol.py`** — `LensModel`, a `Protocol` defining the four things a
  model must provide: `n_layers`, `d_model`, `layers` (indexable residual
  blocks), `encode`, `forward` (residual stack only, no LM head, must build
  an autograd graph through `layers`), and `unembed` (final norm + LM head).
  Fitting and application are written entirely against this interface, so any
  model library can plug in — `hf.py` is the HuggingFace adapter, and
  `tests/tiny.py` is a minimal from-scratch example used by the test suite.

- **`hf.py`** — `from_hf(hf_model, tokenizer)` wraps an already-loaded HF
  `*ForCausalLM` as a `LensModel`. It auto-detects where the residual blocks,
  final norm, embedding, and LM head live via a table of known `Layout`s
  (`_LAYOUTS`, covering Llama/Qwen/Mistral/Gemma/OLMo/StableLM/Phi/GPT-2/GPT-NeoX
  and their multimodal wrappers); pass `layout=` explicitly for anything else.
  Construction mutates the model in place (freezes all params, optionally
  `torch.compile`s each block individually so `ActivationRecorder` hooks still
  fire per-block).

- **`hooks.py`** — `ActivationRecorder` is a context manager that registers
  forward hooks on requested block indices and captures their output tensors
  (undetached, so they can feed `torch.autograd.grad` directly). Its
  `start_graph_at` param makes the captured tensor at that layer the autograd
  leaf/root — needed because model params have `requires_grad=False`, so
  without this the graph wouldn't exist at all.

- **`fitting.py`** — `jacobian_for_prompt` computes `J_l` for one prompt: one
  forward pass (input replicated `dim_batch` times along the batch axis) then
  `ceil(d_model / dim_batch)` backward passes, each injecting a one-hot
  cotangent at `dim_batch` output dims simultaneously at every valid target
  position, and averaging the resulting gradient over source positions. The
  estimator and its rationale are documented in the module docstring — read
  it before touching this function, the reduction order (sum over target
  positions, then mean over source positions) is deliberate and matches the
  paper. `fit()` drives this over a prompt list, accumulating a running mean
  with optional atomic checkpoint/resume (tracks `next_idx` separately from
  `n_done` so a skipped too-short prompt isn't reprocessed on resume). Shard
  a large corpus across machines by fitting disjoint slices and combining
  with `JacobianLens.merge`.

- **`lens.py`** — `JacobianLens` holds the fitted `{layer: J_l}` dict and is
  the load-bearing public API: `save`/`load`/`from_pretrained` (local file,
  local dir, or HF Hub repo — `filename=` lets one Hub repo host lenses for
  many models), `merge` (n_prompts-weighted mean of disjoint-corpus lenses),
  `transport` (bare `J_l @ h`), and `apply` (run the model, read out
  requested layers/positions, transport, unembed). `apply(..., use_jacobian=False)`
  gives the vanilla logit-lens baseline (skip the Jacobian transport) for
  comparison.

- **`vis.py`** — Renders the interactive layer × position slice view.
  `compute_slice` runs two passes over the model's activations (top-K tokens
  per cell, then full rank tracking for a chosen token subset — kept separate
  because retaining logits for every layer at once would blow up memory at
  long seq_len × vocab × n_layers) and returns a `SliceData`. `build_page`
  renders it into HTML two ways: `mode="embed"` (single self-contained file,
  data and d3 both inlined — use for notebooks via `notebook_iframe`, but
  avoid for long prompts since rank data dominates payload size) or
  `mode="fetch"` (writes `meta.json`/`slice.bin`/`ranks/{tid}.bin` sidecars
  for static hosting, d3 from CDN with SRI pinning). The HTML/JS template
  itself lives in `jlens/data/slice_vis.html`, shipped as package data.

- **`examples.py`** — Curated example prompts (`EXAMPLES`) used by the
  walkthrough notebook and a WikiText-103 streaming loader for fitting
  corpora. `resolve_prompt` handles both raw-text and chat-template
  (`system`/`user`/`assistant_prefill`) example forms.

### Data flow

`fit(model, prompts)` → `JacobianLens` (holds `J_l` per layer) →
`lens.apply(model, prompt)` → `(lens_logits, model_logits, input_ids)` →
`vis.compute_slice(...)` → `vis.build_page(...)` → HTML slice page.

### `data/` (package-external, not `jlens/data/`)

Synthetic prompt sets (JSON, authored by Anthropic) used in the paper's
experiments and lens-quality evaluations — not code. `data/experiments/`
and `data/evaluations/` each have their own README documenting the
conventions shared across files (lens readout, workspace band, hit/swap
definitions) and what each `{slug}.json` contains. Consult those READMEs
before writing code that consumes these files — the schema differs per file
and is documented there, not in the JSON itself.

## Working in this codebase

- `tests/tiny.py`'s `TinyDecoder` is the reference "any model works" example:
  an 8-dim, 4-layer, CPU-only decoder. When adding lens-core functionality,
  add or extend a test against it rather than requiring a real HF model
  download — the whole test suite runs CPU-only and offline.
- Layer indices are negative-friendly throughout the public API
  (`source_layers`, `target_layer`, `positions`); `_check_layer_indices` in
  `fitting.py` is the shared normalization/validation logic for layers,
  mirrored by `apply`'s own checks in `lens.py`.
- `SKIP_FIRST_N_POSITIONS` (fitting.py) excludes early attention-sink
  positions from the Jacobian average — don't casually change this default,
  it's a documented modeling choice, not a tunable.

## `guardrail_eval/` (experiment)

`guardrail_eval/` is a research **experiment**, not part of the `jlens`
library or the paper it accompanies — it only ever imports `jlens` as a
library (`pip install -e ..`); nothing under `jlens/` is modified for it.

It exists to answer an open question in a separate, larger project (the
"main project": an RL-based attacker hammering a guardrail in front of a
target LLM, where the guardrail is an autoencoder — `Ataque (RL Hammer) →
Guardrail (Autoencoder) → XAI (?) → LLM Alvo`). That project has no answer
yet for what XAI method should explain the guardrail's decisions. Since the
Jacobian lens reads out the residual stream of a decoder transformer, it
can't be applied directly to an autoencoder-based guardrail — so this
experiment inserts an additional, LLM-based guardrail (Qwen3-1.7B) purely
so the J-lens has something to read, then tests whether J-lens readouts are
a viable, *automatable* XAI method by running them through an automated
audit loop (an investigator agent that reads the lens output, scored by an
LLM-judge — modeled on the paper's Appendix A.22 auditor). The goal is to
validate this approach here, at small scale, before committing to it for
the main project's `XAI (?)` gap.

See [`markdowns-de-referencia/ARCHITECTURE.md`](markdowns-de-referencia/ARCHITECTURE.md) for
the full pipeline (phases 0–3: baseline, guardrail + J-lens, attack + target,
automated auditor),
[`guardrail_eval/PLAN_runpod_audit.md`](guardrail_eval/PLAN_runpod_audit.md)
for the GPU execution plan, and
[`markdowns-de-referencia/VISUALIZATION.md`](markdowns-de-referencia/VISUALIZATION.md)
for how `jlens.vis`'s position × layer slice pages apply to this experiment
(`guardrail_eval/make_slices.py`) — don't duplicate that detail here. Status:
phases 0–2 are validated via local CPU smoke tests; phase 3 (the automated
auditor) is implemented but not yet run at scale (pending RunPod GPU
execution).
