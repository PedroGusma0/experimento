# guardrail_eval — J-lens visualization (reference)

What `jlens.vis` offers, what it's for in `guardrail_eval/`, and how to use
`make_slices.py`. This is a companion to `ARCHITECTURE.md`, not a
replacement — read that first for the pipeline as a whole.

## Why this exists

The audit loop (`run_audit_pipeline.py`) reads the guardrail's J-lens at
individual `(position, layer)` points, never a dense map:

- `GuardrailLens.readout()` — one point, always position `-1` (the decision
  position). Logged unconditionally per row as the `jlens` field of
  `audit_readouts_<attack>.jsonl`.
- `GuardrailLens.readout_multi()` — several points, but chosen live by the
  investigator agent via the `get_jlens_readout` tool (up to
  `--max-tool-calls`, default 5). Sparse and reasoning-driven, not
  exhaustive — see `audit_agent.py::investigate`.

Neither can answer "where in the prompt does the guardrail's malign/benign
concept actually become legible, and is that inside the injected seed or
somewhere in the wrapper template?" — that needs every position, not the
ones an agent happened to pick. `jlens.vis.compute_slice` computes exactly
that: a full position x layer grid for one prompt.

## What `jlens.vis` provides

`compute_slice(model, lens, prompt, ...)` → `SliceData` → `build_page(...)`
→ an interactive HTML page. See `jlens/vis.py` for the implementation;
summary of what the page shows:

| Panel | Data source | Shows |
|---|---|---|
| Top-1 grid (position x layer) | `SliceData.top_ids[:, :, 0]` | Which token the lens decodes to at every cell |
| Cell popup (click a cell) | `top_ids`/`top_ranks` at that cell | Top-N tokens + scores at one `(position, layer)` |
| "By layer" column | vertical slice at fixed position | Concept trajectory through the network at one prompt position |
| "By position" row | horizontal slice at fixed layer | Where in the prompt a concept appears, at one layer |
| Rank heatmap | `SliceData.rank_tensor` (pinned tokens only) | Full-grid rank of a *chosen* token — needs `pinned_token_ids` |
| Rank-vs-layer / rank-vs-position line charts | same `rank_tensor` | 1-D cuts of the rank heatmap for pinned tokens |

The rank heatmap and line charts only exist for **pinned** tokens (`pinned_token_ids`)
— compute_slice tracks full rank (not just top-N membership) only for those,
to keep memory bounded (`vis.py`'s two-pass design, see its module docstring).

## Cost — why this is selective, not part of the main loop

`compute_slice` unembeds **every requested position, at every requested
layer**, twice (once for the top-K pass, once for the pinned-token rank
pass, which also runs a full-vocab `argsort` per position — see
`_ranks_of` in `vis.py`). That's `O(seq_len x n_layers)`, against
`readout()`'s `O(1)` and `readout_multi()`'s `O(len(positions))`. On a long
`baseline-wrapping` prompt (seed wrapped in one of the 18 jailbreak
templates in `data/system_variants_en.csv`), `seq_len` can be several times
longer than the raw seed.

Moving the guardrail to a RunPod GPU (see `PLAN_runpod_audit.md`) makes this
much cheaper in absolute terms than the original CPU-only design assumed,
but it should still never run over a full corpus — the audit loop's own
API-bound cost (180-420 calls for the 60-row main run) is the actual budget
constraint on N; `compute_slice` adds GPU/wall-clock cost on top of that per
row it's run on. Treat it as a **diagnostic tool for a hand-picked subset**,
not a per-row step.

## Picking which rows are worth a dense look

`make_slices.py`'s `--select` implements three criteria, reading
`audit_readouts_<attack>.jsonl` (written by `run_audit_pipeline.py`):

- **`errors`** (default) — `label_true != guardrail_label_pred`: false
  positives/negatives. The most direct "why did the guardrail get this
  wrong" question.
- **`unknown`** — `guardrail_label_pred == "unknown"`: rows where a
  wrapping template's own jailbreak-persona instructions competed with the
  classifier system prompt and the guardrail drifted into roleplay instead
  of a clean verdict (documented in `ARCHITECTURE.md`'s Phase 2 findings).
  Localizing *where* that drift starts is a natural use of the position
  axis.
- **`pool-index`** — an explicit list, e.g. rows where the investigator and
  judge disagreed (`audit_scores_<attack>.jsonl`, low `correctness`) or
  where `fallback_used=true` (tool-calling failed and the investigator fell
  back to a single fixed readout).

## `make_slices.py`

```
python make_slices.py --attack baseline-wrapping --select errors --max-rows 5 \
    --device cuda --dtype bfloat16
```

What it does:

1. Loads `results_audit/audit_readouts_<attack>.jsonl`, filters rows by
   `--select`, caps at `--max-rows`.
2. Loads the guardrail (`GuardrailLens`, same class the audit loop uses).
3. Builds a **narrowed lens** restricted to `--layers` (default the
   workspace band L14-26) via `narrow_lens()` — `compute_slice` has no
   `layers=` filter of its own (unlike `JacobianLens.apply`), so bounding
   the layer sweep means subsetting the lens object itself before passing
   it in.
4. Pins a fixed vocabulary of the guardrail's own verdict concepts
   (`CONCEPT_VOCAB` — `"malign"`, `"benign"`, `"INVALID"`, `"illegal"`,
   `"unsafe"`, plus the CJK variants seen in smoke-test readouts, since the
   guardrail is multilingual) so the rank heatmap and line charts are
   populated without depending on what happens to land in a cell's top-N.
5. Renders each selected row with `--last-n-tokens` (default 80) windowing
   the slice grid to the tail of the prompt — the forward pass still sees
   the full prompt, only the rendered grid is windowed, which is what keeps
   cost bounded on long wrapped prompts. The window covers
   `"INPUT: {seed}...Classification:"`, mirroring
   `run_audit_pipeline.py`'s `--token-span-last-n`.
6. Writes one `mode="fetch"` page per row to
   `results_audit/slices/<attack>/<pool_index>/`.

`mode="fetch"` (not `"embed"`) on purpose: it writes `meta.json` +
`slice.bin` + `ranks/*.bin` sidecars per row instead of one large
self-contained HTML per row — better suited to browsing a batch of selected
prompts than a single-file notebook export, and avoids the payload blow-up
`embed` mode gets on long wrapped prompts (see `build_page`'s docstring in
`jlens/vis.py`). Serve a page locally:

```
python -m http.server --directory results_audit/slices/baseline_wrapping/<pool_index> 8000
```

(Fetch-mode pages load their data via `fetch()`, which most browsers block
against a bare `file://` URL — hence the local server.)

## `make_report.py` — consolidated XAI report (Markdown + PDF)

`make_slices.py` produces one interactive page per row. `make_report.py`
consolidates the auditor's own aggregate metrics *and* a handful of J-lens
slice snapshots into a single static document — `guardrail_eval/docs/xai_report.md`
and `guardrail_eval/docs/xai_report.pdf` — covering `baseline`,
`baseline-wrapping`, and their comparison in one file.

**Standalone by design**: it only reads already-written
`audit_readouts_<attack>.jsonl` / `audit_scores_<attack>.jsonl` /
`audit_summary_<attack>.csv`, so nothing it does — including a crash — can
affect a `run_audit_pipeline.py` run in progress or already on disk.

Two independent phases:

- **Phase 1 (always runs, no model load)**: charts straight from the
  auditor's output — guardrail classification outcome (TP/FP/FN/TN/unknown),
  per-claim investigator accuracy, judge score distribution, investigator
  tool-call usage (`tool_calls`/`fallback_used`). An attack missing from disk
  gets a "skipped" note, not a fatal error.
- **Phase 2 (`--skip-slices` to disable)**: loads the guardrail
  (`GuardrailLens`, same as the audit loop) and renders a static PNG rank
  heatmap — the pinned-concept-vocabulary version of `make_slices.py`'s
  interactive rank heatmap, collapsed to the *best* (lowest) rank across
  `CONCEPT_VOCAB` at each `(position, layer)` cell — for up to
  `--max-slice-rows` error rows per attack (reuses `make_slices.py`'s
  `select_rows`/`narrow_lens`/`pinned_token_ids`). Each row runs in its own
  `try`/`except` so one bad prompt doesn't blank the rest of the report.

```
python make_report.py --max-slice-rows 3 --device cuda --dtype bfloat16
python make_report.py --skip-slices                       # metrics only, no GPU
python make_report.py --results-dir /path/to/results_audit --docs-dir /path/to/docs
```

PDF is assembled natively in Python (`reportlab`, no external `pandoc`
dependency needed on the pod) from the same PNGs the Markdown report embeds.
New dependencies: `matplotlib`, `reportlab` (in `requirements.txt`).

## Not yet built

- **Coverage comparison**: does the investigator's sparse `readout_multi`
  probing (logged only as a `tool_calls` count + `evidence` text in
  `audit_scores_<attack>.jsonl`) actually land near where `compute_slice`'s
  dense map shows the real concept peak? Would need `investigate()` to log
  the actual `(positions, layers)` args of each tool call (currently only
  the count is recorded) before this comparison is possible. This is also
  the "infrastructure to observe convergence" the `--max-tool-calls`
  docstring flags as missing for tuning that default.
- **Emergence-position ground truth**: a `ground_truth.py` `Claim` computed
  directly from a pinned-token rank sweep (e.g. "first position where a
  malign-concept token's rank drops below N in the workspace band") — the
  two categories from `automacao_auditoria_jlens.md` ("momento de
  emergência", "deliberação interna") that `ground_truth.py` currently
  leaves out because they need a genuine sweep, not an agent's sparse
  choices, as their gabarito.
- No batch index page across a `--select` run's output directories (each
  row's page is independent; there's no linking `index.html` across rows
  yet).
