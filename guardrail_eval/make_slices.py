#!/usr/bin/env python
"""Render jlens position x layer slice pages for selected guardrail_eval rows.

The audit loop (`run_audit_pipeline.py`) only ever reads the guardrail's
J-lens at a handful of (position, layer) points -- the anchor readout at the
decision position (-1), plus whatever positions the investigator agent
chooses to probe interactively via `get_jlens_readout`. Neither is a dense
map: they can't show *where in the prompt* a concept becomes legible, only
what's legible at the points someone (or something) decided to look.

This script fills that gap for a *small, selected* set of rows using
`jlens.vis.compute_slice` -- the full position x layer grid, plus rank
tracking for a pinned vocabulary of the guardrail's own malign/benign
verdict concepts (`CONCEPT_VOCAB` below, drawn from tokens observed in
smoke-test readouts -- English plus a few CJK variants the multilingual
guardrail also surfaces). Unlike the audit loop's per-point reads, this is
O(seq_len x n_layers) per row -- expensive
enough that it should never run over a full corpus. Use `--select` to pick
the handful of rows actually worth a dense look (see VISUALIZATION.md in
markdowns-de-referencia/ for the full rationale and selection criteria).

Reads `results_audit/audit_readouts_<attack>.jsonl` (written by
`run_audit_pipeline.py`), so run that first. Writes one `mode="fetch"` page
per selected row to `results_audit/slices/<attack>/<pool_index>/` -- open
`index.html` there via a local static server (e.g. `python -m http.server`
from that directory), since fetch-mode pages load `meta.json`/`slice.bin`
via `fetch()` and most browsers block that against a bare `file://` URL.

Example (after an audit run for the baseline-wrapping attack):
    python make_slices.py --attack baseline-wrapping --select errors --max-rows 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import jlens.vis as vis
from jlens.lens import JacobianLens

HERE = Path(__file__).resolve().parent

# Workspace band -- matches run_audit_pipeline.py's DEFAULT_LAYERS. L1-13
# were confirmed pure formatting noise in the Phase 1 smoke test.
DEFAULT_LAYERS = list(range(14, 27))

# Concept vocabulary the guardrail has been observed to surface around its
# malign/benign verdict (see results_audit/audit_readouts_*.jsonl smoke-test
# output). Pinned so compute_slice tracks their rank at *every* (position,
# layer) -- not just the cells where they happen to land in the top-N.
CONCEPT_VOCAB = [
    " malign", " benign", "malign", "benign",
    "INVALID", "invalid", "Invalid",
    "illegal", "unsafe", "danger", "dangerous", "harmful", "malignant",
    "违法行为", "违法", "无效", "危険", "危險",
]


def slug(attack: str) -> str:
    return attack.replace("-", "_")


def pinned_token_ids(tokenizer, words: list[str]) -> set[int]:
    """Token ids for ``words`` in the model's own vocabulary.

    A word may tokenize to more than one piece; every sub-token id gets
    pinned. Imprecise for multi-token words (each piece gets its own rank
    curve rather than the whole word being tracked as a unit), but harmless
    -- it just means a few extra pinned ids in the page.
    """
    ids: set[int] = set()
    for w in words:
        ids.update(tokenizer.encode(w, add_special_tokens=False))
    return ids


def narrow_lens(lens: JacobianLens, layers: list[int]) -> JacobianLens:
    """A copy of ``lens`` restricted to ``layers``.

    ``compute_slice`` always sweeps every layer in the lens it's given (plus
    the model's final layer) -- there's no ``layers=`` filter like
    ``JacobianLens.apply`` has. Subsetting the lens itself is how you bound
    that cost to the workspace band instead of all 27 fitted layers.
    """
    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"layers {missing} not fitted in this lens: {lens.source_layers}")
    return JacobianLens(
        jacobians={l: lens.jacobians[l] for l in layers},
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )


def load_rows(attack: str) -> list[dict]:
    path = HERE / "results_audit" / f"audit_readouts_{slug(attack)}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `run_audit_pipeline.py --attack {attack}` first"
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(
    rows: list[dict], select: str, pool_indices: list[int] | None
) -> list[dict]:
    if select == "pool-index":
        if not pool_indices:
            raise ValueError("--select pool-index requires --pool-index <id> [<id> ...]")
        wanted = set(pool_indices)
        return [r for r in rows if r["pool_index"] in wanted]
    if select == "errors":
        # FP/FN: guardrail's verdict disagrees with the dataset's true label.
        return [r for r in rows if r["label_true"] != r["guardrail_label_pred"]]
    if select == "unknown":
        # Wrapping-attack rows where the guardrail drifted off a clean
        # verdict entirely (ARCHITECTURE.md's "baseline-wrapping" finding).
        return [r for r in rows if r["guardrail_label_pred"] == "unknown"]
    raise ValueError(f"unknown --select {select!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--attack", choices=["baseline", "baseline-wrapping"], default="baseline"
    )
    parser.add_argument(
        "--select",
        choices=["errors", "unknown", "pool-index"],
        default="errors",
        help="'errors': label_true != guardrail_label_pred (FP/FN). "
        "'unknown': guardrail_label_pred == 'unknown' (persona-drift rows). "
        "'pool-index': explicit --pool-index list.",
    )
    parser.add_argument("--pool-index", type=int, nargs="+", default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Cap on rendered rows -- compute_slice is far more expensive per "
        "row than the audit loop's own point readouts; don't run this over a "
        "full corpus.",
    )
    parser.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layers to include in the slice (subsets the lens -- see narrow_lens).",
    )
    parser.add_argument(
        "--last-n-tokens",
        type=int,
        default=80,
        help="Only compute the slice grid for the trailing N prompt positions "
        "(the forward pass still sees the full prompt). Bounds cost on long "
        "wrapped/jailbreak prompts -- the window covers 'INPUT: {seed}...'. "
        "Pass 0 to disable and render every position.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda".')
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to results_audit/slices/<attack>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here (not at module scope) so --help doesn't require torch.
    from jlens_readout import GuardrailLens

    rows = load_rows(args.attack)
    selected = select_rows(rows, args.select, args.pool_index)[: args.max_rows]
    if not selected:
        print(f"no rows matched --select {args.select!r} in attack={args.attack!r}")
        return
    print(f"{len(selected)} row(s) selected (of {len(rows)} in the readouts file)")

    print(f"loading guardrail {args.guardrail_model} (device={args.device}) ...")
    dtype = getattr(torch, args.dtype)
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)
    lens = narrow_lens(gl.lens, args.layers)
    pins = pinned_token_ids(gl.tok, CONCEPT_VOCAB)
    print(f"{len(pins)} pinned concept token ids (from {len(CONCEPT_VOCAB)} words)")

    out_root = (
        Path(args.out_dir) if args.out_dir else HERE / "results_audit" / "slices" / slug(args.attack)
    )
    out_root.mkdir(parents=True, exist_ok=True)
    last_n = args.last_n_tokens or None

    for row in selected:
        pool_index = row["pool_index"]
        prompt = gl.chat_prompt(row["prompt"])
        print(
            f"[{args.attack}] pool_index={pool_index} "
            f"label_true={row['label_true']} pred={row['guardrail_label_pred']} "
            "-- computing slice ..."
        )
        slice_data = vis.compute_slice(
            gl.model,
            lens,
            prompt,
            top_n=args.top_n,
            pinned_token_ids=pins,
            last_n_tokens=last_n,
            max_seq_len=args.max_seq_len,
        )
        out_dir = out_root / str(pool_index)
        vis.build_page(
            slice_data,
            prompt,
            title=f"{args.attack} #{pool_index} "
            f"({row['label_true']} -> {row['guardrail_label_pred']})",
            description=f"category={row['category']}; seed={row['seed'][:80]!r}",
            mode="fetch",
            out_dir=out_dir,
        )
        print(f"  -> {out_dir}/")

    print(
        f"\ndone. Serve a slice with e.g.:\n"
        f"  python -m http.server --directory {out_root / str(selected[0]['pool_index'])} 8000"
    )


if __name__ == "__main__":
    main()
