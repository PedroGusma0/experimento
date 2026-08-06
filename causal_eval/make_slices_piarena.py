#!/usr/bin/env python
"""Render jlens position x layer slice pages for selected causal pipeline v3
(PIArena-based) rows.

Mirrors `guardrail_eval/make_slices.py` (v1/v2's version of this tool) but
adapted to v3's shape: `run_causal_pipeline_piarena.py` only ever reads the
guardrail's J-lens through the dense per-position ablation sweep -- which
only runs at all when the verdict is `malign` (see ARCHITECTURE.md,
"Planned -- causal pipeline v3", Decision 1) -- so `benign`/`unknown` rows
have *no* J-lens data of their own. This script fills that gap for a small,
selected set of rows using `jlens.vis.compute_slice` -- the full position x
layer grid, regardless of verdict -- most useful for exactly the `unknown`
rows the sweep never touches (the guardrail failing to emit a clean verdict
at all) but works for any row.

Unlike v1/v2's `audit_readouts_<attack>.jsonl` (which embeds the prompt
text directly), v3's `causal_readouts_<config>_<variant>.jsonl` only stores
`sample_index`/`config`/`variant` -- so this script joins back to
`piarena_eval/data/{config}_{variant}.csv` to recover `target_inst`/
`context` before re-rendering via `chat_prompt_v3`.

Reads `causal_eval/results_causal_piarena/causal_readouts_<config>_<variant>.jsonl`
(written by `run_causal_pipeline_piarena.py`) and
`piarena_eval/data/<config>_<variant>.csv` (written by
`prepare_piarena_data.py`), so run both first. Writes one `mode="fetch"`
page per selected row to
`causal_eval/results_causal_piarena/slices/<config>_<variant>/<sample_index>/`
-- open `index.html` there via a local static server (fetch-mode pages load
`meta.json`/`slice.bin` via `fetch()`, which most browsers block against a
bare `file://` URL).

Example (after a run for dolly_closed_qa/direct):
    python causal_eval/make_slices_piarena.py --config dolly_closed_qa \
        --variant direct --select unknown --max-rows 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "guardrail_eval"), os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jlens.vis as vis  # noqa: E402

# Reused as-is from v1/v2's tool -- these are generic, not schema-specific:
# CONCEPT_VOCAB (malign/benign-adjacent tokens observed in guardrail
# readouts), pinned_token_ids, narrow_lens, DEFAULT_LAYERS (the Qwen3-1.7B
# workspace band -- a different guardrail, e.g. gemma-4-e4b, needs its own
# band re-derived; see ARCHITECTURE.md).
from make_slices import CONCEPT_VOCAB, DEFAULT_LAYERS, narrow_lens, pinned_token_ids  # noqa: E402

HERE = Path(__file__).resolve().parent
_RESULTS_DIR = HERE / "results_causal_piarena"
_DATA_DIR = Path(_REPO_ROOT) / "piarena_eval" / "data"


def load_rows(config: str, variant: str) -> list[dict]:
    path = _RESULTS_DIR / f"causal_readouts_{config}_{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `run_causal_pipeline_piarena.py "
            f"--config {config} --variant {variant}` first"
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def attach_prompt_fields(rows: list[dict], config: str, variant: str) -> list[dict]:
    """Join each readout row back to its `target_inst`/`context`/`category`
    -- `causal_readouts_*.jsonl` doesn't store the prompt text itself, only
    `sample_index` (see module docstring)."""
    csv_path = _DATA_DIR / f"{config}_{variant}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run `prepare_piarena_data.py "
            f"--config {config}` first"
        )
    df = pd.read_csv(csv_path).set_index("sample_index")
    merged = []
    for row in rows:
        src = df.loc[int(row["sample_index"])]
        out = dict(row)
        out["target_inst"] = src["target_inst"]
        out["context"] = src["context"]
        out["category"] = src["category"]
        merged.append(out)
    return merged


def select_rows(
    rows: list[dict],
    select: str,
    sample_indices: list[int] | None,
    case: str | None,
) -> list[dict]:
    if select == "sample-index":
        if not sample_indices:
            raise ValueError("--select sample-index requires --sample-index <id> [<id> ...]")
        wanted = set(sample_indices)
        return [r for r in rows if r["sample_index"] in wanted]
    if select == "unknown":
        # The rows the causal sweep never touches at all (gating only fires
        # on `malign`) -- the guardrail failing to emit a clean verdict is
        # exactly the case this script exists to make investigable.
        return [r for r in rows if r["verdict"] == "unknown"]
    if select == "malign":
        # Rows the sweep already covers positionally -- compute_slice gives
        # the dense grid around what the sweep only sampled at a few
        # candidate tokens per position.
        return [r for r in rows if r["verdict"] == "malign"]
    if select == "benign":
        return [r for r in rows if r["verdict"] == "benign"]
    if select == "case":
        if not case:
            raise ValueError("--select case requires --case <e.g. attacked+malign>")
        return [r for r in rows if r["case"] == case]
    raise ValueError(f"unknown --select {select!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="dolly_closed_qa")
    parser.add_argument("--variant", choices=["clean", "direct"], default="direct")
    parser.add_argument(
        "--select",
        choices=["unknown", "malign", "benign", "case", "sample-index"],
        default="unknown",
        help="'unknown': verdict == 'unknown' (no sweep data exists for these -- "
        "the main use case). 'malign'/'benign': by verdict. 'case': exact "
        "--case string (e.g. attacked+malign). 'sample-index': explicit "
        "--sample-index list.",
    )
    parser.add_argument("--sample-index", type=int, nargs="+", default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Cap on rendered rows -- compute_slice is far more expensive per "
        "row than the causal sweep's own candidate reads; don't run this over "
        "a full corpus.",
    )
    parser.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layers to include in the slice (subsets the lens -- see narrow_lens). "
        "Default is Qwen3-1.7B's workspace band; re-derive for other guardrails.",
    )
    parser.add_argument(
        "--last-n-tokens",
        type=int,
        default=150,
        help="Only compute the slice grid for the trailing N prompt positions "
        "(the forward pass still sees the full prompt). PIArena prompts run "
        "longer than guardrail_eval's raw seeds (context + injected_task), so "
        "this defaults higher than make_slices.py's 80. Pass 0 to disable and "
        "render every position.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda".')
    # bfloat16 default (not make_slices.py's float32): this machine's ~7.7GB
    # RAM is tight (Qwen3-1.7B fp32 ~= 6.8GB, see ARCHITECTURE.md's OOM
    # notes) -- bf16 halves the footprint, matching run_causal_pipeline*.py's
    # own default for the same reason.
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to results_causal_piarena/slices/<config>_<variant>/",
    )
    return parser.parse_args()


def _write_landing_page(out_root: Path, rendered: list[dict]) -> None:
    """A plain listing page at `out_root/index.html`: one row per rendered
    prompt, linking to its own `<sample_index>/index.html` slice page. Lets
    you pick a prompt to open its navigable J-lens view from a single served
    directory, instead of tracking `sample_index`es and serving/opening each
    row's folder by hand."""

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    rows_html = "\n".join(
        f"<tr><td><a href=\"{r['sample_index']}/index.html\">{r['sample_index']}</a></td>"
        f"<td>{esc(r['verdict'])}</td><td>{esc(r['case'])}</td>"
        f"<td>{esc(r['category'])}</td>"
        f"<td>{esc(r['target_inst'][:100])}</td></tr>"
        for r in rendered
    )
    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>{esc(out_root.name)} slices</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  tr:hover {{ background: #f7f7f7; }}
  a {{ text-decoration: none; color: #06c; }}
</style>
</head><body>
<h1>{esc(out_root.name)} &mdash; {len(rendered)} prompt(s)</h1>
<p>Click a <code>sample_index</code> to open its navigable J-lens slice.</p>
<table>
<tr><th>sample_index</th><th>verdict</th><th>case</th><th>category</th><th>target_inst</th></tr>
{rows_html}
</table>
</body></html>"""
    (out_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    # Imported here (not at module scope) so --help doesn't require torch.
    from jlens_readout import GuardrailLens

    rows = load_rows(args.config, args.variant)
    rows = attach_prompt_fields(rows, args.config, args.variant)
    selected = select_rows(rows, args.select, args.sample_index, args.case)[: args.max_rows]
    if not selected:
        print(f"no rows matched --select {args.select!r} in "
              f"config={args.config!r} variant={args.variant!r}")
        return
    print(f"{len(selected)} row(s) selected (of {len(rows)} in the readouts file)")

    print(f"loading guardrail {args.guardrail_model} (device={args.device}) ...")
    dtype = getattr(torch, args.dtype)
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)
    lens = narrow_lens(gl.lens, args.layers)
    pins = pinned_token_ids(gl.tok, CONCEPT_VOCAB)
    print(f"{len(pins)} pinned concept token ids (from {len(CONCEPT_VOCAB)} words)")

    out_root = (
        Path(args.out_dir) if args.out_dir
        else _RESULTS_DIR / "slices" / f"{args.config}_{args.variant}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    last_n = args.last_n_tokens or None

    for row in selected:
        sample_index = row["sample_index"]
        prompt = gl.chat_prompt_v3(row["target_inst"], row["context"])
        print(
            f"[{args.config}/{args.variant}] sample_index={sample_index} "
            f"verdict={row['verdict']} case={row['case']} -- computing slice ..."
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
        out_dir = out_root / str(sample_index)
        page, _, _ = vis.build_page(
            slice_data,
            prompt,
            title=f"{args.config}/{args.variant} #{sample_index} "
            f"(verdict={row['verdict']}, case={row['case']})",
            description=f"category={row['category']}; target_inst={row['target_inst'][:80]!r}",
            mode="fetch",
            out_dir=out_dir,
        )
        # build_page(mode="fetch") only writes the sidecar data files
        # (meta.json/slice.bin/ranks/); the HTML shell itself is the
        # returned string and must be written separately (see
        # walkthrough.ipynb's cell 5) -- easy to miss, and
        # guardrail_eval/make_slices.py's own copy of this loop does miss it.
        (out_dir / "index.html").write_text(page, encoding="utf-8")
        print(f"  -> {out_dir}/index.html")

    _write_landing_page(out_root, selected)
    print(
        f"\nwrote landing page: {out_root}/index.html\n"
        f"done. Serve it with e.g.:\n"
        f"  python -m http.server --directory {out_root} 8000\n"
        f"then open http://localhost:8000/ and click a sample_index."
    )


if __name__ == "__main__":
    main()
