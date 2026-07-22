#!/usr/bin/env python
"""Phase 1: Qwen3-1.7B guardrail + J-lens readouts over the HarmBench seeds.

For each seed the guardrail reads, this classifies it (malign/benign) and saves
the Jacobian-lens readout at the decision position — the top-K vocabulary
tokens the model's internal activations are disposed to verbalize at each
layer. That readout is the XAI signal: it shows which internal concepts drive
the malign/benign verdict, on raw (unmutated) seeds, before any target model.

Outputs:
- results/readouts_qwen3_1.7b.jsonl  (one JSON object per prompt)
- results/summary_qwen3_1.7b.csv     (seed, category, label true/prediction)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from jlens_readout import GuardrailLens

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(HERE / "data" / "harmbench_labeled.csv"))
    parser.add_argument(
        "--out-jsonl", default=str(HERE / "results" / "readouts_qwen3_1.7b.jsonl")
    )
    parser.add_argument(
        "--out-csv", default=str(HERE / "results" / "summary_qwen3_1.7b.csv")
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to read the lens at; default = every fitted layer.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument("--limit", type=int, default=None, help="First N rows only.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-row label + raw text."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    dtype = getattr(torch, args.dtype)
    print(f"loading guardrail {args.model} + Jacobian lens (dtype={args.dtype}) ...")
    gl = GuardrailLens(args.model, dtype=dtype)
    print(f"  {gl.model}")
    print(f"  {gl.lens}")

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t0 = time.time()
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, row in tqdm(
            df.iterrows(), total=len(df), desc="guardrail+jlens"
        ):
            seed = row["prompt"]
            prompt_str = gl.chat_prompt(seed)
            label_pred, raw = gl.classify(
                prompt_str, max_new_tokens=args.max_new_tokens
            )
            readout = gl.readout(prompt_str, layers=args.layers, top_k=args.top_k)
            if args.verbose:
                print(f"\nseed: {seed[:80]!r}\nraw: {raw!r}\nparsed: {label_pred}")

            record = {
                "index": int(i),
                "seed": seed,
                "category": row.get("category", ""),
                "label_true": row["label"],
                "label_pred": label_pred,
                "jlens": {str(layer): toks for layer, toks in readout.items()},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary_rows.append(
                {
                    "seed": seed,
                    "category": row.get("category", ""),
                    "label (true)": row["label"],
                    "label (prediction)": label_pred,
                }
            )
    elapsed = time.time() - t0

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_csv, index=False)

    accuracy = (summary["label (true)"] == summary["label (prediction)"]).mean()
    n_unknown = (summary["label (prediction)"] == "unknown").sum()
    print(
        f"n={len(summary)}  accuracy={accuracy:.3f}  unknown={n_unknown}  "
        f"elapsed={elapsed:.1f}s\n  readouts -> {out_jsonl}\n  summary  -> {out_csv}"
    )


if __name__ == "__main__":
    main()
