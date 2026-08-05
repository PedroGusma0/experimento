#!/usr/bin/env python
"""Quick smoke test for pipeline v4 (see ARCHITECTURE.md, "Planned --
pipeline v4"): PIArena Direct attack -> target model (jlens-wrapped) ->
J-lens readout at the decision position -> ASR/Utility LLM-judge
(openai/gpt-oss-20b via Groq, network-only -- see judge.py). No guardrail
anywhere in this path -- the target model IS the thing being interpreted.

NOT a scientific run -- confirms the v4 wiring works end-to-end on a
handful of rows. Deliberately out of scope here (see ARCHITECTURE.md's
still-open questions on pipeline v4): the causal ablation sweep on the
target (causal_eval/causal_sweep.py's score_fn contract assumes one scalar
read at one decision position from vocab logits, not an open-ended
generation -- undecided how to adapt it), corpus-scale runs, --resume, and
the remaining PIArena/ROC-AUC/WRS/AIM/Toxicity metrics.

Setup (once, into the shared guardrail_eval/.venv):
    guardrail_eval\\.venv\\Scripts\\python.exe -m pip install openai groq python-dotenv pydantic

Run (from repo root):
    guardrail_eval\\.venv\\Scripts\\python.exe target_audit_eval\\run_smoke_test.py \\
        --variant direct --n-samples 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "guardrail_eval"), os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from target_lens import TargetLens  # noqa: E402
from judge import score_sample, JUDGE_MODEL  # noqa: E402

_DATA_DIR = os.path.join(_REPO_ROOT, "piarena_eval", "data")
_OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    # bfloat16 default, not float32: Qwen3-1.7B in fp32 is ~6.8GB, right at
    # this machine's ~7.7GB RAM ceiling (see ARCHITECTURE.md) -- every other
    # local CPU driver in this repo (causal_eval/run_causal_pipeline_piarena.py)
    # defaults to bfloat16 for the same reason. float32 loading segfaulted
    # partway through weight-loading in this session's first smoke attempt.
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--config", default="dolly_closed_qa")
    p.add_argument("--variant", default="direct", choices=["clean", "direct", "both"])
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--layer-lo", type=int, default=14)  # workspace band lo (Qwen3)
    p.add_argument("--layer-hi", type=int, default=26)  # workspace band hi (Qwen3)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(_OUT_DIR, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading target model+lens ({args.target_model}, {args.dtype}, {args.device})...")
    tl = TargetLens(model_id=args.target_model, dtype=dtype, device=args.device)
    band = [l for l in tl.lens.source_layers if args.layer_lo <= l <= args.layer_hi]
    print(f"workspace band: {band}")
    print(f"judge model (Groq, API-only): {args.judge_model}")

    variants = ["clean", "direct"] if args.variant == "both" else [args.variant]
    out_path = args.out or os.path.join(
        _OUT_DIR, f"smoke_{args.config}_{args.variant}.jsonl"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for variant in variants:
            csv_path = os.path.join(_DATA_DIR, f"{args.config}_{variant}.csv")
            df = (
                pd.read_csv(csv_path)
                .sort_values("sample_index")
                .head(args.n_samples)
                .reset_index(drop=True)
            )
            for _, row in df.iterrows():
                t0 = time.time()
                prompt = tl.render_prompt(row["target_inst"], row["context"])
                response = tl.generate(prompt, max_new_tokens=args.max_new_tokens)
                readout = tl.readout(prompt, layers=band, position=-1, top_k=args.top_k)
                injected_task = row["injected_task"] if variant == "direct" else None
                scores = score_sample(
                    target_inst=row["target_inst"],
                    injected_task=injected_task,
                    response=response,
                    model=args.judge_model,
                )
                record = {
                    "sample_index": int(row["sample_index"]),
                    "variant": variant,
                    "target_model": args.target_model,
                    "target_inst": row["target_inst"],
                    "injected_task": injected_task,
                    "response": response,
                    "readout": readout,
                    **scores,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{variant} #{row['sample_index']}] "
                    f"utility={scores['utility_completes']} asr={scores['asr_completes']} "
                    f"({record['elapsed_s']}s)"
                )
                if scores["utility_error"] or scores["asr_error"]:
                    print(
                        f"  judge error: utility={scores['utility_error']!r} "
                        f"asr={scores['asr_error']!r}"
                    )
                print(f"  response: {response[:200]!r}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
