#!/usr/bin/env python
"""Causal pipeline v3 -- driver (see ARCHITECTURE.md, "Planned -- causal
pipeline v3").

Wires the guardrail (`GuardrailLens.chat_prompt_v3`, `SYSTEM_PROMPT_V3`) over
the PIArena corpora built by `piarena_eval/prepare_piarena_data.py` into
`causal_sweep.sweep_positions` -- unchanged from v2, this driver only
supplies a different prompt shape and a different case scheme.

Two things make this different from `run_causal_pipeline.py` (v2):

1. **Real gating.** The causal sweep only runs when the guardrail's verdict
   is `malign` -- v2 always ran it, regardless of verdict. When the verdict
   is not `malign` (`benign` or `unknown`), the row is recorded as "would be
   sent to the target model" and nothing else happens: the target model and
   the ASR/Utility judge are not implemented yet (see ARCHITECTURE.md,
   Decisions 1 and 5), so no response is fabricated for that branch.
2. **Case scheme.** `attacked`/`clean` (from which corpus file the row came
   from) crossed with the verdict, replacing `ground_truth.py`'s
   `TP`/`FP`/`FN`/`TN` (which assumed a seed's true label was itself
   malign/benign -- meaningless for a PIArena QA sample). Only
   `attacked+malign` and `clean+malign` ever reach the sweep, by
   construction of the gating above.

`--config` accepts one or more PIArena config names (each must already be
built by `prepare_piarena_data.py`), or the special value `all-main` for all
13 main-eval configs (Table 8) in one run -- one pair of output files per
(config, variant).

Local smoke (cpu, bf16):

    guardrail_eval/.venv/Scripts/python.exe causal_eval/run_causal_pipeline_piarena.py \
        --config dolly_closed_qa --variant both --n-samples 2 --last-n 20

Pod run (cuda, float32), full main-eval corpus:

    python causal_eval/run_causal_pipeline_piarena.py --device cuda --dtype float32 \
        --guardrail-model google/gemma-3-4b-it --config all-main \
        --variant both --n-samples 15 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "guardrail_eval"),
    os.path.join(_REPO_ROOT, "piarena_eval"),
    os.path.dirname(__file__),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jlens_readout import GuardrailLens  # noqa: E402

from causal_sweep import sweep_positions  # noqa: E402
from prepare_piarena_data import MAIN_CONFIGS  # noqa: E402

_DATA_DIR = os.path.join(_REPO_ROOT, "piarena_eval", "data")
_OUT_DIR = os.path.join(os.path.dirname(__file__), "results_causal_piarena")

#: One combined spreadsheet for the guardrail's output (verdict/case, not the
#: position-level sweep detail, which stays in causal_position_scores_*.jsonl)
#: across every (config, variant) processed in a single run -- open this one
#: file in Excel/Sheets to eyeball verdict distribution, malign rate, etc.
#: across the whole PIArena main-eval corpus instead of grepping 26 JSONLs.
_SUMMARY_CSV_FIELDS = [
    "sample_index", "config", "variant", "attacked", "verdict", "case",
    "gated_to_causal_sweep", "would_call_target_model",
    "target_model_implemented", "seq_len", "n_positions",
    "elapsed_seconds", "guardrail_model",
]


def _summary_csv_path() -> str:
    return os.path.join(_OUT_DIR, "guardrail_output.csv")


def _append_summary_row(path: str, row: dict) -> None:
    """Append one guardrail-output row to the combined CSV, writing the
    header only the first time the file is created (or is empty)."""
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _case(attacked: bool, verdict: str) -> str:
    """`attacked`/`clean` x verdict -- see ARCHITECTURE.md Decision 4. Only
    the two `malign` cells are ever swept; `unknown` covers a guardrail that
    failed to emit a clean malign/benign verdict at all."""
    if verdict not in ("malign", "benign"):
        return "unknown"
    return f"{'attacked' if attacked else 'clean'}+{verdict}"


def _variant_path(config: str, variant: str) -> str:
    return os.path.join(_DATA_DIR, f"{config}_{variant}.csv")


def _select(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """First-N rows by `sample_index` (deterministic), mirroring v2's
    `_select`. `clean.csv` and `direct.csv` are built from the same
    underlying rows in the same order (see prepare_piarena_data.py), so
    calling this with the same `n_samples` on both files selects the same
    underlying samples in both conditions."""
    return df.sort_values("sample_index").head(n_samples).reset_index(drop=True)


def _done_sample_indices(path: str) -> set[int]:
    """`sample_index` values already written (for --resume)."""
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["sample_index"]))
    return done


def run_variant(gl: GuardrailLens, config: str, variant: str, args: argparse.Namespace) -> None:
    csv_path = _variant_path(config, variant)
    df = _select(pd.read_csv(csv_path), args.n_samples)

    band = [l for l in gl.lens.source_layers if args.layer_lo <= l <= args.layer_hi]
    malign_id = gl.tok.encode("malign", add_special_tokens=False)[0]
    benign_id = gl.tok.encode("benign", add_special_tokens=False)[0]

    def run_forward(input_ids):
        return gl.hf(input_ids).logits[0]  # [seq_len, vocab]

    def score_fn(logits):
        # log-odds(malign vs benign), not raw P(malign) -- see v2's
        # run_causal_pipeline.py / ARCHITECTURE.md for why (softmax
        # saturation on this guardrail's very confident verdicts).
        logits = logits.float()
        return float(logits[malign_id] - logits[benign_id])

    os.makedirs(_OUT_DIR, exist_ok=True)
    readouts_path = os.path.join(_OUT_DIR, f"causal_readouts_{config}_{variant}.jsonl")
    scores_path = os.path.join(_OUT_DIR, f"causal_position_scores_{config}_{variant}.jsonl")

    done = _done_sample_indices(readouts_path) if args.resume else set()
    r_mode = "a" if args.resume else "w"

    todo = df[~df["sample_index"].astype(int).isin(done)]
    n_todo = len(todo)
    elapsed_total = 0.0
    n_timed = 0

    with open(readouts_path, r_mode, encoding="utf-8") as fr, \
         open(scores_path, r_mode, encoding="utf-8") as fs:
        for i, (_, row) in enumerate(todo.iterrows(), start=1):
            sample_index = int(row["sample_index"])
            attacked = bool(row["attacked"])

            t0 = time.perf_counter()
            prompt = gl.chat_prompt_v3(row["target_inst"], row["context"])
            input_ids = gl.model.encode(prompt, max_length=args.max_seq_len)
            seq_len = int(input_ids.shape[1])
            if seq_len >= args.max_seq_len:
                print(f"  [warn] sample_index={sample_index}: prompt truncated at "
                      f"--max-seq-len={args.max_seq_len} (seq_len == max_seq_len) -- "
                      f"the verdict below may be based on an incomplete prompt")
            verdict, _ = gl.classify(prompt, max_seq_len=args.max_seq_len)
            case = _case(attacked, verdict)
            gated_to_sweep = verdict == "malign"

            n_positions = 0
            if gated_to_sweep:
                # Anchor p_start/p_end to the rendered CONTEXT span (not
                # REQUEST, not the fixed system prompt) -- mirrors v2's
                # seed-anchoring, swapping "seed" for "context" since that's
                # where injected_task lives under the PIArena schema.
                context_text = row["context"]
                try:
                    prefix_start = prompt.index(context_text)
                    prefix_end = prefix_start + len(context_text)
                    p_start = int(gl.model.encode(
                        prompt[:prefix_start], max_length=args.max_seq_len
                    ).shape[1])
                    p_end = int(gl.model.encode(
                        prompt[:prefix_end], max_length=args.max_seq_len
                    ).shape[1])
                except ValueError:
                    p_start = 16
                    p_end = seq_len
                    print(f"  [warn] sample_index={sample_index}: context text "
                          f"not found verbatim in rendered prompt, falling back "
                          f"to p_start=16, p_end=seq_len")
                if args.last_n is not None:
                    p_start = max(p_start, p_end - args.last_n)

                records = sweep_positions(
                    gl.model.layers, gl.lens, gl.hf.get_output_embeddings().weight,
                    input_ids, run_forward, score_fn,
                    layers=band, k=args.k, p_start=p_start, p_end=p_end, seed=args.seed,
                )
                n_positions = len(records)
                if not records:
                    print(f"  [warn] sample_index={sample_index}: 0 positions swept "
                          f"(p_start={p_start}, p_end={p_end})")
                for rec in records:
                    token = gl.tok.decode([int(input_ids[0, rec["position"]])])
                    candidatos = [gl.tok.decode([cid]) for cid in rec["candidate_ids"]]
                    fs.write(json.dumps({
                        "sample_index": sample_index, "config": config,
                        "variant": variant, "attacked": attacked, "case": case,
                        "token": token, "candidatos": candidatos, **rec,
                    }) + "\n")
                fs.flush()

            elapsed = time.perf_counter() - t0
            elapsed_total += elapsed
            n_timed += 1

            # "would_call_target_model" documents what real v3 gating would
            # do next (Decision 1) without actually doing it -- the target
            # model and the ASR/Utility judge aren't implemented yet, so no
            # response is fabricated on either branch (see ARCHITECTURE.md's
            # resolved open question on the blocked branch).
            readout_row = {
                "sample_index": sample_index, "config": config, "variant": variant,
                "attacked": attacked, "verdict": verdict, "case": case,
                "gated_to_causal_sweep": gated_to_sweep,
                "would_call_target_model": not gated_to_sweep,
                "target_model_implemented": False,
                "seq_len": seq_len, "n_positions": n_positions,
                "elapsed_seconds": round(elapsed, 3),
                "guardrail_model": args.guardrail_model,
            }
            fr.write(json.dumps(readout_row) + "\n")
            fr.flush()
            _append_summary_row(_summary_csv_path(), readout_row)

            avg = elapsed_total / n_timed
            remaining = n_todo - i
            eta_min = avg * remaining / 60
            timing = (f"{elapsed:.1f}s ({n_positions} pos) | avg {avg:.1f}s/sample | "
                      f"{i}/{n_todo} done | ETA {eta_min:.1f}min for remaining {remaining}")
            if args.verbose:
                print(f"[{config}/{variant}] sample_index={sample_index} "
                      f"attacked={attacked}->{verdict} ({case}) | {timing}")
            else:
                print(f"[{config}/{variant}] sample_index={sample_index}: {timing}")

    print(f"[{config}/{variant}] wrote {readouts_path} and {scores_path} "
          f"({n_timed} samples, avg {elapsed_total/n_timed:.1f}s/sample)"
          if n_timed else
          f"[{config}/{variant}] nothing to do (all {len(df)} rows already in {readouts_path})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal pipeline v3 driver (PIArena-based).")
    p.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--config", nargs="+", default=["dolly_closed_qa"],
                   help="One or more PIArena config names (matches "
                   "piarena_eval/data/{config}_*.csv), or 'all-main' for all "
                   "13 main-eval configs (Table 8).")
    p.add_argument("--variant", default="both", choices=["clean", "direct", "both"])
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--layer-lo", type=int, default=14)  # workspace band lo (Qwen3)
    p.add_argument("--layer-hi", type=int, default=26)  # workspace band hi (Qwen3)
    p.add_argument("--last-n", type=int, default=None,
                   help="sweep only the last N context positions (keeps local runs feasible)")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Truncate the rendered prompt to this many tokens before "
                   "encoding/classifying (passed to GuardrailLens.classify/model.encode; "
                   "jlens' own default is 512, too short for most PIArena contexts -- "
                   "see ARCHITECTURE.md's truncation-bug note). Raise further for the "
                   "_long configs (up to ~19k tokens of context, Table 8).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Loading {args.guardrail_model} ({args.dtype}, {args.device})...")
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)

    os.makedirs(_OUT_DIR, exist_ok=True)

    # The combined summary CSV spans every (config, variant) processed in
    # this invocation (unlike the per-config JSONLs, which reset with -w
    # unless --resume). Without --resume, start it fresh once here so a new
    # run doesn't append onto a stale spreadsheet from a previous session;
    # each row appended inside run_variant() from here on (see
    # _append_summary_row) is additive and --resume-safe (a resumed row is
    # simply never reprocessed, so it's never re-appended either).
    if not args.resume and os.path.isfile(_summary_csv_path()):
        os.remove(_summary_csv_path())

    with open(os.path.join(_OUT_DIR, "causal_run_meta_piarena.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    configs = MAIN_CONFIGS if args.config == ["all-main"] else args.config
    variants = ["clean", "direct"] if args.variant == "both" else [args.variant]
    for config in configs:
        for variant in variants:
            run_variant(gl, config, variant, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
