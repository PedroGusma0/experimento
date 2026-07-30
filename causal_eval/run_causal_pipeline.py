#!/usr/bin/env python
"""Causal validation pipeline v2 — driver.

Wires the real guardrail (`GuardrailLens`) into `causal_sweep.sweep_positions`
to produce, per prompt, a signed per-token importance score for the `malign`
verdict:

    nota(p) = P_ablate(p)(malign) − P_control(p)(malign)

Negative → the content at position p was supporting `malign`; positive → it
was suppressing it (see ARCHITECTURE.md, "Causal validation pipeline v2").
`ground_truth.py`'s label × verdict logic is reused only to *label* each prompt
TP/FP/FN/TN for later slicing — nothing is graded, no investigator/judge.

This is script-only and API-free. The per-position sweep is ~`2 × n_positions`
forward passes per prompt, so a full sweep over a 15+15 corpus is a pod-scale
run; locally, use `--last-n` (sweep only the last N input tokens — enough to
cover the seed after the system-prompt boilerplate) and small `--n-*` to keep
it feasible.

Local smoke (cpu, bf16 — the only combination that fits ~7.7GB RAM):

    guardrail_eval/.venv/Scripts/python.exe causal_eval/run_causal_pipeline.py \
        --attack baseline --n-malign 1 --n-benign 1 --last-n 12

Pod run (cuda, float32):

    python causal_eval/run_causal_pipeline.py --device cuda --dtype float32 \
        --attack both --n-malign 15 --n-benign 15 --resume
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

from jlens_readout import GuardrailLens  # noqa: E402

from causal_sweep import sweep_positions  # noqa: E402

_ATTACK_FILES = {
    "baseline": "attack_baseline.csv",
    "baseline-wrapping": "attack_baseline_wrapping.csv",
}
_DATA_DIR = os.path.join(_REPO_ROOT, "guardrail_eval", "data")
_OUT_DIR = os.path.join(os.path.dirname(__file__), "results_causal")


def _case(label_true: str, verdict: str) -> str:
    """Confusion-matrix cell (malign = the positive class)."""
    if verdict not in ("malign", "benign"):
        return "unknown"
    if label_true == "malign":
        return "TP" if verdict == "malign" else "FN"
    return "FP" if verdict == "malign" else "TN"


def _slug(attack: str) -> str:
    return attack.replace("-", "_")


def _select(df: pd.DataFrame, n_malign: int, n_benign: int) -> pd.DataFrame:
    """First-N malign + first-N benign (deterministic), mirroring the audit pipeline."""
    malign = df[df["label"] == "malign"].head(n_malign)
    benign = df[df["label"] == "benign"].head(n_benign)
    return pd.concat([malign, benign]).reset_index(drop=True)


def _done_pool_indices(path: str) -> set[int]:
    """pool_index values already written (for --resume)."""
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["pool_index"]))
    return done


def run_attack(gl: GuardrailLens, attack: str, args: argparse.Namespace) -> None:
    csv_path = os.path.join(_DATA_DIR, _ATTACK_FILES[attack])
    df = _select(pd.read_csv(csv_path), args.n_malign, args.n_benign)

    band = [l for l in gl.lens.source_layers if args.layer_lo <= l <= args.layer_hi]
    malign_id = gl.tok.encode("malign", add_special_tokens=False)[0]
    benign_id = gl.tok.encode("benign", add_special_tokens=False)[0]

    def run_forward(input_ids):
        return gl.hf(input_ids).logits[0]  # [seq_len, vocab]

    def score_fn(logits):
        # log-odds(malign vs benign) at the decision position, not raw
        # P(malign): softmax saturates near 1.0 for this guardrail (both TP
        # and FP prompts in the first real run hit P(malign)~1.0), hiding all
        # but the single largest effect below float32 noise. logit_malign -
        # logit_benign stays sensitive in that regime and, unlike a bare
        # logit_malign, is invariant to any shared shift across all logits
        # (e.g. the ablation making the model generally less confident) --
        # the log-sum-exp normalizer cancels exactly in the subtraction.
        logits = logits.float()
        return float(logits[malign_id] - logits[benign_id])

    os.makedirs(_OUT_DIR, exist_ok=True)
    slug = _slug(attack)
    readouts_path = os.path.join(_OUT_DIR, f"causal_readouts_{slug}.jsonl")
    scores_path = os.path.join(_OUT_DIR, f"causal_position_scores_{slug}.jsonl")

    done = _done_pool_indices(readouts_path) if args.resume else set()
    r_mode = "a" if args.resume else "w"

    todo = df[~df["pool_index"].astype(int).isin(done)]
    n_todo = len(todo)
    elapsed_total = 0.0  # sums only the timed (prompt, sweep) work, not I/O
    n_timed = 0

    with open(readouts_path, r_mode, encoding="utf-8") as fr, \
         open(scores_path, r_mode, encoding="utf-8") as fs:
        for i, (_, row) in enumerate(todo.iterrows(), start=1):
            pool_index = int(row["pool_index"])

            t0 = time.perf_counter()
            prompt = gl.chat_prompt(row["prompt"])
            input_ids = gl.model.encode(prompt)
            seq_len = int(input_ids.shape[1])
            verdict, _ = gl.classify(prompt)
            case = _case(row["label"], verdict)

            # p_start: skip past the fixed system prompt + "INPUT: " boilerplate
            # dynamically, per prompt -- not SKIP_FIRST_N_POSITIONS (that constant
            # is jlens.fitting's attention-sink skip, unrelated; reusing it left
            # the sweep deep inside SYSTEM_PROMPT, which is identical across every
            # prompt and therefore cannot explain any one prompt's verdict; see
            # ARCHITECTURE.md, "First real-guardrail run...", Finding 4). Locate
            # where the seed text begins in the rendered string and tokenize only
            # that prefix. p_end is NOT given the equivalent treatment: trailing
            # boilerplate (Classification:/<think></think>/etc.) comes AFTER the
            # seed, so causal attention lets it legitimately carry seed-derived
            # signal even though its surface tokens are fixed.
            seed_text = row["prompt"]
            try:
                prefix_end = prompt.index(seed_text)
                seed_start = int(gl.model.encode(prompt[:prefix_end]).shape[1])
            except ValueError:
                seed_start = 16  # SKIP_FIRST_N_POSITIONS fallback: seed not found
                print(f"  [warn] pool_index={pool_index}: seed text not found "
                      f"verbatim in rendered prompt, falling back to p_start=16")

            p_start = seed_start
            if args.last_n is not None:
                p_start = max(seed_start, seq_len - args.last_n)

            records = sweep_positions(
                gl.model.layers, gl.lens, gl.hf.get_output_embeddings().weight,
                input_ids, run_forward, score_fn,
                layers=band, k=args.k, p_start=p_start, seed=args.seed,
            )
            elapsed = time.perf_counter() - t0
            elapsed_total += elapsed
            n_timed += 1
            sec_per_pos = elapsed / len(records) if records else float("nan")

            # one readout row per prompt (clean verdict + case), then one row
            # per swept position (the signed score) — the write happens before
            # nothing else, so a crash mid-corpus leaves a consistent checkpoint.
            fr.write(json.dumps({
                "pool_index": pool_index, "attack": attack,
                "label_true": row["label"], "verdict": verdict, "case": case,
                "seq_len": seq_len, "n_positions": len(records),
                "elapsed_seconds": round(elapsed, 3),
                "seconds_per_position": round(sec_per_pos, 4),
                "guardrail_model": args.guardrail_model,
            }) + "\n")
            fr.flush()
            for rec in records:
                token = gl.tok.decode([int(input_ids[0, rec["position"]])])
                fs.write(json.dumps({
                    "pool_index": pool_index, "attack": attack, "case": case,
                    "token": token, **rec,
                }) + "\n")
            fs.flush()

            avg = elapsed_total / n_timed
            remaining = n_todo - i
            eta_min = avg * remaining / 60
            timing = (f"{elapsed:.1f}s ({sec_per_pos:.3f}s/pos, {len(records)} pos) | "
                      f"avg {avg:.1f}s/prompt | {i}/{n_todo} done | "
                      f"ETA {eta_min:.1f}min for remaining {remaining}")
            if args.verbose:
                top = min(records, key=lambda r: r["nota"], default=None)
                msg = (f"[{attack}] pool_index={pool_index} {row['label']}"
                       f"->{verdict} ({case}) | {timing}")
                if top is not None:
                    tt = gl.tok.decode([int(input_ids[0, top['position']])])
                    msg += f"\n    most malign-driving: {tt!r} @ {top['position']} (nota={top['nota']:.4f})"
                print(msg)
            else:
                print(f"[{attack}] pool_index={pool_index}: {timing}")

    print(f"[{attack}] wrote {readouts_path} and {scores_path} "
          f"({n_timed} prompts, avg {elapsed_total/n_timed:.1f}s/prompt)"
          if n_timed else f"[{attack}] nothing to do (all {len(df)} rows already in {readouts_path})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal validation pipeline v2 driver.")
    p.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--attack", default="baseline",
                   choices=["baseline", "baseline-wrapping", "both"])
    p.add_argument("--n-malign", type=int, default=5)
    p.add_argument("--n-benign", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--layer-lo", type=int, default=14)  # workspace band lo (Qwen3)
    p.add_argument("--layer-hi", type=int, default=26)  # workspace band hi (Qwen3)
    p.add_argument("--last-n", type=int, default=None,
                   help="sweep only the last N input positions (keeps local runs feasible)")
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
    with open(os.path.join(_OUT_DIR, "causal_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    attacks = ["baseline", "baseline-wrapping"] if args.attack == "both" else [args.attack]
    for attack in attacks:
        run_attack(gl, attack, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
