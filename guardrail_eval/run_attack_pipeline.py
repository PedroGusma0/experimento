#!/usr/bin/env python
"""Phase 2: guardrail (Qwen3-1.7B + J-lens) + target (gemma-3-1b-it) over the
baseline / baseline-wrapping attack corpora.

For each prompt in the smoke subset: the guardrail classifies it and its
J-lens readout is recorded at the decision position, and -- regardless of
the guardrail's verdict -- the prompt is forwarded to the target model,
which generates an open-ended response. Nothing is blocked in this phase;
the guardrail's classification is recorded, never gated on.

Runs in two sequential phases so the guardrail and target model are never
resident in memory at the same time (this machine has ~7.7GB RAM; Qwen3-1.7B
+ gemma-3-1b-it loaded together in fp32 exceeds that). Phase A loads only the
guardrail and writes the J-lens readouts + guardrail summary for every attack
in scope; the guardrail is then freed before Phase B loads only the target
model and writes its outputs for the same rows.

Per attack ("baseline" / "baseline-wrapping"), writes three artifacts:
  results/readouts_qwen3_1.7b_<slug>.jsonl   (guardrail J-lens readouts)
  results/summary_qwen3_1.7b_<slug>.csv      (guardrail verdicts)
  results/target_<slug>.csv                  (target model outputs)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# Workspace-band default: reading all 27 fitted layers is the dominant cost
# (a full-vocab unembed per layer); L1-13 were confirmed noise/formatting
# tokens in the Phase 1 smoke test, so default to the interpretable band.
DEFAULT_LAYERS = list(range(14, 27))

ATTACKS = ["baseline", "baseline-wrapping"]


def slug(attack: str) -> str:
    return attack.replace("-", "_")


def select_subset(attack: str, n_malign: int, n_benign: int) -> pd.DataFrame:
    df = pd.read_csv(HERE / "data" / f"attack_{slug(attack)}.csv")
    malign = df[df["label"] == "malign"].head(n_malign)
    benign = df[df["label"] == "benign"].head(n_benign)
    return pd.concat([malign, benign], ignore_index=True)


def out_paths(attack: str) -> dict[str, Path]:
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": results_dir / f"readouts_qwen3_1.7b_{slug(attack)}.jsonl",
        "summary": results_dir / f"summary_qwen3_1.7b_{slug(attack)}.csv",
        "target": results_dir / f"target_{slug(attack)}.csv",
    }


# --------------------------------------------------------------------------- #
# Phase A: guardrail (classify + J-lens readout)
# --------------------------------------------------------------------------- #


def run_guardrail_phase(attack: str, gl, args: argparse.Namespace) -> None:
    subset = select_subset(attack, args.n_malign, args.n_benign)
    paths = out_paths(attack)

    summary_rows = []
    t0 = time.time()
    with paths["jsonl"].open("w", encoding="utf-8") as fh:
        for _, row in tqdm(
            subset.iterrows(), total=len(subset), desc=f"guardrail[{attack}]"
        ):
            prompt = row["prompt"]
            # Wrap the raw/wrapped attack text in the guardrail's own
            # classifier template (SYSTEM_PROMPT + "INPUT: ...") -- this is
            # what a guardrail sitting in front of the target would actually
            # see. Both classify() and readout() must see the identical
            # templated string so the lens's decision position lines up with
            # what classify() generated from.
            guardrail_prompt = gl.chat_prompt(prompt)
            label_pred, guardrail_raw = gl.classify(
                guardrail_prompt, max_new_tokens=args.guardrail_max_new_tokens
            )
            readout = gl.readout(guardrail_prompt, layers=args.layers, top_k=args.top_k)

            if args.verbose:
                print(
                    f"\n[{attack}] pool_index={row['pool_index']} label_true={row['label']} "
                    f"guardrail={label_pred!r}\nprompt: {prompt[:80]!r}"
                )

            record = {
                "pool_index": int(row["pool_index"]),
                "attack": attack,
                "seed": row["seed"],
                "prompt": prompt,
                "category": row["category"],
                "source": row["source"],
                "source_id": row["source_id"],
                "label_true": row["label"],
                "guardrail_label_pred": label_pred,
                "guardrail_raw": guardrail_raw,
                "jlens": {str(layer): toks for layer, toks in readout.items()},
            }
            if attack == "baseline-wrapping":
                record["variant_index"] = int(row["variant_index"])
                record["variant_prompt_type"] = row["variant_prompt_type"]
                record["variant_attack_category"] = row["variant_attack_category"]
                record["variant_attack_technique"] = row["variant_attack_technique"]
                record["variant_attack_pattern"] = row["variant_attack_pattern"]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary_rows.append(
                {
                    "pool_index": row["pool_index"],
                    "attack": attack,
                    "seed": row["seed"],
                    "prompt": prompt,
                    "category": row["category"],
                    "label (true)": row["label"],
                    "label (prediction)": label_pred,
                }
            )
    elapsed = time.time() - t0

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["summary"], index=False)
    accuracy = (summary["label (true)"] == summary["label (prediction)"]).mean()
    n_unknown = (summary["label (prediction)"] == "unknown").sum()
    print(
        f"[{attack}] guardrail: n={len(summary)}  accuracy={accuracy:.3f}  "
        f"unknown={n_unknown}  elapsed={elapsed:.1f}s\n"
        f"  readouts -> {paths['jsonl']}\n  summary  -> {paths['summary']}"
    )


# --------------------------------------------------------------------------- #
# Phase B: target model (open-ended generation)
# --------------------------------------------------------------------------- #


def run_target_phase(attack: str, tm, args: argparse.Namespace) -> None:
    subset = select_subset(attack, args.n_malign, args.n_benign)
    paths = out_paths(attack)

    target_rows = []
    t0 = time.time()
    for _, row in tqdm(
        subset.iterrows(), total=len(subset), desc=f"target[{attack}]"
    ):
        prompt = row["prompt"]
        # No gating: every prompt is forwarded regardless of the guardrail's
        # (already-recorded, in Phase A) verdict.
        output = tm.generate(prompt, max_new_tokens=args.target_max_new_tokens)
        if args.verbose:
            print(f"\n[{attack}] prompt: {prompt[:80]!r}\noutput: {output[:120]!r}")
        target_rows.append({"seed": row["seed"], "prompt": prompt, "output": output})
    elapsed = time.time() - t0

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(paths["target"], index=False)
    print(
        f"[{attack}] target: n={len(target_df)}  elapsed={elapsed:.1f}s\n"
        f"  target -> {paths['target']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", choices=[*ATTACKS, "both"], default="both")
    parser.add_argument("--n-malign", type=int, default=3)
    parser.add_argument("--n-benign", type=int, default=3)
    parser.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target-model", default="google/gemma-3-1b-it")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layers to read the guardrail's J-lens at; default = workspace band L14-L26.",
    )
    parser.add_argument("--guardrail-max-new-tokens", type=int, default=8)
    parser.add_argument("--target-max-new-tokens", type=int, default=60)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attacks = ATTACKS if args.attack == "both" else [args.attack]
    dtype = getattr(torch, args.dtype)
    t0 = time.time()

    # Phase A: guardrail only. Imported here (not at module scope) alongside
    # the explicit `del` + gc.collect() below, so the model is fully released
    # before Phase B loads the target -- the two are never resident together.
    from jlens_readout import GuardrailLens

    print(f"[phase A] loading guardrail {args.guardrail_model} + J-lens (dtype={args.dtype}) ...")
    gl = GuardrailLens(args.guardrail_model, dtype=dtype)
    print(f"  {gl.model}\n  {gl.lens}")
    for attack in attacks:
        run_guardrail_phase(attack, gl, args)
    del gl
    gc.collect()

    # Phase B: target only.
    from target_model import TargetModel

    print(f"\n[phase B] loading target {args.target_model} (dtype={args.dtype}) ...")
    tm = TargetModel(args.target_model, dtype=dtype)
    print(f"  {tm}")
    for attack in attacks:
        run_target_phase(attack, tm, args)
    del tm
    gc.collect()

    print(f"\ntotal elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
