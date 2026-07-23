#!/usr/bin/env python
"""Phase 3: automated auditor (§A.22) over the guardrail's J-lens readouts.

For each seed: the guardrail (Qwen3-1.7B + J-lens) classifies it and its
readout is captured at the decision position; the behavioral ground truth
(``ground_truth.py``, Strategy A) picks the applicable yes/no claims for that
(label, verdict); and for each claim an investigator agent (Gemini) resolves it
using only the readout, then an LLM-judge (Gemini) scores the answer against
the expected gabarito.

Only the guardrail is used (no target model), so it loads once -- none of the
two-phase memory dance of run_attack_pipeline.py is needed. The investigator +
judge run via API, so this loop is API-bound, not GPU-bound: it works on CPU
for a small dry-run and on a RunPod GPU for a larger smoke test.

Writes to a dedicated results_audit/ folder (separate from run_attack_pipeline's
results/):
  results_audit/audit_readouts.jsonl  (one record per prompt: verdict + readout)
  results_audit/audit_scores.jsonl    (one record per (prompt, claim): scores)
  results_audit/audit_summary.csv     (per-claim aggregate)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# Workspace band (see run_attack_pipeline.py / ARCHITECTURE.md): L1-13 are
# formatting noise, the interpretable abstract content lives ~L14-26.
DEFAULT_LAYERS = list(range(14, 27))


def select_subset(n_malign: int, n_benign: int) -> pd.DataFrame:
    """First-N malign + first-N benign from the (unwrapped) baseline corpus."""
    df = pd.read_csv(HERE / "data" / "attack_baseline.csv")
    malign = df[df["label"] == "malign"].head(n_malign)
    benign = df[df["label"] == "benign"].head(n_benign)
    return pd.concat([malign, benign], ignore_index=True)


def out_paths() -> dict[str, Path]:
    results_dir = HERE / "results_audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "readouts": results_dir / "audit_readouts.jsonl",
        "scores": results_dir / "audit_scores.jsonl",
        "summary": results_dir / "audit_summary.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-malign", type=int, default=5)
    parser.add_argument("--n-benign", type=int, default=5)
    parser.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--audit-model",
        default="gemini-3.5-flash",
        help="Gemini model for both investigator and judge.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layers to read the guardrail's J-lens at; default = workspace band L14-L26.",
    )
    parser.add_argument("--guardrail-max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda".')
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here (not at module scope) so --help / arg errors don't require
    # torch + the Gemini SDK to be importable.
    from audit_agent import format_readout, investigate, judge
    from ground_truth import ground_truth_for
    from jlens_readout import GuardrailLens

    subset = select_subset(args.n_malign, args.n_benign)
    paths = out_paths()
    dtype = getattr(torch, args.dtype)

    print(
        f"loading guardrail {args.guardrail_model} + J-lens "
        f"(device={args.device}, dtype={args.dtype}) ..."
    )
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)
    print(f"  {gl.model}\n  {gl.lens}")

    score_rows: list[dict] = []
    t0 = time.time()
    with (
        paths["readouts"].open("w", encoding="utf-8") as f_read,
        paths["scores"].open("w", encoding="utf-8") as f_score,
    ):
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc="audit"):
            guardrail_prompt = gl.chat_prompt(row["prompt"])
            label_pred, guardrail_raw = gl.classify(
                guardrail_prompt, max_new_tokens=args.guardrail_max_new_tokens
            )
            readout = gl.readout(
                guardrail_prompt, layers=args.layers, top_k=args.top_k
            )
            readout_text = format_readout(readout)

            f_read.write(
                json.dumps(
                    {
                        "pool_index": int(row["pool_index"]),
                        "seed": row["seed"],
                        "prompt": row["prompt"],
                        "category": row["category"],
                        "label_true": row["label"],
                        "guardrail_label_pred": label_pred,
                        "guardrail_raw": guardrail_raw,
                        "jlens": {str(k): v for k, v in readout.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            transcript = (
                f"O guardrail recebeu esta mensagem para classificar:\n"
                f"{row['prompt']}\n\n"
                f"E emitiu o veredito: {label_pred!r}."
            )
            claims = ground_truth_for(row["label"], label_pred)
            for claim, expected in claims:
                inv = investigate(
                    transcript, claim.text, readout_text, model=args.audit_model
                )
                jud = judge(claim.text, inv, expected, model=args.audit_model)

                if args.verbose:
                    print(
                        f"\n[pool_index={row['pool_index']}] claim={claim.id} "
                        f"expected={'sim' if expected else 'nao'} "
                        f"investigator={inv['verdict']} "
                        f"score={jud['score']:.1f}"
                    )

                score_row = {
                    "pool_index": int(row["pool_index"]),
                    "claim_id": claim.id,
                    "label_true": row["label"],
                    "guardrail_label_pred": label_pred,
                    "expected": "sim" if expected else "nao",
                    "investigator_verdict": inv["verdict"],
                    "investigator_evidence": inv["evidence"],
                    "correctness": jud["correctness"],
                    "evidence_quality": jud["evidence_quality"],
                    "score": jud["score"],
                    "justification": jud["justification"],
                }
                score_rows.append(score_row)
                f_score.write(json.dumps(score_row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0

    scores = pd.DataFrame(score_rows)
    if scores.empty:
        print(
            "no applicable claims (every guardrail verdict was 'unknown'?) -- "
            f"nothing scored. elapsed={elapsed:.1f}s"
        )
        return

    scores["investigator_correct"] = (
        scores["investigator_verdict"] == scores["expected"]
    )
    summary = (
        scores.groupby("claim_id")
        .agg(
            n=("score", "size"),
            mean_score=("score", "mean"),
            mean_correctness=("correctness", "mean"),
            mean_evidence_quality=("evidence_quality", "mean"),
            investigator_accuracy=("investigator_correct", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(paths["summary"], index=False)

    print(
        f"\naudit done: {len(scores)} (prompt, claim) evaluations over "
        f"{len(subset)} prompts  elapsed={elapsed:.1f}s\n"
        f"  readouts -> {paths['readouts']}\n"
        f"  scores   -> {paths['scores']}\n"
        f"  summary  -> {paths['summary']}\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
