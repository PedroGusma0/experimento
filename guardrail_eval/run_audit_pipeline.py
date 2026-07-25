#!/usr/bin/env python
"""Phase 3: automated auditor (§A.22) over the guardrail's J-lens readouts.

For each seed: the guardrail (Qwen3-1.7B + J-lens) classifies it and an anchor
readout is captured at the decision position; the behavioral ground truth
(``ground_truth.py``, Strategy A) picks the applicable yes/no claims for that
(label, verdict); and for each claim an investigator agent (DeepSeek) resolves
it by interactively probing the J-lens at positions/layers of its own choosing
(up to ``--max-tool-calls`` queries -- see ``audit_agent.investigate``), then an
LLM-judge (Groq) scores the answer against the expected gabarito.

Only the guardrail is used (no target model), so it loads once -- none of the
two-phase memory dance of run_attack_pipeline.py is needed. By default the
investigator + judge run via API (DeepSeek + Groq), so this loop is
API-bound, not GPU-bound; ``--investigator-backend local``/
``--judge-backend local`` switch either role to an open-weight model loaded
locally via ``transformers`` (no vLLM -- see ``local_agent.py`` and
``PLAN_runpod_audit.md``), in which case guardrail + investigator + judge all
stay resident together for the whole run (still no phase separation needed,
since local loading doesn't reserve VRAM the way a vLLM server would).

Runs over one or both attack corpora (``--attack``, mirroring
run_attack_pipeline.py's baseline / baseline-wrapping split). Writes to a
dedicated results_audit/ folder (separate from run_attack_pipeline's
results/), one set of files per attack:
  results_audit/audit_readouts_<attack>.jsonl  (one record per prompt)
  results_audit/audit_scores_<attack>.jsonl    (one record per (prompt, claim))
  results_audit/audit_summary_<attack>.csv     (per-claim aggregate)
With ``--attack both``, an additional results_audit/audit_summary_combined.csv
aggregates across both attacks.

Each row triggers at most one claim (a guardrail verdict is malign XOR
benign, never both, and ``ground_truth.py``'s two claims each apply to
exactly one of those) -- so API call volume scales close to linearly with
prompt count, not with claim count. With interactive tool-calling this is
~3-7 API calls per claim (investigator: 1 opening turn + up to
``--max-tool-calls`` tool rounds + 1 finalize; judge: 1 call) -- budget
accordingly for a large run; ``--api-pacing-seconds`` and ``--resume`` exist
to make that survivable on a free-tier API budget.
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

ATTACKS = ["baseline", "baseline-wrapping"]


def slug(attack: str) -> str:
    return attack.replace("-", "_")


def select_subset(attack: str, n_malign: int, n_benign: int) -> pd.DataFrame:
    """First-N malign + first-N benign from the given attack corpus."""
    df = pd.read_csv(HERE / "data" / f"attack_{slug(attack)}.csv")
    malign = df[df["label"] == "malign"].head(n_malign)
    benign = df[df["label"] == "benign"].head(n_benign)
    return pd.concat([malign, benign], ignore_index=True)


def out_paths(attack: str) -> dict[str, Path]:
    results_dir = HERE / "results_audit"
    results_dir.mkdir(parents=True, exist_ok=True)
    s = slug(attack)
    return {
        "readouts": results_dir / f"audit_readouts_{s}.jsonl",
        "scores": results_dir / f"audit_scores_{s}.jsonl",
        "summary": results_dir / f"audit_summary_{s}.csv",
        "meta": results_dir / f"audit_run_meta_{s}.json",
    }


def _done_pool_indices(readouts_path: Path) -> set[int]:
    """``pool_index`` values already present in a prior run's readouts file.

    Source of truth for ``--resume``: the readout is written unconditionally
    per row, before the costlier claim/API step, so skipping on it protects
    the expensive GPU work from being redone. Known gap: if a run died
    exactly between writing a row's readout and its score, that row is
    treated as done here but never got scored -- accepted as rare, not worth
    tracking both states separately for a research script.
    """
    done: set[int] = set()
    if not readouts_path.exists():
        return done
    with readouts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["pool_index"])
    return done


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    scores["investigator_correct"] = scores["investigator_verdict"] == scores["expected"]
    return (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", choices=[*ATTACKS, "both"], default="both")
    parser.add_argument("--n-malign", type=int, default=15)
    parser.add_argument("--n-benign", type=int, default=15)
    parser.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--investigator-backend",
        choices=["api", "local"],
        default="api",
        help="'api' (default): DeepSeek via the NVIDIA endpoint, same as "
        "always -- used by the CPU dry-run. 'local': open-weight model "
        "loaded locally via transformers, no vLLM -- RunPod only, see "
        "PLAN_runpod_audit.md ('Modelos abertos locais pro investigador/judge').",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["api", "local"],
        default="api",
        help="'api' (default): Groq, same as always. 'local': open-weight "
        "model loaded locally via transformers. Independent from "
        "--investigator-backend -- mixed combinations are supported.",
    )
    parser.add_argument(
        "--investigator-model",
        default=None,
        help="Investigator model id. Default depends on --investigator-backend "
        "('deepseek-ai/deepseek-v4-pro' for api, 'Qwen/Qwen2.5-14B-Instruct-AWQ' "
        "for local -- see local_agent.DEFAULT_LOCAL_INVESTIGATOR_MODEL).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model id. Default depends on --judge-backend "
        "('openai/gpt-oss-120b' for api, 'meta-llama/Llama-3.1-8B-Instruct' "
        "for local -- see local_agent.DEFAULT_LOCAL_JUDGE_MODEL). Different "
        "provider/family from the investigator by default, to reduce "
        "self-evaluation bias.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layers to read the guardrail's J-lens at; default = workspace band L14-L26. "
        "Also the default layer set for the investigator's tool when it omits `layers`.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=5,
        help="Cap on the investigator's get_jlens_readout queries per claim. "
        "Starting guess -- tune empirically once runnable (see PLAN_runpod_audit.md).",
    )
    parser.add_argument(
        "--token-span-last-n",
        type=int,
        default=60,
        help="How many trailing prompt tokens to show the investigator's token "
        "map (window around INPUT: {seed}...Classification:).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip pool_index rows already present in that attack's "
        "audit_readouts_<attack>.jsonl (append instead of overwrite), so a "
        "long run survives a pod interruption without redoing GPU work.",
    )
    parser.add_argument(
        "--api-pacing-seconds",
        type=float,
        default=0.0,
        help="Sleep this long after each claim's investigate+judge cycle -- "
        "raise if hitting sustained rate limits during a large run.",
    )
    parser.add_argument("--guardrail-max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda".')
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_audit_for_attack(
    attack: str,
    gl,
    args: argparse.Namespace,
    *,
    local_investigator=None,
    local_judge=None,
) -> pd.DataFrame:
    """Runs the audit loop for one attack corpus.

    Guardrail + investigator + judge stay resident together the whole loop
    (no phase separation -- see PLAN_runpod_audit.md for why that's safe now
    that the local path doesn't use vLLM). ``local_investigator``/
    ``local_judge`` are ``local_agent.LocalChatModel`` instances, required
    exactly when ``args.investigator_backend``/``args.judge_backend`` is
    ``"local"`` (loaded once by ``main()``, reused across attacks).

    Returns the *full* accumulated scores for this attack (reloaded from disk
    after writing) -- not just the rows processed in this invocation -- so
    both the per-attack summary and the caller's cross-attack combined
    summary reflect everything on disk, correctly handling ``--resume``.
    """
    from audit_agent import format_readout_multi, format_token_span
    from audit_agent import investigate as investigate_api
    from audit_agent import judge as judge_api
    from ground_truth import ground_truth_for

    if args.investigator_backend == "local":
        from local_agent import investigate_local
        assert local_investigator is not None
    if args.judge_backend == "local":
        from local_agent import judge_local
        assert local_judge is not None

    subset = select_subset(attack, args.n_malign, args.n_benign)
    paths = out_paths(attack)

    # Provenance: which model/backend produced this invocation's rows --
    # overwritten each run, reflects the config of the *current* command.
    with paths["meta"].open("w", encoding="utf-8") as f_meta:
        json.dump(vars(args), f_meta, ensure_ascii=False, indent=2, default=str)

    skip: set[int] = set()
    mode = "w"
    if args.resume:
        skip = _done_pool_indices(paths["readouts"])
        if skip:
            mode = "a"
            print(f"[{attack}] --resume: skipping {len(skip)} already-processed rows")

    n_new = 0
    t0 = time.time()
    with (
        paths["readouts"].open(mode, encoding="utf-8") as f_read,
        paths["scores"].open(mode, encoding="utf-8") as f_score,
    ):
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"audit[{attack}]"):
            pool_index = int(row["pool_index"])
            if pool_index in skip:
                continue

            guardrail_prompt = gl.chat_prompt(row["prompt"])
            label_pred, guardrail_raw = gl.classify(
                guardrail_prompt, max_new_tokens=args.guardrail_max_new_tokens
            )
            # Anchor readout at the decision position -- the investigator no
            # longer reads this directly; it probes readout_multi()
            # interactively via the tool (below). Kept for continuity of
            # audit_readouts_<attack>.jsonl and as the --resume checkpoint.
            readout = gl.readout(
                guardrail_prompt, layers=args.layers, top_k=args.top_k
            )

            f_read.write(
                json.dumps(
                    {
                        "attack": attack,
                        "pool_index": pool_index,
                        "seed": row["seed"],
                        "prompt": row["prompt"],
                        "category": row["category"],
                        "label_true": row["label"],
                        "guardrail_label_pred": label_pred,
                        "guardrail_raw": guardrail_raw,
                        "guardrail_model": args.guardrail_model,
                        "jlens": {str(k): v for k, v in readout.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_read.flush()  # survive a mid-run interruption for --resume

            transcript = (
                f"O guardrail recebeu esta mensagem para classificar:\n"
                f"{row['prompt']}\n\n"
                f"E emitiu o veredito: {label_pred!r}."
            )
            token_span_text = format_token_span(
                gl.token_span(guardrail_prompt, last_n=args.token_span_last_n)
            )

            def readout_fn(
                positions: list[int],
                layers: list[int] | None,
                *,
                _prompt: str = guardrail_prompt,
            ) -> str:
                """Bound to this row's guardrail_prompt; called by the
                investigator's tool loop (see audit_agent.investigate)."""
                multi = gl.readout_multi(
                    _prompt,
                    positions=positions,
                    layers=layers or args.layers,
                    top_k=args.top_k,
                )
                return format_readout_multi(multi)

            claims = ground_truth_for(row["label"], label_pred)
            for claim, expected in claims:
                if args.investigator_backend == "local":
                    inv = investigate_local(
                        transcript,
                        claim.text,
                        readout_fn=readout_fn,
                        token_span_text=token_span_text,
                        model=local_investigator,
                        max_tool_calls=args.max_tool_calls,
                    )
                else:
                    inv = investigate_api(
                        transcript,
                        claim.text,
                        readout_fn=readout_fn,
                        token_span_text=token_span_text,
                        model=args.investigator_model,
                        max_tool_calls=args.max_tool_calls,
                    )

                if args.judge_backend == "local":
                    jud = judge_local(claim.text, inv, expected, model=local_judge)
                else:
                    jud = judge_api(claim.text, inv, expected, model=args.judge_model)

                if args.verbose:
                    print(
                        f"\n[{attack}][pool_index={pool_index}] claim={claim.id} "
                        f"expected={'sim' if expected else 'nao'} "
                        f"investigator={inv['verdict']} "
                        f"tool_calls={inv.get('tool_calls', 0)} "
                        f"fallback={inv.get('fallback_used', False)} "
                        f"score={jud['score']:.1f}"
                    )

                score_row = {
                    "attack": attack,
                    "pool_index": pool_index,
                    "claim_id": claim.id,
                    "label_true": row["label"],
                    "guardrail_label_pred": label_pred,
                    "expected": "sim" if expected else "nao",
                    "investigator_verdict": inv["verdict"],
                    "investigator_evidence": inv["evidence"],
                    "tool_calls": inv.get("tool_calls", 0),
                    "fallback_used": inv.get("fallback_used", False),
                    "investigator_model": args.investigator_model,
                    "investigator_backend": args.investigator_backend,
                    "judge_model": args.judge_model,
                    "judge_backend": args.judge_backend,
                    "correctness": jud["correctness"],
                    "evidence_quality": jud["evidence_quality"],
                    "score": jud["score"],
                    "justification": jud["justification"],
                }
                n_new += 1
                f_score.write(json.dumps(score_row, ensure_ascii=False) + "\n")
                f_score.flush()

                if args.api_pacing_seconds > 0:
                    time.sleep(args.api_pacing_seconds)

    elapsed = time.time() - t0

    # Reload the *complete* scores file (old + newly appended rows) so the
    # summary reflects everything accumulated for this attack, not just the
    # rows processed in this invocation -- matters for --resume.
    if paths["scores"].exists() and paths["scores"].stat().st_size > 0:
        scores = pd.read_json(paths["scores"], lines=True)
    else:
        scores = pd.DataFrame()

    if scores.empty:
        print(
            f"[{attack}] no claims scored (new this run: {n_new}) "
            f"elapsed={elapsed:.1f}s"
        )
        return scores

    summary = _summarize(scores)
    summary.to_csv(paths["summary"], index=False)

    print(
        f"\n[{attack}] {n_new} new (prompt, claim) evaluations this run "
        f"({len(scores)} total in {paths['scores'].name})  elapsed={elapsed:.1f}s\n"
        f"  readouts -> {paths['readouts']}\n"
        f"  scores   -> {paths['scores']}\n"
        f"  summary  -> {paths['summary']}\n"
    )
    print(summary.to_string(index=False))
    return scores


def main() -> None:
    args = parse_args()
    # Imported here (not at module scope) so --help / arg errors don't require
    # torch to be importable.
    from jlens_readout import GuardrailLens

    needs_local = args.investigator_backend == "local" or args.judge_backend == "local"
    if needs_local:
        # local_agent imports transformers -- only pay for that when actually
        # needed (the default api/api path never touches it).
        from local_agent import (
            DEFAULT_LOCAL_INVESTIGATOR_MODEL,
            DEFAULT_LOCAL_JUDGE_MODEL,
            LocalChatModel,
        )

    if args.investigator_model is None:
        args.investigator_model = (
            DEFAULT_LOCAL_INVESTIGATOR_MODEL
            if args.investigator_backend == "local"
            else "deepseek-ai/deepseek-v4-pro"
        )
    if args.judge_model is None:
        args.judge_model = (
            DEFAULT_LOCAL_JUDGE_MODEL
            if args.judge_backend == "local"
            else "openai/gpt-oss-120b"
        )

    attacks = ATTACKS if args.attack == "both" else [args.attack]
    dtype = getattr(torch, args.dtype)

    print(
        f"loading guardrail {args.guardrail_model} + J-lens "
        f"(device={args.device}, dtype={args.dtype}) ..."
    )
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)
    print(f"  {gl.model}\n  {gl.lens}")

    # Guardrail + investigator + judge stay resident together the whole run
    # (no phase separation -- see PLAN_runpod_audit.md). Loaded once, reused
    # across attacks.
    local_investigator = local_judge = None
    if args.investigator_backend == "local":
        print(f"loading local investigator {args.investigator_model} (device={args.device}) ...")
        local_investigator = LocalChatModel.load(
            args.investigator_model, dtype=dtype, device=args.device
        )
    if args.judge_backend == "local":
        print(f"loading local judge {args.judge_model} (device={args.device}) ...")
        local_judge = LocalChatModel.load(args.judge_model, dtype=dtype, device=args.device)

    all_scores = [
        run_audit_for_attack(
            attack, gl, args, local_investigator=local_investigator, local_judge=local_judge
        )
        for attack in attacks
    ]

    if len(attacks) > 1:
        combined = pd.concat([s for s in all_scores if not s.empty], ignore_index=True)
        if not combined.empty:
            combined_summary = _summarize(combined)
            combined_path = HERE / "results_audit" / "audit_summary_combined.csv"
            combined_summary.to_csv(combined_path, index=False)
            print(f"\ncombined summary ({' + '.join(attacks)}) -> {combined_path}")
            print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
