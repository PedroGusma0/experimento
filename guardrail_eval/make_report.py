#!/usr/bin/env python
"""Static XAI report (Markdown + PDF) over a guardrail_eval audit run.

Consolidates the two interpretability artifacts `guardrail_eval` produces
separately -- aggregate auditor metrics (`run_audit_pipeline.py`'s
`audit_readouts_<attack>.jsonl` / `audit_scores_<attack>.jsonl` /
`audit_summary_<attack>.csv`) and J-lens position x layer evidence
(`make_slices.py`) -- into one document, covering both attacks
(`baseline` + `baseline-wrapping`) plus their comparison. See
`markdowns-de-referencia/VISUALIZATION.md` for the rationale behind the
J-lens side of this.

Deliberately a **standalone script**, run manually after
`run_audit_pipeline.py` finishes: it only ever reads already-written
JSONL/CSV files, so nothing it does (including a crash) can affect a run in
progress or already-written results. Runs in two independent phases:

- **Phase 1 (always runs)**: aggregate charts from the auditor's own
  output -- guardrail classification outcome (TP/FP/FN/TN/unknown), per-claim
  investigator accuracy, judge score distribution, investigator tool-call
  usage. No model load required. An attack with no data on disk gets a
  "skipped" note instead of a fatal error.
- **Phase 2 (`--skip-slices` to disable)**: loads the guardrail + J-lens
  (same `GuardrailLens` the audit loop uses) and renders a static rank
  heatmap (position x layer, pinned malign/benign concept vocabulary) for a
  handful of error rows per attack (`--max-slice-rows`, reusing
  `make_slices.py`'s row-selection logic). Each row's snapshot runs in its
  own try/except so one bad row can't blank the rest of the report.

Example (after `run_audit_pipeline.py --attack both ...`):
    python make_report.py --max-slice-rows 3 --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ATTACKS = ["baseline", "baseline-wrapping"]

# Workspace band -- matches run_audit_pipeline.py / make_slices.py.
DEFAULT_LAYERS = list(range(14, 27))


def slug(attack: str) -> str:
    return attack.replace("-", "_")


def _results_paths(results_dir: Path, attack: str) -> dict[str, Path]:
    s = slug(attack)
    return {
        "readouts": results_dir / f"audit_readouts_{s}.jsonl",
        "scores": results_dir / f"audit_scores_{s}.jsonl",
        "summary": results_dir / f"audit_summary_{s}.csv",
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class AttackData:
    attack: str
    readouts: list[dict]
    scores: list[dict]
    summary: pd.DataFrame  # empty if audit_summary_<attack>.csv is missing

    @property
    def available(self) -> bool:
        return bool(self.readouts)


def load_attack_data(results_dir: Path, attack: str) -> AttackData:
    paths = _results_paths(results_dir, attack)
    readouts = _read_jsonl(paths["readouts"])
    scores = _read_jsonl(paths["scores"])
    summary = pd.read_csv(paths["summary"]) if paths["summary"].exists() else pd.DataFrame()
    return AttackData(attack, readouts, scores, summary)


def _confusion_counts(readouts: list[dict]) -> dict[str, int]:
    """TP/FP/FN/TN/unknown from label_true x guardrail_label_pred."""
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "unknown": 0}
    for r in readouts:
        label, pred = r["label_true"], r["guardrail_label_pred"]
        if pred == "unknown":
            counts["unknown"] += 1
        elif label == "malign" and pred == "malign":
            counts["TP"] += 1
        elif label == "benign" and pred == "benign":
            counts["TN"] += 1
        elif label == "benign" and pred == "malign":
            counts["FP"] += 1
        elif label == "malign" and pred == "benign":
            counts["FN"] += 1
    return counts


def _grouped_bar(ax, labels: list[str], series: dict[str, list[float]]) -> None:
    """Draw one bar cluster per label, one bar per series, on ``ax``."""
    n = len(series)
    width = 0.8 / max(n, 1)
    x = range(len(labels))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        ax.bar([xi + offset for xi in x], values, width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()


def _plot_confusion(attacks_data: list[AttackData], assets_dir: Path) -> Path | None:
    available = [d for d in attacks_data if d.available]
    if not available:
        return None
    categories = ["TP", "TN", "FP", "FN", "unknown"]
    series = {d.attack: [_confusion_counts(d.readouts)[c] for c in categories] for d in available}
    fig, ax = plt.subplots(figsize=(6, 4))
    _grouped_bar(ax, categories, series)
    ax.set_ylabel("prompts")
    ax.set_title("Guardrail classification outcome (label_true x prediction)")
    fig.tight_layout()
    out = assets_dir / "confusion.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_claim_accuracy(attacks_data: list[AttackData], assets_dir: Path) -> Path | None:
    available = [d for d in attacks_data if not d.summary.empty]
    if not available:
        return None
    claim_ids = sorted({cid for d in available for cid in d.summary["claim_id"]})
    series = {}
    for d in available:
        by_claim = d.summary.set_index("claim_id")["investigator_accuracy"]
        series[d.attack] = [float(by_claim.get(cid, 0.0)) for cid in claim_ids]
    fig, ax = plt.subplots(figsize=(6, 4))
    _grouped_bar(ax, claim_ids, series)
    ax.set_ylim(0, 1)
    ax.set_ylabel("investigator accuracy")
    ax.set_title("Investigator accuracy by claim")
    fig.tight_layout()
    out = assets_dir / "claim_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_score_distribution(attacks_data: list[AttackData], assets_dir: Path) -> Path | None:
    available = [d for d in attacks_data if d.scores]
    if not available:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in available:
        scores = [s["score"] for s in d.scores]
        ax.hist(scores, bins=range(0, 12), alpha=0.6, label=d.attack)
    ax.set_xlabel("judge score (0-10, mean of correctness/evidence_quality)")
    ax.set_ylabel("count")
    ax.set_title("Judge score distribution")
    ax.legend()
    fig.tight_layout()
    out = assets_dir / "score_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_tool_calls(attacks_data: list[AttackData], assets_dir: Path) -> Path | None:
    available = [d for d in attacks_data if d.scores]
    if not available:
        return None
    max_calls = max((s["tool_calls"] for d in available for s in d.scores), default=0)
    bins = range(0, max_calls + 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in available:
        calls = [s["tool_calls"] for s in d.scores]
        ax.hist(calls, bins=bins, alpha=0.6, label=d.attack, align="left")
    ax.set_xlabel("tool_calls used by investigator")
    ax.set_ylabel("count")
    ax.set_title("Investigator tool-call usage")
    ax.legend()
    fig.tight_layout()
    out = assets_dir / "tool_calls.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _pinned_rank_heatmap(slice_data, pinned_ids: set[int]) -> np.ndarray | None:
    """Best (lowest) rank across pinned tokens at every (position, layer)."""
    idx = [i for i, tid in enumerate(slice_data.tracked_token_ids) if tid in pinned_ids]
    if not idx:
        return None
    sub = slice_data.rank_tensor[:, :, idx]  # [seq_len, n_layers, k]
    return sub.min(axis=-1)  # [seq_len, n_layers]


def _plot_slice_snapshot(slice_data, pinned_ids: set[int], title: str, out_path: Path) -> Path | None:
    heat = _pinned_rank_heatmap(slice_data, pinned_ids)
    if heat is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(np.log10(heat.T + 1), aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_yticks(range(len(slice_data.layers)))
    ax.set_yticklabels(slice_data.layers)
    n_pos = heat.shape[0]
    step = max(1, n_pos // 10)
    xticks = list(range(0, n_pos, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([slice_data.ctx_offset + x for x in xticks])
    ax.set_xlabel("prompt position")
    ax.set_ylabel("layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(best pinned-concept rank + 1) -- lower = more salient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_slice_snapshots(
    attacks_data: list[AttackData], args: argparse.Namespace, assets_dir: Path
) -> list[tuple[str, dict, Path]]:
    """Phase 2: render one static rank-heatmap PNG per selected error row.

    Returns ``[(attack, row, png_path), ...]``. A per-row failure is caught
    and warned about, not raised -- one bad prompt shouldn't blank the rest
    of the report.
    """
    import torch

    from jlens.vis import compute_slice
    from jlens_readout import GuardrailLens
    from make_slices import CONCEPT_VOCAB, narrow_lens, pinned_token_ids, select_rows

    print(f"loading guardrail {args.guardrail_model} (device={args.device}) ...")
    dtype = getattr(torch, args.dtype)
    gl = GuardrailLens(args.guardrail_model, dtype=dtype, device=args.device)
    lens = narrow_lens(gl.lens, args.layers)
    pins = pinned_token_ids(gl.tok, CONCEPT_VOCAB)
    last_n = args.last_n_tokens or None

    snapshots: list[tuple[str, dict, Path]] = []
    for d in attacks_data:
        if not d.available:
            continue
        rows = select_rows(d.readouts, "errors", None)[: args.max_slice_rows]
        for row in rows:
            pool_index = row["pool_index"]
            try:
                prompt = gl.chat_prompt(row["prompt"])
                slice_data = compute_slice(
                    gl.model, lens, prompt, top_n=10, pinned_token_ids=pins, last_n_tokens=last_n
                )
                out_path = assets_dir / f"slice_{slug(d.attack)}_{pool_index}.png"
                title = f"{d.attack} #{pool_index} ({row['label_true']} -> {row['guardrail_label_pred']})"
                _plot_slice_snapshot(slice_data, pins, title, out_path)
                snapshots.append((d.attack, row, out_path))
                print(f"  [{d.attack}] pool_index={pool_index} -> {out_path.name}")
            except Exception as exc:  # noqa: BLE001 - isolate one bad row
                print(f"  [warn] slice failed for {d.attack}#{pool_index}: {exc}")
    return snapshots


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _md_table(records: list[dict]) -> str:
    if not records:
        return "_no rows_"
    cols = list(records[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in records:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_markdown(
    attacks_data: list[AttackData],
    chart_paths: dict[str, Path | None],
    snapshots: list[tuple[str, dict, Path]],
    docs_dir: Path,
    assets_dir: Path,
) -> Path:
    a = assets_dir.name
    lines = ["# guardrail_eval — XAI report", ""]

    lines.append("## Guardrail classification outcome")
    if chart_paths.get("confusion"):
        lines.append(f"![confusion]({a}/{chart_paths['confusion'].name})")
    lines.append("")
    for d in attacks_data:
        if not d.available:
            lines.append(f"_No data for `{d.attack}` (audit_readouts_{slug(d.attack)}.jsonl not found -- skipped)._")
            continue
        counts = _confusion_counts(d.readouts)
        lines.append(f"**{d.attack}** (n={len(d.readouts)}): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines.append("")

    lines.append("## Auditor performance")
    if chart_paths.get("claim_accuracy"):
        lines.append(f"![claim accuracy]({a}/{chart_paths['claim_accuracy'].name})")
    lines.append("")
    for d in attacks_data:
        if d.summary.empty:
            continue
        lines.append(f"### {d.attack}")
        lines.append(_md_table(d.summary.to_dict("records")))
        lines.append("")

    if chart_paths.get("score_distribution"):
        lines.append("## Judge score distribution")
        lines.append(f"![scores]({a}/{chart_paths['score_distribution'].name})")
        lines.append("")

    if chart_paths.get("tool_calls"):
        lines.append("## Investigator tool-call usage")
        lines.append(f"![tool calls]({a}/{chart_paths['tool_calls'].name})")
        for d in attacks_data:
            if not d.scores:
                continue
            n = len(d.scores)
            fb = sum(1 for s in d.scores if s.get("fallback_used"))
            lines.append(f"- **{d.attack}**: fallback_used in {fb}/{n} claims ({fb / n:.0%})")
        lines.append("")

    if snapshots:
        lines.append("## J-lens spatial evidence (selected error rows)")
        lines.append(
            "Heatmap of the best (lowest) rank across a pinned malign/benign "
            "concept vocabulary, at every (position, layer) -- darker/lower = "
            "the guardrail's internal activation is more disposed to verbalize "
            "that concept there. See `markdowns-de-referencia/VISUALIZATION.md`."
        )
        lines.append("")
        for attack, row, png in snapshots:
            lines.append(f"### {attack} #{row['pool_index']} — {row['label_true']} -> {row['guardrail_label_pred']}")
            lines.append(f"seed: {row['seed'][:200]!r}")
            lines.append(f"![slice]({a}/{png.name})")
            lines.append(
                f"_interactive version:_ `python make_slices.py --attack {attack} "
                f"--select pool-index --pool-index {row['pool_index']}`"
            )
            lines.append("")

    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "xai_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _pdf_table(records: list[dict]):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, Table, TableStyle

    if not records:
        return Paragraph("no rows", getSampleStyleSheet()["BodyText"])
    cols = list(records[0].keys())
    data = [cols] + [[_fmt(r[c]) for c in cols] for r in records]
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    return t


def write_pdf(
    attacks_data: list[AttackData],
    chart_paths: dict[str, Path | None],
    snapshots: list[tuple[str, dict, Path]],
    docs_dir: Path,
) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = [Paragraph("guardrail_eval — XAI report", styles["Title"]), Spacer(1, 12)]

    story.append(Paragraph("Guardrail classification outcome", styles["Heading2"]))
    if chart_paths.get("confusion"):
        story.append(Image(str(chart_paths["confusion"]), width=5.5 * inch, height=3.7 * inch))
    for d in attacks_data:
        if not d.available:
            story.append(Paragraph(f"No data for {d.attack} (skipped).", styles["BodyText"]))
            continue
        counts = _confusion_counts(d.readouts)
        story.append(
            Paragraph(
                f"<b>{d.attack}</b> (n={len(d.readouts)}): "
                + ", ".join(f"{k}={v}" for k, v in counts.items()),
                styles["BodyText"],
            )
        )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Auditor performance", styles["Heading2"]))
    if chart_paths.get("claim_accuracy"):
        story.append(Image(str(chart_paths["claim_accuracy"]), width=5.5 * inch, height=3.7 * inch))
    for d in attacks_data:
        if d.summary.empty:
            continue
        story.append(Paragraph(d.attack, styles["Heading3"]))
        story.append(_pdf_table(d.summary.to_dict("records")))
        story.append(Spacer(1, 8))

    if chart_paths.get("score_distribution"):
        story.append(Paragraph("Judge score distribution", styles["Heading2"]))
        story.append(Image(str(chart_paths["score_distribution"]), width=5.5 * inch, height=3.7 * inch))

    if chart_paths.get("tool_calls"):
        story.append(Paragraph("Investigator tool-call usage", styles["Heading2"]))
        story.append(Image(str(chart_paths["tool_calls"]), width=5.5 * inch, height=3.7 * inch))
        for d in attacks_data:
            if not d.scores:
                continue
            n = len(d.scores)
            fb = sum(1 for s in d.scores if s.get("fallback_used"))
            story.append(
                Paragraph(f"{d.attack}: fallback_used in {fb}/{n} claims ({fb / n:.0%})", styles["BodyText"])
            )

    if snapshots:
        story.append(PageBreak())
        story.append(Paragraph("J-lens spatial evidence (selected error rows)", styles["Heading2"]))
        for attack, row, png in snapshots:
            story.append(
                Paragraph(
                    f"{attack} #{row['pool_index']} — {row['label_true']} -> {row['guardrail_label_pred']}",
                    styles["Heading3"],
                )
            )
            story.append(Paragraph(f"seed: {row['seed'][:200]}", styles["BodyText"]))
            story.append(Image(str(png), width=6.5 * inch, height=3.25 * inch))
            story.append(Spacer(1, 12))

    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "xai_report.pdf"
    SimpleDocTemplate(str(out), pagesize=letter).build(story)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir", default=None, help="Defaults to results_audit/ next to this script."
    )
    parser.add_argument("--docs-dir", default=None, help="Defaults to docs/ next to this script.")
    parser.add_argument("--format", choices=["md", "pdf", "both"], default="both")
    parser.add_argument(
        "--skip-slices",
        action="store_true",
        help="Skip Phase 2 (J-lens snapshots) -- report is metrics-only, no guardrail load.",
    )
    parser.add_argument(
        "--max-slice-rows",
        type=int,
        default=3,
        help="Cap on error rows rendered per attack in Phase 2.",
    )
    parser.add_argument("--guardrail-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--last-n-tokens", type=int, default=80)
    parser.add_argument("--device", default="cpu", help='"cpu" or "cuda".')
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else HERE / "results_audit"
    docs_dir = Path(args.docs_dir) if args.docs_dir else HERE / "docs"
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    attacks_data = [load_attack_data(results_dir, a) for a in ATTACKS]
    if not any(d.available for d in attacks_data):
        raise FileNotFoundError(
            f"no audit_readouts_<attack>.jsonl found under {results_dir} -- "
            "run run_audit_pipeline.py first"
        )
    for d in attacks_data:
        if not d.available:
            print(f"[skip] no data for attack={d.attack!r} under {results_dir}")

    chart_paths = {
        "confusion": _plot_confusion(attacks_data, assets_dir),
        "claim_accuracy": _plot_claim_accuracy(attacks_data, assets_dir),
        "score_distribution": _plot_score_distribution(attacks_data, assets_dir),
        "tool_calls": _plot_tool_calls(attacks_data, assets_dir),
    }

    snapshots: list[tuple[str, dict, Path]] = []
    if args.skip_slices:
        print("--skip-slices: Phase 2 (J-lens snapshots) skipped")
    else:
        snapshots = generate_slice_snapshots(attacks_data, args, assets_dir)

    if args.format in ("md", "both"):
        out = write_markdown(attacks_data, chart_paths, snapshots, docs_dir, assets_dir)
        print(f"wrote {out}")
    if args.format in ("pdf", "both"):
        out = write_pdf(attacks_data, chart_paths, snapshots, docs_dir)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
