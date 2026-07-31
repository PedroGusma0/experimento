#!/usr/bin/env python
"""Per-token causal-attribution heatmap for a small, selected set of prompts.

`run_causal_pipeline.py` produces one signed score (`nota`) per swept input
position, but reading that as a table of numbers hides the thing it's for:
seeing, at a glance, *which part of the seed* pushed the guardrail's verdict
and by how much. This renders that as inline-highlighted text over the
prompt itself -- colored by sign (supports/suppresses `malign`), intensity
by magnitude -- following the visual-mapping conventions surveyed in
`papers-de-referencia/exemplos_de_mapeamento_visual_xai.pdf` (Captum's
`visualize_text`, SHAP's `plots.text`), adapted for what `nota` actually is:
an independent per-position ablation effect, not a Shapley-style additive
decomposition, so the page never claims the marks sum to anything.

Deliberately for a **small, selected** set of prompts, not a full corpus --
mirrors `guardrail_eval/make_slices.py`'s scoping rationale: one page per
prompt at scale would be 230 pages nobody reads end to end; the value is in
looking at a handful of well-chosen cases (see `--select` below), not in
exhaustive coverage. Pure post-processing over already-written JSONL --
no model load, no GPU, runs in well under a second.

Reads `<results-dir>/causal_readouts_<attack>.jsonl` +
`causal_position_scores_<attack>.jsonl` (written by `run_causal_pipeline.py`;
`--results-dir` also accepts a locally-downloaded pod results folder, e.g.
`../run_2_pod`). Writes one self-contained HTML file (no external assets,
data embedded inline -- open directly via `file://`, no local server needed).

Example (after a local or downloaded run_causal_pipeline.py run):
    guardrail_eval/.venv/Scripts/python.exe causal_eval/make_readout_viz.py \
        --attack baseline --select top-per-case --per-case 1

Example (against the archived Run 2 pod results, specific prompts):
    guardrail_eval/.venv/Scripts/python.exe causal_eval/make_readout_viz.py \
        --results-dir ../run_2_pod --select pool-index --pool-index 48 227 214
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Canonical display order; any other case value (e.g. "unknown" -- a
# wrapping-attack prompt where the guardrail drifted off a clean verdict,
# see ARCHITECTURE.md's Phase 2 findings) is appended, sorted, after these.
_CASE_ORDER = ("TP", "FP", "FN", "TN")


def slug(attack: str) -> str:
    return attack.replace("-", "_")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `run_causal_pipeline.py --attack ...` "
            "first, or point --results-dir at a folder that has it"
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def seed_only(rows: list[dict]) -> list[dict]:
    """Trim any trailing template tail a *legacy* run may still contain.

    Results written by the current `run_causal_pipeline.py` already stop at
    the seed's own end (`p_end = seed_end`, see ARCHITECTURE.md, "Causal
    validation pipeline v2") -- this is a no-op on that data. It's kept only
    so this script also works unmodified on results_causal/ folders produced
    before that fix (e.g. the archived Run 2 / `run_2_pod/`), which swept all
    the way through the fixed "Classification:" boilerplate.
    """
    rows = sorted(rows, key=lambda r: r["position"])
    cut = next((i for i, r in enumerate(rows) if r["token"] == "Classification"), None)
    if cut is None:
        return rows  # post-fix data: sweep already stops at the seed's own end
    rows = rows[:cut]
    while rows and rows[-1]["token"].strip() == "":
        rows = rows[:-1]
    return rows


def load_examples(results_dir: Path, attack: str) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    s = slug(attack)
    readouts = {
        r["pool_index"]: r for r in _read_jsonl(results_dir / f"causal_readouts_{s}.jsonl")
    }
    scores = _read_jsonl(results_dir / f"causal_position_scores_{s}.jsonl")
    by_pool: dict[int, list[dict]] = {}
    for r in scores:
        by_pool.setdefault(r["pool_index"], []).append(r)
    return readouts, {pi: seed_only(rs) for pi, rs in by_pool.items()}


def select_top_per_case(
    readouts: dict[int, dict], by_pool: dict[int, list[dict]], per_case: int
) -> list[int]:
    """`per_case` prompt(s) per confusion-matrix case, ranked by the largest
    |nota| reached anywhere in that prompt's (seed-only) position span --
    the same criterion used to pick the worked examples first prototyped as
    a Claude Artifact for this pipeline. Cases are discovered from the data
    (not hardcoded to TP/FP/FN/TN), so an "unknown"-verdict prompt is still
    selectable, not silently invisible."""
    by_case: dict[str, list[tuple[int, float]]] = {}
    for pool_index, rows in by_pool.items():
        if not rows:
            continue
        case = readouts.get(pool_index, {}).get("case", "unknown")
        peak = max(abs(r["nota"]) for r in rows)
        by_case.setdefault(case, []).append((pool_index, peak))
    cases = list(_CASE_ORDER) + sorted(c for c in by_case if c not in _CASE_ORDER)
    selected: list[int] = []
    for case in cases:
        # tie-break by pool_index ascending, for determinism
        ranked = sorted(by_case.get(case, []), key=lambda x: (-x[1], x[0]))
        selected.extend(pi for pi, _ in ranked[:per_case])
    return selected


def build_html(
    readouts: dict[int, dict], by_pool: dict[int, list[dict]], selected: list[int], attack: str
) -> str:
    examples = []
    for pool_index in selected:
        rows = by_pool.get(pool_index, [])
        if not rows:
            continue
        ro = readouts.get(pool_index, {})
        examples.append(
            {
                "pool": pool_index,
                "case": ro.get("case", "unknown"),
                "label_true": ro.get("label_true", "?"),
                "verdict": ro.get("verdict", "?"),
                "score_clean": rows[0]["score_clean"],
                "rows": [
                    {
                        "t": r["token"],
                        "n": r["nota"],
                        "kl": r["kl_ablate"],
                        "c": r["candidatos"],
                    }
                    for r in rows
                ],
            }
        )
    # `</` -> `<\/` (not just `</script`) so a vocab string can't close the
    # <script> tag regardless of what follows -- matches jlens.vis.build_page's
    # own precedent for embedding untrusted vocab strings into an HTML page.
    examples_json = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE_TEMPLATE.replace("__EXAMPLES_JSON__", examples_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--attack", choices=["baseline", "baseline-wrapping"], default="baseline")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Folder with causal_readouts_<attack>.jsonl / "
        "causal_position_scores_<attack>.jsonl. Defaults to "
        "causal_eval/results_causal/ (accepts any downloaded results folder, "
        "e.g. a pod run's run_2_pod/).",
    )
    parser.add_argument(
        "--select",
        choices=["top-per-case", "pool-index"],
        default="top-per-case",
        help="'top-per-case': highest peak |nota| per TP/FP/FN/TN case present. "
        "'pool-index': explicit --pool-index list.",
    )
    parser.add_argument("--per-case", type=int, default=1)
    parser.add_argument("--pool-index", type=int, nargs="+", default=None)
    parser.add_argument("--out", default=None, help="Defaults to <results-dir>/readout_viz_<attack>.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else HERE / "results_causal"
    readouts, by_pool = load_examples(results_dir, args.attack)
    print(f"{len(readouts)} prompt(s) in {results_dir} (attack={args.attack!r})")

    if args.select == "pool-index":
        if not args.pool_index:
            raise ValueError("--select pool-index requires --pool-index <id> [<id> ...]")
        selected = args.pool_index
    else:
        selected = select_top_per_case(readouts, by_pool, args.per_case)

    missing = [pi for pi in selected if pi not in by_pool or not by_pool[pi]]
    if missing:
        print(f"  [warn] no seed-span positions for pool_index {missing}, skipping")
    selected = [pi for pi in selected if pi not in missing]
    if not selected:
        print("no rows to render")
        return
    print(f"rendering {selected}")

    html = build_html(readouts, by_pool, selected, args.attack)
    out_path = Path(args.out) if args.out else results_dir / f"readout_viz_{slug(args.attack)}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"-> {out_path}")


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>readout</title>
<style>
  :root {
    --bg: #ffffff;
    --ink: #1a1a1a;
    --ink-muted: #808080;
    --line: #e6e6e6;
    --neg: #FF0051;
    --pos: #008BFB;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #111111; --ink: #eaeaea; --ink-muted: #8a8a8a; --line: #2a2a2a; --neg: #FF3D74; --pos: #3FA7FF; }
  }
  :root[data-theme="dark"] { --bg: #111111; --ink: #eaeaea; --ink-muted: #8a8a8a; --line: #2a2a2a; --neg: #FF3D74; --pos: #3FA7FF; }
  :root[data-theme="light"] { --bg: #ffffff; --ink: #1a1a1a; --ink-muted: #808080; --line: #e6e6e6; --neg: #FF0051; --pos: #008BFB; }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--ink);
    font-family: Helvetica, Arial, sans-serif;
    font-size: 14px; -webkit-font-smoothing: antialiased;
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  main {
    max-width: 860px; margin: 0 auto; padding: 28px 20px;
    display: flex; flex-direction: column; gap: 26px;
  }
  .case { padding-bottom: 22px; border-bottom: 1px solid var(--line); }
  .case:last-child { border-bottom: none; padding-bottom: 0; }
  .case-head {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
    font-size: 11px; color: var(--ink-muted); margin: 0 0 10px;
    font-variant-numeric: tabular-nums;
  }
  .case-head b { color: var(--ink); font-weight: 600; }
  .tokens { font-size: 15px; line-height: 2.6; white-space: pre-wrap; }
  .tok { --a: 0; padding: 2px 1px; border-radius: 2px; }
  .tok.neg { background: color-mix(in srgb, var(--neg) calc(var(--a) * 100%), transparent); }
  .tok.pos { background: color-mix(in srgb, var(--pos) calc(var(--a) * 100%), transparent); }
  .tok .cal {
    display: inline-block; font-size: 9px; vertical-align: super;
    font-variant-numeric: tabular-nums; margin-left: 1px;
  }
  .tok.neg .cal { color: var(--neg); }
  .tok.pos .cal { color: var(--pos); }
</style>
</head>
<body>
<main>
  <div id="cases"></div>
</main>

<script>
  const EXAMPLES = __EXAMPLES_JSON__;
  const container = document.getElementById("cases");

  EXAMPLES.forEach(ex => {
    const maxAbs = Math.max(...ex.rows.map(r => Math.abs(r.n)), 1e-9);
    const logMax = Math.log1p(maxAbs) || 1;
    const topN = [...ex.rows].sort((a, b) => Math.abs(b.n) - Math.abs(a.n)).slice(0, 3);
    const callouts = new Set(topN.map(r => ex.rows.indexOf(r)));

    const article = document.createElement("article");
    article.className = "case";

    const head = document.createElement("div");
    head.className = "case-head";
    head.innerHTML = `<span>#${ex.pool}</span><b>${ex.case}</b>` +
      `<span>${ex.label_true} → ${ex.verdict}</span><span>${ex.score_clean.toFixed(2)}</span>`;
    article.appendChild(head);

    const tokWrap = document.createElement("div");
    tokWrap.className = "tokens";

    ex.rows.forEach((r, i) => {
      const span = document.createElement("span");
      const alpha = 0.08 + 0.80 * Math.min(1, Math.log1p(Math.abs(r.n)) / logMax);
      span.className = "tok " + (r.n < 0 ? "neg" : "pos");
      span.style.setProperty("--a", alpha.toFixed(3));
      span.textContent = r.t;
      // native title attribute -- browser escapes it, no HTML injection risk
      // even though token/candidatos are decoded, attacker-influenced vocab strings
      span.title = `nota ${r.n > 0 ? "+" : ""}${r.n.toFixed(4)} | kl_ablate ${r.kl.toExponential(2)} | ${r.c.join(", ")}`;
      if (callouts.has(i)) {
        const cal = document.createElement("span");
        cal.className = "cal";
        cal.textContent = (r.n > 0 ? "+" : "") + r.n.toFixed(2);
        span.appendChild(cal);
      }
      tokWrap.appendChild(span);
    });

    article.appendChild(tokWrap);
    container.appendChild(article);
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
