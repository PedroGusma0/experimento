#!/usr/bin/env python
"""Build the PIArena corpora for causal pipeline v3 (see
`markdowns-de-referencia/ARCHITECTURE.md`, "Planned -- causal pipeline v3").

Downloads one config of the PIArena benchmark
(https://huggingface.co/datasets/sleeepeer/PIArena) directly from its Hub
parquet file (the `datasets` library's config auto-detection doesn't apply
here -- the repo has no declared builder configs, just one parquet per
config under `data/{config}-00000-of-00001.parquet`). Each row already has
`target_inst`, a *clean* `context`, and a separately-authored `injected_task`
(Figure 3 schema of the paper) -- so building both experimental conditions
needs no extra data, just whether `injected_task` gets inserted:

  data/{config}_clean.csv  -- context used as-is (PIArena's "No Attack")
  data/{config}_direct.csv -- injected_task inserted into context verbatim
                              (PIArena's Direct attack, no rewriting)

Both files carry a shared `sample_index` (0..N-1) so a row in one joins to
its counterpart in the other. Default config is `dolly_closed_qa` (200 rows,
short QA context -- the first test named in ARCHITECTURE.md; the `_long`
configs run 16-19k tokens of context and are deferred).

Reads (downloads, never mutates): the HF Hub parquet for the chosen config.
Writes: the two CSVs above, under `data/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

REPO_ID = "sleeepeer/PIArena"
DEFAULT_CONFIG = "dolly_closed_qa"

#: The paper's 13 main-eval configs (Table 8, 1700 rows total) -- excludes
#: the 3 `_knowledge_corruption` variants (nq/hotpotqa/msmarco_rag), which
#: belong to the separate §5.4 task-alignment experiment, not this one (see
#: ARCHITECTURE.md, Decision 2). Pass ``--config all-main`` to build all 13
#: in one invocation instead of naming them one at a time.
MAIN_CONFIGS: list[str] = [
    "squad_v2",
    "dolly_closed_qa",
    "dolly_information_extraction",
    "dolly_summarization",
    "nq_rag",
    "msmarco_rag",
    "hotpotqa_rag",
    "hotpotqa_long",
    "qasper_long",
    "gov_report_long",
    "multi_news_long",
    "passage_retrieval_en_long",
    "lcc_long",
]

_EXPECTED_COLUMNS = {
    "target_inst",
    "context",
    "injected_task",
    "target_task_answer",
    "injected_task_answer",
    "category",
}
_CATEGORIES = {
    "phishing_injection",
    "content_promotion",
    "access_denial",
    "infrastructure_failure",
}


def download_config(config: str) -> pd.DataFrame:
    """Download one PIArena config's parquet file from the HF Hub and
    validate it against the paper's Figure 3 schema.

    Raises:
        ValueError: If the columns, nulls, empty strings, or category values
            don't match what every PIArena config is documented to contain.
    """
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/{config}-00000-of-00001.parquet",
    )
    df = pd.read_parquet(path)
    if set(df.columns) != _EXPECTED_COLUMNS:
        raise ValueError(
            f"{REPO_ID}/{config} has columns {list(df.columns)}, "
            f"expected {_EXPECTED_COLUMNS}"
        )
    if df.isna().any().any():
        raise ValueError(f"{REPO_ID}/{config}: unexpected null values")
    for col in ("target_inst", "context", "injected_task"):
        if (df[col].str.strip() == "").any():
            raise ValueError(f"{REPO_ID}/{config}: empty {col!r} somewhere")
    bad_categories = set(df["category"]) - _CATEGORIES
    if bad_categories:
        raise ValueError(
            f"{REPO_ID}/{config}: unknown category values {bad_categories}"
        )

    df = df.reset_index(drop=True)
    df.insert(0, "sample_index", range(len(df)))
    return df


def build_clean(df: pd.DataFrame, *, config: str) -> pd.DataFrame:
    """PIArena's own "No Attack" condition: `context` used as-is, nothing
    inserted -- the `clean` half of the `attacked`/`clean` case scheme
    (ARCHITECTURE.md, Decision 4)."""
    clean = df[
        ["sample_index", "target_inst", "context", "target_task_answer", "category"]
    ].copy()
    clean["attacked"] = False
    clean.insert(1, "config", config)
    return clean


def insert_injected_task(context: str, injected_task: str, position: str) -> str:
    """Insert `injected_task` verbatim into `context` -- the Direct attack
    (PIArena Section 4.4: "the injected prompt is then inserted at any
    specified position (beginning, middle, or end)"). No rewriting, no
    attacker LLM -- Direct is the simplest of PIArena's attack modes.
    """
    if position == "end":
        return f"{context} {injected_task}"
    if position == "start":
        return f"{injected_task} {context}"
    if position == "middle":
        mid = len(context) // 2
        # Split on a whitespace boundary near the midpoint so the injection
        # doesn't land inside a word.
        split = context.rfind(" ", 0, mid)
        split = mid if split == -1 else split
        return f"{context[:split]} {injected_task} {context[split:]}"
    raise ValueError(f"unknown position {position!r}, expected start/middle/end")


def build_direct(df: pd.DataFrame, *, config: str, position: str) -> pd.DataFrame:
    """PIArena's Direct attack: `injected_task` inserted verbatim into
    `context` at `position`. Also records the injected task's exact
    character span inside the contaminated context -- not yet consumed by
    the causal driver, but the ground-truth span attribution described in
    ARCHITECTURE.md's Decision 4 needs it, so it's captured here while the
    clean/contaminated strings are already in hand. Carries `config` (the
    source dataset name) on every row -- once multiple configs' `direct`
    rows are combined into one file (see `combine_direct`), `sample_index`
    alone is no longer unique across the combined corpus, only
    `(config, sample_index)` is."""
    rows = []
    for row in df.itertuples():
        contaminated = insert_injected_task(row.context, row.injected_task, position)
        if contaminated.count(row.injected_task) != 1:
            raise ValueError(
                f"sample_index={row.sample_index}: injected_task does not "
                "appear exactly once in the contaminated context (either "
                "missing, or it coincidentally duplicates existing context "
                "text -- span attribution would be ambiguous)"
            )
        span_start = contaminated.index(row.injected_task)
        span_end = span_start + len(row.injected_task)
        rows.append(
            {
                "sample_index": row.sample_index,
                "config": config,
                "target_inst": row.target_inst,
                "context": contaminated,
                "injected_task": row.injected_task,
                "injected_task_answer": row.injected_task_answer,
                "target_task_answer": row.target_task_answer,
                "category": row.category,
                "attacked": True,
                "insert_position": position,
                "injected_task_char_start": span_start,
                "injected_task_char_end": span_end,
            }
        )
    return pd.DataFrame(rows)


def combine_direct(direct_frames: list[pd.DataFrame], *, out_path: Path) -> pd.DataFrame:
    """Merge every config's `direct` rows (already built, one DataFrame per
    config) into a single JSON file -- together, these *are* PIArena's full
    Direct-attack corpus (Table 8's 1700 rows once all 13 main configs are
    built), just split across per-config CSVs until now."""
    combined = pd.concat(direct_frames, ignore_index=True)
    combined.to_json(out_path, orient="records", indent=2, force_ascii=False)
    return combined


def build_one(config: str, position: str) -> pd.DataFrame:
    print(f"downloading {REPO_ID}/{config} ...")
    df = download_config(config)
    print(f"  {len(df)} rows, categories: {dict(df['category'].value_counts())}")

    clean = build_clean(df, config=config)
    clean_path = DATA / f"{config}_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"wrote {len(clean)} rows to {clean_path}")

    direct = build_direct(df, config=config, position=position)
    direct_path = DATA / f"{config}_direct.csv"
    direct.to_csv(direct_path, index=False)
    print(f"wrote {len(direct)} rows to {direct_path} (position={position})")
    return direct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        default=[DEFAULT_CONFIG],
        help="One or more PIArena config names, or the special value "
        "'all-main' to build all 13 main-eval configs (Table 8) in one run.",
    )
    parser.add_argument(
        "--position", default="end", choices=["start", "middle", "end"]
    )
    parser.add_argument(
        "--combined-out",
        default="direct_combined.json",
        help="Filename (under data/) for the merged Direct-attack JSON "
        "across every config built this run. Pass '' to skip writing it.",
    )
    args = parser.parse_args()

    configs = MAIN_CONFIGS if args.config == ["all-main"] else args.config

    DATA.mkdir(exist_ok=True)
    failures = []
    direct_frames = []
    for i, config in enumerate(configs, start=1):
        print(f"\n[{i}/{len(configs)}] {config}")
        try:
            direct_frames.append(build_one(config, args.position))
        except Exception as exc:  # noqa: BLE001 -- one bad config shouldn't sink the batch
            print(f"  [error] {config}: {exc}")
            failures.append(config)

    if len(configs) > 1:
        ok = len(configs) - len(failures)
        print(f"\ndone: {ok}/{len(configs)} configs built"
              + (f", failed: {failures}" if failures else ""))

    if args.combined_out and direct_frames:
        combined_path = DATA / args.combined_out
        combined = combine_direct(direct_frames, out_path=combined_path)
        built_configs = sorted(set(combined["config"]))
        print(f"\nwrote combined Direct-attack corpus: {len(combined)} rows "
              f"across {len(built_configs)} config(s) {built_configs} -> {combined_path}")


if __name__ == "__main__":
    main()
