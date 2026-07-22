#!/usr/bin/env python
"""Build the labeled HarmBench corpus for the guardrail baseline eval.

Reads the repo-root ``harmbench.csv`` (never mutated) and writes
``data/harmbench_labeled.csv`` with an added ``label`` column, filled with
``"malign"`` for every row (HarmBench prompts are all harmful by
construction; this baseline has no benign examples yet).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "harmbench.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "harmbench_labeled.csv"

EXPECTED_ROWS = 200
EXPECTED_COLUMNS = {"prompt", "category"}


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    if set(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{INPUT_CSV} has columns {list(df.columns)}, expected "
            f"{sorted(EXPECTED_COLUMNS)}"
        )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{INPUT_CSV} has {len(df)} rows, expected {EXPECTED_ROWS}")

    df["label"] = "malign"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"wrote {len(df)} labeled rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
