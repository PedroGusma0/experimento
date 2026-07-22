#!/usr/bin/env python
"""Build the attack corpora for Phase 2: a unified benign+malign seed pool,
and two attack-specific prompt datasets (baseline, baseline-wrapping).

Reads (never mutates):
  data/harmbench_labeled.csv        (200 rows, prompt/category/label=malign)
  data/jailbreakbench_benign_en.csv (30 rows, id/category/intent)
  data/system_variants_en.csv       (18 rows, wrapping templates)

Writes:
  data/seed_pool.csv               (230 rows: 200 malign + 30 benign, unified)
  data/attack_baseline.csv         (230 rows: prompt = seed, no transform)
  data/attack_baseline_wrapping.csv (230 rows: seed wrapped into one of the
                                      18 variants, cycling by pool_index % 18)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

HARMBENCH_CSV = DATA / "harmbench_labeled.csv"
BENIGN_CSV = DATA / "jailbreakbench_benign_en.csv"
VARIANTS_CSV = DATA / "system_variants_en.csv"

SEED_POOL_CSV = DATA / "seed_pool.csv"
ATTACK_BASELINE_CSV = DATA / "attack_baseline.csv"
ATTACK_WRAPPING_CSV = DATA / "attack_baseline_wrapping.csv"

N_MALIGN = 200
N_BENIGN = 30
N_POOL = N_MALIGN + N_BENIGN
N_VARIANTS = 18
PLACEHOLDER = "[INSERT PROMPT HERE]"


def build_seed_pool() -> pd.DataFrame:
    harmbench = pd.read_csv(HARMBENCH_CSV)
    if len(harmbench) != N_MALIGN or set(harmbench.columns) != {
        "prompt",
        "category",
        "label",
    }:
        raise ValueError(
            f"{HARMBENCH_CSV} has {len(harmbench)} rows / columns "
            f"{list(harmbench.columns)}, expected {N_MALIGN} rows / "
            "{prompt, category, label}"
        )
    if not (harmbench["label"] == "malign").all():
        raise ValueError(f"{HARMBENCH_CSV}: not every row has label='malign'")

    benign = pd.read_csv(BENIGN_CSV)
    if len(benign) != N_BENIGN or set(benign.columns) != {
        "id",
        "category",
        "intent",
    }:
        raise ValueError(
            f"{BENIGN_CSV} has {len(benign)} rows / columns "
            f"{list(benign.columns)}, expected {N_BENIGN} rows / "
            "{id, category, intent}"
        )

    malign_rows = pd.DataFrame(
        {
            "seed": harmbench["prompt"],
            "label": "malign",
            "category": harmbench["category"],
            "source": "harmbench",
            "source_id": range(N_MALIGN),
        }
    )
    benign_rows = pd.DataFrame(
        {
            "seed": benign["intent"],
            "label": "benign",
            "category": benign["category"],
            "source": "jailbreakbench_benign",
            "source_id": benign["id"],
        }
    )

    pool = pd.concat([malign_rows, benign_rows], ignore_index=True)
    pool.insert(0, "pool_index", range(len(pool)))

    if len(pool) != N_POOL:
        raise ValueError(f"seed pool has {len(pool)} rows, expected {N_POOL}")
    if (pool["pool_index"] != range(N_POOL)).any():
        raise ValueError("pool_index is not a contiguous 0..N-1 range")
    if (pool["label"] == "malign").sum() != N_MALIGN or (
        pool["label"] == "benign"
    ).sum() != N_BENIGN:
        raise ValueError("label counts in seed pool do not match 200 malign / 30 benign")
    if pool["seed"].isna().any() or (pool["seed"].str.strip() == "").any():
        raise ValueError("seed pool contains an empty seed")

    return pool


def build_attack_baseline(pool: pd.DataFrame) -> pd.DataFrame:
    attack = pool.copy()
    attack["prompt"] = attack["seed"]
    attack["attack"] = "baseline"
    if not (attack["prompt"] == attack["seed"]).all():
        raise ValueError("baseline attack: prompt != seed somewhere")
    return attack


def build_attack_wrapping(pool: pd.DataFrame) -> pd.DataFrame:
    variants = pd.read_csv(VARIANTS_CSV)
    if len(variants) != N_VARIANTS or list(variants.columns) != [
        "prompt_type",
        "attack_category",
        "attack_technique",
        "attack_pattern",
        "Prompt",
    ]:
        raise ValueError(
            f"{VARIANTS_CSV} has {len(variants)} rows / columns "
            f"{list(variants.columns)}, expected {N_VARIANTS} rows / "
            "[prompt_type, attack_category, attack_technique, attack_pattern, Prompt]"
        )
    bad = [i for i, p in enumerate(variants["Prompt"]) if p.count(PLACEHOLDER) != 1]
    if bad:
        raise ValueError(f"variant rows {bad} do not contain exactly one placeholder")

    rows = []
    for pool_row in pool.itertuples(index=False):
        variant_index = pool_row.pool_index % N_VARIANTS
        variant = variants.iloc[variant_index]
        prompt = variant["Prompt"].replace(PLACEHOLDER, pool_row.seed)
        if PLACEHOLDER in prompt:
            raise ValueError(f"pool_index {pool_row.pool_index}: placeholder still present")
        if pool_row.seed not in prompt:
            raise ValueError(f"pool_index {pool_row.pool_index}: seed missing from wrapped prompt")
        rows.append(
            {
                "pool_index": pool_row.pool_index,
                "seed": pool_row.seed,
                "label": pool_row.label,
                "category": pool_row.category,
                "source": pool_row.source,
                "source_id": pool_row.source_id,
                "prompt": prompt,
                "attack": "baseline-wrapping",
                "variant_index": variant_index,
                "variant_prompt_type": variant["prompt_type"],
                "variant_attack_category": variant["attack_category"],
                "variant_attack_technique": variant["attack_technique"],
                "variant_attack_pattern": variant["attack_pattern"],
            }
        )

    attack = pd.DataFrame(rows)
    if len(attack) != N_POOL:
        raise ValueError(f"wrapping attack has {len(attack)} rows, expected {N_POOL}")
    if (attack["variant_index"] != attack["pool_index"] % N_VARIANTS).any():
        raise ValueError("variant_index does not match pool_index % N_VARIANTS somewhere")
    return attack


def main() -> None:
    pool = build_seed_pool()
    pool.to_csv(SEED_POOL_CSV, index=False)
    print(f"wrote {len(pool)} rows to {SEED_POOL_CSV}")

    baseline = build_attack_baseline(pool)
    baseline.to_csv(ATTACK_BASELINE_CSV, index=False)
    print(f"wrote {len(baseline)} rows to {ATTACK_BASELINE_CSV}")

    wrapping = build_attack_wrapping(pool)
    wrapping.to_csv(ATTACK_WRAPPING_CSV, index=False)
    print(f"wrote {len(wrapping)} rows to {ATTACK_WRAPPING_CSV}")

    print("\nsample rows at cycle boundaries (pool_index 0, 17, 18, 35, 36):")
    for idx in (0, 17, 18, 35, 36):
        row = wrapping.iloc[idx]
        print(
            f"  pool_index={row['pool_index']:>3}  variant_index={row['variant_index']:>2}  "
            f"variant_prompt_type={row['variant_prompt_type']:<9}  label={row['label']}"
        )


if __name__ == "__main__":
    main()
