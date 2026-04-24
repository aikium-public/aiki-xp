"""Build tier_predictions_lookup_v2.parquet with is_mega + species.

Reads 25 prediction parquets from GCS (5 recipes × 5 folds) and produces a
single lookup keyed on gene_id with columns:
  gene_id, species, is_mega, true_expression,
  tier_a_prediction, tier_b_prediction, tier_b_plus_prediction,
  tier_c_prediction, tier_d_prediction
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from google.cloud import storage  # noqa: F401 (we use gsutil via shell)
import subprocess

GCS_PREFIX = "gs://aikium-data/yotta_display/binding_model/results/recipe_5fold_cv"
LOCAL = Path("/tmp/lookup_rebuild")
LOCAL.mkdir(parents=True, exist_ok=True)

TIER_RECIPES = {
    "tier_a_prediction":      "esmc_prott5_seed42",
    "tier_b_prediction":      "deploy_protein_cds_features_6mod_seed42",
    "tier_b_plus_prediction": "tier_b_evo2_init_window_classical_rna_init_prott5_seed42",
    "tier_c_prediction":      "evo2_prott5_seed42",
    "tier_d_prediction":      "balanced_nonmega_5mod",
}


def download_all():
    for recipe in TIER_RECIPES.values():
        for fold in range(5):
            name = f"{recipe}_fold{fold}_predictions.parquet"
            local = LOCAL / name
            if local.exists() and local.stat().st_size > 0:
                continue
            uri = f"{GCS_PREFIX}/{name}"
            print(f"  gs://...  → {local.name}")
            subprocess.run(["gsutil", "-q", "cp", uri, str(local)], check=True)


def load_tier(tier_col: str, recipe: str) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        local = LOCAL / f"{recipe}_fold{fold}_predictions.parquet"
        df = pd.read_parquet(local)
        df["fold"] = fold
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    n_before = len(out)
    out = out.drop_duplicates("gene_id", keep="first")
    n_after = len(out)
    if n_before != n_after:
        print(f"  {recipe}: dropped {n_before - n_after} duplicate gene_ids")
    return out


def main():
    print("Downloading 25 prediction parquets...")
    download_all()

    print("\nLoading Tier D (will contribute shared species/is_mega/truth)...")
    tier_d = load_tier("tier_d_prediction", TIER_RECIPES["tier_d_prediction"])
    print(f"  {len(tier_d)} rows, cols: {list(tier_d.columns)}")

    base = tier_d[["gene_id", "species", "is_mega", "fold", "y_true"]].copy()
    base = base.rename(columns={"y_true": "true_expression", "fold": "cv_fold"})

    tiers = {"tier_d_prediction": tier_d}
    for tier_col, recipe in TIER_RECIPES.items():
        if tier_col == "tier_d_prediction":
            continue
        print(f"\nLoading {tier_col} ({recipe})...")
        df = load_tier(tier_col, recipe)
        tiers[tier_col] = df

    # Merge on gene_id: start from Tier D's base and attach each tier's y_pred
    out = base
    for tier_col, df in tiers.items():
        pred = df[["gene_id", "y_pred"]].rename(columns={"y_pred": tier_col})
        before = len(out)
        out = out.merge(pred, on="gene_id", how="left")
        n_nan = out[tier_col].isna().sum()
        print(f"  merged {tier_col}: {before} rows, {n_nan} NaN predictions")

    # Reorder columns for output
    cols = ["gene_id", "species", "is_mega", "cv_fold", "true_expression",
            "tier_a_prediction", "tier_b_prediction", "tier_b_plus_prediction",
            "tier_c_prediction", "tier_d_prediction"]
    out = out[cols]

    print("\n=== Final schema ===")
    print(out.dtypes)
    print(f"\nRows: {len(out)}")
    print(f"Unique species: {out['species'].nunique()}")
    print(f"Mega rate: {out['is_mega'].mean():.3%}")

    # NaN check — fail loudly (Rule 1 of CLAUDE.md)
    for col in ["gene_id", "species", "is_mega", "cv_fold", "true_expression"]:
        n_nan = out[col].isna().sum()
        if n_nan > 0:
            raise ValueError(f"Column {col} has {n_nan} NaN — cannot publish")
        print(f"  {col}: 0 NaN ✓")
    for col in [c for c in cols if c.endswith("_prediction")]:
        n_nan = out[col].isna().sum()
        if n_nan > 0:
            print(f"  ⚠ {col}: {n_nan} NaN (genes missing from that tier's fold predictions)")

    out_path = LOCAL / "tier_predictions_lookup_v2.parquet"
    out.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024**2:.2f} MB)")

    # SHA256 for provenance
    import hashlib
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
