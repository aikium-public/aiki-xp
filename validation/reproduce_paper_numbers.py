"""Reproduce Aiki-XP manuscript headline numbers from the public lookup parquet.

Downloads `tier_predictions_lookup.parquet` from the Modal volume (or use a
local copy) and computes per-fold-mean Spearman rho per tier, filtered by
mega status. Compare to Table 1 / Fig 2 of the manuscript.

Usage:
    python reproduce_paper_numbers.py /path/to/tier_predictions_lookup.parquet

Expected output (Tier D, seed 42):
    rho_overall  = 0.6675 +/- 0.0139   (paper 0.667 +/- 0.014)
    rho_non_mega = 0.5904 +/- 0.0121   (paper 0.590 +/- 0.012)
    rho_mega     = 0.7388 +/- 0.0269

Any subset filter (species, non-mega, cv_fold) can be applied the same way.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def per_fold_mean(df: pd.DataFrame, pred_col: str, mask: pd.Series | None = None):
    """Spearman rho per cv_fold, then mean +/- std across folds."""
    d = df if mask is None else df[mask]
    rhos = []
    for f in sorted(d["cv_fold"].unique()):
        grp = d[d["cv_fold"] == f]
        if len(grp) < 2:
            continue
        rhos.append(spearmanr(grp["true_expression"], grp[pred_col]).statistic)
    rhos = np.asarray(rhos)
    return rhos.mean(), rhos.std(), rhos


def main(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows, {df['species'].nunique()} species, "
          f"{df['is_mega'].mean():.1%} mega, folds={sorted(df['cv_fold'].unique())}")
    print()

    tiers = ["tier_a", "tier_b", "tier_b_plus", "tier_c", "tier_d"]
    print(f"{'tier':12s}  {'rho_ov':>14s}  {'rho_non_mega':>14s}  {'rho_mega':>14s}")
    for tier in tiers:
        col = f"{tier}_prediction"
        m_ov, s_ov, _ = per_fold_mean(df, col)
        m_nm, s_nm, _ = per_fold_mean(df, col, ~df["is_mega"])
        m_mg, s_mg, _ = per_fold_mean(df, col, df["is_mega"])
        print(f"{tier:12s}  {m_ov:6.4f}+/-{s_ov:.4f}  {m_nm:6.4f}+/-{s_nm:.4f}  "
              f"{m_mg:6.4f}+/-{s_mg:.4f}")

    # Per-species spot-check for the champion (Tier D)
    print()
    print("Tier D per-species rho (top 10 by gene count):")
    counts = df["species"].value_counts().head(10)
    for sp in counts.index:
        sub = df[df["species"] == sp]
        if sub["cv_fold"].nunique() < 2:
            continue
        m, s, _ = per_fold_mean(sub, "tier_d_prediction")
        m_nm, s_nm, _ = per_fold_mean(sub, "tier_d_prediction", ~sub["is_mega"])
        print(f"  {sp:30s}  n={len(sub):>5d}  ov={m:5.3f}+/-{s:.3f}  "
              f"nm={m_nm:5.3f}+/-{s_nm:.3f}  (non-mega n={(~sub['is_mega']).sum()})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
