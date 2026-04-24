#!/usr/bin/env python3
"""Build supplementary tables for NBT submission as Excel workbooks."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parent.parent / "zenodo_staging"
RESULTS = STAGING / "results"
OUTPUT = Path(__file__).resolve().parent.parent / "supplementary_tables"


def load_5fold_recipes() -> dict:
    """Load all 5-fold recipe results, compute mean/std."""
    recipe_dir = RESULTS / "recipe_5fold_cv"
    recipes = defaultdict(list)
    for jf in sorted(recipe_dir.glob("*.json")):
        name = jf.stem
        if "_fold" not in name:
            continue
        base = name.rsplit("_fold", 1)[0]
        with open(jf) as f:
            d = json.load(f)
        sa = d.get("results", {}).get("single_adapter", {})
        if not sa:
            continue
        recipes[base].append({
            "fold": name.rsplit("_fold", 1)[1].split("_")[0],
            "rho_overall": sa.get("mean_spearman"),
            "rho_nc": sa.get("rho_non_mega"),
            "n_modalities": len(d.get("embedders", [])),
            "params": sa.get("mean_model_trainable_params"),
            "embedders": d.get("embedders", []),
        })
    return recipes


def table_recipe_comparison() -> pd.DataFrame:
    recipes = load_5fold_recipes()
    rows = []
    for name, folds in sorted(recipes.items()):
        if len(folds) < 3:
            continue
        rho_ov = [f["rho_overall"] for f in folds if f["rho_overall"] is not None]
        rho_nc = [f["rho_nc"] for f in folds if f["rho_nc"] is not None]
        rows.append({
            "recipe": name,
            "n_modalities": folds[0]["n_modalities"],
            "n_folds": len(folds),
            "rho_overall_mean": f"{np.mean(rho_ov):.4f}" if rho_ov else "",
            "rho_overall_std": f"{np.std(rho_ov, ddof=1):.4f}" if len(rho_ov) > 1 else "",
            "rho_nc_mean": f"{np.mean(rho_nc):.4f}" if rho_nc else "",
            "rho_nc_std": f"{np.std(rho_nc, ddof=1):.4f}" if len(rho_nc) > 1 else "",
            "per_fold_rho_nc": ", ".join(f"{r:.4f}" for r in rho_nc) if rho_nc else "",
            "modalities": ", ".join(folds[0]["embedders"]),
        })
    df = pd.DataFrame(rows)
    if "rho_nc_mean" in df.columns and len(df) > 0:
        df = df.sort_values("rho_nc_mean", ascending=False)
    return df


def table_losco_holdouts() -> pd.DataFrame:
    losco_dir = RESULTS / "losco_xp5"
    rows = []
    for cluster_dir in sorted(losco_dir.iterdir()):
        if not cluster_dir.is_dir():
            continue
        jsons = list(cluster_dir.glob("fusion_results_*.json"))
        if not jsons:
            continue
        with open(jsons[0]) as f:
            d = json.load(f)
        sa = d["results"]["single_adapter"]
        rows.append({
            "cluster": cluster_dir.name,
            "rho_overall": f"{sa['mean_spearman']:.4f}",
            "rho_nc": f"{sa['rho_non_mega']:.4f}",
        })
    if rows:
        rho_nc_vals = [float(r["rho_nc"]) for r in rows]
        rows.append({
            "cluster": "MEAN +/- STD",
            "rho_overall": "",
            "rho_nc": f"{np.mean(rho_nc_vals):.4f} +/- {np.std(rho_nc_vals, ddof=1):.4f}",
        })
    return pd.DataFrame(rows)


def table_tier_locks() -> pd.DataFrame:
    jf = RESULTS / "nbt_acceptance_push" / "t7_multiseed_tier_locks.json"
    if not jf.exists():
        return pd.DataFrame()
    with open(jf) as f:
        d = json.load(f)
    rows = []
    for tier_name, val in d.get("recipes", {}).items():
        for seed_key in ["seed_42", "seed_0"]:
            sv = val.get(seed_key, {})
            folds = sv.get("per_fold", [])
            if folds:
                rows.append({
                    "tier": tier_name,
                    "seed": seed_key,
                    "rho_nc_mean": f"{np.mean(folds):.4f}",
                    "rho_nc_std": f"{np.std(folds, ddof=1):.4f}",
                    "per_fold": ", ".join(f"{v:.4f}" for v in folds),
                })
    return pd.DataFrame(rows)


def table_ensemble_breakdown() -> pd.DataFrame:
    jf = RESULTS / "nbt_acceptance_push" / "t3e_ensemble_breakdown.json"
    if not jf.exists():
        return pd.DataFrame()
    with open(jf) as f:
        d = json.load(f)
    rows = []
    for group_name, gval in d.get("groups", {}).items():
        per_fold = gval.get("per_fold_rho_nc", [])
        rows.append({
            "group": group_name,
            "n_members": gval.get("n_members"),
            "members": ", ".join(gval.get("members", [])),
            "rho_nc_mean": f"{np.mean(per_fold):.4f}" if per_fold else "",
            "rho_nc_std": f"{np.std(per_fold, ddof=1):.4f}" if len(per_fold) > 1 else "",
            "delta_vs_baseline": gval.get("delta_vs_baseline"),
            "p_value": gval.get("p_value"),
            "per_fold": ", ".join(f"{v:.4f}" for v in per_fold) if per_fold else "",
        })
    return pd.DataFrame(rows)


def table_contamination() -> pd.DataFrame:
    csv_path = RESULTS / "landscape_2026" / "landscape_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def table_scaling() -> pd.DataFrame:
    jf = RESULTS / "nbt_acceptance_push" / "fig4_xp5_scaling.json"
    if not jf.exists():
        return pd.DataFrame()
    with open(jf) as f:
        d = json.load(f)
    rows = []
    for frac_key, val in d.get("per_fraction", {}).items():
        rows.append({
            "training_fraction_pct": val.get("fraction_pct", frac_key),
            "rho_nc_mean": f"{val['rho_nc_mean']:.4f}",
            "rho_nc_std": f"{val['rho_nc_std']:.4f}",
            "rho_overall_mean": f"{val['rho_overall_mean']:.4f}",
            "rho_overall_std": f"{val['rho_overall_std']:.4f}",
            "per_fold_rho_nc": ", ".join(f"{v:.4f}" for v in val.get("per_fold_rho_nc", [])),
        })
    return pd.DataFrame(rows)


def table_bootstrap_cis() -> pd.DataFrame:
    jf = RESULTS / "nbt_acceptance_push" / "t5_bootstrap_cis.json"
    if not jf.exists():
        return pd.DataFrame()
    with open(jf) as f:
        d = json.load(f)
    rows = []
    for recipe, val in d.get("recipes", {}).items():
        pooled = val.get("pooled", {})
        nm = pooled.get("non_mega", {})
        ov = pooled.get("overall", {})
        rows.append({
            "recipe": recipe,
            "rho_nc": f"{nm.get('rho', 0):.4f}",
            "rho_nc_ci_low": f"{nm.get('ci_low', 0):.4f}",
            "rho_nc_ci_high": f"{nm.get('ci_high', 0):.4f}",
            "rho_overall": f"{ov.get('rho', 0):.4f}",
            "rho_overall_ci_low": f"{ov.get('ci_low', 0):.4f}",
            "rho_overall_ci_high": f"{ov.get('ci_high', 0):.4f}",
            "n_test": pooled.get("n_test_total"),
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tables = [
        ("S4_recipe_comparison", table_recipe_comparison),
        ("ED_losco_holdouts", table_losco_holdouts),
        ("ED_tier_locks_multiseed", table_tier_locks),
        ("S_ensemble_breakdown", table_ensemble_breakdown),
        ("S_contamination", table_contamination),
        ("S_scaling", table_scaling),
        ("S_bootstrap_cis", table_bootstrap_cis),
    ]

    combined_path = OUTPUT / "aikixp_supplementary_tables.xlsx"
    with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
        for name, builder in tables:
            print(f"Building {name}...")
            df = builder()
            if len(df) == 0:
                print(f"  EMPTY — skipped")
                continue
            sheet = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet)
            print(f"  {len(df)} rows x {len(df.columns)} cols -> sheet '{sheet}'")

    print(f"\nWrote {combined_path} ({combined_path.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
