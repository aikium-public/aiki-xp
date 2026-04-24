#!/usr/bin/env python3
"""
External Validation of AIKI-XP (ProTEx) Expression Predictions
=============================================================
Correlates LOSO E. coli K12 predictions against three orthogonal datasets:
  1. Mori et al. 2021 — DIA/SWATH mass-spec, 60+ conditions, absolute mass fractions
  2. Taniguchi et al. 2010 — Single-cell YFP fluorescence, mean copies + noise (CV²)
  3. Li et al. 2014 — Ribosome profiling synthesis rates, 3 conditions

All validation uses LOSO predictions (E. coli held out entirely during training).
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "results/protex_qc/external_validation/raw_downloads"
OUT = BASE / "results/protex_qc/external_validation"
PRED_PATH = BASE / "results/protex_qc/final_sprint"


def load_gene_mapping():
    mapping = pd.read_csv(OUT / "ecoli_k12_gene_mapping.csv")
    mapping["np_base"] = mapping["np_accession"].str.replace(r"\.\d+$", "", regex=True)
    return mapping


def load_loso_predictions(seed=42):
    path = PRED_PATH / f"loso_Escherichia_coli_K12_f10_seed{seed}_predictions.parquet"
    preds = pd.read_parquet(path)
    ecoli = preds[preds["species"] == "Escherichia_coli_K12"].copy()
    ecoli["np_accession"] = ecoli["gene_id"].str.split("|").str[1]
    ecoli["np_base"] = ecoli["np_accession"].str.replace(r"\.\d+$", "", regex=True)
    return ecoli


def merge_via_gene_name(ecoli, external, mapping, ext_gene_col="gene_name", ext_value_col="value"):
    ecoli_mapped = ecoli.merge(
        mapping[["np_base", "gene_name"]].drop_duplicates(subset="gene_name"),
        on="np_base", how="inner",
    )
    merged = ecoli_mapped.merge(
        external[[ext_gene_col, ext_value_col]].rename(columns={ext_gene_col: "gene_name"}),
        on="gene_name", how="inner",
    )
    return merged


def merge_via_b_number(ecoli, external, mapping, ext_b_col="b_number", ext_value_col="value"):
    ecoli_mapped = ecoli.merge(
        mapping[["np_base", "b_number"]].drop_duplicates(subset="b_number"),
        on="np_base", how="inner",
    )
    merged = ecoli_mapped.merge(
        external[[ext_b_col, ext_value_col]].rename(columns={ext_b_col: "b_number"}),
        on="b_number", how="inner",
    )
    return merged


def validate_mori_2021(ecoli, mapping):
    """Mori et al. 2021 — Condition-resolved absolute mass fractions."""
    print("\n" + "=" * 70)
    print("DATASET 1: Mori et al. 2021 — Condition-resolved E. coli proteomics")
    print("=" * 70)

    ev9 = pd.read_excel(
        RAW / "PMC8144880/MSB-17-e9536-s007.xlsx",
        sheet_name="EV9-AbsoluteMassFractions-2",
    )
    samples = pd.read_excel(
        RAW / "PMC8144880/MSB-17-e9536-s011.xlsx",
        sheet_name="EV3-Samples-2",
    )

    id_cols = ["Gene name", "Gene locus", "Protein ID"]
    condition_cols = [c for c in ev9.columns if c not in id_cols]

    ecoli_mapped = ecoli.merge(
        mapping[["np_base", "gene_name"]].drop_duplicates(subset="gene_name"),
        on="np_base", how="inner",
    )

    ev9_renamed = ev9.rename(columns={"Gene name": "gene_name"})
    merged_base = ecoli_mapped.merge(
        ev9_renamed[["gene_name"] + condition_cols], on="gene_name", how="inner"
    )

    print(f"  Genes matched: {len(merged_base)} / {len(ecoli)} predictions")
    print(f"  Conditions: {len(condition_cols)}")

    # Condition-averaged correlation
    merged_base["mori_avg"] = merged_base[condition_cols].mean(axis=1)
    nonzero = merged_base[merged_base["mori_avg"] > 0].copy()
    nonzero["mori_log_avg"] = np.log10(nonzero["mori_avg"])

    rho_avg, p_avg = spearmanr(nonzero["y_pred"], nonzero["mori_log_avg"])
    r_avg, _ = pearsonr(nonzero["y_pred"], nonzero["mori_log_avg"])
    print(f"\n  Condition-averaged (log10 mass fraction):")
    print(f"    N={len(nonzero)}, Spearman ρ={rho_avg:.4f}, Pearson r={r_avg:.4f}, p={p_avg:.2e}")

    # Per-condition correlations
    per_cond = []
    for cond in condition_cols:
        subset = merged_base[merged_base[cond] > 0].copy()
        if len(subset) < 30:
            continue
        subset["ext_log"] = np.log10(subset[cond])
        rho, pval = spearmanr(subset["y_pred"], subset["ext_log"])
        per_cond.append({
            "condition": cond,
            "n_genes": len(subset),
            "spearman_rho": rho,
            "p_value": pval,
        })

    per_cond_df = pd.DataFrame(per_cond).sort_values("spearman_rho", ascending=False)
    print(f"\n  Per-condition Spearman ρ distribution (N={len(per_cond_df)} conditions):")
    print(f"    Mean ρ = {per_cond_df['spearman_rho'].mean():.4f}")
    print(f"    Median ρ = {per_cond_df['spearman_rho'].median():.4f}")
    print(f"    Min ρ = {per_cond_df['spearman_rho'].min():.4f} ({per_cond_df.iloc[-1]['condition']})")
    print(f"    Max ρ = {per_cond_df['spearman_rho'].max():.4f} ({per_cond_df.iloc[0]['condition']})")

    per_cond_df.to_csv(OUT / "mori_2021_per_condition_rho.csv", index=False)

    sample_meta = samples[["Sample ID", "Description"]].dropna(subset=["Sample ID"])
    per_cond_annotated = per_cond_df.merge(
        sample_meta.rename(columns={"Sample ID": "condition"}), on="condition", how="left"
    )
    per_cond_annotated.to_csv(OUT / "mori_2021_per_condition_rho_annotated.csv", index=False)

    return {
        "dataset": "Mori et al. 2021",
        "measurement": "DIA/SWATH mass-spec absolute mass fractions",
        "n_matched": int(len(merged_base)),
        "n_nonzero_avg": int(len(nonzero)),
        "condition_averaged": {
            "spearman_rho": round(rho_avg, 4),
            "pearson_r": round(r_avg, 4),
            "p_value": float(f"{p_avg:.2e}"),
            "n": int(len(nonzero)),
        },
        "per_condition_summary": {
            "n_conditions": int(len(per_cond_df)),
            "mean_rho": round(per_cond_df["spearman_rho"].mean(), 4),
            "median_rho": round(per_cond_df["spearman_rho"].median(), 4),
            "min_rho": round(per_cond_df["spearman_rho"].min(), 4),
            "max_rho": round(per_cond_df["spearman_rho"].max(), 4),
        },
    }


def validate_taniguchi_2010(ecoli, mapping):
    """Taniguchi et al. 2010 — Single-cell YFP fluorescence."""
    print("\n" + "=" * 70)
    print("DATASET 2: Taniguchi et al. 2010 — Single-cell fluorescence")
    print("=" * 70)

    tang = pd.read_excel(
        RAW / "taniguchi_2010_bionumbers.xlsx",
        sheet_name="Table S6",
    )
    print(f"  Taniguchi genes: {len(tang)}")
    print(f"  Columns: {tang.columns.tolist()[:10]}...")

    tang_clean = tang[["Gene Name", "B Number", "Mean_Protein", "Noise_Protein"]].copy()
    tang_clean = tang_clean.rename(columns={
        "Gene Name": "gene_name",
        "B Number": "b_number",
        "Mean_Protein": "mean_protein",
        "Noise_Protein": "noise_protein",
    })
    tang_clean = tang_clean.dropna(subset=["mean_protein"])
    tang_clean["mean_protein"] = pd.to_numeric(tang_clean["mean_protein"], errors="coerce")
    tang_clean = tang_clean.dropna(subset=["mean_protein"])

    # Merge via gene name
    merged = merge_via_gene_name(
        ecoli, tang_clean, mapping,
        ext_gene_col="gene_name", ext_value_col="mean_protein"
    )
    merged = merged[merged["mean_protein"] > 0].copy()
    merged["log_mean_protein"] = np.log10(merged["mean_protein"])

    print(f"  Genes matched (mean > 0): {len(merged)}")

    rho, pval = spearmanr(merged["y_pred"], merged["log_mean_protein"])
    r, _ = pearsonr(merged["y_pred"], merged["log_mean_protein"])
    print(f"\n  Mean protein expression (log10 copies/cell):")
    print(f"    N={len(merged)}, Spearman ρ={rho:.4f}, Pearson r={r:.4f}, p={pval:.2e}")

    # Noise correlation
    merged_noise = merge_via_gene_name(
        ecoli, tang_clean[tang_clean["noise_protein"].notna()], mapping,
        ext_gene_col="gene_name", ext_value_col="noise_protein"
    )
    merged_noise["noise_protein"] = pd.to_numeric(merged_noise["noise_protein"], errors="coerce")
    merged_noise = merged_noise.dropna(subset=["noise_protein"])
    merged_noise = merged_noise[merged_noise["noise_protein"] > 0].copy()
    merged_noise["log_noise"] = np.log10(merged_noise["noise_protein"])

    rho_noise, pval_noise = spearmanr(merged_noise["y_pred"], merged_noise["log_noise"])
    print(f"\n  Protein noise (log10 CV²):")
    print(f"    N={len(merged_noise)}, Spearman ρ={rho_noise:.4f}, p={pval_noise:.2e}")

    return {
        "dataset": "Taniguchi et al. 2010",
        "measurement": "Single-cell YFP fluorescence (copies/cell)",
        "n_matched_mean": int(len(merged)),
        "mean_expression": {
            "spearman_rho": round(rho, 4),
            "pearson_r": round(r, 4),
            "p_value": float(f"{pval:.2e}"),
            "n": int(len(merged)),
        },
        "noise_cv2": {
            "spearman_rho": round(rho_noise, 4),
            "p_value": float(f"{pval_noise:.2e}"),
            "n": int(len(merged_noise)),
        },
    }


def validate_li_2014(ecoli, mapping):
    """Li et al. 2014 — Ribosome profiling synthesis rates."""
    print("\n" + "=" * 70)
    print("DATASET 3: Li et al. 2014 — Ribosome profiling synthesis rates")
    print("=" * 70)

    li = pd.read_excel(
        RAW / "li_2014_mmc1.xlsx",
        sheet_name="TableS1",
    )
    print(f"  Li genes: {len(li)}")
    print(f"  Conditions: {li.columns.tolist()[1:]}")

    conditions = ["MOPS complete", "MOPS minimal", "MOPS complete without methionine"]
    results_per_cond = {}

    for cond in conditions:
        li_cond = li[["Gene", cond]].copy()
        li_cond[cond] = li_cond[cond].astype(str).str.strip("[]")
        li_cond[cond] = pd.to_numeric(li_cond[cond], errors="coerce")
        li_cond = li_cond.dropna(subset=[cond])
        li_cond = li_cond[li_cond[cond] > 0].copy()
        li_cond["log_rate"] = np.log10(li_cond[cond])

        merged = merge_via_gene_name(
            ecoli, li_cond.rename(columns={"Gene": "gene_name", "log_rate": "value"}),
            mapping, ext_gene_col="gene_name", ext_value_col="value"
        )

        rho, pval = spearmanr(merged["y_pred"], merged["value"])
        r, _ = pearsonr(merged["y_pred"], merged["value"])
        print(f"\n  {cond} (log10 proteins/cell/generation):")
        print(f"    N={len(merged)}, Spearman ρ={rho:.4f}, Pearson r={r:.4f}, p={pval:.2e}")

        results_per_cond[cond] = {
            "spearman_rho": round(rho, 4),
            "pearson_r": round(r, 4),
            "p_value": float(f"{pval:.2e}"),
            "n": int(len(merged)),
        }

    # Condition-averaged
    li_avg = li.copy()
    for cond in conditions:
        li_avg[cond] = li_avg[cond].astype(str).str.strip("[]")
        li_avg[cond] = pd.to_numeric(li_avg[cond], errors="coerce")
    li_avg["avg_rate"] = li_avg[conditions].mean(axis=1)
    li_avg = li_avg[li_avg["avg_rate"] > 0].copy()
    li_avg["log_avg_rate"] = np.log10(li_avg["avg_rate"])

    merged_avg = merge_via_gene_name(
        ecoli,
        li_avg[["Gene", "log_avg_rate"]].rename(columns={"Gene": "gene_name", "log_avg_rate": "value"}),
        mapping, ext_gene_col="gene_name", ext_value_col="value"
    )
    rho_avg, pval_avg = spearmanr(merged_avg["y_pred"], merged_avg["value"])
    r_avg, _ = pearsonr(merged_avg["y_pred"], merged_avg["value"])
    print(f"\n  Condition-averaged synthesis rate:")
    print(f"    N={len(merged_avg)}, Spearman ρ={rho_avg:.4f}, Pearson r={r_avg:.4f}, p={pval_avg:.2e}")

    return {
        "dataset": "Li et al. 2014",
        "measurement": "Ribosome profiling synthesis rates (proteins/cell/generation)",
        "per_condition": results_per_cond,
        "condition_averaged": {
            "spearman_rho": round(rho_avg, 4),
            "pearson_r": round(r_avg, 4),
            "p_value": float(f"{pval_avg:.2e}"),
            "n": int(len(merged_avg)),
        },
    }


def validate_multi_seed(mapping):
    """Run validation across both seeds for robustness."""
    all_results = {}
    for seed in [42, 123]:
        print(f"\n{'#' * 70}")
        print(f"# SEED {seed}")
        print(f"{'#' * 70}")
        ecoli = load_loso_predictions(seed=seed)
        print(f"Loaded {len(ecoli)} E. coli K12 LOSO predictions (seed={seed})")

        results = {
            "seed": seed,
            "n_predictions": int(len(ecoli)),
        }
        results["mori_2021"] = validate_mori_2021(ecoli, mapping)
        results["taniguchi_2010"] = validate_taniguchi_2010(ecoli, mapping)
        results["li_2014"] = validate_li_2014(ecoli, mapping)
        all_results[f"seed_{seed}"] = results

    return all_results


def main():
    print("=" * 70)
    print("AIKI-XP External Validation — E. coli K12 LOSO Predictions")
    print("=" * 70)

    mapping = load_gene_mapping()
    print(f"Gene mapping loaded: {len(mapping)} entries")

    results = validate_multi_seed(mapping)

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Dataset':<35} {'Metric':<30} {'Seed 42':>10} {'Seed 123':>10}"
    print(header)
    print("-" * len(header))

    def get_rho(res, path):
        obj = res
        for key in path:
            obj = obj[key]
        return obj

    rows = [
        ("Mori 2021 (avg)", "condition_averaged.spearman_rho",
         ["mori_2021", "condition_averaged", "spearman_rho"]),
        ("Mori 2021 (per-cond mean)", "per_condition_summary.mean_rho",
         ["mori_2021", "per_condition_summary", "mean_rho"]),
        ("Taniguchi 2010 (mean expr)", "mean_expression.spearman_rho",
         ["taniguchi_2010", "mean_expression", "spearman_rho"]),
        ("Taniguchi 2010 (noise CV²)", "noise_cv2.spearman_rho",
         ["taniguchi_2010", "noise_cv2", "spearman_rho"]),
        ("Li 2014 (avg rate)", "condition_averaged.spearman_rho",
         ["li_2014", "condition_averaged", "spearman_rho"]),
    ]

    for label, _, path in rows:
        v42 = get_rho(results["seed_42"], path)
        v123 = get_rho(results["seed_123"], path)
        print(f"  {label:<33} {'Spearman ρ':<28} {v42:>10.4f} {v123:>10.4f}")

    # Save results
    out_path = OUT / "external_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Blocker report
    print("\n\n" + "=" * 70)
    print("BLOCKER REPORT")
    print("=" * 70)
    print(f"{'Check':<40} {'Mori 2021':>12} {'Taniguchi 2010':>15} {'Li 2014':>12}")
    print("-" * 80)
    print(f"{'Data downloadable?':<40} {'YES (PMC OA)':>12} {'YES (BioNum)':>15} {'YES (Cell)':>12}")
    print(f"{'Gene ID type':<40} {'gene+b-num':>12} {'gene+b-num':>15} {'gene name':>12}")
    print(f"{'PaXDb overlap?':<40} {'Partial':>12} {'Low':>15} {'Low':>12}")
    print(f"{'Log-scale needed?':<40} {'YES':>12} {'YES':>15} {'YES':>12}")


if __name__ == "__main__":
    main()
