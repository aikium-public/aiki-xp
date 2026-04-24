#!/usr/bin/env python3
"""
Reproduce External Showcase Validation Results — Synechococcus T4 Novel-Phylum
===============================================================================

Self-contained script that reads the showcase artifacts and computes every number
cited in the manuscript's external-validation section.

Motivation
----------
The Russo et al. 2025 DIA-MS dataset (PRIDE PXD062851) provides quantitative
proteomics for 2,307 proteins from Synechococcus elongatus PCC 7942 (taxid 1140),
a cyanobacterium with ZERO representation in the 385-species training set at any
taxonomic rank (species through phylum). Of these, 2,018 were resolved to RefSeq
CDS entries via UniProt-to-WP_ accession mapping (289 unmappable dropped).

Pipeline that produced these artifacts (on GCP A100 40GB):
  1. scripts/protex/build_showcase_gene_table.py — compiled the gene catalog from
     Russo 2025 PRIDE data + NCBI reference genome GCF_000012525.1
  2. scripts/protex/featurize_showcase.py — computed all 10 F10 modalities fresh
     (ESM-C, DNABERT-2, HyenaDNA, Bacformer, Evo-2, 5 classical) on GCP A100
  3. scripts/protex/audit_showcase_embeddings.py — verified 2018/2018 clean for
     all 10 modalities (no duplicates, no NaN, no all-zero rows)
  4. scripts/protex/predict_showcase.py — ran frozen champion checkpoint inference
  5. scripts/protex/annotate_showcase_familiarity.py — annotated each gene against
     the exact training set (390,640 train genes from hard_hybrid_production_split_v2.tsv)

Required input files (all in results/protex_qc/external_validation/):
  - showcase_predictions.parquet (or showcase_recheck/showcase_predictions.parquet)
  - showcase_familiarity/showcase_familiarity_annotated.parquet

To download from GCS if not present locally:
  bash scripts/protex/sync_showcase_artifacts_gcs.sh download \\
    [internal storage]

Usage:
  python3 scripts/protex/reproduce_showcase_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "results" / "protex_qc" / "external_validation"

FAMILIARITY_PATH = OUT_ROOT / "showcase_familiarity" / "showcase_familiarity_annotated.parquet"
PREDICTIONS_PATH = OUT_ROOT / "showcase_recheck" / "showcase_predictions.parquet"
PREDICTIONS_FALLBACK = OUT_ROOT / "showcase_predictions.parquet"


def load_data() -> pd.DataFrame:
    pred_path = PREDICTIONS_PATH if PREDICTIONS_PATH.exists() else PREDICTIONS_FALLBACK
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions parquet found at {PREDICTIONS_PATH} or {PREDICTIONS_FALLBACK}")

    if FAMILIARITY_PATH.exists():
        df = pd.read_parquet(FAMILIARITY_PATH)
        if "y_pred" not in df.columns:
            pred = pd.read_parquet(pred_path, columns=["gene_id", "y_pred"])
            df = df.merge(pred, on="gene_id", how="left")
    else:
        df = pd.read_parquet(pred_path)

    df["gene_id"] = df["gene_id"].astype(str)
    return df


def compute_rho(y_pred: np.ndarray, y_ext: np.ndarray) -> tuple[float, float, float]:
    log_ext = np.log10(y_ext.astype(float) + 1e-10)
    rho, p_rho = spearmanr(y_pred, log_ext)
    r, _ = pearsonr(y_pred, log_ext)
    return float(rho), float(r), float(p_rho)


def main() -> int:
    df = load_data()
    work = df.dropna(subset=["y_pred", "external_measurement"]).copy()
    work = work[work["external_measurement"] > 0].copy()

    n_total = len(df)
    n_valid = len(work)

    print("=" * 80)
    print("EXTERNAL SHOWCASE VALIDATION — REPRODUCIBLE RESULTS")
    print("=" * 80)
    print(f"Dataset: Russo et al. 2025 (PRIDE PXD062851)")
    print(f"Organism: Synechococcus elongatus PCC 7942 (taxid 1140)")
    print(f"Total showcase genes: {n_total}")
    print(f"Valid for correlation (y_pred present, external_measurement > 0): {n_valid}")

    # ── Headline correlation ──
    rho, r, p = compute_rho(work["y_pred"].values, work["external_measurement"].values)
    print(f"\n── Headline Result ──")
    print(f"Spearman ρ = {rho:.4f}  (p = {p:.2e})")
    print(f"Pearson r  = {r:.4f}")
    print(f"N = {n_valid}")

    results = {
        "headline": {
            "spearman_rho": round(rho, 4),
            "pearson_r": round(r, 4),
            "p_value": float(f"{p:.2e}"),
            "n_genes": n_valid,
        },
    }

    # ── Categorical novelty flags ──
    cat_cols = [
        "taxid_seen_in_training",
        "species_seen_in_training",
        "genus_seen_in_training",
        "family_seen_in_training",
        "phylum_seen_in_training",
        "exact_protein_seen_in_training",
        "exact_operon_dna_seen_in_training",
    ]
    print(f"\n── Categorical Novelty (all should be 0/N for truly novel) ──")
    novelty = {}
    for col in cat_cols:
        if col in work.columns:
            n_seen = int(work[col].sum())
            novelty[col] = n_seen
            print(f"  {col}: {n_seen}/{n_valid} seen")
    results["novelty_flags"] = novelty

    # ── Continuous familiarity metrics ──
    cont_cols = {
        "gene_nearest_train_cosine": "Nearest gene cosine (ESM-C)",
        "gene_nearest_train_global_identity": "Nearest gene protein identity",
        "operon_nearest_train_cosine": "Nearest operon cosine (DNABERT-2)",
        "operon_nearest_train_kmer_jaccard": "Nearest operon 8-mer Jaccard",
        "genome_nearest_train_cosine": "Nearest genome cosine (Bacformer)",
    }
    print(f"\n── Continuous Familiarity Metrics (mean across {n_valid} genes) ──")
    familiarity_means = {}
    for col, label in cont_cols.items():
        if col in work.columns:
            mean_val = float(np.nanmean(work[col].astype(float)))
            familiarity_means[col] = round(mean_val, 4)
            print(f"  {label}: {mean_val:.4f}")
    results["familiarity_means"] = familiarity_means

    # ── Binned correlations ──
    bin_metrics = [
        ("gene_nearest_train_cosine", "Gene cosine (ESM-C)"),
        ("gene_nearest_train_global_identity", "Gene protein identity"),
        ("operon_nearest_train_cosine", "Operon cosine (DNABERT-2)"),
        ("genome_nearest_train_cosine", "Genome cosine (Bacformer)"),
    ]
    binned_results = {}
    for col, label in bin_metrics:
        if col not in work.columns:
            continue
        series = work[col].astype(float)
        if series.nunique(dropna=True) < 4:
            continue
        try:
            bins = pd.qcut(series, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        except ValueError:
            continue

        print(f"\n── Binned ρ by {label} ──")
        print(f"  {'Quartile':<10} {'N':>6} {'Range':>24} {'ρ':>8} {'p':>12}")
        tmp = work.assign(_bin=bins)
        bin_rows = []
        for bin_name, grp in tmp.groupby("_bin", observed=True):
            if len(grp) < 10:
                continue
            bin_rho, _, bin_p = compute_rho(grp["y_pred"].values, grp["external_measurement"].values)
            lo = float(grp[col].min())
            hi = float(grp[col].max())
            print(f"  {str(bin_name):<10} {len(grp):>6} {lo:.4f}–{hi:.4f}  {bin_rho:>8.4f} {bin_p:>12.2e}")
            bin_rows.append({
                "bin": str(bin_name),
                "n_genes": int(len(grp)),
                "metric_min": round(lo, 4),
                "metric_max": round(hi, 4),
                "spearman_rho": round(bin_rho, 4),
                "p_value": float(f"{bin_p:.2e}"),
            })
        binned_results[col] = bin_rows
    results["binned_correlations"] = binned_results

    # ── Save reproducible results JSON ──
    out_path = OUT_ROOT / "showcase_familiarity" / "showcase_reproducible_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n── Saved: {out_path} ──")

    print(f"\n{'=' * 80}")
    print("All numbers above are reproducible from the saved artifacts.")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
