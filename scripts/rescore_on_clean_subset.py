#!/usr/bin/env python3
"""Phase 2 — re-score a tool's per-gene predictions on the contamination-cleaned subset.

Given a per-gene predictions parquet (with `gene_id`, `<label_col>`, `y_pred` at minimum)
and a clean_gene_ids.txt file (one gene_id per line, produced by
`compute_external_contamination.py`), compute:
    - ρ_full      = Spearman ρ on the full benchmark
    - ρ_clean     = Spearman ρ on the clean subset only
    - Δρ          = ρ_clean − ρ_full
    - bootstrap CIs (95%) on each via per-gene resampling (n=1000)
    - AUROC_full / AUROC_clean if a binary label can be derived from the label column
      via the threshold convention used by NetSolP/MPEPE (label ≥ threshold → positive)

The Spearman ρ on the clean subset is the honest evaluation surface. Δρ is the
"how much published number was inflated by training-set memorization" gap.

Output: a single JSON written to results/landscape_2026/<tool>/<benchmark>__rescore.json
plus a one-row append to results/landscape_2026/landscape_summary.csv.

Usage:
    python scripts/protex/rescore_on_clean_subset.py \\
        --tool netsolp_usability \\
        --benchmark boel_2016 \\
        --predictions results/external_benchmark/boel/netsolp_usability_predictions.parquet \\
        --label-col expression_score \\
        --clean-ids results/landscape_2026/contamination/boel_2016__clean_gene_ids.txt \\
        --binary-threshold 3 \\
        --citation "Hon et al. 2022 Bioinformatics 38:941"

The tool name and benchmark name end up in the JSON + summary CSV. Use them as identifiers.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rescore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANDSCAPE_ROOT = PROJECT_ROOT / "results" / "landscape_2026"
SUMMARY_CSV = LANDSCAPE_ROOT / "landscape_summary.csv"

SUMMARY_COLUMNS = [
    "tool",
    "benchmark",
    "label_col",
    "n_full",
    "n_clean",
    "n_contaminated",
    "pct_contaminated",
    "rho_full",
    "rho_full_ci_low",
    "rho_full_ci_high",
    "rho_clean",
    "rho_clean_ci_low",
    "rho_clean_ci_high",
    "delta_rho",
    "delta_rho_ci_low",
    "delta_rho_ci_high",
    "auroc_full",
    "auroc_clean",
    "binary_threshold",
    "citation",
    "predictions_path",
    "clean_ids_path",
    "scored_at",
]


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────


def spearman_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Spearman ρ + percentile bootstrap CI on per-gene resamples."""
    if len(y_true) < 3:
        return float("nan"), float("nan"), float("nan")
    rho, _ = stats.spearmanr(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(y_true[idx], y_pred[idx])
        boot[i] = r
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return float(rho), ci_low, ci_high


def delta_rho_with_ci(
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
    clean_mask: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Δρ = ρ_clean − ρ_full, with CI from joint bootstrap (resample full, recompute both)."""
    if clean_mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    rho_full, _ = stats.spearmanr(y_true_full, y_pred_full)
    rho_clean, _ = stats.spearmanr(y_true_full[clean_mask], y_pred_full[clean_mask])
    delta = rho_clean - rho_full

    rng = np.random.default_rng(seed)
    n = len(y_true_full)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true_full[idx]
        yp = y_pred_full[idx]
        cm = clean_mask[idx]
        rf, _ = stats.spearmanr(yt, yp)
        if cm.sum() < 3:
            boot[i] = np.nan
            continue
        rc, _ = stats.spearmanr(yt[cm], yp[cm])
        boot[i] = rc - rf
    boot = boot[~np.isnan(boot)]
    ci_low = float(np.percentile(boot, 2.5)) if len(boot) else float("nan")
    ci_high = float(np.percentile(boot, 97.5)) if len(boot) else float("nan")
    return float(delta), ci_low, ci_high


def auroc_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """AUROC for the binary task `y_true >= threshold` vs `y_pred` continuous."""
    try:
        from sklearn.metrics import roc_auc_score

        y_bin = (y_true >= threshold).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return float("nan")
        return float(roc_auc_score(y_bin, y_pred))
    except Exception as e:
        log.warning(f"AUROC failed: {e}")
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────


def load_clean_ids(path: Path) -> set[str]:
    with open(path) as f:
        ids = {line.strip() for line in f if line.strip()}
    log.info(f"loaded {len(ids):,} clean gene_ids from {path}")
    return ids


def load_predictions(
    path: Path, label_col: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (gene_ids, y_true, y_pred)."""
    df = pd.read_parquet(path)
    required = {"gene_id", "y_pred", label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"predictions parquet missing required columns {missing}; got {df.columns.tolist()}"
        )

    # Rule 1: explicit NaN check
    n_nan_pred = df["y_pred"].isna().sum()
    n_nan_true = df[label_col].isna().sum()
    if n_nan_pred > 0:
        log.warning(f"{n_nan_pred} NaN in y_pred — dropping (transparent, not silent)")
    if n_nan_true > 0:
        log.warning(f"{n_nan_true} NaN in {label_col} — dropping")
    df = df.dropna(subset=["y_pred", label_col])

    log.info(f"loaded {len(df):,} predictions from {path}")
    return (
        df["gene_id"].astype(str).reset_index(drop=True),
        df[label_col].astype(float).reset_index(drop=True),
        df["y_pred"].astype(float).reset_index(drop=True),
    )


def append_summary_row(row: dict) -> None:
    LANDSCAPE_ROOT.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV)
        # Drop any prior row with the same (tool, benchmark) before appending
        existing = existing[
            ~((existing["tool"] == row["tool"]) & (existing["benchmark"] == row["benchmark"]))
        ]
    else:
        existing = pd.DataFrame(columns=SUMMARY_COLUMNS)
    new_df = pd.DataFrame([{c: row.get(c) for c in SUMMARY_COLUMNS}])
    out = pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(SUMMARY_CSV, index=False)
    log.info(f"summary row appended: {SUMMARY_CSV} ({len(out)} rows total)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, help="Tool identifier (e.g. netsolp_usability)")
    parser.add_argument("--benchmark", required=True, help="Benchmark identifier (e.g. boel_2016)")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label-col", default="expression_score")
    parser.add_argument("--clean-ids", type=Path, required=True)
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=None,
        help="If provided, AUROC is computed for `label >= threshold` as binary positive",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--citation", default="", help="Free-text citation for the SI table"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: results/landscape_2026/<tool>/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (LANDSCAPE_ROOT / args.tool)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.benchmark}__rescore.json"

    t0 = time.time()
    gene_ids, y_true, y_pred = load_predictions(args.predictions, args.label_col)
    clean_set = load_clean_ids(args.clean_ids)

    clean_mask = gene_ids.isin(clean_set).to_numpy()
    n_full = len(y_true)
    n_clean = int(clean_mask.sum())
    n_contam = n_full - n_clean
    pct_contam = 100.0 * n_contam / max(n_full, 1)

    y_true_arr = y_true.to_numpy()
    y_pred_arr = y_pred.to_numpy()

    rho_full, rho_full_lo, rho_full_hi = spearman_with_ci(
        y_true_arr, y_pred_arr, args.n_bootstrap, args.seed
    )
    rho_clean, rho_clean_lo, rho_clean_hi = spearman_with_ci(
        y_true_arr[clean_mask], y_pred_arr[clean_mask], args.n_bootstrap, args.seed
    )
    delta, delta_lo, delta_hi = delta_rho_with_ci(
        y_true_arr, y_pred_arr, clean_mask, args.n_bootstrap, args.seed
    )

    auroc_full = float("nan")
    auroc_clean = float("nan")
    if args.binary_threshold is not None:
        auroc_full = auroc_at_threshold(y_true_arr, y_pred_arr, args.binary_threshold)
        if n_clean >= 3:
            auroc_clean = auroc_at_threshold(
                y_true_arr[clean_mask], y_pred_arr[clean_mask], args.binary_threshold
            )

    result = {
        "tool": args.tool,
        "benchmark": args.benchmark,
        "label_col": args.label_col,
        "n_full": n_full,
        "n_clean": n_clean,
        "n_contaminated": n_contam,
        "pct_contaminated": round(pct_contam, 4),
        "rho_full": round(rho_full, 6),
        "rho_full_ci_low": round(rho_full_lo, 6),
        "rho_full_ci_high": round(rho_full_hi, 6),
        "rho_clean": round(rho_clean, 6),
        "rho_clean_ci_low": round(rho_clean_lo, 6),
        "rho_clean_ci_high": round(rho_clean_hi, 6),
        "delta_rho": round(delta, 6),
        "delta_rho_ci_low": round(delta_lo, 6),
        "delta_rho_ci_high": round(delta_hi, 6),
        "auroc_full": round(auroc_full, 6) if not np.isnan(auroc_full) else None,
        "auroc_clean": round(auroc_clean, 6) if not np.isnan(auroc_clean) else None,
        "binary_threshold": args.binary_threshold,
        "citation": args.citation,
        "predictions_path": str(args.predictions),
        "clean_ids_path": str(args.clean_ids),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "wall_seconds": round(time.time() - t0, 2),
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    append_summary_row(result)

    log.info("─" * 60)
    log.info(f"tool        : {args.tool}")
    log.info(f"benchmark   : {args.benchmark}")
    log.info(f"N full      : {n_full:,}")
    log.info(f"N clean     : {n_clean:,}  ({100 - pct_contam:.2f}%)")
    log.info(f"ρ full      : {rho_full:+.4f}  [{rho_full_lo:+.4f}, {rho_full_hi:+.4f}]")
    log.info(f"ρ clean     : {rho_clean:+.4f}  [{rho_clean_lo:+.4f}, {rho_clean_hi:+.4f}]")
    log.info(f"Δρ          : {delta:+.4f}  [{delta_lo:+.4f}, {delta_hi:+.4f}]")
    if not np.isnan(auroc_full):
        log.info(f"AUROC full  : {auroc_full:.4f}")
        log.info(f"AUROC clean : {auroc_clean:.4f}")
    log.info(f"output json : {out_json}")
    log.info("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
