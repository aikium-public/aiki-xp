#!/usr/bin/env python3
"""
Generate two new SI figures:
  1. Synechococcus stratified analysis (universal vs phylum-specific genes)
  2. Confound analysis summary (Simpson's paradox test, A8 native vs z-scored)

Usage:
    python scripts/protex/generate_si_synechococcus_and_confounds.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "protex" / "latex" / "figures" / "nbt"

# Colors
CORRECT_COL = "#2166AC"
WRONG_COL = "#D6604D"
NEUTRAL_COL = "#878787"
HIGHLIGHT_COL = "#E08214"
BG = "#FFFFFF"
INK = "#1A1A2E"


def generate_si_synechococcus():
    """SI figure: Synechococcus cross-phylum stratified analysis."""
    log.info("Generating SI: Synechococcus stratified analysis...")

    # Load predictions
    pred_path = PROJECT_ROOT / "results" / "protex_qc" / "external_validation" / "showcase_predictions.parquet"
    if not pred_path.exists():
        log.warning(f"  Predictions not found: {pred_path}")
        return

    cat = pd.read_parquet(str(pred_path))
    cat["external_measurement"] = pd.to_numeric(cat["external_measurement"], errors="coerce")
    cat = cat[cat["external_measurement"] > 0].copy()
    cat["log_expr"] = np.log10(cat["external_measurement"])
    cat["prot_len"] = cat["protein_sequence"].str.len()

    # Load genome annotations: GFF has new locus tags, predictions have old ones.
    # Bridge via protein_id (wp_accession).
    gff_path = PROJECT_ROOT / "results" / "protex_qc" / "external_validation" / "raw_downloads" / "genomes" / "syn7942_genomic.gff"
    combined_path = PROJECT_ROOT / "results" / "protex_qc" / "external_validation" / "syn7942_combined_mapping.csv"
    gene_map_path = PROJECT_ROOT / "results" / "protex_qc" / "external_validation" / "syn7942_gene_mapping.csv"
    tag_to_product = {}
    if gff_path.exists() and combined_path.exists() and gene_map_path.exists():
        # Parse GFF: new_locus_tag -> product
        new_tag_to_product = {}
        with open(gff_path) as gf:
            for line in gf:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[2] != "CDS":
                    continue
                attrs = dict(x.split("=", 1) for x in parts[8].split(";") if "=" in x)
                lt = attrs.get("locus_tag", "")
                prod = attrs.get("product", "hypothetical protein")
                if lt:
                    new_tag_to_product[lt] = prod
        # Bridge: old_locus_tag -> wp_accession -> new_locus_tag -> product
        combined = pd.read_csv(combined_path)
        gene_map = pd.read_csv(gene_map_path)
        bridge = combined.merge(gene_map[["protein_id", "locus_tag"]],
                                left_on="wp_accession", right_on="protein_id", how="inner",
                                suffixes=("_old", "_new"))
        for _, row in bridge.iterrows():
            old_tag = row["locus_tag_old"]
            new_tag = row["locus_tag_new"]
            if new_tag in new_tag_to_product:
                tag_to_product[old_tag] = new_tag_to_product[new_tag]
        log.info(f"  Mapped {len(tag_to_product)} old locus tags to products via GFF bridge")
    else:
        log.warning("  GFF/mapping files not available, using labels without gene names")

    # Classify genes
    cat["product"] = cat["locus_tag"].map(tag_to_product).fillna("unknown")
    cat["is_ribosomal"] = cat["product"].str.contains("ribosom|elongation factor|chaperone|polymerase|tRNA synthetase", case=False, na=False)
    cat["is_photosynthetic"] = cat["product"].str.contains("phycobili|photosystem|allophycocyanin|ribulose|carboxyl|phycocyanin|thylakoid", case=False, na=False)
    cat["is_operon"] = cat["num_genes_in_operon"].fillna(1) > 1 if "num_genes_in_operon" in cat.columns else False

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
    })

    fig = plt.figure(figsize=(7.2, 3.2), facecolor=BG)
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    # ── Panel a: Predicted vs actual, colored by gene type ────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(-0.15, 1.08, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="top")

    # Background points
    other = cat[~cat["is_ribosomal"] & ~cat["is_photosynthetic"]]
    ax_a.scatter(other["y_pred"], other["log_expr"], s=3, alpha=0.15, color=NEUTRAL_COL, rasterized=True)
    # Ribosomal/housekeeping
    ribo = cat[cat["is_ribosomal"]]
    ax_a.scatter(ribo["y_pred"], ribo["log_expr"], s=12, alpha=0.7, color=CORRECT_COL,
                 edgecolors="white", linewidths=0.3, label=f"Housekeeping (n={len(ribo)})", zorder=3)
    # Photosynthetic
    photo = cat[cat["is_photosynthetic"]]
    ax_a.scatter(photo["y_pred"], photo["log_expr"], s=12, alpha=0.7, color=WRONG_COL,
                 edgecolors="white", linewidths=0.3, label=f"Cyanobacteria-specific (n={len(photo)})", zorder=3)

    ax_a.set_xlabel("Model prediction (z-scored)")
    ax_a.set_ylabel("Actual expression (log₁₀ DIA intensity)")
    ax_a.set_title("Universal genes correctly ranked;\nphylum-specific genes invisible", fontsize=7, fontweight="bold")
    ax_a.legend(fontsize=5, loc="upper left", frameon=True, facecolor="white")

    rho_all, _ = spearmanr(cat["y_pred"], cat["log_expr"])
    ax_a.text(0.95, 0.05, f"Overall ρ = {rho_all:.3f}", transform=ax_a.transAxes,
              fontsize=6, ha="right", va="bottom")

    # Panel b (decile) and d (model comparison) dropped per coauthor — a+c sufficient.

    # ── Panel b: Multi-gene operon vs singleton (was panel c) ─────────
    ax_c = fig.add_subplot(gs[0, 1])
    ax_c.text(-0.15, 1.08, "b", transform=ax_c.transAxes, fontsize=12, fontweight="bold", va="top")

    operon_counts = cat.groupby("operon_id").size()
    cat["op_size"] = cat["operon_id"].map(operon_counts)
    singleton = cat[cat["op_size"] == 1]
    multigene = cat[cat["op_size"] > 1]

    categories = ["Singleton genes", "Multi-gene operon"]
    rhos = []
    ns = []
    for sub, label in [(singleton, "Singleton"), (multigene, "Multi-gene")]:
        if len(sub) > 10:
            r, p = spearmanr(sub["y_pred"], sub["log_expr"])
            rhos.append(r)
            ns.append(len(sub))
        else:
            rhos.append(0)
            ns.append(0)

    colors_c = [NEUTRAL_COL, CORRECT_COL]
    bars = ax_c.bar(categories, rhos, color=colors_c, edgecolor="white", width=0.5)
    for bar, r, n in zip(bars, rhos, ns):
        ax_c.text(bar.get_x() + bar.get_width() / 2, r + 0.01, f"ρ={r:.3f}\n(n={n})",
                  ha="center", fontsize=5.5, fontweight="bold")
    ax_c.set_ylabel("Spearman ρ")
    ax_c.set_title("Operon context transfers across phyla;\nsingleton prediction is noise-level", fontsize=7, fontweight="bold")
    ax_c.set_ylim(0, max(rhos) * 1.4 if max(rhos) > 0 else 0.3)

    # Panel d (all-tier model comparison) REMOVED per coauthor — panels a+c sufficient.
    # Key values preserved in caption: 1B baseline ρ=0.147, XP5 dropout ρ=0.073.

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"si_synechococcus_stratified.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
        log.info(f"  Saved: {path}")
    plt.close(fig)


def generate_si_confounds():
    """SI figure: Confound analysis summary."""
    log.info("Generating SI: Confound analysis summary...")

    # Load results
    confound_path = PROJECT_ROOT / "results" / "expression_confound_analysis" / "confound_analysis_results.json"
    if not confound_path.exists():
        log.warning(f"  Confound results not found: {confound_path}")
        return

    with open(confound_path) as f:
        results = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
    })

    fig = plt.figure(figsize=(7.2, 5.5), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── Panel a: Simpson's paradox test ───────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(-0.15, 1.08, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="top")

    simpsons = results.get("A10_simpsons_paradox", {})
    nm5_data = simpsons.get("nm5_vs_shuffled_labels", {})

    if nm5_data:
        actual = nm5_data.get("actual_rho", 0.663)
        shuffled = nm5_data.get("shuffled_rho_mean", 0.006)
        genuine = nm5_data.get("genuine_within_species_signal", 0.657)

        bars = ax_a.bar(
            ["Full model\nρ", "Shuffled\n(species only)", "Genuine\nwithin-species"],
            [actual, shuffled, genuine],
            color=[CORRECT_COL, NEUTRAL_COL, HIGHLIGHT_COL],
            edgecolor="white",
            width=0.5,
        )
        for bar, val in zip(bars, [actual, shuffled, genuine]):
            ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                      ha="center", fontsize=6, fontweight="bold")

    ax_a.set_ylabel("Spearman ρ")
    ax_a.set_ylim(0, 0.75)
    ax_a.set_title("Simpson's paradox test:\n99% of ρ is within-species", fontsize=7, fontweight="bold")

    # ── Panel b: Per-species ρ distribution ───────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(-0.15, 1.08, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="top")

    # Load per-species data
    sp_path = PROJECT_ROOT / "results" / "expression_confound_analysis" / "a3_per_species_rho_nm5_fold0.csv"
    if sp_path.exists():
        sp_df = pd.read_csv(sp_path)
        ax_b.hist(sp_df["rho"], bins=30, color=CORRECT_COL, edgecolor="white", alpha=0.8)
        median_rho = sp_df["rho"].median()
        ax_b.axvline(median_rho, color=HIGHLIGHT_COL, linewidth=1.5, linestyle="--")
        ax_b.text(median_rho - 0.02, ax_b.get_ylim()[1] * 0.85,
                  f"median = {median_rho:.3f}", fontsize=6, color=HIGHLIGHT_COL,
                  fontweight="bold", ha="right")
        ax_b.set_xlabel("Per-species Spearman ρ")
        ax_b.set_ylabel("Number of species")
        # Legend box moved to top-LEFT (per user feedback 2026-04-13).
        ax_b.text(0.05, 0.95, f"N = {len(sp_df)} species\n76% > 0.6\n1% < 0.3",
                  transform=ax_b.transAxes, fontsize=5.5, ha="left", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E8E8E8"))
    ax_b.set_title("Per-species prediction quality:\nmedian ρ > overall ρ", fontsize=7, fontweight="bold")

    # ── Panel c: A8 native vs z-scored ────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.text(-0.15, 1.08, "c", transform=ax_c.transAxes, fontsize=12, fontweight="bold", va="top")

    metrics = {
        "XP5\nz-scored": {"overall": 0.656, "per_sp": 0.678},
        "XP5\nnative": {"overall": 0.884, "per_sp": 0.614},
    }

    x = np.arange(2)
    width = 0.3
    overall_vals = [metrics["XP5\nz-scored"]["overall"], metrics["XP5\nnative"]["overall"]]
    persp_vals = [metrics["XP5\nz-scored"]["per_sp"], metrics["XP5\nnative"]["per_sp"]]

    bars1 = ax_c.bar(x - width / 2, overall_vals, width, label="Overall ρ", color=CORRECT_COL, edgecolor="white")
    bars2 = ax_c.bar(x + width / 2, persp_vals, width, label="Per-species median ρ", color=HIGHLIGHT_COL, edgecolor="white")

    for bar, val in zip(list(bars1) + list(bars2), overall_vals + persp_vals):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                  ha="center", fontsize=5.5, fontweight="bold")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["XP5\nz-scored", "XP5\nnative"])
    ax_c.set_ylabel("Spearman ρ")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlim(-0.55, 1.75)  # leave room on the right for arrow labels
    ax_c.legend(fontsize=5.5, loc="upper left")
    ax_c.set_title("Native labels inflate overall ρ\nbut hurt per-species ρ", fontsize=7, fontweight="bold")

    # Arrow showing the paradox — arrows close to the second x position
    # (NM5 native), with labels pulled INSIDE the panel to avoid overflow
    # past the right axis line.
    ax_c.annotate("", xy=(1.17, 0.884), xytext=(1.17, 0.656),
                  arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5))
    ax_c.text(1.22, 0.77, "+0.228\n(inflated)", fontsize=5, color="#2ca02c",
              ha="left", va="center")
    ax_c.annotate("", xy=(1.40, 0.614), xytext=(1.40, 0.678),
                  arrowprops=dict(arrowstyle="->", color=WRONG_COL, lw=1.5))
    ax_c.text(1.45, 0.645, "−0.064\n(real loss)", fontsize=5, color=WRONG_COL,
              ha="left", va="center")

    # ── Panel d: Trivial baselines ────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.text(-0.15, 1.08, "d", transform=ax_d.transAxes, fontsize=12, fontweight="bold", va="top")

    baselines = results.get("A9_trivial_baselines", {})
    zs = baselines.get("z-scored", {})

    baseline_items = [
        ("Species mean", zs.get("species_mean", {}).get("rho", -0.056)),
        ("Protein length", zs.get("protein_length", {}).get("rho", -0.066)),
        ("Operon size", zs.get("operon_size", {}).get("rho", -0.019)),
    ]

    # Add model for comparison
    nm5 = baselines.get("nm5_champion", {})
    if nm5:
        baseline_items.append(("XP5 champion", nm5.get("rho_overall", 0.663)))

    names_d = [b[0] for b in baseline_items]
    vals_d = [b[1] for b in baseline_items]
    colors_d = [NEUTRAL_COL if abs(v) < 0.1 else CORRECT_COL for v in vals_d]

    ax_d.barh(range(len(baseline_items)), vals_d, color=colors_d, edgecolor="white", height=0.5)
    ax_d.set_yticks(range(len(baseline_items)))
    ax_d.set_yticklabels(names_d, fontsize=6)
    ax_d.set_xlabel("Spearman ρ (z-scored test labels)")
    ax_d.axvline(0, color=INK, linewidth=0.5)
    for i, v in enumerate(vals_d):
        ax_d.text(max(v, 0) + 0.01, i, f"{v:.3f}", fontsize=5.5, va="center", fontweight="bold")
    ax_d.set_title("Trivial baselines at noise floor;\nmodel signal is genuine", fontsize=7, fontweight="bold")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"si_confound_analysis.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
        log.info(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    generate_si_synechococcus()
    generate_si_confounds()
    log.info("Done.")
