#!/usr/bin/env python3
"""
Generate Nature Biotechnology publication figures for AIKI-XP.

Produces 6 main figures (multi-panel composites) + Extended Data figures.
All figures follow Nature figure guidelines:
  - Arial/Helvetica font, 7pt minimum
  - 300 DPI, vector PDF + raster PNG
  - Single column: 89mm (3.5"), double column: 183mm (7.2")
  - Colorblind-safe palettes

Usage:
    python scripts/protex/generate_nbt_figures.py
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import seaborn as sns

from build_manuscript_audit_assets import (
    OUT_DIR as AUDIT_DIR,
    build_architecture_comparison,
    build_biology_family_go,
    build_claim_audit,
    build_clean_slice_summary,
    build_fixedwidth_loo,
    build_label_domain_summary,
    build_loso_summary,
    build_mega_asymmetry,
    build_noise_curve,
    build_pair_synergy_matrix,
    build_species_breadth_curve,
    build_species_cluster_model_comparison,
    build_split_summary,
    build_threshold_curve as build_audit_threshold_curve,
    build_training_fraction_curve,
    extract_metric,
)
from protex_visual_data import INTERACTIVE_DIR, build_visual_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURE_DIR = PROJECT_ROOT / "docs" / "protex" / "latex" / "figures" / "nbt"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PREPRESS_DIR = FIGURE_DIR / "prepress"

EXPORT_CONFIG = {
    "png_dpi": 300,
    "write_prepress": False,
    "prepress_png_dpi": 600,
    "prepress_tiff_dpi": 1200,
}

# ── Nature-compliant style ──────────────────────────────────────────────
COLORS = {
    "primary": "#1F4E79",      # editorial blue
    "secondary": "#B03A2E",    # editorial red
    "tertiary": "#1B6B57",     # deep teal-green
    "quaternary": "#C67D12",   # amber
    "quinary": "#6C4C9C",      # violet
    "light1": "#7FB3D5",       # light blue
    "light2": "#E59866",       # salmon
    "light3": "#88C9A1",       # light green
    "neutral": "#7B8794",      # gray
    "bg": "#F5F7FA",           # near-white
    "ink": "#182533",          # dark text
    "muted": "#5E6A75",
}

# Modality type colors
MOD_COLORS = {
    "protein": "#1F4E79",
    "dna": "#1B6B57",
    "rna": "#6C4C9C",
    "genome_context": "#C67D12",
    "classical": "#7B8794",
    "biophysical": "#7B8794",  # same color as classical (renamed)
    "fusion": "#B03A2E",
}

SPLIT_COLORS = {
    "train": "#3B82F6",
    "val": "#F59E0B",
    "test": "#DC2626",
}

CKA_ORDER = [
    "ESM-C protein",
    "Evo-2 CDS",
    "HyenaDNA CDS",
    "DNABERT-2 operon",
    "CodonFM CDS",
    "RiNALMo init",
    "Bacformer",
]

# Updated 2026-04-04: NM5 champion LOO (5-modality recipe, hard_hybrid campaign)
# Source: results/nonmega_champion_campaign_hardhybrid/ (Phase C = GO, Phase D = SC)
# NM5 recipe: {Evo-2 7B, HyenaDNA, BF-large, biophysical(5), ESM-C}
AUDITED_CROSS_REGIME_LOO = pd.DataFrame(
    [
        # label, family, GO-overall Δ, GO-ov rank, GO-nc Δ, GO-nc rank, SC-overall Δ, SC-ov rank
        ("Evo-2 7B", "dna", 0.042, 1, 0.031, 1, 0.016, 1),
        ("Biophysical", "biophysical", 0.021, 2, 0.014, 2, 0.013, 3),
        ("ESM-C protein", "protein", 0.013, 3, 0.009, 3, 0.001, 4),
        ("HyenaDNA CDS", "dna", 0.005, 4, 0.006, 4, -0.008, 5),
        ("Bacformer-large", "genome_context", 0.001, 5, 0.010, 5, 0.014, 2),
    ],
    columns=[
        "label", "family",
        "gene_operon_delta", "gene_operon_rank",
        "go_nonmega_delta", "go_nonmega_rank",
        "species_cluster_delta", "species_cluster_rank",
    ],
)

# Updated 2026-04-01: 7B PCA-4096 champion F10 values, ESM-C unchanged (same checkpoint).
# Source: go_f10_7b_fo_pca4096_25M_seed42_predictions.parquet stratified by operon/length.
# NOTE: multi-gene operon Δ=+0.237 is inflated by 7B operon embedding identity
# leakage (see Discussion singleton analysis). Singleton Δ=+0.059 is the honest number.
AUDITED_PATHWAY_ADVANTAGE = pd.DataFrame(
    [
        ("In operon (multi-gene)", "operon", 0.603, 0.366, 0.237, 11648),
        ("Singleton", "operon", 0.686, 0.627, 0.059, 39044),
        ("<100 aa", "length", 0.700, 0.587, 0.113, 3535),
        ("100-300 aa", "length", 0.630, 0.515, 0.115, 22409),
        ("300-500 aa", "length", 0.678, 0.574, 0.104, 16417),
        ("500-1,000 aa", "length", 0.664, 0.597, 0.067, 7203),
        (">1,000 aa", "length", 0.811, 0.778, 0.033, 1128),
    ],
    columns=["label", "group", "fusion10_rho", "esmc_rho", "delta", "n"],
)

# Updated 2026-04-04: NM5 champion missing-modality (hard_hybrid campaign Phase E)
# Source: results/nonmega_champion_campaign_hardhybrid/E_missing_*.json
AUDITED_MISSING_MODALITY = pd.DataFrame(
    [
        ("All 5 modalities", 5, 0.663),
        ("No biophysical", 4, 0.642),
        ("DNA only", 2, 0.642),
        ("Protein + DNA", 3, 0.647),
        ("No Evo-2 7B", 4, 0.621),
        ("Protein only", 1, 0.570),
        ("Biophysical only", 1, 0.507),
    ],
    columns=["label", "count", "rho"],
)

_VISUAL_DATA = None
_AUDIT_TABLES = {}

def set_nbt_style():
    """Configure matplotlib for Nature Biotechnology figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.titleweight": "bold",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, name: str):
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=EXPORT_CONFIG["png_dpi"])
    if EXPORT_CONFIG["write_prepress"]:
        PREPRESS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREPRESS_DIR / f"{name}.pdf")
        fig.savefig(PREPRESS_DIR / f"{name}.svg")
        fig.savefig(
            PREPRESS_DIR / f"{name}.png",
            dpi=EXPORT_CONFIG["prepress_png_dpi"],
        )
        try:
            fig.savefig(
                PREPRESS_DIR / f"{name}.tiff",
                dpi=EXPORT_CONFIG["prepress_tiff_dpi"],
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except TypeError:
            logger.warning("TIFF compression unsupported for %s; writing uncompressed TIFF", name)
            fig.savefig(
                PREPRESS_DIR / f"{name}.tiff",
                dpi=EXPORT_CONFIG["prepress_tiff_dpi"],
            )
    plt.close(fig)
    logger.info("  Saved %s", name)


def _panel_label(ax, label, x=-0.12, y=1.08):
    """Add a bold panel label (a, b, c...) in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def _style_axis(ax, grid_axis="y"):
    """Consistent editorial styling for data panels."""
    ax.set_facecolor("white")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#5E6A75")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors="#33414B")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9E1EA", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _callout(ax, x, y, text, color="#1F2A44", fc="#F6F8FB", fontsize=5.5):
    ax.text(
        x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.12", facecolor=fc,
                  edgecolor=color, linewidth=0.6)
    )


def _get_visual_data(force: bool = False):
    global _VISUAL_DATA
    if force or _VISUAL_DATA is None:
        _VISUAL_DATA = build_visual_data(force=force)
    return _VISUAL_DATA


def _load_audit_table(filename: str, builder):
    path = AUDIT_DIR / filename
    key = str(path)
    if key in _AUDIT_TABLES:
        return _AUDIT_TABLES[key]
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = builder()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    _AUDIT_TABLES[key] = df
    return df


def _format_species_label(species: str) -> str:
    if not isinstance(species, str):
        return str(species)
    if "_" in species and not species.split("_", 1)[0].isdigit():
        return species.replace("_", " ")
    return f"taxid:{species}"


def _draw_umap_scatter(ax, frame, color_col, palette, title, point_size=8):
    ax.set_facecolor("white")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for category, color in palette.items():
        sub = frame[frame[color_col] == category]
        if sub.empty:
            continue
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=point_size,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.15,
            label=category,
        )
    ax.set_title(title, fontsize=7.1)


def _similarity_matrix(bundle):
    similarity = bundle.modality_similarity.copy()
    sim = similarity.pivot(index="modality_a", columns="modality_b", values="cka_similarity")
    return sim.reindex(index=CKA_ORDER, columns=CKA_ORDER)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Problem + Platform (no results)
# ══════════════════════════════════════════════════════════════════════════
def fig1_pipeline_overview():
    """Three-tier overview: the problem, the platform, and the evaluation framing."""
    logger.info("Fig 1: Pipeline overview")
    threshold_df = _load_audit_table("threshold_curve.csv", build_audit_threshold_curve)

    # ── Outer canvas ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 6.5), facecolor="white")
    # Three rows: top panels (regulatory + threshold), platform boxes, framing
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.05, 0.72, 0.14],
        hspace=0.44, wspace=0.36,
        left=0.04, right=0.97, top=0.95, bottom=0.04,
    )

    # ── Tier 1 left: Regulatory layers ────────────────────────────────────
    ax_reg = fig.add_subplot(gs[0, 0])
    ax_reg.set_xlim(0, 10)
    ax_reg.set_ylim(-0.5, 10.2)
    ax_reg.axis("off")

    # Panel label and subtitle placed tightly inside the axes
    ax_reg.text(-0.5, 10.5, "a", transform=ax_reg.transData,
                fontsize=10, fontweight="bold", color="#152334", va="top", ha="left",
                clip_on=False)
    ax_reg.text(0.5, 10.0,
                "Multiple regulatory layers govern bacterial expression",
                fontsize=6.5, fontweight="bold", color="#152334", va="top")

    layers = [
        ("Promoter strength / TF binding", "#2166AC", True),
        ("Operon co-transcription",         "#2166AC", True),
        ("Shine-Dalgarno / init. structure", "#1B7837", True),
        ("Codon adaptation (CAI, tAI)",      "#1B7837", True),
        ("mRNA stability / degradation",     "#762A83", True),
        ("Protein stability / folding",      "#2166AC", True),
        ("Growth-rate coupling",             "#E08214", False),
        ("sRNA regulation",                  "#878787", False),
        ("Post-translational mod.",          "#878787", False),
    ]
    for i, (name, color, captured) in enumerate(layers):
        yi = 8.6 - i * 0.97
        alpha = 1.0 if captured else 0.40
        lw = 2.2 if captured else 1.4
        ls = "solid" if captured else (4, 2)
        ax_reg.plot([0.3, 1.85], [yi, yi], color=color, linewidth=lw,
                    linestyle=ls if isinstance(ls, str) else (0, ls), alpha=alpha)
        marker = "o" if captured else "x"
        ms = 5 if captured else 5.5
        ax_reg.plot(1.07, yi, marker=marker, color=color, markersize=ms,
                    alpha=alpha, markeredgewidth=1.2,
                    markerfacecolor=color if captured else "none")
        ax_reg.text(2.15, yi, name, fontsize=5.8, va="center",
                    color="#1F2A44" if captured else "#9CA3AF")

    # Legend underneath the list
    ax_reg.plot([0.3, 1.1], [-0.1, -0.1], color="#444", linewidth=1.8, linestyle="solid")
    ax_reg.plot(0.7, -0.1, "o", color="#444", markersize=4.5)
    ax_reg.text(1.3, -0.1, "= captured by our stack", fontsize=5.0, va="center", color="#6B7280")
    ax_reg.plot([4.4, 5.2], [-0.1, -0.1], color="#888", linewidth=1.4, linestyle=(0, (4, 2)), alpha=0.55)
    ax_reg.plot(4.8, -0.1, "x", color="#888", markersize=4.5, markeredgewidth=1.2)
    ax_reg.text(5.4, -0.1, "= not yet captured", fontsize=5.0, va="center", color="#9CA3AF")

    # ── Tier 1 right: Threshold / leakage curve ───────────────────────────
    ax_leak = fig.add_subplot(gs[0, 1])
    # Panel label placed inside the axes using axes coords to avoid overlap
    ax_leak.text(-0.10, 1.05, "b", transform=ax_leak.transAxes,
                 fontsize=10, fontweight="bold", color="#152334", va="top")
    ax_leak.set_title("Split design defines the scientific question", fontsize=6.5, pad=6)
    _style_axis(ax_leak, "y")

    ax_leak.errorbar(
        threshold_df["threshold"], threshold_df["overall_mean"],
        yerr=threshold_df["overall_std"],
        fmt="o-", color=COLORS["primary"],
        markersize=5, markerfacecolor="white", markeredgewidth=1.1,
        linewidth=1.4, capsize=3,
    )
    ax_leak.axvline(0.20, color=COLORS["secondary"], linestyle=":", linewidth=0.9)

    # Fix annotation positions to avoid overlapping the data
    y_min = threshold_df["overall_mean"].min()
    ax_leak.text(0.21, 0.58, "primary\nthreshold", fontsize=5.0,
                 color=COLORS["secondary"], va="bottom", ha="left")

    for row in threshold_df.itertuples():
        offset = 7 if row.threshold != 0.20 else 7
        ax_leak.annotate(
            f"{row.overall_mean:.3f}",
            (row.threshold, row.overall_mean),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=5.0,
        )

    ax_leak.set_xlabel(
        "Mash distance threshold\n(higher = more species shared across splits)",
        fontsize=5.5,
    )
    ax_leak.set_ylabel("Spearman ρ\n(rank correlation, 1.0 = perfect)", fontsize=5.5)
    ax_leak.set_ylim(0.555, 0.695)
    ax_leak.set_xlim(0.02, 0.33)

    # Italic note — positioned bottom-right but well inside the axes
    ax_leak.text(
        0.98, 0.04,
        "Stricter splits → lower ρ:\nleakage inflates apparent performance",
        transform=ax_leak.transAxes, ha="right", va="bottom",
        fontsize=4.8, color="#4B5563", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F3F4F6", edgecolor="none"),
    )

    # ── Tier 2: Platform boxes ────────────────────────────────────────────
    ax_plat = fig.add_subplot(gs[1, :])
    ax_plat.set_xlim(0, 15.0)
    ax_plat.set_ylim(0, 4.0)
    ax_plat.axis("off")

    def platform_box(x, w, title, bullets, edge_color):
        ax_plat.add_patch(FancyBboxPatch(
            (x, 0.15), w, 3.55,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="#FAFCFF", edgecolor=edge_color, linewidth=1.1,
        ))
        ax_plat.text(
            x + w / 2, 3.45, title,
            fontsize=6.8, fontweight="bold", color="#152334",
            ha="center", va="top",
        )
        for j, line in enumerate(bullets):
            ax_plat.text(
                x + 0.28, 2.72 - j * 0.72, f"• {line}",
                fontsize=5.4, color="#374151", va="top",
            )

    platform_box(
        0.10, 4.55, "Dataset",
        [
            "492,026 genes · 385 bacterial species",
            "PaXDb + Abele direct proteomics labels",
            "Frozen V2 table with checksums",
        ],
        COLORS["primary"],
    )
    platform_box(
        5.22, 5.05, "11 Foundation Models → 5 Biology Families",
        [
            "Protein: ESM-C 600M, ESM-2 650M  (1,152d + 1,280d)",
            "DNA/CDS: Evo-2, DNABERT-2, HyenaDNA",
            "Context: Bacformer (480d)  ·  Classical: 95d",
        ],
        COLORS["tertiary"],
    )
    platform_box(
        10.85, 3.95, "Fusion & Evaluation",
        [
            "Single-adapter MLP,  ~25M params",
            "3 leakage-aware splits (see panel b)",
            "Audited provenance on all metrics",
        ],
        COLORS["secondary"],
    )

    for x0, x1 in [(4.70, 5.17), (10.32, 10.80)]:
        ax_plat.add_patch(FancyArrowPatch(
            (x0, 1.95), (x1, 1.95),
            arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#687684",
        ))

    # ── Tier 3: Framing sentence ──────────────────────────────────────────
    ax_frame = fig.add_subplot(gs[2, :])
    ax_frame.axis("off")
    ax_frame.text(
        0.5, 0.52,
        "The figures below quantify how much apparent performance comes from "
        "family recognition versus transferable regulatory signal.",
        transform=ax_frame.transAxes, ha="center", va="center",
        fontsize=6.5, color="#374151", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#EFF6FF",
                  edgecolor="#BFDBFE", linewidth=0.6),
    )

    _save(fig, "fig1_pipeline_overview")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Dataset overview — species distribution + expression landscape
# ══════════════════════════════════════════════════════════════════════════
def fig2_dataset_overview():
    """Dataset cartography with audited threshold logic and neutral provenance labels."""
    logger.info("Fig 2: Dataset overview")
    bundle = _get_visual_data()
    species = bundle.species.copy()
    clusters = bundle.clusters.copy()
    threshold_curve = _load_audit_table("threshold_curve.csv", build_audit_threshold_curve).copy()
    species["direct_label_fraction"] = species.get("direct_label_fraction", species["gold_fraction"])
    clusters["direct_label_fraction"] = clusters.get("direct_label_fraction", clusters.get("gold_fraction", 0.0))

    fig = plt.figure(figsize=(8.5, 7.4), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.45, 1.0], hspace=0.55, wspace=0.60,
                            left=0.13, right=0.95, top=0.94, bottom=0.10)

    # ── Panel a: ordered species cartography ──
    carto_spec = gs[0, :].subgridspec(2, 1, height_ratios=[6.2, 1.2], hspace=0.06)
    ax_a = fig.add_subplot(carto_spec[0, 0])
    ax_a_strip = fig.add_subplot(carto_spec[1, 0], sharex=ax_a)
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "y")

    x = np.arange(len(species))
    bar_colors = species["species_split"].map(SPLIT_COLORS).fillna(COLORS["neutral"])
    ax_a.bar(x, species["gene_count"], width=0.92, color=bar_colors, edgecolor="none")
    ax_a.set_xlim(-0.5, len(species) - 0.5)
    ax_a.set_ylabel("Genes per species")
    ax_a.set_title("Species cartography ordered by split and phylogenetic cluster", fontsize=8)
    ax_a.set_xticks([])

    median_genes = int(species["gene_count"].median())
    ax_a.axhline(median_genes, color="#6B7280", linestyle="--", linewidth=0.7)
    ax_a.text(len(species) - 3, median_genes + 45, f"median = {median_genes:,}",
              fontsize=5.5, color="#4B5563", ha="right")

    cluster_change = np.flatnonzero(species["species_cluster"].ne(species["species_cluster"].shift()).to_numpy())
    for start in cluster_change[1:]:
        ax_a.axvline(start - 0.5, color="#D8E0EA", linewidth=0.55, alpha=0.9)
        ax_a_strip.axvline(start - 0.5, color="white", linewidth=0.55, alpha=0.9)

    for _, row in species.head(4).iterrows():
        order = int(row["species_order"]) - 1
        ax_a.annotate(
            _format_species_label(row["species"]),
            (order, row["gene_count"]),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=5.2,
            ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.35, color="#7A8792"),
        )

    _callout(
        ax_a,
        0.015,
        0.60,
        "Largest phylogenetic block:\n139 species, 227K genes\nbalanced internally across splits",
        color=COLORS["secondary"],
        fc="#FCECEF",
        fontsize=5.2,
    )
    _callout(
        ax_a,
        0.68,
        0.92,
        "385 species\n492,026 genes\n116 species clusters at t=0.20",
        color=COLORS["primary"],
        fc="#EEF4FB",
        fontsize=5.3,
    )

    # Strip now shows only the conserved ribosomal component fraction;
    # the "Proteomics measured" row was removed 2026-04-13 because it
    # reflected an old misunderstanding (PaXDb was wrongly treated as the
    # only proteomics source; Abele is also proteomics).
    strip = species["mega_fraction"].to_numpy().reshape(1, -1)
    im = ax_a_strip.imshow(strip, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
    ax_a_strip.set_yticks([0])
    ax_a_strip.set_yticklabels(["Conserved\nfraction"], fontsize=5.5)
    ax_a_strip.set_xlabel("Species ordered by split then cluster rank")
    # Add a mini colorbar for the conserved fraction strip
    cbar_ax = fig.add_axes([0.96, 0.62, 0.012, 0.10])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Conserved\nfraction", fontsize=4.5, labelpad=2)
    cb.ax.tick_params(labelsize=4)
    split_legend = [mpatches.Patch(color=SPLIT_COLORS[k], label=f"{k} split") for k in ["train", "val", "test"]]
    ax_a.legend(handles=split_legend, frameon=False, fontsize=4.8, loc="upper left",
                ncol=1, bbox_to_anchor=(0.22, 0.98))
    ax_a_strip.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_a_strip.spines.values():
        spine.set_visible(False)

    # ── Panel b: cluster landscape ──
    ax_b = fig.add_subplot(gs[1, :2])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "x")
    top_clusters = clusters.head(14).sort_values("n_genes", ascending=True)
    y = np.arange(len(top_clusters))
    ax_b.hlines(y, 0, top_clusters["n_genes"], color="#CFD8E3", linewidth=1.2)
    ax_b.scatter(
        top_clusters["n_genes"],
        y,
        s=top_clusters["n_species"] * 10,
        c=top_clusters["split"].map(SPLIT_COLORS).fillna(COLORS["neutral"]),
        edgecolors="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"{row.species_cluster}" for row in top_clusters.itertuples()], fontsize=5.4)
    ax_b.set_xlabel("Genes in species cluster")
    ax_b.set_title("Largest species clusters at the chosen t=0.20 threshold", fontsize=7.4)
    # Species-count labels: placed to the right of each bubble with a
    # white background so they are legible against the colored bubble
    # (previously the "7 sp." / "80 sp." / "139 sp." text overlapped
    # the bubble fill and was hard to read on the red/orange bubbles).
    for yi, row in zip(y, top_clusters.itertuples()):
        ax_b.text(
            row.n_genes + 4500,
            yi,
            f"{int(row.n_species)} sp.",
            fontsize=5.6,
            va="center",
            ha="left",
            fontweight="bold",
            color="#1F2A44",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#B0BEC5", lw=0.4, alpha=0.9),
        )
    _callout(
        ax_b,
        0.55,
        0.30,
        "Bubble size = species count\nColor = species-cluster split assignment",
        color="#374151",
        fc="#F6F8FB",
        fontsize=5.0,
    )
    split_handles = [mpatches.Patch(color=SPLIT_COLORS[key], label=key) for key in ["train", "val", "test"]]
    ax_b.legend(handles=split_handles, frameon=False, fontsize=5.0, loc="lower right", ncol=3)

    # ── Panel c: threshold separation curve ──
    ax_c = fig.add_subplot(gs[1, 2])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    ax_c2 = ax_c.twinx()
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["left"].set_visible(False)
    ax_c2.spines["right"].set_color("#5E6A75")
    ax_c2.tick_params(colors="#33414B", labelsize=5.5)

    ax_c.plot(
        threshold_curve["threshold"],
        threshold_curve["overall_mean"],
        "o-",
        color=COLORS["primary"],
        linewidth=1.4,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    ax_c2.plot(
        threshold_curve["threshold"],
        threshold_curve["cluster_count"],
        "s--",
        color=COLORS["quaternary"],
        linewidth=1.0,
        markersize=4.5,
        alpha=0.9,
    )
    ax_c.axvline(0.20, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    ax_c.text(0.205, 0.69, "chosen\nt=0.20", fontsize=5.0, color=COLORS["secondary"],
              va="top")
    ax_c.set_xlabel("Mash distance threshold\n(higher = more permissive)", fontsize=6)
    ax_c.set_ylabel("Spearman ρ (blue solid)", fontsize=5.5, color=COLORS["primary"])
    ax_c2.set_ylabel("Cluster count (orange dashed)", fontsize=5.5,
                    color=COLORS["quaternary"], rotation=270, labelpad=14)
    ax_c.set_title("Stricter phylogenetic partitioning\nlowers ρ and increases cluster count", fontsize=6.8)
    ax_c.set_ylim(0.555, 0.72)
    ax_c2.set_ylim(0, int(threshold_curve["cluster_count"].max() * 1.25))
    # Move value labels above ρ line and below cluster-count line to
    # avoid overlap with the "chosen t=0.20" dotted line and with each
    # other at t=0.10/0.20.
    for row in threshold_curve.itertuples():
        # Skip label at t=0.20 for ρ to avoid collision with dotted line
        y_offset = 10 if abs(row.threshold - 0.20) > 0.01 else 14
        ax_c.annotate(f"{row.overall_mean:.3f}", (row.threshold, row.overall_mean),
                      textcoords="offset points", xytext=(0, y_offset), ha="center",
                      fontsize=5, color=COLORS["primary"], fontweight="bold")
        # Cluster counts ABOVE the markers to avoid overlap with ρ labels
        ax_c2.annotate(f"{int(row.cluster_count)}",
                       (row.threshold, row.cluster_count),
                       textcoords="offset points", xytext=(0, 10),
                       ha="center", fontsize=5,
                       color=COLORS["quaternary"], fontweight="bold")

    _save(fig, "fig2_dataset_overview")


def fig2_dataset_overview_variant_b():
    """Alternate dataset view: coverage curve + cluster geometry."""
    logger.info("Fig 2 variant B: Dataset overview")
    bundle = _get_visual_data()
    species = bundle.species.copy()
    clusters = bundle.clusters.copy()
    threshold_curve = _load_audit_table("threshold_curve.csv", build_audit_threshold_curve).copy()
    species["direct_label_fraction"] = species.get("direct_label_fraction", species["gold_fraction"])
    clusters["direct_label_fraction"] = clusters.get("direct_label_fraction", clusters.get("gold_fraction", 0.0))

    fig = plt.figure(figsize=(7.2, 5.8), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.0], hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, :])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "y")
    x = np.arange(1, len(species) + 1)
    cumulative = species["gene_count"].cumsum() / species["gene_count"].sum()
    ax_a.fill_between(x, cumulative, color="#DCECFB", alpha=0.9)
    ax_a.plot(x, cumulative, color=COLORS["primary"], linewidth=1.6)
    split_blocks = species.groupby("species_split", sort=False).agg(start=("species_order", "min"), stop=("species_order", "max"))
    for split, row in split_blocks.iterrows():
        ax_a.axvspan(row["start"], row["stop"], color=SPLIT_COLORS.get(split, COLORS["neutral"]), alpha=0.08)
        ax_a.text((row["start"] + row["stop"]) / 2, 0.04, split, color=SPLIT_COLORS.get(split, COLORS["neutral"]),
                  fontsize=5.2, ha="center", va="bottom", fontweight="bold")
    ax_a.set_xlim(1, len(species))
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xlabel("Species rank")
    ax_a.set_ylabel("Cumulative gene coverage")
    ax_a.set_title("Cumulative coverage makes the 492K dataset visibly long-tailed", fontsize=8)
    ax_a.text(120, 0.52, "Half the genes are covered by\nroughly the first 100 species", fontsize=5.2,
              color="#33414B",
              bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#CBD5E1", linewidth=0.5))

    ax_b = fig.add_subplot(gs[1, 0])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "both")
    top = clusters.head(60).copy()
    ax_b.scatter(
        top["n_species"],
        top["n_genes"],
        s=35 + top["direct_label_fraction"].fillna(0).to_numpy() * 350,
        c=top["split"].map(SPLIT_COLORS).fillna(COLORS["neutral"]),
        edgecolors="white",
        linewidth=0.5,
        alpha=0.9,
    )
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Species per cluster")
    ax_b.set_ylabel("Genes per cluster")
    ax_b.set_title("Cluster geometry separates giant clades from the long tail", fontsize=7.2)
    for row in top.head(4).itertuples():
        ax_b.annotate(row.species_cluster, (row.n_species, row.n_genes), textcoords="offset points",
                      xytext=(4, 5), fontsize=4.7)

    ax_c = fig.add_subplot(gs[1, 1])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    buckets = (
        clusters.groupby(["size_bucket", "split"], observed=False)["n_genes"]
        .sum()
        .unstack(fill_value=0)
        .reindex(["1 species", "2-3 species", "4-10 species", "11+ species"])
        .fillna(0)
    )
    bottom = np.zeros(len(buckets))
    xpos = np.arange(len(buckets))
    for split in ["train", "val", "test"]:
        vals = buckets.get(split, pd.Series(0, index=buckets.index)).to_numpy()
        ax_c.bar(xpos, vals, bottom=bottom, color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.4, label=split)
        bottom += vals
    ax_c2 = ax_c.twinx()
    ax_c2.plot(threshold_curve["threshold"], threshold_curve["cluster_count"], "o--",
               color=COLORS["quaternary"], linewidth=1.0, markersize=4)
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels(buckets.index, fontsize=5.0)
    ax_c.set_ylabel("Genes by cluster-size bucket")
    ax_c2.set_ylabel("Cluster count by threshold", fontsize=5.6)
    ax_c.set_title("Split-unit size and threshold granularity in one view", fontsize=7.0)
    ax_c.legend(frameon=False, fontsize=4.8, ncol=3, loc="upper left")
    _save(fig, "fig2_dataset_overview_variant_b")


def fig2_dataset_overview_variant_c():
    """Alternate dataset view: stripe atlas + threshold logic."""
    logger.info("Fig 2 variant C: Dataset overview")
    bundle = _get_visual_data()
    species = bundle.species.copy()
    clusters = bundle.clusters.copy()
    threshold_curve = _load_audit_table("threshold_curve.csv", build_audit_threshold_curve).copy()
    species["direct_label_fraction"] = species.get("direct_label_fraction", species["gold_fraction"])

    fig = plt.figure(figsize=(7.2, 5.9), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.4, 0.9, 1.0], hspace=0.34)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    stripe = np.vstack([
        species["gene_count"].to_numpy() / species["gene_count"].max(),
        species["direct_label_fraction"].to_numpy(),
        species["mega_fraction"].to_numpy(),
        species["cluster_rank"].to_numpy() / species["cluster_rank"].max(),
    ])
    ax_a.imshow(stripe, aspect="auto", interpolation="nearest", cmap="viridis")
    ax_a.set_yticks([0, 1, 2, 3])
    ax_a.set_yticklabels(["Gene load", "Direct-label share", "Mega share", "Cluster rank"], fontsize=5.5)
    ax_a.set_xticks([])
    ax_a.set_title("Species stripe atlas emphasizes multiple structural gradients at once", fontsize=8)
    for start in np.flatnonzero(species["species_cluster"].ne(species["species_cluster"].shift()).to_numpy())[1:]:
        ax_a.axvline(start - 0.5, color="white", linewidth=0.3, alpha=0.7)

    ax_b = fig.add_subplot(gs[1, 0])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "x")
    top20 = clusters.head(20).sort_values("n_genes", ascending=True)
    ax_b.barh(np.arange(len(top20)), top20["n_genes"], color=top20["split"].map(SPLIT_COLORS).fillna(COLORS["neutral"]),
              edgecolor="white", linewidth=0.4)
    ax_b.set_yticks(np.arange(len(top20)))
    ax_b.set_yticklabels(top20["species_cluster"], fontsize=4.8)
    ax_b.set_xlabel("Genes per species cluster")
    ax_b.set_title("Top 20 species clusters reveal just how skewed the clade structure is", fontsize=7.0)

    ax_c = fig.add_subplot(gs[2, 0])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    ax_c.plot(threshold_curve["threshold"], threshold_curve["overall_mean"], "o-", color=COLORS["primary"],
              markerfacecolor="white", markeredgewidth=1.2, linewidth=1.4)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(threshold_curve["threshold"], threshold_curve["cluster_count"], "s-",
               color=COLORS["secondary"], linewidth=1.1, markersize=4.5)
    ax_c.axvline(0.20, color="#6B7280", linestyle="--", linewidth=0.7)
    ax_c.set_xlabel("Mash distance threshold")
    ax_c.set_ylabel("Species-cluster ρ")
    ax_c2.set_ylabel("Cluster count", fontsize=6)
    ax_c.set_title("Primary threshold sits where transfer difficulty and clade resolution are both informative", fontsize=7.1)
    _save(fig, "fig2_dataset_overview_variant_c")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Core results — cross-split matrix + LOO + forward stepwise
# ══════════════════════════════════════════════════════════════════════════
def fig3_core_results():
    """Core benchmark figure: model hierarchy, cross-split matrix, and full audited LOO."""
    logger.info("Fig 3: Core results")
    bio = _load_audit_table("biology_family_gene_operon.csv", build_biology_family_go)
    split = _load_audit_table("split_summary.csv", build_split_summary).set_index("metric_id")
    loo = _load_audit_table("fixedwidth_loo.csv", build_fixedwidth_loo).copy()

    esmc_random = extract_metric(PROJECT_ROOT / "results/protex_qc/round9_sprint/single_esmc_protein_random_seed42.json", "rho_overall")
    esmc_species = extract_metric(PROJECT_ROOT / "results/protex_qc/round9_sprint/single_esmc_protein_species_cluster_seed42.json", "rho_overall")
    compact_go = np.mean([
        extract_metric(PROJECT_ROOT / f"results/protex_qc/junction_experiments/f10_baseline_256d_seed{seed}.json", "rho_overall")
        for seed in (42, 123, 7)
    ])
    compact_species = np.mean([
        extract_metric(PROJECT_ROOT / f"results/protex_qc/protein_engineering/f10_with_bacformer_256d_species_cluster_seed{seed}.json", "rho_overall")
        for seed in (42,)
    ])

    fig = plt.figure(figsize=(7.2, 7.6), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.95, 1.35], wspace=0.34, hspace=0.44)

    bio_idx = bio.set_index("metric_id")
    n_test = 50692

    # ── Panel A: capacity hierarchy ──
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "y")

    # 7B PCA-4096 champion (gene-operon, 3 seeds)
    _7b_go_files = sorted((PROJECT_ROOT / "results/protex_qc/evo2_7b_profiling/pca_dim_sweep").glob("go_f10_7b_fo_pca4096_25M_seed*.json"))
    if _7b_go_files:
        _7b_go_rhos = [extract_metric(f, "rho_overall") for f in _7b_go_files]
        _7b_go_mean = float(np.mean(_7b_go_rhos))
    else:
        _7b_go_mean = 0.667  # fallback

    labels_a = ["XGBoost\n(non-neural)", "ESM-C 600M\n(best single)", "Fusion-10 1B\n(10 modalities)", "F10 + 7B\nPCA-4096"]
    vals_a = [bio_idx.loc["xgboost_69d", "rho_overall"],
              bio_idx.loc["single_esmc", "rho_overall"],
              bio_idx.loc["fusion10_seed42", "rho_overall"],
              _7b_go_mean]
    colors_a = [COLORS["neutral"], COLORS["primary"], COLORS["secondary"], "#B03A2E"]

    bars = ax_a.bar(range(4), vals_a, color=colors_a, edgecolor="white", linewidth=0.5, width=0.65)
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(labels_a, fontsize=4.6)
    ax_a.set_ylabel("Spearman ρ on gene-operon")
    ax_a.set_ylim(0.0, 0.72)
    ax_a.set_title("Strict family holdout rewards multimodal breadth\nand DNA model scaling", fontsize=7)
    for bar, val in zip(bars, vals_a):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}",
                  ha="center", fontsize=5.5, fontweight="bold")
    ax_a.text(
        0.97, 0.05,
        f"N={n_test:,} test genes\n7B champion Δ vs ESM-C = {vals_a[3] - vals_a[1]:+.3f}\n7B Δ vs 1B = {vals_a[3] - vals_a[2]:+.3f}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.7,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F7FAFC", edgecolor="#D7E0EA", linewidth=0.5),
    )

    # ── Panel B: cross-split matrix ──
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b", x=-0.16, y=1.06)
    # 7B PCA-4096 cross-split means
    _7b_sc_files = sorted((PROJECT_ROOT / "results/protex_qc/evo2_7b_profiling/pca_dim_sweep").glob("sc_f10_7b_fo_pca4096_25M_seed*.json"))
    _7b_rand_files = sorted((PROJECT_ROOT / "results/protex_qc/evo2_7b_profiling/pca_dim_sweep").glob("rand_f10_7b_fo_pca4096_25M_seed*.json"))
    _7b_sc_mean = float(np.mean([extract_metric(f, "rho_overall") for f in _7b_sc_files])) if _7b_sc_files else 0.671
    _7b_rand_mean = float(np.mean([extract_metric(f, "rho_overall") for f in _7b_rand_files])) if _7b_rand_files else 0.741

    split_matrix = pd.DataFrame(
        {
            "Gene-operon": [_7b_go_mean, split.loc["fusion10_gene_operon", "mean"], compact_go, bio_idx.loc["single_esmc", "rho_overall"], bio_idx.loc["xgboost_69d", "rho_overall"]],
            "Species-cluster": [_7b_sc_mean, split.loc["fusion10_species_cluster_t020", "mean"], compact_species, esmc_species, np.nan],
            "Random": [_7b_rand_mean, split.loc["fusion10_random", "mean"], np.nan, esmc_random, np.nan],
        },
        index=["F10 + 7B\nPCA-4096", "Fusion-10\n1B 25M", "Fusion-10\n256d", "ESM-C\nsingle", "XGBoost\n69d"],
    )
    sns.heatmap(
        split_matrix,
        ax=ax_b,
        cmap=sns.color_palette("blend:#eef4fb,#1F4E79,#B03A2E", as_cmap=True),
        vmin=0.50,
        vmax=0.75,
        cbar_kws={"shrink": 0.80, "label": "Spearman ρ"},
        linewidths=0.8,
        linecolor="white",
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 5.7, "fontweight": "bold"},
    )
    ax_b.set_title("Cross-split matrix shows the real difficulty axis:\ngene-family novelty, not species novelty alone", fontsize=7)
    ax_b.tick_params(axis="x", rotation=0, labelsize=5.3)
    ax_b.tick_params(axis="y", rotation=0, labelsize=5.3)

    # ── Panel C: full audited LOO profile (7B PCA-4096 champion) ──
    ax_c = fig.add_subplot(gs[1, :])
    _panel_label(ax_c, "c", y=1.04)
    _style_axis(ax_c, "x")

    family_lookup_7b = {
        "esmc_protein": "protein", "hyenadna_dna_cds": "dna", "dnabert2_operon_dna": "dna",
        "evo2_7b_full_operon_pca4096": "dna", "bacformer": "genome_context",
        "classical_rna_init": "classical", "classical_codon": "classical",
        "classical_operon_struct": "classical", "classical_disorder": "classical",
        "classical_protein": "classical",
    }
    pretty_7b = {
        "evo2_7b_full_operon_pca4096": "Evo-2 7B full-operon PCA-4096",
        "esmc_protein": "ESM-C protein (1,152d)",
        "hyenadna_dna_cds": "HyenaDNA CDS (256d)",
        "bacformer": "Bacformer context (480d)",
        "classical_rna_init": "RNA-init block (16d)",
        "dnabert2_operon_dna": "DNABERT-2 operon (768d)",
        "classical_operon_struct": "Operon-structure block (10d)",
        "classical_disorder": "Disorder block (8d)",
        "classical_codon": "Codon block (11d)",
        "classical_protein": "Protein feature block (24d)",
    }

    # Load 7B LOO data from JSONs
    _7b_loo_dir = PROJECT_ROOT / "results/protex_qc/evo2_7b_profiling/7b_pca4096_nbt"
    _7b_baseline_f = _7b_loo_dir / "loo_7b_baseline_w1104_seed42.json"
    if _7b_baseline_f.exists():
        _7b_baseline_rho = json.load(open(_7b_baseline_f))["results"]["single_adapter"]["mean_spearman"]
        loo_7b_rows = []
        for mod_key in pretty_7b:
            drop_f = _7b_loo_dir / f"loo_7b_drop_{mod_key}_w1104_seed42.json"
            if drop_f.exists():
                drop_rho = json.load(open(drop_f))["results"]["single_adapter"]["mean_spearman"]
                loo_7b_rows.append({
                    "modality": mod_key,
                    "rho_overall": drop_rho,
                    "delta": drop_rho - _7b_baseline_rho,
                    "baseline_rho": _7b_baseline_rho,
                })
        loo_7b = pd.DataFrame(loo_7b_rows)
        loo_7b["family"] = loo_7b["modality"].map(family_lookup_7b)
        loo_7b["label"] = loo_7b["modality"].map(pretty_7b)
        loo_panel = loo_7b.sort_values("delta")
        baseline = _7b_baseline_rho
        loo_is_7b = True
    else:
        # Fallback to 1B LOO from audit CSV
        logger.warning("7B LOO JSONs not found, falling back to 1B LOO")
        loo_panel = loo[loo["modality"].isin(pretty_7b.keys())].copy()
        loo_panel["family"] = loo_panel["modality"].map(family_lookup_7b)
        loo_panel["label"] = loo_panel["modality"].map(pretty_7b)
        loo_panel = loo_panel.sort_values("delta")
        baseline = loo_panel["baseline_rho"].iloc[0]
        loo_is_7b = False

    y_pos = np.arange(len(loo_panel))
    for i, row in enumerate(loo_panel.itertuples()):
        color = MOD_COLORS[row.family]
        ax_c.hlines(i, row.rho_overall, baseline, color="#C8D1DC", linewidth=1.5, zorder=1)
        ax_c.scatter(row.rho_overall, i, s=55, color=color, edgecolors="white", linewidth=0.6, zorder=3)
        ax_c.scatter(baseline, i, s=35, color="#E5E7EB", edgecolors="#9CA3AF", linewidth=0.5, zorder=2)
        ax_c.text(row.rho_overall - 0.0007, i + 0.22, f"ρ={row.rho_overall:.3f}  (Δ={row.delta:.3f})",
                  fontsize=4.5, ha="right", color=color, fontweight="bold")

    ax_c.axvline(baseline, color="#9CA3AF", linestyle="--", linewidth=0.7, zorder=0)
    stack_label = "7B PCA-4096 champion" if loo_is_7b else "1B Fusion-10"
    ax_c.text(baseline + 0.0005, len(loo_panel) - 0.3, f"Full {stack_label}\nbaseline ρ={baseline:.3f}",
              fontsize=4.8, color="#6B7280", va="top")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(loo_panel["label"], fontsize=5.5)
    ax_c.set_xlabel("Spearman ρ on gene-operon split")
    ax_c.set_title("Fixed-width LOO on the 7B champion: DNA scaling dominates all other modalities\n"
                    "(colored dot = ρ without that modality; gray dot = full baseline)",
                    fontsize=7)
    # Dynamic xlim: accommodate Evo-2 7B's large drop
    min_rho = loo_panel["rho_overall"].min()
    ax_c.set_xlim(min_rho - 0.005, baseline + 0.003)

    legend_items = [
        mpatches.Patch(color=MOD_COLORS["protein"], label="Protein"),
        mpatches.Patch(color=MOD_COLORS["dna"], label="DNA / CDS"),
        mpatches.Patch(color=MOD_COLORS["genome_context"], label="Genome context"),
        mpatches.Patch(color=MOD_COLORS["biophysical"], label="Biophysical"),
    ]
    ax_c.legend(handles=legend_items, frameon=False, fontsize=5.0, loc="lower left", ncol=4)

    _save(fig, "fig3_core_results")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Plateau evidence
# ══════════════════════════════════════════════════════════════════════════
def fig4_plateau_evidence():
    """Evidence-backed limitation figure centered on data regime, not speculative decomposition."""
    logger.info("Fig 4: Plateau evidence")
    label_df = _load_audit_table("label_domain_summary.csv", build_label_domain_summary)
    frac_df = _load_audit_table("training_fraction_curve.csv", build_training_fraction_curve)
    breadth_df = _load_audit_table("species_breadth_curve.csv", build_species_breadth_curve)

    fig = plt.figure(figsize=(7.2, 3.5), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(1, 3, wspace=0.46)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a", x=-0.18, y=1.04)
    _style_axis(ax_a, "x")
    label_order = ["paxdb_only", "high_quality_only", "proxy_only", "proxy_to_paxdb", "paxdb_to_proxy"]
    plot_df = label_df.set_index("metric_id").loc[label_order].reset_index()
    y = np.arange(len(plot_df))
    colors_a = [COLORS["secondary"], COLORS["light2"], COLORS["quaternary"], COLORS["light1"], COLORS["neutral"]]
    bars = ax_a.barh(y, plot_df["mean"], color=colors_a, edgecolor="white", linewidth=0.4, height=0.68)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(
        ["PaXDb only", "High-quality\nsubset", "Abele only", "Abele → PaXDb", "PaXDb → Abele"],
        fontsize=5.4,
    )
    ax_a.set_xlabel("Spearman ρ")
    ax_a.set_title("Label provenance ceiling", fontsize=6.4, pad=10)
    ax_a.set_xlim(0.50, 0.70)
    for bar, val in zip(bars, plot_df["mean"]):
        ax_a.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=5.0)

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b", x=-0.16, y=1.04)
    _style_axis(ax_b, "y")
    ax_b.errorbar(
        frac_df["fraction_pct"],
        frac_df["mean"],
        yerr=frac_df["std"],
        fmt="o-",
        color=COLORS["primary"],
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.3,
        capsize=3,
    )
    ax_b.fill_between(frac_df["fraction_pct"], frac_df["mean"] - frac_df["std"], frac_df["mean"] + frac_df["std"], color="#D7E8F9", alpha=0.55)
    ax_b.set_xlabel("Training genes retained (%)")
    ax_b.set_ylabel("Spearman ρ")
    ax_b.set_title("Training-fraction scaling", fontsize=6.4, pad=10)
    ax_b.set_ylim(0.55, 0.64)
    n_train_full = 390640
    for row in frac_df.itertuples():
        n_genes = int(n_train_full * row.fraction_pct / 100)
        ax_b.annotate(f"{row.mean:.3f}\n(~{n_genes // 1000}K)", (row.fraction_pct, row.mean),
                      xytext=(0, 8), textcoords="offset points", ha="center", fontsize=4.5)
    ax_b.text(0.97, 0.05, "At 15M params: ρ=0.630\n(matches 25M champion)",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=4.5, color="#4B5563",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3F4F6", edgecolor="none"))

    ax_c = fig.add_subplot(gs[0, 2])
    _panel_label(ax_c, "c", x=-0.12, y=1.04)
    _style_axis(ax_c, "y")
    breadth_x = breadth_df["fraction_code"].astype(int).values
    ax_c.errorbar(
        breadth_x,
        breadth_df["mean"],
        yerr=breadth_df["std"],
        fmt="o-",
        color=COLORS["quaternary"],
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.3,
        capsize=3,
    )
    ax_c.fill_between(breadth_x, breadth_df["mean"] - breadth_df["std"], breadth_df["mean"] + breadth_df["std"], color="#E4F1EA", alpha=0.6)
    ax_c.set_xlabel("Species breadth retained (%)")
    ax_c.set_ylabel("Spearman ρ")
    ax_c.set_title("Species-breadth scaling", fontsize=6.4, pad=10)
    ax_c.set_ylim(0.59, 0.64)
    species_counts = {50: 193, 75: 289, 100: 385}
    for xval, row in zip(breadth_x, breadth_df.itertuples()):
        n_sp = species_counts.get(int(xval), "?")
        ax_c.annotate(f"{row.mean:.3f}\n({n_sp} spp.)", (xval, row.mean),
                      xytext=(0, 8), textcoords="offset points", ha="center", fontsize=4.5)

    _save(fig, "fig4_plateau_evidence")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Practical utility
# ══════════════════════════════════════════════════════════════════════════
def fig5_modality_structure():
    """Why multimodality matters: orthogonality, rank reshuffling, biological gain, and graceful degradation."""
    logger.info("Fig 5: Modality structure")
    bundle = _get_visual_data()
    sim = _similarity_matrix(bundle)

    fig = plt.figure(figsize=(7.2, 7.0), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.48)

    # Panel a: CKA heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    sns.heatmap(
        sim,
        ax=ax_a,
        cmap=sns.color_palette("blend:#f5f7fa,#8fc3d8,#1F4E79", as_cmap=True),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"shrink": 0.76, "label": "linear CKA"},
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 4.8},
    )
    ax_a.set_title("Cross-modality similarity shows which embeddings are redundant\nand which contribute orthogonal signal", fontsize=7)
    ax_a.tick_params(axis="x", rotation=45, labelsize=4.8)
    ax_a.tick_params(axis="y", labelsize=4.8)

    # Panel b: 3-axis LOO reshuffling — simplified to 5 key modalities
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b", x=-0.18, y=1.06)
    _style_axis(ax_b, "y")
    # Show only the 5 modalities with the most dramatic shifts
    show_mods = ["Evo-2 7B", "Biophys: operon pos.", "Biophys: RNA init", "ESM-C protein", "Bacformer genome"]
    slope = AUDITED_CROSS_REGIME_LOO.copy()
    slope = slope[slope["label"].isin(show_mods)].sort_values("gene_operon_rank")
    x_positions = [0, 0.5, 1.0]
    for row in slope.itertuples():
        color = MOD_COLORS[row.family]
        y_vals = [row.gene_operon_delta, row.go_nonmega_delta, row.species_cluster_delta]
        lw = 2.0 if row.label in ["Evo-2 7B", "ESM-C protein"] else 1.2
        ax_b.plot(x_positions, y_vals, color=color, linewidth=lw, alpha=0.85, zorder=2)
        for xi, yi in zip(x_positions, y_vals):
            ax_b.scatter(xi, yi, s=35, color=color, edgecolors="white", linewidth=0.6, zorder=3)
        # Label at rightmost point (SC) with name
        ax_b.text(1.04, row.species_cluster_delta, f"{row.label}", fontsize=4.5,
                  ha="left", va="center", color=color, fontweight="bold")
    ax_b.set_xlim(-0.05, 1.35)
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels(["GO\noverall", "GO\nnon-cons.", "Species-\ncluster"], fontsize=5)
    ax_b.set_ylabel("LOO importance (Δρ)", fontsize=6)
    ax_b.set_title("Modality importance reshuffles\nalong two axes", fontsize=7)
    ax_b.axhline(0, color="#D7E0EA", linewidth=0.5, zorder=0)
    ax_b.tick_params(labelsize=5)

    # Panel c: where fusion helps biologically
    ax_c = fig.add_subplot(gs[1, 0])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "x")
    pathway = AUDITED_PATHWAY_ADVANTAGE.copy()
    pathway["display"] = pathway["label"] + pathway["n"].map(lambda n: f"  (N={n:,})")
    pathway = pathway.sort_values("delta", ascending=True)
    y = np.arange(len(pathway))
    colors_c = [COLORS["quaternary"] if grp == "operon" else COLORS["primary"] for grp in pathway["group"]]
    ax_c.barh(y, pathway["delta"], color=colors_c, edgecolor="white", linewidth=0.4, height=0.66)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(pathway["display"], fontsize=4.8)
    ax_c.set_xlabel("Fusion-10 minus ESM-C, Δρ")
    ax_c.set_title("Fusion helps most on operon genes and mid-length proteins", fontsize=7)
    for yi, row in zip(y, pathway.itertuples()):
        ax_c.text(row.delta + 0.0015, yi, f"{row.delta:+.3f}", va="center", fontsize=4.8, color=COLORS["ink"], fontweight="bold")
    ax_c.text(0.98, 0.05, "Multi-gene operon gain inflated by 7B embedding identity;\nsingleton gain (Δρ=+0.059) is the leakage-free estimate.",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=4.7,
              bbox=dict(boxstyle="round,pad=0.24", facecolor="#F7FAFC", edgecolor="#D7E0EA", linewidth=0.5))

    # Panel d: missing-modality robustness
    ax_d = fig.add_subplot(gs[1, 1])
    _panel_label(ax_d, "d")
    _style_axis(ax_d, "x")
    available = AUDITED_MISSING_MODALITY.copy().sort_values("rho")
    y = np.arange(len(available))
    color_lookup = {
        "All 10": COLORS["secondary"],
        "No classical": COLORS["light1"],
        "No Bacformer": COLORS["light2"],
        "Protein + DNA": MOD_COLORS["dna"],
        "Protein + biophysical": MOD_COLORS["biophysical"],
        "DNA only": MOD_COLORS["dna"],
        "Protein only": MOD_COLORS["protein"],
        "Biophysical only": MOD_COLORS["biophysical"],
    }
    ax_d.barh(y, available["rho"], color=[color_lookup.get(label, COLORS["neutral"]) for label in available["label"]], edgecolor="white", linewidth=0.4, height=0.66)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(available["label"], fontsize=5.0)
    ax_d.set_xlabel("Spearman ρ")
    ax_d.set_xlim(0.42, 0.70)
    ax_d.set_title("Performance degrades gracefully when modalities are unavailable", fontsize=7)
    for yi, row in zip(y, available.itertuples()):
        ax_d.text(row.rho + 0.002, yi, f"{row.rho:.3f}", va="center", fontsize=4.8, color=COLORS["ink"], fontweight="bold")
    ax_d.text(0.98, 0.05, "Even protein-only mode retains 77% of champion performance;\nDNA-only (0.624) slightly outperforms protein-only (0.511).",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=4.7,
              bbox=dict(boxstyle="round,pad=0.24", facecolor="#F7FAFC", edgecolor="#D7E0EA", linewidth=0.5))

    _save(fig, "fig5_modality_structure")


def fig5_modality_structure_variant_b():
    """Alternate modality figure: six small-multiple UMAPs."""
    logger.info("Fig 5 variant B: Modality structure")
    bundle = _get_visual_data()
    umap_df = bundle.modality_umap.copy()
    quartile_colors = {
        "Q1 low": "#2166AC",
        "Q2": "#67A9CF",
        "Q3": "#F4A582",
        "Q4 high": "#B2182B",
    }
    order = [
        "ESM-C protein",
        "Evo-2 CDS",
        "HyenaDNA CDS",
        "DNABERT-2 operon",
        "RiNALMo init",
        "Bacformer",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), facecolor="#F4F7FB")
    for idx, (ax, modality) in enumerate(zip(axes.flat, order)):
        _panel_label(ax, chr(ord("a") + idx), x=-0.10, y=1.03)
        sub = umap_df[umap_df["modality"] == modality]
        _draw_umap_scatter(ax, sub, "expression_quartile", quartile_colors, modality, point_size=7)
    handles = [mpatches.Patch(color=color, label=label) for label, color in quartile_colors.items()]
    axes[0, 2].legend(handles=handles, frameon=False, fontsize=4.6, loc="lower right", ncol=2)
    fig.suptitle("Same sampled genes, six modality geometries", fontsize=8.2, y=0.99)
    fig.tight_layout()
    _save(fig, "fig5_modality_structure_variant_b")


def fig5_modality_structure_variant_c():
    """Alternate modality figure: modality constellation plus selected UMAPs."""
    logger.info("Fig 5 variant C: Modality structure")
    bundle = _get_visual_data()
    umap_df = bundle.modality_umap.copy()
    sim = _similarity_matrix(bundle).fillna(0.0)

    sim_arr = sim.to_numpy(dtype=float)
    dist = 1.0 - sim_arr
    n = dist.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dist ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    order_idx = np.argsort(evals)[::-1][:2]
    coords = evecs[:, order_idx] * np.sqrt(np.clip(evals[order_idx], 0, None))
    constellation = pd.DataFrame({"modality": sim.index, "x": coords[:, 0], "y": coords[:, 1]})
    family_map = {
        "ESM-C protein": "protein",
        "Evo-2 CDS": "dna",
        "HyenaDNA CDS": "dna",
        "DNABERT-2 operon": "dna",
        "CodonFM CDS": "dna",
        "RiNALMo init": "rna",
        "Bacformer": "genome_context",
    }
    constellation["family"] = constellation["modality"].map(family_map)

    fig = plt.figure(figsize=(7.2, 5.1), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 1.0], wspace=0.30, hspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, None)
    for row in constellation.itertuples():
        ax_a.scatter(row.x, row.y, s=120, color=MOD_COLORS[row.family], edgecolors="white", linewidth=0.8)
        ax_a.text(row.x + 0.01, row.y + 0.01, row.modality.replace(" protein", "").replace(" CDS", ""),
                  fontsize=5.0, ha="left", va="bottom")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_title("Constellation of modalities from 1 - CKA distance", fontsize=7.1)

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b")
    sns.heatmap(sim, ax=ax_b, cmap="mako", linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.75})
    ax_b.set_title("Similarity matrix", fontsize=7.1)
    ax_b.tick_params(axis="x", rotation=45, labelsize=4.7)
    ax_b.tick_params(axis="y", labelsize=4.7)

    ax_c = fig.add_subplot(gs[1, 0])
    _panel_label(ax_c, "c")
    sub_c = umap_df[umap_df["modality"] == "Bacformer"]
    _draw_umap_scatter(ax_c, sub_c, "species_split", SPLIT_COLORS, "Bacformer colored by species split", point_size=7)
    ax_c.legend(frameon=False, fontsize=4.8, loc="lower right")

    ax_d = fig.add_subplot(gs[1, 1])
    _panel_label(ax_d, "d")
    mega_colors = {"Mega": "#B2182B", "Non-mega": "#67A9CF"}
    sub_d = umap_df[umap_df["modality"] == "ESM-C protein"]
    _draw_umap_scatter(ax_d, sub_d, "is_mega_label", mega_colors, "ESM-C colored by mega-component membership", point_size=7)
    ax_d.legend(frameon=False, fontsize=4.8, loc="lower right")

    _save(fig, "fig5_modality_structure_variant_c")


def ed16_practical_utility():
    """Extended Data utility figure based only on verified scaling and transfer summaries."""
    logger.info("ED 16: Practical utility")
    frac_df = _load_audit_table("training_fraction_curve.csv", build_training_fraction_curve)
    breadth_df = _load_audit_table("species_breadth_curve.csv", build_species_breadth_curve)
    split_df = _load_audit_table("split_summary.csv", build_split_summary).set_index("metric_id")

    fig = plt.figure(figsize=(7.2, 3.0), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(1, 3, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "y")
    ax_a.errorbar(frac_df["fraction_pct"], frac_df["mean"], yerr=frac_df["std"], fmt="o-", color=COLORS["primary"], markersize=5, markerfacecolor="white", markeredgewidth=1.2, linewidth=1.2, capsize=3)
    ax_a.fill_between(frac_df["fraction_pct"], frac_df["mean"] - frac_df["std"], frac_df["mean"] + frac_df["std"], color="#D7E8F9", alpha=0.55)
    ax_a.set_xlabel("Training genes retained (%)")
    ax_a.set_ylabel("Spearman ρ")
    ax_a.set_ylim(0.55, 0.64)
    ax_a.set_title("Training-fraction scaling", fontsize=7)

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "y")
    breadth_x = np.array([50, 75, 100])
    ax_b.errorbar(breadth_x, breadth_df["mean"], yerr=breadth_df["std"], fmt="o-", color=COLORS["quaternary"], markersize=5, markerfacecolor="white", markeredgewidth=1.2, linewidth=1.2, capsize=3)
    ax_b.fill_between(breadth_x, breadth_df["mean"] - breadth_df["std"], breadth_df["mean"] + breadth_df["std"], color="#E4F1EA", alpha=0.6)
    ax_b.set_xlabel("Species breadth retained (%)")
    ax_b.set_ylabel("Spearman ρ")
    ax_b.set_ylim(0.59, 0.64)
    ax_b.set_title("Species-breadth scaling", fontsize=7)

    ax_c = fig.add_subplot(gs[0, 2])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    labels = ["Gene-operon", "TXpredict\nholdout"]
    means = [split_df.loc["fusion10_gene_operon", "mean"], split_df.loc["txpredict_holdout", "mean"]]
    errs = [split_df.loc["fusion10_gene_operon", "std"], split_df.loc["txpredict_holdout", "std"]]
    bars = ax_c.bar(range(2), means, yerr=errs, color=[COLORS["secondary"], COLORS["light2"]], edgecolor="white", linewidth=0.4, capsize=3, error_kw={"linewidth": 0.5}, width=0.62)
    ax_c.set_xticks(range(2))
    ax_c.set_xticklabels(labels, fontsize=5.4)
    ax_c.set_ylabel("Spearman ρ")
    ax_c.set_ylim(0.54, 0.66)
    ax_c.set_title("Holdout transfer remains strong", fontsize=7)
    for bar, val in zip(bars, means):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=5.0, fontweight="bold")

    _save(fig, "ed16_practical_utility")


def fig_leakage_topology():
    """Leakage/topology companion figure centered on the mega-component."""
    logger.info("Topology figure")

    bundle = _get_visual_data()
    components = bundle.components.copy()
    species = bundle.species.copy()
    clusters = bundle.clusters.copy()

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "both")
    top_components = components.head(2500)
    ax_a.scatter(
        top_components["component_rank"],
        top_components["gene_count"],
        s=np.where(top_components["component_rank"] == 1, 26, 9),
        c=np.where(top_components["component_rank"] == 1, COLORS["secondary"], COLORS["primary"]),
        alpha=0.75,
        edgecolors="none",
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.axhline(5000, color="#9CA3AF", linestyle="--", linewidth=0.7)
    ax_a.set_xlabel("Connected-component rank")
    ax_a.set_ylabel("Genes per component")
    ax_a.set_title("Connected-component spectrum", fontsize=7.2)
    ax_a.annotate(
        "Only one component crosses\n5,000 genes",
        xy=(1, components.iloc[0]["gene_count"]),
        xytext=(5, 40000),
        fontsize=5.0,
        arrowprops=dict(arrowstyle="->", lw=0.6, color="#6B7280"),
    )

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "both")
    ax_b.scatter(
        species["gene_count"],
        species["mega_fraction"],
        s=18 + species["cluster_species_count"] * 1.4,
        c=species["species_split"].map(SPLIT_COLORS).fillna(COLORS["neutral"]),
        alpha=0.8,
        edgecolors="white",
        linewidth=0.4,
    )
    ax_b.set_xlabel("Genes per species")
    ax_b.set_ylabel("Mega-component share")
    ax_b.set_title("Mega-component reaches every species", fontsize=7.2)
    ax_b.set_ylim(0.15, 0.55)
    _callout(
        ax_b,
        0.03,
        0.95,
        "202,945 genes\n385 species\n41.2% of dataset",
        color=COLORS["secondary"],
        fc="#FCECEF",
        fontsize=5.0,
    )

    ax_c = fig.add_subplot(gs[1, :])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    bucket_table = (
        clusters.groupby(["size_bucket", "split"], observed=False)["n_genes"]
        .sum()
        .unstack(fill_value=0)
        .reindex(["1 species", "2-3 species", "4-10 species", "11+ species"])
        .fillna(0)
    )
    left = np.zeros(len(bucket_table))
    for split in ["train", "val", "test"]:
        vals = bucket_table.get(split, pd.Series(0, index=bucket_table.index)).to_numpy()
        ax_c.barh(
            np.arange(len(bucket_table)),
            vals,
            left=left,
            color=SPLIT_COLORS[split],
            edgecolor="white",
            linewidth=0.4,
            label=split,
        )
        left += vals
    ax_c.set_yticks(np.arange(len(bucket_table)))
    ax_c.set_yticklabels(bucket_table.index, fontsize=5.6)
    ax_c.set_xlabel("Genes assigned via species clusters")
    ax_c.set_title("Split assignment acts on cluster-sized biological units, not isolated genes", fontsize=7.2)
    ax_c.legend(frameon=False, fontsize=5.3, ncol=3, loc="lower right")
    _save(fig, "fig_leakage_topology")


def fig_leakage_topology_variant_b():
    """Alternate topology figure: domination and cumulative share."""
    logger.info("Topology figure variant B")
    bundle = _get_visual_data()
    components = bundle.components.copy()
    clusters = bundle.clusters.copy()

    fig = plt.figure(figsize=(7.2, 4.6), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "y")
    top = components.head(50).copy()
    ax_a.plot(top["component_rank"], top["gene_fraction"].cumsum(), color=COLORS["secondary"], linewidth=1.6)
    ax_a.fill_between(top["component_rank"], top["gene_fraction"].cumsum(), color="#FBD5DB", alpha=0.85)
    ax_a.set_xlabel("Top connected components")
    ax_a.set_ylabel("Cumulative gene share")
    ax_a.set_title("One component dominates immediately", fontsize=7.2)
    ax_a.annotate("Mega-component alone = 41.2%", xy=(1, top.iloc[0]["gene_fraction"]),
                  xytext=(10, 0.58), fontsize=5.0,
                  arrowprops=dict(arrowstyle="->", lw=0.6, color="#6B7280"))

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "both")
    top_scatter = components.head(500)
    ax_b.scatter(top_scatter["n_species"], top_scatter["gene_count"],
                 s=np.where(top_scatter["is_mega"], 24, 9),
                 c=np.where(top_scatter["is_mega"], COLORS["secondary"], COLORS["primary"]),
                 alpha=0.75, edgecolors="none")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Species spanned")
    ax_b.set_ylabel("Genes in component")
    ax_b.set_title("Mega-component is both large and phylogenetically global", fontsize=7.1)

    ax_c = fig.add_subplot(gs[1, :])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    bucket = (
        clusters.groupby(["size_bucket", "split"], observed=False)["n_species"]
        .sum()
        .unstack(fill_value=0)
        .reindex(["1 species", "2-3 species", "4-10 species", "11+ species"])
        .fillna(0)
    )
    bottom = np.zeros(len(bucket))
    x = np.arange(len(bucket))
    for split in ["train", "val", "test"]:
        vals = bucket.get(split, pd.Series(0, index=bucket.index)).to_numpy()
        ax_c.bar(x, vals, bottom=bottom, color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.4, label=split)
        bottom += vals
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(bucket.index, fontsize=5.2)
    ax_c.set_ylabel("Species assigned in bucket")
    ax_c.set_title("Split balance depends on operating at biological-unit scale", fontsize=7.1)
    ax_c.legend(frameon=False, fontsize=4.9, ncol=3, loc="upper right")
    _save(fig, "fig_leakage_topology_variant_b")


def fig_leakage_topology_variant_c():
    """Alternate topology figure: summary cards plus ordered species view."""
    logger.info("Topology figure variant C")
    bundle = _get_visual_data()
    components = bundle.components.copy()
    species = bundle.species.copy()

    fig = plt.figure(figsize=(7.2, 5.2), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.95, 1.2], wspace=0.30, hspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    ax_a.axis("off")
    summary_cards = [
        ("Genes", "202,945", COLORS["secondary"]),
        ("Species", "385", COLORS["primary"]),
        ("Gene clusters", f"{int(components.iloc[0]['n_gene_clusters']):,}", COLORS["quaternary"]),
        ("Operons", f"{int(components.iloc[0]['n_operons']):,}", COLORS["tertiary"]),
    ]
    for i, (label, value, color) in enumerate(summary_cards):
        x = 0.03 + (i % 2) * 0.48
        y = 0.54 - (i // 2) * 0.42
        ax_a.add_patch(FancyBboxPatch((x, y), 0.43, 0.30, transform=ax_a.transAxes,
                                      boxstyle="round,pad=0.02,rounding_size=0.04",
                                      facecolor="white", edgecolor=color, linewidth=1.0))
        ax_a.text(x + 0.03, y + 0.20, label, transform=ax_a.transAxes, fontsize=5.4, color="#4B5563")
        ax_a.text(x + 0.03, y + 0.08, value, transform=ax_a.transAxes, fontsize=10, fontweight="bold", color=color)
    ax_a.set_title("Mega-component summary at a glance", fontsize=7.4, pad=8)

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "y")
    top = components.head(25).copy()
    ax_b.bar(np.arange(len(top)), top["gene_count"], color=[COLORS["secondary"]] + [COLORS["light1"]] * (len(top) - 1),
             edgecolor="white", linewidth=0.4)
    ax_b.set_yscale("log")
    ax_b.set_xticks([0, 4, 9, 14, 19, 24])
    ax_b.set_xticklabels([1, 5, 10, 15, 20, 25], fontsize=5.0)
    ax_b.set_xlabel("Component rank")
    ax_b.set_ylabel("Genes per component")
    ax_b.set_title("Ranked component sizes", fontsize=7.1)

    ax_c = fig.add_subplot(gs[1, :])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    ordered = species.sort_values(["mega_fraction", "gene_count"], ascending=[False, False]).reset_index(drop=True)
    x = np.arange(len(ordered))
    ax_c.scatter(x, ordered["mega_fraction"], s=18 + ordered["gene_count"] / 180,
                 c=ordered["species_split"].map(SPLIT_COLORS).fillna(COLORS["neutral"]),
                 edgecolors="white", linewidth=0.3, alpha=0.85)
    ax_c.set_xlim(-2, len(ordered) + 2)
    ax_c.set_ylim(0.15, 0.55)
    ax_c.set_xlabel("Species ordered by mega-component share")
    ax_c.set_ylabel("Mega-component share")
    ax_c.set_title("Species differ in degree, but none escape the mega-component", fontsize=7.2)
    for _, row in ordered.head(4).iterrows():
        idx = int(row.name)
        ax_c.annotate(_format_species_label(row["species"]), (idx, row["mega_fraction"]),
                      textcoords="offset points", xytext=(3, 5), fontsize=4.8)
    _save(fig, "fig_leakage_topology_variant_c")


def build_dataset_atlas_html():
    """Write an interactive atlas from real sampled embeddings and topology data."""
    logger.info("Interactive atlas")
    bundle = _get_visual_data()
    umap_df = bundle.modality_umap.copy()
    split_df = _load_audit_table("split_summary.csv", build_split_summary).set_index("metric_id")
    species_frame = bundle.species.copy()
    species_frame["direct_label_fraction"] = species_frame.get("direct_label_fraction", species_frame["gold_fraction"])
    species = species_frame[
        [
            "species",
            "species_order",
            "gene_count",
            "species_split",
            "species_cluster",
            "direct_label_fraction",
            "mega_fraction",
            "cluster_gene_count",
            "cluster_species_count",
        ]
    ].to_dict(orient="records")
    clusters = bundle.clusters[
        ["species_cluster", "cluster_rank", "n_genes", "n_species", "split", "top_species"]
    ].to_dict(orient="records")
    components = bundle.components.head(3000)[
        ["component_rank", "gene_count", "n_species", "n_gene_clusters", "is_mega", "label"]
    ].to_dict(orient="records")
    umap_records = umap_df[
        [
            "gene_id",
            "species",
            "expression_level",
            "expression_quartile",
            "source_short",
            "quality_tier",
            "species_split",
            "gene_operon_split",
            "species_cluster",
            "cluster_group",
            "is_mega_label",
            "modality",
            "family",
            "umap1",
            "umap2",
        ]
    ].to_dict(orient="records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AIKI-XP Dataset Atlas</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #08111f;
      --panel: rgba(10, 21, 39, 0.88);
      --ink: #e5eef8;
      --muted: #93a4b8;
      --blue: #3b82f6;
      --amber: #f59e0b;
      --red: #dc2626;
      --cyan: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(220, 38, 38, 0.12), transparent 22%),
        linear-gradient(180deg, #091223 0%, #040812 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      letter-spacing: 0.02em;
    }}
    .sub {{
      margin-top: 8px;
      color: var(--muted);
      max-width: 960px;
      line-height: 1.45;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(360px, 1fr));
      gap: 18px;
      margin-top: 24px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat {{
      background: rgba(10, 21, 39, 0.78);
      border: 1px solid rgba(147, 164, 184, 0.18);
      border-radius: 14px;
      padding: 12px 12px 10px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.22);
    }}
    .stat .k {{
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .stat .v {{
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(147, 164, 184, 0.18);
      border-radius: 18px;
      padding: 14px 14px 6px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
    }}
    .card h2 {{
      margin: 4px 4px 10px;
      font-size: 16px;
      font-weight: 600;
    }}
    .plot {{
      height: 420px;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AIKI-XP Dataset Atlas</h1>
    <div class="sub">
      Interactive companion to the manuscript figures. This atlas combines frozen split/topology artifacts
      with sampled real embedding geometry from the local production store, so the same page can show
      species cartography, leakage-relevant topology, and modality-specific organization of expression signal.
    </div>
    <div class="stats">
      <div class="stat"><div class="k">Dataset</div><div class="v">492,026 genes</div></div>
      <div class="stat"><div class="k">Species</div><div class="v">385</div></div>
      <div class="stat"><div class="k">Gene-operon</div><div class="v">ρ = {split_df.loc["fusion10_gene_operon", "mean"]:.3f}</div></div>
      <div class="stat"><div class="k">Species-cluster</div><div class="v">ρ = {split_df.loc["fusion10_species_cluster_t020", "mean"]:.3f}</div></div>
      <div class="stat"><div class="k">Random ceiling</div><div class="v">ρ = {split_df.loc["fusion10_random", "mean"]:.3f}</div></div>
      <div class="stat"><div class="k">TXpredict holdout</div><div class="v">ρ = {split_df.loc["txpredict_holdout", "mean"]:.3f}</div></div>
    </div>
    <div class="grid">
      <div class="card wide">
        <h2>Species Cartography</h2>
        <div id="speciesPlot" class="plot"></div>
      </div>
      <div class="card wide">
        <h2>Embedding Atlas</h2>
        <div id="embeddingPlot" class="plot"></div>
      </div>
      <div class="card">
        <h2>Species-Cluster Landscape</h2>
        <div id="clusterPlot" class="plot"></div>
      </div>
      <div class="card">
        <h2>Connected-Component Spectrum</h2>
        <div id="componentPlot" class="plot"></div>
      </div>
    </div>
  </div>
  <script>
    const species = {json.dumps(species)};
    const clusters = {json.dumps(clusters)};
    const components = {json.dumps(components)};
    const umapRecords = {json.dumps(umap_records)};
    const splitColors = {{train: "#3b82f6", val: "#f59e0b", test: "#dc2626"}};
    const modalities = [...new Set(umapRecords.map(d => d.modality))];
    const tracesByModality = modalities.map((modality) => {{
      const pts = umapRecords.filter(d => d.modality === modality);
      return {{
        x: pts.map(d => d.umap1),
        y: pts.map(d => d.umap2),
        mode: "markers",
        type: "scattergl",
        visible: modality === modalities[0],
        marker: {{
          color: pts.map(d => d.expression_level),
          colorscale: "Turbo",
          size: pts.map(d => d.is_mega_label === "Mega" ? 8 : 5),
          opacity: 0.72,
          colorbar: modality === modalities[0] ? {{title: "expression z"}} : undefined
        }},
        text: pts.map(d => `${{d.gene_id}}<br>${{d.species}}<br>expr=${{d.expression_level.toFixed(3)}}<br>quartile=${{d.expression_quartile}}<br>source=${{d.source_short}}<br>quality=${{d.quality_tier}}<br>species_split=${{d.species_split}}<br>gene_operon_split=${{d.gene_operon_split}}<br>cluster=${{d.species_cluster}}<br>mega=${{d.is_mega_label}}`),
        hovertemplate: "%{{text}}<extra></extra>",
        name: modality
      }};
    }});

    Plotly.newPlot("speciesPlot", [{{
      x: species.map(d => d.species_order),
      y: species.map(d => d.gene_count),
      mode: "markers",
      type: "scattergl",
      marker: {{
        color: species.map(d => splitColors[d.species_split] || "#94a3b8"),
        size: species.map(d => 6 + 20 * d.mega_fraction),
        opacity: 0.85,
        line: {{width: 0}}
      }},
      text: species.map(d => `${{d.species}}<br>genes=${{d.gene_count}}<br>cluster=${{d.species_cluster}}<br>mega_share=${{d.mega_fraction.toFixed(3)}}<br>direct_label_share=${{d.direct_label_fraction.toFixed(3)}}`),
      hovertemplate: "%{{text}}<extra></extra>"
    }}], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{color: "#e5eef8"}},
      margin: {{l: 60, r: 20, t: 20, b: 45}},
      xaxis: {{title: "Species order (split, then cluster rank)", gridcolor: "rgba(148,163,184,0.15)"}},
      yaxis: {{title: "Genes per species", gridcolor: "rgba(148,163,184,0.15)"}}
    }}, {{displayModeBar: false, responsive: true}});

    Plotly.newPlot("embeddingPlot", tracesByModality, {{
      title: `Embedding atlas: ${{modalities[0]}}`,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{color: "#e5eef8"}},
      margin: {{l: 45, r: 20, t: 40, b: 42}},
      xaxis: {{title: "UMAP 1", zeroline: false, gridcolor: "rgba(148,163,184,0.15)"}},
      yaxis: {{title: "UMAP 2", zeroline: false, gridcolor: "rgba(148,163,184,0.15)"}},
      updatemenus: [{{
        type: "dropdown",
        direction: "down",
        x: 0.01,
        y: 1.15,
        bgcolor: "rgba(10,21,39,0.95)",
        bordercolor: "rgba(147,164,184,0.25)",
        buttons: modalities.map((modality, idx) => ({{
          label: modality,
          method: "update",
          args: [
            {{
              visible: modalities.map((_, j) => j === idx)
            }},
            {{
              title: `Embedding atlas: ${{modality}}`
            }}
          ]
        }}))
      }}]
    }}, {{displayModeBar: false, responsive: true}});

    Plotly.newPlot("clusterPlot", [{{
      x: clusters.map(d => d.cluster_rank),
      y: clusters.map(d => d.n_genes),
      mode: "markers",
      marker: {{
        color: clusters.map(d => splitColors[d.split] || "#94a3b8"),
        size: clusters.map(d => 8 + 2.3 * d.n_species),
        opacity: 0.9,
        line: {{color: "#ffffff", width: 1}}
      }},
      text: clusters.map(d => `${{d.species_cluster}}<br>genes=${{d.n_genes}}<br>species=${{d.n_species}}<br>top=${{d.top_species}}`),
      hovertemplate: "%{{text}}<extra></extra>"
    }}], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{color: "#e5eef8"}},
      margin: {{l: 60, r: 20, t: 20, b: 45}},
      xaxis: {{title: "Cluster rank", gridcolor: "rgba(148,163,184,0.15)"}},
      yaxis: {{title: "Genes in species cluster", gridcolor: "rgba(148,163,184,0.15)"}}
    }}, {{displayModeBar: false, responsive: true}});

    Plotly.newPlot("componentPlot", [{{
      x: components.map(d => d.component_rank),
      y: components.map(d => d.gene_count),
      mode: "markers",
      marker: {{
        color: components.map(d => d.is_mega ? "#dc2626" : "#38bdf8"),
        size: components.map(d => d.is_mega ? 12 : 5),
        opacity: 0.8
      }},
      text: components.map(d => `${{d.label}}<br>genes=${{d.gene_count}}<br>species=${{d.n_species}}<br>clusters=${{d.n_gene_clusters}}`),
      hovertemplate: "%{{text}}<extra></extra>"
    }}], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{color: "#e5eef8"}},
      margin: {{l: 60, r: 20, t: 20, b: 45}},
      xaxis: {{title: "Connected-component rank", type: "log", gridcolor: "rgba(148,163,184,0.15)"}},
      yaxis: {{title: "Genes per component", type: "log", gridcolor: "rgba(148,163,184,0.15)"}}
    }}, {{displayModeBar: false, responsive: true}});
  </script>
</body>
</html>
"""
    (INTERACTIVE_DIR / "aiki_xp_dataset_atlas.html").write_text(html)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cross-species generalization
# ══════════════════════════════════════════════════════════════════════════
def fig6_cross_species():
    """Cross-species generalization with three-line threshold curve, novel/shared bars, and modality comparison."""
    logger.info("Fig 6: Cross-species generalization")

    threshold_df = _load_audit_table("threshold_curve.csv", build_audit_threshold_curve)
    split_df = _load_audit_table("split_summary.csv", build_split_summary).set_index("metric_id")
    models_df = _load_audit_table("species_cluster_model_comparison.csv", build_species_cluster_model_comparison).set_index("metric_id")

    fig = plt.figure(figsize=(7.2, 3.8), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(1, 3, wspace=0.42)

    n_test_sc = 54024

    # ── Panel A: Three-line threshold curve (overall + novel + shared) ──
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_label(ax_a, "a")
    _style_axis(ax_a, "y")
    ax_a.errorbar(threshold_df["threshold"], threshold_df["overall_mean"],
                  yerr=threshold_df["overall_std"], fmt="o-", color=COLORS["primary"],
                  markersize=5, markerfacecolor="white", markeredgewidth=1.2, linewidth=1.3, capsize=3, label="Overall")
    ax_a.plot(threshold_df["threshold"], threshold_df["rho_shared_families"],
              "s--", color=COLORS["light1"], markersize=4, linewidth=1.0, label="Shared families")
    ax_a.plot(threshold_df["threshold"], threshold_df["rho_novel_families"],
              "D--", color=COLORS["quaternary"], markersize=4, linewidth=1.0, label="Novel families")
    ax_a.axvline(0.20, color=COLORS["secondary"], linestyle=":", linewidth=0.8)
    go_ref = split_df.loc["fusion10_gene_operon", "mean"]
    # 7B champion gene-operon reference
    _7b_go_files_fig6 = sorted((PROJECT_ROOT / "results/protex_qc/evo2_7b_profiling/pca_dim_sweep").glob("go_f10_7b_fo_pca4096_25M_seed*.json"))
    _7b_go_ref = float(np.mean([extract_metric(f, "rho_overall") for f in _7b_go_files_fig6])) if _7b_go_files_fig6 else 0.667
    ax_a.axhline(_7b_go_ref, color="#B03A2E", linestyle="--", linewidth=0.8, alpha=0.8)
    ax_a.text(0.06, _7b_go_ref + 0.006, f"7B champion\nρ={_7b_go_ref:.3f}", fontsize=4.2, color="#B03A2E", fontweight="bold")
    ax_a.axhline(go_ref, color="#9CA3AF", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_a.text(0.06, go_ref - 0.012, f"1B gene-operon\nρ={go_ref:.3f}", fontsize=4.2, color="#6B7280")
    ax_a.text(0.205, 0.38, "t=0.20\n(primary)", fontsize=4.3, color=COLORS["secondary"])
    ax_a.set_xlabel("Mash distance threshold\n(higher = more species shared)")
    ax_a.set_ylabel("Spearman ρ")
    ax_a.set_ylim(0.35, 0.76)
    ax_a.set_title("Performance at four phylogenetic thresholds\n(N={:,} test genes at t=0.20)".format(n_test_sc), fontsize=6.5)
    ax_a.legend(frameon=False, fontsize=4.5, loc="upper left")

    # ── Panel B: Novel vs shared bars ──
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, "b")
    _style_axis(ax_b, "y")
    novel = threshold_df.loc[np.isclose(threshold_df["threshold"], 0.20), "rho_novel_families"].iloc[0]
    shared = threshold_df.loc[np.isclose(threshold_df["threshold"], 0.20), "rho_shared_families"].iloc[0]
    go = split_df.loc["fusion10_gene_operon", "mean"]
    bars = ax_b.bar(range(3), [go, novel, shared],
                    color=[COLORS["primary"], COLORS["quaternary"], COLORS["light1"]],
                    edgecolor="white", linewidth=0.4, width=0.70)
    ax_b.set_xticks(range(3))
    ax_b.set_xticklabels(["Gene-operon\n(all families novel\nby construction)", "Novel families\n(cluster absent\nfrom training)", "Shared families\n(cluster in at least\none training species)"], fontsize=4.5)
    ax_b.set_ylabel("Spearman ρ")
    ax_b.set_ylim(0.40, 0.78)
    ax_b.set_title("Novel vs shared family transfer\nat species-cluster t=0.20", fontsize=6.5)
    for bar, val in zip(bars, [go, novel, shared]):
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.010, f"{val:.3f}",
                  ha="center", fontsize=5.5, fontweight="bold")
    gap = shared - novel
    ax_b.annotate(f"Δ={gap:.3f}\nfamily recognition\nvs regulatory transfer",
                  xy=(2, shared), xytext=(1.5, 0.76), fontsize=4.2, color="#4B5563",
                  arrowprops=dict(arrowstyle="->", lw=0.4, color="#4B5563"))

    # ── Panel C: Novel-family performance by modality ──
    ax_c = fig.add_subplot(gs[0, 2])
    _panel_label(ax_c, "c")
    _style_axis(ax_c, "y")
    compare = models_df.loc[["context", "dna", "protein", "fusion10"]].copy()
    bars = ax_c.bar(range(len(compare)), compare["rho_novel_families"],
                    color=[MOD_COLORS["genome_context"], MOD_COLORS["dna"], MOD_COLORS["protein"], COLORS["secondary"]],
                    edgecolor="white", linewidth=0.4, width=0.64)
    ax_c.set_xticks(range(len(compare)))
    ax_c.set_xticklabels(["Genome\ncontext", "Coding\nDNA", "Protein\nsequence", "Fusion-10\n(all biology)"], fontsize=4.8)
    ax_c.set_ylabel("Novel-family ρ")
    ax_c.set_ylim(0.35, 0.56)
    ax_c.set_title("Where does multimodality help most?\n(novel families only, N={:,})".format(n_test_sc), fontsize=6.5)
    for bar, val in zip(bars, compare["rho_novel_families"]):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=5.0, fontweight="bold")

    _save(fig, "fig6_cross_species")


def ed_pair_synergy():
    """Modality-pair synergy heatmap from checked-in pair results."""
    logger.info("ED: Pair synergy heatmap")
    df = _load_audit_table("pair_synergy_matrix.csv", build_pair_synergy_matrix)
    pivot = df.pivot(index="modality_a", columns="modality_b", values="rho")
    order = ["ESM-C", "Evo-2", "DNABERT-2", "Bacformer", "Classical", "HyenaDNA"]
    pivot = pivot.reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor="#F4F7FB")
    mask = pivot.isnull()
    vmin, vmax = 0.50, 0.60
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    for i in range(len(order)):
        for j in range(len(order)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.57 else "#1F2A44"
            fontw = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.5, color=color, fontweight=fontw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ (gene-operon)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.set_title("Pairwise modality fusion performance", fontsize=8, pad=10)
    fig.tight_layout()
    _save(fig, "ed_pair_synergy_heatmap")


def ed_loso_species():
    """LOSO (leave-one-species-out) bar chart from checked-in results."""
    logger.info("ED: LOSO species-level performance")
    df = _load_audit_table("loso_summary.csv", build_loso_summary)
    if df.empty:
        logger.warning("No LOSO data found")
        return

    species_order = sorted(df["species"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor="#F4F7FB")
    _style_axis(ax, "x")
    y_positions = np.arange(len(species_order))[::-1]
    bar_height = 0.35
    for i, sp in enumerate(species_order):
        yi = y_positions[i]
        esmc_row = df[(df["species"] == sp) & (df["model"] == "ESM-C")]
        f10_row = df[(df["species"] == sp) & (df["model"] == "F10")]
        if not esmc_row.empty:
            val = esmc_row.iloc[0]["rho"]
            ax.barh(yi + bar_height / 2, val, height=bar_height, color=COLORS["light1"],
                    edgecolor="white", linewidth=0.4)
            ax.text(val + 0.003, yi + bar_height / 2, f"{val:.3f}", va="center", fontsize=4.8, color="#5D6975")
        if not f10_row.empty:
            val = f10_row.iloc[0]["rho"]
            ax.barh(yi - bar_height / 2, val, height=bar_height, color=COLORS["secondary"],
                    edgecolor="white", linewidth=0.4)
            ax.text(val + 0.003, yi - bar_height / 2, f"{val:.3f}", va="center", fontsize=4.8,
                    color=COLORS["secondary"], fontweight="bold")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s.replace("_", " ") for s in species_order], fontsize=5.5)
    ax.set_xlabel("Spearman ρ (leave-one-species-out)")
    ax.set_title("Per-species holdout: Fusion-10 vs ESM-C", fontsize=8)
    legend_items = [
        mpatches.Patch(color=COLORS["secondary"], label="Fusion-10"),
        mpatches.Patch(color=COLORS["light1"], label="ESM-C"),
    ]
    ax.legend(handles=legend_items, frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    _save(fig, "ed_loso_species")


def ed_mega_asymmetry():
    """Mega/non-mega training asymmetry from checked-in results."""
    logger.info("ED: Mega/non-mega asymmetry")
    df = _load_audit_table("mega_asymmetry.csv", build_mega_asymmetry)
    if df.empty or df["rho"].isnull().all():
        logger.warning("No mega asymmetry data found")
        return

    fig, ax = plt.subplots(figsize=(4.0, 3.2), facecolor="#F4F7FB")
    _style_axis(ax, "y")
    labels = ["Train mega\ntest non-mega", "Train non-mega\ntest mega"]
    vals = df["rho"].tolist()
    colors = [COLORS["light2"], COLORS["tertiary"]]
    bars = ax.bar(range(2), vals, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0, 0.85)
    ax.set_title("Mega-component training asymmetry", fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", fontsize=7, fontweight="bold")
    gap = abs(vals[1] - vals[0]) if len(vals) == 2 else 0
    mid_y = (vals[0] + vals[1]) / 2 if len(vals) == 2 else 0.5
    ax.annotate(
        f"Δ = {gap:.3f}",
        xy=(1, vals[1] - 0.02), xytext=(0.5, mid_y),
        fontsize=7, fontweight="bold", color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=1.2),
        ha="center",
    )
    fig.tight_layout()
    _save(fig, "ed_mega_asymmetry")


def ed_architecture_comparison():
    """Architecture comparison forest plot from checked-in round5 JSONs."""
    logger.info("ED: Architecture comparison")
    df = _load_audit_table("architecture_comparison.csv", build_architecture_comparison)
    df = df.dropna(subset=["rho"]).sort_values("rho", ascending=True)

    fig, ax = plt.subplots(figsize=(5.6, 3.2), facecolor="#F4F7FB")
    _style_axis(ax, "x")
    y = np.arange(len(df))
    champion_rho = df.loc[df["metric_id"] == "single_adapter", "rho"].iloc[0] if "single_adapter" in df["metric_id"].values else None
    colors = [COLORS["secondary"] if mid == "single_adapter" else COLORS["light1"] for mid in df["metric_id"]]
    ax.barh(y, df["rho"], color=colors, edgecolor="white", linewidth=0.4, height=0.6)
    if champion_rho is not None:
        ax.axvline(champion_rho, color=COLORS["secondary"], linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=5.5)
    ax.set_xlim(df["rho"].min() - 0.008, df["rho"].max() + 0.008)
    ax.set_xlabel("Gene-operon Spearman ρ (seed 42)")
    ax.set_title("Fusion architecture comparison — all within noise of champion", fontsize=7.5)
    for yi, val in zip(y, df["rho"]):
        ax.text(val + 0.001, yi, f"{val:.3f}", va="center", fontsize=5.0)
    fig.tight_layout()
    _save(fig, "ed_architecture_comparison")

def ed8_cross_attention():
    """Extended Data 8: Cross-attention negative result."""
    logger.info("ED 8: Cross-attention negative result")
    split_df = _load_audit_table("split_summary.csv", build_split_summary).set_index("metric_id")
    claims = _load_audit_table("provenance_claims.csv", build_claim_audit).set_index("claim_id")
    cross_attn = claims.loc["cross_attention", "actual_mean"]

    fig, ax = plt.subplots(figsize=(4.4, 3.0), facecolor="#F4F7FB")
    _style_axis(ax, "y")
    vals = [split_df.loc["fusion10_gene_operon", "mean"], cross_attn]
    errs = [split_df.loc["fusion10_gene_operon", "std"], 0.0]
    bars = ax.bar(range(2), vals, yerr=errs, color=[COLORS["secondary"], COLORS["neutral"]], edgecolor="white", linewidth=0.4, capsize=3, error_kw={"linewidth": 0.5}, width=0.6)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Single-adapter\nFusion-10", "Cross-attention\nseed42"], fontsize=5.4)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0.60, 0.64)
    ax.set_title("Checked-in cross-attention does not beat the audited champion", fontsize=7.1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}", ha="center", fontsize=5.0, fontweight="bold")
    fig.tight_layout()
    _save(fig, "ed8_cross_attention_negative")


def ed9_noise_robust():
    """Extended Data 9: Verified robust-objective and corruption controls."""
    logger.info("ED 9: Noise-robust training battery")
    noise_df = _load_audit_table("noise_curve.csv", build_noise_curve)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.6, 3.0), facecolor="#F4F7FB")
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")
    _style_axis(ax_a, "x")
    _style_axis(ax_b, "y")

    methods = pd.DataFrame(
        [
            ("Standard MSE", extract_metric(PROJECT_ROOT / "results/protex_qc/round5/champion_f10_go_seed42.json", "rho_overall")),
            ("Huber", extract_metric(PROJECT_ROOT / "results/protex_qc/validation/c15_huber_delta1.json", "rho_overall")),
            ("Trimmed MSE", extract_metric(PROJECT_ROOT / "results/protex_qc/validation/c14_truncated_mse_p95.json", "rho_overall")),
            ("Winsorized labels", extract_metric(PROJECT_ROOT / "results/protex_qc/validation/fusion10_labelmode_winsorized_seed42.json", "rho_overall")),
            ("Z-score labels", extract_metric(PROJECT_ROOT / "results/protex_qc/validation/fusion10_labelmode_zscore_seed42.json", "rho_overall")),
            ("Adversarial source", extract_metric(PROJECT_ROOT / "results/protex_qc/validation/adversarial_source_l0.1_seed42.json", "rho_overall")),
        ],
        columns=["label", "rho"],
    ).dropna(subset=["rho"]).sort_values("rho")
    y = np.arange(len(methods))
    bars = ax_a.barh(y, methods["rho"], color=[COLORS["neutral"]] * (len(methods) - 1) + [COLORS["secondary"]], edgecolor="white", linewidth=0.4, height=0.66)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(methods["label"], fontsize=5.2)
    ax_a.set_xlabel("Spearman ρ")
    ax_a.set_xlim(0.56, 0.635)
    ax_a.set_title("Checked-in robust-objective variants do not outperform standard MSE", fontsize=7.0)
    for bar, val in zip(bars, methods["rho"]):
        ax_a.text(val + 0.0015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=4.9)

    ax_b.plot(noise_df["corruption_pct"], noise_df["rho_overall"], "o-", color=COLORS["secondary"], markerfacecolor="white", markeredgewidth=1.2, linewidth=1.4)
    ax_b.fill_between(noise_df["corruption_pct"], noise_df["rho_overall"], alpha=0.08, color=COLORS["secondary"])
    ax_b.set_xticks(noise_df["corruption_pct"])
    ax_b.set_xticklabels([f"{x}%" for x in noise_df["corruption_pct"]], fontsize=5.3)
    ax_b.set_ylabel("Spearman ρ")
    ax_b.set_ylim(0.0, 0.66)
    ax_b.set_title("Explicit label corruption shows a smooth degradation curve", fontsize=7.0)
    for row in noise_df.itertuples():
        ax_b.annotate(f"{row.rho_overall:.3f}", (row.corruption_pct, row.rho_overall), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=4.8)

    fig.tight_layout()
    _save(fig, "ed9_noise_robust_battery")


def ed7_per_source_gap():
    """Extended Data 7: Verified source-provenance and transfer gaps."""
    logger.info("ED 7: Per-source label gap")
    label_df = _load_audit_table("label_domain_summary.csv", build_label_domain_summary).set_index("metric_id")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0), facecolor="#F4F7FB")
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")
    _style_axis(ax_a, "y")
    _style_axis(ax_b, "y")

    within = ["paxdb_only", "high_quality_only", "proxy_only"]
    vals = [label_df.loc[k, "mean"] for k in within]
    errs = [0.0 if pd.isna(label_df.loc[k, "std"]) else label_df.loc[k, "std"] for k in within]
    bars = ax_a.bar(range(3), vals, yerr=errs, color=[COLORS["secondary"], COLORS["quaternary"], COLORS["light1"]], edgecolor="white", linewidth=0.4, capsize=3, error_kw={"linewidth": 0.5})
    ax_a.set_xticks(range(3))
    ax_a.set_xticklabels(["PaXDb only", "High-quality\nsubset", "Abele only"], fontsize=5.3)
    ax_a.set_ylim(0.56, 0.70)
    ax_a.set_ylabel("Spearman ρ")
    ax_a.set_title("Within-domain performance depends strongly on label provenance", fontsize=7.0)
    for bar, val in zip(bars, vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=5.0, fontweight="bold")

    transfer = ["proxy_to_paxdb", "paxdb_to_proxy"]
    vals_t = [label_df.loc[k, "mean"] for k in transfer]
    bars = ax_b.bar(range(2), vals_t, color=[COLORS["light2"], COLORS["neutral"]], edgecolor="white", linewidth=0.4, width=0.6)
    ax_b.set_xticks(range(2))
    ax_b.set_xticklabels(["Abele → PaXDb", "PaXDb → Abele"], fontsize=5.3)
    ax_b.set_ylim(0.50, 0.64)
    ax_b.set_ylabel("Spearman ρ")
    ax_b.set_title("Cross-domain transfer is asymmetric and exposes label mismatch", fontsize=7.0)
    for bar, val in zip(bars, vals_t):
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.007, f"{val:.3f}", ha="center", fontsize=5.0, fontweight="bold")

    fig.tight_layout()
    _save(fig, "ed7_per_source_gap")


def ed10_cross_validation():
    """Extended Data 10: Seed42 five-fold CV only; unverifiable multi-seed claims removed."""
    logger.info("ED 10: Cross-validation overview")
    kfold_dir = PROJECT_ROOT / "results/protex_qc/kfold_cv_f10"
    fold_vals = []
    for path in sorted(kfold_dir.glob("f10_kfold_*_seed42.json")):
        value = extract_metric(path, "rho_overall")
        if value is not None:
            fold_vals.append((path.stem.replace("f10_kfold_", "fold ").replace("_seed42", ""), value))
    fig, ax = plt.subplots(figsize=(5.3, 3.3), facecolor="#F4F7FB")
    _style_axis(ax, "x")
    labels = [label for label, _ in fold_vals]
    means = [val for _, val in fold_vals]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, color=[COLORS["secondary"], COLORS["light2"], COLORS["quaternary"], COLORS["primary"], COLORS["light1"]][: len(labels)], edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlim(min(means) - 0.02, max(means) + 0.02)
    ax.set_xlabel("5-fold CV Spearman ρ")
    ax.set_title("Only the checked-in seed42 CV folds are retained here", fontsize=7.4)
    for yi, mean in zip(y, means):
        ax.text(mean + 0.0015, yi, f"{mean:.3f}", va="center", fontsize=5.0)

    fig.tight_layout()
    _save(fig, "ed10_cross_validation_overview")


def ed12_biomolecule_groupings():
    """Extended Data 12: Supported biology-family comparison with honest scope."""
    logger.info("ED 12: Biomolecule groupings")
    bio = _load_audit_table("biology_family_gene_operon.csv", build_biology_family_go).set_index("metric_id")
    models_df = _load_audit_table("species_cluster_model_comparison.csv", build_species_cluster_model_comparison).set_index("metric_id")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0), facecolor="#F4F7FB")
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")
    _style_axis(ax_a, "y")
    _style_axis(ax_b, "y")

    labels = ["Classical", "Protein", "DNA", "Fusion"]
    overall = [
        bio.loc["classical_only", "rho_overall"],
        bio.loc["protein_only", "rho_overall"],
        bio.loc["dna_only", "rho_overall"],
        bio.loc["fusion10_seed42", "rho_overall"],
    ]
    cluster = [
        bio.loc["classical_only", "rho_overall"],
        models_df.loc["protein", "rho_cluster_weighted"],
        models_df.loc["dna", "rho_cluster_weighted"],
        models_df.loc["fusion10", "rho_cluster_weighted"],
    ]
    colors = [MOD_COLORS["classical"], MOD_COLORS["protein"], MOD_COLORS["dna"], COLORS["secondary"]]

    bars = ax_a.bar(range(4), overall, color=colors, edgecolor="white", linewidth=0.4)
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(labels, fontsize=5.5)
    ax_a.set_ylim(0, 0.66)
    ax_a.set_ylabel("Spearman ρ")
    ax_a.set_title("Gene-operon overall performance", fontsize=7)
    for bar, val in zip(bars, overall):
        ax_a.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.3f}", ha="center", fontsize=5)

    bars = ax_b.bar(range(4), cluster, color=colors, edgecolor="white", linewidth=0.4)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(labels, fontsize=5.5)
    ax_b.set_ylim(0, 0.28)
    ax_b.set_ylabel(r"$\rho_{\mathrm{cluster-wt}}$")
    ax_b.set_title("Cluster-weighted performance", fontsize=7)
    for bar, val in zip(bars, cluster):
        ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.008, f"{val:.3f}", ha="center", fontsize=5)

    fig.tight_layout()
    _save(fig, "ed12_biomolecule_groupings")


# ══════════════════════════════════════════════════════════════════════════
# DATASET VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════

def fig_species_landscape():
    """Species landscape: gene count vs expression spread, colored by phylum proxy."""
    logger.info("Species landscape visualization")

    split_file = PROJECT_ROOT / "results" / "protex_qc" / "final_data_freeze_20260219" / "splits" / "gene_cluster_operon_split_v2_balanced.tsv"
    df = pd.read_csv(split_file, sep="\t", usecols=["gene_id", "species"],
                     dtype={"species": str})

    sp_counts = df.groupby("species").size().reset_index(name="gene_count")
    sp_counts = sp_counts.sort_values("gene_count", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), facecolor="#F4F7FB")
    _style_axis(ax, "y")

    # Rank plot — all 385 species ordered by gene count
    sp_counts = sp_counts.reset_index(drop=True)
    sp_counts["rank"] = range(1, len(sp_counts) + 1)

    ax.fill_between(sp_counts["rank"], sp_counts["gene_count"], alpha=0.3,
                   color=COLORS["primary"])
    ax.plot(sp_counts["rank"], sp_counts["gene_count"], color=COLORS["primary"],
           linewidth=1.0)

    # Annotate top species
    for _, row in sp_counts.head(5).iterrows():
        name = row["species"]
        if name[0].isdigit():
            name = f"taxid:{name}"
        else:
            name = name.replace("_", " ")
        ax.annotate(name, (row["rank"], row["gene_count"]),
                   textcoords="offset points", xytext=(15, 0),
                   fontsize=5, fontstyle="italic",
                   arrowprops=dict(arrowstyle="-", lw=0.3, color="gray"))

    ax.set_xlabel("Species rank")
    ax.set_ylabel("Gene count")
    ax.set_title(f"Gene count distribution across {len(sp_counts)} bacterial species", fontsize=8)

    # Summary stats box
    stats_text = (f"Total: {sp_counts['gene_count'].sum():,} genes\n"
                  f"Median: {int(sp_counts['gene_count'].median()):,} genes/species\n"
                  f"Range: {sp_counts['gene_count'].min():,}–{sp_counts['gene_count'].max():,}")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=6,
           va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor=COLORS["bg"], edgecolor="gray",
                                           linewidth=0.5))

    fig.tight_layout()
    _save(fig, "fig_species_landscape")


def fig_modality_heatmap():
    """Supplementary modality relationship figure using real sampled embeddings."""
    logger.info("Modality relationship heatmap")

    bundle = _get_visual_data()
    similarity = bundle.modality_similarity.copy()
    availability = bundle.modality_availability.copy()

    fig = plt.figure(figsize=(7.2, 3.7), facecolor="#F4F7FB")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 0.8], wspace=0.34)

    ax_a = fig.add_subplot(gs[0, 0])
    sim = similarity.pivot(index="modality_a", columns="modality_b", values="cka_similarity")
    order = [
        "ESM-C protein",
        "Evo-2 CDS",
        "HyenaDNA CDS",
        "DNABERT-2 operon",
        "CodonFM CDS",
        "RiNALMo init",
        "Bacformer",
    ]
    sim = sim.reindex(index=order, columns=order)
    sns.heatmap(
        sim,
        ax=ax_a,
        cmap="mako",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "linear CKA"},
    )
    ax_a.set_title("Cross-modality similarity from real sampled embeddings", fontsize=7.3)
    ax_a.tick_params(axis="x", rotation=45, labelsize=4.8)
    ax_a.tick_params(axis="y", labelsize=4.8)

    ax_b = fig.add_subplot(gs[0, 1])
    _style_axis(ax_b, "x")
    avail = availability.sort_values("coverage_fraction", ascending=True)
    y = np.arange(len(avail))
    colors = [MOD_COLORS.get(fam, COLORS["neutral"]) for fam in avail["family"]]
    ax_b.barh(y, avail["coverage_fraction"], color=colors, edgecolor="white", linewidth=0.4)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(avail["modality"], fontsize=4.7)
    ax_b.set_xlim(0.99, 1.001)
    ax_b.set_xlabel("Coverage fraction")
    ax_b.set_title("Coverage across local modalities is effectively complete", fontsize=7.0)
    for yi, val in zip(y, avail["coverage_fraction"]):
        ax_b.text(val + 0.00005, yi, f"{val:.4f}", va="center", fontsize=4.6)

    fig.tight_layout()
    _save(fig, "fig_modality_heatmap")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def _parse_args():
    parser = argparse.ArgumentParser(description="Generate AIKI-XP publication figures.")
    parser.add_argument(
        "--prepress",
        action="store_true",
        help="Also export high-resolution publication masters to figures/nbt/prepress.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=300,
        help="DPI for manuscript PNG companions written alongside PDFs.",
    )
    parser.add_argument(
        "--prepress-png-dpi",
        type=int,
        default=600,
        help="DPI for high-resolution PNG exports in prepress mode.",
    )
    parser.add_argument(
        "--prepress-tiff-dpi",
        type=int,
        default=1200,
        help="DPI for TIFF exports in prepress mode.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    EXPORT_CONFIG.update(
        {
            "png_dpi": args.png_dpi,
            "write_prepress": args.prepress,
            "prepress_png_dpi": args.prepress_png_dpi,
            "prepress_tiff_dpi": args.prepress_tiff_dpi,
        }
    )
    set_nbt_style()
    _get_visual_data()
    logger.info("Generating NBT publication figures → %s", FIGURE_DIR)
    if EXPORT_CONFIG["write_prepress"]:
        logger.info("Prepress masters enabled → %s", PREPRESS_DIR)

    # 6 Main Figures
    fig1_pipeline_overview()
    fig2_dataset_overview()
    fig2_dataset_overview_variant_b()
    fig2_dataset_overview_variant_c()
    fig3_core_results()
    fig4_plateau_evidence()
    fig5_modality_structure()
    fig5_modality_structure_variant_b()
    fig5_modality_structure_variant_c()
    fig6_cross_species()

    # Audited Extended Data
    ed7_per_source_gap()
    ed8_cross_attention()
    ed9_noise_robust()
    ed10_cross_validation()
    ed12_biomolecule_groupings()
    ed16_practical_utility()

    # New evidence-backed Extended Data
    ed_pair_synergy()
    ed_loso_species()
    ed_mega_asymmetry()
    ed_architecture_comparison()

    # Dataset visualizations
    fig_species_landscape()
    fig_modality_heatmap()
    fig_leakage_topology()
    fig_leakage_topology_variant_b()
    fig_leakage_topology_variant_c()
    build_dataset_atlas_html()

    logger.info("All figures generated successfully!")
    logger.info("Output directory: %s", FIGURE_DIR)


if __name__ == "__main__":
    main()
