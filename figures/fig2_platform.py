#!/usr/bin/env python3
"""
Fig 2 v5: Aiki-XP platform — biological scales + fusion architecture.

Redesign with:
  - Panel a: Clean nested rectangles with biological scale hierarchy
  - Panel b: Architecture diagram with G/O/C/P/B abbreviations
  - Panel c: Fused UMAP of 492K genes colored by expression + ρ_nc overlay

Usage:
    python scripts/protex/generate_fig2_v5_platform.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "protex" / "latex" / "figures" / "nbt"

# Colors
GENOME_COL = "#1B4332"     # dark green
OPERON_COL = "#2D6A4F"     # green
CDS_COL = "#40916C"        # medium green
PROTEIN_COL = "#74C69D"    # light green
BIOPHYS_COL = "#E76F51"    # warm coral
FUSION_COL = "#264653"     # dark teal
BG = "#FFFFFF"
INK = "#1A1A2E"
LIGHT_BG = "#F8F9FA"


def generate_fig2():
    log.info("Generating Fig 2 v5 platform...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.5,
    })

    fig = plt.figure(figsize=(7.2, 5.4), facecolor=BG)

    # ── Panel a: Biological operon diagram (v3d style) ────────────────
    ax_a = fig.add_axes([0.02, 0.42, 0.55, 0.53])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.text(0.0, 1.02, "a", fontsize=14, fontweight="bold", color=INK)
    ax_a.text(0.04, 1.02, "What regulates bacterial protein expression?",
              fontsize=8, fontweight="bold", color=INK)

    # Chromosome backbone
    cy = 0.62
    ax_a.plot([0.03, 0.97], [cy, cy], color="#CFD8DC", lw=5, zorder=0, solid_capstyle="round")

    # Operon bracket — extended LEFT to include promoter (biologically the
    # promoter is inside the operon unit, upstream of the first gene).
    ax_a.plot([0.04, 0.85], [cy + 0.13, cy + 0.13], color="#E65100", lw=1.5)
    ax_a.plot([0.04, 0.04], [cy + 0.13, cy + 0.11], color="#E65100", lw=1.5)
    ax_a.plot([0.85, 0.85], [cy + 0.13, cy + 0.11], color="#E65100", lw=1.5)
    ax_a.text(0.445, cy + 0.15, "operon (polycistronic transcription unit)",
              fontsize=5.5, color="#E65100", ha="center", fontweight="bold")

    # Genes in the operon (rounded rectangles with arrows)
    genes = [(0.10, 0.10), (0.22, 0.14), (0.38, 0.22), (0.62, 0.10), (0.74, 0.10)]
    for i, (gx, gw) in enumerate(genes):
        is_target = (i == 2)
        fc = "#E0F2F1" if is_target else "#ECEFF1"
        ec = "#00897B" if is_target else "#B0BEC5"
        rect = FancyBboxPatch((gx, cy - 0.04), gw, 0.08,
                              boxstyle="round,pad=0.003,rounding_size=0.010",
                              facecolor=fc, edgecolor=ec, linewidth=1.5 if is_target else 0.6)
        ax_a.add_patch(rect)
        if is_target:
            ax_a.annotate("", xy=(gx + gw - 0.02, cy), xytext=(gx + 0.02, cy),
                         arrowprops=dict(arrowstyle="-|>", color="#00897B", lw=1.5, mutation_scale=10))
            ax_a.text(gx + gw/2, cy + 0.065, "target gene",
                     fontsize=6, color="#00897B", ha="center", fontweight="bold")

    # Biological signals with annotation lines
    # Target gene = gene 2 at x ∈ [0.38, 0.60]. Adjustments 2026-04-13:
    #   - 5' UTR: x=0.37 → intergenic region immediately upstream of target
    #     (previously x=0.35 landed inside gene 1, the left neighbour).
    #   - Protein: x=0.54 → inside the target gene (previously x=0.63 landed
    #     inside gene 3, the right neighbour).
    #   - Promoter: x=0.08 → now inside the extended operon bracket.
    signals = [
        (0.08, "Promoter", "σ-factor binding", "#AD1457", -0.18),
        (0.37, "5' UTR", "mRNA folding,\nSD accessibility", "#7B1FA2", -0.30),
        (0.49, "Codon usage", "tRNA availability", "#2E7D32", -0.18),
        (0.54, "Protein", "folding, degradation", "#1565C0", -0.30),
        (0.43, "Operon\nposition", "gene order,\nco-expression", "#607D8B", -0.42),
        (0.90, "Genome context", "chromosomal\nneighborhood", "#C62828", -0.30),
    ]
    for sx, label, desc, color, yo in signals:
        ax_a.plot([sx, sx], [cy - 0.06, cy + yo + 0.06], color=color, lw=0.6, ls="--", alpha=0.5)
        ax_a.text(sx, cy + yo + 0.02, label, fontsize=5.5, color=color, ha="center", fontweight="bold")
        ax_a.text(sx, cy + yo - 0.04, desc, fontsize=3.8, color="#78909C", ha="center", linespacing=1.2)

    # Floating tagline removed — redundant with main caption; also caused visual overlap
    # with the "Operon position" annotation when placed inside panel a's bounds.

    # ── Panel b: Architecture (right 40%) ─────────────────────────────
    ax_b = fig.add_axes([0.60, 0.42, 0.38, 0.53])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis("off")
    ax_b.text(-0.05, 1.05, "b", transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")

    # Five input modalities as colored circles with abbreviations
    modalities = [
        ("G", GENOME_COL, "Genome\n960d"),
        ("O", OPERON_COL, "Operon\n4,096d"),
        ("C", CDS_COL, "CDS\n256d"),
        ("P", PROTEIN_COL, "Protein\n1,152d"),
        ("B", BIOPHYS_COL, "Biophys.\n69d"),
    ]

    y_top = 9.0
    x_positions = np.linspace(1, 9, 5)

    for i, (abbr, color, desc) in enumerate(modalities):
        x = x_positions[i]
        circle = plt.Circle((x, y_top), 0.7, facecolor=color, alpha=0.2,
                            edgecolor=color, linewidth=1.5)
        ax_b.add_patch(circle)
        ax_b.text(x, y_top, abbr, fontsize=11, fontweight="bold",
                  ha="center", va="center", color=color)
        ax_b.text(x, y_top - 1.1, desc, fontsize=4.5, ha="center", va="top", color=INK)

    # Pyramid adapters → arrows down
    for x in x_positions:
        ax_b.annotate("", xy=(x, 6.5), xytext=(x, 7.5),
                      arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    # Adapter box
    adapter_rect = FancyBboxPatch((0.5, 5.8), 9.0, 1.0, boxstyle="round,pad=0.15",
                                   facecolor="#E8E8E8", alpha=0.5, edgecolor="#888", linewidth=1.0)
    ax_b.add_patch(adapter_rect)
    ax_b.text(5.0, 6.3, "Per-modality pyramid MLPs → 1,262d latent per modality",
              fontsize=5.5, ha="center", va="center", color=INK)

    # Arrow down to fusion
    ax_b.annotate("", xy=(5.0, 4.3), xytext=(5.0, 5.6),
                  arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))

    # Fusion MLP box
    fusion_rect = FancyBboxPatch((2.0, 3.3), 6.0, 1.2, boxstyle="round,pad=0.15",
                                  facecolor=FUSION_COL, alpha=0.15, edgecolor=FUSION_COL, linewidth=1.5)
    ax_b.add_patch(fusion_rect)
    ax_b.text(5.0, 4.05, "Fusion MLP (~25M params)", fontsize=7, fontweight="bold",
              ha="center", va="center", color=FUSION_COL)
    ax_b.text(5.0, 3.65, "9 heads × 1,262d = 11,358d → 1,262d → 631d → 1", fontsize=4.5,
              ha="center", va="center", color=FUSION_COL, alpha=0.7)

    # Arrow down to prediction
    ax_b.annotate("", xy=(5.0, 2.0), xytext=(5.0, 3.1),
                  arrowprops=dict(arrowstyle="->", color=FUSION_COL, lw=1.5))

    # Prediction output
    ax_b.text(5.0, 1.5, "Per-species z-scored\nprotein abundance",
              fontsize=6.5, ha="center", va="center", color=FUSION_COL, fontweight="bold")

    ax_b.text(5.0, 10.4, "Fusion architecture", fontsize=8, fontweight="bold",
              ha="center", color=INK)

    # ── Panel c: Fused UMAP colored by expression (bottom-left) ───────
    ROOT = Path(__file__).resolve().parents[2]
    ATLAS = ROOT / "results" / "protex_qc" / "heavyweight_visual_data" / "manifold_atlas_492k.parquet"
    PROD = ROOT / "datasets" / "protex_aggregated" / "protex_aggregated_v1.1_final_freeze.parquet"

    ax_c = fig.add_axes([0.06, 0.04, 0.40, 0.32])
    ax_c.text(-0.08, 1.10, "c", transform=ax_c.transAxes, fontsize=14, fontweight="bold", va="top")

    atlas = pd.read_parquet(ATLAS, columns=["gene_id", "fused_umap2_x", "fused_umap2_y"])
    meta = pd.read_parquet(PROD, columns=["gene_id", "expression_level"])
    df = atlas.merge(meta, on="gene_id", how="left")

    x_u = df["fused_umap2_x"].values
    y_u = df["fused_umap2_y"].values
    c_u = df["expression_level"].values

    sc = ax_c.scatter(x_u, y_u, c=c_u, cmap="RdYlBu_r", s=0.03, alpha=0.20,
                      vmin=-3, vmax=3, rasterized=True)
    cb = plt.colorbar(sc, ax=ax_c, shrink=0.6, pad=0.02, aspect=15)
    cb.set_label("Expression (z-scored)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_title("Fused representation: 492,026 genes", fontsize=7, fontweight="bold", pad=3)

    # Overlay ρ_nc in a semi-transparent box
    ax_c.text(0.98, 0.95,
              r"$\rho_\mathrm{nc} = 0.592 \pm 0.011$" + "\n(pooled 2-seed 5-fold CV)",
              transform=ax_c.transAxes, fontsize=6.5, fontweight="bold",
              va="top", ha="right", color=FUSION_COL,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C8D1DC", alpha=0.85))

    # ── Panel d: Monotone deployment ladder (bottom-right) ────────────
    # Central proof of the "more biological context → better prediction"
    # claim. Each tier's input is a strict superset of the previous one;
    # every step both adds a distinct class of biological input and yields
    # a statistically significant ρ_nc gain.
    ax_d = fig.add_axes([0.56, 0.04, 0.42, 0.32])
    ax_d.text(-0.10, 1.10, "d", transform=ax_d.transAxes, fontsize=14, fontweight="bold", va="top")

    tier_data = [
        ("Tier A\nprotein only", 0.518, "#1f77b4"),
        ("Tier B\n+ coding DNA", 0.530, "#9467bd"),
        ("Tier B$^{+}$\n+ 5$^{\\prime}$ UTR", 0.543, "#5dade2"),
        ("Tier C\n+ full operon", 0.576, "#17becf"),
        ("Tier D\n+ host genome", 0.592, "#B03A2E"),
    ]
    y_t = np.arange(len(tier_data))
    for i, (label, rho, color) in enumerate(tier_data):
        ax_d.barh(i, rho, color=color, edgecolor="white", linewidth=0.5,
                  height=0.58, alpha=0.88)
        ax_d.text(rho + 0.003, i, f"{rho:.3f}", va="center", fontsize=6,
                  color=INK, fontweight="bold")

    ax_d.set_yticks(y_t)
    ax_d.set_yticklabels([t[0] for t in tier_data], fontsize=5.5)
    ax_d.set_xlabel(r"$\rho_\mathrm{nc}$ (non-conserved)", fontsize=6.5)
    ax_d.tick_params(axis="x", labelsize=5.5)
    ax_d.set_xlim(0.42, 0.62)
    ax_d.set_title("Monotone deployment ladder:\nmore biological context $\\rightarrow$ better prediction",
                   fontsize=7, fontweight="bold", pad=3)
    for spine in ("top", "right"):
        ax_d.spines[spine].set_visible(False)

    # Δ arrow A → D
    ax_d.annotate("", xy=(0.592, 4.0), xytext=(0.518, 0.0),
                  arrowprops=dict(arrowstyle="->", color="#888", lw=1, ls="--"))
    ax_d.text(0.435, 2.3, r"$\Delta$=+0.074" + "\n(+14%)",
              fontsize=5.5, ha="left", color="#555", style="italic",
              bbox=dict(boxstyle="round,pad=0.22", fc=BG, ec="#888",
                        lw=0.5, alpha=0.95))

    # Save
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"fig2_v5_platform.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
        log.info(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    generate_fig2()
