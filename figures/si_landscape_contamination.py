#!/usr/bin/env python3
"""Generate publication-quality SI figure for the landscape analysis.

Panel A: contamination bar chart (benchmarks + published training sets)
Panel B: ρ_full vs ρ_clean paired bars (tools × benchmarks)
Panel C: Δρ forest plot with bootstrap CIs

Reads from:
    results/landscape_2026/contamination/contamination_matrix.csv
    results/landscape_2026/landscape_summary.csv

Outputs:
    docs/protex/latex/figures/nbt/si_landscape_contamination.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTAM_CSV = PROJECT_ROOT / "results" / "landscape_2026" / "contamination" / "contamination_matrix.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "landscape_2026" / "landscape_summary.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "protex" / "latex" / "figures" / "nbt"

TOOL_DISPLAY = {
    "netsolp_usability": "NetSolP\nusability",
    "netsolp_solubility": "NetSolP\nsolubility",
    "mpepe": "MPEPE",
    "xp5_protein_only": "XP5\nprotein-only",
    "xp5_champion_f10": "XP5 F10\n(multimodal)",
    "xp5_tier_a_esmc_prott5": "XP5 Tier A\n(ESM-C+ProtT5)",
}

BENCH_DISPLAY = {
    "boel_2016": "Boël 2016",
    "price_nesg_2011": "Price/NESG",
    "targettrack_ssgcid": "SSGCID",
    "synechococcus_elongatus": "Synechococcus",
}

TOOL_COLORS = {
    "netsolp_usability": "#E53935",
    "netsolp_solubility": "#EF9A9A",
    "mpepe": "#FB8C00",
    "xp5_protein_only": "#7CB342",
    "xp5_champion_f10": "#1E88E5",
    "xp5_tier_a_esmc_prott5": "#5E35B1",
}


def panel_schematic_contamination(ax):
    """Panel 0: Venn-diagram-like schematic explaining what 'contamination' means.

    Two overlapping circles: Aiki-XP 492K training set (left, blue) and a
    published benchmark (right, orange). The intersection (purple) is
    labeled 'shared proteins (≥30% id ∧ ≥80% cov)' — this is what gets
    removed in the contamination-cleaned subset.
    """
    from matplotlib.patches import Circle, FancyArrowPatch
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("a   What is contamination?",
                 fontsize=9, fontweight="bold", loc="left")

    # Training set circle (left)
    c1 = Circle((-0.9, 0), 1.45, facecolor="#2196F3", alpha=0.35,
                edgecolor="#1565C0", linewidth=1.3)
    ax.add_patch(c1)
    ax.text(-1.85, 1.55, "Aiki-XP\ntraining set\n(492,026 proteins)",
            fontsize=6.5, fontweight="bold", color="#0D47A1",
            ha="center", va="center")

    # Benchmark circle (right)
    c2 = Circle((0.9, 0), 1.45, facecolor="#FF9800", alpha=0.35,
                edgecolor="#E65100", linewidth=1.3)
    ax.add_patch(c2)
    ax.text(1.85, 1.55, "Published\nbenchmark\n(e.g. Boël 2016)",
            fontsize=6.5, fontweight="bold", color="#BF360C",
            ha="center", va="center")

    # Intersection callout
    ax.text(0, 0, "shared\nproteins\n≥30% id\n≥80% cov",
            fontsize=6, fontweight="bold", color="#4A148C",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#E1BEE7",
                      ec="#7B1FA2", lw=0.8, alpha=0.95))

    # Bottom caption
    ax.text(0, -1.85,
            "Contamination = proteins in both sets. Published benchmark "
            "numbers may include\nprediction on these shared proteins, where "
            "the tool has seen near-identical training data.\nCleaned "
            "subset = benchmark minus the intersection.",
            fontsize=5.5, color="#444",
            ha="center", va="top", style="italic")


def panel_a_contamination(ax, contam_df):
    """Panel A: horizontal bar chart of contamination %."""
    bench_mask = contam_df["set_name"].isin(BENCH_DISPLAY.keys())
    bench = contam_df[bench_mask].copy()
    train = contam_df[~bench_mask].copy()
    data = pd.concat([bench, train], ignore_index=True)

    labels = []
    for s in data["set_name"]:
        if s in BENCH_DISPLAY:
            labels.append(f"{BENCH_DISPLAY[s]} (N={data.loc[data['set_name']==s, 'n_query'].iloc[0]:,})")
        else:
            labels.append(s.replace("_", " ").title())

    y = np.arange(len(data))
    colors = ["#2196F3" if s in BENCH_DISPLAY else "#FF9800" for s in data["set_name"]]
    ax.barh(y, data["pct_contaminated"], color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    for i, (val, nc) in enumerate(zip(data["pct_contaminated"], data["n_clean"])):
        ax.text(val + 1.5, i, f"{val:.1f}%", va="center", fontsize=7, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("% overlap with Aiki-XP 492K\n(≥30% id ∧ ≥80% cov)", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_title("b   Benchmark and training-set contamination", fontsize=9, fontweight="bold", loc="left")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Separator between benchmarks and training sets
    n_bench = bench_mask.sum()
    ax.axhline(n_bench - 0.5, color="gray", linestyle="--", linewidth=0.5)

    blue_patch = mpatches.Patch(color="#2196F3", label="Benchmarks")
    orange_patch = mpatches.Patch(color="#FF9800", label="Training sets")
    ax.legend(handles=[blue_patch, orange_patch], loc="lower right", fontsize=6, framealpha=0.8)


def panel_b_rho_bars(ax, summary_df):
    """Panel B: paired bars ρ_full vs ρ_clean for each (tool, benchmark) on Boël only."""
    boel = summary_df[summary_df["benchmark"] == "boel_2016"].copy()
    boel = boel.sort_values("rho_full", ascending=True)

    y = np.arange(len(boel))
    w = 0.35
    ax.barh(y - w/2, boel["rho_full"], height=w, color="#90CAF9", label="ρ full (N=6,348)", edgecolor="white")
    ax.barh(y + w/2, boel["rho_clean"], height=w, color="#1565C0", label="ρ clean (N=2,481)", edgecolor="white")

    labels = [TOOL_DISPLAY.get(t, t) for t in boel["tool"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Spearman ρ", fontsize=8)
    ax.set_title("c   Boël 2016: full vs contamination-cleaned", fontsize=9, fontweight="bold", loc="left")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="gray", linewidth=0.5)


def panel_c_delta_forest(ax, summary_df):
    """Panel C: forest plot of Δρ with bootstrap CIs, all (tool, benchmark) pairs."""
    df = summary_df.copy()
    df["label"] = df.apply(
        lambda r: f"{TOOL_DISPLAY.get(r['tool'], r['tool']).replace(chr(10), ' ')} × {BENCH_DISPLAY.get(r['benchmark'], r['benchmark'])}",
        axis=1,
    )
    df = df.sort_values("delta_rho")

    y = np.arange(len(df))
    xerr_lo = df["delta_rho"] - df["delta_rho_ci_low"]
    xerr_hi = df["delta_rho_ci_high"] - df["delta_rho"]
    xerr = np.array([xerr_lo.values, xerr_hi.values])

    colors = []
    for _, row in df.iterrows():
        if row["delta_rho_ci_high"] < 0:
            colors.append("#E53935")  # significant negative
        elif row["delta_rho_ci_low"] > 0:
            colors.append("#43A047")  # significant positive
        else:
            colors.append("#757575")  # not significant

    ax.errorbar(df["delta_rho"], y, xerr=xerr, fmt="o", color="#333",
                ecolor="#999", elinewidth=1, capsize=2, markersize=4, zorder=3)
    for i, (d, c) in enumerate(zip(df["delta_rho"], colors)):
        ax.plot(d, y.tolist()[i], "o", color=c, markersize=5, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=6)
    ax.set_xlabel("Δρ = ρ_clean − ρ_full", fontsize=8)
    ax.set_title("d   Contamination effect (Δρ, 95% CI)", fontsize=9, fontweight="bold", loc="left")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    neg_patch = mpatches.Patch(color="#E53935", label="Inflated (CI < 0)")
    pos_patch = mpatches.Patch(color="#43A047", label="Underest. (CI > 0)")
    ns_patch = mpatches.Patch(color="#757575", label="Not significant")
    ax.legend(handles=[neg_patch, pos_patch, ns_patch], fontsize=6, loc="lower right", framealpha=0.8)


def _build_figure(contam_df, summary_df):
    """Build the 4-panel landscape figure (schematic + 3 data panels).

    Layout:
        row 0: schematic intro (left) + contamination bar chart (right)
        row 1: Boël ρ_full/clean paired bars (left) + placeholder (right)
        row 2: forest plot (full width)
    """
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                          height_ratios=[1.0, 1.0, 1.3])

    ax_schematic = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :])

    panel_schematic_contamination(ax_schematic)
    panel_a_contamination(ax_a, contam_df)
    panel_b_rho_bars(ax_b, summary_df)
    panel_c_delta_forest(ax_c, summary_df)
    return fig


def main():
    contam_df = pd.read_csv(CONTAM_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = _build_figure(contam_df, summary_df)
    out_path = OUT_DIR / "si_landscape_contamination.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to {out_path}")

    # PNG copy for quick review
    fig2 = _build_figure(contam_df, summary_df)
    png_path = OUT_DIR / "si_landscape_contamination.png"
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved PNG to {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
