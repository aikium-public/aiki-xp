#!/usr/bin/env python3
"""
SI Figure: Protein language model scaling saturates across architectures.
Three families: ESM-2 (35M-15B), ESM-C (600M), ProtT5-XL (3B).

Uses overall ρ on gene-operon split, seed 42.
ESM-2 15B and ProtT5-XL via Ridge regression on 29K genes.

Sources:
  - ESM-2 35M/650M, ESM-C: results/protex_qc/v2_production_sweep/p1_*.json
  - ESM-2 15B: results/recipe_5fold_cv/ or Ridge scout
  - ProtT5-XL: results/recipe_5fold_cv/prot_t5_xl_protein_solo_seed42_fold0.json
    or tier_selection_analysis.json (ProtT5 solo ρ_nm = 0.511)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'protex' / 'latex' / 'figures' / 'nbt'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
})

# Data points: (params_M, overall_rho, label, family)
# ESM-2 family (ρ overall, gene-operon split, seed 42)
# ESM-2 35M: from v2_production_sweep
# ESM-2 650M: from v2_production_sweep
# ESM-2 15B: Ridge on 29K genes (from run_llm_scaling_scout.py)
# ESM-C 600M: from v2_production_sweep
# ProtT5-XL 3B: solo ρ_overall ≈ 0.572 (from tier_selection_analysis.json: ProtT5 solo)

data = {
    'ESM-2': [
        (35, 0.532, 'ESM-2\n35M'),
        (650, 0.562, 'ESM-2\n650M'),
        (15000, 0.532, 'ESM-2\n15B'),
    ],
    'ESM-C': [
        (600, 0.572, 'ESM-C\n600M'),
    ],
    'ProtT5-XL': [
        (3000, 0.572, 'ProtT5-XL\n3B'),
    ],
}

colors = {
    'ESM-2': '#1f77b4',
    'ESM-C': '#ff7f0e',
    'ProtT5-XL': '#2ca02c',
}

fig, ax = plt.subplots(figsize=(4.2, 3.2))

for family, pts in data.items():
    params = [p[0] for p in pts]
    rhos = [p[1] for p in pts]
    labels = [p[2] for p in pts]

    if len(pts) > 1:
        ax.plot(params, rhos, 'o-', color=colors[family], markersize=6,
                linewidth=1.2, label=family, zorder=3)
    else:
        ax.plot(params, rhos, 's', color=colors[family], markersize=7,
                label=family, zorder=3)

    for p, r, l in zip(params, rhos, labels):
        # Stagger labels to avoid collisions at 600M/650M/3B clusters.
        # ESM-C 600M sits above the ceiling line; ESM-2 650M just below
        # it. ProtT5-XL 3B is further right at the ceiling.
        if '35M' in l:
            offset = (0, 14)
            ha = 'center'
        elif '650M' in l:
            offset = (-35, -22)  # left of data point (below-left, avoids crossing lines)
            ha = 'right'
        elif '600M' in l:
            offset = (-35, 14)   # above-left (away from ceiling line)
            ha = 'right'
        elif '15B' in l:
            offset = (0, 14)     # above the data point (no more overlap)
            ha = 'center'
        elif '3B' in l:
            offset = (20, 14)    # above-right
            ha = 'left'
        else:
            offset = (0, 10)
            ha = 'center'
        ax.annotate(f'{l}\nρ={r:.3f}', (p, r), textcoords='offset points',
                   xytext=offset, fontsize=6, ha=ha, color=colors[family])

ax.set_xscale('log')
ax.set_xlabel('Parameters (millions)')
ax.set_ylabel(r'Overall $\rho$ (gene-operon, seed 42)')
ax.set_xlim(18, 32000)
ax.set_ylim(0.48, 0.62)

# Ceiling line — place label to the far right so it does not overlap
# any data-point annotation.
ax.axhline(y=0.572, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
ax.text(0.98, 0.572, ' Protein PLM ceiling',
        transform=ax.get_yaxis_transform(),
        fontsize=5.5, color='gray', va='bottom', ha='right')

ax.legend(fontsize=6, loc='lower right', frameon=False)

# Note about Ridge — factual reason (slow extraction / pre-computed
# embeddings only), not "training impractical".
ax.text(0.02, 0.02,
        'ESM-2 15B and ProtT5-XL evaluated via Ridge regression on 29K genes\n(same fold-0 hold-out split; pre-computed embeddings only)',
        transform=ax.transAxes, fontsize=5, color='gray', va='bottom')

for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'si_protein_scaling.{ext}')
print(f'Saved: si_protein_scaling.pdf + .png')
plt.close(fig)
