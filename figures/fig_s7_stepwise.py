#!/usr/bin/env python3
"""
S7 stepwise trajectory — single panel, non-conserved ρ, corrected labels.
Drops panel a (greedy R1 pick over-interpreted per coauthor).
Fixes: "non-mega" → "non-conserved", "classical" → "biophysical".
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'protex' / 'latex' / 'figures' / 'nbt'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

PRETTY = {
    'evo2_7b_full_operon_pca4096': 'Evo-2 7B\n(operon)',
    'hyenadna_dna_cds': '+HyenaDNA\n(CDS)',
    'bacformer_large': '+Bacformer-lg\n(genome)',
    'esmc_protein': '+ESM-C\n(protein)',
    'esm2_protein': '+ESM-2\n(protein)',
    'codonfm_cds': '+CodonFM\n(CDS)',
    'bacformer_base': '+BF-base\n(genome)',
}

def prettify(mod_list):
    """Get the newly added modality name."""
    for m in mod_list:
        if m in PRETTY:
            return PRETTY[m]
    # Biophysical block
    classical = [m for m in mod_list if m.startswith('classical_')]
    if classical:
        return '+Biophysical\n(69d)'
    return '+' + mod_list[0].replace('_', ' ')

# Load fold 1 non-conserved-optimized stepwise (greedy_rho_key = rho_non_mega)
f0 = json.load(open(ROOT / 'results/stepwise_nonmega_fold1/fold1/stepwise_progress.json'))

rounds = []
prev_set = set()
for h in f0['history']:
    if 'set' not in h:
        break  # incomplete round
    new_mods = set(h['set']) - prev_set
    label = prettify(list(new_mods)) if new_mods else 'Evo-2 7B\n(solo)'
    rounds.append({'round': h['round'], 'rho_nc': h['rho'], 'label': label, 'n_mods': len(h['set'])})
    prev_set = set(h['set'])

# Plot
fig, ax = plt.subplots(figsize=(5.5, 3.0), facecolor='#FAFBFC')

x = [r['round'] for r in rounds]
y = [r['rho_nc'] for r in rounds]
labels = [r['label'] for r in rounds]

# Color by biological scale
colors = []
for lab in labels:
    if 'operon' in lab.lower() or 'Evo' in lab:
        colors.append('#2ca02c')
    elif 'genome' in lab.lower() or 'Bacformer' in lab or 'BF-' in lab:
        colors.append('#9467bd')
    elif 'CDS' in lab or 'Codon' in lab or 'HyenaDNA' in lab:
        colors.append('#17becf')
    elif 'protein' in lab.lower() or 'ESM' in lab:
        colors.append('#1f77b4')
    elif 'Biophysical' in lab:
        colors.append('#ff7f0e')
    else:
        colors.append('#888888')

bars = ax.bar(x, y, color=colors, edgecolor='white', linewidth=0.5, width=0.7, alpha=0.55)

for i, (xi, yi) in enumerate(zip(x, y)):
    ax.text(xi, yi + 0.003, f'{yi:.3f}', ha='center', fontsize=6, fontweight='bold', color=colors[i])
    ax.text(xi, 0.555, labels[i], ha='center', fontsize=5.5, va='bottom', color='#2C3E50', rotation=0)

# Champion reference — pooled 2-seed 5-fold CV (§A.T7, 2026-04-11)
champ_nc = 0.592
ax.axhline(champ_nc, color='#e74c3c', ls='--', lw=0.8, alpha=0.6)
ax.text(len(x) - 0.5, champ_nc + 0.002, f'XP5 champion\n$\\rho_{{nc}}$={champ_nc:.3f}',
        fontsize=5.5, color='#e74c3c', ha='right', va='bottom')

ax.set_xlabel('Greedy round (fold 1, non-conserved optimization)')
ax.set_ylabel(r'Validation $\rho_\mathrm{nc}$ (fold 1)')
ax.set_title('Stepwise selection order\n(non-conserved optimization)', fontsize=9, fontweight='bold')
ax.set_ylim(0.55, 0.66)

# Note about validation vs test — moved to top-left to avoid overlapping
# the R4/R5 ESM-C/ESM-2 bars in the bottom-right.
ax.text(0.02, 0.97, 'Validation set values;\npooled 2-seed 5-fold test mean = 0.592',
        transform=ax.transAxes, fontsize=5, ha='left', va='top',
        color='#888', style='italic')
ax.set_xticks(x)
ax.set_xticklabels([f'R{i}' for i in x])

for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'si_nonmega_stepwise.{ext}')
plt.close(fig)
print(f'Saved: si_nonmega_stepwise.pdf + .png')
