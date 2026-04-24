#!/usr/bin/env python3
"""
Fig 6 v5: Cross-species evaluation — combined threshold curve + LOSCO + LOSO.
Merges old Fig 6 (threshold) with old ED3 (LOSCO/LOSO) per coauthor V3-8.
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMP = ROOT / 'results' / 'nonmega_champion_campaign_hardhybrid'
OUT = ROOT / 'docs' / 'protex' / 'latex' / 'figures' / 'nbt'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

BG = '#FAFBFC'

def load_rho(path):
    d = json.load(open(path))
    r = d['results']['single_adapter']
    return r['rho_overall'], r['rho_non_mega']

# ── Data ──
thresholds = [0.05, 0.10, 0.20, 0.30]
sc_rho, sc_nc = [], []
for t in thresholds:
    t_str = f'{t:.2f}'.replace('.', '')
    ro, rn = load_rho(CAMP / f'F_sc_t{t_str}_seed42.json')
    sc_rho.append(ro); sc_nc.append(rn)

# Gene-operon reference
go_d = json.load(open(CAMP / 'A_champion_go_seed42.json'))
go_rho = go_d['results']['single_adapter']['rho_overall']
go_nc = go_d['results']['single_adapter']['rho_non_mega']

# LOSCO (23 clusters)
losco = []
for d in sorted(glob.glob(str(ROOT / 'results/losco_xp5/losco_phylo_*_seed42'))):
    jsons = glob.glob(f'{d}/*.json')
    if jsons:
        data = json.load(open(jsons[0]))
        r = data['results']['single_adapter']
        cluster_id = Path(d).name.split('_')[2]
        # Count species from split file if available
        losco.append({'cluster': cluster_id, 'rho_nc': r['rho_non_mega'], 'rho': r['rho_overall']})
losco.sort(key=lambda x: x['rho_nc'])

# LOSO (10 species)
loso = []
for f in sorted(glob.glob(str(CAMP / 'G_loso_*_seed42.json'))):
    d = json.load(open(f))
    r = d['results']['single_adapter']
    sp = Path(f).name.replace('G_loso_', '').replace('_seed42.json', '').replace('_', ' ')
    loso.append({'species': sp, 'rho': r['rho_overall'], 'rho_nc': r['rho_non_mega']})
loso.sort(key=lambda x: x['rho'], reverse=True)

# ── Figure ──
# 3-panel layout: a (threshold curve) + b (family recognition) on top row,
# c (LOSCO forest) spanning full width on bottom. Panel d (LOSO) moved to
# Extended Data Table 2 (tab:loso).
fig = plt.figure(figsize=(7.2, 5.5), facecolor=BG)
gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.40, height_ratios=[1, 1.2])

# Panel a: threshold curve (top left)
ax_a = fig.add_subplot(gs[0, 0])
ax_a.text(-0.12, 1.10, 'a', transform=ax_a.transAxes, fontsize=12, fontweight='bold')
ax_a.plot(thresholds, sc_rho, 'o-', color='#2166AC', markersize=6, linewidth=1.5, label='Overall ρ', zorder=3)
ax_a.plot(thresholds, sc_nc, 's-', color='#B2182B', markersize=6, linewidth=1.5, label='ρ_nc', zorder=3)
ax_a.axhline(go_nc, color='#B2182B', ls='--', lw=0.8, alpha=0.5, label=f'GO ref (ρ_nc={go_nc:.3f})')
ax_a.axhline(go_rho, color='#2166AC', ls='--', lw=0.8, alpha=0.5)
for i, t in enumerate(thresholds):
    ax_a.text(t, sc_rho[i] + 0.008, f'{sc_rho[i]:.3f}', ha='center', fontsize=5.5, color='#2166AC')
    ax_a.text(t, sc_nc[i] - 0.015, f'{sc_nc[i]:.3f}', ha='center', fontsize=5.5, color='#B2182B')
ax_a.set_xlabel('Mash distance threshold\n(higher = coarser clusters, more shared families)')
ax_a.set_ylabel('Spearman ρ')
ax_a.set_title('Species-cluster split rigor\nvs performance', fontsize=8, fontweight='bold')
ax_a.legend(fontsize=6, loc='lower right')
ax_a.set_ylim(0.48, 0.72)

# Panel b: shared vs novel at t=0.20 (top right)
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.12, 1.10, 'b', transform=ax_b.transAxes, fontsize=12, fontweight='bold')
cats = ['Novel\nfamilies', 'Overall', 'Shared\nfamilies']
vals = [0.492, 0.671, 0.724]
colors = ['#B2182B', '#888888', '#2166AC']
bars = ax_b.bar(cats, vals, color=colors, edgecolor='white', width=0.55, alpha=0.85)
for bar, v in zip(bars, vals):
    ax_b.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
              ha='center', fontsize=7, fontweight='bold')
ax_b.set_ylabel('Spearman ρ')
ax_b.set_title('Family recognition\nvs regulatory transfer (t=0.20)', fontsize=8, fontweight='bold')
ax_b.set_ylim(0, 0.82)
ax_b.annotate('', xy=(2.3, 0.724), xytext=(2.3, 0.492),
              arrowprops=dict(arrowstyle='<->', color='#888', lw=1))
ax_b.text(2.45, 0.608, 'Δ=0.232', fontsize=6, color='#888', va='center')

# Panel c: LOSCO vertical bars (bottom row)
ax_c = fig.add_subplot(gs[1, :])
ax_c.text(-0.06, 1.06, 'c', transform=ax_c.transAxes, fontsize=12, fontweight='bold')
x_c = np.arange(len(losco))
nc_vals = [l['rho_nc'] for l in losco]
cmap = plt.cm.YlGnBu
norm_vals = [(v - min(nc_vals)) / (max(nc_vals) - min(nc_vals) + 1e-6) for v in nc_vals]
bar_colors = [cmap(0.3 + 0.6 * nv) for nv in norm_vals]
ax_c.bar(x_c, nc_vals, color=bar_colors, edgecolor='white', linewidth=0.3, width=0.8)
mean_nc = np.mean(nc_vals)
ax_c.axhline(mean_nc, color='#B2182B', ls='--', lw=0.8)
ax_c.text(len(losco) - 0.5, mean_nc + 0.002, f'mean={mean_nc:.3f} ± {np.std(nc_vals):.3f}',
          fontsize=5.5, color='#B2182B', ha='right')
_cluster_sp_counts = {
    '0000': 139, '0001': 80, '0002': 12, '0003': 7, '0004': 7,
    '0005': 6, '0006': 5, '0007': 4, '0008': 3, '0009': 3,
    '0010': 2, '0011': 2, '0012': 2, '0013': 2, '0014': 2,
    '0015': 2, '0016': 2, '0017': 2, '0018': 2, '0019': 2,
    '0020': 2, '0021': 2, '0022': 2,
}
_labels_c = [f"{_cluster_sp_counts.get(l['cluster'], '?')}" for l in losco]
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(_labels_c, fontsize=4.5, rotation=0)
ax_c.set_xlabel('Species per held-out cluster')
ax_c.set_ylabel(r'$\rho_\mathrm{nc}$ (non-conserved)')
ax_c.set_title(f'LOSCO: 23 phylogenetic clusters held out one at a time',
               fontsize=8, fontweight='bold')
ax_c.set_ylim(0.50, 0.62)

for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig6_cross_species.{ext}')
plt.close(fig)
print('Saved: fig6_cross_species.pdf + .png')
