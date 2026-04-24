#!/usr/bin/env python3
"""
Fig ED1 v3: Comprehensive external validation — defining figure.

Six panels:
  a) Deployment tier overview with bio-icons showing accumulating information
  b) Native expression external validation (tier-colored)
  c) Difficulty-vs-performance across evaluation regimes
  d) Cambray case study — ONLY 5-fold XP5 variants on 492K training
  e) Bespoke model schematic — general recipe for task-specific models
  f) Solubility cross-evaluation — endpoint conflation story

Sources:
  - deployment_tiers.yaml (commit 99f84b7c)
  - results/comprehensive_external_eval/tier_a_canonical/
  - results/comprehensive_external_eval/cambray_2018/cambray_case_study_results.json
  - results/external_validation_nm5/external_validation_nm5_vs_f10.json
  - results/synechococcus_all_tiers/all_tier_results.json
  - results/zscore_vs_raw/v21_temporal_holdout_nm5_per_species.json
  - see deployment_tiers.yaml
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'protex' / 'latex' / 'figures' / 'nbt'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'pdf.fonttype': 42,
})

# ═══════════════════════════════════════════════════════════════════
# Colors
# ═══════════════════════════════════════════════════════════════════
TC = {'A': '#3B7DD8', 'B': '#7B68AE', 'C': '#43A047', 'D': '#C0392B'}
CAM = '#D4A017'   # amber (bespoke)
INK = '#2C3E50'
BG = '#FAFBFC'
LG = '#D5D8DC'
GRAY = '#888888'

# Bio-icon colors (matching generate_fig1_nbt_clean.py aesthetic)
BIO_PROTEIN = '#1565C0'   # blue
BIO_DNA = '#2E7D32'        # green
BIO_OPERON = '#558B2F'     # olive green
BIO_GENOME = '#E65100'     # orange (bacterium)


def plabel(ax, s, x=-0.08, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='right', color=INK)


# ═══════════════════════════════════════════════════════════════════
# Bio-icon drawing functions (adapted from generate_fig1_nbt_clean.py)
# All coordinates are in AXES data units unless noted.
# ═══════════════════════════════════════════════════════════════════

def draw_protein_icon(ax, x, y, scale=1.0, color=BIO_PROTEIN):
    """Folded protein blob with helix suggestion."""
    s = scale * 0.25
    theta = np.linspace(0, 2*np.pi, 60)
    r = s * (1 + 0.25*np.sin(3*theta) + 0.15*np.cos(5*theta))
    px = x + r * np.cos(theta)
    py = y + r * np.sin(theta)
    ax.fill(px, py, color=color, alpha=0.18, zorder=9)
    ax.plot(px, py, color=color, lw=1.0, zorder=10)
    # Alpha helix suggestion
    t2 = np.linspace(-0.6*s, 0.6*s, 25)
    hx = x + t2
    hy = y + 0.25*s*np.sin(12*t2/s)
    ax.plot(hx, hy, color=color, lw=0.8, alpha=0.65, zorder=11)


def draw_dna_icon(ax, x, y, w=0.45, h=0.12, color=BIO_DNA):
    """Short double-helix glyph."""
    t = np.linspace(0, 3*np.pi, 80)
    sx = x + np.linspace(-w/2, w/2, 80)
    sy1 = y + h * np.sin(t)
    sy2 = y + h * np.sin(t + np.pi)
    ax.plot(sx, sy1, color=color, lw=1.2, solid_capstyle='round', zorder=10)
    ax.plot(sx, sy2, color=color, lw=1.2, solid_capstyle='round', zorder=10)
    # Rungs
    for i in range(0, 80, 8):
        if np.sin(t[i]) > np.sin(t[i] + np.pi):
            ax.plot([sx[i], sx[i]], [sy1[i], sy2[i]], color=color,
                    lw=0.5, alpha=0.45, zorder=9)


def draw_operon_icon(ax, x, y, w=0.6, color=BIO_OPERON):
    """Operon: multiple gene arrows on a DNA backbone."""
    h = 0.09
    # Backbone
    ax.plot([x - w/2, x + w/2], [y, y], color=color, lw=1.2, zorder=9)
    ax.plot([x - w/2, x + w/2], [y - h*0.45, y - h*0.45], color=color, lw=1.2, zorder=9)
    # Gene arrows (3 genes)
    gene_w = w / 3.3
    for i in range(3):
        gx = x - w/2 + i * gene_w * 1.1 + 0.02
        gene = FancyBboxPatch((gx, y - h*0.3), gene_w*0.85, h*0.6,
                               boxstyle="round,pad=0.005,rounding_size=0.02",
                               facecolor=color, edgecolor=color, alpha=0.4, lw=0.6,
                               zorder=10)
        ax.add_patch(gene)
        # Arrow tip
        tip = Polygon([[gx + gene_w*0.85, y - h*0.3],
                       [gx + gene_w*1.02, y],
                       [gx + gene_w*0.85, y + h*0.3]],
                      facecolor=color, edgecolor=color, alpha=0.6, zorder=11)
        ax.add_patch(tip)


def draw_bacterium_icon(ax, x, y, scale=1.0, color=BIO_GENOME):
    """Rod-shaped bacterium with circular chromosome inside."""
    s = scale * 0.22
    # Body (capsule)
    body = FancyBboxPatch((x - 1.6*s, y - 0.6*s), 3.2*s, 1.2*s,
                           boxstyle=f"round,pad=0,rounding_size={0.6*s}",
                           facecolor=color, alpha=0.18, edgecolor=color,
                           lw=1.0, zorder=10)
    ax.add_patch(body)
    # Internal chromosome (wavy circle)
    ch_t = np.linspace(0, 1.85*np.pi, 40)
    cx = x + 0.55*s * np.cos(ch_t)
    cy = y + 0.32*s * np.sin(ch_t)
    ax.plot(cx, cy, color=color, lw=0.8, alpha=0.7, zorder=11)


def draw_features_icon(ax, x, y, color='#D4A017'):
    """Bar chart icon for biophysical features."""
    heights = [0.30, 0.50, 0.20, 0.40, 0.25]
    w = 0.04
    for i, h in enumerate(heights):
        rx = x - 0.12 + i * 0.05
        ax.add_patch(FancyBboxPatch((rx, y - 0.02), w, h*0.3,
                                     boxstyle="round,pad=0.002",
                                     facecolor=color, edgecolor='none',
                                     alpha=0.75, zorder=10))


# ═══════════════════════════════════════════════════════════════════
# DATA (all from verified JSONs)
# ═══════════════════════════════════════════════════════════════════

# Tier-stratified comparison on the LOSO E. coli benchmarks (panel b)
# Source: results/external_validation_deployment_tiers/loso_ecoli/*.json
# Column recipes (verified against each JSON's `embedders` field):
#   Tier A*: deploy_esmc_solo_seed42 — ESM-C solo (PRIOR Tier A lock, pre-ProtT5).
#            The current Tier A lock esmc_prott5_seed42 has not been re-evaluated
#            on LOSO E. coli (would require a full 5-fold LOSO retrain). Shown
#            with an asterisk in the panel.
#   Tier B:  deploy_protein_cds_features_6mod_seed42 — current Tier B lock.
#   Tier D:  balanced_nonmega_5mod (NM5 champion) — current Tier D lock,
#            loaded from results/nonmega_champion_campaign_hardhybrid/
#            G_loso_Escherichia_coli_K12_seed42.json (embedders verified).
# Tier C omitted: its input contract (operon DNA) doesn't apply to LOSO
# E. coli; Tier C is shown directly on Cambray in panel d.
LOSO_TIER_COMPARE = [
    # (dataset_short, n, tier_A_rho, tier_B_rho, tier_D_rho)
    ('Li 2014\nribosome profiling',  3409, 0.6017, 0.6129, 0.6726),
    ('Mori 2021\nDIA-MS',            2076, 0.6105, 0.6319, 0.6205),
    ('Taniguchi 2010\nYFP',           685, 0.5552, 0.5895, 0.6099),
]

# Tier D extras and OOD (panel c) — broader generalization story.
# All values are from current-lock recipes:
#   - temporal / LOSCO / phylum / Synechococcus NM5: NM5 champion (Tier D lock)
#   - Boël / Price: Tier A canonical (esmc_prott5_seed42)
# Source JSONs cited in the .tex comment block.
# Synechococcus: NM5 standard rho=0.0286 (p=0.199, NOT significant); NM5 dropout
# rho=0.0729 (p=0.001). Shown here as NM5 dropout because that is the deployed
# variant (train_fusion.py modality_dropout=0.2). The retired F10-1B baseline
# reached rho=0.147 but is not a current tier recipe.
# Source: results/synechococcus_all_tiers/all_tier_results.json
TIER_D_EXTRAS = [
    # (label, rho, category, detail)
    ('V2.1 temporal\nholdout',     0.657, 'native',  '56K genes, 89 sp.'),
    ('LOSCO\n23 clusters',          0.580, 'phylo',   r'$\rho_{nc}$'),
    ('Phylum holdouts\n(3 phyla)',  0.605, 'phylo',   'mean'),
    ('Bo\u00ebl 2016\n(T7 overexpr.)', 0.165, 'hetero',  '6,348 genes'),
    ('Price/NESG\n(cell-free)',     0.113, 'hetero',  '9,703 genes'),
    ('Synechococcus\n(novel phylum)', 0.073, 'extreme', 'NM5 dropout'),
]

# Difficulty vs performance (panel c)
# Ordinal difficulty scale: 1 (easiest) to 7 (hardest)
DIFFICULTY = [
    # (difficulty, rho, label, tier)
    (1, 0.592, '5-fold CV', 'D'),                      # in-distribution (T7 pooled 2-seed)
    (1.3, 0.580, 'LOSCO\n23 clusters', 'D'),           # phylogenetic holdout
    (2, 0.657, 'Temporal\nV2.1', 'D'),                 # novel time
    (2.3, 0.673, 'LOSO Li\n(ribosome)', 'D'),          # LOSO E. coli
    (2.5, 0.621, 'LOSO Mori', 'D'),
    (2.7, 0.610, 'LOSO\nTaniguchi', 'D'),
    (3, 0.651, 'LOSO 10\nspecies mean', 'D'),          # full LOSO panel
    (3.5, 0.605, 'Phylum\nholdouts', 'D'),              # cross-phylum
    (5, 0.165, 'Bo\u00ebl T7\noverexpression', 'A'),    # heterologous
    (5.3, 0.113, 'Price/NESG\ncell-free', 'A'),         # heterologous
    (6, 0.073, 'Synechococcus\n(NM5 dropout)', 'D'),    # extreme OOD, NM5 dropout
    (7, 0.434, 'Cambray\n(codon library)', 'D'),  # classical_only corrected; see panel d
]

# Cambray case study (panel d) — all values are 5-fold XP5 ensembles
# trained on 492K (transfer to Cambray, no Cambray labels seen during training).
# Canonical sources (corrected via XP5Ensemble, Rule 17):
#   - results/cambray_comprehensive/phase5_cambray_eval.json (Phase 5)
#   - results/comprehensive_external_eval/cambray_2018/cambray_r{1,2}_eval.json (R1/R2)
#   - results/comprehensive_external_eval/cambray_2018/tier_c_canonical/cambray_tier_c__evo2_prott5_seed42.json
# Three clusters visualize the dilution rule (classical → FM+classical → FM-only).
CAMBRAY_CLASSICAL = [
    # Classical biophysics only — no foundation model readouts.
    # Both values are XP5Ensemble (Rule 17) 5-fold transfers from 492K.
    # Adding 11d codon features on top of 16d rna_init nearly doubles the
    # residual gain: 0.291 → 0.434. Composition of orthogonal mechanistic
    # signals, not a "classical features transfer cleanly" oversimplification.
    ('codon + rna\\_init (27d)\nclassical biophysics', 0.4335),
    ('rna\\_init solo (16d)\nno codon features', 0.2912),
]
CAMBRAY_FM_CLASSICAL = [
    # FM readouts co-mingled with classical biophysics
    ('init\\_window best\n(4 FMs + rna\\_init)', 0.2886),
    ('RiNALMo + rna\\_init\n(1 FM + classical)', 0.2768),
    ('kitchen\\_sink\n(NM5, 9 mods)', 0.1716),
]
CAMBRAY_FM_ONLY = [
    # Pure FM readouts — no classical features at all
    ('evo2\\_7b multiwindow\n(3 windows, FM only)', 0.1015),
    ('RiNALMo solo\n(FM only)', 0.0847),
    ('Tier C lock\n(Evo-2 + ProtT5-XL)', 0.0747),
]

# Cambray-internal Ridge ceiling (upper bound fitting directly on Cambray,
# 5-fold CV) — quantifies the cross-domain transfer gap.
# Source: results/cambray_comprehensive/phase2_ridge_screening.json
CAMBRAY_RIDGE_CEILING = 0.6574  # evo2_7b_init_70nt_pca4096, 4096d
CAMBRAY_RIDGE_CLASSICAL = 0.4782  # classical_rna_init, 16d (same sample)

# Solubility cross-evaluation (panel f)
SOLUBILITY = [
    # (name, auroc, n, note)
    ('Solubility mixed\n(hybrid DB)', 0.641, 103906, 'Strong signal'),
    ('Solubility overall\n(binary)', 0.630, 149103, ''),
    ('Solubility pure\n(NESG pure-sol)', 0.593, 33501, 'Weaker signal'),
    ('Price/NESG\nsolubility', 0.218, 9703,
     r'$\rho$ (not AUROC)'),
    ('eSol control\n(intrinsic folding)', 0.487, 2702, 'Chance baseline'),
]


# ═══════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.5, 10.5))
gs = gridspec.GridSpec(4, 2, figure=fig,
                       height_ratios=[1.35, 1.4, 1.55, 1.0],
                       width_ratios=[1, 1],
                       hspace=0.65, wspace=0.35,
                       left=0.08, right=0.97, top=0.96, bottom=0.05)


# ──────────────────────────────────────────────────────────────────
# Panel a: Tier overview with bio-icons (full width)
# ──────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
plabel(ax_a, 'a', x=-0.02, y=1.20)
ax_a.set_xlim(0, 12)
ax_a.set_ylim(-1.15, 4.80)  # extended vertical range for breathing room
ax_a.axis('off')
ax_a.set_title('Aiki-XP deployment tiers: match the model to your available data',
                fontsize=9, fontweight='bold', color=INK, loc='left', pad=8)

# Information accumulation arrow (left → right) at the bottom
ax_a.annotate('', xy=(11.3, -0.60), xytext=(0.7, -0.60),
              arrowprops=dict(arrowstyle='->', color='#B0B0B0', lw=2,
                              shrinkA=0, shrinkB=0))
ax_a.text(6.0, -1.00, 'increasing information available to the model',
          ha='center', va='top', fontsize=6.5, color=GRAY, style='italic',
          fontweight='bold')

# Compact icon legend at the top of panel a
legend_y = 4.35
spec = [
    ('protein', 'protein', 0.50),
    ('dna', 'CDS DNA', 1.50),
    ('operon', 'operon', 2.55),
    ('genome', 'host genome', 3.65),
    ('features', 'biophysical', 5.10),
]
ax_a.text(0.0, legend_y, 'icons:', fontsize=5.5, color='#666',
          va='center', ha='left', style='italic')
for kind, label, dx in spec:
    ix = 0.55 + dx
    if kind == 'protein':
        draw_protein_icon(ax_a, ix, legend_y, scale=0.55)
    elif kind == 'dna':
        draw_dna_icon(ax_a, ix, legend_y, w=0.30, h=0.07)
    elif kind == 'operon':
        draw_operon_icon(ax_a, ix, legend_y, w=0.40)
    elif kind == 'genome':
        draw_bacterium_icon(ax_a, ix, legend_y, scale=0.55)
    elif kind == 'features':
        draw_features_icon(ax_a, ix, legend_y)
    ax_a.text(ix + 0.27, legend_y, label, fontsize=5.0, color='#666',
              va='center', ha='left')

tier_spec = [
    # (x, tier, title, modalities_str, rho, icons_to_draw)
    # MONOTONE LADDER A → B → B+ → C → D. This ladder is the central
    # story of the paper: each tier's input is a strict superset of the
    # previous tier's input, and every step both adds biological context
    # AND improves ρ_nc — demonstrating that bacterial expression
    # prediction benefits from ALL scales of biological information. This
    # is a cleaner version of the signal we originally chased with
    # Evo-2 7B alone.
    #
    # NAMING (§A.TIER_B_PLUS_DISPLAY_RENAME, 2026-04-13):
    #   • External display label = "Tier B+"  (reads naturally as
    #     "Tier B augmented with +5′ UTR input").
    #   • Internal config key = "A_plus_protein_init_window"
    #     (kept for code/pipeline stability in
    #     configs/protex/deployment_tiers.yaml).
    #
    # Tier B+ = R2 (Tier B body + ProtT5-XL + Evo-2 7B init_70nt, 8 mod)
    # at ρ_nc = 0.555 ± 0.014 from §R2 / §A.TIER_A_PLUS_MISALIGNMENT
    # (2026-04-13 fix). Previously this box showed an init-window-only
    # ablation at 0.538 with a "parallel branch" narrative — that was a
    # misrepresentation because a real user with upstream DNA also has
    # the CDS that follows the ATG.
    (0.10, 'A', 'Protein AA only',
     'ESM-C +\nProtT5-XL', 0.518,
     ['protein']),
    (2.52, 'B', '+ coding DNA',
     'ESM-C + HyenaDNA\n+ CodonFM + biophys', 0.530,
     ['protein', 'dna', 'features']),
    (4.94, 'B+', '+ 5\u2032 UTR',
     'Tier B + ProtT5-XL + Evo-2 1B\ninit\\_window + ViennaRNA 5\u2032 UTR', 0.543,
     ['protein', 'dna', 'features']),
    (7.36, 'C', '+ full operon',
     'Evo-2 7B +\nProtT5-XL', 0.576,
     ['protein', 'operon']),
    (9.78, 'D', '+ host genome',
     'NM5 champion\n(9 mod, 5 biophys)', 0.592,
     ['protein', 'operon', 'genome', 'features']),
]

# Add Tier A+ to TC color map (between A and B visually — dark teal)
TC['B+'] = '#1F8AB7'  # teal between B (purple) and C (green) in the monotone ladder

BOX_W = 2.10
BOX_H = 3.85  # taller boxes — more vertical space between icons and text

for x0, tier, title, recipe, rho, icons in tier_spec:
    c = TC[tier]
    # Background box (colored)
    ax_a.add_patch(FancyBboxPatch((x0, 0.0), BOX_W, BOX_H,
                                    boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor='white',
                                    alpha=0.08, lw=0, zorder=1))
    # Border
    ax_a.add_patch(FancyBboxPatch((x0, 0.0), BOX_W, BOX_H,
                                    boxstyle="round,pad=0.08",
                                    facecolor='none', edgecolor=c,
                                    lw=1.6, zorder=2))

    # Tier label
    ax_a.text(x0 + BOX_W/2, BOX_H - 0.15, f'Tier {tier}',
              ha='center', va='top', fontsize=9, fontweight='bold', color=c)
    ax_a.text(x0 + BOX_W/2, BOX_H - 0.55, title,
              ha='center', va='top', fontsize=6.5, color=INK)

    # Icon row — slightly larger, with clear separation from text above/below
    n_icons = len(icons)
    icon_spacing = BOX_W * 0.78 / max(n_icons, 1)
    icon_y = 2.30
    icon_x_start = x0 + BOX_W/2 - (n_icons - 1) * icon_spacing / 2
    for i, icon_type in enumerate(icons):
        ix = icon_x_start + i * icon_spacing
        if icon_type == 'protein':
            draw_protein_icon(ax_a, ix, icon_y, scale=0.85)
        elif icon_type == 'dna':
            draw_dna_icon(ax_a, ix, icon_y, w=0.42, h=0.10)
        elif icon_type == 'operon':
            draw_operon_icon(ax_a, ix, icon_y, w=0.58)
        elif icon_type == 'genome':
            draw_bacterium_icon(ax_a, ix, icon_y, scale=0.90)
        elif icon_type == 'features':
            draw_features_icon(ax_a, ix, icon_y)

    # Recipe text (below icons) — pulled further down so it doesn't
    # touch the icons above
    ax_a.text(x0 + BOX_W/2, 1.25, recipe,
              ha='center', va='center', fontsize=5.3, color='#555',
              style='italic')

    # Rho badge at bottom
    ax_a.add_patch(FancyBboxPatch((x0 + BOX_W/2 - 0.65, 0.28), 1.3, 0.44,
                                    boxstyle="round,pad=0.04",
                                    facecolor=c, edgecolor='none', alpha=0.92,
                                    zorder=3))
    ax_a.text(x0 + BOX_W/2, 0.50,
              f'$\\rho_{{nc}}$ = {rho:.3f}',
              ha='center', va='center', fontsize=7.2, fontweight='bold',
              color='white')

# (Icon legend removed — icons are labeled inline in each tier box)


# ──────────────────────────────────────────────────────────────────
# Panel b: Tier-stratified comparison on E. coli LOSO (row 2, left)
# Shows that adding upstream/genome context monotonically improves
# prediction across three independent measurement technologies.
# ──────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])
plabel(ax_b, 'b', x=-0.10, y=1.08)
ax_b.set_facecolor(BG)
ax_b.set_title('Tier comparison on LOSO ${E.~coli}$ benchmarks',
                fontsize=8, fontweight='bold', color=INK, loc='left', pad=4)

# Grouped bars: 3 datasets × 3 tiers
n_datasets = len(LOSO_TIER_COMPARE)
group_w = 0.78
bar_w = group_w / 3
x_pos = np.arange(n_datasets)
tier_colors = [TC['A'], TC['B'], TC['D']]
tier_labels = ['ESM-C solo*\n(Tier A contract)',
               'Tier B lock\n(+CDS DNA)',
               'Tier D lock\n(NM5, full genome)']

for ti, (col, tlbl) in enumerate(zip(tier_colors, tier_labels)):
    vals = [d[2 + ti] for d in LOSO_TIER_COMPARE]
    offsets = (ti - 1) * bar_w
    bars = ax_b.bar(x_pos + offsets, vals, bar_w, color=col,
                     edgecolor='white', linewidth=0.5, alpha=0.88,
                     label=tlbl)
    for xi, v in zip(x_pos + offsets, vals):
        ax_b.text(xi, v + 0.008, f'{v:.3f}', ha='center', va='bottom',
                  fontsize=5.0, color=INK, fontweight='bold')

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([d[0] for d in LOSO_TIER_COMPARE], fontsize=6)
ax_b.set_ylim(0, 0.82)
ax_b.set_ylabel(r'Spearman $\rho$', fontsize=7)
ax_b.axhline(0.5, color=LG, lw=0.5, ls='--', zorder=0)
ax_b.legend(fontsize=5, loc='upper left', frameon=True,
             framealpha=0.9, edgecolor=LG, ncol=3,
             columnspacing=0.6, handletextpad=0.3,
             bbox_to_anchor=(0.0, 1.0))
# Footnote moved to figure-relative coordinates via ax_b.annotate-style
# with xycoords="axes fraction"; placed well BELOW the x-axis labels
# (y=-0.35 → below tick labels at ~−0.10). Removed \texttt{} which was
# rendering as raw LaTeX.
ax_b.text(0.5, -0.28,
          '*Tier A bars use deploy_esmc_solo (prior lock, pre-ProtT5);\n'
          'the current esmc_prott5 lock has not been re-evaluated on LOSO ${E.~coli}$.\n'
          'Tier C (operon-DNA contract) is shown directly on Cambray (panel d).',
          transform=ax_b.transAxes, fontsize=4.5, ha='center', va='top',
          color='#888', style='italic')

# ──────────────────────────────────────────────────────────────────
# Panel c (NEW): Tier D extras — temporal, phylo, hetero, OOD
# Compact bar chart that absorbs the broader generalization story
# previously crammed into the old NATIVE_EXPR panel.
# ──────────────────────────────────────────────────────────────────
ax_c2 = fig.add_subplot(gs[1, 1])
plabel(ax_c2, 'c', x=-0.10, y=1.08)
ax_c2.set_facecolor(BG)
ax_c2.set_title('Generalization range (Tier D unless noted)',
                 fontsize=8, fontweight='bold', color=INK, loc='left', pad=4)

cat_colors = {
    'native':  TC['D'],
    'phylo':   '#E67E22',
    'hetero':  TC['A'],
    'extreme': '#8D6E63',
}
y_c2 = np.arange(len(TIER_D_EXTRAS))
vals_c2 = [d[1] for d in TIER_D_EXTRAS]
colors_c2 = [cat_colors[d[2]] for d in TIER_D_EXTRAS]

ax_c2.barh(y_c2, vals_c2, color=colors_c2, edgecolor='white',
            height=0.65, alpha=0.88)
for i, (n, v, cat, det) in enumerate(TIER_D_EXTRAS):
    ax_c2.text(v + 0.008, i - 0.18, f'{v:.3f}', va='center', fontsize=6,
                fontweight='bold', color=INK)
    ax_c2.text(v + 0.008, i + 0.20, det, va='center', ha='left', fontsize=4.5,
                color='#888', style='italic')

ax_c2.set_yticks(y_c2)
ax_c2.set_yticklabels([d[0] for d in TIER_D_EXTRAS], fontsize=5.5)
ax_c2.invert_yaxis()
ax_c2.set_xlim(0, 0.80)
ax_c2.set_xlabel(r'Spearman $\rho$ or $\rho_{nc}$', fontsize=6.5)
ax_c2.axvline(0.5, color=LG, lw=0.5, ls='--', zorder=0)

legend_c2 = [
    mpatches.Patch(facecolor=TC['D'], label='Tier D native'),
    mpatches.Patch(facecolor='#E67E22', label='phylogenetic'),
    mpatches.Patch(facecolor=TC['A'], label='Tier A heterologous'),
    mpatches.Patch(facecolor='#8D6E63', label='novel phylum'),
]
ax_c2.legend(handles=legend_c2, fontsize=4.5, loc='lower right',
              frameon=True, framealpha=0.9, edgecolor=LG, ncol=1,
              handletextpad=0.3, borderpad=0.3)


# (Removed: difficulty-vs-performance scatter — its content is now
#  visible directly in panels b (tier comparison) and c (Tier D extras).)


# ──────────────────────────────────────────────────────────────────
# Panel f: Solubility cross-evaluation (row 4, full width)
# ──────────────────────────────────────────────────────────────────
ax_f = fig.add_subplot(gs[3, :])
plabel(ax_f, 'f', x=-0.04, y=1.10)
ax_f.set_facecolor(BG)

x_f = np.arange(len(SOLUBILITY))
vals_f = [d[1] for d in SOLUBILITY]
is_rho = ['rho' in d[3].lower() for d in SOLUBILITY]
colors_f = [TC['A'] if not r else '#95a5a6' for r in is_rho]

ax_f.bar(x_f, vals_f, color=colors_f, edgecolor='white',
          width=0.65, alpha=0.88)
for i, (n, v, nn, note) in enumerate(SOLUBILITY):
    ax_f.text(i, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=6,
              fontweight='bold', color=INK)
    ax_f.text(i, -0.04, f'n={nn:,}', ha='center', va='top', fontsize=4.5,
              color='#888', style='italic')

# Chance line
ax_f.axhline(0.5, color='#B71C1C', lw=0.6, ls='--', zorder=1, alpha=0.6)
ax_f.text(len(SOLUBILITY) - 0.4, 0.51, 'chance baseline',
          fontsize=5, color='#B71C1C', style='italic', ha='right', va='bottom')

ax_f.set_xticks(x_f)
ax_f.set_xticklabels([d[0] for d in SOLUBILITY], fontsize=5.5)
ax_f.set_ylim(0.0, 0.85)
ax_f.set_ylabel('AUROC (or $\\rho$, gray)', fontsize=6.5)
ax_f.set_title('Endpoint conflation: solubility $\\neq$ expression',
               fontsize=8, fontweight='bold', color=INK, loc='left')

# Caption
ax_f.text(0.5, -0.30, 'Native-expression features transfer marginally to solubility prediction '
          '(AUROC 0.59–0.64, near eSol intrinsic-folding chance baseline 0.49); '
          'pure-expression endpoints yield substantially higher correlations than endpoints that '
          'conflate solubility and expression. Direction flips across construct batches in '
          'designed-protein solubility (Aikium scaffolds, see Discussion).',
          transform=ax_f.transAxes, fontsize=5.5, ha='center', va='top',
          color='#666', style='italic', wrap=True)


# ──────────────────────────────────────────────────────────────────
# Panel d (row 3, FULL WIDTH): Cambray — ONLY 5-fold XP variants on 492K
# Previously shared row 3 with a bespoke-recipe schematic (panel e),
# which was dropped 2026-04-13 because its Cambray example (ρ=0.434) was
# not spectacular enough to warrant a whole sub-panel. Giving panel d
# the full row width resolves the crowded y-axis and annotation overlap.
# ──────────────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[2, :])
plabel(ax_d, 'd', x=-0.05, y=1.12)
ax_d.set_facecolor(BG)

# Build the three-group bar chart (classical / FM+classical / FM-only)
GAP = 0.45  # vertical gap between consecutive groups
FM_COLOR = '#8E44AD'     # muted purple for FM+classical
FM_ONLY_COLOR = '#888888'  # gray for FM-only
groups = [
    (CAMBRAY_CLASSICAL, CAM),
    (CAMBRAY_FM_CLASSICAL, FM_COLOR),
    (CAMBRAY_FM_ONLY, FM_ONLY_COLOR),
]

y_cam = []
labels_cam = []
vals_cam = []
colors_cam = []
group_bounds = []  # (start_idx, end_idx) for each group in y_cam
gap_positions = []  # y-positions of dividing lines
idx = 0
for gi, (group, base_color) in enumerate(groups):
    start = len(y_cam)
    for item in group:
        n, v = item[0], item[1]
        y_cam.append(idx)
        labels_cam.append(n)
        vals_cam.append(v)
        # Override base color for canonical tier-locked recipes
        if 'Tier C lock' in n:
            colors_cam.append(TC['C'])
        elif 'NM5' in n or 'Tier D' in n or 'kitchen' in n:
            colors_cam.append(base_color)
        else:
            colors_cam.append(base_color)
        idx += 1
    end = len(y_cam)
    group_bounds.append((start, end))
    if gi < len(groups) - 1:
        gap_positions.append(idx - 0.5 + GAP / 2)
        idx += GAP

y_cam = np.array(y_cam)
bars = ax_d.barh(y_cam, vals_cam, color=colors_cam, edgecolor='white',
                  height=0.68, alpha=0.88)
for yi, v, lab in zip(y_cam, vals_cam, labels_cam):
    ax_d.text(v + 0.006, yi, f'{v:.3f}', va='center', fontsize=6.0,
              fontweight='bold', color=INK)

# Dividing lines
for gp in gap_positions:
    ax_d.axhline(gp, color=INK, lw=0.4, ls=':', alpha=0.5)

# Group badges — y positions computed from actual bar positions.
# Now on the RIGHT side of the plot so they don't overlap the "0.433"
# value label on the first bar.
group_labels_cfg = [
    ('Classical only (no FM)', CAM, '#FFF8E1', r'$\approx 0.43$'),
    ('FM + classical (diluted)', FM_COLOR, '#F3E5F5', r'$\approx 0.17$--$0.29$'),
    ('FM only (no classical)', FM_ONLY_COLOR, '#F5F5F5', r'$\approx 0.07$--$0.10$'),
]
for (start, end), (lbl, col, bg, rng) in zip(group_bounds, group_labels_cfg):
    if start == end:
        continue
    mid_y = np.mean(y_cam[start:end])
    ax_d.text(0.74, mid_y, f'{lbl}\n{rng}',
              fontsize=6.0, fontweight='bold', color=col, ha='left', va='center',
              bbox=dict(boxstyle='round,pad=0.22', facecolor=bg,
                        edgecolor='none', alpha=0.85))

# Cambray-internal Ridge ceiling (shows the cross-domain transfer gap)
ax_d.axvline(CAMBRAY_RIDGE_CEILING, color='#B71C1C', lw=0.8, ls='--',
             alpha=0.7, zorder=1)
ax_d.text(CAMBRAY_RIDGE_CEILING + 0.005, y_cam[-1] + 0.5,
          f'Cambray-internal Ridge ceiling ({CAMBRAY_RIDGE_CEILING:.3f})',
          fontsize=5.5, color='#B71C1C', ha='left', va='top', style='italic')

# Matplotlib renders backslash-underscore as literal "\_"; use plain
# underscores in the y-axis labels (no LaTeX escape needed).
clean_labels = [lbl.replace('\\_', '_') for lbl in labels_cam]
ax_d.set_yticks(y_cam)
ax_d.set_yticklabels(clean_labels, fontsize=5.6)
ax_d.invert_yaxis()
ax_d.set_xlim(0, 0.95)  # wider to fit group badges on the right
ax_d.set_xlabel(r'Spearman $\rho$ (Cambray 2018, N = 24,200)', fontsize=7)
ax_d.set_title('Cambray: classical features transfer; FM readouts do not',
                fontsize=8, fontweight='bold', color=INK, pad=6, loc='left')
# Moved source footnote to bottom-right of figure (outside axes) to
# avoid clashing with the x-axis label.
ax_d.text(1.0, -0.24,
          'All values: 5-fold XP5 ensemble, trained on 492K (Phase 5 & R1/R2, XP5Ensemble, Rule 17)',
          transform=ax_d.transAxes, fontsize=5.2, ha='right', va='top',
          color='#777', style='italic')


# Panel e (bespoke recipe schematic) removed 2026-04-13 after user review.
# The "recipe" was redundant with the Cambray example already in panel d
# and the example's ρ=0.434 was not spectacular enough to warrant its own
# sub-panel. Panel d now spans gs[2, :] (full row width).


# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
# IMPORTANT: write to BOTH the tex-included name (no suffix) and the
# versioned name. The .tex includes `fig_comprehensive_external.pdf`,
# so the unsuffixed file is what the manuscript actually uses. The
# versioned copy preserves the v3 history alongside v1/v2.
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'fig_comprehensive_external.{ext}')
    fig.savefig(OUT / f'fig_comprehensive_external_v3.{ext}')
    print(f'Saved: fig_comprehensive_external.{ext} (and _v3 copy)')
plt.close(fig)
