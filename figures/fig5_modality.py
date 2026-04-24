#!/usr/bin/env python3
"""
Fig 5 v5: Modality complementarity (3 panels).
  a) CKA heatmap (5 modalities including biophysical, N=20K)
  b) LOO importance reshuffling (GO vs SC, legend top-left)
  c) Deployment tiers (corrected: no duplicate values)

Standalone script — does NOT overwrite other figures.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'results' / 'nonmega_champion_campaign_hardhybrid'
EMBED_DIR = ROOT / 'datasets' / 'protex_aggregated' / 'embeddings_finalized'
OUT = ROOT / 'docs' / 'protex' / 'latex' / 'figures' / 'nbt'
OUT.mkdir(parents=True, exist_ok=True)

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

BG = '#FAFBFC'
INK = '#2C3E50'
MOD_COLORS = {
    'protein': '#1f77b4', 'dna': '#2ca02c', 'cds': '#17becf',
    'genome_context': '#9467bd', 'biophysical': '#ff7f0e',
}


def extract_rho(d, key='rho_overall'):
    if 'results' in d and 'single_adapter' in d['results']:
        sa = d['results']['single_adapter']
        if key in sa and sa[key] is not None:
            return sa[key]
    return d.get(key)


def load_json(name):
    with open(CAMPAIGN / name) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Panel a: CKA (5 modalities, N=20K, proper array unpacking)
# ═══════════════════════════════════════════════════════════════════════════
def compute_cka_matrix():
    print('Panel a: Computing CKA (N=20,000 genes)...')
    nm5_embeds = {
        'Genome': ('bacformer_large_embeddings.parquet', 'bacformer_embedding'),
        'Operon': ('evo2_7b_full_operon_pca4096.parquet', 'evo2_7b_full_operon_pca4096_embedding'),
        'CDS': ('hyenadna_dna_cds_embeddings.parquet', 'hyenadna_dna_cds_embedding'),
        'Protein': ('esmc_protein_embeddings.parquet', 'esmc_protein_embedding'),
        'Biophys.': None,
    }
    classical_files = [
        'classical_codon_features.parquet', 'classical_rna_thermo_features.parquet',
        'classical_protein_features.parquet', 'classical_disorder_features.parquet',
        'classical_operon_structural_features.parquet',
    ]

    N = 20000
    np.random.seed(42)
    ref = pd.read_parquet(EMBED_DIR / 'hyenadna_dna_cds_embeddings.parquet', columns=['gene_id'])
    sample_ids = ref['gene_id'].sample(N, random_state=42).values
    id_set = set(sample_ids)

    matrices, gene_ids = {}, {}
    for name, info in nm5_embeds.items():
        if info is not None:
            fname, emb_col = info
            edf = pd.read_parquet(EMBED_DIR / fname)
            edf = edf[edf['gene_id'].isin(id_set)].set_index('gene_id')
            matrices[name] = np.stack(edf[emb_col].values)
            gene_ids[name] = edf.index.values
        else:
            parts = []
            for cf in classical_files:
                p = pd.read_parquet(EMBED_DIR / cf)
                p = p[p['gene_id'].isin(id_set)].set_index('gene_id')
                parts.append(p.select_dtypes(include=[np.number]))
            combined = pd.concat(parts, axis=1)
            for col in combined.columns:
                if combined[col].isna().any():
                    combined[col] = combined[col].fillna(combined[col].median())
            matrices[name] = combined.values
            gene_ids[name] = combined.index.values

    # Align to common gene set
    common = set(gene_ids[list(gene_ids.keys())[0]])
    for k in gene_ids:
        common &= set(gene_ids[k])
    common = sorted(common)
    print(f'  Common genes: {len(common)}')

    aligned = {}
    for name in matrices:
        idx = {gid: i for i, gid in enumerate(gene_ids[name])}
        aligned[name] = matrices[name][[idx[g] for g in common]]

    def linear_cka(X, Y):
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        denom = np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
        if denom < 1e-12:
            return 0.0
        return np.linalg.norm(X.T @ Y, 'fro') ** 2 / denom

    names = list(aligned.keys())
    n = len(names)
    cka = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka[i, j] = linear_cka(aligned[names[i]], aligned[names[j]])
    return names, cka, len(common)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Load LOO data ──
    base_d = load_json('A_champion_go_seed42.json')
    base_nc = extract_rho(base_d, 'rho_non_mega')

    base_sc_d = load_json('B_champion_sc020_seed42.json')
    base_sc_ov = extract_rho(base_sc_d, 'rho_overall')
    base_sc_nc = extract_rho(base_sc_d, 'rho_non_mega')

    mods = {
        'evo2_7b': ('Operon context\n(Evo-2 7B)', 'dna'),
        'biophysical': ('Biophysical\n(per-gene, 69d)', 'biophysical'),
        'bacformer_large': ('Genome context\n(Bacformer-lg)', 'genome_context'),
        'esmc': ('Protein identity\n(ESM-C)', 'protein'),
        'hyenadna': ('CDS composition\n(HyenaDNA)', 'cds'),
    }

    # 23-cluster LOSCO paired-test corrections (§A.SC_LOO_23CLUSTER, 2026-04-13).
    # The single-partition SC values for CDS and genome-context were refuted
    # (or are pending) by the full 23-cluster campaign. Use the corrected mean
    # where available; flag which are single-partition vs multi-partition.
    sc_nc_override = {
        'hyenadna': -0.008,   # 23-cluster paired mean, p=0.00005 (corrected; single-part was +0.006)
    }
    sc_nc_source = {
        'hyenadna': '23-cluster',  # others: single-partition
    }

    loo_data = []
    for mod_key, (label, family) in mods.items():
        go_d = load_json(f'C_loo_go_drop_{mod_key}.json')
        sc_d = load_json(f'D_loo_sc_drop_{mod_key}.json')
        sc_nc_raw = extract_rho(sc_d, 'rho_non_mega') - base_sc_nc
        sc_nc_final = sc_nc_override.get(mod_key, sc_nc_raw)
        loo_data.append({
            'label': label, 'family': family,
            'go_nc': extract_rho(go_d, 'rho_non_mega') - base_nc,
            'sc_nc': sc_nc_final,
            'sc_source': sc_nc_source.get(mod_key, 'single-partition'),
        })
    loo_df = pd.DataFrame(loo_data)

    # ── CKA computation ──
    cka_names, cka_mat, n_genes = compute_cka_matrix()

    # ── Build figure (2 panels, 1 row — tier ladder moved to Fig 2d) ──
    fig = plt.figure(figsize=(6.4, 3.4), facecolor=BG)
    gs = gridspec.GridSpec(1, 2, wspace=0.85, left=0.09, right=0.97, top=0.86, bottom=0.18,
                           width_ratios=[1.0, 1.15])

    # ── Panel a: CKA heatmap ──
    ax_a = fig.add_subplot(gs[0, 0])

    import seaborn as sns
    cka_df = pd.DataFrame(cka_mat, index=cka_names, columns=cka_names)
    sns.heatmap(cka_df, ax=ax_a, vmin=0, vmax=1,
                cmap='Blues', linewidths=1, linecolor='white',
                annot=True, fmt='.2f', annot_kws={'fontsize': 5, 'fontweight': 'normal'},
                cbar_kws={'shrink': 0.6, 'label': 'Linear CKA', 'pad': 0.02})
    ax_a.set_title(f'Representational similarity\n(N={n_genes:,} genes)', fontsize=8, fontweight='bold', pad=14)
    ax_a.tick_params(axis='x', rotation=30, labelsize=6)
    ax_a.tick_params(axis='y', rotation=0, labelsize=6.5)

    # ── Panel b: LOO importance (paired bars, legend top-left) ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(BG)

    loo_sorted = loo_df.sort_values('go_nc')
    n_mods = len(loo_sorted)
    y = np.arange(n_mods)
    bar_h = 0.35

    for i, row in enumerate(loo_sorted.itertuples()):
        color = MOD_COLORS[row.family]
        # GO bar (dark)
        ax_b.barh(i + bar_h / 2, row.go_nc, height=bar_h, color=color, alpha=0.9,
                  edgecolor='white', linewidth=0.4, zorder=3)
        # SC bar (light) — now uses Δρ_nc (same metric as GO) with 23-cluster
        # correction for CDS (§A.SC_LOO_23CLUSTER).
        ax_b.barh(i - bar_h / 2, row.sc_nc, height=bar_h, color=color, alpha=0.45,
                  edgecolor='white', linewidth=0.4, zorder=3)
        # Value labels
        go_x = row.go_nc - 0.001 if row.go_nc < 0 else row.go_nc + 0.001
        sc_x = row.sc_nc - 0.001 if row.sc_nc < 0 else row.sc_nc + 0.001
        ax_b.text(go_x, i + bar_h / 2, f'{row.go_nc:+.3f}',
                  ha='right' if row.go_nc < 0 else 'left', va='center', fontsize=5,
                  color=color, fontweight='bold')
        sc_suffix = r'$^{\dagger}$' if row.sc_source == '23-cluster' else ''
        ax_b.text(sc_x, i - bar_h / 2, f'{row.sc_nc:+.3f}' + sc_suffix,
                  ha='right' if row.sc_nc < 0 else 'left', va='center', fontsize=5,
                  color=color, alpha=0.7)

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(loo_sorted['label'], fontsize=6)
    ax_b.axvline(0, color='#D7E0EA', linewidth=0.8, zorder=0)
    ax_b.set_xlabel(r'LOO importance ($\Delta\rho_\mathrm{nc}$)')
    ax_b.set_title('LOO importance under GO\n(SC values partition-dependent, see caption)', fontsize=8, fontweight='bold', pad=14)

    legend_b = [
        mpatches.Patch(color='#555', alpha=0.9, label='Gene-operon (GO)'),
        mpatches.Patch(color='#555', alpha=0.45, label='Species-cluster (SC)'),
    ]
    ax_b.legend(handles=legend_b, fontsize=5.5, loc='upper left', frameon=True,
                facecolor=BG, edgecolor='#D0D8E0')
    # Footnote for 23-cluster corrections
    ax_b.text(0.02, 0.02, r'$\dagger$ 23-cluster LOSCO paired mean',
              transform=ax_b.transAxes, fontsize=4.5, color='#888',
              style='italic', va='bottom')

    # ── Panel c: Deployment tiers (corrected values, no duplicates) ──
    # (The monotone deployment ladder — previously Fig 5c — is now Fig 2d.
    # It establishes the paper's core "more biological context → better
    # prediction" claim right next to the architecture diagram, so the
    # reader does not have to flip pages to see the proof.)

    # Panel letters placed with absolute figure coordinates (above titles)
    # to avoid overlapping title text.
    fig.text(0.030, 0.985, 'a', fontsize=13, fontweight='bold', ha='left', va='top')
    fig.text(0.530, 0.985, 'b', fontsize=13, fontweight='bold', ha='left', va='top')

    # Save
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'fig5_modality_structure.{ext}')
    plt.close(fig)
    print(f'Saved: fig5_modality_structure.pdf + .png')


if __name__ == '__main__':
    main()
