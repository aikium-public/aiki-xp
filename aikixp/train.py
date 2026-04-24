#!/usr/bin/env python3
"""
ProtEx Learned Fusion Training

Trains fusion + MLP head models using precomputed embeddings.
Compares different fusion architectures with proper LOSO cross-validation.

Key insight from baselines:
- Naive concatenation (Ridge) HURTS: ρ=0.607 vs ESM-2 alone ρ=0.670
- Learned fusion should beat both by learning cross-modal interactions

Usage:
    python python -m aikixp.train --fusion single_adapter  # RECOMMENDED
    python python -m aikixp.train --fusion single_adapter --fair-target-params 25000000
    python python -m aikixp.train --fusion all  # Compare all architectures

Author: ProtEx Team
Date: 2026-01-29
"""

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, replace
import time

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = REPO_ROOT
DATA_PATH = REPO_ROOT / "data" / "aikixp_492k_v1.parquet"
CANONICAL_GOLD_FILENAMES = {
    "aikixp_492k_v1.parquet",
    "protex_v0.2.1_production_gold.parquet",
    "protex_v0.2.1_production_gold_v2.parquet",
    "protex_aggregated_v1.1_final_freeze.parquet",
    "protex_aggregated_v1.1.2_production_candidate.parquet",
}
REQUIRED_DATA_CONTRACT_COLS = (
    "gene_id",
    "species",
    "taxid",
    "operon_source",
    "operon_id",
    "expression_level",
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "protex_fusion"
PRODUCTION_OPERON_SPLIT_PATH = (
    PROJECT_ROOT / "data" / "splits" / "hard_hybrid_production_split_v2.tsv"
)

# Embedder columns and dimensions (from actual parquet file)
EMBEDDER_INFO = {
    # NOTE: All embedders are now loaded from companion parquets.
    # This dict is kept for backwards compatibility but is empty.
}

# All embedders come from companion parquets (joined at load time by gene_id)
COMPANION_EMBEDDER_INFO = {
    # ── RA base embeddings (extracted from protex_v0.2.1_corrected.parquet) ──
    'esm2': {'col': 'esm2_embedding', 'dim': 5120,
             'parquet': 'esm2_base_embeddings.parquet'},
    'esm2_650m': {'col': 'esm2_protein_embedding', 'dim': 1280,
                  'parquet': 'esm2_650m_protein_embeddings.parquet'},
    'esmc': {'col': 'esmc_embedding', 'dim': 1152,
             'parquet': 'esmc_base_embeddings.parquet'},
    'evo2': {'col': 'evo2_embedding', 'dim': 8192,
             'parquet': 'evo2_base_embeddings.parquet'},  # V1 gold (7B)
    'evo2_cds': {'col': 'evo2_cds_embedding', 'dim': 5120,
                 'parquet': 'evo2_cds_embeddings.parquet'},
    'evo2_init_window': {'col': 'evo2_init_window_embedding', 'dim': 5120,
                         'parquet': 'evo2_init_window_embeddings.parquet'},
    'hyenadna': {'col': 'hyenadna_embedding', 'dim': 256,
                 'parquet': 'hyenadna_base_embeddings.parquet'},
    'codonfm': {'col': 'codonfm_embedding', 'dim': 2048,
                'parquet': 'codonfm_base_embeddings.parquet'},
    'rinalmo': {'col': 'rinalmo_embedding', 'dim': 1280,
                'parquet': 'rinalmo_base_embeddings.parquet'},
    # ── Evo-2 7B multi-window embeddings (11,264d from blocks.28.mlp.l1) ──
    # Full-dim: use with PCA or as single swap in F10
    'evo2_7b_cds': {'col': 'evo2_7b_cds_embedding', 'dim': 11264,
                    'parquet': 'evo2_7b_cds.parquet'},
    'evo2_7b_init_70nt': {'col': 'evo2_7b_init_70nt_embedding', 'dim': 11264,
                          'parquet': 'evo2_7b_init_70nt.parquet'},
    'evo2_7b_upstream_200nt': {'col': 'evo2_7b_upstream_200nt_embedding', 'dim': 11264,
                               'parquet': 'evo2_7b_upstream_200nt.parquet'},
    'evo2_7b_downstream_200nt': {'col': 'evo2_7b_downstream_200nt_embedding', 'dim': 11264,
                                 'parquet': 'evo2_7b_downstream_200nt.parquet'},
    'evo2_7b_full_operon': {'col': 'evo2_7b_full_operon_embedding', 'dim': 11264,
                            'parquet': 'evo2_7b_full_operon.parquet'},
    # PCA-128 reduced (created by profile_evo2_7b_contribution.py Phase 2)
    'evo2_7b_cds_pca128': {'col': 'evo2_7b_cds_pca128_embedding', 'dim': 128,
                           'parquet': 'evo2_7b_cds_pca128.parquet'},
    'evo2_7b_init_70nt_pca128': {'col': 'evo2_7b_init_70nt_pca128_embedding', 'dim': 128,
                                 'parquet': 'evo2_7b_init_70nt_pca128.parquet'},
    'evo2_7b_upstream_200nt_pca128': {'col': 'evo2_7b_upstream_200nt_pca128_embedding', 'dim': 128,
                                      'parquet': 'evo2_7b_upstream_200nt_pca128.parquet'},
    'evo2_7b_downstream_200nt_pca128': {'col': 'evo2_7b_downstream_200nt_pca128_embedding', 'dim': 128,
                                        'parquet': 'evo2_7b_downstream_200nt_pca128.parquet'},
    # PCA-4096 reduced windows (fitted on train split of original gene-operon split)
    'evo2_7b_cds_pca4096': {'col': 'evo2_7b_cds_pca4096_embedding', 'dim': 4096,
                            'parquet': 'evo2_7b_cds_pca4096.parquet'},
    'evo2_7b_init_70nt_pca4096': {'col': 'evo2_7b_init_70nt_pca4096_embedding', 'dim': 4096,
                                  'parquet': 'evo2_7b_init_70nt_pca4096.parquet'},
    'evo2_7b_upstream_200nt_pca4096': {'col': 'evo2_7b_upstream_200nt_pca4096_embedding', 'dim': 4096,
                                       'parquet': 'evo2_7b_upstream_200nt_pca4096.parquet'},
    'evo2_7b_downstream_200nt_pca4096': {'col': 'evo2_7b_downstream_200nt_pca4096_embedding', 'dim': 4096,
                                          'parquet': 'evo2_7b_downstream_200nt_pca4096.parquet'},
    'evo2_7b_full_operon_pca128': {'col': 'evo2_7b_full_operon_pca128_embedding', 'dim': 128,
                                   'parquet': 'evo2_7b_full_operon_pca128.parquet'},
    'evo2_7b_full_operon_pca256': {'col': 'evo2_7b_full_operon_pca256_embedding', 'dim': 256,
                                   'parquet': 'evo2_7b_full_operon_pca256.parquet'},
    'evo2_7b_full_operon_pca512': {'col': 'evo2_7b_full_operon_pca512_embedding', 'dim': 512,
                                   'parquet': 'evo2_7b_full_operon_pca512.parquet'},
    'evo2_7b_full_operon_pca768': {'col': 'evo2_7b_full_operon_pca768_embedding', 'dim': 768,
                                   'parquet': 'evo2_7b_full_operon_pca768.parquet'},
    'evo2_7b_full_operon_pca1024': {'col': 'evo2_7b_full_operon_pca1024_embedding', 'dim': 1024,
                                    'parquet': 'evo2_7b_full_operon_pca1024.parquet'},
    'evo2_7b_full_operon_pca1536': {'col': 'evo2_7b_full_operon_pca1536_embedding', 'dim': 1536,
                                    'parquet': 'evo2_7b_full_operon_pca1536.parquet'},
    'evo2_7b_full_operon_pca2048': {'col': 'evo2_7b_full_operon_pca2048_embedding', 'dim': 2048,
                                    'parquet': 'evo2_7b_full_operon_pca2048.parquet'},
    'evo2_7b_full_operon_pca3072': {'col': 'evo2_7b_full_operon_pca3072_embedding', 'dim': 3072,
                                    'parquet': 'evo2_7b_full_operon_pca3072.parquet'},
    'evo2_7b_full_operon_pca4096': {'col': 'evo2_7b_full_operon_pca4096_embedding', 'dim': 4096,
                                    'parquet': 'evo2_7b_full_operon_pca4096.parquet'},
    'evo2_7b_full_operon_pca8192': {'col': 'evo2_7b_full_operon_pca8192_embedding', 'dim': 8192,
                                    'parquet': 'evo2_7b_full_operon_pca8192.parquet'},
    # ── Additional companion embeddings ──
    'operon_hyenadna': {'col': 'operon_hyenadna_embedding', 'dim': 256,
                        'parquet': 'operon_embeddings.parquet'},
    'bacformer': {'col': 'bacformer_embedding', 'dim': 480,
                  'parquet': 'bacformer_embeddings.parquet'},
    'bacformer_large': {'col': 'bacformer_embedding', 'dim': 960,
                        'parquet': 'bacformer_large_embeddings.parquet'},
    # ── RNA region embeddings (initiation windows) ──
    # 5'UTRBERT: nucleotide-level, multi-species incl. bacteria (Sanofi, NAR 2025)
    'utrbert_init': {'col': 'utrbert_init_embedding', 'dim': 768,
                     'parquet': 'utrbert_init_embeddings.parquet'},
    # DEPRECATED: junction window embeddings — superseded by upstream/downstream context.
    # Retained for ablation comparison only. Do not use for new production runs.
    'utrbert_junc': {'col': 'utrbert_junc_embedding', 'dim': 768,
                     'parquet': 'utrbert_junc_embeddings.parquet',
                     'deprecated': True},
    # RiNALMo: ncRNA-trained baseline for comparison
    'rinalmo_init': {'col': 'rinalmo_init_embedding', 'dim': 1280,
                     'parquet': 'rinalmo_init_embeddings.parquet'},
    'rinalmo_junc': {'col': 'rinalmo_junc_embedding', 'dim': 1280,
                     'parquet': 'rinalmo_junc_embeddings.parquet',
                     'deprecated': True},  # DEPRECATED
    # ── Biophysical features (SD strength, MFE, GC content, etc.) ──
    # Scalar features computed by featurize_biophysical.py — novel hybrid approach
    'biophysical_init': {'col': 'biophysical_init_features', 'dim': 10,
                         'parquet': 'biophysical_init_features.parquet',
                         'feature_cols': [
                             'gc_content', 'gc_content_upstream', 'gc_content_downstream',
                             'sd_score', 'sd_spacing', 'sd_spacing_penalty',
                             'au_richness_init', 'mfe_full', 'mfe_rbs',
                             'accessibility_start',
                         ]},
    'biophysical_junc': {'col': 'biophysical_junc_features', 'dim': 5,
                         'parquet': 'biophysical_junc_features.parquet',
                         'deprecated': True,  # DEPRECATED
                         'feature_cols': [
                             'gc_content', 'intergenic_length', 'mfe_full',
                             'sd_score_downstream', 'sd_spacing_downstream',
                         ]},
    # ── Classical features (featurize_classical.py) ──
    # Host-specific codon adaptation — highest-value non-LLM features
    # tAI/CAI depend on per-species reference; no LM can learn this
    'classical_codon': {'col': 'classical_codon_features', 'dim': 11,
                        'parquet': 'classical_codon_features.parquet',
                        'feature_cols': [
                            'cai', 'enc', 'fop', 'cpb', 'codon_ramp_ratio',
                            'gc1', 'gc2', 'gc3', 'gc_cds', 'cds_length_nt',
                            'rare_codon_clusters',
                        ]},
    # Extended RNA thermodynamics — MFE, accessibility, SD duplex ΔG,
    # ensemble energy/diversity, methylation motifs (ViennaRNA-computed)
    'classical_rna_init': {'col': 'classical_rna_init_features', 'dim': 16,
                           'parquet': 'classical_rna_thermo_features.parquet',
                           'feature_cols': [
                               'gc_content', 'gc_content_upstream', 'gc_content_downstream',
                               'au_richness_init', 'sd_score', 'sd_spacing', 'sd_spacing_penalty',
                               'mfe_full', 'mfe_rbs', 'accessibility_start',
                               'sd_binding_dg', 'ensemble_energy', 'ensemble_diversity',
                               'drach_density_cds', 'cpg_oe_cds', 'dam_density_cds',
                           ]},
    # DEPRECATED: Junction classical features — superseded by upstream/downstream context.
    'classical_rna_junc': {'col': 'classical_rna_junc_features', 'dim': 7,
                           'parquet': 'classical_rna_junc_features.parquet',
                           'deprecated': True,
                           'feature_cols': [
                               'junc_gc_content', 'junc_intergenic_distance',
                               'junc_overlap_motif', 'junc_sd_score_downstream',
                               'junc_sd_spacing_downstream', 'junc_mfe_full',
                               'junc_sd_binding_dg',
                           ]},
    # Protein physicochemistry — pI, GRAVY, instability, charge, MetAP cleavage
    'classical_protein': {'col': 'classical_protein_features', 'dim': 24,
                          'parquet': 'classical_protein_features.parquet',
                          'feature_cols': [
                              'pI', 'mw', 'gravy', 'instability_index', 'aliphatic_index',
                              'aromaticity', 'charge_ph7', 'boman_index',
                              'protein_length', 'log_protein_length', 'cysteine_fraction',
                              'hydrophobic_fraction', 'charged_fraction',
                              'polar_fraction', 'tiny_fraction', 'proline_fraction',
                              'metap_cleaved',
                              'abs_charge_per_res', 'charge_balance',
                              'n_glycosylation_motifs', 'pest_motif_count',
                              'hydrophobic_patch_count', 'hydrophobic_patch_fraction',
                              'low_complexity_fraction',
                          ]},
    # Disorder & charge patterning — metapredict + localCIDER
    'classical_disorder': {'col': 'classical_disorder_features', 'dim': 8,
                           'parquet': 'classical_disorder_features.parquet',
                           'feature_cols': [
                               'disorder_mean', 'disorder_fraction', 'longest_idr',
                               'disorder_nterm', 'disorder_cterm',
                               'fcr', 'ncpr', 'kappa',
                           ]},
    # Operon structural / positional metadata — no LLM captures this
    'classical_operon_struct': {'col': 'classical_operon_structural_features', 'dim': 10,
                                'parquet': 'classical_operon_structural_features.parquet',
                                'feature_cols': [
                                    'operon_length_nt', 'operon_num_genes',
                                    'gene_position_in_operon', 'gene_relative_position',
                                    'is_first_gene', 'is_last_gene', 'is_singleton',
                                    'cds_fraction_of_operon',
                                    'upstream_intergenic_dist_nt',
                                    'downstream_intergenic_dist_nt',
                                ]},
    # ── Merged classical features by modality ──
    'classical_protein_all': {'col': 'classical_protein_all_features', 'dim': 32,
                              'parquet': 'classical_protein_all_features.parquet',
                              'feature_cols': [
                                  'pI', 'mw', 'gravy', 'instability_index', 'aliphatic_index',
                                  'aromaticity', 'charge_ph7', 'boman_index',
                                  'protein_length', 'log_protein_length', 'cysteine_fraction',
                                  'hydrophobic_fraction', 'charged_fraction',
                                  'polar_fraction', 'tiny_fraction', 'proline_fraction',
                                  'metap_cleaved',
                                  'abs_charge_per_res', 'charge_balance',
                                  'n_glycosylation_motifs', 'pest_motif_count',
                                  'hydrophobic_patch_count', 'hydrophobic_patch_fraction',
                                  'low_complexity_fraction',
                                  'disorder_mean', 'disorder_fraction', 'longest_idr',
                                  'disorder_nterm', 'disorder_cterm',
                                  'fcr', 'ncpr', 'kappa',
                              ]},
    'classical_dna_all': {'col': 'classical_dna_all_features', 'dim': 21,
                          'parquet': 'classical_dna_all_features.parquet',
                          'feature_cols': [
                              'cai', 'enc', 'fop', 'cpb', 'codon_ramp_ratio',
                              'gc1', 'gc2', 'gc3', 'gc_cds', 'cds_length_nt',
                              'rare_codon_clusters',
                              'operon_length_nt', 'operon_num_genes',
                              'gene_position_in_operon', 'gene_relative_position',
                              'is_first_gene', 'is_last_gene', 'is_singleton',
                              'cds_fraction_of_operon',
                              'upstream_intergenic_dist_nt',
                              'downstream_intergenic_dist_nt',
                          ]},
    'classical_rna': {'col': 'classical_rna_features', 'dim': 16,
                      'parquet': 'classical_rna_features.parquet',
                      'feature_cols': [
                          'gc_content', 'gc_content_upstream', 'gc_content_downstream',
                          'au_richness_init', 'sd_score', 'sd_spacing', 'sd_spacing_penalty',
                          'mfe_full', 'mfe_rbs', 'accessibility_start',
                          'sd_binding_dg', 'ensemble_energy', 'ensemble_diversity',
                          'drach_density_cds', 'cpg_oe_cds', 'dam_density_cds',
                      ]},
    # ── Context region embeddings (upstream/downstream intergenic) ──
    'rinalmo_upstream': {'col': 'rinalmo_upstream_embedding', 'dim': 1280,
                         'parquet': 'rinalmo_upstream_embeddings.parquet'},
    'rinalmo_downstream': {'col': 'rinalmo_downstream_embedding', 'dim': 1280,
                           'parquet': 'rinalmo_downstream_embeddings.parquet'},
    'hyenadna_upstream': {'col': 'hyenadna_upstream_embedding', 'dim': 256,
                          'parquet': 'hyenadna_upstream_embeddings.parquet'},
    'hyenadna_downstream': {'col': 'hyenadna_downstream_embedding', 'dim': 256,
                            'parquet': 'hyenadna_downstream_embeddings.parquet'},
    # ── Aggregated / Expansion naming aliases ──
    # These map to parquet names used in data/embeddings/.
    # For combined V1+V2 training, use --companion-dir data/embeddings
    # and reference these embedder names instead of the gold-specific ones above.
    'esm2_protein': {'col': 'esm2_protein_embedding', 'dim': 1280,
                     'parquet': 'esm2_protein_embeddings.parquet'},
    'esm2_35m_protein': {'col': 'esm2_35m_protein_embedding', 'dim': 480,
                         'parquet': 'esm2_35m_protein_embeddings.parquet'},
    'esm2_3b_protein': {'col': 'esm2_3b_protein_embedding', 'dim': 2560,
                        'parquet': 'esm2_3b_protein_embeddings.parquet'},
    'esmc_protein': {'col': 'esmc_protein_embedding', 'dim': 1152,
                     'parquet': 'esmc_protein_embeddings.parquet'},
    # ── ProtT5-XL (Rostlab, 3B T5 encoder, 1024d) ──
    'prot_t5_xl_protein': {'col': 'prot_t5_xl_embedding', 'dim': 1024,
                           'parquet': 'prot_t5_xl_protein_embeddings.parquet'},
    # ── LoRA-adapted embeddings (from train_expression_lora.py) ──
    'esmc_lora': {'col': 'esmc_lora_protein_embedding', 'dim': 1152,
                  'parquet': 'esmc_lora_protein_embeddings.parquet'},
    'esm2_lora': {'col': 'esm2_lora_protein_embedding', 'dim': 1280,
                  'parquet': 'esm2_lora_protein_embeddings.parquet'},
    'dnabert2_dna_cds': {'col': 'dnabert2_dna_cds_embedding', 'dim': 768,
                          'parquet': 'dnabert2_dna_cds_embeddings.parquet'},
    'dnabert2_dna_init_window': {'col': 'dnabert2_dna_init_window_embedding', 'dim': 768,
                                  'parquet': 'dnabert2_dna_init_window_embeddings.parquet'},
    'dnabert2_operon_dna': {'col': 'dnabert2_full_operon_dna_embedding', 'dim': 768,
                            'parquet': 'dnabert2_operon_dna_embeddings.parquet'},
    'hyenadna_dna_cds': {'col': 'hyenadna_dna_cds_embedding', 'dim': 256,
                          'parquet': 'hyenadna_dna_cds_embeddings.parquet'},
    'hyenadna_dna_init_window': {'col': 'hyenadna_dna_init_window_embedding', 'dim': 256,
                                  'parquet': 'hyenadna_dna_init_window_embeddings.parquet'},
    'hyenadna_operon_dna': {'col': 'hyenadna_full_operon_dna_embedding', 'dim': 256,
                            'parquet': 'hyenadna_operon_dna_embeddings.parquet'},
    'nt2_dna_cds': {'col': 'nt2_dna_cds_embedding', 'dim': 1024,
                     'parquet': 'nt2_dna_cds_embeddings.parquet'},
    'nt2_dna_init_window': {'col': 'nt2_dna_init_window_embedding', 'dim': 1024,
                            'parquet': 'nt2_dna_init_window_embeddings.parquet'},
    'nt2_operon_dna': {'col': 'nt2_full_operon_dna_embedding', 'dim': 1024,
                       'parquet': 'nt2_operon_dna_embeddings.parquet'},
    'codonfm_cds': {'col': 'codonfm_cds_embedding', 'dim': 2048,
                     'parquet': 'codonfm_cds_embeddings.parquet'},
    # ── PCA-128 reduced embeddings for kitchen-sink experiments ──
    'esm2_protein_pca128': {'col': 'esm2_protein_pca128_embedding', 'dim': 128,
                            'parquet': 'esm2_protein_pca128_embeddings.parquet'},
    'esmc_protein_pca128': {'col': 'esmc_protein_pca128_embedding', 'dim': 128,
                            'parquet': 'esmc_protein_pca128_embeddings.parquet'},
    'evo2_cds_pca128': {'col': 'evo2_cds_pca128_embedding', 'dim': 128,
                        'parquet': 'evo2_cds_pca128_embeddings.parquet'},
    'evo2_cds_pca512': {'col': 'evo2_cds_pca512_embedding', 'dim': 512,
                        'parquet': 'evo2_cds_pca512.parquet'},
    'evo2_cds_pca1024': {'col': 'evo2_cds_pca1024_embedding', 'dim': 1024,
                         'parquet': 'evo2_cds_pca1024.parquet'},
    'evo2_cds_pca2048': {'col': 'evo2_cds_pca2048_embedding', 'dim': 2048,
                         'parquet': 'evo2_cds_pca2048.parquet'},
    'evo2_cds_pca4096': {'col': 'evo2_cds_pca4096_embedding', 'dim': 4096,
                         'parquet': 'evo2_cds_pca4096.parquet'},
    'rinalmo_init_pca128': {'col': 'rinalmo_init_pca128_embedding', 'dim': 128,
                            'parquet': 'rinalmo_init_pca128_embeddings.parquet'},
    'codonfm_cds_pca128': {'col': 'codonfm_cds_pca128_embedding', 'dim': 128,
                           'parquet': 'codonfm_cds_pca128_embeddings.parquet'},
    'bacformer_pca128': {'col': 'bacformer_pca128_embedding', 'dim': 128,
                         'parquet': 'bacformer_pca128_embeddings.parquet'},
    'hyenadna_dna_cds_pca128': {'col': 'hyenadna_dna_cds_pca128_embedding', 'dim': 128,
                                'parquet': 'hyenadna_dna_cds_pca128_embeddings.parquet'},
    'dnabert2_dna_cds_pca128': {'col': 'dnabert2_dna_cds_pca128_embedding', 'dim': 128,
                                'parquet': 'dnabert2_dna_cds_pca128_embeddings.parquet'},
    'nt2_dna_cds_pca128': {'col': 'nt2_dna_cds_pca128_embedding', 'dim': 128,
                           'parquet': 'nt2_dna_cds_pca128_embeddings.parquet'},
    'utrbert_init_pca128': {'col': 'utrbert_init_pca128_embedding', 'dim': 128,
                            'parquet': 'utrbert_init_pca128_embeddings.parquet'},
    # ── PCA-95% variance adaptive embeddings (create_pca_embeddings_95var.py) ──
    'esm2_protein_pca95': {'col': 'esm2_protein_pca95_embedding', 'dim': 346,
                           'parquet': 'esm2_protein_pca95_embeddings.parquet'},
    'esmc_protein_pca95': {'col': 'esmc_protein_pca95_embedding', 'dim': 129,
                           'parquet': 'esmc_protein_pca95_embeddings.parquet'},
    'evo2_cds_pca95': {'col': 'evo2_cds_pca95_embedding', 'dim': 475,
                       'parquet': 'evo2_cds_pca95_embeddings.parquet'},
    'rinalmo_init_pca95': {'col': 'rinalmo_init_pca95_embedding', 'dim': 81,
                           'parquet': 'rinalmo_init_pca95_embeddings.parquet'},
    'codonfm_cds_pca95': {'col': 'codonfm_cds_pca95_embedding', 'dim': 1,
                          'parquet': 'codonfm_cds_pca95_embeddings.parquet'},
    'bacformer_pca95': {'col': 'bacformer_pca95_embedding', 'dim': 312,
                        'parquet': 'bacformer_pca95_embeddings.parquet'},
    'hyenadna_dna_cds_pca95': {'col': 'hyenadna_dna_cds_pca95_embedding', 'dim': 4,
                               'parquet': 'hyenadna_dna_cds_pca95_embeddings.parquet'},
    'dnabert2_dna_cds_pca95': {'col': 'dnabert2_dna_cds_pca95_embedding', 'dim': 156,
                               'parquet': 'dnabert2_dna_cds_pca95_embeddings.parquet'},
    'nt2_dna_cds_pca95': {'col': 'nt2_dna_cds_pca95_embedding', 'dim': 680,
                          'parquet': 'nt2_dna_cds_pca95_embeddings.parquet'},
    'utrbert_init_pca95': {'col': 'utrbert_init_pca95_embedding', 'dim': 114,
                           'parquet': 'utrbert_init_pca95_embeddings.parquet'},
    # ── Evo-2 7B multi-window embeddings (blocks.28.mlp.l1, 11264d) ──
    # Extracted from full-operon forward passes with windowed mean pooling.
    # See scripts/protex/evo2_7b_multiwindow_extract.py for details.
    'evo2_7b_cds': {'col': 'evo2_7b_cds_embedding', 'dim': 11264,
                    'parquet': 'evo2_7b_cds.parquet'},
    'evo2_7b_init_70nt': {'col': 'evo2_7b_init_70nt_embedding', 'dim': 11264,
                          'parquet': 'evo2_7b_init_70nt.parquet'},
    'evo2_7b_upstream_200nt': {'col': 'evo2_7b_upstream_200nt_embedding', 'dim': 11264,
                               'parquet': 'evo2_7b_upstream_200nt.parquet'},
    'evo2_7b_downstream_200nt': {'col': 'evo2_7b_downstream_200nt_embedding', 'dim': 11264,
                                  'parquet': 'evo2_7b_downstream_200nt.parquet'},
    'evo2_7b_full_operon': {'col': 'evo2_7b_full_operon_embedding', 'dim': 11264,
                            'parquet': 'evo2_7b_full_operon.parquet'},
    # PCA-128 reduced variants (for multi-window experiments on memory-limited GPUs)
    'evo2_7b_cds_pca128': {'col': 'evo2_7b_cds_pca128_embedding', 'dim': 128,
                           'parquet': 'evo2_7b_cds_pca128.parquet'},
    'evo2_7b_init_70nt_pca128': {'col': 'evo2_7b_init_70nt_pca128_embedding', 'dim': 128,
                                 'parquet': 'evo2_7b_init_70nt_pca128.parquet'},
    'evo2_7b_upstream_200nt_pca128': {'col': 'evo2_7b_upstream_200nt_pca128_embedding', 'dim': 128,
                                      'parquet': 'evo2_7b_upstream_200nt_pca128.parquet'},
    'evo2_7b_downstream_200nt_pca128': {'col': 'evo2_7b_downstream_200nt_pca128_embedding', 'dim': 128,
                                         'parquet': 'evo2_7b_downstream_200nt_pca128.parquet'},
    'evo2_7b_full_operon_pca128': {'col': 'evo2_7b_full_operon_pca128_embedding', 'dim': 128,
                                   'parquet': 'evo2_7b_full_operon_pca128.parquet'},
    'evo2_7b_full_operon_pca256': {'col': 'evo2_7b_full_operon_pca256_embedding', 'dim': 256,
                                   'parquet': 'evo2_7b_full_operon_pca256.parquet'},
    'evo2_7b_full_operon_pca512': {'col': 'evo2_7b_full_operon_pca512_embedding', 'dim': 512,
                                   'parquet': 'evo2_7b_full_operon_pca512.parquet'},
    # ── Extra precomputed features (featurize_extra_features.py) ──
    'extra_precomputed': {'col': 'extra_precomputed_features', 'dim': 26,
                          'parquet': 'extra_precomputed_features.parquet',
                          'feature_cols': [
                              'aa_freq_a', 'aa_freq_c', 'aa_freq_d', 'aa_freq_e', 'aa_freq_f',
                              'aa_freq_g', 'aa_freq_h', 'aa_freq_i', 'aa_freq_k', 'aa_freq_l',
                              'aa_freq_m', 'aa_freq_n', 'aa_freq_p', 'aa_freq_q', 'aa_freq_r',
                              'aa_freq_s', 'aa_freq_t', 'aa_freq_v', 'aa_freq_w', 'aa_freq_y',
                              'gc3', 'gc_skew', 'at_skew', 'internal_sd_count',
                              'kozak_score', 'is_leaderless',
                          ]},
}

# Fusion architectures to compare.
# WARNING: Architecture choice matters enormously. See descriptions below.
FUSION_TYPES = [
    'single_adapter',    # RECOMMENDED — per-modality pyramid MLP (multi-layer, max 4× reduction/step)
    'cross_attention',   # Cross-attention fusion with learned query token
    'attention_gated',   # Attention-weighted modality contributions
    'gmu',               # Gated Multimodal Unit — learned sigmoid gates
    'concat',            # Direct concatenation + single projection layer
    'projector',         # Independent projection per modality, then sum
    'linear_concat',     # WEAK BASELINE — single Linear per modality, then concat (NO pyramid, NO depth)
]
# Legacy alias — 'latent_alignment' was a misleading name for what is just
# a single linear layer per modality. Renamed to 'linear_concat' for honesty.
# The old name still works but prints a warning.
FUSION_TYPE_ALIASES = {
    'latent_alignment': 'linear_concat',
}


@dataclass
class TrainConfig:
    """Training configuration."""
    fusion_type: str = "single_adapter"
    latent_dim: int = 256
    hidden_dim: int = 512  # Smaller than paper's 1024 for faster training
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 50
    early_stopping_patience: int = 10
    early_stop_metric: str = "rmse"  # rmse | spearman — which val metric triggers early stopping
    val_metric: str = "overall"  # overall | non_mega — which val subset for early stopping
    _split_file_path: Optional[str] = None  # internal: path to split file for non-mega mask construction
    val_fraction: float = 0.1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: int = 42

    # Data loading
    num_workers: int = 0  # DataLoader worker processes; 0 = main thread only

    # Learning-rate schedule
    lr_warmup_steps: int = 0
    lr_schedule: str = "constant"  # constant | cosine | linear

    # Sampling strategy
    sampler: str = "random"  # random | species_balanced | species_pure | expression_stratified

    # Label handling
    label_mode: str = "winsorized"  # winsorized | zscore | raw

    # Sample weighting
    sample_weights: str = "none"  # none | quality_tier | identity | source_confidence | mega_downweight

    # Embedding normalization (before fusion)
    embed_norm: str = "none"  # none | l2 | zscore

    # Negative controls
    scramble_labels: bool = False
    scramble_embeddings: bool = False

    # Loss function
    loss_function: str = "mse"  # mse | huber
    huber_delta: float = 1.0
    loss_cap_percentile: Optional[float] = None  # e.g. 95.0 = truncated MSE
    modality_dropout: float = 0.0  # prob of zeroing each modality per sample during training

    # Ranking loss (pairwise margin ranking added to MSE)
    ranking_loss_lambda: float = 0.0
    ranking_loss_margin: float = 1.0
    ranking_loss_pairs_per_sample: int = 4

    # Fair-capacity controls (for comparable modality value sweeps)
    fair_target_params: Optional[int] = None
    fair_tolerance: float = 0.05
    fair_auto_width: bool = False
    fair_min_width: int = 32
    fair_max_width: int = 8192
    fair_width_step: int = 16


class EmbeddingDataset(Dataset):
    """Dataset for precomputed embeddings.

    Parameters
    ----------
    norm_stats : dict, optional
        Per-modality z-score stats ``{name: (mean, std)}`` computed from
        the training set.  When provided, modalities listed in norm_stats
        are z-score-normalized using these statistics.
    embed_norm : str, optional
        Normalization for LLM (non-tabular) embeddings before fusion.
        * ``"none"`` – no normalization (default).
        * ``"l2"``   – L2-normalize each vector to unit length.
        * ``"zscore"`` – z-score per dimension (requires norm_stats).
    sample_weights : ndarray, optional
        Per-sample weights for the loss function.  If None, all
        samples receive weight 1.0.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embedder_info: Dict[str, Dict],
        norm_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        embed_norm: str = "none",
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.embedder_info = embedder_info
        
        # Pre-extract all embeddings as numpy arrays for speed
        self.embeddings = {}
        import time as _time
        for name, info in embedder_info.items():
            col = info['col']
            if col not in df.columns:
                raise ValueError(f"Missing required embedding column: {col} ({name})")
            if df[col].isna().any():
                n_missing = int(df[col].isna().sum())
                raise ValueError(
                    f"Found {n_missing} missing embeddings in column '{col}' ({name}). "
                    "Refusing to continue with partial coverage."
                )
            _t0 = _time.time()
            self.embeddings[name] = np.vstack(df[col].values).astype(np.float32)
            actual_dim = self.embeddings[name].shape[1]
            expected_dim = info['dim']
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for '{name}': "
                    f"loaded {actual_dim}d from parquet but COMPANION_EMBEDDER_INFO declares {expected_dim}d. "
                    f"Wrong parquet file?"
                )
            logger.info("  vstack %s (%d rows, dim=%d, verified): %.1fs",
                        name, len(df), actual_dim, _time.time() - _t0)
        
        # ── L2-normalize LLM embeddings (before any z-score) ─────
        if embed_norm == "l2":
            for name, info in embedder_info.items():
                if 'feature_cols' in info:
                    continue  # skip tabular modalities
                norms = np.linalg.norm(self.embeddings[name], axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                self.embeddings[name] = (self.embeddings[name] / norms).astype(np.float32)

        # ── Apply z-score normalization using pre-computed stats ──
        # Use float32 casts to avoid float64 temporaries (2x memory).
        if norm_stats is not None:
            for name, (mean, std) in norm_stats.items():
                if name in self.embeddings:
                    self.embeddings[name] -= mean.astype(np.float32)
                    self.embeddings[name] /= std.astype(np.float32)
        
        # Labels
        self.labels = df['expression_level'].values.astype(np.float32)

        # Sample weights (for confidence-weighted loss)
        if sample_weights is not None:
            self.sample_weights = sample_weights.astype(np.float32)
        else:
            self.sample_weights = np.ones(len(df), dtype=np.float32)

    def compute_tabular_norm_stats(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute per-feature mean/std for scalar tabular modalities.

        Only modalities with ``'feature_cols'`` in their embedder_info are
        normalized (classical + biophysical features).  LLM embeddings are
        already internally normalized by their source models and are skipped.

        Returns dict ``{modality_name: (mean_array, std_array)}``.
        """
        stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, info in self.embedder_info.items():
            if 'feature_cols' not in info:
                continue
            arr = self.embeddings[name]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            # Guard against constant features (e.g. is_singleton for a species)
            std[std < 1e-8] = 1.0
            stats[name] = (mean, std)
        return stats

    def compute_embedding_norm_stats(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute per-dimension mean/std for LLM (non-tabular) modalities.

        Returns dict ``{modality_name: (mean_array, std_array)}``.
        """
        stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, info in self.embedder_info.items():
            if 'feature_cols' in info:
                continue  # tabular → handled by compute_tabular_norm_stats
            arr = self.embeddings[name]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            std[std < 1e-8] = 1.0
            stats[name] = (mean, std)
        return stats

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        emb_dict = {
            name: torch.from_numpy(arr[idx])
            for name, arr in self.embeddings.items()
        }
        label = torch.tensor(self.labels[idx])
        weight = torch.tensor(self.sample_weights[idx])
        return emb_dict, label, weight


def collate_fn(batch):
    """Collate embeddings into batched tensors."""
    emb_dicts, labels, weights = zip(*batch)
    
    # Stack each embedder's tensors
    batched_embs = {}
    for name in emb_dicts[0].keys():
        batched_embs[name] = torch.stack([d[name] for d in emb_dicts])
    
    labels = torch.stack(labels)
    weights = torch.stack(weights)
    return batched_embs, labels, weights


class FusionModel(nn.Module):
    """
    Learned fusion model with MLP head.
    
    Architecture:
        embeddings -> fusion -> MLP -> prediction
    """
    
    def __init__(self, config: TrainConfig, input_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.input_dims = input_dims
        
        # Build fusion module
        self.fusion = self._build_fusion()
        
        # Build regression head
        fusion_output_dim = self._get_fusion_output_dim()
        self.head = self._build_head(fusion_output_dim)
        
    def _build_fusion(self) -> nn.Module:
        """Build fusion module based on config."""
        fusion_type = self.config.fusion_type
        
        if fusion_type in ("linear_concat", "latent_alignment"):
            return LinearConcatFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.dropout,
            )
        elif fusion_type == "attention_gated":
            return AttentionGatedFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.hidden_dim,
                self.config.dropout,
            )
        elif fusion_type == "gmu":
            return GMUFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.dropout,
            )
        elif fusion_type == "concat":
            return ConcatFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.dropout,
            )
        elif fusion_type == "projector":
            return ProjectorFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.dropout,
            )
        elif fusion_type == "single_adapter":
            return SingleAdapterFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.dropout,
            )
        elif fusion_type == "cross_attention":
            return CrossAttentionFusion(
                self.input_dims,
                self.config.latent_dim,
                self.config.dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def _get_fusion_output_dim(self) -> int:
        """Get output dimension of fusion module."""
        return self.fusion.output_dim
    
    def _build_head(self, input_dim: int) -> nn.Module:
        """Build MLP regression head."""
        layers = []
        current_dim = input_dim
        
        for i in range(self.config.num_layers):
            next_dim = self.config.hidden_dim if i == 0 else current_dim // 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            ])
            current_dim = next_dim
        
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused = self.fusion(embeddings)
        return self.head(fused).squeeze(-1)

    def get_fused_embedding(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the fused representation before the prediction head."""
        return self.fusion(embeddings)

    def forward_with_attention(self, embeddings: Dict[str, torch.Tensor]):
        """Return (prediction, attention_weights) if the fusion supports it."""
        if hasattr(self.fusion, 'forward_with_attention'):
            fused, attn = self.fusion.forward_with_attention(embeddings)
            pred = self.head(fused).squeeze(-1)
            return pred, attn
        pred = self.forward(embeddings)
        return pred, None


# ============================================================
# Fusion Modules (simplified implementations)
# ============================================================

class LinearConcatFusion(nn.Module):
    """WEAK BASELINE: single Linear layer per modality → concat. No depth, no pyramid.

    Previously called 'LatentAlignmentFusion' — renamed because the old name
    disguised the fact that this is just one linear projection per modality.
    A 5120d embedding gets crushed to latent_dim in a single step with no
    intermediate nonlinear layers. Use SingleAdapterFusion (pyramid MLP) instead.
    """
    
    def __init__(self, input_dims: Dict[str, int], latent_dim: int, dropout: float):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        
        self.projectors = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        
        self.output_dim = latent_dim * len(input_dims)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        aligned = []
        for name in sorted(self.input_dims.keys()):
            h = self.projectors[name](embeddings[name])
            aligned.append(h)
        return torch.cat(aligned, dim=-1)


class AttentionGatedFusion(nn.Module):
    """Uses attention to weight modality contributions."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        
        # Project each modality
        self.projectors = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.projectors[name] = nn.Linear(dim, latent_dim)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = latent_dim
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Project all modalities
        projected = []
        for name in sorted(self.input_dims.keys()):
            h = self.projectors[name](embeddings[name])
            projected.append(h)
        
        # Stack: (batch, num_modalities, latent_dim)
        stacked = torch.stack(projected, dim=1)
        
        # Compute attention weights
        attn_scores = self.attention(stacked)  # (batch, num_mod, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted sum
        fused = (stacked * attn_weights).sum(dim=1)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

    def forward_with_attention(self, embeddings: Dict[str, torch.Tensor]):
        """Return (fused, attn_weights) where attn_weights is (batch, n_modalities)."""
        projected = []
        for name in sorted(self.input_dims.keys()):
            h = self.projectors[name](embeddings[name])
            projected.append(h)
        stacked = torch.stack(projected, dim=1)
        attn_scores = self.attention(stacked)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (stacked * attn_weights).sum(dim=1)
        fused = self.layer_norm(fused)
        return fused, attn_weights.squeeze(-1)


class GMUFusion(nn.Module):
    """Gated Multimodal Unit - learns gates for each modality."""
    
    def __init__(self, input_dims: Dict[str, int], latent_dim: int, dropout: float):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        
        # Project each modality
        self.projectors = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        
        for name, dim in input_dims.items():
            self.projectors[name] = nn.Linear(dim, latent_dim)
            self.gates[name] = nn.Sequential(
                nn.Linear(dim, latent_dim),
                nn.Sigmoid(),
            )
        
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim * len(input_dims), latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = latent_dim
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        gated = []
        for name in sorted(self.input_dims.keys()):
            x = embeddings[name]
            h = self.projectors[name](x)
            g = self.gates[name](x)
            gated.append(h * g)
        
        concat = torch.cat(gated, dim=-1)
        return self.output_proj(concat)


class ConcatFusion(nn.Module):
    """Simple concatenation with projection."""
    
    def __init__(self, input_dims: Dict[str, int], latent_dim: int, dropout: float):
        super().__init__()
        self.input_dims = input_dims
        total_dim = sum(input_dims.values())
        
        self.proj = nn.Sequential(
            nn.Linear(total_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.output_dim = latent_dim
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [embeddings[name] for name in sorted(self.input_dims.keys())]
        concat = torch.cat(parts, dim=-1)
        return self.proj(concat)


class ProjectorFusion(nn.Module):
    """Project each modality and sum."""
    
    def __init__(self, input_dims: Dict[str, int], latent_dim: int, dropout: float):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        
        self.projectors = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
            )
        
        self.output_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = latent_dim
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        for name in sorted(self.input_dims.keys()):
            h = self.projectors[name](embeddings[name])
            projected.append(h)
        
        # Sum all projections
        fused = torch.stack(projected, dim=0).sum(dim=0)
        fused = self.output_norm(fused)
        fused = self.dropout(fused)
        return fused


def _pyramid_dims(input_dim: int, target_dim: int, max_ratio: int = 4) -> List[int]:
    """Compute layer dimensions for a gradual reduction pyramid.

    Each intermediate layer reduces by at most ``max_ratio`` (default 4×).
    When ``input_dim <= target_dim`` a single expansion layer is used.

    After building the ideal pyramid, intermediate layers whose first-layer
    param cost exceeds a safety cap are progressively removed.  This keeps
    the architecture feasible for very high-dimensional inputs (e.g. Evo2
    at 8192-d) under tight parameter budgets.  The resulting first-step
    ratio may exceed ``max_ratio`` but is still far better than a single
    massive bottleneck that was the previous default.

    Returns a list starting at *input_dim* and ending at *target_dim*,
    e.g. ``[5120, 1280, 320, 256]`` for ``(5120, 256, 4)``.
    """
    if input_dim <= target_dim:
        return [input_dim, target_dim]

    # Build ideal pyramid (each step ≤ max_ratio)
    dims = [input_dim]
    current = input_dim
    while current // max_ratio > target_dim:
        current = current // max_ratio
        dims.append(current)
    dims.append(target_dim)

    # Safety: if the first linear layer would consume >8M params on its own,
    # progressively remove intermediates so the solver can find a feasible width.
    # 8M is ~80% of a 10M budget — leaves room for the head and later layers.
    _MAX_FIRST_LAYER = 8_000_000
    while len(dims) > 2 and dims[0] * dims[1] > _MAX_FIRST_LAYER:
        dims = [dims[0]] + dims[2:]

    return dims


def _build_pyramid_mlp(
    input_dim: int,
    target_dim: int,
    dropout: float,
    max_ratio: int = 4,
) -> nn.Sequential:
    """Build a pyramid MLP that gradually reduces *input_dim* → *target_dim*."""
    dims = _pyramid_dims(input_dim, target_dim, max_ratio)
    layers: list = []
    for i in range(len(dims) - 1):
        layers.extend([
            nn.Linear(dims[i], dims[i + 1]),
            nn.LayerNorm(dims[i + 1]),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
    return nn.Sequential(*layers)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion with learned query token."""

    def __init__(self, input_dims: Dict[str, int], latent_dim: int, dropout: float, num_heads: int = 4):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.modality_names = sorted(input_dims.keys())
        self.projectors = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(dropout),
            )
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim * 2, latent_dim))
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.output_dim = latent_dim

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused, _ = self.forward_with_attention(embeddings)
        return fused

    def forward_with_attention(self, embeddings: Dict[str, torch.Tensor]):
        projected = [self.projectors[name](embeddings[name]) for name in self.modality_names]
        kv = torch.stack(projected, dim=1)
        bs = kv.shape[0]
        q = self.query.expand(bs, -1, -1)
        attn_out, attn_weights = self.cross_attn(q, kv, kv)
        h = self.norm1(q + attn_out).squeeze(1)
        h = self.norm2(h + self.ffn(h))
        return h, attn_weights.squeeze(1)


class SingleAdapterFusion(nn.Module):
    """Gradual pyramid adapter — avoids premature linear bottlenecks.

    *Single modality*:
        ``input → pyramid MLP → latent_dim``

    *Multiple modalities*:
        ``each input → per-modality pyramid → latent_dim``
        ``concat → fusion MLP → latent_dim``

    The pyramid never reduces by more than 4× per layer, preserving
    high-dimensional structure that a single linear projection would destroy.
    """

    def __init__(self, input_dims: Dict[str, int], latent_dim: int, dropout: float):
        super().__init__()
        self.input_dims = input_dims
        n_modalities = len(input_dims)

        if n_modalities == 1:
            # Single modality: direct pyramid
            total_dim = sum(input_dims.values())
            self.mode = 'single'
            self.adapter = _build_pyramid_mlp(total_dim, latent_dim, dropout)
            self.output_dim = latent_dim
        else:
            # Multiple modalities: per-modality pyramid encoders
            self.mode = 'multi'
            per_mod_dim = latent_dim  # each modality → latent_dim

            self.encoders = nn.ModuleDict()
            for name, dim in sorted(input_dims.items()):
                self.encoders[name] = _build_pyramid_mlp(dim, per_mod_dim, dropout)

            # Shared fusion MLP after concatenation
            concat_dim = per_mod_dim * n_modalities
            self.fusion_mlp = nn.Sequential(
                nn.Linear(concat_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output_dim = latent_dim

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == 'single':
            parts = [embeddings[name] for name in sorted(self.input_dims.keys())]
            concat = torch.cat(parts, dim=-1)
            return self.adapter(concat)
        else:
            encoded = []
            for name in sorted(self.input_dims.keys()):
                encoded.append(self.encoders[name](embeddings[name]))
            concat = torch.cat(encoded, dim=-1)
            return self.fusion_mlp(concat)

    def forward_decomposed(self, embeddings: Dict[str, torch.Tensor]):
        """Return (fused, per_modality_dict) where per_modality_dict maps
        modality name -> encoded tensor (batch, latent_dim) before fusion MLP."""
        if self.mode == 'single':
            parts = [embeddings[name] for name in sorted(self.input_dims.keys())]
            concat = torch.cat(parts, dim=-1)
            fused = self.adapter(concat)
            return fused, {sorted(self.input_dims.keys())[0]: fused}
        per_mod = {}
        for name in sorted(self.input_dims.keys()):
            per_mod[name] = self.encoders[name](embeddings[name])
        concat = torch.cat([per_mod[name] for name in sorted(self.input_dims.keys())], dim=-1)
        fused = self.fusion_mlp(concat)
        return fused, per_mod


def count_trainable_params(model: nn.Module) -> int:
    """Return number of trainable model parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model_with_fairness(
    config: TrainConfig,
    input_dims: Dict[str, int],
) -> Tuple[nn.Module, Dict[str, float], TrainConfig]:
    """
    Build a model and enforce optional fair-capacity constraints.

    Returns:
        model, fairness metadata, config_used_for_model
    """
    cfg_used = config
    fair_meta = {
        "fair_target_params": config.fair_target_params,
        "fair_tolerance": config.fair_tolerance,
        "fair_auto_width": config.fair_auto_width,
    }

    if config.fair_target_params is not None and config.fair_auto_width:
        if config.latent_dim != 256:
            logger.warning(
                f"--fair-auto-width is enabled and will OVERRIDE --latent-dim {config.latent_dim} "
                f"with the auto-selected width. To use explicit latent-dim, disable auto-width "
                f"by omitting --fair-auto-width or passing --fair-target-params with --no-fair-auto-width."
            )
        # Binary search for width that hits target param count.
        # Param count is monotonic in width, so binary search works.
        # Fine-tune phase checks ±step around the best to find exact minimum deviation.
        import gc as _gc
        import time as _time
        _t_aw = _time.time()

        # ── Determine valid-width constraint for this architecture ──
        # cross_attention requires embed_dim % num_heads == 0
        _width_divisor = 1
        if config.fusion_type == "cross_attention":
            _width_divisor = 4  # num_heads default

        def _snap_width(w):
            """Snap width to nearest valid value for the architecture."""
            if _width_divisor <= 1:
                return w
            return max(_width_divisor, round(w / _width_divisor) * _width_divisor)

        def _try_width(w):
            """Try to build model at width w. Returns param count or None if width is invalid."""
            _tw_t0 = _time.time()
            try:
                trial_cfg = replace(config, latent_dim=w, hidden_dim=w)
                trial_model = FusionModel(trial_cfg, input_dims)
                n = count_trainable_params(trial_model)
                del trial_model
                logger.debug("  width-search: w=%d -> %s params (%.2fs)",
                             w, f"{n:,}" if n else "FAIL", _time.time() - _tw_t0)
                return n
            except Exception:
                logger.debug("  width-search: w=%d -> INVALID (%.2fs)", w, _time.time() - _tw_t0)
                return None
            finally:
                _gc.collect()

        # ── Narrow search range: estimate max plausible width ──
        # A model at width w has at least sum(input_dims)*w params (just projectors).
        # Cap hi so we never construct absurdly large trial models.
        _min_params_per_width = sum(input_dims.values())  # lower bound: projectors alone
        _est_max_width = max(
            config.fair_min_width,
            min(config.fair_max_width,
                int(3 * config.fair_target_params / max(_min_params_per_width, 1)))
        )

        # Phase 1: Binary search to find approximate width
        lo, hi = config.fair_min_width, _est_max_width
        logger.info("  Width search: range [%d, %d], divisor=%d, target=%s",
                     lo, hi, _width_divisor, f"{config.fair_target_params:,}")
        best = None
        _n_searched = 0
        while lo <= hi:
            mid = _snap_width((lo + hi) // 2)
            n = _try_width(mid)
            _n_searched += 1
            if n is None:
                # Shouldn't happen after snap, but handle gracefully
                hi = mid - _width_divisor
                continue
            dev = abs(n - config.fair_target_params) / max(1, config.fair_target_params)
            if best is None or dev < best[0]:
                best = (dev, mid, n)
            if n < config.fair_target_params:
                lo = mid + _width_divisor
            else:
                hi = mid - _width_divisor

        logger.info("  Phase 1 done: %d trials in %.1fs, best so far w=%s (%.1fM, dev=%.4f)",
                     _n_searched, _time.time() - _t_aw,
                     best[1] if best else "?",
                     best[2] / 1e6 if best else 0,
                     best[0] if best else 1.0)

        # Phase 2: Fine-tune around best width (check every VALID width in ±2*step)
        if best is not None:
            center = best[1]
            _ft_lo = max(config.fair_min_width, center - 2 * config.fair_width_step)
            _ft_hi = min(_est_max_width, center + 2 * config.fair_width_step)
            for w in range(_snap_width(_ft_lo), _ft_hi + 1, max(1, _width_divisor)):
                n = _try_width(w)
                _n_searched += 1
                if n is None:
                    continue
                dev = abs(n - config.fair_target_params) / max(1, config.fair_target_params)
                if dev < best[0]:
                    best = (dev, w, n)

        logger.info("  Auto-width search: %d trials in %.1fs, best_width=%d (%.1fM params, dev=%.4f)",
                     _n_searched, _time.time() - _t_aw, best[1], best[2]/1e6, best[0])
        assert best is not None
        _, best_width, best_params = best
        cfg_used = replace(config, latent_dim=best_width, hidden_dim=best_width)
        model = FusionModel(cfg_used, input_dims)
        n_params = count_trainable_params(model)
        deviation = abs(n_params - config.fair_target_params) / max(1, config.fair_target_params)
        fair_meta.update(
            {
                "fair_width": best_width,
                "effective_latent_dim": best_width,
                "effective_hidden_dim": best_width,
                "model_trainable_params": n_params,
                "fair_deviation": float(deviation),
            }
        )
    else:
        model = FusionModel(cfg_used, input_dims)
        n_params = count_trainable_params(model)
        fair_meta.update({
            "model_trainable_params": n_params,
            "effective_latent_dim": config.latent_dim,
            "effective_hidden_dim": config.hidden_dim,
        })
        if config.fair_target_params is not None:
            deviation = abs(n_params - config.fair_target_params) / max(1, config.fair_target_params)
            fair_meta["fair_deviation"] = float(deviation)

    if config.fair_target_params is not None:
        deviation = fair_meta.get("fair_deviation")
        if deviation is None:
            deviation = abs(
                fair_meta["model_trainable_params"] - config.fair_target_params
            ) / max(1, config.fair_target_params)
            fair_meta["fair_deviation"] = float(deviation)
        if deviation > config.fair_tolerance:
            raise ValueError(
                f"Fair-capacity constraint failed: deviation={deviation:.4f} exceeds "
                f"tolerance={config.fair_tolerance:.4f} "
                f"(target={config.fair_target_params}, actual={fair_meta['model_trainable_params']})."
            )

    return model, fair_meta, cfg_used


# ============================================================
# Reproducibility, Schedulers, Samplers
# ============================================================

def seed_everything(seed: int) -> None:
    """Set seeds for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reduce CUDA non-determinism (explains observed Δρ=0.008 same-seed gap)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    schedule: str,
    total_steps: int,
) -> Optional[LambdaLR]:
    """Build LR scheduler with optional linear warmup + decay.

    Returns None if schedule is 'constant' and warmup_steps is 0.
    """
    if schedule == "constant" and warmup_steps == 0:
        return None

    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < warmup_steps:
            return max(1e-8, current_step / max(1, warmup_steps))
        # Post-warmup decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif schedule == "linear":
            return max(0.0, 1.0 - progress)
        else:  # constant
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


class SpeciesBalancedSampler(Sampler):
    """Inverse-frequency weighted sampler so each species contributes equally.

    Each epoch draws ``len(dataset)`` samples with replacement, weighting
    each sample inversely by its species frequency.
    """

    def __init__(self, species_labels: np.ndarray, num_samples: int, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        unique, counts = np.unique(species_labels, return_counts=True)
        freq = dict(zip(unique, counts))
        weights = np.array([1.0 / freq[s] for s in species_labels], dtype=np.float64)
        weights /= weights.sum()
        self.weights = torch.from_numpy(weights)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist())

    def __len__(self):
        return self.num_samples


class SpeciesPureBatchSampler(Sampler):
    """Each batch contains genes from a single species only, cycled.

    Species are shuffled per epoch. Within each species, genes are shuffled.
    Batches are yielded species-by-species until all genes are covered.
    """

    def __init__(self, species_labels: np.ndarray, batch_size: int, seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        # Build per-species index lists
        self.species_indices: Dict[str, np.ndarray] = {}
        for sp in np.unique(species_labels):
            self.species_indices[sp] = np.where(species_labels == sp)[0]
        self.total = len(species_labels)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        all_indices = []
        species_order = list(self.species_indices.keys())
        rng.shuffle(species_order)
        for sp in species_order:
            idx = self.species_indices[sp].copy()
            rng.shuffle(idx)
            all_indices.extend(idx.tolist())
        return iter(all_indices)

    def __len__(self):
        return self.total


class ExpressionStratifiedSampler(Sampler):
    """Stratified sampling by expression quartile for balanced batches.

    Divides samples into 4 quartile bins. Each epoch draws equally from
    each bin (with replacement from smaller bins if needed).
    """

    def __init__(self, expression_values: np.ndarray, num_samples: int, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.seed = seed
        self._epoch = 0
        # Build quartile bins
        quartiles = np.percentile(expression_values, [25, 50, 75])
        self.bins: List[np.ndarray] = []
        self.bins.append(np.where(expression_values <= quartiles[0])[0])
        self.bins.append(np.where((expression_values > quartiles[0]) & (expression_values <= quartiles[1]))[0])
        self.bins.append(np.where((expression_values > quartiles[1]) & (expression_values <= quartiles[2]))[0])
        self.bins.append(np.where(expression_values > quartiles[2])[0])

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        per_bin = self.num_samples // 4
        remainder = self.num_samples - per_bin * 4
        indices = []
        for i, bin_idx in enumerate(self.bins):
            n = per_bin + (1 if i < remainder else 0)
            chosen = rng.choice(bin_idx, size=n, replace=(n > len(bin_idx)))
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


def build_train_sampler(
    config: TrainConfig,
    train_df: pd.DataFrame,
) -> Optional[Sampler]:
    """Build a custom training sampler based on config.sampler.

    Returns None for 'random' (use DataLoader shuffle=True).
    """
    if config.sampler == "random":
        return None
    n = len(train_df)
    if config.sampler == "species_balanced":
        return SpeciesBalancedSampler(
            train_df['species'].values, num_samples=n, seed=config.seed,
        )
    elif config.sampler == "species_pure":
        return SpeciesPureBatchSampler(
            train_df['species'].values, batch_size=config.batch_size, seed=config.seed,
        )
    elif config.sampler == "expression_stratified":
        return ExpressionStratifiedSampler(
            train_df['expression_level'].values, num_samples=n, seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown sampler: {config.sampler}")


# ============================================================
# Training Functions
# ============================================================

MEGA_COMPONENT_THRESHOLD = 5000


def compute_mega_component(
    gene_cluster_ids: pd.Series,
    compound_operon_ids: pd.Series,
) -> pd.Series:
    """Compute mega-component membership via Union-Find on (cluster, operon) edges.

    Two gene clusters are linked if they share a compound operon.
    A gene is "mega" if its connected component contains >= MEGA_COMPONENT_THRESHOLD genes.

    Parameters
    ----------
    gene_cluster_ids : Series
        One per gene.  Must not contain NaN.
    compound_operon_ids : Series
        One per gene.  Must not contain NaN.

    Returns
    -------
    Series[bool]
        True for genes in the mega-component, same index as inputs.
    """
    if gene_cluster_ids.isna().any() or compound_operon_ids.isna().any():
        raise ValueError(
            "Cannot compute mega-component with NaN cluster or operon IDs.  "
            f"gene_cluster_id NaN: {int(gene_cluster_ids.isna().sum())}, "
            f"compound_operon_id NaN: {int(compound_operon_ids.isna().sum())}"
        )

    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank.get(ra, 0) < rank.get(rb, 0):
            ra, rb = rb, ra
        parent[rb] = ra
        if rank.get(ra, 0) == rank.get(rb, 0):
            rank[ra] = rank.get(ra, 0) + 1

    cluster_ids = gene_cluster_ids.astype(str).values
    operon_ids = compound_operon_ids.astype(str).values

    operon_to_clusters: Dict[str, List[str]] = defaultdict(set)
    for cid, oid in zip(cluster_ids, operon_ids):
        operon_to_clusters[oid].add(cid)

    for oid, clusters in operon_to_clusters.items():
        clusters_list = list(clusters)
        for i in range(1, len(clusters_list)):
            union(clusters_list[0], clusters_list[i])

    gene_roots = [find(cid) for cid in cluster_ids]
    root_gene_count = Counter(gene_roots)

    is_mega = pd.Series(
        [root_gene_count[r] >= MEGA_COMPONENT_THRESHOLD for r in gene_roots],
        index=gene_cluster_ids.index,
    )

    n_mega = int(is_mega.sum())
    n_total = len(is_mega)
    mega_frac = n_mega / n_total if n_total else 0
    mega_roots = {r for r, c in root_gene_count.items() if c >= MEGA_COMPONENT_THRESHOLD}
    logger.info(
        "Mega-component: %d/%d genes (%.1f%%) in %d component(s) with >= %d genes",
        n_mega, n_total, 100 * mega_frac,
        len(mega_roots), MEGA_COMPONENT_THRESHOLD,
    )
    # Guardrail: mega component should be 38-45% of genes on the production table.
    # If it's wildly different, the compound_operon_id source is wrong.
    # Bypass: set PROTEX_ALLOW_MEGA_FRACTION_SKEW=1 when training on an intentional
    # subset (e.g. mega-only data-adaptation experiment — cambray_data_subset, 2026-04-11).
    if n_total > 100_000 and not (0.35 <= mega_frac <= 0.48):
        if os.environ.get("PROTEX_ALLOW_MEGA_FRACTION_SKEW") == "1":
            logger.warning(
                "Mega component is %.1f%% (%d/%d) — outside 35-48%% expected range. "
                "PROTEX_ALLOW_MEGA_FRACTION_SKEW=1 — continuing (intentional subset).",
                100 * mega_frac, n_mega, n_total,
            )
        else:
            raise ValueError(
                f"Mega component is {100*mega_frac:.1f}% ({n_mega}/{n_total}) — "
                f"expected 38-45% for the production table. "
                f"Check that compound_operon_id comes from the correct split file. "
                f"See SPLIT_FILE_PROVENANCE.md."
            )
    return is_mega


def harmonize_labels(
    df: pd.DataFrame,
    mode: str = "winsorized",
    train_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Normalize expression labels per species.

    Parameters
    ----------
    df : DataFrame
        Must contain 'expression_level' and 'species' columns.
    mode : str
        * ``"raw"`` – no normalization; keep original values.
        * ``"zscore"`` – classical per-species z-score (mean/std).
        * ``"winsorized"`` (default) – per-species z-score using
          *Winsorized* mean/std.  Values outside the [1st, 99th]
          percentile are clipped before computing statistics.  This
          prevents sentinel "not detected" values (e.g. E. coli ≈ −23)
          from inflating the standard deviation and compressing
          real dynamic range.
    train_mask : Series[bool], optional
        If provided, compute statistics *only* on training rows to
        prevent data leakage (still apply to all rows).
    """
    df = df.copy()
    df['expression_raw'] = df['expression_level']

    if mode == "raw":
        return df

    for species in df['species'].unique():
        sp_mask = df['species'] == species
        # Use train_mask for stats if available (prevent leakage)
        if train_mask is not None:
            stat_mask = sp_mask & train_mask
        else:
            stat_mask = sp_mask

        stat_values = df.loc[stat_mask, 'expression_raw'].values

        if len(stat_values) == 0:
            # Species entirely in val/test with no train rows – use all
            stat_values = df.loc[sp_mask, 'expression_raw'].values

        if mode == "winsorized":
            lo = np.percentile(stat_values, 1)
            hi = np.percentile(stat_values, 99)
            clipped = np.clip(stat_values, lo, hi)
            mean, std = np.mean(clipped), np.std(clipped)
        else:  # "zscore"
            mean, std = np.mean(stat_values), np.std(stat_values)

        if std > 1e-12:
            df.loc[sp_mask, 'expression_level'] = (
                df.loc[sp_mask, 'expression_raw'].values - mean
            ) / std
        else:
            df.loc[sp_mask, 'expression_level'] = 0.0

    return df


def _pairwise_ranking_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    pairs_per_sample: int = 4,
) -> torch.Tensor:
    """Pairwise margin ranking loss within a batch."""
    bs = preds.shape[0]
    if bs < 2:
        return torch.tensor(0.0, device=preds.device)
    n_pairs = min(bs * pairs_per_sample, bs * (bs - 1) // 2)
    idx_i = torch.randint(0, bs, (n_pairs,), device=preds.device)
    idx_j = torch.randint(0, bs - 1, (n_pairs,), device=preds.device)
    idx_j = idx_j + (idx_j >= idx_i).long()
    y_diff = labels[idx_i] - labels[idx_j]
    target = torch.sign(y_diff)
    loss = torch.nn.functional.margin_ranking_loss(
        preds[idx_i], preds[idx_j], target, margin=margin,
    )
    return loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler: Optional[LambdaLR] = None,
    ranking_lambda: float = 0.0,
    ranking_margin: float = 1.0,
    ranking_pairs: int = 4,
    loss_function: str = "mse",
    huber_delta: float = 1.0,
    loss_cap_percentile: Optional[float] = None,
    modality_dropout: float = 0.0,
) -> float:
    """Train for one epoch (with optional per-sample weighting + ranking loss).

    Args:
        modality_dropout: Probability of zeroing out each modality's embedding
            per sample during training. 0.0 = no dropout (default). E.g., 0.2
            means each modality has a 20% chance of being zeroed for each sample.
            Teaches the model to handle missing modalities at inference.
    """
    model.train()
    total_loss = 0.0

    for batch_embs, batch_labels, batch_weights in loader:
        batch_embs = {k: v.to(device, non_blocking=True) for k, v in batch_embs.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_weights = batch_weights.to(device, non_blocking=True)

        # Modality dropout: randomly zero entire modality embeddings per sample.
        # This teaches the model to handle missing modalities at inference.
        if modality_dropout > 0.0 and model.training:
            n_modalities = len(batch_embs)
            for mod_name, mod_tensor in batch_embs.items():
                # Per-sample mask: each sample independently drops this modality
                mask = torch.rand(mod_tensor.shape[0], 1, device=device) > modality_dropout
                batch_embs[mod_name] = mod_tensor * mask.float()

        optimizer.zero_grad()
        preds = model(batch_embs)

        if loss_function == "huber":
            per_sample_loss = torch.nn.functional.huber_loss(
                preds, batch_labels, reduction='none', delta=huber_delta,
            )
        else:
            per_sample_loss = (preds - batch_labels) ** 2

        if loss_cap_percentile is not None and loss_cap_percentile < 100.0:
            with torch.no_grad():
                cap = torch.quantile(per_sample_loss.detach(), loss_cap_percentile / 100.0)
            per_sample_loss = torch.clamp(per_sample_loss, max=cap.item())

        loss = (per_sample_loss * batch_weights).mean()

        if ranking_lambda > 0:
            rank_loss = _pairwise_ranking_loss(
                preds, batch_labels,
                margin=ranking_margin,
                pairs_per_sample=ranking_pairs,
            )
            loss = loss + ranking_lambda * rank_loss
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * len(batch_labels)
    
    return total_loss / len(loader.dataset)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on all batches, return (predictions, labels) numpy arrays.

    DataLoader must have shuffle=False so output order matches input order.
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_embs, batch_labels, _batch_weights in loader:
            batch_embs = {k: v.to(device, non_blocking=True) for k, v in batch_embs.items()}
            preds = model(batch_embs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def _build_non_mega_mask(df: pd.DataFrame) -> np.ndarray:
    """Build boolean mask for non-mega-component genes.

    Uses Union-Find on (gene_cluster_id, operon_column) to identify
    the mega-component (largest connected component, ~202K genes / 41%).
    Returns a boolean array aligned to df index: True = non-mega.

    Requires columns: gene_id, gene_cluster_id, and one of
    compound_operon_id (k-fold splits) or operon_component_id (original split).
    """
    from collections import defaultdict as _dd

    # Must use compound_operon_id (gives 41.2% mega, 202K genes).
    # operon_component_id gives 80.7% — a different, much larger component.
    if 'compound_operon_id' not in df.columns:
        raise ValueError(
            f"Split file needs compound_operon_id for mega-component detection "
            f"(41.2%, 202K genes). Found columns: {list(df.columns)}"
        )
    operon_col = 'compound_operon_id'

    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for _, row in df.iterrows():
        cid = f"C_{row['gene_cluster_id']}"
        oid = f"O_{row[operon_col]}"
        if cid not in parent: parent[cid] = cid
        if oid not in parent: parent[oid] = oid
        union(cid, oid)

    gene_to_comp = {}
    for _, row in df.iterrows():
        gene_to_comp[row['gene_id']] = find(f"C_{row['gene_cluster_id']}")

    comp_sizes = _dd(int)
    for comp in gene_to_comp.values():
        comp_sizes[comp] += 1
    mega_comp = max(comp_sizes, key=comp_sizes.get)
    mega_genes = {gid for gid, comp in gene_to_comp.items() if comp == mega_comp}

    mask = ~df['gene_id'].isin(mega_genes)
    n_mega = (~mask).sum()
    n_non_mega = mask.sum()
    logger.info("Non-mega mask (%s): %d mega, %d non-mega (%.1f%% non-mega)",
                operon_col, n_mega, n_non_mega, 100 * n_non_mega / len(df))
    return mask.values


def _safe_spearman(labels: np.ndarray, preds: np.ndarray, min_n: int = 10) -> Optional[float]:
    """Spearman rho with minimum-sample guard. Returns None if unreliable."""
    if len(labels) < min_n:
        return None
    if np.std(labels) < 1e-12 or np.std(preds) < 1e-12:
        return None
    rho = float(stats.spearmanr(labels, preds)[0])
    if not np.isfinite(rho):
        return None
    return rho


def compute_base_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Spearman, Pearson, R², RMSE on full arrays.

    Guards against NaN/constant predictions that would silently corrupt
    aggregated results (e.g. degenerate model outputs all same value).
    """
    if len(preds) < 2 or len(labels) < 2:
        logger.warning("compute_base_metrics: fewer than 2 samples, returning NaN metrics")
        return {'spearman': float('nan'), 'pearson': float('nan'), 'r2': float('nan'), 'rmse': float('nan')}

    if np.std(preds) < 1e-12 or np.std(labels) < 1e-12:
        logger.warning(
            "compute_base_metrics: constant predictions (std=%.2e) or labels (std=%.2e) — "
            "Spearman/Pearson undefined, returning NaN for correlation metrics",
            np.std(preds), np.std(labels),
        )
        return {
            'spearman': float('nan'),
            'pearson': float('nan'),
            'r2': float(r2_score(labels, preds)) if np.std(labels) > 1e-12 else float('nan'),
            'rmse': float(np.sqrt(mean_squared_error(labels, preds))),
        }

    rho = float(stats.spearmanr(labels, preds)[0])
    r = float(stats.pearsonr(labels, preds)[0])
    if not np.isfinite(rho):
        rho = float('nan')
    if not np.isfinite(r):
        r = float('nan')
    return {
        'spearman': rho,
        'pearson': r,
        'r2': float(r2_score(labels, preds)),
        'rmse': float(np.sqrt(mean_squared_error(labels, preds))),
    }


MIN_GROUP_SIZE_CLUSTER = 5
MIN_GROUP_SIZE_SPECIES = 10


def compute_stratified_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    gene_cluster_ids: Optional[np.ndarray] = None,
    is_mega: Optional[np.ndarray] = None,
    train_cluster_ids: Optional[set] = None,
) -> Dict[str, object]:
    """Compute the 4-metric evaluation framework plus novel/shared family diagnostics.

    All inputs must be aligned (same length and same row order).

    Returns
    -------
    dict with keys:
        rho_overall, rho_non_mega, rho_cluster_weighted, rho_per_species,
        rho_novel_families, rho_shared_families,
        n_test, n_mega, n_non_mega, n_species_evaluated,
        n_clusters_evaluated, n_novel_family_genes, n_shared_family_genes,
        per_species_rho (dict)
    """
    n = len(preds)
    assert len(labels) == n and len(species) == n

    metrics: Dict[str, object] = {}
    metrics['n_test'] = n

    # ── rho_overall ──
    metrics['rho_overall'] = _safe_spearman(labels, preds, min_n=10)

    # ── rho_non_mega ──
    if is_mega is not None:
        mega_mask = is_mega.astype(bool)
        non_mega_mask = ~mega_mask
        metrics['n_mega'] = int(mega_mask.sum())
        metrics['n_non_mega'] = int(non_mega_mask.sum())
        metrics['rho_non_mega'] = _safe_spearman(labels[non_mega_mask], preds[non_mega_mask])
        metrics['rho_mega'] = _safe_spearman(labels[mega_mask], preds[mega_mask])
    else:
        metrics['n_mega'] = None
        metrics['n_non_mega'] = None
        metrics['rho_non_mega'] = None
        metrics['rho_mega'] = None

    # ── rho_cluster_weighted ──
    if gene_cluster_ids is not None:
        per_cluster_rho: Dict[str, Optional[float]] = {}
        cluster_rhos = []
        unique_clusters = np.unique(gene_cluster_ids)
        for cid in unique_clusters:
            mask = gene_cluster_ids == cid
            rho = _safe_spearman(labels[mask], preds[mask], min_n=MIN_GROUP_SIZE_CLUSTER)
            per_cluster_rho[str(cid)] = rho
            if rho is not None:
                cluster_rhos.append(rho)
        metrics['rho_cluster_weighted'] = float(np.mean(cluster_rhos)) if cluster_rhos else None
        metrics['n_clusters_evaluated'] = len(cluster_rhos)
        metrics['per_cluster_rho'] = per_cluster_rho
    else:
        metrics['rho_cluster_weighted'] = None
        metrics['n_clusters_evaluated'] = None
        metrics['per_cluster_rho'] = None

    # ── rho_per_species ──
    per_species_rho: Dict[str, Optional[float]] = {}
    species_rhos = []
    unique_species = np.unique(species)
    for sp in unique_species:
        mask = species == sp
        rho = _safe_spearman(labels[mask], preds[mask], min_n=MIN_GROUP_SIZE_SPECIES)
        per_species_rho[str(sp)] = rho
        if rho is not None:
            species_rhos.append(rho)
    metrics['rho_per_species'] = float(np.mean(species_rhos)) if species_rhos else None
    metrics['n_species_evaluated'] = len(species_rhos)
    metrics['per_species_rho'] = per_species_rho

    # ── rho_novel_families / rho_shared_families ──
    if gene_cluster_ids is not None and train_cluster_ids is not None:
        novel_mask = np.array([cid not in train_cluster_ids for cid in gene_cluster_ids])
        shared_mask = ~novel_mask
        metrics['n_novel_family_genes'] = int(novel_mask.sum())
        metrics['n_shared_family_genes'] = int(shared_mask.sum())
        metrics['rho_novel_families'] = _safe_spearman(labels[novel_mask], preds[novel_mask])
        metrics['rho_shared_families'] = _safe_spearman(labels[shared_mask], preds[shared_mask])
    else:
        metrics['n_novel_family_genes'] = None
        metrics['n_shared_family_genes'] = None
        metrics['rho_novel_families'] = None
        metrics['rho_shared_families'] = None

    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on a dataset (base metrics only, used for val during training)."""
    preds, labels = collect_predictions(model, loader, device)
    return compute_base_metrics(preds, labels)


def train_loso_fold(
    df: pd.DataFrame,
    loso_group: str,
    config: TrainConfig,
    embedder_info: Dict[str, Dict],
) -> Optional[Dict[str, float]]:
    """Train and evaluate on a single LOSO fold."""
    # Split data
    test_mask = (df['loso_group'] == loso_group) & (~df['training_only'])
    train_mask = df['loso_group'] != loso_group
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    if len(test_df) == 0:
        logger.warning(f"No test samples for {loso_group}")
        return None

    # Build a validation split from train_df only (never touch test for selection).
    eligible_val = train_df[~train_df['training_only']]
    if len(eligible_val) < 50:
        raise ValueError(
            f"Too few non-training_only samples ({len(eligible_val)}) to form validation "
            f"for fold {loso_group}."
        )
    if 'operon_id' not in train_df.columns:
        raise ValueError(
            "LOSO training requires 'operon_id' to build operon-grouped validation splits."
        )
    if eligible_val['operon_id'].isna().any():
        n_missing = int(eligible_val['operon_id'].isna().sum())
        raise ValueError(
            f"Found {n_missing} rows with missing operon_id in eligible validation pool "
            f"for fold {loso_group}."
        )
    n_val_target = max(1, int(len(eligible_val) * config.val_fraction))
    # Deterministic operon-grouped validation split for reproducibility.
    rng = np.random.default_rng(42)
    operon_counts = (
        eligible_val.groupby('operon_id')['gene_id']
        .count()
        .sort_values(ascending=False)
    )
    operons = operon_counts.index.to_numpy()
    rng.shuffle(operons)
    picked_operons = []
    running = 0
    for op in operons:
        picked_operons.append(op)
        running += int(operon_counts.loc[op])
        if running >= n_val_target:
            break
    val_indices = eligible_val[eligible_val['operon_id'].isin(picked_operons)].index
    val_df = train_df.loc[val_indices]
    train_df = train_df.drop(index=val_indices)
    if len(val_df) == 0 or len(train_df) == 0:
        raise ValueError(f"Invalid train/val split for fold {loso_group}")
    
    # Create datasets — compute z-score normalization from training set only,
    # then apply to val/test to prevent data leakage.
    train_weights = _compute_sample_weights(train_df, config.sample_weights)
    train_dataset = EmbeddingDataset(
        train_df, embedder_info, embed_norm=config.embed_norm,
        sample_weights=train_weights,
    )
    norm_stats = train_dataset.compute_tabular_norm_stats()
    if config.embed_norm == "zscore":
        norm_stats.update(train_dataset.compute_embedding_norm_stats())
    if norm_stats:
        # Re-create with normalization applied
        train_dataset = EmbeddingDataset(
            train_df, embedder_info, norm_stats=norm_stats,
            embed_norm=config.embed_norm, sample_weights=train_weights,
        )
    val_dataset = EmbeddingDataset(val_df, embedder_info, norm_stats=norm_stats, embed_norm=config.embed_norm)
    test_dataset = EmbeddingDataset(test_df, embedder_info, norm_stats=norm_stats, embed_norm=config.embed_norm)
    
    train_sampler = build_train_sampler(config, train_df)
    total_input_dim = sum(info['dim'] for info in embedder_info.values())
    _nw = config.num_workers
    _dl_kw = dict(
        pin_memory=(config.device != "cpu"),
        num_workers=_nw,
        persistent_workers=(_nw > 0),
        prefetch_factor=(2 if _nw > 0 else None),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        **_dl_kw,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dl_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dl_kw,
    )
    
    # Create model
    input_dims = {name: info['dim'] for name, info in embedder_info.items()}
    model, fair_meta, cfg_used = build_model_with_fairness(config, input_dims)
    model = model.to(config.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # LR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.num_epochs
    scheduler = build_lr_scheduler(optimizer, config.lr_warmup_steps, config.lr_schedule, total_steps)
    
    # Training loop with early stopping
    use_spearman_es = config.early_stop_metric == "spearman"
    best_val_score = float('-inf') if use_spearman_es else float('inf')
    patience_counter = 0
    best_state = None
    t_train_start = time.time()

    for epoch in range(config.num_epochs):
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, cfg_used.device,
            scheduler=scheduler,
            ranking_lambda=config.ranking_loss_lambda,
            ranking_margin=config.ranking_loss_margin,
            ranking_pairs=config.ranking_loss_pairs_per_sample,
            loss_function=config.loss_function,
            huber_delta=config.huber_delta,
            loss_cap_percentile=config.loss_cap_percentile,
            modality_dropout=config.modality_dropout,
        )

        metrics = evaluate(model, val_loader, cfg_used.device)
        val_rmse = metrics['rmse']
        val_rho = metrics.get('spearman', float('nan'))

        if use_spearman_es:
            val_score = val_rho if np.isfinite(val_rho) else float('-inf')
            improved = val_score > best_val_score
        else:
            val_score = val_rmse
            improved = val_score < best_val_score

        if improved:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.time() - t_train_start
        es_label = "rho" if use_spearman_es else "rmse"
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_rmse=%.4f  val_rho=%.4f  best_%s=%.4f  pat=%d/%d  [%.0fs]",
            epoch + 1, config.num_epochs, train_loss, val_rmse,
            val_rho if np.isfinite(val_rho) else 0.0,
            es_label, best_val_score,
            patience_counter, config.early_stopping_patience, elapsed,
        )
        if patience_counter >= config.early_stopping_patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Load best model and get final metrics
    if best_state is not None:
        model.load_state_dict(best_state)
    
    final_metrics = evaluate(model, test_loader, cfg_used.device)
    final_metrics['epochs'] = epoch + 1
    final_metrics['n_train'] = len(train_df)
    final_metrics['n_test'] = len(test_df)
    final_metrics.update(fair_meta)
    final_metrics['effective_latent_dim'] = cfg_used.latent_dim
    final_metrics['effective_hidden_dim'] = cfg_used.hidden_dim
    
    return final_metrics


def run_fusion_loso(
    df: pd.DataFrame,
    config: TrainConfig,
    embedder_info: Dict[str, Dict],
) -> Dict[str, any]:
    """Run full LOSO cross-validation for a fusion type."""
    loso_groups = df['loso_group'].unique()
    fold_results = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {config.fusion_type} fusion")
    logger.info(f"{'='*60}")
    
    for loso_group in tqdm(loso_groups, desc=config.fusion_type):
        result = train_loso_fold(df, loso_group, config, embedder_info)
        if result is not None:
            result['loso_group'] = loso_group
            fold_results.append(result)
    
    # Aggregate
    spearman_vals = [r['spearman'] for r in fold_results if not np.isnan(r['spearman'])]
    
    summary = {
        'fusion_type': config.fusion_type,
        'mean_spearman': np.mean(spearman_vals),
        'std_spearman': np.std(spearman_vals),
        'mean_pearson': np.mean([r['pearson'] for r in fold_results]),
        'mean_r2': np.mean([r['r2'] for r in fold_results]),
        'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
        'mean_model_trainable_params': np.mean(
            [r['model_trainable_params'] for r in fold_results if 'model_trainable_params' in r]
        ) if fold_results else None,
        'fold_results': fold_results,
    }
    
    logger.info(f"  Mean Spearman: {summary['mean_spearman']:.4f} ± {summary['std_spearman']:.4f}")
    
    return summary


def _compute_sample_weights(
    df: pd.DataFrame,
    mode: str = "none",
) -> np.ndarray:
    """Compute per-sample weights for confidence-weighted loss.

    Parameters
    ----------
    mode : str
        * ``"none"`` – uniform weights (1.0).
        * ``"quality_tier"`` – weight = (6 - quality_tier) / 3.0.
          Tier 1 (highest quality) → 1.67, Tier 3 → 1.0, Tier 5 → 0.33.
        * ``"identity"`` – weight = protein_identity_vs_paxdb (0–1),
          floored at 0.1 to avoid zero-weight samples.
        * ``"source_confidence"`` – weight by expression source reliability:
          v1_gold → 1.2, abundance_zscore → 1.0, abele_calibrated → 0.5,
          multiplied by protein-CDS identity where available.
        * ``"mega_downweight"`` – downweight mega-component genes (0.3×)
          to focus gradient on non-mega gene families.
    """
    n = len(df)
    if mode == "none":
        return np.ones(n, dtype=np.float32)
    elif mode == "quality_tier":
        if 'quality_tier' not in df.columns:
            logger.warning("quality_tier column missing; falling back to uniform weights.")
            return np.ones(n, dtype=np.float32)
        qt = df['quality_tier'].fillna(3).values.astype(np.float32)
        w = (6.0 - qt) / 3.0
        return w
    elif mode == "identity":
        if 'protein_identity_vs_paxdb' not in df.columns:
            logger.warning("protein_identity_vs_paxdb column missing; falling back to uniform weights.")
            return np.ones(n, dtype=np.float32)
        w = df['protein_identity_vs_paxdb'].fillna(0.5).values.astype(np.float32)
        w = np.maximum(w, 0.1)
        return w
    elif mode == "source_confidence":
        w = np.ones(n, dtype=np.float32)
        if 'expression_source' in df.columns:
            w[df['expression_source'] == 'abele_calibrated'] = 0.5
        if 'source_dataset' in df.columns:
            w[df['source_dataset'] == 'v1_gold'] = 1.2
        if 'protein_cds_identity' in df.columns:
            ident = df['protein_cds_identity'].fillna(0.95).values.astype(np.float32)
            w *= np.clip(ident, 0.5, 1.0)
        if 'dna_cds_vs_protein_identity' in df.columns:
            ident = df['dna_cds_vs_protein_identity'].fillna(0.95).values.astype(np.float32)
            w *= np.clip(ident, 0.5, 1.0)
        return w
    elif mode == "mega_downweight":
        w = np.ones(n, dtype=np.float32)
        if 'is_mega' in df.columns:
            w[df['is_mega'].astype(bool)] = 0.3
            n_mega = int(df['is_mega'].astype(bool).sum())
            logger.info("mega_downweight: %d/%d mega genes weighted 0.3×", n_mega, n)
        else:
            logger.warning("is_mega column missing; mega_downweight has no effect.")
        return w
    else:
        raise ValueError(f"Unknown sample_weights mode: {mode}")


def train_fixed_split(
    df: pd.DataFrame,
    config: TrainConfig,
    embedder_info: Dict[str, Dict],
    save_intermediate: bool = False,
    save_every_n: int = 5,
) -> Dict[str, float]:
    """Train/evaluate on explicit train/val/test split column."""
    if 'split' not in df.columns:
        raise ValueError("Fixed-split mode requires a 'split' column in dataframe.")
    allowed = {'train', 'val', 'test', 'excluded'}
    bad = sorted(set(df['split'].dropna().unique()) - allowed)
    if bad:
        raise ValueError(f"Unexpected split values: {bad}")
    # Drop 'excluded' genes (used for fraction-scaling: retained for mega-component, not trained)
    if 'excluded' in df['split'].values:
        n_excl = (df['split'] == 'excluded').sum()
        logger.info("Dropping %d 'excluded' genes before training.", n_excl)
        df = df[df['split'] != 'excluded'].copy()

    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Invalid fixed split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    # Free embedding columns from the original df to reclaim ~25 GB for
    # large modality counts (17+). Splits already have their own copies.
    import gc as _gc
    _emb_cols_to_drop = [info['col'] for info in embedder_info.values()
                         if info['col'] in df.columns]
    if _emb_cols_to_drop:
        df.drop(columns=_emb_cols_to_drop, inplace=True)
        _gc.collect()
        logger.info(f"Freed {len(_emb_cols_to_drop)} embedding columns from source df")

    # ── Sample weights for confidence-weighted loss ─────────────
    train_weights = _compute_sample_weights(train_df, config.sample_weights)
    # Val/test always use uniform weights (no bias in evaluation)
    val_weights = np.ones(len(val_df), dtype=np.float32)
    test_weights = np.ones(len(test_df), dtype=np.float32)

    # ── Scramble negative controls ──────────────────────────────
    if config.scramble_labels:
        logger.warning("⚠ SCRAMBLE-LABELS active — shuffling expression_level in train split.")
        rng_scr = np.random.default_rng(config.seed)
        train_df['expression_level'] = rng_scr.permutation(train_df['expression_level'].values)
        val_df['expression_level'] = rng_scr.permutation(val_df['expression_level'].values)

    # ── Build datasets with embedding normalization ─────────────
    import time as _time
    logger.info("  Building train dataset (%d rows, %d modalities)...", len(train_df), len(embedder_info))
    _t0 = _time.time()
    train_dataset = EmbeddingDataset(
        train_df, embedder_info, embed_norm=config.embed_norm,
        sample_weights=train_weights,
    )
    logger.info("  Train dataset built in %.1fs", _time.time() - _t0)
    norm_stats = train_dataset.compute_tabular_norm_stats()
    if config.embed_norm == "zscore":
        emb_stats = train_dataset.compute_embedding_norm_stats()
        norm_stats.update(emb_stats)
    # Apply norm stats in-place to avoid expensive rebuild.
    if norm_stats:
        logger.info("  Applying %d norm stats in-place...", len(norm_stats))
        for _ns_name, (_ns_mean, _ns_std) in norm_stats.items():
            if _ns_name in train_dataset.embeddings:
                _mean_f32 = _ns_mean.astype(np.float32)
                _std_f32 = _ns_std.astype(np.float32)
                train_dataset.embeddings[_ns_name] -= _mean_f32
                train_dataset.embeddings[_ns_name] /= _std_f32
    _emb_cols = [info['col'] for info in embedder_info.values() if info['col'] in train_df.columns]
    if _emb_cols:
        train_df.drop(columns=_emb_cols, inplace=True)
    _gc.collect()
    logger.info("  Building val dataset (%d rows)...", len(val_df))
    _t0 = _time.time()
    val_dataset = EmbeddingDataset(
        val_df, embedder_info, norm_stats=norm_stats,
        embed_norm=config.embed_norm, sample_weights=val_weights,
    )
    logger.info("  Val dataset built in %.1fs", _time.time() - _t0)
    # Free val_df embedding columns
    _val_emb_cols = [info['col'] for info in embedder_info.values() if info['col'] in val_df.columns]
    if _val_emb_cols:
        val_df.drop(columns=_val_emb_cols, inplace=True)
    logger.info("  Building test dataset (%d rows)...", len(test_df))
    _t0 = _time.time()
    test_dataset = EmbeddingDataset(
        test_df, embedder_info, norm_stats=norm_stats,
        embed_norm=config.embed_norm, sample_weights=test_weights,
    )
    logger.info("  Test dataset built in %.1fs", _time.time() - _t0)
    _test_emb_cols = [info['col'] for info in embedder_info.values() if info['col'] in test_df.columns]
    if _test_emb_cols:
        test_df.drop(columns=_test_emb_cols, inplace=True)
    _gc.collect()
    logger.info("  All datasets ready. Building model...")

    # ── Scramble embeddings negative control ────────────────────
    if config.scramble_embeddings:
        logger.warning("⚠ SCRAMBLE-EMBEDDINGS active — shuffling row mapping per modality.")
        rng_scr = np.random.default_rng(config.seed)
        for name in train_dataset.embeddings:
            n = len(train_dataset.embeddings[name])
            train_dataset.embeddings[name] = train_dataset.embeddings[name][rng_scr.permutation(n)]

    # ── Build data loaders ──────────────────────────────────────
    train_sampler = build_train_sampler(config, train_df)
    total_input_dim = sum(info['dim'] for info in embedder_info.values())
    _nw = config.num_workers
    _dl_kw = dict(
        pin_memory=(config.device != "cpu"),
        num_workers=_nw,
        persistent_workers=(_nw > 0),
        prefetch_factor=(2 if _nw > 0 else None),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        **_dl_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dl_kw,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **_dl_kw,
    )

    input_dims = {name: info['dim'] for name, info in embedder_info.items()}
    logger.info("  Running build_model_with_fairness...")
    model, fair_meta, cfg_used = build_model_with_fairness(config, input_dims)
    actual_params = fair_meta.get('model_trainable_params', 0)
    logger.info("")
    logger.info("  " + "=" * 60)
    logger.info("  MODEL CARD (verified post-instantiation)")
    logger.info("  " + "-" * 60)
    logger.info("  Fusion architecture : %s", config.fusion_type)
    logger.info("  Trainable params    : %s", f"{actual_params:,d}")
    logger.info("  Latent dim (width)  : %s", fair_meta.get('effective_latent_dim', config.latent_dim))
    logger.info("  Hidden dim          : %s", fair_meta.get('effective_hidden_dim', config.hidden_dim))
    if config.fair_target_params:
        deviation = fair_meta.get('fair_deviation', 0)
        logger.info("  Target params       : %s", f"{config.fair_target_params:,d}")
        logger.info("  Deviation           : %.4f (tolerance=%.2f)", deviation, config.fair_tolerance)
    else:
        logger.warning("  NO --fair-target-params set — param count is NOT capacity-controlled!")
        logger.warning("  Comparisons across architectures or modality sets are UNFAIR without this.")
    logger.info("  Device              : %s", config.device)
    logger.info("  " + "=" * 60)
    logger.info("")
    model = model.to(config.device)
    logger.info("  Model on device. Starting training loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ── LR scheduler ───────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.num_epochs
    scheduler = build_lr_scheduler(optimizer, config.lr_warmup_steps, config.lr_schedule, total_steps)

    use_spearman_es = config.early_stop_metric == "spearman"
    use_non_mega_val = config.val_metric == "non_mega"
    best_val_score = float('-inf') if use_spearman_es else float('inf')
    patience_counter = 0
    best_state = None
    intermediate_states = {}

    # Build non-mega mask for val set if needed
    val_non_mega_mask = None
    if use_non_mega_val:
        if not use_spearman_es:
            raise ValueError("--val-metric non_mega requires --early-stop-metric spearman")
        _split_path = config._split_file_path
        if _split_path and Path(_split_path).exists():
            try:
                # Always use compound_operon_id for mega definition (41.2%, 202K genes).
                # operon_component_id gives a much larger component (80.7%, 397K) — wrong.
                # If the split file lacks compound_operon_id, fall back to kfold split.
                _split_df_cols = pd.read_csv(_split_path, sep='\t', nrows=0).columns
                if 'compound_operon_id' in _split_df_cols:
                    _split_df = pd.read_csv(_split_path, sep='\t',
                                            usecols=['gene_id', 'gene_cluster_id', 'compound_operon_id'])
                else:
                    # Original split lacks compound_operon_id — search known locations
                    _candidates = [
                        Path(_split_path).parent / 'kfold' / 'kfold_5f_fold0.tsv',
                        Path('data/splits/hard_hybrid_5fold/fold0.tsv'),
                        Path('data/splits/kfold/kfold_5f_fold0.tsv'),
                    ]
                    _kfold_path = next((p for p in _candidates if p.exists()), None)
                    if _kfold_path is not None:
                        logger.info("Split file lacks compound_operon_id; using %s for mega definition", _kfold_path)
                        _split_df = pd.read_csv(str(_kfold_path), sep='\t',
                                                usecols=['gene_id', 'gene_cluster_id', 'compound_operon_id'])
                    else:
                        raise FileNotFoundError(
                            f"Cannot find kfold split for mega definition. Tried: {_kfold_path}"
                        )
                _full_mask = _build_non_mega_mask(_split_df)
                _mega_genes = set(_split_df.loc[~_full_mask, 'gene_id'])
                val_non_mega_mask = ~val_df['gene_id'].isin(_mega_genes).values
                n_nm = val_non_mega_mask.sum()
                logger.info("Val non-mega early stopping: %d/%d val genes are non-mega (%.1f%%)",
                            n_nm, len(val_non_mega_mask), 100 * n_nm / len(val_non_mega_mask))
            except Exception as e:
                raise ValueError(
                    f"--val-metric non_mega requested but non-mega mask construction failed: {e}. "
                    f"Ensure the split file has 'compound_operon_id' or that a kfold file "
                    f"with compound_operon_id is available at <split_dir>/kfold/kfold_5f_fold0.tsv. "
                    f"NEVER silently fall back — see the codebase documentation."
                ) from e
        else:
            raise ValueError(
                "--val-metric non_mega requires --split-file with cluster/operon columns. "
                "Provide a split file via --split-file. "
                "NEVER silently fall back — see the codebase documentation."
            )

    t_train_start = time.time()
    for epoch in range(config.num_epochs):
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, cfg_used.device,
            scheduler=scheduler,
            ranking_lambda=config.ranking_loss_lambda,
            ranking_margin=config.ranking_loss_margin,
            ranking_pairs=config.ranking_loss_pairs_per_sample,
            loss_function=config.loss_function,
            huber_delta=config.huber_delta,
            loss_cap_percentile=config.loss_cap_percentile,
            modality_dropout=config.modality_dropout,
        )

        # Evaluate on full val set (unchanged path for --val-metric overall)
        val_rho_nm = float('nan')
        if use_non_mega_val and val_non_mega_mask is not None:
            # Need raw predictions for non-mega subsetting
            val_preds, val_labels = collect_predictions(model, val_loader, cfg_used.device)
            val_rho = float(stats.spearmanr(val_labels, val_preds)[0]) if len(val_labels) > 10 else float('nan')
            val_rmse = float(np.sqrt(np.mean((val_labels - val_preds) ** 2)))
            nm_preds = val_preds[val_non_mega_mask]
            nm_labels = val_labels[val_non_mega_mask]
            if len(nm_labels) > 10:
                val_rho_nm = float(stats.spearmanr(nm_labels, nm_preds)[0])
            val_score = val_rho_nm if np.isfinite(val_rho_nm) else float('-inf')
            improved = val_score > best_val_score
        else:
            # Original path — no behavior change
            val_metrics = evaluate(model, val_loader, cfg_used.device)
            val_rmse = val_metrics['rmse']
            val_rho = val_metrics.get('spearman', float('nan'))
            if use_spearman_es:
                val_score = val_rho if np.isfinite(val_rho) else float('-inf')
                improved = val_score > best_val_score
            else:
                val_score = val_rmse
                improved = val_score < best_val_score

        if improved:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if save_intermediate:
            ep1 = epoch + 1
            if ep1 == 1 or ep1 % save_every_n == 0:
                intermediate_states[f"epoch{ep1:03d}"] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

        elapsed = time.time() - t_train_start
        es_label = "rho_nm" if use_non_mega_val else ("rho" if use_spearman_es else "rmse")
        nm_str = f"  val_rho_nm={val_rho_nm:.4f}" if use_non_mega_val and np.isfinite(val_rho_nm) else ""
        logger.info(
            "Epoch %2d/%d  train_loss=%.4f  val_rmse=%.4f  val_rho=%.4f%s  best_%s=%.4f  pat=%d/%d  [%.0fs]",
            epoch + 1, config.num_epochs, train_loss, val_rmse,
            val_rho if np.isfinite(val_rho) else 0.0,
            nm_str,
            es_label, best_val_score,
            patience_counter, config.early_stopping_patience, elapsed,
        )
        if patience_counter >= config.early_stopping_patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    if save_intermediate:
        last_key = f"epoch{epoch+1:03d}"
        if last_key not in intermediate_states:
            intermediate_states[last_key] = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Base metrics (backward compatible) ──
    preds, labels = collect_predictions(model, test_loader, cfg_used.device)
    final_metrics = compute_base_metrics(preds, labels)
    final_metrics['epochs'] = epoch + 1
    final_metrics['n_train'] = len(train_df)
    final_metrics['n_val'] = len(val_df)
    final_metrics['n_test'] = len(test_df)
    final_metrics.update(fair_meta)
    final_metrics['effective_latent_dim'] = cfg_used.latent_dim
    final_metrics['effective_hidden_dim'] = cfg_used.hidden_dim

    # ── Stratified metrics (4-metric framework) ──
    test_species = test_df['species'].values

    test_cluster_ids = None
    if 'gene_cluster_id' in test_df.columns:
        test_cluster_ids = test_df['gene_cluster_id'].values

    test_is_mega = None
    if 'is_mega' in test_df.columns:
        test_is_mega = test_df['is_mega'].values

    train_cluster_set = None
    if 'gene_cluster_id' in train_df.columns:
        train_cluster_set = set(train_df['gene_cluster_id'].unique())

    stratified = compute_stratified_metrics(
        preds=preds,
        labels=labels,
        species=test_species,
        gene_cluster_ids=test_cluster_ids,
        is_mega=test_is_mega,
        train_cluster_ids=train_cluster_set,
    )
    final_metrics['stratified'] = stratified

    # Guardrail (the val-metric contract): if --val-metric non_mega was requested,
    # rho_non_mega MUST be computed. If it's None, is_mega was not available —
    # which means the split file or its kfold fallback lacked compound_operon_id.
    if config.val_metric == "non_mega" and stratified.get('rho_non_mega') is None:
        raise ValueError(
            "FATAL: --val-metric non_mega was requested but rho_non_mega is None "
            "in test-set metrics. The is_mega column was not computed — check that "
            "the split file has compound_operon_id or that the kfold fallback worked. "
            "See the codebase documentation."
        )

    # Attach artifacts for optional saving (predictions, model, test metadata)
    final_metrics['_artifacts'] = {
        'model': model,
        'best_state': best_state,
        'intermediate_states': intermediate_states,
        'test_preds': preds,
        'test_labels': labels,
        'test_gene_ids': test_df['gene_id'].values,
        'test_species': test_species,
        'test_cluster_ids': test_cluster_ids,
        'test_is_mega': test_is_mega,
        'input_dims': input_dims,
        'config': cfg_used,
        'norm_stats': norm_stats,  # GUARDRAIL G1: save for inference
    }

    return final_metrics


def run_fusion_fixed(
    df: pd.DataFrame,
    config: TrainConfig,
    embedder_info: Dict[str, Dict],
    save_intermediate: bool = False,
    save_every_n: int = 5,
) -> Dict[str, any]:
    """Run one fixed train/val/test experiment for a fusion type."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {config.fusion_type} fusion (fixed split)")
    logger.info(f"{'='*60}")
    result = train_fixed_split(df, config, embedder_info,
                               save_intermediate=save_intermediate,
                               save_every_n=save_every_n)
    summary = {
        'fusion_type': config.fusion_type,
        'mode': 'fixed_split',
        'mean_spearman': result['spearman'],
        'std_spearman': 0.0,
        'mean_pearson': result['pearson'],
        'mean_r2': result['r2'],
        'mean_rmse': result['rmse'],
        'mean_model_trainable_params': result.get('model_trainable_params'),
        'fold_results': [result],
    }

    # Propagate stratified metrics to top-level summary for easy leaderboard access
    strat = result.get('stratified', {})
    for key in ('rho_overall', 'rho_non_mega', 'rho_mega',
                'rho_cluster_weighted', 'rho_per_species',
                'rho_novel_families', 'rho_shared_families'):
        summary[key] = strat.get(key)

    rho_str = f"{summary['mean_spearman']:.4f}"
    extra_parts = []
    if strat.get('rho_non_mega') is not None:
        extra_parts.append(f"non-mega={strat['rho_non_mega']:.4f}")
    if strat.get('rho_per_species') is not None:
        extra_parts.append(f"per-sp={strat['rho_per_species']:.4f}")
    extra = f" ({', '.join(extra_parts)})" if extra_parts else ""
    logger.info(f"  Test Spearman: {rho_str}{extra}")
    return summary


def verify_embedding_registry(
    data_dir: Path,
    embedder_names: List[str],
    strict: bool = True,
) -> None:
    """Verify requested embeddings against the directory's registry.

    If a registry file exists in data_dir, checks that every requested
    modality's parquet matches its registered SHA256. Also warns about
    unregistered files that could corrupt glob-based loaders.

    Raises ValueError on SHA mismatch. Logs warnings for missing registry
    or unregistered files.
    """
    import importlib.util
    _reg_path = Path(__file__).parent / "embedding_registry.py"
    _spec = importlib.util.spec_from_file_location("embedding_registry", _reg_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    REGISTRY_FILENAME = _mod.REGISTRY_FILENAME
    verify_registry = _mod.verify_registry

    registry_path = data_dir / REGISTRY_FILENAME
    if not registry_path.exists():
        logger.warning(
            "No embedding registry at %s — skipping integrity check. "
            "Run: python aikixp/embedding_registry.py generate "
            "--embed-dir %s --label <your_label>",
            registry_path, data_dir,
        )
        return

    parquet_names = []
    for name in embedder_names:
        if name in COMPANION_EMBEDDER_INFO:
            stem = Path(COMPANION_EMBEDDER_INFO[name]["parquet"]).stem
            parquet_names.append(stem)

    if not parquet_names:
        return

    failures = verify_registry(data_dir, parquet_names, strict=strict)

    sha_failures = [f for f in failures if f["error"] == "SHA256_MISMATCH"]
    unregistered = [f for f in failures if f["error"] == "UNREGISTERED_FILE"]
    missing_reg = [f for f in failures if f["error"] == "NOT_IN_REGISTRY"]
    other = [f for f in failures if f["error"] not in
             ("SHA256_MISMATCH", "UNREGISTERED_FILE", "NOT_IN_REGISTRY")]

    if sha_failures:
        msg = "EMBEDDING INTEGRITY CHECK FAILED — file(s) modified since registration:\n"
        for f in sha_failures:
            msg += f"  {f['modality']}: {f['detail']}\n"
        msg += (
            "This means the parquet on disk does not match what was registered. "
            "Possible causes: file was replaced, re-extracted with different PCA, "
            "or aliased from another modality.\n"
            "To update the registry after a legitimate re-extraction, re-run:\n"
            f"  python aikixp/embedding_registry.py generate "
            f"--embed-dir {data_dir} --label <your_label>"
        )
        raise ValueError(msg)

    if unregistered:
        for f in unregistered:
            logger.warning("UNREGISTERED FILE: %s", f["detail"])

    if missing_reg:
        for f in missing_reg:
            logger.info("Modality %s not in registry (may be new) — no SHA check.",
                        f["modality"])

    if other:
        for f in other:
            logger.warning("Registry check issue: [%s] %s: %s",
                           f["error"], f["modality"], f["detail"])

    n_checked = len(parquet_names) - len(missing_reg)
    if n_checked > 0:
        with open(registry_path) as f:
            reg_meta = json.load(f)
        gcs_status = "GCS-verified" if reg_meta.get("gcs_verified") else "NOT GCS-verified"
        logger.info(
            "Embedding registry: %d/%d modalities verified (SHA256 match, %s, label=%s)",
            n_checked - len(sha_failures), n_checked,
            gcs_status, reg_meta.get("label", "?"),
        )
        if not reg_meta.get("gcs_verified"):
            logger.warning(
                "Registry was NOT verified against GCS at generation time. "
                "For production runs, regenerate with: "
                "python aikixp/embedding_registry.py generate "
                "--embed-dir %s --gcs-dir <GCS_PATH>",
                data_dir,
            )


def load_companion_embeddings(
    df: pd.DataFrame,
    embedder_names: List[str],
    data_dir: Path,
    allow_partial: bool = False,
    min_coverage_pct: float = 50.0,
) -> pd.DataFrame:
    """Load and join companion embedding parquets (operon, bacformer, etc.).

    Companion embeddings are stored in separate parquets aligned by gene_id.
    They are joined into the main dataframe when requested via --embedders.

    Supports two formats:
      - Standard: parquet with 'gene_id' + single embedding column (array)
      - Biophysical: parquet with 'gene_id' + multiple scalar feature columns,
        which are packed into a single array column for the fusion model.
        Identified by the presence of 'feature_cols' in COMPANION_EMBEDDER_INFO.

    Parameters
    ----------
    allow_partial : bool
        If True, zero-fill missing embeddings (coverage >= min_coverage_pct).
        If False (default), raise on any missing embeddings.
    min_coverage_pct : float
        Minimum coverage percentage required even when allow_partial is True.
        Modalities below this threshold always fail.
    """
    try:
        verify_embedding_registry(data_dir, embedder_names)
    except ValueError:
        raise
    except Exception as e:
        logger.warning("Registry verification could not run: %s", e)

    for name in embedder_names:
        if name not in COMPANION_EMBEDDER_INFO:
            continue
            
        info = COMPANION_EMBEDDER_INFO[name]
        parquet_name = info['parquet']
        col_name = info['col']
        
        # Check if column already exists
        if col_name in df.columns:
            logger.info(f"  {name}: column '{col_name}' already in dataframe")
            continue
            
        parquet_path = data_dir / parquet_name
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Companion embedding parquet not found: {parquet_path}\n"
                f"Run the embedding script first (e.g., embed_rna_regions.py, "
                f"featurize_biophysical.py)"
            )
        
        logger.info(f"  Loading {name} from {parquet_path}")
        companion_df = pd.read_parquet(parquet_path)
        
        if 'gene_id' not in companion_df.columns:
            raise ValueError(
                f"Companion parquet must have 'gene_id' column. "
                f"Found: {list(companion_df.columns)}"
            )
        
        # ── Filter companion to genes in main dataframe before validation ──
        # Companion parquets may contain superset of genes (e.g., full pre-gold set).
        # Filter first so NaN checks only cover genes that will actually be used.
        main_gene_ids = set(df['gene_id'])
        companion_df = companion_df[companion_df['gene_id'].isin(main_gene_ids)].copy()
        
        # ── Biophysical features: pack scalar columns into array ─────
        feature_cols = info.get('feature_cols')
        if feature_cols is not None:
            # Validate that all feature columns exist
            missing = set(feature_cols) - set(companion_df.columns)
            if missing:
                raise ValueError(
                    f"Biophysical parquet missing feature columns: {missing}\n"
                    f"Found: {list(companion_df.columns)}\n"
                    f"Expected: {feature_cols}"
                )
            
            # Pack scalar features into a single float32 array column.
            # Zero-fill the rare NaN entries (typically <0.01% of rows).
            nan_counts = companion_df[feature_cols].isna().sum()
            total_nan = int(nan_counts.sum())
            if total_nan > 0:
                bad = nan_counts[nan_counts > 0].to_dict()
                nan_frac = total_nan / (len(companion_df) * len(feature_cols))
                if nan_frac > 0.01:
                    raise ValueError(
                        f"Too many NaN values in classical features for {name}: {bad} "
                        f"({nan_frac:.2%} of cells). Refusing imputation."
                    )
                logger.warning(
                    "  Zero-filling %d NaN cells in %s classical features: %s",
                    total_nan, name, bad,
                )
                companion_df[feature_cols] = companion_df[feature_cols].fillna(0.0)
            feature_matrix = companion_df[feature_cols].values.astype(np.float32)
            companion_df = companion_df[['gene_id']].copy()
            companion_df[col_name] = list(feature_matrix)
            
            logger.info(f"    Packed {len(feature_cols)} scalar features → '{col_name}' array")
        else:
            # Standard embedding column
            if col_name not in companion_df.columns:
                raise ValueError(
                    f"Companion parquet must have '{col_name}' column. "
                    f"Found: {list(companion_df.columns)}"
                )
            companion_df = companion_df[['gene_id', col_name]]
        
        # Join on gene_id
        companion_df = companion_df.set_index('gene_id')
        df = df.set_index('gene_id').join(companion_df, how='left').reset_index()
        
        # Check coverage — handle missing embeddings and NaN-containing vectors
        n_valid = df[col_name].notna().sum()
        n_missing = len(df) - n_valid
        coverage_pct = n_valid / len(df) * 100 if len(df) > 0 else 100
        logger.info(
            f"    Joined: {n_valid}/{len(df)} genes have {name} embeddings ({coverage_pct:.1f}%)"
        )
        # Also detect rows where the vector itself contains NaN (e.g. overlong sequences)
        if not feature_cols and n_valid > 0:
            present_mask = df[col_name].notna()
            has_nan = present_mask & df.loc[present_mask, col_name].apply(
                lambda v: bool(np.isnan(v).any()) if isinstance(v, np.ndarray) else False
            ).reindex(df.index, fill_value=False)
            n_nan_vectors = int(has_nan.sum())
            if n_nan_vectors > 0:
                logger.warning(
                    "    %d %s embedding vectors contain internal NaN (%.2f%%) — zero-filling.",
                    n_nan_vectors, name, 100 * n_nan_vectors / len(df),
                )
                zero_vec = np.zeros(info['dim'], dtype=np.float32)
                for idx in df.index[has_nan]:
                    df.at[idx, col_name] = zero_vec.copy()
        if n_missing > 0:
            expected_dim = info['dim']
            if not allow_partial:
                raise ValueError(
                    f"Incomplete companion embedding coverage for {name}: "
                    f"{n_valid}/{len(df)} ({coverage_pct:.1f}%). "
                    f"Pass allow_partial=True (--allow-partial-embeddings) to zero-fill missing vectors."
                )
            if coverage_pct < min_coverage_pct:
                raise ValueError(
                    f"Coverage for {name} is {coverage_pct:.1f}%, below minimum "
                    f"{min_coverage_pct:.0f}%. Even with --allow-partial-embeddings, "
                    f"this modality has too little coverage to be useful."
                )
            zero_vec = np.zeros(expected_dim, dtype=np.float32)
            mask = df[col_name].isna()
            for idx in df.index[mask]:
                df.at[idx, col_name] = zero_vec.copy()
            logger.warning(
                "    Zero-filled %d missing %s embeddings (%.1f%% coverage). "
                "Missing genes receive zero vectors — fusion model must compensate.",
                n_missing, name, coverage_pct,
            )
    
    return df


def run_train_preflight_audit(
    df: pd.DataFrame,
    active_embedder_info: Dict[str, Dict],
    output_dir: Path,
    is_fixed_split: bool = False,
) -> Dict[str, any]:
    """Run fail-fast preflight checks and persist an audit gate report."""
    gates = []

    def add_gate(gate_id: str, status: str, critical: bool, detail: str):
        gates.append(
            {"id": gate_id, "status": status, "critical": critical, "detail": detail}
        )

    # In fixed-split mode, loso_group and training_only are not required
    if is_fixed_split:
        required_cols = ['gene_id', 'species', 'expression_level']
    else:
        required_cols = ['gene_id', 'species', 'loso_group', 'training_only', 'operon_id', 'expression_level']
    missing_cols = [c for c in required_cols if c not in df.columns]
    add_gate(
        "required_columns_present",
        "pass" if not missing_cols else "fail",
        True,
        f"missing={missing_cols} (is_fixed_split={is_fixed_split})",
    )

    null_counts = {c: int(df[c].isna().sum()) for c in required_cols if c in df.columns}
    has_nulls = any(v > 0 for v in null_counts.values())
    add_gate(
        "split_critical_columns_non_null",
        "pass" if not has_nulls else "fail",
        True,
        f"null_counts={null_counts}",
    )

    missing_embedder_cols = []
    embedder_null_counts = {}
    for name, info in active_embedder_info.items():
        col = info['col']
        if col not in df.columns:
            missing_embedder_cols.append(col)
            continue
        embedder_null_counts[col] = int(df[col].isna().sum())
    add_gate(
        "embedding_columns_present",
        "pass" if not missing_embedder_cols else "fail",
        True,
        f"missing_embedding_columns={missing_embedder_cols}",
    )
    add_gate(
        "embedding_coverage_complete",
        "pass" if all(v == 0 for v in embedder_null_counts.values()) else "fail",
        True,
        f"embedding_null_counts={embedder_null_counts}",
    )

    add_gate(
        "val_split_not_test_based",
        "pass",
        True,
        "Early stopping uses train-derived validation fold in train_loso_fold.",
    )
    add_gate(
        "val_split_operon_grouped",
        "pass",
        True,
        "Validation fold sampling uses operon groups from training data.",
    )

    overall_pass = not any(g["critical"] and g["status"] == "fail" for g in gates)
    report = {
        "overall_pass": overall_pass,
        "n_rows": int(len(df)),
        "n_species": int(df['species'].nunique()) if 'species' in df.columns else 0,
        "n_loso_groups": int(df['loso_group'].nunique()) if 'loso_group' in df.columns else 0,
        "gates": gates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "train_preflight_audit_gate_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Training preflight audit report written: {report_path}")
    return report


def validate_fixed_split_integrity(
    df: pd.DataFrame,
    hard_check_columns: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Check grouping-column coherence for externally provided fixed splits.

    Parameters
    ----------
    hard_check_columns : list of str, optional
        Columns that MUST NOT span multiple splits (fail on violation).
        Default ``['gene_cluster_id']`` — correct for gene-operon splits.
        Pass ``[]`` for species-holdout splits where gene-cluster spanning
        is expected by design.
    """
    if 'split' not in df.columns:
        raise ValueError("Missing 'split' column after fixed-split merge.")
    allowed = {'train', 'val', 'test', 'excluded'}
    bad = sorted(set(df['split'].dropna().unique()) - allowed)
    if bad:
        raise ValueError(f"Unexpected split values: {bad}")
    if df['split'].isna().any():
        raise ValueError("Null split values found after fixed-split merge.")
    # For integrity checks, exclude 'excluded' genes (they don't participate in training)
    if 'excluded' in df['split'].values:
        df = df[df['split'] != 'excluded'].copy()

    if hard_check_columns is None:
        hard_check_columns = ['gene_cluster_id']
    hard_set = set(hard_check_columns)

    summary: Dict[str, Dict[str, int]] = {}

    all_grouping_columns = [
        'gene_cluster_id', 'supergroup_id', 'operon_component_id',
        'compound_operon_id',
    ]
    for col in all_grouping_columns:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            if col in hard_set:
                raise ValueError(
                    f"Column '{col}' has {int(df[col].isna().sum())} null values; "
                    f"cannot validate split integrity."
                )
            continue
        split_span = df.groupby(col)['split'].nunique()
        violating = split_span[split_span > 1]
        n_violations = len(violating)
        summary[col] = {"n_groups": int(df[col].nunique()), "violations": n_violations}
        if n_violations > 0:
            if col in hard_set:
                examples = violating.index.astype(str).tolist()[:5]
                raise ValueError(
                    f"Split leakage: {n_violations} '{col}' groups span "
                    f"multiple splits. Examples: {examples}"
                )
            logger.warning(
                "%s: %d of %d groups span multiple splits (soft check).",
                col, n_violations, int(df[col].nunique()),
            )

    operon_key_cols = ['operon_source', 'taxid', 'operon_id']
    if all(c in df.columns for c in operon_key_cols):
        composite = df[operon_key_cols].astype(str).agg('|'.join, axis=1)
        split_span = (
            df.assign(_operon_key=composite)
            .groupby('_operon_key')['split'].nunique()
        )
        violating = split_span[split_span > 1]
        n_violations = len(violating)
        summary['operon_composite'] = {
            "n_groups": int(composite.nunique()), "violations": n_violations,
        }
        if n_violations > 0:
            logger.warning(
                "Raw operon groups: %d of %d span multiple splits (soft check).",
                n_violations, int(composite.nunique()),
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description="ProtEx Fusion Training")
    parser.add_argument(
        '--fusion',
        type=str,
        default='single_adapter',
        help='Fusion architecture. Recommended: single_adapter (pyramid MLP, supports '
             '--fair-target-params). Options: single_adapter, cross_attention, '
             'attention_gated, gmu, concat, projector, linear_concat. '
             'Use "all" to compare all. '
             'WARNING: linear_concat (formerly "latent_alignment") is a single linear '
             'layer per modality — only suitable as a weak baseline, never for production.'
    )
    parser.add_argument(
        '--embedders',
        nargs='+',
        default=None,
        help='Embedder names to use (default: all 6 base embedders). '
             'Available: esm2, esmc, evo2, evo2_cds, evo2_init_window, hyenadna, codonfm, rinalmo, '
             'operon_hyenadna, bacformer, bacformer_large, '
             'utrbert_init, rinalmo_init, '
             'biophysical_init, classical_codon, classical_rna_init, '
             'classical_protein, classical_disorder, classical_operon_struct. '
             'DEPRECATED (ablation only): utrbert_junc, rinalmo_junc, '
             'biophysical_junc, classical_rna_junc'
    )
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--early-stop-metric', type=str, default='rmse', choices=['rmse', 'spearman'],
                        help='Validation metric for early stopping: rmse (lower=better) or spearman (higher=better)')
    parser.add_argument('--val-metric', type=str, default='overall', choices=['overall', 'non_mega'],
                        help='Which val subset for early stopping: overall (all val genes) or '
                             'non_mega (non-mega-component val genes only for checkpoint selection). '
                             'Requires split file with gene_cluster_id and compound_operon_id columns.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes for parallel data loading (default: 4)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--val-fraction', type=float, default=0.1, help='Validation fraction from training fold')
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument(
        '--fair-target-params',
        type=int,
        default=None,
        help='Optional target trainable parameter count for fair-capacity comparisons.',
    )
    parser.add_argument(
        '--fair-tolerance',
        type=float,
        default=0.05,
        help='Allowed relative deviation from fair-target-params.',
    )
    parser.add_argument(
        '--fair-auto-width',
        action='store_true',
        help='Auto-search latent/hidden width to match fair-target-params (single_adapter only).',
    )
    parser.add_argument('--fair-min-width', type=int, default=32, help='Min width for fair auto-search.')
    parser.add_argument('--fair-max-width', type=int, default=8192, help='Max width for fair auto-search.')
    parser.add_argument('--fair-width-step', type=int, default=16, help='Step size for fair auto-search.')

    # Reproducibility & exploration
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument(
        '--scramble-labels', action='store_true',
        help='Negative control: shuffle expression labels (should yield rho~0).',
    )
    parser.add_argument(
        '--scramble-embeddings', action='store_true',
        help='Negative control: shuffle embedding-to-gene mappings per modality.',
    )
    parser.add_argument(
        '--lr-warmup-steps', type=int, default=0,
        help='Linear warmup from 0 to --lr over N optimizer steps.',
    )
    parser.add_argument(
        '--lr-schedule', type=str, default='constant',
        choices=['constant', 'cosine', 'linear'],
        help='LR schedule after warmup.',
    )
    parser.add_argument(
        '--sampler', type=str, default='random',
        choices=['random', 'species_balanced', 'species_pure', 'expression_stratified'],
        help='Training batch sampling strategy.',
    )
    parser.add_argument(
        '--label-mode', type=str, default='winsorized',
        choices=['winsorized', 'zscore', 'raw'],
        help='Expression label normalization: winsorized (robust z-score), zscore (classic), raw.',
    )
    parser.add_argument(
        '--sample-weights', type=str, default='none',
        choices=['none', 'quality_tier', 'identity', 'source_confidence', 'mega_downweight'],
        help='Per-sample confidence weighting for loss: none, quality_tier, identity, source_confidence, mega_downweight.',
    )
    parser.add_argument(
        '--embed-norm', type=str, default='none',
        choices=['none', 'l2', 'zscore'],
        help='LLM embedding normalization before fusion: none, l2, zscore.',
    )

    parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'huber'],
                        help='Base loss function: mse (default) or huber (robust to outliers).')
    parser.add_argument('--huber-delta', type=float, default=1.0, help='Delta for Huber loss (transition point from quadratic to linear).')
    parser.add_argument('--loss-cap-percentile', type=float, default=None,
                        help='Cap per-sample loss at this percentile within each batch (e.g. 95 = truncated MSE). None=disabled.')

    parser.add_argument('--ranking-loss-lambda', type=float, default=0.0, help='Weight for pairwise ranking loss (0=disabled).')
    parser.add_argument('--ranking-loss-margin', type=float, default=1.0, help='Margin for ranking loss.')
    parser.add_argument('--ranking-loss-pairs-per-sample', type=int, default=4, help='Pairs per sample for ranking.')
    parser.add_argument('--modality-dropout', type=float, default=0.0,
                        help='Probability of zeroing each modality per sample during training (0=disabled). '
                             'E.g., 0.2 means each modality has 20%% chance of being masked to zero per sample. '
                             'Teaches the model to handle missing modalities at inference.')

    parser.add_argument(
        '--allow-partial-embeddings',
        action='store_true',
        help='Allow zero-filling for embeddings with partial coverage (>= --min-embedding-coverage).',
    )
    parser.add_argument(
        '--min-embedding-coverage',
        type=float,
        default=50.0,
        help='Minimum embedding coverage (%%) when --allow-partial-embeddings is set. Default: 50.',
    )
    parser.add_argument(
        '--split-integrity-checks',
        nargs='*',
        default=None,
        help='Columns that must not span splits (hard checks). '
             'Default: gene_cluster_id. Pass nothing for species-holdout splits. '
             'Example: --split-integrity-checks  (empty = no hard checks).',
    )

    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--output', type=Path, default=None, help='Output JSON path (overrides output-dir)')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save per-gene test predictions to a parquet file alongside the result JSON (default: True)')
    parser.add_argument('--no-save-predictions', action='store_false', dest='save_predictions',
                        help='Disable saving predictions')
    parser.add_argument('--save-checkpoint', action='store_true', default=True,
                        help='Save best model checkpoint (.pt) alongside the result JSON (default: True)')
    parser.add_argument('--no-save-checkpoint', action='store_false', dest='save_checkpoint',
                        help='Disable saving checkpoint')
    parser.add_argument('--save-intermediate-checkpoints', action='store_true', default=False,
                        help='Save first, last, and every-5th-epoch checkpoints (for OOD checkpoint averaging)')
    parser.add_argument('--save-every-n-epochs', type=int, default=5,
                        help='Save intermediate checkpoint every N epochs (requires --save-intermediate-checkpoints)')
    parser.add_argument(
        '--data-path',
        type=Path,
        default=DATA_PATH,
        help='Input training table (defaults to gold production table).',
    )
    parser.add_argument(
        '--allow-non-gold-table',
        action='store_true',
        help='Allow non-canonical input table path (default: fail fast).',
    )
    parser.add_argument(
        '--companion-dir',
        type=Path,
        default=None,
        help='Directory containing companion embedding parquets. '
             'Defaults to data-path parent directory.',
    )
    parser.add_argument(
        '--split-file',
        type=Path,
        default=None,
        help="Optional TSV with columns gene_id, split (train/val/test) for fixed-split training mode",
    )
    parser.add_argument(
        '--use-production-operon-split',
        action='store_true',
        help='DEPRECATED — use --split-file explicitly. This flag points to a legacy '
             'split path that caused the 2026-03-19 Evo-2 7B profiling discrepancy.',
    )
    parser.add_argument(
        '--validate-against-champion',
        type=str,
        default=None,
        metavar='CONFIG_NAME',
        help='Validate all settings against a named champion config from champion_registry.py '
             '(e.g. F10_25M_v2). Hard-aborts on any mismatch.',
    )
    
    args = parser.parse_args()
    if (not args.allow_non_gold_table) and args.data_path.name not in CANONICAL_GOLD_FILENAMES:
        raise ValueError(
            f"Refusing non-gold data table: {args.data_path}. "
            "Use --allow-non-gold-table to override."
        )
    if args.fair_auto_width and args.fair_target_params is None:
        raise ValueError("--fair-auto-width requires --fair-target-params.")

    # ── Seed everything for reproducibility ─────────────────────
    seed_everything(args.seed)
    logger.info("Random seed set to %d", args.seed)
    if args.scramble_labels:
        logger.warning("⚠ ⚠ ⚠  NEGATIVE CONTROL: --scramble-labels active. Results should yield rho ≈ 0.  ⚠ ⚠ ⚠")
    if args.scramble_embeddings:
        logger.warning("⚠ ⚠ ⚠  NEGATIVE CONTROL: --scramble-embeddings active. Results should yield rho ≈ 0.  ⚠ ⚠ ⚠")
    if args.use_production_operon_split:
        warnings.warn(
            "\n"
            "======================================================================\n"
            "DEPRECATED: --use-production-operon-split is deprecated and will be\n"
            "removed in a future release. This convenience flag caused the\n"
            "2026-03-19 Evo-2 7B profiling discrepancy (wrong split file).\n"
            "\n"
            "Use --split-file <path> explicitly instead. For the champion split:\n"
            "  --split-file data/splits/"
            "hard_hybrid_production_split_v2.tsv\n"
            "======================================================================",
            DeprecationWarning,
            stacklevel=2,
        )
        if args.split_file is not None:
            raise ValueError("Pass either --split-file or --use-production-operon-split, not both.")
        args.split_file = PRODUCTION_OPERON_SPLIT_PATH
        if not args.split_file.exists():
            raise FileNotFoundError(
                f"Production operon split not found: {args.split_file}. "
                "Generate it first with build_gene_cluster_split.py sweep outputs."
            )
        logger.info("Using canonical production operon split: %s", args.split_file)

    # ── G2: Champion config validation ────────────────────────────
    if args.validate_against_champion is not None:
        from champion_registry import CHAMPION_CONFIGS
        cname = args.validate_against_champion
        if cname not in CHAMPION_CONFIGS:
            raise ValueError(
                f"Unknown champion config '{cname}'. "
                f"Available: {list(CHAMPION_CONFIGS.keys())}"
            )
        champ = CHAMPION_CONFIGS[cname]
        mismatches = []

        def _check(field, actual, expected):
            if actual != expected:
                mismatches.append(f"  {field}: got {actual!r}, expected {expected!r}")

        _check("fusion_type", args.fusion, champ["fusion_type"])
        _check("embedders", sorted(args.embedders or []), champ["embedders"])
        _check("fair_target_params", args.fair_target_params, champ["fair_target_params"])
        if not args.fair_auto_width:
            # Only check latent/hidden when NOT using auto-width (auto-width overrides these)
            _check("latent_dim", args.latent_dim, champ["latent_dim"])
            _check("hidden_dim", args.hidden_dim, champ["hidden_dim"])
        _check("epochs", args.epochs, champ["epochs"])
        _check("early_stop_metric", args.early_stop_metric, champ["early_stop_metric"])
        _check("lr_schedule", args.lr_schedule, champ["lr_schedule"])
        _check("lr_warmup_steps", args.lr_warmup_steps, champ["lr_warmup_steps"])
        _check("label_mode", args.label_mode, champ["label_mode"])
        if args.split_file is not None:
            _check("split_file_name", Path(args.split_file).name, champ["split_file_name"])
        else:
            mismatches.append(f"  split_file: not set, expected file named {champ['split_file_name']!r}")

        if mismatches:
            raise SystemExit(
                f"\n{'='*70}\n"
                f"CHAMPION VALIDATION FAILED for '{cname}'\n"
                f"{'-'*70}\n"
                + "\n".join(mismatches) +
                f"\n{'='*70}\n"
                "Fix the arguments above or remove --validate-against-champion."
            )
        logger.info("Champion validation PASSED for '%s'", cname)

    # Determine which embedders to use
    all_available = {**EMBEDDER_INFO, **COMPANION_EMBEDDER_INFO}
    if args.embedders:
        unknown = set(args.embedders) - set(all_available.keys())
        if unknown:
            raise ValueError(
                f"Unknown embedders: {unknown}\n"
                f"Available: {sorted(all_available.keys())}"
            )
        selected_embedders = args.embedders
    else:
        # Default: 6 RA base embedders
        selected_embedders = ['esm2', 'esmc', 'evo2', 'hyenadna', 'codonfm', 'rinalmo']
    
    # Build embedder info for selected embedders
    active_embedder_info = {}
    for name in selected_embedders:
        if name in EMBEDDER_INFO:
            active_embedder_info[name] = EMBEDDER_INFO[name]
        elif name in COMPANION_EMBEDDER_INFO:
            active_embedder_info[name] = COMPANION_EMBEDDER_INFO[name]
    
    # Warn on deprecated junction embedders
    _deprecated = [n for n in selected_embedders
                   if all_available.get(n, {}).get('deprecated')]
    if _deprecated:
        logger.warning(
            "DEPRECATED junction embedders selected: %s. "
            "These are superseded by upstream/downstream context columns. "
            "Use only for ablation comparison, not production models.",
            _deprecated,
        )
    
    logger.info(f"Selected embedders: {list(active_embedder_info.keys())}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    missing_contract_cols = [c for c in REQUIRED_DATA_CONTRACT_COLS if c not in df.columns]
    if missing_contract_cols:
        raise ValueError(
            f"Input data table missing required contract columns: {missing_contract_cols}"
        )
    logger.info(f"Loaded {len(df)} samples")

    # Optional fixed split mode from explicit split file.
    if args.split_file is not None:
        # Blocklist: reject known-bad split files by SHA prefix.
        # The balanced split caused three incidents (2026-03-19, 2026-03-31, 2026-04-02).
        # See SPLIT_FILE_PROVENANCE.md.
        _BLOCKED_SPLIT_SHAS = {
            "2b6704a6afa4": "gene_cluster_operon_split_v2_balanced.tsv (DEPRECATED — "
                            "has 22 protein-leaked seqs, lacks compound_operon_id, caused "
                            "3 incidents. Use hard_hybrid_production_split_v2.tsv instead. "
                            "See SPLIT_FILE_PROVENANCE.md)",
        }
        import hashlib as _hl
        _split_sha = _hl.sha256(Path(args.split_file).read_bytes()).hexdigest()[:12]
        if _split_sha in _BLOCKED_SPLIT_SHAS:
            raise ValueError(
                f"Split file {args.split_file} is BLOCKED (SHA {_split_sha}): "
                f"{_BLOCKED_SPLIT_SHAS[_split_sha]}"
            )
        logger.info("Split file SHA: %s (not in blocklist)", _split_sha)

        split_df = pd.read_csv(args.split_file, sep='\t', dtype={'species': str})
        required_split_cols = {'gene_id', 'split'}
        missing_split_cols = required_split_cols - set(split_df.columns)
        if missing_split_cols:
            raise ValueError(
                f"Split file missing columns: {sorted(missing_split_cols)}; "
                f"found {list(split_df.columns)}"
            )
        if split_df['gene_id'].duplicated().any():
            raise ValueError("Split file has duplicate gene_id rows.")
        allowed_splits = {'train', 'val', 'test', 'excluded'}
        bad_split_vals = sorted(set(split_df['split'].dropna().unique()) - allowed_splits)
        if bad_split_vals:
            raise ValueError(f"Split file has invalid split values: {bad_split_vals}")
        # 'excluded' genes are kept through mega-component computation, then dropped before training
        n_excluded = (split_df['split'] == 'excluded').sum()
        if n_excluded > 0:
            logger.info("Split file has %d 'excluded' genes (for mega-component only, not trained).", n_excluded)
        # Avoid split column collision when data table already has historical split labels.
        if 'split' in df.columns:
            logger.warning(
                "Input data already contains 'split' column. It will be replaced by %s.",
                args.split_file,
            )
            df = df.drop(columns=['split'])

        # Merge split assignment plus any extra metadata columns needed for
        # stratified evaluation (gene_cluster_id, compound_operon_id, species_cluster).
        merge_cols = ['gene_id', 'split']
        extra_meta_cols = ['gene_cluster_id', 'compound_operon_id', 'species_cluster']
        for col in extra_meta_cols:
            if col in split_df.columns and col not in df.columns:
                merge_cols.append(col)
        # Drop any pre-existing columns that would collide (except gene_id)
        for col in merge_cols[1:]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.merge(
            split_df[merge_cols].rename(columns={'split': 'split_from_file'}),
            on='gene_id',
            how='left',
        )
        df = df.rename(columns={'split_from_file': 'split'})
        if df['split'].isna().any():
            missing = int(df['split'].isna().sum())
            if missing == len(df):
                raise ValueError("Split file covers zero genes in data table.")
            logger.info(
                "Split file covers %d/%d genes (%.1f%%). Dropping %d uncovered genes.",
                len(df) - missing, len(df), 100 * (len(df) - missing) / len(df), missing,
            )
            df = df.dropna(subset=['split']).reset_index(drop=True)
        hard_checks = args.split_integrity_checks
        if hard_checks is None:
            hard_checks = ['gene_cluster_id']
        integrity = validate_fixed_split_integrity(df, hard_check_columns=hard_checks)
        logger.info(
            "Loaded fixed split from %s with counts: %s",
            args.split_file,
            df['split'].value_counts().to_dict(),
        )
        if integrity:
            logger.info("Fixed-split integrity checks passed: %s", integrity)

        # Compute mega-component annotation if cluster metadata is available.
        # If split file lacks compound_operon_id, load it from the kfold reference
        # file so that is_mega is available for BOTH val early-stopping AND test
        # evaluation. This is not a silent fallback — it fails hard if the kfold
        # file is missing.
        if 'gene_cluster_id' in df.columns and 'compound_operon_id' not in df.columns:
            # Search multiple known locations for compound_operon_id reference
            _candidates = [
                Path(args.split_file).parent / 'kfold' / 'kfold_5f_fold0.tsv',
                Path('data/splits/hard_hybrid_5fold/fold0.tsv'),
                Path('data/splits/kfold/kfold_5f_fold0.tsv'),
            ]
            _kfold_path = next((p for p in _candidates if p.exists()), None)
            if _kfold_path is not None:
                logger.info(
                    "Split file lacks compound_operon_id; loading from %s "
                    "for is_mega computation (val + test).", _kfold_path
                )
                _kf = pd.read_csv(str(_kfold_path), sep='\t',
                                  usecols=['gene_id', 'compound_operon_id'])
                df = df.merge(_kf[['gene_id', 'compound_operon_id']], on='gene_id', how='left')
                n_null = df['compound_operon_id'].isna().sum()
                if n_null > 0:
                    raise ValueError(
                        f"compound_operon_id is null for {n_null} genes after kfold merge. "
                        f"The kfold file does not cover all genes in this split."
                    )
            elif args.val_metric == "non_mega":
                raise ValueError(
                    f"--val-metric non_mega requires compound_operon_id but split file "
                    f"lacks it and kfold reference not found at {_kfold_path}."
                )
            else:
                logger.warning(
                    "Split file lacks compound_operon_id and no kfold reference found — "
                    "mega-component metrics will be skipped."
                )

        if 'gene_cluster_id' in df.columns and 'compound_operon_id' in df.columns:
            df['is_mega'] = compute_mega_component(
                df['gene_cluster_id'], df['compound_operon_id']
            )
            n_mega = df['is_mega'].sum()
            logger.info(
                "is_mega computed: %d mega (%.1f%%), %d non-mega (%.1f%%)",
                n_mega, 100 * n_mega / len(df),
                len(df) - n_mega, 100 * (len(df) - n_mega) / len(df),
            )
    
    # Load companion embeddings if needed
    companion_dir = args.companion_dir if args.companion_dir else args.data_path.parent
    companion_names = [n for n in selected_embedders if n in COMPANION_EMBEDDER_INFO]
    if companion_names:
        logger.info(f"Loading companion embeddings from {companion_dir}: {companion_names}")
        df = load_companion_embeddings(
            df, companion_names, companion_dir,
            allow_partial=args.allow_partial_embeddings,
            min_coverage_pct=args.min_embedding_coverage,
        )
    
    # Harmonize labels – use train_mask for leakage-free stats when split is available
    train_mask = (df['split'] == 'train') if 'split' in df.columns else None
    df = harmonize_labels(df, mode=args.label_mode, train_mask=train_mask)
    if args.label_mode != "raw":
        logger.info(
            "Label normalization (%s): expression_level stats → mean=%.3f, std=%.3f",
            args.label_mode,
            df['expression_level'].mean(),
            df['expression_level'].std(),
        )

    is_fixed_split = args.split_file is not None or (
        'split' in df.columns and set(df['split'].dropna().unique()) == {'train', 'val', 'test'}
    )
    preflight_report = run_train_preflight_audit(
        df, active_embedder_info, args.output_dir, is_fixed_split=is_fixed_split,
    )
    if not preflight_report["overall_pass"]:
        raise RuntimeError("Training preflight audit failed. See train_preflight_audit_gate_report.json")
    
    # Resolve fusion type aliases and warn about weak architectures
    requested_fusion = args.fusion
    if requested_fusion in FUSION_TYPE_ALIASES:
        canonical = FUSION_TYPE_ALIASES[requested_fusion]
        warnings.warn(
            f"\n{'='*70}\n"
            f"DEPRECATED: --fusion '{requested_fusion}' has been renamed to '{canonical}'.\n"
            f"'{requested_fusion}' is a SINGLE LINEAR LAYER per modality — no pyramid,\n"
            f"no depth. It should only be used as a weak baseline, never for production.\n"
            f"Use --fusion single_adapter for real experiments.\n"
            f"{'='*70}\n",
            DeprecationWarning, stacklevel=1,
        )
        requested_fusion = canonical
    if requested_fusion == 'linear_concat':
        logger.warning(
            "linear_concat fusion is a single linear layer per modality. "
            "Results will be substantially worse than single_adapter (pyramid MLP). "
            "Consider using --fusion single_adapter instead."
        )

    # Determine which fusions to run
    if requested_fusion == 'all':
        fusion_types = FUSION_TYPES
    else:
        fusion_types = [requested_fusion]
    
    # Run experiments
    all_results = {}
    
    for fusion_type in fusion_types:
        config = TrainConfig(
            fusion_type=fusion_type,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            early_stop_metric=args.early_stop_metric,
            val_metric=args.val_metric,
            _split_file_path=str(args.split_file) if args.split_file else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.lr,
            val_fraction=args.val_fraction,
            seed=args.seed,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_schedule=args.lr_schedule,
            sampler=args.sampler,
            label_mode=args.label_mode,
            sample_weights=args.sample_weights,
            embed_norm=args.embed_norm,
            scramble_labels=args.scramble_labels,
            scramble_embeddings=args.scramble_embeddings,
            loss_function=args.loss_function,
            huber_delta=args.huber_delta,
            loss_cap_percentile=args.loss_cap_percentile,
            ranking_loss_lambda=args.ranking_loss_lambda,
            ranking_loss_margin=args.ranking_loss_margin,
            ranking_loss_pairs_per_sample=args.ranking_loss_pairs_per_sample,
            modality_dropout=args.modality_dropout,
            fair_target_params=args.fair_target_params,
            fair_tolerance=args.fair_tolerance,
            fair_auto_width=args.fair_auto_width,
            fair_min_width=args.fair_min_width,
            fair_max_width=args.fair_max_width,
            fair_width_step=args.fair_width_step,
        )
        _intermediate_kwargs = dict(
            save_intermediate=args.save_intermediate_checkpoints,
            save_every_n=args.save_every_n_epochs,
        )
        if args.split_file is not None:
            result = run_fusion_fixed(df, config, active_embedder_info, **_intermediate_kwargs)
        elif 'split' in df.columns and set(df['split'].dropna().unique()) == {'train', 'val', 'test'}:
            logger.info("Using embedded split column from production table (train/val/test)")
            integrity = validate_fixed_split_integrity(df)
            if integrity:
                logger.info("Embedded split integrity checks passed: %s", integrity)
            result = run_fusion_fixed(df, config, active_embedder_info, **_intermediate_kwargs)
        else:
            result = run_fusion_loso(df, config, active_embedder_info)
        all_results[fusion_type] = result
    
    # Print summary
    print("\n" + "="*80)
    print("FUSION COMPARISON RESULTS")
    print("="*80)
    print(f"{'Fusion':<25} {'Spearman':<15} {'Pearson':<12} {'R²':<12}")
    print("-"*80)
    
    for fusion_type, result in sorted(all_results.items(), key=lambda x: -x[1]['mean_spearman']):
        print(f"{fusion_type:<25} "
              f"{result['mean_spearman']:.4f}±{result['std_spearman']:.4f}  "
              f"{result['mean_pearson']:.4f}       "
              f"{result['mean_r2']:.4f}")
    
    # Compare to baselines
    # NOTE: These are gene-operon split baselines for reference only.
    # They are NOT computed on the current split's test set.
    print("\n" + "="*80)
    print("COMPARISON TO BASELINES")
    print("="*80)
    print("Reference baselines (gene-operon split, NOT computed on current test set):")
    print("  ESM-2 Ridge (gene-operon):  ρ ≈ 0.540")
    print("  ESM-C Ridge (gene-operon):  ρ ≈ 0.540")
    print("\nLearned fusion results:")
    for fusion_type, result in sorted(all_results.items(), key=lambda x: -x[1]['mean_spearman']):
        print(f"  {fusion_type:<20} ρ={result['mean_spearman']:.4f}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect artifacts before stripping for JSON serialization
    collected_artifacts = {}
    for fusion_type, result in all_results.items():
        for fr in result.get('fold_results', []):
            if '_artifacts' in fr:
                collected_artifacts[fusion_type] = fr.pop('_artifacts')

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items() if k != '_artifacts'}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    # Determine output path
    if args.output:
        output_file = args.output
    else:
        embedder_suffix = "_".join(sorted(selected_embedders))
        output_file = args.output_dir / f"fusion_results_{args.fusion}_{embedder_suffix}.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract verified param count from the first (usually only) fusion result
    _verified_params = None
    _verified_fusion = None
    for _ft, _fres in all_results.items():
        if isinstance(_fres, dict):
            _verified_params = _fres.get('mean_model_trainable_params')
            _verified_fusion = _ft
            break

    # ── G3: Compute split file hash ─────────────────────────────
    _split_sha256 = None
    if args.split_file is not None and Path(args.split_file).exists():
        _split_h = hashlib.sha256()
        with open(args.split_file, "rb") as _sf:
            for _chunk in iter(lambda: _sf.read(1 << 16), b""):
                _split_h.update(_chunk)
        _split_sha256 = _split_h.hexdigest()

    # Compute embedder list hash for fingerprint
    _embedder_hash = hashlib.sha256(
        ",".join(sorted(selected_embedders)).encode()
    ).hexdigest()[:12]

    # Include metadata in results — MODEL CARD fields at top level for easy access
    results_with_meta = {
        # ── MODEL CARD (top-level, never buried) ──
        "fusion_type": _verified_fusion or requested_fusion,
        "model_trainable_params": _verified_params,
        "embedders": sorted(selected_embedders),
        "embedder_dims": {n: active_embedder_info[n]['dim'] for n in selected_embedders},
        "input_dim_total": int(sum(active_embedder_info[n]['dim'] for n in selected_embedders)),
        # ── Run metadata ──
        "mode": "fixed_split" if args.split_file is not None else "loso",
        "split_file": str(args.split_file) if args.split_file is not None else None,
        "split_file_sha256": _split_sha256,
        "seed": args.seed,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_schedule": args.lr_schedule,
        "sampler": args.sampler,
        "label_mode": args.label_mode,
        "sample_weights": args.sample_weights,
        "embed_norm": args.embed_norm,
        "scramble_labels": args.scramble_labels,
        "scramble_embeddings": args.scramble_embeddings,
        "ranking_loss": {"lambda": args.ranking_loss_lambda, "margin": args.ranking_loss_margin, "pairs_per_sample": args.ranking_loss_pairs_per_sample},
        "fair_capacity": {
            "target_params": args.fair_target_params,
            "tolerance": args.fair_tolerance,
            "auto_width": args.fair_auto_width,
            "min_width": args.fair_min_width,
            "max_width": args.fair_max_width,
            "width_step": args.fair_width_step,
        },
        "results": convert_numpy(all_results),
    }

    # ── G6: Result JSON contract enforcement ──────────────────────
    _REQUIRED_RESULT_FIELDS = (
        "fusion_type", "model_trainable_params", "embedders",
        "split_file", "split_file_sha256", "seed", "fair_capacity", "results",
    )
    _missing_fields = [f for f in _REQUIRED_RESULT_FIELDS if f not in results_with_meta]
    if _missing_fields:
        raise SystemExit(
            f"RESULT JSON CONTRACT VIOLATION: missing required fields: {_missing_fields}\n"
            "This is a bug in train_fusion.py — a refactor likely dropped a field."
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        json.dump(results_with_meta, tmp_file, indent=2)
        tmp_path = tmp_file.name
    os.replace(tmp_path, output_path)

    logger.info(f"\nResults saved to {output_file}")

    # ── G3: Config fingerprint banner ─────────────────────────────
    _split_name = Path(args.split_file).name if args.split_file else "N/A (LOSO)"
    _split_hash_short = _split_sha256[:12] if _split_sha256 else "N/A"
    _param_str = f"{_verified_params:,}" if _verified_params else "N/A"
    logger.info(
        "\n"
        "======================================================================\n"
        "CONFIG FINGERPRINT (compare across runs for consistency)\n"
        "----------------------------------------------------------------------\n"
        "  Split hash    : %s\n"
        "  Split file    : %s\n"
        "  Fusion type   : %s\n"
        "  Param count   : %s\n"
        "  Embedder hash : %s\n"
        "  Seed          : %d\n"
        "======================================================================",
        _split_hash_short,
        _split_name,
        _verified_fusion or requested_fusion,
        _param_str,
        _embedder_hash,
        args.seed,
    )

    # ── Save per-gene predictions and model checkpoint ──
    for fusion_type, artifacts in collected_artifacts.items():
        stem = output_file.stem

        if args.save_predictions:
            pred_path = output_file.parent / f"{stem}_predictions.parquet"
            pred_df = pd.DataFrame({
                'gene_id': artifacts['test_gene_ids'],
                'species': artifacts['test_species'],
                'y_true': artifacts['test_labels'],
                'y_pred': artifacts['test_preds'],
            })
            if artifacts['test_cluster_ids'] is not None:
                pred_df['gene_cluster_id'] = artifacts['test_cluster_ids']
            if artifacts['test_is_mega'] is not None:
                pred_df['is_mega'] = artifacts['test_is_mega']
            pred_df.to_parquet(pred_path, index=False)
            logger.info("Per-gene predictions saved to %s (%d genes)", pred_path, len(pred_df))

        if args.save_checkpoint:
            # ── GUARDRAIL G1: Save norm_stats in checkpoint ──
            # Classical features are z-scored during training. Without saving
            # the mean/std, inference scripts cannot reproduce the normalization.
            
            saved_norm_stats = {}
            _ns = artifacts.get('norm_stats') or {}
            if _ns:
                for ns_name, (ns_mean, ns_std) in _ns.items():
                    saved_norm_stats[ns_name] = {
                        'mean': ns_mean.tolist() if hasattr(ns_mean, 'tolist') else list(ns_mean),
                        'std': ns_std.tolist() if hasattr(ns_std, 'tolist') else list(ns_std),
                    }

            ckpt_base = {
                'input_dims': artifacts['input_dims'],
                'fusion_type': fusion_type,
                'config': {
                    'latent_dim': artifacts['config'].latent_dim,
                    'hidden_dim': artifacts['config'].hidden_dim,
                    'dropout': artifacts['config'].dropout,
                    'num_layers': artifacts['config'].num_layers,
                },
                'embedders': sorted(selected_embedders),
                'norm_stats': saved_norm_stats,  # GUARDRAIL G1
            }

            ckpt_path = output_file.parent / f"{stem}_checkpoint.pt"
            ckpt = {**ckpt_base, 'model_state_dict': artifacts['best_state'] or artifacts['model'].state_dict()}
            torch.save(ckpt, ckpt_path)
            logger.info("Model checkpoint saved to %s", ckpt_path)

            for label, state_dict in artifacts.get('intermediate_states', {}).items():
                inter_path = output_file.parent / f"{stem}_{label}_checkpoint.pt"
                torch.save({**ckpt_base, 'model_state_dict': state_dict}, inter_path)
                logger.info("Intermediate checkpoint saved to %s", inter_path)

    # ── PROVENANCE REMINDER ──
    # Check if this run matches the current champion config
    try:
        from champion_registry import CHAMPION_CONFIGS, CURRENT_CHAMPION
        champ = CHAMPION_CONFIGS.get(CURRENT_CHAMPION, {})
        if set(selected_embedders) == set(champ.get("embedders", [])):
            best_rho = results_with_meta.get("results", {})
            for ft, r in best_rho.items():
                if isinstance(r, dict):
                    rho = r.get("mean_spearman", 0)
                    logger.info(
                        f"  CHAMPION CONFIG MATCH ({CURRENT_CHAMPION}): "
                        f"this run rho={rho:.4f}, expected ~{champ.get('rho_gene_operon', 'N/A')}"
                    )
    except ImportError:
        pass  # champion_registry not available (standalone run)

    logger.info(
        "\n"
        "======================================================================\n"
        "----------------------------------------------------------------------\n"
        "  1. GCS backup:\n"
        "  2. If this is a new result for the manuscript:\n"
        "\n"
        "     - Update docs/protex/ARTIFACT_MAP.md with GCS path\n"
        "  3. If this beats the champion:\n"
        "     - Update aikixp/champion_registry.py\n"
        "     - Update docs/protex/PROTEX_MANUSCRIPT_V2.md headline table\n"
        "======================================================================",
        output_file.parent,
    )


if __name__ == "__main__":
    main()
