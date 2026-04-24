#!/usr/bin/env python3
"""
Champion Registry — shared champion config definitions and hash utilities.

This module is intentionally lightweight (no torch dependency) so that audit
scripts can import it without pulling in the full training stack.

Author: ProtEx Team
Date: 2026-03-19
"""

import hashlib
from pathlib import Path


def _file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _embedder_list_hash(embedders: list[str]) -> str:
    """Stable 12-char hash of a sorted embedder list."""
    canonical = ",".join(sorted(embedders))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ── Champion configurations ──────────────────────────────────────────────
# Each entry records the *exact* settings of a validated champion model.
# Keys must match argparse dest names in train_fusion.py.

CHAMPION_CONFIGS = {
    # ── Current champion (2026-03-22): 7B PCA-4096 ──
    # ρ=0.666 ± 0.002 gene-operon (6 seeds), 0.671 species-cluster, 0.741 random.
    # Swaps evo2_cds (1B, 5120d) for evo2_7b_full_operon_pca4096 (7B, 4096d).
    "F10_7B_PCA4096": {
        "fusion_type": "single_adapter",
        "embedders": sorted([
            "esmc_protein",
            "dnabert2_operon_dna",
            "evo2_7b_full_operon_pca4096",
            "hyenadna_dna_cds",
            "bacformer",
            "classical_codon",
            "classical_rna_init",
            "classical_protein",
            "classical_disorder",
            "classical_operon_struct",
        ]),
        "fair_target_params": 25_000_000,
        "latent_dim": 1104,
        "hidden_dim": 1104,
        "split_file_name": "hard_hybrid_production_split_v2.tsv",
        "epochs": 30,
        "early_stop_metric": "rmse",
        "lr_schedule": "cosine",
        "lr_warmup_steps": 200,
        "label_mode": "raw",
        "checkpoint_pattern": "results/protex_qc/evo2_7b_profiling/pca_dim_sweep/"
                              "go_f10_7b_fo_pca4096_25M_seed{seed}_checkpoint.pt",
        "validated_seeds": [42, 0, 314, 1, 2, 3],
        "rho_gene_operon": "0.666 ± 0.002 (6s)",
    },

    # ── Previous champion (2026-03-07): 1B F10 ──
    # ρ=0.629 ± 0.003 gene-operon (5 seeds).
    "F10_25M_v2": {
        "fusion_type": "single_adapter",
        "embedders": sorted([
            "esmc_protein",
            "dnabert2_operon_dna",
            "evo2_cds",
            "hyenadna_dna_cds",
            "bacformer",
            "classical_codon",
            "classical_rna_init",
            "classical_protein",
            "classical_disorder",
            "classical_operon_struct",
        ]),
        "fair_target_params": 25_000_000,
        "latent_dim": 1104,
        "hidden_dim": 1104,
        "split_file_name": "hard_hybrid_production_split_v2.tsv",
        "epochs": 30,
        "early_stop_metric": "rmse",
        "lr_schedule": "cosine",
        "lr_warmup_steps": 200,
        "label_mode": "raw",
        "checkpoint_pattern": "results/protex_qc/fraction_scaling/"
                              "f10_go_trainfrac_100_seed{seed}_checkpoint.pt",
        "validated_seeds": [42, 7, 123, 0, 99],
        "rho_gene_operon": "0.629 ± 0.003 (5s)",
    },
}

# Convenience: the current champion key
CURRENT_CHAMPION = "F10_7B_PCA4096"
