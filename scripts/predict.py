#!/usr/bin/env python3
"""Aiki-XP end-to-end prediction CLI.

Accepts pre-extracted embedding parquets and runs XP5Ensemble inference.
For raw sequences, use the embedding extraction scripts first (see README).

Examples:
    # Tier A (protein only) — requires ESM-C + ProtT5 embeddings
    python scripts/predict.py \
        --tier A \
        --embeddings esmc_protein=emb/esmc.parquet prot_t5_xl_protein=emb/prott5.parquet \
        --output predictions.csv

    # Tier D (full native) — requires all 9 modalities
    python scripts/predict.py \
        --tier D \
        --embed-dir data/embeddings/ \
        --gene-ids genes.txt \
        --output predictions.csv

    # Auto-detect tier from available embeddings
    python scripts/predict.py \
        --embed-dir data/embeddings/ \
        --output predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aikixp.inference import XP5Ensemble, TIER_CONFIG_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TIER_TO_RECIPE = {
    "A": "esmc_prott5_seed42",
    "B": "deploy_protein_cds_features_6mod_seed42",
    "B+": "tier_b_evo2_init_window_classical_rna_init_prott5_seed42",
    "C": "evo2_prott5_seed42",
    "D": "balanced_nonmega_5mod",
}

TIER_MODALITIES = {
    "A": ["esmc_protein", "prot_t5_xl_protein"],
    "B": ["esmc_protein", "hyenadna_dna_cds", "codonfm_cds",
           "classical_codon", "classical_protein", "classical_disorder"],
    "B+": ["esmc_protein", "hyenadna_dna_cds", "codonfm_cds",
            "classical_codon", "classical_protein", "classical_disorder",
            "prot_t5_xl_protein", "evo2_init_window", "classical_rna_init"],
    "C": ["evo2_7b_full_operon_pca4096", "prot_t5_xl_protein"],
    "D": ["evo2_7b_full_operon_pca4096", "hyenadna_dna_cds",
           "bacformer_large", "classical_codon", "classical_rna_init",
           "classical_protein", "classical_disorder",
           "classical_operon_struct", "esmc_protein"],
}

MODALITY_PARQUET_NAMES = {
    "esmc_protein": "esmc_protein_embeddings.parquet",
    "prot_t5_xl_protein": "prot_t5_xl_protein_embeddings.parquet",
    "hyenadna_dna_cds": "hyenadna_dna_cds_embeddings.parquet",
    "codonfm_cds": "codonfm_cds_embeddings.parquet",
    "evo2_7b_full_operon_pca4096": "evo2_7b_full_operon_pca4096.parquet",
    "evo2_init_window": "evo2_init_window_embeddings.parquet",
    "bacformer_large": "bacformer_large_embeddings.parquet",
    "classical_codon": "classical_codon_features.parquet",
    "classical_rna_init": "classical_rna_thermo_features.parquet",
    "classical_protein": "classical_protein_features.parquet",
    "classical_disorder": "classical_disorder_features.parquet",
    "classical_operon_struct": "classical_operon_structural_features.parquet",
}


def load_embeddings_from_dir(
    embed_dir: Path, modalities: list[str], gene_ids: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Load embedding parquets from a directory, aligned by gene_id."""
    arrays = {}
    reference_ids = None

    for mod in modalities:
        parquet_name = MODALITY_PARQUET_NAMES.get(mod)
        if parquet_name is None:
            log.warning("No known parquet name for modality '%s', skipping", mod)
            continue

        path = embed_dir / parquet_name
        if not path.exists():
            log.warning("Embedding not found: %s (will be zero-filled)", path.name)
            continue

        df = pd.read_parquet(path)
        if "gene_id" not in df.columns:
            raise ValueError(f"{path.name} has no gene_id column")

        if reference_ids is None:
            if gene_ids is not None:
                reference_ids = gene_ids
            else:
                reference_ids = df["gene_id"].values

        df = df.set_index("gene_id")
        df = df.reindex(reference_ids)

        feature_cols = [c for c in df.columns if c != "gene_id"]
        if len(feature_cols) == 1 and df[feature_cols[0]].dtype == object:
            arr = np.stack(df[feature_cols[0]].apply(
                lambda x: x if isinstance(x, np.ndarray) else np.zeros(1)
            ).values).astype(np.float32)
        else:
            arr = df[feature_cols].values.astype(np.float32)

        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            log.warning("%s: %d NaN values (genes not in embedding file)", mod, n_nan)
            arr = np.nan_to_num(arr, nan=0.0)

        arrays[mod] = arr
        log.info("  %s: %d genes x %d dims", mod, arr.shape[0], arr.shape[1])

    return arrays, reference_ids


def load_embeddings_from_args(
    embed_args: list[str],
) -> dict[str, np.ndarray]:
    """Parse modality=path pairs from CLI arguments."""
    arrays = {}
    reference_ids = None

    for arg in embed_args:
        if "=" not in arg:
            raise ValueError(f"Expected modality=path, got: {arg}")
        mod, path_str = arg.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        df = pd.read_parquet(path)
        if "gene_id" not in df.columns:
            raise ValueError(f"{path.name} has no gene_id column")

        if reference_ids is None:
            reference_ids = df["gene_id"].values

        df = df.set_index("gene_id").reindex(reference_ids)
        feature_cols = [c for c in df.columns if c != "gene_id"]
        if len(feature_cols) == 1 and df[feature_cols[0]].dtype == object:
            arr = np.stack(df[feature_cols[0]].apply(
                lambda x: x if isinstance(x, np.ndarray) else np.zeros(1)
            ).values).astype(np.float32)
        else:
            arr = df[feature_cols].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arrays[mod] = arr
        log.info("  %s: %d genes x %d dims from %s", mod, arr.shape[0], arr.shape[1], path.name)

    return arrays, reference_ids


def detect_tier(available_modalities: set[str]) -> str:
    """Pick the highest tier whose required modalities are all available."""
    for tier in ["D", "C", "B+", "B", "A"]:
        required = set(TIER_MODALITIES[tier])
        if required.issubset(available_modalities):
            return tier
    return "A"


def main():
    parser = argparse.ArgumentParser(
        description="Aiki-XP expression prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tier", choices=["A", "B", "B+", "C", "D"],
                        help="Deployment tier (auto-detected if --embed-dir used)")
    parser.add_argument("--recipe", help="Override recipe name (advanced)")
    parser.add_argument("--embed-dir", type=Path,
                        help="Directory of embedding parquets (auto-loads by modality name)")
    parser.add_argument("--embeddings", nargs="+", metavar="MOD=PATH",
                        help="Explicit modality=path pairs")
    parser.add_argument("--gene-ids", type=Path,
                        help="File with gene_id column or one ID per line (subset)")
    parser.add_argument("--ckpt-dir", type=Path,
                        help="Checkpoint directory (default: checkpoints/)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output CSV path")
    args = parser.parse_args()

    if not args.embed_dir and not args.embeddings:
        parser.error("Provide --embed-dir or --embeddings")

    gene_ids = None
    if args.gene_ids:
        gf = pd.read_csv(args.gene_ids)
        if "gene_id" in gf.columns:
            gene_ids = gf["gene_id"].values
        else:
            gene_ids = gf.iloc[:, 0].values
        log.info("Subsetting to %d genes from %s", len(gene_ids), args.gene_ids)

    if args.embed_dir:
        all_mods = set()
        for mod, pq_name in MODALITY_PARQUET_NAMES.items():
            if (args.embed_dir / pq_name).exists():
                all_mods.add(mod)
        log.info("Found %d modalities in %s", len(all_mods), args.embed_dir)

        tier = args.tier or detect_tier(all_mods)
        log.info("Using Tier %s", tier)

        needed = TIER_MODALITIES[tier]
        arrays, ref_ids = load_embeddings_from_dir(args.embed_dir, needed, gene_ids)
    else:
        arrays, ref_ids = load_embeddings_from_args(args.embeddings)
        tier = args.tier or detect_tier(set(arrays.keys()))
        log.info("Using Tier %s", tier)

    recipe = args.recipe or TIER_TO_RECIPE[tier]
    log.info("Recipe: %s", recipe)

    ckpt_dir_env = args.ckpt_dir
    if ckpt_dir_env:
        import os
        os.environ["AIKIXP_CKPT_DIR"] = str(ckpt_dir_env)

    model = XP5Ensemble(recipe, device=args.device)
    predictions = model.predict(arrays)

    out_df = pd.DataFrame({"gene_id": ref_ids, "predicted_expression": predictions})
    out_df = out_df.sort_values("predicted_expression", ascending=False)
    out_df["rank"] = range(1, len(out_df) + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    log.info("Wrote %d predictions to %s", len(out_df), args.output)
    log.info("  Prediction range: [%.3f, %.3f], std=%.3f",
             predictions.min(), predictions.max(), predictions.std())


if __name__ == "__main__":
    main()
