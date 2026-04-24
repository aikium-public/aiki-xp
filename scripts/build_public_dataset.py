#!/usr/bin/env python3
"""Build the public Aiki-XP dataset from the production table.

Resolves all nulls, coalesces protein_sequence, corrects the init window,
fills expression_source, recovers strand and locus_tag from genome pickles.

Usage:
    python scripts/build_public_dataset.py \
        --production-table datasets/protex_aggregated/protex_aggregated_v1.1_final_freeze.parquet \
        --split-file results/protex_qc/final_data_freeze_20260219/splits/hard_hybrid_production_split_v2.tsv \
        --v112-mapping results/protex_qc/bacformer_mapping_v1.1.2/bacformer_resolved_mapping_v1.1.2.v1gold_pseudogene_fix.parquet \
        --genome-cache cache/protex/genomes/ \
        --output-dir data/
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Init window spec (Nieuwkoop NAR 2023 asymmetric) ──
INIT_UPSTREAM = 35
INIT_DOWNSTREAM = 25
INIT_TARGET_LEN = INIT_UPSTREAM + INIT_DOWNSTREAM  # 60

# ── V1 gold expression source per species ──
# From PROTEX_DATASET_V1.md lines 143-154 and PROTEX_DATASET_V2.md
# MassIVE MSV000096603 is the Abele study's data deposit — same source as
# the v2_expansion "abele_calibrated" genes, just pulled earlier for v1_gold.
# Only Legionella (GPMDB 272624) is a genuinely separate source.
V1_EXPRESSION_SOURCE = {
    "Cronobacter_sakazakii": "abele_calibrated",
    "Shewanella_putrefaciens": "abele_calibrated",
    "Legionella_pneumophila": "gpmdb_272624",
}
V1_DEFAULT_SOURCE = "paxdb_v6"


def coalesce_protein_sequence(df: pd.DataFrame) -> pd.Series:
    seq = df["protein_sequence"].copy()
    mask = seq.isna()
    seq.loc[mask] = df.loc[mask, "protein_sequence_paxdb"]
    n_null = seq.isna().sum()
    if n_null > 0:
        raise ValueError(f"protein_sequence has {n_null} NaN after coalesce")
    return seq


def fix_init_window(df: pd.DataFrame) -> pd.Series:
    """Recompute rna_init_window_seq as 60nt (35 up + 25 down) for ALL rows.

    The production table has mixed 60/70nt windows (v2=60, v1_gold=70).
    Recomputing from full_operon_dna + cds_start_in_operon gives a uniform
    60nt window matching the corrected init60 pipeline.
    """
    seqs = []
    for i in range(len(df)):
        operon = df["full_operon_dna"].iat[i]
        cs = int(df["cds_start_in_operon"].iat[i])
        op_len = len(operon)
        start = max(0, cs - INIT_UPSTREAM)
        end = min(op_len, cs + INIT_DOWNSTREAM)
        win = operon[start:end]
        seqs.append(win.replace("T", "U").replace("t", "u"))
    result = pd.Series(seqs, index=df.index)
    lengths = result.str.len()
    print(f"  Init window lengths: min={lengths.min()}, median={int(lengths.median())}, max={lengths.max()}")
    n_target = (lengths == INIT_TARGET_LEN).sum()
    print(f"  Exactly {INIT_TARGET_LEN}nt: {n_target:,} / {len(df):,} ({n_target/len(df)*100:.1f}%)")
    return result


def fill_expression_source(df: pd.DataFrame) -> pd.Series:
    """Fill expression_source for v1_gold genes by species lookup."""
    src = df["expression_source"].copy()
    v1_mask = src.isna()
    for idx in df.index[v1_mask]:
        species = df.at[idx, "species"]
        src.at[idx] = V1_EXPRESSION_SOURCE.get(species, V1_DEFAULT_SOURCE)
    n_null = src.isna().sum()
    if n_null > 0:
        raise ValueError(f"expression_source still has {n_null} NaN after fill")
    return src


def derive_num_genes_in_operon(df: pd.DataFrame) -> pd.Series:
    """Count genes per operon group — same method used during featurization."""
    return df.groupby(["operon_source", "taxid", "operon_id"])["gene_id"].transform("count").astype(int)


def recover_strand_and_locus_tag(
    df: pd.DataFrame, v112_mapping: pd.DataFrame, genome_cache: Path
) -> tuple[pd.Series, pd.Series]:
    """Recover strand and locus_tag for v1_gold genes from genome pickles.

    Uses the corrected v1.1.2 CDS indices to look up CDS features in the
    BioPython SeqRecord genome pickles.
    """
    strand_col = df["strand"].copy() if "strand" in df.columns else pd.Series("", index=df.index)
    lt_col = df["locus_tag"].copy() if "locus_tag" in df.columns else pd.Series("", index=df.index)

    v1_mask = strand_col.isna() | (strand_col == "")
    if v1_mask.sum() == 0:
        return strand_col, lt_col

    merged = df.loc[v1_mask, ["gene_id"]].merge(
        v112_mapping[["gene_id", "bacformer_resolved_pickle_v112", "bacformer_resolved_cds_index_v112"]],
        on="gene_id", how="left"
    )

    recovered = 0
    missing_pkl = 0
    genome_cache_dict = {}

    for pkl_name, grp in merged.groupby("bacformer_resolved_pickle_v112"):
        pkl_path = genome_cache / f"{pkl_name}.pkl"
        if not pkl_path.exists():
            missing_pkl += len(grp)
            continue

        if pkl_name not in genome_cache_dict:
            with open(pkl_path, "rb") as f:
                genome = pickle.load(f)
            cds_feats = [
                feat for feat in genome.features
                if feat.type == "CDS"
                and "translation" in feat.qualifiers
                and "pseudo" not in feat.qualifiers
            ]
            genome_cache_dict[pkl_name] = cds_feats

        cds_feats = genome_cache_dict[pkl_name]
        for _, row in grp.iterrows():
            idx = int(row["bacformer_resolved_cds_index_v112"])
            if idx < len(cds_feats):
                feat = cds_feats[idx]
                orig_idx = df.index[df["gene_id"] == row["gene_id"]]
                if len(orig_idx) == 1:
                    strand_col.at[orig_idx[0]] = "+" if feat.location.strand == 1 else "-"
                    lt_col.at[orig_idx[0]] = feat.qualifiers.get("locus_tag", [""])[0]
                    recovered += 1

    print(f"  Strand/locus_tag recovered: {recovered:,}")
    if missing_pkl > 0:
        print(f"  WARNING: {missing_pkl} genes from missing genome pickles")

    return strand_col, lt_col


def main():
    parser = argparse.ArgumentParser(description="Build public Aiki-XP dataset")
    parser.add_argument("--production-table", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--v112-mapping", required=True, type=Path)
    parser.add_argument("--genome-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.production_table}...")
    df = pd.read_parquet(args.production_table)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print(f"Loading v1.1.2 mapping...")
    v112 = pd.read_parquet(args.v112_mapping)
    print(f"  {len(v112)} rows")

    # ── 1. Coalesce protein sequence ──
    print("\n1. Coalescing protein_sequence...")
    protein_seq = coalesce_protein_sequence(df)

    # ── 2. Fix init window to uniform 60nt ──
    print("\n2. Recomputing rna_init_window_seq (uniform 60nt)...")
    init_window = fix_init_window(df)

    # ── 3. Fill expression_source ──
    print("\n3. Filling expression_source for v1_gold...")
    expr_source = fill_expression_source(df)
    print(f"  Values: {expr_source.value_counts().to_dict()}")

    # ── 4. Derive num_genes_in_operon ──
    print("\n4. Deriving num_genes_in_operon...")
    num_genes = derive_num_genes_in_operon(df)
    print(f"  Range: {num_genes.min()} to {num_genes.max()}, median={num_genes.median():.0f}")

    # ── 5. Recover strand and locus_tag from genome pickles ──
    print("\n5. Recovering strand and locus_tag from genome pickles...")
    strand, locus_tag = recover_strand_and_locus_tag(df, v112, args.genome_cache)

    # ── 6. Genome accession from v1.1.2 mapping (100% coverage) ──
    print("\n6. Filling genome_accession from v1.1.2 mapping...")
    genome_acc = df[["gene_id"]].merge(
        v112[["gene_id", "bacformer_resolved_pickle_v112"]], on="gene_id", how="left"
    )["bacformer_resolved_pickle_v112"]
    genome_acc.index = df.index
    print(f"  Non-null: {genome_acc.notna().sum():,}")

    # ── 7. Load split assignments ──
    print(f"\n7. Loading split assignments from {args.split_file}...")
    splits = pd.read_csv(args.split_file, sep="\t", low_memory=False)
    split_cols = ["gene_id"]
    if "gene_cluster_id" in splits.columns:
        split_cols.append("gene_cluster_id")
    if "compound_operon_id" in splits.columns:
        split_cols.append("compound_operon_id")
    splits = splits[split_cols]

    # ── Build output table ──
    print("\n8. Building output table...")
    out = pd.DataFrame({
        "gene_id": df["gene_id"],
        "species": df["species"],
        "taxid": df["taxid"],
        "locus_tag": locus_tag,
        "protein_sequence": protein_seq,
        "protein_length": protein_seq.str.len().astype(int),
        "source_dataset": df["source_dataset"],
        "expression_level": df["expression_level"],
        "expression_level_native": df["expression_level_native"],
        "expression_source": expr_source,
        "dna_cds_seq": df["dna_cds_seq"],
        "full_operon_dna": df["full_operon_dna"],
        "rna_init_window_seq": init_window,
        "cds_start_in_operon": df["cds_start_in_operon"],
        "upstream_context_seq": df["upstream_context_seq"],
        "upstream_context_length_nt": df["upstream_context_length_nt"],
        "downstream_context_seq": df["downstream_context_seq"],
        "downstream_context_length_nt": df["downstream_context_length_nt"],
        "operon_id": df["operon_id"],
        "num_genes_in_operon": num_genes,
        "strand": strand,
        "genome_accession": genome_acc,
    })

    # Merge split assignments
    n_before = len(out)
    out = out.merge(splits, on="gene_id", how="left")
    assert len(out) == n_before

    # ── Validate: zero nulls in critical columns ──
    print("\n9. Validation...")
    critical = [
        "gene_id", "species", "taxid", "protein_sequence", "expression_level",
        "expression_source", "dna_cds_seq", "full_operon_dna", "rna_init_window_seq",
        "cds_start_in_operon", "operon_id", "num_genes_in_operon", "genome_accession",
        "gene_cluster_id", "compound_operon_id",
    ]
    all_clean = True
    for col in critical:
        if col in out.columns:
            n_null = out[col].isna().sum()
            if n_null > 0:
                print(f"  FAIL: {col} has {n_null} NaN")
                all_clean = False

    # Report columns with remaining nulls (non-critical)
    for col in out.columns:
        if col not in critical:
            n_null = out[col].isna().sum()
            n_empty = (out[col] == "").sum() if out[col].dtype == object else 0
            if n_null > 0 or n_empty > 0:
                print(f"  INFO: {col} has {n_null} NaN + {n_empty} empty")

    # ── Write ──
    pq_path = args.output_dir / "aikixp_492k_v1.parquet"
    out.to_parquet(pq_path, index=False)
    pq_sha = hashlib.sha256(pq_path.read_bytes()).hexdigest()
    print(f"\nWrote {pq_path} ({pq_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  SHA256: {pq_sha}")

    csv_cols = [c for c in out.columns if not c.endswith("_seq") and c != "protein_sequence"]
    csv_path = args.output_dir / "aikixp_492k_v1_metadata.csv"
    out[csv_cols].to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Genes: {len(out):,}")
    print(f"Species: {out['species'].nunique()}")
    print(f"Columns: {len(out.columns)}")
    print(f"Expression sources: {out['expression_source'].value_counts().to_dict()}")
    print(f"Init window lengths: {out['rna_init_window_seq'].str.len().value_counts().to_dict()}")
    print(f"Strand: {out['strand'].value_counts(dropna=False).head().to_dict()}")
    print(f"Locus tag coverage: {(out['locus_tag'].notna() & (out['locus_tag'] != '')).sum():,} / {len(out):,}")
    if all_clean:
        print("\nAll critical columns have zero NaN.")
    else:
        print("\nWARNING: Some critical columns have NaN — check output above.")


if __name__ == "__main__":
    main()
