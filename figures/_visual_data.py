#!/usr/bin/env python3
"""
Reusable visual-data preparation for manuscript figures and interactive assets.

This module only uses frozen split artifacts that are present in the workspace,
so every downstream visualization is traceable to manuscript-safe data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPECIES_SPLIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "protex_qc"
    / "final_data_freeze_20260219"
    / "splits"
    / "species_cluster_split_t0.20.tsv"
)
GENE_OPERON_FILE = (
    PROJECT_ROOT
    / "results"
    / "protex_qc"
    / "final_data_freeze_20260219"
    / "splits"
    / "hard_hybrid_production_split_v2.tsv"
)
THRESHOLD_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "protex_qc"
    / "manuscript_upgrade"
    / "tables"
    / "species_threshold_summary.csv"
)
MAIN_FREEZE_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "protex_aggregated"
    / "protex_aggregated_v1.1_final_freeze.parquet"
)
EMBEDDINGS_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "protex_aggregated"
    / "embeddings_finalized"
)
VISUAL_DATA_DIR = PROJECT_ROOT / "results" / "protex_qc" / "manuscript_visual_data"
INTERACTIVE_DIR = PROJECT_ROOT / "docs" / "protex" / "interactive"

ATLAS_SAMPLE_SIZE = 5000
SIMILARITY_SAMPLE_SIZE = 2000

EMBEDDING_MODALITIES = {
    "ESM-C protein": {
        "file": "esmc_protein_embeddings.parquet",
        "column": "esmc_protein_embedding",
        "family": "protein",
    },
    "Evo-2 CDS": {
        "file": "evo2_cds_embeddings.parquet",
        "column": "evo2_cds_embedding",
        "family": "dna",
    },
    "Bacformer": {
        "file": "bacformer_embeddings.parquet",
        "column": "bacformer_embedding",
        "family": "genome_context",
    },
    "RiNALMo init": {
        "file": "rinalmo_init_embeddings.parquet",
        "column": "rinalmo_init_embedding",
        "family": "rna",
    },
    "CodonFM CDS": {
        "file": "codonfm_cds_embeddings.parquet",
        "column": "codonfm_cds_embedding",
        "family": "dna",
    },
    "DNABERT-2 operon": {
        "file": "dnabert2_operon_dna_embeddings.parquet",
        "column": "dnabert2_full_operon_dna_embedding",
        "family": "dna",
    },
    "HyenaDNA CDS": {
        "file": "hyenadna_dna_cds_embeddings.parquet",
        "column": "hyenadna_dna_cds_embedding",
        "family": "dna",
    },
}


def _ensure_dirs() -> None:
    VISUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "")
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class VisualDataBundle:
    species: pd.DataFrame
    clusters: pd.DataFrame
    components: pd.DataFrame
    threshold_curve: pd.DataFrame
    modality_points: pd.DataFrame
    modality_family_summary: pd.DataFrame
    sample_metadata: pd.DataFrame
    modality_umap: pd.DataFrame
    modality_similarity: pd.DataFrame
    modality_availability: pd.DataFrame
    paths: Dict[str, Path]


def build_visual_data(force: bool = False) -> VisualDataBundle:
    _ensure_dirs()

    paths = {
        "species": VISUAL_DATA_DIR / "species_summary.csv",
        "clusters": VISUAL_DATA_DIR / "species_cluster_summary.csv",
        "components": VISUAL_DATA_DIR / "component_summary.csv",
        "threshold_curve": VISUAL_DATA_DIR / "species_threshold_curve.csv",
        "modality_points": VISUAL_DATA_DIR / "modality_points.csv",
        "modality_family_summary": VISUAL_DATA_DIR / "modality_family_summary.csv",
        "sample_metadata": VISUAL_DATA_DIR / "embedding_sample_metadata.csv",
        "modality_umap": VISUAL_DATA_DIR / "modality_umap.csv",
        "modality_similarity": VISUAL_DATA_DIR / "modality_similarity.csv",
        "modality_availability": VISUAL_DATA_DIR / "modality_availability.csv",
        "metadata": VISUAL_DATA_DIR / "visualization_metadata.json",
    }

    if force or not all(path.exists() for path in paths.values()):
        species_df = _load_species_split()
        gene_operon_df = _load_gene_operon_split()

        component_assignment = _compute_component_assignment(gene_operon_df)
        gene_operon_df["component_id"] = component_assignment

        components = _build_component_summary(gene_operon_df)
        species = _build_species_summary(species_df, gene_operon_df, components)
        clusters = _build_cluster_summary(species_df, species)
        threshold_curve = _build_threshold_curve()
        modality_points, modality_family_summary = _build_modality_tables()
        sample_metadata = _build_embedding_sample_metadata(gene_operon_df, species_df, components)
        modality_umap = _build_modality_umap(sample_metadata)
        modality_similarity = _build_modality_similarity(sample_metadata)
        modality_availability = _build_modality_availability()

        species.to_csv(paths["species"], index=False)
        clusters.to_csv(paths["clusters"], index=False)
        components.to_csv(paths["components"], index=False)
        threshold_curve.to_csv(paths["threshold_curve"], index=False)
        modality_points.to_csv(paths["modality_points"], index=False)
        modality_family_summary.to_csv(paths["modality_family_summary"], index=False)
        sample_metadata.to_csv(paths["sample_metadata"], index=False)
        modality_umap.to_csv(paths["modality_umap"], index=False)
        modality_similarity.to_csv(paths["modality_similarity"], index=False)
        modality_availability.to_csv(paths["modality_availability"], index=False)

        metadata = {
            "n_species": int(species["species"].nunique()),
            "n_clusters": int(clusters["species_cluster"].nunique()),
            "n_components": int(components["component_id"].nunique()),
            "mega_component_genes": int(components.iloc[0]["gene_count"]),
            "mega_component_species": int(components.iloc[0]["n_species"]),
            "data_sources": {
                "species_split": str(SPECIES_SPLIT_FILE.relative_to(PROJECT_ROOT)),
                "gene_operon_split": str(GENE_OPERON_FILE.relative_to(PROJECT_ROOT)),
                "threshold_curve": str(THRESHOLD_SUMMARY_FILE.relative_to(PROJECT_ROOT)),
                "main_freeze": str(MAIN_FREEZE_FILE.relative_to(PROJECT_ROOT)),
                "embeddings_dir": str(EMBEDDINGS_DIR.relative_to(PROJECT_ROOT)),
            },
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2))

    species = pd.read_csv(paths["species"])
    clusters = pd.read_csv(paths["clusters"])
    components = pd.read_csv(paths["components"])
    threshold_curve = pd.read_csv(paths["threshold_curve"])
    modality_points = pd.read_csv(paths["modality_points"])
    modality_family_summary = pd.read_csv(paths["modality_family_summary"])
    sample_metadata = pd.read_csv(paths["sample_metadata"])
    modality_umap = pd.read_csv(paths["modality_umap"])
    modality_similarity = pd.read_csv(paths["modality_similarity"])
    modality_availability = pd.read_csv(paths["modality_availability"])
    return VisualDataBundle(
        species=species,
        clusters=clusters,
        components=components,
        threshold_curve=threshold_curve,
        modality_points=modality_points,
        modality_family_summary=modality_family_summary,
        sample_metadata=sample_metadata,
        modality_umap=modality_umap,
        modality_similarity=modality_similarity,
        modality_availability=modality_availability,
        paths=paths,
    )


def _load_species_split() -> pd.DataFrame:
    return pd.read_csv(
        SPECIES_SPLIT_FILE,
        sep="\t",
        usecols=[
            "gene_id",
            "species",
            "split",
            "gene_cluster_id",
            "compound_operon_id",
            "source_dataset",
            "species_cluster",
        ],
        dtype=str,
    )


def _load_gene_operon_split() -> pd.DataFrame:
    return pd.read_csv(
        GENE_OPERON_FILE,
        sep="\t",
        usecols=[
            "gene_id",
            "species",
            "split",
            "gene_cluster_id",
            "compound_operon_id",
            "source_dataset",
        ],
        dtype=str,
    )


def _compute_component_assignment(df: pd.DataFrame) -> list[str]:
    uf = UnionFind()
    cluster_ids = df["gene_cluster_id"].fillna("missing_cluster").astype(str).to_numpy()
    operon_ids = df["compound_operon_id"].fillna("missing_operon").astype(str).to_numpy()

    for cluster_id, operon_id in zip(cluster_ids, operon_ids):
        uf.union(f"g::{cluster_id}", f"o::{operon_id}")

    return [uf.find(f"g::{cluster_id}") for cluster_id in cluster_ids]


def _build_component_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("component_id", dropna=False)
        .agg(
            gene_count=("gene_id", "size"),
            n_species=("species", "nunique"),
            n_gene_clusters=("gene_cluster_id", "nunique"),
            n_operons=("compound_operon_id", "nunique"),
        )
        .reset_index()
        .sort_values(["gene_count", "n_species"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["component_rank"] = np.arange(1, len(summary) + 1)
    summary["gene_fraction"] = summary["gene_count"] / summary["gene_count"].sum()
    summary["is_mega"] = summary["gene_count"] >= 5000
    summary["label"] = np.where(
        summary["component_rank"] == 1,
        "Mega-component",
        "Component " + summary["component_rank"].astype(str),
    )
    return summary


def _build_species_summary(
    species_df: pd.DataFrame,
    gene_operon_df: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    mega_component_id = components.iloc[0]["component_id"]
    gene_operon_df["is_mega"] = gene_operon_df["component_id"] == mega_component_id

    species_summary = (
        species_df.groupby("species", dropna=False)
        .agg(
            gene_count=("gene_id", "size"),
            species_split=("split", _mode),
            species_cluster=("species_cluster", _mode),
            n_gene_clusters=("gene_cluster_id", "nunique"),
            n_operons=("compound_operon_id", "nunique"),
            gold_fraction=("source_dataset", lambda s: (s == "v1_gold").mean()),
            expanded_fraction=("source_dataset", lambda s: (s != "v1_gold").mean()),
        )
        .reset_index()
    )
    species_summary["genus"] = species_summary["species"].map(lambda s: s.split("_")[0])

    mega_by_species = (
        gene_operon_df.groupby("species", dropna=False)
        .agg(
            mega_fraction=("is_mega", "mean"),
            gene_operon_test_fraction=("split", lambda s: (s == "test").mean()),
            gene_operon_train_fraction=("split", lambda s: (s == "train").mean()),
        )
        .reset_index()
    )
    species_summary = species_summary.merge(mega_by_species, on="species", how="left")

    cluster_sizes = (
        species_summary.groupby("species_cluster", dropna=False)
        .agg(
            cluster_gene_count=("gene_count", "sum"),
            cluster_species_count=("species", "nunique"),
            cluster_gold_fraction=("gold_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["cluster_gene_count", "cluster_species_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    cluster_sizes["cluster_rank"] = np.arange(1, len(cluster_sizes) + 1)
    species_summary = species_summary.merge(cluster_sizes, on="species_cluster", how="left")

    split_order = {"train": 0, "val": 1, "test": 2}
    species_summary["split_order"] = species_summary["species_split"].map(split_order).fillna(3)
    species_summary = species_summary.sort_values(
        ["split_order", "cluster_rank", "gene_count"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    species_summary["species_order"] = np.arange(1, len(species_summary) + 1)
    return species_summary


def _build_cluster_summary(species_df: pd.DataFrame, species_summary: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = (
        species_df.groupby("species_cluster", dropna=False)
        .agg(
            n_genes=("gene_id", "size"),
            n_species=("species", "nunique"),
            split=("split", _mode),
            gold_fraction=("source_dataset", lambda s: (s == "v1_gold").mean()),
        )
        .reset_index()
    )

    top_species = (
        species_summary.sort_values(["cluster_rank", "gene_count"], ascending=[True, False])
        .groupby("species_cluster", dropna=False)["species"]
        .apply(lambda s: ", ".join(s.head(3)))
        .reset_index(name="top_species")
    )
    cluster_summary = cluster_summary.merge(top_species, on="species_cluster", how="left")
    cluster_summary = cluster_summary.sort_values(["n_genes", "n_species"], ascending=[False, False]).reset_index(drop=True)
    cluster_summary["cluster_rank"] = np.arange(1, len(cluster_summary) + 1)
    cluster_summary["size_bucket"] = pd.cut(
        cluster_summary["n_species"],
        bins=[0, 1, 3, 10, 1000],
        labels=["1 species", "2-3 species", "4-10 species", "11+ species"],
        include_lowest=True,
    )
    return cluster_summary


def _build_threshold_curve() -> pd.DataFrame:
    curve = pd.read_csv(THRESHOLD_SUMMARY_FILE)
    curve["threshold_label"] = curve["threshold"].map(lambda x: f"t={x:.2f}")
    curve["headline_cluster_count"] = curve["threshold"].map(
        {
            0.05: 330,
            0.10: 291,
            0.20: 116,
            0.30: 17,
        }
    )
    return curve


def _build_modality_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.DataFrame(
        [
            {
                "modality": "ESM-C protein",
                "family": "protein",
                "single_rho": 0.572,
                "loo_drop": 0.027,
                "stepwise_gain": 0.000,
                "fusion_rho_after_add": 0.572,
            },
            {
                "modality": "DNABERT-2 operon",
                "family": "dna",
                "single_rho": 0.490,
                "loo_drop": 0.010,
                "stepwise_gain": 0.024,
                "fusion_rho_after_add": 0.595,
            },
            {
                "modality": "HyenaDNA CDS",
                "family": "dna",
                "single_rho": 0.250,
                "loo_drop": 0.018,
                "stepwise_gain": 0.010,
                "fusion_rho_after_add": 0.605,
            },
            {
                "modality": "Bacformer",
                "family": "genome_context",
                "single_rho": 0.532,
                "loo_drop": 0.014,
                "stepwise_gain": 0.010,
                "fusion_rho_after_add": 0.615,
            },
            {
                "modality": "Classical stack",
                "family": "classical",
                "single_rho": 0.518,
                "loo_drop": 0.014,
                "stepwise_gain": 0.012,
                "fusion_rho_after_add": 0.627,
            },
            {
                "modality": "Evo-2 CDS",
                "family": "dna",
                "single_rho": 0.565,
                "loo_drop": 0.008,
                "stepwise_gain": 0.008,
                "fusion_rho_after_add": 0.635,
            },
            {
                "modality": "Classical operon",
                "family": "classical",
                "single_rho": 0.518,
                "loo_drop": 0.009,
                "stepwise_gain": 0.000,
                "fusion_rho_after_add": np.nan,
            },
            {
                "modality": "Classical codon",
                "family": "classical",
                "single_rho": 0.518,
                "loo_drop": 0.007,
                "stepwise_gain": 0.000,
                "fusion_rho_after_add": np.nan,
            },
            {
                "modality": "Classical protein",
                "family": "classical",
                "single_rho": 0.518,
                "loo_drop": 0.003,
                "stepwise_gain": 0.000,
                "fusion_rho_after_add": np.nan,
            },
            {
                "modality": "Classical disorder",
                "family": "classical",
                "single_rho": 0.518,
                "loo_drop": 0.008,
                "stepwise_gain": 0.000,
                "fusion_rho_after_add": np.nan,
            },
        ]
    )
    points["redundancy_score"] = points["single_rho"] - points["loo_drop"]
    points["family_slug"] = points["family"].map(_safe_slug)

    family_summary = pd.DataFrame(
        [
            {"family": "protein", "overall_rho": 0.575, "cluster_weighted_rho": 0.091},
            {"family": "dna", "overall_rho": 0.573, "cluster_weighted_rho": 0.170},
            {"family": "rna", "overall_rho": 0.283, "cluster_weighted_rho": 0.054},
            {"family": "classical", "overall_rho": 0.518, "cluster_weighted_rho": 0.095},
        ]
    )
    family_summary["family_slug"] = family_summary["family"].map(_safe_slug)
    return points, family_summary


def _build_embedding_sample_metadata(
    gene_operon_df: pd.DataFrame,
    species_df: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    main = pd.read_parquet(
        MAIN_FREEZE_FILE,
        columns=["gene_id", "species", "taxid", "expression_level", "source_dataset", "quality_tier"],
    )
    main["gene_id"] = main["gene_id"].astype(str)
    main["species"] = main["species"].astype(str)
    main["source_short"] = np.where(main["source_dataset"].eq("v1_gold"), "PaXDb", "Proxy")

    gene_operon_aug = gene_operon_df[["gene_id", "split", "component_id"]].rename(
        columns={"split": "gene_operon_split"}
    )
    mega_component_id = components.iloc[0]["component_id"]
    gene_operon_aug["is_mega"] = gene_operon_aug["component_id"].eq(mega_component_id)
    species_aug = species_df[["gene_id", "split", "species_cluster"]].rename(
        columns={"split": "species_split"}
    )

    merged = (
        main.merge(gene_operon_aug, on="gene_id", how="left")
        .merge(species_aug, on="gene_id", how="left")
        .drop_duplicates("gene_id")
    )
    merged["expression_quartile"] = pd.qcut(
        merged["expression_level"],
        4,
        labels=["Q1 low", "Q2", "Q3", "Q4 high"],
        duplicates="drop",
    )
    merged["is_mega_label"] = np.where(merged["is_mega"], "Mega", "Non-mega")
    merged["quality_tier"] = merged["quality_tier"].fillna("unknown").astype(str)

    cluster_counts = merged["species_cluster"].value_counts()
    top_clusters = set(cluster_counts.head(6).index.tolist())
    merged["cluster_group"] = np.where(
        merged["species_cluster"].isin(top_clusters),
        merged["species_cluster"],
        "other clusters",
    )

    sample = _stratified_sample(
        merged,
        n_total=ATLAS_SAMPLE_SIZE,
        group_cols=["expression_quartile", "source_short", "species_split"],
        random_state=42,
    ).reset_index(drop=True)
    sample["sample_rank"] = np.arange(1, len(sample) + 1)
    return sample


def _build_modality_umap(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    gene_ids = sample_metadata["gene_id"].astype(str).tolist()
    meta_cols = [
        "gene_id", "species", "taxid", "expression_level", "expression_quartile",
        "source_short", "quality_tier", "species_split", "gene_operon_split",
        "species_cluster", "cluster_group", "is_mega_label",
    ]
    results = []
    for modality, config in list(EMBEDDING_MODALITIES.items())[:5]:
        vectors = _load_embedding_matrix(config["file"], config["column"], gene_ids)
        coords = _project_umap(vectors)
        frame = sample_metadata[meta_cols].copy()
        frame["modality"] = modality
        frame["family"] = config["family"]
        frame["umap1"] = coords[:, 0]
        frame["umap2"] = coords[:, 1]
        results.append(frame)

    fusion10_path = VISUAL_DATA_DIR / "fusion10_fused_embeddings_sample.parquet"
    if fusion10_path.exists():
        f10 = pd.read_parquet(fusion10_path)
        f10["gene_id"] = f10["gene_id"].astype(str)
        f10_ids = set(f10["gene_id"])
        common = [gid for gid in gene_ids if gid in f10_ids]
        if len(common) >= 500:
            f10_sub = f10[f10["gene_id"].isin(set(common))].set_index("gene_id")
            f10_sub = f10_sub.loc[[g for g in common if g in f10_sub.index]]
            vecs = np.stack(f10_sub["fusion10_embedding"].tolist()).astype(np.float32)
            nan_mask = np.isnan(vecs).any(axis=1)
            if nan_mask.any():
                vecs[nan_mask] = 0.0
            coords = _project_umap(vecs)
            sub_meta = sample_metadata.copy()
            sub_meta["_gid"] = sub_meta["gene_id"].astype(str)
            sub_meta = sub_meta[sub_meta["_gid"].isin(set(f10_sub.index))].copy()
            sub_meta = sub_meta.set_index("_gid").loc[f10_sub.index].reset_index(drop=True)
            frame = sub_meta[meta_cols].head(len(coords)).copy()
            frame["modality"] = "Fusion-10"
            frame["family"] = "fusion"
            frame["umap1"] = coords[:, 0]
            frame["umap2"] = coords[:, 1]
            results.append(frame)

    return pd.concat(results, ignore_index=True)


def _build_modality_similarity(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    gene_ids = _stratified_sample(
        sample_metadata,
        n_total=SIMILARITY_SAMPLE_SIZE,
        group_cols=["expression_quartile", "source_short"],
        random_state=7,
    )["gene_id"].astype(str).tolist()

    matrices: Dict[str, np.ndarray] = {}
    for modality, config in EMBEDDING_MODALITIES.items():
        matrices[modality] = _reduce_for_similarity(
            _load_embedding_matrix(config["file"], config["column"], gene_ids)
        )

    rows = []
    for left_name, left_matrix in matrices.items():
        for right_name, right_matrix in matrices.items():
            rows.append(
                {
                    "modality_a": left_name,
                    "modality_b": right_name,
                    "cka_similarity": _linear_cka(left_matrix, right_matrix),
                    "family_a": EMBEDDING_MODALITIES[left_name]["family"],
                    "family_b": EMBEDDING_MODALITIES[right_name]["family"],
                }
            )
    return pd.DataFrame(rows)


def _build_modality_availability() -> pd.DataFrame:
    records = []
    total_rows = 492026
    for modality, config in EMBEDDING_MODALITIES.items():
        file_path = EMBEDDINGS_DIR / config["file"]
        table = ds.dataset(file_path, format="parquet")
        records.append(
            {
                "modality": modality,
                "family": config["family"],
                "file": config["file"],
                "n_rows": table.count_rows(),
                "coverage_fraction": table.count_rows() / total_rows,
            }
        )
    return pd.DataFrame(records).sort_values(["family", "modality"]).reset_index(drop=True)


def _stratified_sample(
    df: pd.DataFrame,
    n_total: int,
    group_cols: list[str],
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    pieces = []
    groups = list(df.groupby(group_cols, dropna=False))
    target_each = max(1, n_total // max(len(groups), 1))
    chosen = set()

    for _, grp in groups:
        n = min(len(grp), target_each)
        if n == 0:
            continue
        idx = rng.choice(grp.index.to_numpy(), size=n, replace=False)
        chosen.update(idx.tolist())
        pieces.append(df.loc[idx])

    sampled = pd.concat(pieces, axis=0).drop_duplicates("gene_id") if pieces else df.head(0).copy()
    if len(sampled) < n_total:
        remaining = df.loc[~df.index.isin(chosen)]
        if len(remaining) > 0:
            extra_n = min(n_total - len(sampled), len(remaining))
            extra = remaining.sample(extra_n, random_state=random_state)
            sampled = pd.concat([sampled, extra], axis=0)

    return sampled.head(n_total).copy()


def _load_embedding_matrix(filename: str, column: str, gene_ids: list[str]) -> np.ndarray:
    dataset = ds.dataset(EMBEDDINGS_DIR / filename, format="parquet")
    table = dataset.to_table(
        columns=["gene_id", column],
        filter=pc.field("gene_id").isin(gene_ids),
    )
    frame = table.to_pandas()
    frame["gene_id"] = frame["gene_id"].astype(str)
    frame = frame.set_index("gene_id").reindex(gene_ids)
    values = frame[column].tolist()
    first_non_null = next(v for v in values if isinstance(v, (list, np.ndarray)))
    dim = len(first_non_null)
    matrix = np.zeros((len(gene_ids), dim), dtype=np.float32)
    for i, value in enumerate(values):
        if isinstance(value, np.ndarray):
            matrix[i] = value.astype(np.float32, copy=False)
        elif isinstance(value, list):
            matrix[i] = np.asarray(value, dtype=np.float32)
    return matrix


def _project_umap(matrix: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    n_comp = min(50, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=n_comp, random_state=42, svd_solver="randomized").fit_transform(scaled)
    projector = umap.UMAP(
        n_neighbors=30,
        min_dist=0.12,
        metric="cosine",
        random_state=42,
        transform_seed=42,
    )
    return projector.fit_transform(reduced).astype(np.float32)


def _reduce_for_similarity(matrix: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(matrix)
    n_comp = min(32, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=n_comp, random_state=42, svd_solver="randomized").fit_transform(scaled)
    reduced = reduced - reduced.mean(axis=0, keepdims=True)
    return reduced.astype(np.float32)


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denom_left = np.linalg.norm(left.T @ left, ord="fro")
    denom_right = np.linalg.norm(right.T @ right, ord="fro")
    denom = max(denom_left * denom_right, 1e-12)
    return float(numerator / denom)


def _mode(values: Iterable[str]) -> str:
    series = pd.Series(list(values), dtype="object")
    if series.empty:
        return ""
    mode = series.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(series.iloc[0])


def main() -> None:
    build_visual_data(force=True)


if __name__ == "__main__":
    main()
