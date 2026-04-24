#!/usr/bin/env python3
"""
V2.1 Temporal Holdout: Zero-shot inference + familiarity analysis.

Runs the frozen V2 champion model (F10, trained on 492K genes) on the
124K V2.1 recovery genes it has never seen. Removes exact protein/CDS
duplicates to avoid memorization, then reports Spearman rho stratified
by similarity to training data.

Self-contained script for the A100 node. Requires:
  - V2.1 production table + embeddings (from featurize_v21_a100.py)
  - V2 production table (for dedup)
  - Champion checkpoint
  - V2 split file (to identify training genes)

Usage:
  python scripts/protex/predict_v21_temporal_holdout.py
  python scripts/protex/predict_v21_temporal_holdout.py --skip-familiarity
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/protex"))

V21_TABLE = PROJECT_ROOT / "datasets/protex_aggregated_v21/protex_aggregated_v2.1_production.parquet"
V2_TABLE = PROJECT_ROOT / "datasets/protex_aggregated/protex_aggregated_v1.1_final_freeze.parquet"
V2_EMBED_DIR = PROJECT_ROOT / "datasets/protex_aggregated/embeddings_finalized"
CKPT_PATH = PROJECT_ROOT / "results/protex_qc/fraction_scaling/f10_go_trainfrac_100_seed42_checkpoint.pt"
SPLIT_FILE = PROJECT_ROOT / "results/protex_qc/final_data_freeze_20260219/splits/hard_hybrid_production_split_v2.tsv"
OUT_DIR = PROJECT_ROOT / "results/protex_v21_recovery/temporal_holdout"

MODALITY_FILE_CANDIDATES = {
    "esmc_protein": ["esmc_protein_embeddings.parquet"],
    "dnabert2_operon_dna": ["dnabert2_operon_dna_embeddings.parquet"],
    "hyenadna_dna_cds": ["hyenadna_dna_cds_embeddings.parquet"],
    "bacformer": ["bacformer_embeddings.parquet"],
    "classical_codon": ["classical_codon_features.parquet", "classical_codon_embeddings.parquet"],
    "classical_rna_init": ["classical_rna_thermo_features.parquet", "classical_rna_init_embeddings.parquet"],
    "classical_protein": ["classical_protein_features.parquet", "classical_protein_embeddings.parquet"],
    "classical_disorder": ["classical_disorder_features.parquet", "classical_disorder_embeddings.parquet"],
    "classical_operon_struct": [
        "classical_operon_structural_features.parquet",
        "classical_operon_struct_embeddings.parquet",
    ],
    "evo2_cds": ["evo2_cds_embeddings.parquet"],
}


def resolve_v21_embed_dir() -> Path:
    """Prefer merged V2.1 embeddings on analysis nodes, fall back to delta dir."""
    candidates = [
        PROJECT_ROOT / "datasets/protex_aggregated_v21/embeddings_merged",
        PROJECT_ROOT / "datasets/protex_aggregated_v21/embeddings",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find V2.1 embeddings directory. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def _load_vectors_from_frame(df: pd.DataFrame) -> np.ndarray:
    emb_col = [c for c in df.columns if c.endswith("_embedding")]
    if emb_col:
        return np.stack(df[emb_col[0]].map(lambda x: np.asarray(x, dtype=np.float32)).values)
    num_cols = [c for c in df.columns if c != "gene_id"]
    return df[num_cols].values.astype(np.float32)


def load_embeddings(embed_dir: Path, gene_ids: set, input_dims: dict[str, int]) -> dict:
    """Load only the required modality files, handling both embedding and feature schemas."""
    embeddings = {}
    for mod_name in input_dims:
        candidates = MODALITY_FILE_CANDIDATES.get(mod_name, [f"{mod_name}_embeddings.parquet"])
        fpath = next((embed_dir / name for name in candidates if (embed_dir / name).exists()), None)
        if fpath is None:
            continue
        df = pd.read_parquet(fpath)
        df = df[df["gene_id"].isin(gene_ids)]
        vectors = _load_vectors_from_frame(df)
        embeddings[mod_name] = {"gene_ids": df["gene_id"].values, "vectors": vectors}
    return embeddings


def build_tensors(gene_ids, embeddings, input_dims, zero_fill_allowed=None):
    """Align embeddings to gene_id order and fail if required modalities are missing.

    Parameters
    ----------
    zero_fill_allowed : set[str] or None
        Modality names explicitly permitted to be zero-filled when missing.
        Passed via --zero-fill-modalities on the command line. If a modality
        has 0 coverage and is NOT in this set, the script crashes.
    """
    if zero_fill_allowed is None:
        zero_fill_allowed = set()

    n = len(gene_ids)
    gid_to_idx = {g: i for i, g in enumerate(gene_ids)}

    tensors = {}
    coverage = {}
    for mod_name, dim in input_dims.items():
        tensor = np.zeros((n, dim), dtype=np.float32)
        filled = 0
        if mod_name in embeddings:
            emb = embeddings[mod_name]
            for j, gid in enumerate(emb["gene_ids"]):
                if gid in gid_to_idx:
                    idx = gid_to_idx[gid]
                    vec = emb["vectors"][j]
                    tensor[idx] = vec[:dim] if len(vec) >= dim else np.pad(vec, (0, dim - len(vec)))
                    filled += 1
        if filled == 0:
            if mod_name in zero_fill_allowed:
                print(f"  ZERO-FILLED: {mod_name}: 0/{n} genes — explicitly allowed via --zero-fill-modalities")
            else:
                raise RuntimeError(
                    f"Required modality '{mod_name}' had 0/{n} filled genes. "
                    f"Check that the correct embedding directory is present and compatible "
                    f"with the frozen V2 champion checkpoint. "
                    f"If this modality is genuinely unavailable, pass "
                    f"--zero-fill-modalities {mod_name} to zero-fill it explicitly."
                )
        elif filled < n:
            print(f"  WARNING: {mod_name}: {filled}/{n} genes filled ({n - filled} zero-filled)")
        else:
            print(f"  {mod_name}: {filled}/{n} (100%), dim={dim}")
        tensors[mod_name] = torch.tensor(tensor)
        coverage[mod_name] = filled
    return tensors, coverage


def load_model(device, ckpt_path=None):
    """Load a frozen champion model checkpoint."""
    from train_fusion import FusionModel, TrainConfig

    ckpt_path = ckpt_path or CKPT_PATH
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_dims = ckpt["input_dims"]
    config = ckpt["config"]

    tc = TrainConfig(
        fusion_type=ckpt.get("fusion_type", "single_adapter"),
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config.get("dropout", 0.1),
        num_layers=config.get("num_layers", 2),
    )
    model = FusionModel(tc, input_dims)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Champion model: {n_params:,} params, input_dims={list(input_dims.keys())}")
    return model, input_dims


def predict(model, tensors, input_dims, device, batch_size=512):
    """Run batched inference."""
    n = next(iter(tensors.values())).shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            batch = {k: tensors[k][i:j].to(device) for k in input_dims}
            out = model(batch).cpu().numpy().flatten()
            preds.extend(out)
    return np.array(preds)


def dedup_against_v2(recovery: pd.DataFrame) -> pd.DataFrame:
    """Remove recovery genes with exact protein or CDS match to V2."""
    print("\nDeduplicating against V2 production table...")
    v2 = pd.read_parquet(V2_TABLE, columns=["protein_sequence", "protein_sequence_paxdb", "dna_cds_seq"])
    v2["protein"] = v2["protein_sequence"].fillna(v2.get("protein_sequence_paxdb", ""))
    v2_proteins = set(v2["protein"].dropna().values) - {""}
    v2_cds = set(v2["dna_cds_seq"].dropna().values) - {""}

    prot_match = recovery["protein_sequence"].fillna("").isin(v2_proteins)
    cds_match = recovery["dna_cds_seq"].fillna("").isin(v2_cds)
    either = prot_match | cds_match

    clean = recovery[~either].copy()
    print(f"  Removed {either.sum()} exact duplicates ({either.mean()*100:.1f}%)")
    print(f"  Clean holdout: {len(clean)} genes")
    return clean


def annotate_familiarity(
    holdout: pd.DataFrame,
    train_embeddings: dict | None = None,
    v21_embed_dir: Path | None = None,
) -> pd.DataFrame:
    """Add familiarity flags: taxid_in_v2, and optionally cosine similarity to nearest training gene."""
    v2_taxids = set(pd.read_parquet(V2_TABLE, columns=["taxid"])["taxid"].values)
    holdout["taxid_in_v2"] = holdout["taxid"].isin(v2_taxids)
    holdout["species_status"] = holdout["taxid_in_v2"].map({True: "existing_species", False: "new_species"})

    if train_embeddings is not None:
        print("\n  Computing cosine similarity to nearest training gene (ESM-C)...")
        from sklearn.metrics.pairwise import cosine_similarity

        if v21_embed_dir is None:
            v21_embed_dir = resolve_v21_embed_dir()
        holdout_esmc = v21_embed_dir / "esmc_protein_embeddings.parquet"
        if holdout_esmc.exists():
            h_df = pd.read_parquet(holdout_esmc)
            h_df = h_df[h_df["gene_id"].isin(set(holdout["gene_id"]))]
            emb_col = [c for c in h_df.columns if c.endswith("_embedding")][0]
            h_vecs = np.stack(h_df[emb_col].map(lambda x: np.asarray(x, dtype=np.float32)).values)
            h_gids = h_df["gene_id"].values

            t_vecs = train_embeddings["vectors"]
            chunk = 5000
            nearest_cos = np.zeros(len(h_gids))
            for i in range(0, len(h_gids), chunk):
                end = min(i + chunk, len(h_gids))
                sims = cosine_similarity(h_vecs[i:end], t_vecs)
                nearest_cos[i:end] = sims.max(axis=1)
                if i % 20000 == 0:
                    print(f"    {i}/{len(h_gids)}")

            cos_map = dict(zip(h_gids, nearest_cos))
            holdout["nearest_train_cosine"] = holdout["gene_id"].map(cos_map)

            bins = pd.qcut(holdout["nearest_train_cosine"].dropna(), q=4,
                           labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
            holdout.loc[bins.index, "familiarity_quartile"] = bins

    return holdout


def stratified_analysis(holdout: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rho stratified by species status and familiarity."""
    print(f"\n{'='*80}")
    print("STRATIFIED CORRELATION ANALYSIS")
    print(f"{'='*80}")

    valid = holdout.dropna(subset=["y_pred", "expression_level"])
    rho_all, p_all = spearmanr(valid["y_pred"], valid["expression_level"])
    r_all, _ = pearsonr(valid["y_pred"], valid["expression_level"])
    print(f"\n  OVERALL: N={len(valid)}, Spearman rho={rho_all:.4f}, Pearson r={r_all:.4f}, p={p_all:.2e}")

    rows = [{"stratum": "OVERALL", "n": len(valid), "spearman_rho": round(rho_all, 4),
             "pearson_r": round(r_all, 4), "p_value": float(p_all)}]

    for col, label in [("species_status", "Species Status"),
                       ("familiarity_quartile", "Familiarity Quartile")]:
        if col not in holdout.columns:
            continue
        print(f"\n  --- By {label} ---")
        for val, grp in valid.groupby(col, observed=True):
            if len(grp) < 10:
                continue
            rho, p = spearmanr(grp["y_pred"], grp["expression_level"])
            r, _ = pearsonr(grp["y_pred"], grp["expression_level"])
            print(f"  {val:<25s} N={len(grp):>6}, rho={rho:.4f}, r={r:.4f}")
            rows.append({"stratum": f"{label}: {val}", "n": len(grp),
                         "spearman_rho": round(rho, 4), "pearson_r": round(r, 4),
                         "p_value": float(p)})

    for col, label in [("taxid", "Per-species (top 15)")]:
        species_rhos = []
        for val, grp in valid.groupby(col):
            if len(grp) < 20:
                continue
            rho, _ = spearmanr(grp["y_pred"], grp["expression_level"])
            species_rhos.append({"taxid": val, "n": len(grp), "rho": rho})

        if species_rhos:
            sp_df = pd.DataFrame(species_rhos).sort_values("rho", ascending=False)
            print(f"\n  --- {label} ---")
            for _, r in sp_df.head(15).iterrows():
                print(f"  taxid={r['taxid']:>10}  N={r['n']:>5}  rho={r['rho']:.4f}")
            print(f"  ...")
            print(f"  Median per-species rho: {sp_df['rho'].median():.4f} (N={len(sp_df)} species)")
            rows.append({"stratum": "median_per_species", "n": len(sp_df),
                         "spearman_rho": round(sp_df["rho"].median(), 4)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_PATH,
                        help="Model checkpoint path (default: F10 1B champion)")
    parser.add_argument("--skip-familiarity", action="store_true",
                        help="Skip cosine similarity computation (faster)")
    parser.add_argument("--zero-fill-modalities", nargs="+", default=[],
                        help="Modality names to zero-fill if missing (e.g. bacformer_large). "
                             "Each must be named explicitly — no wildcards. The model receives "
                             "zero vectors for these modalities, which degrades predictions.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V2.1 TEMPORAL HOLDOUT: Zero-Shot Inference")
    print("=" * 80)

    print("\n1. Loading V2.1 recovery genes...")
    v21 = pd.read_parquet(V21_TABLE)
    recovery = v21[v21.source_dataset == "v21_recovery"].copy()
    print(f"   Recovery genes: {len(recovery)}")

    print("\n2. Deduplicating against V2...")
    holdout = dedup_against_v2(recovery)

    print(f"\n3. Loading model from {args.checkpoint}...")
    model, input_dims = load_model(args.device, ckpt_path=args.checkpoint)

    print("\n4. Loading V2.1 embeddings...")
    v21_embed_dir = resolve_v21_embed_dir()
    print(f"   Using embeddings from: {v21_embed_dir}")
    all_gene_ids = set(recovery["gene_id"].values)
    embeddings = load_embeddings(v21_embed_dir, all_gene_ids, input_dims)
    print(f"   Loaded {len(embeddings)} modalities")
    if len(embeddings) == 0:
        raise RuntimeError(
            f"No V2.1 embedding parquets were found in {v21_embed_dir}. "
            "Track A inference cannot proceed with zero modalities."
        )

    print("\n5. Building tensors (all 124K genes for inference)...")
    if args.zero_fill_modalities:
        print(f"   Zero-fill allowed for: {args.zero_fill_modalities}")
    gene_ids = recovery["gene_id"].values.tolist()
    tensors, coverage = build_tensors(gene_ids, embeddings, input_dims,
                                       zero_fill_allowed=set(args.zero_fill_modalities))
    print(f"   Required modalities covered: {len(coverage)}/{len(input_dims)}")

    print("\n6. Running inference...")
    preds = predict(model, tensors, input_dims, args.device)
    recovery["y_pred"] = preds
    print(f"   Predictions: N={len(preds)}, mean={preds.mean():.4f}, std={preds.std():.4f}")
    if np.isclose(preds.std(), 0.0):
        raise RuntimeError(
            "Predictions are effectively constant (std ~ 0). This usually means the model "
            "was fed zero-filled or otherwise incompatible embeddings. Aborting before "
            "writing invalid temporal-holdout summary files."
        )

    holdout = holdout.merge(recovery[["gene_id", "y_pred"]], on="gene_id", how="left")

    print("\n7. Annotating familiarity...")
    train_embs = None
    if not args.skip_familiarity:
        train_esmc = V2_EMBED_DIR / "esmc_protein_embeddings.parquet"
        if train_esmc.exists():
            print("   Loading V2 training ESM-C embeddings for cosine similarity...")
            split = pd.read_csv(SPLIT_FILE, sep="\t")
            train_ids = set(split[split["split"] == "train"]["gene_id"].values)

            t_df = pd.read_parquet(train_esmc)
            t_df = t_df[t_df["gene_id"].isin(train_ids)]
            emb_col = [c for c in t_df.columns if c.endswith("_embedding")][0]
            t_vecs = np.stack(t_df[emb_col].map(lambda x: np.asarray(x, dtype=np.float32)).values)
            train_embs = {"gene_ids": t_df["gene_id"].values, "vectors": t_vecs}
            print(f"   Training ESM-C vectors: {t_vecs.shape}")

    holdout = annotate_familiarity(holdout, train_embs, v21_embed_dir=v21_embed_dir)

    print("\n8. Stratified analysis...")
    results = stratified_analysis(holdout)

    holdout.to_parquet(args.out_dir / "v21_temporal_holdout_predictions.parquet", index=False)
    results.to_csv(args.out_dir / "v21_temporal_holdout_results.csv", index=False)

    summary = {
        "total_recovery_genes": len(recovery),
        "exact_duplicates_removed": len(recovery) - len(holdout),
        "clean_holdout_genes": len(holdout),
        "existing_species_genes": int((holdout.get("species_status") == "existing_species").sum()),
        "new_species_genes": int((holdout.get("species_status") == "new_species").sum()),
        "results": results.to_dict(orient="records"),
    }
    with open(args.out_dir / "v21_temporal_holdout_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"  Predictions: {args.out_dir / 'v21_temporal_holdout_predictions.parquet'}")
    print(f"  Results: {args.out_dir / 'v21_temporal_holdout_results.csv'}")
    print(f"  Summary: {args.out_dir / 'v21_temporal_holdout_summary.json'}")


if __name__ == "__main__":
    main()
