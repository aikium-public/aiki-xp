#!/usr/bin/env python3
"""Build a FAISS similarity index over the 492K ESM-C PCA128 protein embeddings.

Input:  esmc_protein_pca128_embeddings.parquet (gene_id, embedding)
Output: sim_index.faiss + sim_meta.parquet (gene_id, species, is_mega, true_expression, tier_d)

The index is L2-normalised inner-product (cosine similarity). Query with:

    import faiss, pandas as pd
    index = faiss.read_index("sim_index.faiss")
    meta  = pd.read_parquet("sim_meta.parquet")
    q = query_vec / np.linalg.norm(query_vec)
    D, I = index.search(q[None, :].astype("float32"), k=5)
    hits = meta.iloc[I[0]]

Usage:
    python scripts/build_similarity_index.py \
        --embeddings  embeddings_finalized/esmc_protein_pca128_embeddings.parquet \
        --lookup      tier_predictions_lookup.parquet \
        --out-dir     /tmp/sim_index/
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, type=pathlib.Path,
                    help="Path to esmc_protein_pca128_embeddings.parquet")
    ap.add_argument("--lookup", required=True, type=pathlib.Path,
                    help="Path to tier_predictions_lookup.parquet")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path,
                    help="Directory to write sim_index.faiss + sim_meta.parquet")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import faiss

    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] Loading embeddings from {args.embeddings} ...", flush=True)
    emb_df = pd.read_parquet(args.embeddings)
    print(f"[{time.time()-t0:5.1f}s]   rows: {len(emb_df):,}, cols: {list(emb_df.columns)}", flush=True)

    # Materialize (N, 128) float32 array
    print(f"[{time.time()-t0:5.1f}s] Materializing embedding matrix ...", flush=True)
    emb_col = [c for c in emb_df.columns if "embedding" in c][0]
    emb = np.asarray(emb_df[emb_col].tolist(), dtype="float32")
    d = emb.shape[1]
    n = emb.shape[0]
    print(f"[{time.time()-t0:5.1f}s]   shape: {emb.shape}, dtype: {emb.dtype}", flush=True)

    # L2-normalise for cosine via inner product
    print(f"[{time.time()-t0:5.1f}s] L2-normalising ...", flush=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_norm = emb / norms

    # Sanity: mean norm should be 1.0
    m = float(np.linalg.norm(emb_norm, axis=1).mean())
    print(f"[{time.time()-t0:5.1f}s]   mean ||v|| post-norm: {m:.6f}", flush=True)

    # Build FAISS IndexFlatIP — for 492K × 128 this is ~250 MB, trivial to query
    print(f"[{time.time()-t0:5.1f}s] Building FAISS IndexFlatIP ...", flush=True)
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)
    print(f"[{time.time()-t0:5.1f}s]   ntotal: {index.ntotal}", flush=True)

    # Write index
    index_path = args.out_dir / "sim_index.faiss"
    print(f"[{time.time()-t0:5.1f}s] Writing {index_path} ...", flush=True)
    faiss.write_index(index, str(index_path))
    print(f"[{time.time()-t0:5.1f}s]   wrote {index_path.stat().st_size / 1e6:.1f} MB", flush=True)

    # Build metadata side-table: gene_id + species + is_mega + true_expression + tier_d_prediction
    print(f"[{time.time()-t0:5.1f}s] Joining with {args.lookup} for metadata ...", flush=True)
    lookup = pd.read_parquet(args.lookup)
    meta = emb_df[["gene_id"]].copy()
    meta = meta.merge(
        lookup[["gene_id", "species", "is_mega", "cv_fold",
                "true_expression", "tier_d_prediction"]],
        on="gene_id", how="left",
    )
    print(f"[{time.time()-t0:5.1f}s]   joined: {len(meta):,} rows; "
          f"{meta['true_expression'].notna().sum():,} have CV ground truth "
          f"({100 * meta['true_expression'].notna().mean():.1f}%)", flush=True)
    # row order must match the FAISS index exactly
    assert len(meta) == index.ntotal, \
        f"meta ({len(meta)}) != index.ntotal ({index.ntotal})"

    meta_path = args.out_dir / "sim_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    print(f"[{time.time()-t0:5.1f}s] Wrote {meta_path} ({meta_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Smoke test: query index with a random row and verify self-hit is top-1
    print(f"[{time.time()-t0:5.1f}s] Smoke test ...", flush=True)
    rng = np.random.default_rng(42)
    i = int(rng.integers(0, n))
    q = emb_norm[i:i+1]
    D, I = index.search(q, 5)
    assert I[0][0] == i, f"self-hit should be top-1, got {I[0][0]} != {i}"
    print(f"[{time.time()-t0:5.1f}s]   OK: gene_id[{i}]={emb_df['gene_id'].iloc[i]} "
          f"top-5 sims: {D[0].tolist()}", flush=True)
    print(f"[{time.time()-t0:5.1f}s] DONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
