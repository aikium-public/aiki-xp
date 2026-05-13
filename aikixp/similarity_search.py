"""FAISS-backed similarity search over the 492K ESM-C PCA128 protein embeddings.

The index is built offline by ``scripts/build_similarity_index.py`` and lives
on the ``aikixp-lookups`` Modal volume at:

    /lookups/similarity/sim_index.faiss    (~252 MB, IndexFlatIP, L2-normalised)
    /lookups/similarity/sim_meta.parquet   (~8 MB, gene_id + species + truth + tier_d CV)

Two usage patterns:

1. **Neighbour-by-gene-id**: user pastes a gene_id already in the corpus, we look
   up its row in the index and return the top-K nearest neighbours.
2. **Neighbour-by-sequence**: user pastes a novel protein, we run ESM-C + the
   production PCA to get a 128-d query vector, then query the index.
   (Requires ESM-C in the container — not available on the landing ASGI today;
    wire through the Tier A container or a dedicated embedding endpoint.)

Example usage inside a Modal function with `/lookups` mounted:

    from aikixp.similarity_search import SimilarityIndex
    sim = SimilarityIndex.load("/lookups/similarity")
    hits = sim.search_by_gene_id("Escherichia_coli_K12|NP_417556.2", k=5)
    for h in hits:
        print(h["gene_id"], h["similarity"], h["true_expression"], h["tier_d_prediction"])
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SimilarityIndex:
    """Lazy-loaded FAISS index + metadata table for the 492K corpus."""

    index_path: Path
    meta_path: Path
    _index: Optional[object] = None
    _meta: Optional[object] = None
    _gene_id_to_row: Optional[dict] = None

    @classmethod
    def load(cls, base_dir: str | Path) -> "SimilarityIndex":
        base = Path(base_dir)
        return cls(index_path=base / "sim_index.faiss",
                   meta_path=base / "sim_meta.parquet")

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        import faiss
        import pandas as pd
        self._index = faiss.read_index(str(self.index_path))
        self._meta = pd.read_parquet(self.meta_path)
        # Build a gene_id -> row lookup once
        self._gene_id_to_row = dict(zip(self._meta["gene_id"], range(len(self._meta))))

    # -------- search by pre-embedded query --------

    def search(self, query_vec, k: int = 5, exclude_row: Optional[int] = None) -> List[dict]:
        """Query the index with an L2-normalised 128-d vector.

        If ``exclude_row`` is set, hits whose FAISS row index equals it are dropped
        (use when querying with a vector that itself lives in the index, to avoid
        self-hits swamping the top of the list).

        Returns a list of dicts, one per neighbour, with similarity + metadata.
        """
        import numpy as np
        self._ensure_loaded()
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        q = q / norm

        # Over-fetch so self-hit exclusion doesn't short-change us
        k_query = k + (1 if exclude_row is not None else 0)
        D, I = self._index.search(q, k_query)

        out = []
        meta = self._meta
        for sim, row in zip(D[0], I[0]):
            if row < 0:
                continue
            if exclude_row is not None and int(row) == exclude_row:
                continue
            m = meta.iloc[int(row)]
            out.append({
                "gene_id": m["gene_id"],
                "species": m.get("species") if "species" in meta.columns else None,
                "is_mega": bool(m["is_mega"]) if "is_mega" in meta.columns and not _isna(m["is_mega"]) else None,
                "cv_fold": int(m["cv_fold"]) if "cv_fold" in meta.columns and not _isna(m["cv_fold"]) else None,
                "true_expression": float(m["true_expression"]) if not _isna(m.get("true_expression")) else None,
                "tier_d_prediction": float(m["tier_d_prediction"]) if not _isna(m.get("tier_d_prediction")) else None,
                "similarity": float(sim),
            })
            if len(out) >= k:
                break
        return out

    # -------- search by gene id --------

    def search_by_gene_id(self, gene_id: str, k: int = 5) -> List[dict]:
        """Look up ``gene_id`` in the index and return the top-K most-similar corpus members.

        The query gene itself is excluded from the returned hits (so a cosine=1.0
        self-match never pads the top of the list).
        """
        self._ensure_loaded()
        row = self._gene_id_to_row.get(gene_id)
        if row is None:
            raise KeyError(f"Gene ID not in similarity corpus: {gene_id}")

        import numpy as np
        vec = np.array(self._index.reconstruct(int(row)), dtype="float32")[None, :]
        return self.search(vec, k=k, exclude_row=int(row))


def _isna(v) -> bool:
    """True if v is None or pandas NaN."""
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
