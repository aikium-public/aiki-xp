# Similarity search — integration notes

The FAISS index is built and uploaded to the Modal volume, and the
library code is in `aikixp/similarity_search.py`. This note captures
the minimum edit needed to expose it as a public endpoint.

## What's shipped

- `scripts/build_similarity_index.py` — offline builder. Takes the 492K
  ESM-C PCA128 parquet + the CV lookup, emits `sim_index.faiss` (~252 MB)
  + `sim_meta.parquet` (~8 MB).
- `aikixp/similarity_search.py` — `SimilarityIndex` class with `load()`,
  `search()`, `search_by_gene_id()`.
- Index artifacts uploaded to the `aikixp-lookups` Modal volume at
  `/lookups/similarity/sim_index.faiss` and `/lookups/similarity/sim_meta.parquet`.
  Verified locally: searching `Escherichia_coli_K12|NP_417556.2` returns 5
  close homologues at cosine ≥ 0.97 with their PaxDb truth + Tier D CV.

## What needs to be added

### 1. Add `faiss-cpu` to the landing image (modal_app.py)

In the `tier_ab_image = (modal.Image.debian_slim()...` chain:

```python
tier_ab_image = (
    tier_ab_image
    .pip_install(
        "faiss-cpu>=1.8",  # for similarity_search.SimilarityIndex.load()
    )
)
```

`faiss-cpu` is a 1-2 MB wheel — negligible image bloat.

### 2. Mount the lookups volume on the landing ASGI (already mounted)

Existing:

```python
volumes={
    "/lookups": modal.Volume.from_name("aikixp-lookups", ...),
    "/genomes": modal.Volume.from_name("aikixp-genomes", ...),
}
```

No change needed — the similarity index sits under `/lookups/similarity/`.

### 3. Add the `/find_neighbors` route to the landing ASGI

Inside `landing_page()` after the existing `/find_in_corpus` route, paste:

```python
# ── Load similarity index once per container ───────────────────────────
_SIM_INDEX = None

def _sim():
    nonlocal_sim = globals().get("_SIM_INDEX")
    # (Use a module-level mutable dict or a closure; the pattern here is
    # standard for Modal — load-on-first-call, then cache.)
    import pathlib as _p
    from aikixp.similarity_search import SimilarityIndex
    global _SIM_INDEX
    if _SIM_INDEX is None:
        _SIM_INDEX = SimilarityIndex.load("/lookups/similarity")
    return _SIM_INDEX

@fastapi_app.post("/find_neighbors", tags=["corpus"])
async def _find_neighbors(payload: dict) -> dict:
    """Return the top-K most-similar 492K-corpus proteins to a given query.

    Query modes:
      - {"gene_id": "<species>|<protein_id>"}  — exact corpus row
      - {"embedding": [128 floats]}             — pre-computed ESM-C PCA128 vector

    Response:
      {"query": "...", "hits": [{gene_id, species, similarity,
                                  true_expression, tier_d_prediction, ...}, ...]}
    """
    k = int(payload.get("k") or 5)
    if k < 1 or k > 50:
        return {"error": "k must be between 1 and 50"}

    sim = _sim()
    if payload.get("gene_id"):
        try:
            hits = sim.search_by_gene_id(payload["gene_id"], k=k)
        except KeyError:
            return {"error": f"Gene ID not in corpus: {payload['gene_id']}"}
        return {"query": {"gene_id": payload["gene_id"]}, "hits": hits}

    if payload.get("embedding"):
        import numpy as np
        vec = np.asarray(payload["embedding"], dtype="float32")
        if vec.shape != (128,):
            return {"error": f"embedding must be length 128, got {vec.shape}"}
        hits = sim.search(vec, k=k)
        return {"query": {"embedding_dim": 128}, "hits": hits}

    return {"error": "provide either 'gene_id' or 'embedding' in the payload"}
```

### 4. Smoke test

```bash
curl -s -X POST https://aikium--aikixp-tier-a-landing-page.modal.run/find_neighbors \
  -H 'Content-Type: application/json' \
  -d '{"gene_id": "Escherichia_coli_K12|NP_417556.2", "k": 5}' | python3 -m json.tool
```

Expected: 5 neighbours with `similarity` ≥ 0.95 on the hybF example, each with
`true_expression` and `tier_d_prediction` populated for genes in the 244K test split.

### 5. Frontend panel (in `web/index.html`)

Minimum UI: after the existing "find_in_corpus" callout rows in the results table,
add a collapsible "Similar proteins in our corpus" panel that hits `/find_neighbors`
with the matched gene_id (or the user's own query if they paste a gene_id in a
new text field). Render a small scatter of `true_expression` vs `tier_d_prediction`
for the K neighbours, coloured by similarity.

This is a ~2h UI task; keep it out of scope for the initial backend wire-up.

### 6. Sequence-query mode (follow-up)

The current integration supports "find neighbours for a gene already in the corpus."
To support "paste any protein, find neighbours," we need an embedding endpoint
(Tier A container has ESM-C loaded — add `POST /embed_protein` there) and then
the landing ASGI can forward the protein to it and use the returned 128-d vector
as the query. Defer to post-launch week 1.

## Cost to run

- FAISS IndexFlatIP on 492K × 128d: ~252 MB in memory, queries return in <5 ms
  on the existing 4-vCPU / 16 GB landing container.
- Cold load (FAISS + metadata parquet): ~300 ms once per container.
- No GPU required.

## File manifest

```
aikixp/similarity_search.py                   (library)
scripts/build_similarity_index.py             (offline builder)
docs/similarity_search_integration.md         (this file)

Modal volume aikixp-lookups:
  /lookups/similarity/sim_index.faiss         (252 MB)
  /lookups/similarity/sim_meta.parquet        (8 MB)
```
