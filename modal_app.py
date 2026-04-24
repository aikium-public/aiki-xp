"""Modal deployment of Aiki-XP for browser-based bacterial expression prediction.

Architecture:
  - Tier A (CPU): ESM-C + ProtT5-XL from FASTA. Scale-to-zero.
  - Tier D (GPU, planned): adds Evo-2 7B + Bacformer-large. Separate app.

Deploy:
    modal deploy modal_app.py

Test locally (uses Modal's remote compute but keeps output here):
    modal run modal_app.py::predict_tier_a --fasta tests/small.fasta

Once deployed, the FastAPI endpoint becomes:
    https://<your-workspace>--aikixp-tier-a-fastapi-app.modal.run/
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import fastapi
import modal

# ── Container image ──────────────────────────────────────────────────────────

tier_ab_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "pyyaml",
        "joblib",
        "tqdm",
        "pyarrow",
        "transformers>=4.36",
        "sentencepiece",
        "accelerate",
        "huggingface-hub",
        "esm",
        "fastapi[standard]",
        # biopython is needed to unpickle the /genomes/*.pkl SeqRecord
        # objects served by find_in_corpus / cds_for_protein / request_genome.
        "biopython>=1.83",
        # faiss-cpu backs the /find_neighbors similarity-search endpoint against
        # the 492K ESM-C PCA128 index on the lookups volume.
        "faiss-cpu>=1.8",
    )
    # Copy the aikixp package into the image at build time
    .add_local_python_source("aikixp")
    .add_local_dir("configs", "/app/configs")
    .add_local_dir("scripts", "/app/scripts")
)

# ── Persistent storage for checkpoints and model weights ─────────────────────

checkpoints_volume = modal.Volume.from_name(
    "aikixp-checkpoints", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    "aikixp-hf-cache", create_if_missing=True
)
genomes_volume = modal.Volume.from_name(
    "aikixp-genomes", create_if_missing=True
)

# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App("aikixp-tier-a")

# ── Static HTML landing page ─────────────────────────────────────────────────

# Landing page image = full tier_ab image + web/ dir. Having pandas + BioPython
# on hand lets the ASGI app host the species_scatter and find_in_corpus routes
# alongside the static assets, keeping us under the Starter-plan 8-endpoint cap.
web_image = tier_ab_image.add_local_dir("web", "/web")


# NCBI API key lives in the `aikium-ncbi-api-key` Modal secret (NCBI_API_KEY,
# NCBI_EMAIL). Raises the NCBI polite-use cap from 3 req/s to 10 req/s.
@app.function(
    image=web_image,
    volumes={"/lookups": modal.Volume.from_name("aikixp-lookups", create_if_missing=False),
             "/genomes": modal.Volume.from_name("aikixp-genomes", create_if_missing=False)},
    secrets=[modal.Secret.from_name("aikium-ncbi-api-key")],
    timeout=300,
    memory=4096,
    # Keep one container warm so first-click from a LinkedIn share doesn't eat an
    # 8-second cold-start (measured 2026-04-20). At CPU-Starter rates this is
    # under $1/day and is strictly worth it for launch week; can be dropped to
    # min_containers=0 once the traffic tails off.
    min_containers=1,
    scaledown_window=300,
)
@modal.asgi_app()
def landing_page():
    """Serve the demo landing page + static assets + CPU-only lookup/scatter routes.

    Routes:
      GET  /                     -> landing page HTML
      GET  /aikium_logo.png      -> logo asset
      GET  /hosts.json           -> host manifest (baseline + user-added)
      GET  /genome_status        -> check cache + Bacformer-precompute status
      POST /species_scatter      -> per-species truth/pred scatter data
      POST /sample_lookup        -> random sample from the 244K CV lookup
      POST /find_in_corpus       -> check if a user's protein is in our 492K set
      POST /request_genome       -> download + cache a new NCBI genome on demand
    """
    from fastapi import FastAPI, Header
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from typing import Optional
    from pathlib import Path
    import pickle, pathlib, json, os, time, threading
    import pandas as pd
    from scipy.stats import spearmanr

    fastapi_app = FastAPI()

    # ── Genome-request state (per-container) ─────────────────────────────────
    # Rate-limit is in-memory per container. Modal's min_containers=1 keeps a
    # single warm container; under burst traffic additional containers may get
    # their own counters (acceptable slack during launch week).
    _RATE_LIMIT_LOCK = threading.Lock()
    _RATE_LIMIT_PER_HOUR = 10
    _USER_ADDED_CAP = 100
    _request_counts: dict = {}  # (ip, hour_bucket) -> count

    def _client_ip(xff: Optional[str]) -> str:
        # Modal puts the client IP in X-Forwarded-For.
        if xff:
            return xff.split(",")[0].strip()
        return "unknown"

    def _check_rate_limit(ip: str) -> bool:
        now = int(time.time())
        hour_bucket = now // 3600
        with _RATE_LIMIT_LOCK:
            # Evict old buckets to keep the dict small.
            for k in list(_request_counts.keys()):
                if k[1] != hour_bucket:
                    _request_counts.pop(k, None)
            key = (ip, hour_bucket)
            count = _request_counts.get(key, 0)
            if count >= _RATE_LIMIT_PER_HOUR:
                return False
            _request_counts[key] = count + 1
            return True

    # ── Read-endpoint rate limiting (separate from /request_genome) ──────────
    # Heavy scripted use of these read endpoints is what we want to funnel into
    # partnership conversations rather than letting users anonymously scrape the
    # full PaxDB/Abele truth corpus for free. The 429 body points heavy users
    # at partnerships@aikium.com + the Zenodo archive + the local Docker image.
    _READ_RATE_LOCK = threading.Lock()
    _read_counts: dict = {}  # (ip, endpoint, hour_bucket) -> count
    _READ_CAPS = {
        "sample_lookup":   20,    # reproduce-paper-numbers button uses this
        "species_scatter": 60,    # fine for interactive host picks
        "find_in_corpus":  60,
        "find_neighbors":  60,
        "lookup_gene":     120,   # interactive single-gene lookups are common
        "umap_landscape":  10,    # ~4 MB payload, one load per session is enough
        "umap_project":    60,    # frequent per session
        "fold_structure":  20,    # ESMFold Atlas proxy — courteous cap
        "embed_protein":   30,    # ESM-C + linear PCA; cross-container call
    }
    _PARTNERSHIP_MAILTO = (
        "mailto:partnerships@aikium.com?"
        "subject=Aiki-XP%20high-volume%20inquiry&"
        "body=Hi%20Aikium%20team%2C%0A%0AI%27m%20using%20Aiki-XP%20at%20scale%20and%20"
        "ran%20into%20the%20interactive-use%20rate%20limit.%20Context%3A%0A%0A"
        "-%20Organization%3A%20%0A-%20Use%20case%3A%20%0A-%20Approximate%20"
        "request%20volume%3A%20%0A%0AThanks%21"
    )

    def _check_read_rate(ip: str, endpoint: str) -> tuple[bool, int, int]:
        """Return (allowed, current_count, cap). Evicts expired buckets."""
        cap = _READ_CAPS.get(endpoint, 60)
        now = int(time.time())
        hour_bucket = now // 3600
        with _READ_RATE_LOCK:
            for k in list(_read_counts.keys()):
                if k[2] != hour_bucket:
                    _read_counts.pop(k, None)
            key = (ip, endpoint, hour_bucket)
            count = _read_counts.get(key, 0)
            if count >= cap:
                return False, count, cap
            _read_counts[key] = count + 1
            return True, count + 1, cap

    def _partnership_429(endpoint: str, cap: int) -> JSONResponse:
        msg = (
            f"You've hit the interactive-use rate limit for {endpoint} "
            f"({cap} requests/hour per IP).\n\n"
            "Running Aiki-XP at scale? We'd love to talk.\n"
            "  → Partnership & high-volume inquiries: partnerships@aikium.com\n"
            "  → Full 492K corpus (proteomics truth + 5-tier CV predictions, "
            "properly citable): https://doi.org/10.5281/zenodo.19639621\n"
            "  → Run locally without rate limits: "
            "docker pull ghcr.io/aikium-public/aiki-xp:inference\n"
        )
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "3600"},
            content={
                "error": "rate_limit_exceeded",
                "endpoint": endpoint,
                "cap_per_hour": cap,
                "message": msg,
                "partnerships_contact": "partnerships@aikium.com",
                "bulk_data_doi": "10.5281/zenodo.19639621",
                "docker_image": "ghcr.io/aikium-public/aiki-xp:inference",
            },
        )

    def _user_added_manifest_path() -> Path:
        return Path("/genomes/user_added_genomes.jsonl")

    def _bulk_added_manifest_path() -> Path:
        # Generated by scripts/bulk_upload_genomes.py. Uploaded once after a
        # GCS sync. Appends here are not expected — the file is rewritten in
        # full when the cache is extended.
        return Path("/genomes/bulk_added_genomes.jsonl")

    def _load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _load_user_added() -> list[dict]:
        return _load_jsonl(_user_added_manifest_path())

    def _load_bulk_added() -> list[dict]:
        return _load_jsonl(_bulk_added_manifest_path())

    def _append_user_added(entry: dict) -> None:
        path = _user_added_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def _append_request_log(entry: dict) -> None:
        path = Path("/genomes/_request_log.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as e:
            print(f"request-log write failed: {e}")

    @fastapi_app.get("/", response_class=HTMLResponse)
    async def _index():
        return HTMLResponse(Path("/web/index.html").read_text())

    @fastapi_app.get("/aikium_logo.png")
    async def _logo():
        return FileResponse("/web/aikium_logo.png", media_type="image/png")

    @fastapi_app.get("/favicon.ico")
    async def _favicon():
        return FileResponse("/web/favicon.ico", media_type="image/x-icon")

    @fastapi_app.get("/favicon-16.png")
    async def _favicon_16():
        return FileResponse("/web/favicon-16.png", media_type="image/png")

    @fastapi_app.get("/favicon-32.png")
    async def _favicon_32():
        return FileResponse("/web/favicon-32.png", media_type="image/png")

    @fastapi_app.get("/apple-touch-icon.png")
    async def _apple_touch_icon():
        return FileResponse("/web/apple-touch-icon.png", media_type="image/png")

    @fastapi_app.get("/og_image.png")
    async def _og_image():
        return FileResponse("/web/og_image.png", media_type="image/png")

    @fastapi_app.get("/hosts.json")
    async def _hosts():
        """Merge three manifests into the host typeahead:

          1. Baseline — /web/hosts.json (image-baked, 1,831 entries with
             species_keys + n_test).
          2. Bulk — /genomes/bulk_added_genomes.jsonl (extended cache pushed
             by scripts/bulk_upload_genomes.py; no CV split metadata).
          3. User-added — /genomes/user_added_genomes.jsonl (per-request
             entries from /request_genome; also no CV metadata).

        Deduped by accession in load order so baseline always wins.
        """
        # Pull in pickles/manifests written by sibling containers since this
        # one was started. Without this, /request_genome additions and
        # bulk-uploaded manifests don't surface until the warm container
        # cycles (min_containers=1 keeps it alive for hours).
        try:
            modal.Volume.from_name("aikixp-genomes").reload()
        except Exception as e:
            print(f"volume reload warning: {e}")
        baseline = json.loads(Path("/web/hosts.json").read_text())
        bulk = _load_bulk_added()
        user = _load_user_added()
        if not (bulk or user):
            return FileResponse("/web/hosts.json", media_type="application/json")
        seen = {h["acc"] for h in baseline}
        merged = list(baseline)
        for h in bulk + user:
            acc = h.get("acc")
            if acc and acc not in seen:
                merged.append(h)
                seen.add(acc)
        return JSONResponse(merged, media_type="application/json")

    # ---- species_scatter ----
    @fastapi_app.post("/species_scatter")
    async def _scatter(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "species_scatter")
        if not ok:
            return _partnership_429("species_scatter", cap)

        keys = payload.get("species_keys") or (
            [payload["species_key"]] if payload.get("species_key") else []
        )
        if not keys:
            return {"error": "Provide species_keys (list) or species_key (single)"}
        # Batch cap: interactive host picks use 1-3 keys; cap at 5 to prevent
        # a single request from dumping the entire corpus via species enumeration.
        if len(keys) > 5:
            return {"error": "species_keys list capped at 5 per request. "
                             "For bulk export see https://doi.org/10.5281/zenodo.19639621"}

        df = pd.read_parquet("/lookups/tier_predictions_lookup.parquet")
        sub = df[df["species"].isin(keys)]
        if sub.empty:
            return {"error": f"No test-split genes found for species {keys}",
                    "n": 0, "points": []}

        nm = sub[~sub["is_mega"]]
        rho_ov = float(spearmanr(sub["true_expression"], sub["tier_d_prediction"]).statistic) if len(sub) >= 2 else None
        rho_nm = float(spearmanr(nm["true_expression"], nm["tier_d_prediction"]).statistic) if len(nm) >= 2 else None

        points = [
            {
                "gene_id": row["gene_id"],
                "true": float(row["true_expression"]),
                "tier_a": float(row["tier_a_prediction"]),
                "tier_c": float(row["tier_c_prediction"]),
                "tier_d": float(row["tier_d_prediction"]),
                "is_mega": bool(row["is_mega"]),
                "cv_fold": int(row["cv_fold"]),
            }
            for _, row in sub.iterrows()
        ]
        return {
            "species_keys": keys,
            "n": len(sub),
            "n_nonmega": int((~sub["is_mega"]).sum()),
            "rho_overall": rho_ov,
            "rho_nonmega": rho_nm,
            "points": points,
        }

    # ---- sample_lookup ----
    @fastapi_app.post("/sample_lookup")
    async def _sample(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Return a random sample of N rows from the 244K CV lookup.

        Used by the landing page's 'Reproduce paper numbers' button to compute
        per-fold mean Spearman rho on a fresh sample in the browser. N capped
        at 500 per call and 20 calls/hour per IP so this endpoint stays an
        *interactive* reproducibility surface, not a bulk-data mirror.

        Request:  {"n": 500, "seed": 42}  (both optional)
        Response: {"n": 500, "seed": 42, "rows": [
          {"gene_id": "...", "species": "...", "is_mega": false, "cv_fold": 1,
           "true_expression": -0.14, "tier_a": ..., "tier_b": ...,
           "tier_b_plus": ..., "tier_c": ..., "tier_d": ...}, ...
        ]}
        """
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "sample_lookup")
        if not ok:
            return _partnership_429("sample_lookup", cap)

        import pandas as pd
        # Per-call size cap: 500 rows/call × 20 calls/hour = 10K/hour/IP.
        # Still reproduces Tier D rho_nm within sampling noise on a single call
        # (per-fold SD ~0.05 at N=500 vs ~0.03 at N=2000; paper point estimate
        # is unambiguously within the band).
        n = min(int(payload.get("n", 500)), 500)
        seed = int(payload.get("seed", 42))
        df = pd.read_parquet("/lookups/tier_predictions_lookup.parquet")
        sample = df.sample(n=min(n, len(df)), random_state=seed)
        rows = [
            {
                "gene_id": r["gene_id"],
                "species": r["species"],
                "is_mega": bool(r["is_mega"]),
                "cv_fold": int(r["cv_fold"]),
                "true_expression": float(r["true_expression"]),
                "tier_a": float(r["tier_a_prediction"]),
                "tier_b": float(r["tier_b_prediction"]),
                "tier_b_plus": float(r["tier_b_plus_prediction"]),
                "tier_c": float(r["tier_c_prediction"]),
                "tier_d": float(r["tier_d_prediction"]),
            }
            for _, r in sample.iterrows()
        ]
        return {"n": len(rows), "seed": seed, "rows": rows}

    # ---- find_in_corpus ----
    @fastapi_app.post("/find_in_corpus")
    async def _find(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "find_in_corpus")
        if not ok:
            return _partnership_429("find_in_corpus", cap)

        protein = (payload.get("protein") or "").strip().replace(" ", "").replace("\n", "").upper()
        host = (payload.get("host") or "").strip()
        species_keys = payload.get("species_keys") or []
        if not protein or not host:
            return {"matched": False, "error": "Provide both 'protein' and 'host' fields"}

        host_path = pathlib.Path("/genomes") / f"{host}.pkl"
        if not host_path.exists():
            return {"matched": False, "reason": f"Host genome not found on volume: {host}"}

        import sys
        sys.path.insert(0, "/app")
        from aikixp.genome_lookup import load_genome
        genome = load_genome(host_path)

        matched_pid = None
        for feat in genome.features:
            if feat.type != "CDS" or "translation" not in feat.qualifiers or "pseudo" in feat.qualifiers:
                continue
            if feat.qualifiers["translation"][0] == protein:
                pids = feat.qualifiers.get("protein_id", [])
                if pids:
                    matched_pid = pids[0]
                    break

        if not matched_pid:
            return {"matched": False, "reason": "no exact protein match in host genome"}

        df = pd.read_parquet("/lookups/tier_predictions_lookup.parquet")
        candidates = [f"{sk}|{matched_pid}" for sk in species_keys] if species_keys else []
        hits = df[df["gene_id"].isin(candidates)] if candidates else pd.DataFrame()
        if hits.empty:
            hits = df[df["gene_id"].str.endswith(f"|{matched_pid}")]

        if hits.empty:
            return {
                "matched": False,
                "protein_id": matched_pid,
                "reason": f"Protein matches {matched_pid} in host {host}, but that gene is not in our 492K training / test set.",
            }

        row = hits.iloc[0]
        return {
            "matched": True,
            "gene_id": row["gene_id"],
            "protein_id": matched_pid,
            "species": row["species"],
            "truth": float(row["true_expression"]),
            "tier_a_cv": float(row["tier_a_prediction"]),
            "tier_b_cv": float(row["tier_b_prediction"]),
            "tier_b_plus_cv": float(row["tier_b_plus_prediction"]),
            "tier_c_cv": float(row["tier_c_prediction"]),
            "tier_d_cv": float(row["tier_d_prediction"]),
            "is_mega": bool(row["is_mega"]),
            "cv_fold": int(row["cv_fold"]),
        }

    # ---- find_neighbors ----
    _similarity_cache: dict = {}

    @fastapi_app.post("/find_neighbors")
    async def _find_neighbors(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Top-K most-similar proteins in the 492K corpus.

        Two modes:
          - {"gene_id": "<species>|<protein_id>", "k": 5}
            Look up a gene already in the corpus and return its nearest
            neighbours (self-hit excluded).
          - {"embedding": [128 floats], "k": 5}
            Query with a user-supplied ESM-C PCA128 vector.

        Returns each hit with `similarity` (cosine), `true_expression`, and the
        held-out `tier_d_prediction` where available.
        """
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "find_neighbors")
        if not ok:
            return _partnership_429("find_neighbors", cap)

        k = int(payload.get("k") or 5)
        # Tightened from 50 → 20 so one call can't return a large slice of
        # the corpus. Interactive UI uses k=5.
        if k < 1 or k > 20:
            return {"error": "k must be between 1 and 20"}

        # Lazy-load the FAISS index once per container
        sim = _similarity_cache.get("index")
        if sim is None:
            from aikixp.similarity_search import SimilarityIndex
            sim = SimilarityIndex.load("/lookups/similarity")
            _similarity_cache["index"] = sim

        if payload.get("gene_id"):
            try:
                hits = sim.search_by_gene_id(payload["gene_id"], k=k)
            except KeyError:
                return {"error": f"Gene ID not in the 492K corpus: {payload['gene_id']}"}
            return {"query": {"gene_id": payload["gene_id"]}, "hits": hits}

        if payload.get("embedding"):
            import numpy as _np
            vec = _np.asarray(payload["embedding"], dtype="float32")
            if vec.shape != (128,):
                return {"error": f"embedding must be length 128, got shape {vec.shape}"}
            hits = sim.search(vec, k=k)
            return {"query": {"embedding_dim": 128}, "hits": hits}

        return {"error": "provide either 'gene_id' or 'embedding' in the payload"}

    # ---- umap_landscape / umap_project ----
    _umap_cache: dict = {}

    _UMAP_MODALITIES = {
        "fused":           ("fused_umap2_x", "fused_umap2_y"),
        "esmc_protein":    ("esmc_protein_umap2_x", "esmc_protein_umap2_y"),
        "evo2_operon":     ("evo2_7b_full_operon_pca4096_umap2_x",
                            "evo2_7b_full_operon_pca4096_umap2_y"),
        "bacformer":       ("bacformer_umap2_x", "bacformer_umap2_y"),
        "hyenadna_cds":    ("hyenadna_dna_cds_umap2_x", "hyenadna_dna_cds_umap2_y"),
        "biophysical":     ("biophysical_umap2_x", "biophysical_umap2_y"),
        "rinalmo_init":    ("rinalmo_init_umap2_x", "rinalmo_init_umap2_y"),
    }

    def _umap_atlas_40k() -> "pd.DataFrame":
        df = _umap_cache.get("atlas_40k")
        if df is None:
            df = pd.read_parquet("/lookups/umap/atlas_40k.parquet")
            _umap_cache["atlas_40k"] = df
        return df

    def _umap_atlas_full() -> "pd.DataFrame":
        df = _umap_cache.get("atlas_full")
        if df is None:
            # Keyed by gene_id for fast .loc lookup
            df = pd.read_parquet("/lookups/umap/atlas_full.parquet").set_index("gene_id")
            _umap_cache["atlas_full"] = df
        return df

    @fastapi_app.get("/umap_landscape")
    async def _umap_landscape(
        modality: str = "fused",
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Pre-computed UMAP landscape of 40K stratified corpus genes.

        Use the `modality` query param: fused / esmc_protein / evo2_operon /
        bacformer / hyenadna_cds / biophysical / rinalmo_init.

        Returns `points` = [{x, y, gene_id, species, is_test, tier_d, truth}, ...].
        Colour by `tier_d` (test set) or `truth` (test set); grey out training-only rows.
        """
        # Rate limit (heavier read payload — 40K rows)
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "umap_landscape")
        if not ok:
            return _partnership_429("umap_landscape", cap)

        if modality not in _UMAP_MODALITIES:
            return {"error": f"modality must be one of {list(_UMAP_MODALITIES.keys())}"}
        xcol, ycol = _UMAP_MODALITIES[modality]

        df = _umap_atlas_40k()
        if xcol not in df.columns or ycol not in df.columns:
            return {"error": f"modality {modality} not in the atlas"}

        points = [
            {
                "gene_id": r["gene_id"],
                "species": r["species"] if r["is_test"] else None,
                "x": float(r[xcol]),
                "y": float(r[ycol]),
                "is_test": bool(r["is_test"]),
                "is_mega": bool(r["is_mega"]) if r["is_test"] else None,
                "truth": float(r["true_expression"]) if r["is_test"] else None,
                "tier_d": float(r["tier_d_prediction"]) if r["is_test"] else None,
            }
            for _, r in df.iterrows()
        ]
        return {
            "modality": modality,
            "n": len(points),
            "source": "manifold_atlas_492k.parquet (40K stratified subsample)",
            "points": points,
        }

    @fastapi_app.post("/umap_project")
    async def _umap_project(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Project a user's protein into the UMAP landscape via K nearest neighbours.

        Two query modes (same as /find_neighbors):
          - {"gene_id": "<species>|<protein_id>"}  — lookup by gene
          - {"embedding": [128 floats]}             — user-supplied ESM-C PCA128

        Returns:
          {
            "neighbors":  [{gene_id, similarity, <modality>_x, <modality>_y, ...}, ...],
            "centroid":   {"fused": [x, y], "esmc_protein": [x, y], ...},
            "note":       "inferred position = mean of neighbour UMAP coords."
          }

        The centroid is the similarity-weighted mean of the K-NN UMAP coordinates —
        an APPROXIMATE projection, not an exact UMAP.transform(). The original
        UMAP reducer was not persisted; see docs/protex/PLAN_UMAP_LANDSCAPE.md.
        """
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "umap_project")
        if not ok:
            return _partnership_429("umap_project", cap)

        k = int(payload.get("k") or 10)
        if k < 3 or k > 20:
            return {"error": "k must be between 3 and 20"}

        sim = _similarity_cache.get("index")
        if sim is None:
            from aikixp.similarity_search import SimilarityIndex
            sim = SimilarityIndex.load("/lookups/similarity")
            _similarity_cache["index"] = sim

        # 1. Resolve neighbour set
        if payload.get("gene_id"):
            try:
                hits = sim.search_by_gene_id(payload["gene_id"], k=k)
            except KeyError:
                return {"error": f"Gene ID not in corpus: {payload['gene_id']}"}
            query_desc = {"gene_id": payload["gene_id"]}
        elif payload.get("embedding"):
            import numpy as _np
            vec = _np.asarray(payload["embedding"], dtype="float32")
            if vec.shape != (128,):
                return {"error": f"embedding must be length 128, got {vec.shape}"}
            hits = sim.search(vec, k=k)
            query_desc = {"embedding_dim": 128}
        else:
            return {"error": "provide either 'gene_id' or 'embedding'"}

        # 2. Lookup UMAP coords for each neighbour
        atlas = _umap_atlas_full()
        import numpy as _np
        nb_coords = []
        for h in hits:
            gid = h["gene_id"]
            if gid not in atlas.index:
                continue
            row = atlas.loc[gid]
            hit = dict(h)
            for mod_name, (xcol, ycol) in _UMAP_MODALITIES.items():
                if xcol in atlas.columns and not _np.isnan(row[xcol]):
                    hit[f"{mod_name}_x"] = float(row[xcol])
                    hit[f"{mod_name}_y"] = float(row[ycol])
                else:
                    hit[f"{mod_name}_x"] = None
                    hit[f"{mod_name}_y"] = None
            nb_coords.append(hit)

        # 3. Similarity-weighted centroid per modality
        centroid = {}
        if nb_coords:
            sims = _np.array([h["similarity"] for h in nb_coords], dtype="float32")
            # Normalise weights to sum to 1
            w = sims / (sims.sum() if sims.sum() > 0 else 1.0)
            for mod_name in _UMAP_MODALITIES:
                xs, ys, ws = [], [], []
                for h, wi in zip(nb_coords, w):
                    if h.get(f"{mod_name}_x") is not None and h.get(f"{mod_name}_y") is not None:
                        xs.append(h[f"{mod_name}_x"])
                        ys.append(h[f"{mod_name}_y"])
                        ws.append(wi)
                if xs:
                    wsum = sum(ws)
                    if wsum > 0:
                        cx = sum(x * wi for x, wi in zip(xs, ws)) / wsum
                        cy = sum(y * wi for y, wi in zip(ys, ws)) / wsum
                        centroid[mod_name] = [float(cx), float(cy)]

        # Defensive: strip NaN/Inf and convert numpy scalars before JSON
        # serialization. FastAPI's json.dumps rejects non-finite floats and
        # doesn't know about np.float32; we coerce everything here.
        import math as _math
        import numpy as _np_mod

        def _sanitize(o):
            # numpy scalar → python scalar
            if isinstance(o, (_np_mod.generic,)):
                o = o.item()
            if isinstance(o, float):
                return o if _math.isfinite(o) else None
            if isinstance(o, int):
                return o
            if isinstance(o, dict):
                return {k: _sanitize(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_sanitize(v) for v in o]
            return o

        return _sanitize({
            "query": query_desc,
            "k_requested": k,
            "k_returned": len(nb_coords),
            "neighbors": nb_coords,
            "centroid": centroid,
            "note": ("Inferred position = similarity-weighted mean of K-NN "
                     "UMAP coords; approximate projection. The UMAP reducer "
                     "was not persisted (see PLAN_UMAP_LANDSCAPE.md)."),
        })

    # ---- embed_protein (ESM-C PCA128 via the Tier A container) ----
    @fastapi_app.post("/embed_protein")
    async def _embed_protein(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Extract the 128-d ESM-C PCA embedding for a single protein.

        Input:   {"protein": "MKT..."}
        Output:  {"esmc_pca128": [128 floats], "length": int, "dim_full": 1152, "dim_pca": 128}

        Calls the Tier A container (`AikixpTierA.embed_esmc_pca128`) which
        already has ESM-C warm. The resulting 128-d vector lives in the same
        basis as the FAISS similarity index, so you can pass it straight to
        /find_neighbors or /umap_project.
        """
        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "embed_protein")
        if not ok:
            return _partnership_429("embed_protein", cap)

        protein = (payload.get("protein") or "").strip()
        if not protein:
            return {"error": "Provide 'protein' (amino-acid sequence)."}
        try:
            TierACls = modal.Cls.from_name("aikixp-tier-a", "AikixpTierA")
            result = TierACls().embed_esmc_pca128.remote(protein)
        except Exception as e:
            return {"error": f"Embedding call failed: {type(e).__name__}: {e}"}
        return result

    # ---- fold_structure (ESMFold Atlas proxy) ----
    @fastapi_app.post("/fold_structure")
    async def _fold_structure(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Proxy to the ESMFold Atlas API (https://api.esmatlas.com/foldSequence/v1/pdb/).

        Input:   {"protein": "MKT..."} (≤400 residues)
        Output:  {"source_used": "esm_atlas",
                  "pdb": "ATOM...\\n...",
                  "length": int,
                  "pLDDT_mean": float (parsed from PDB B-factors; 0-100),
                  "latency_s": float,
                  "note": str}

        Length cap: 400 residues (ESMFold Atlas limit). For longer proteins
        we return a 400 with a redirect to the Docker image or the ESM-2
        HuggingFace Spaces deployment.

        No auth; rate-limited per partnership pattern. Uses urllib (no extra
        deps) so the request doesn't pull a heavy HTTP client into the image.
        """
        import time
        import urllib.request
        import urllib.error

        ok, _, cap = _check_read_rate(_client_ip(x_forwarded_for), "fold_structure")
        if not ok:
            return _partnership_429("fold_structure", cap)

        seq = (payload.get("protein") or "").strip().replace(" ", "").replace("\n", "").upper()
        if not seq:
            return {"error": "Provide 'protein' (amino-acid sequence)."}
        import re as _re
        if not _re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq):
            return {"error": "Protein must contain only standard amino-acid letters."}
        if len(seq) > 400:
            return {
                "error": f"Sequence length {len(seq)} exceeds ESMFold Atlas's 400-residue cap. "
                         "For longer proteins use the Docker image locally "
                         "(ghcr.io/aikium-public/aiki-xp:inference) with a local ESMFold install, "
                         "or email partnerships@aikium.com to discuss an extended-length API.",
                "length": len(seq),
            }

        t0 = time.time()
        try:
            req = urllib.request.Request(
                url="https://api.esmatlas.com/foldSequence/v1/pdb/",
                data=seq.encode("ascii"),
                method="POST",
                headers={
                    "Content-Type": "text/plain",
                    "User-Agent": "aikixp-landing/1.0 (demo.aikium.com)",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                pdb_text = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return {
                "error": f"ESMFold Atlas HTTP {e.code}: {e.reason}",
                "length": len(seq),
                "latency_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            return {
                "error": f"ESMFold Atlas request failed: {type(e).__name__}: {e}",
                "length": len(seq),
                "latency_s": round(time.time() - t0, 2),
            }

        # Parse mean pLDDT from PDB B-factor column. ESMFold Atlas stores
        # pLDDT on a 0-1 scale in the B-factor column (NOT 0-100), so we
        # multiply by 100 for the reader-facing value (matches the AlphaFold
        # / pLDDT convention most biologists expect).
        plddts = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and len(line) >= 66:
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                try:
                    plddts.append(float(line[60:66].strip()))
                except ValueError:
                    pass
        plddt_mean = (100.0 * sum(plddts) / len(plddts)) if plddts else None

        return {
            "source_used": "esm_atlas",
            "pdb": pdb_text,
            "length": len(seq),
            "n_ca_atoms": len(plddts),
            "pLDDT_mean": round(plddt_mean, 1) if plddt_mean is not None else None,
            "latency_s": round(time.time() - t0, 2),
            "note": ("Structure predicted by ESMFold (Meta AI). Confidence encoded "
                     "in the B-factor column (pLDDT, 0-100). Served via the public "
                     "ESMFold Atlas API; no GPU required on Aikium's side."),
        }

    # ---- request_genome ----
    @fastapi_app.post("/request_genome")
    async def _request_genome(
        payload: dict,
        x_forwarded_for: Optional[str] = Header(default=None),
    ):
        """Download a new NCBI chromosome accession on demand.

        Request body:
            {"accession": "NC_000913.3", "compute_bacformer": false}

        Response:
            {"status": "ok" | "already_cached" | "queued_for_review",
             "acc": "NC_000913.3",
             "name": "Escherichia coli K12",
             "n_cds": 4243, "n_pseudo": 90, "genome_bp": 4641652,
             "warnings": [...],
             "bacformer_status": "queued" | "done" | "skipped" | "unavailable"}

        On validation failure, returns HTTP 400 with {"error": "..."}.
        """
        import sys
        sys.path.insert(0, "/app")

        ip = _client_ip(x_forwarded_for)
        if not _check_rate_limit(ip):
            return JSONResponse(
                {"error": (
                    "Rate limit reached (10 requests/hour per IP). "
                    "Try again in an hour, or email venkatesh@aikium.com for bulk requests."
                )},
                status_code=429,
            )

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        raw_acc = (payload.get("accession") or "").strip()
        compute_bacformer = bool(payload.get("compute_bacformer", False))

        from aikixp.genome_cache import (
            GenomeDownloadError,
            download_and_cache_genome,
            validate_accession_format,
        )

        t0 = time.time()
        log_entry: dict = {
            "ts": t0,
            "ip": ip,
            "raw_accession": raw_acc,
            "compute_bacformer": compute_bacformer,
        }

        try:
            accession = validate_accession_format(raw_acc)
        except GenomeDownloadError as e:
            log_entry.update(status="rejected", reason=str(e))
            _append_request_log(log_entry)
            return JSONResponse({"error": str(e)}, status_code=400)

        log_entry["accession"] = accession

        pkl_path = Path("/genomes") / f"{accession}.pkl"
        if pkl_path.exists():
            log_entry["status"] = "already_cached"
            _append_request_log(log_entry)
            try:
                from aikixp.genome_lookup import load_genome
                genome = load_genome(pkl_path)
                n_valid = sum(
                    1 for f in genome.features
                    if f.type == "CDS"
                    and "translation" in f.qualifiers
                    and "pseudo" not in f.qualifiers
                )
                genome_bp = len(genome.seq) if genome.seq is not None else 0
                organism = (
                    getattr(genome, "annotations", {}).get("organism")
                    or genome.description or accession
                )
            except Exception:
                n_valid, genome_bp, organism = 0, 0, accession
            bacformer_cached = Path(f"/genomes/_bacformer_cache/{accession}.npy").exists()
            bf_status: str
            if bacformer_cached:
                bf_status = "done"
            elif compute_bacformer:
                try:
                    EmbeddingsCls = modal.Cls.from_name("aikixp-tier-d", "AikixpEmbeddings")
                    EmbeddingsCls().precompute_bacformer.spawn(accession)
                    bf_status = "queued"
                except Exception as e:
                    print(f"bacformer precompute spawn failed: {type(e).__name__}: {e}")
                    bf_status = "unavailable"
            else:
                bf_status = "skipped"
            return {
                "status": "already_cached",
                "acc": accession,
                "name": str(organism).strip().rstrip("."),
                "n_cds": n_valid,
                "genome_bp": genome_bp,
                "bacformer_status": bf_status,
            }

        # Enforce the user-added cap during launch week.
        added = _load_user_added()
        if len(added) >= _USER_ADDED_CAP:
            log_entry.update(status="queued_for_review",
                             reason=f"cap reached ({_USER_ADDED_CAP})")
            _append_request_log(log_entry)
            return JSONResponse({
                "status": "queued_for_review",
                "acc": accession,
                "message": (
                    f"User-added genome cap reached ({_USER_ADDED_CAP}). "
                    "Your request has been logged and will be reviewed. "
                    "Email venkatesh@aikium.com to prioritize."
                ),
            })

        api_key = os.environ.get("NCBI_API_KEY") or None
        try:
            stats = download_and_cache_genome(
                accession, target_dir=Path("/genomes"), api_key=api_key,
                overwrite=False,
            )
        except GenomeDownloadError as e:
            log_entry.update(status="failed", reason=str(e))
            _append_request_log(log_entry)
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            log_entry.update(status="error", reason=f"{type(e).__name__}: {e}")
            _append_request_log(log_entry)
            return JSONResponse(
                {"error": f"Unexpected error: {type(e).__name__}: {e}"},
                status_code=500,
            )

        # Commit the volume so other containers (Tier D precompute) see the pickle.
        try:
            genomes_vol = modal.Volume.from_name("aikixp-genomes")
            genomes_vol.commit()
        except Exception as e:
            print(f"volume commit warning: {e}")

        # Append to user-added manifest so /hosts.json surfaces the new host.
        manifest_entry = {
            "acc": accession,
            "name": stats.name,
            "n": stats.n_cds,
            "species_keys": [],   # no CV split for user-added genomes
            "user_added": True,
            "added_at": int(t0),
            "sha256": stats.sha256,
            "genome_bp": stats.genome_bp,
            "n_pseudo": stats.n_pseudo,
        }
        _append_user_added(manifest_entry)
        try:
            modal.Volume.from_name("aikixp-genomes").commit()
        except Exception as e:
            print(f"manifest commit warning: {e}")

        # Optional: spawn the Tier D Bacformer precompute in the background.
        bacformer_status = "skipped"
        if compute_bacformer:
            try:
                EmbeddingsCls = modal.Cls.from_name("aikixp-tier-d", "AikixpEmbeddings")
                EmbeddingsCls().precompute_bacformer.spawn(accession)
                bacformer_status = "queued"
            except Exception as e:
                print(f"bacformer precompute spawn failed: {type(e).__name__}: {e}")
                bacformer_status = "unavailable"

        log_entry.update(
            status="ok",
            n_cds=stats.n_cds,
            n_pseudo=stats.n_pseudo,
            genome_bp=stats.genome_bp,
            bacformer_status=bacformer_status,
            elapsed_s=round(time.time() - t0, 2),
        )
        _append_request_log(log_entry)

        return {
            "status": "ok",
            "acc": accession,
            "name": stats.name,
            "n_cds": stats.n_cds,
            "n_pseudo": stats.n_pseudo,
            "genome_bp": stats.genome_bp,
            "pseudogene_fraction": round(stats.pseudogene_fraction, 3),
            "warnings": stats.warnings,
            "bacformer_status": bacformer_status,
            "elapsed_s": round(time.time() - t0, 2),
        }

    # ---- genome_status ----
    @fastapi_app.get("/genome_status")
    async def _genome_status(acc: str = ""):
        """Poll the cache + Bacformer-precompute status for a given accession.

        Used by the UI to render progress after POST /request_genome and to
        decide whether Tier D will have a warm or cold Bacformer start.
        """
        acc = (acc or "").strip()
        if not acc:
            return JSONResponse({"error": "Provide ?acc=<accession>"}, status_code=400)

        pkl_path = Path("/genomes") / f"{acc}.pkl"
        bf_path = Path(f"/genomes/_bacformer_cache/{acc}.npy")
        progress_path = Path(f"/genomes/_bacformer_progress/{acc}.json")

        cached = pkl_path.exists()
        bacformer_cached = bf_path.exists()

        progress: dict = {}
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text())
            except Exception:
                progress = {}

        # Derive a user-facing Bacformer status string.
        if bacformer_cached:
            bacformer_status = "done"
        elif progress.get("status") == "running":
            bacformer_status = "running"
        elif progress.get("status") == "failed":
            bacformer_status = "failed"
        elif cached:
            bacformer_status = "not_started"
        else:
            bacformer_status = "no_genome"

        return {
            "acc": acc,
            "cached": cached,
            "bacformer_cached": bacformer_cached,
            "bacformer_status": bacformer_status,
            "progress": progress,
        }

    return fastapi_app


@app.cls(
    image=tier_ab_image,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/models": hf_cache_volume,
        "/lookups":    modal.Volume.from_name("aikixp-lookups", create_if_missing=False),
    },
    cpu=4,
    memory=16 * 1024,  # 16 GB
    scaledown_window=60,  # idle for 60s -> spin down
    timeout=600,
)
class AikixpTierA:
    """Tier A inference: ESM-C + ProtT5-XL from FASTA."""

    @modal.enter()
    def startup(self):
        """Load the XP5 ensemble once per container."""
        import os, sys
        os.environ["AIKIXP_CKPT_DIR"] = "/checkpoints"
        os.environ["AIKIXP_TIER_CONFIG"] = "/app/configs/deployment_tiers.yaml"
        # norm_stats lives on the checkpoints volume alongside the per-fold
        # .pt files — the image-bundled copy has been flaky across redeploys.
        os.environ["AIKIXP_NORM_STATS"] = "/checkpoints/norm_stats_492k.json"
        os.environ["HF_HOME"] = "/models"
        sys.path.insert(0, "/app")

        from aikixp.inference import XP5Ensemble
        self.model = XP5Ensemble("esmc_prott5_seed42", device="cpu")

        # Lazy-load flag for the ESM-C PCA transformer (for /embed_protein / UMAP overlay).
        # File: /lookups/umap/esmc_1152_to_128.npz (W: 1152×128, b: 128).
        # LSTSQ-derived from the existing PCA128 parquet; RMSE 0.0 vs original.
        self._esmc_pca_W = None
        self._esmc_pca_b = None

        print("AikixpTierA: model loaded, ready")

    @modal.method()
    def predict(self, fasta_text: str) -> dict:
        """Run Tier A end-to-end on a FASTA string."""
        import os, sys
        sys.path.insert(0, "/app")

        from aikixp.extract import extract_tier_a_embeddings, parse_fasta
        import numpy as np
        import pandas as pd

        tmp = Path("/tmp/aikixp_run")
        tmp.mkdir(exist_ok=True)
        fasta_path = tmp / "input.fasta"
        fasta_path.write_text(fasta_text)

        gene_ids, sequences = parse_fasta(fasta_path)
        if not gene_ids:
            return {"error": "No sequences parsed from FASTA", "predictions": []}

        emb_dir = tmp / "emb"
        paths = extract_tier_a_embeddings(sequences, gene_ids, emb_dir, device="cpu")

        arrays = {}
        for mod, path in paths.items():
            df = pd.read_parquet(path).set_index("gene_id").reindex(gene_ids)
            col = [c for c in df.columns if c != "gene_id"][0]
            arr = np.stack(df[col].values).astype(np.float32)
            arrays[mod] = arr

        predictions = self.model.predict(arrays)
        return {
            "n_sequences": len(gene_ids),
            "tier": "A",
            "recipe": "esmc_prott5_seed42",
            "predictions": [
                {"gene_id": gid, "predicted_expression": float(p)}
                for gid, p in zip(gene_ids, predictions)
            ],
        }

    @modal.method()
    def embed_esmc_pca128(self, protein: str) -> dict:
        """Return the 128-d ESM-C PCA embedding for a single protein.

        Used by the landing ASGI's /embed_protein wrapper so that novel
        proteins (not in the 492K corpus) can still be projected onto the
        UMAP landscape via K-NN in the FAISS index.

        The linear transformer (W, b) was LSTSQ-fit from the existing
        esmc_protein_embeddings.parquet (1152d) against
        esmc_protein_pca128_embeddings.parquet (128d) — RMSE 0.0 vs the
        original PCA, so the resulting 128-d vector matches the corpus
        basis exactly.
        """
        import sys, os, tempfile, pathlib
        sys.path.insert(0, "/app")
        import numpy as np
        import pandas as pd

        protein = (protein or "").strip().replace(" ", "").replace("\n", "").upper()
        if not protein:
            return {"error": "Provide 'protein' (amino-acid sequence)."}
        import re as _re
        if not _re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", protein):
            return {"error": "Protein must contain only standard AA letters (ACDEFGHIKLMNPQRSTVWY)."}
        if len(protein) > 4000:
            return {"error": f"Length {len(protein)} exceeds 4000 aa hard cap; trim or contact partnerships@aikium.com."}

        # Lazy-load the transformer once per container
        if self._esmc_pca_W is None:
            transform_path = pathlib.Path("/lookups/umap/esmc_1152_to_128.npz")
            if not transform_path.exists():
                return {"error": "ESM-C PCA transformer not present on the lookups volume."}
            data = np.load(transform_path)
            self._esmc_pca_W = data["W"].astype(np.float32)  # (1152, 128)
            self._esmc_pca_b = data["b"].astype(np.float32)  # (128,)
            print(f"embed_esmc_pca128: loaded transformer {self._esmc_pca_W.shape}")

        from aikixp.extract import extract_esmc
        with tempfile.TemporaryDirectory() as td:
            out_dir = pathlib.Path(td)
            # extract_esmc takes (sequences, gene_ids, out_dir, device) and writes a parquet
            parq = extract_esmc([protein], ["query"], out_dir, device="cpu")
            df = pd.read_parquet(parq).set_index("gene_id").reindex(["query"])
            col = [c for c in df.columns if c != "gene_id"][0]
            esmc_1152 = np.stack(df[col].values).astype(np.float32)[0]  # (1152,)

        pca128 = esmc_1152 @ self._esmc_pca_W + self._esmc_pca_b        # (128,)
        return {
            "esmc_pca128": pca128.tolist(),
            "length": len(protein),
            "dim_full": int(esmc_1152.shape[0]),
            "dim_pca": int(pca128.shape[0]),
        }


# ── Lookup endpoint: all 5 tiers for any of the 244K test genes ────────────

lookups_volume = modal.Volume.from_name("aikixp-lookups", create_if_missing=True)

# Per-container in-memory rate limiter + shared partnership 429 body for
# /lookup_gene. Scraping the 244K CV corpus via this endpoint is the obvious
# abuse vector (every row carries PaxDB/Abele truth + 5-tier CV). Cap the
# batch size to 20 IDs/call and 120 calls/hour/IP — interactive use is fine,
# systematic scraping runs into walls quickly.
import threading as _lg_threading
_LOOKUP_GENE_LOCK = _lg_threading.Lock()
_LOOKUP_GENE_COUNTS: dict = {}  # (ip, hour_bucket) -> count
_LOOKUP_GENE_CAP_PER_HOUR = 120
_LOOKUP_GENE_BATCH_CAP = 20


def _lookup_gene_check_rate(ip: str) -> tuple[bool, int]:
    import time as _t
    hour_bucket = int(_t.time()) // 3600
    with _LOOKUP_GENE_LOCK:
        for k in list(_LOOKUP_GENE_COUNTS.keys()):
            if k[1] != hour_bucket:
                _LOOKUP_GENE_COUNTS.pop(k, None)
        key = (ip, hour_bucket)
        count = _LOOKUP_GENE_COUNTS.get(key, 0)
        if count >= _LOOKUP_GENE_CAP_PER_HOUR:
            return False, count
        _LOOKUP_GENE_COUNTS[key] = count + 1
        return True, count + 1


@app.function(
    image=tier_ab_image,
    volumes={"/lookups": lookups_volume},
    timeout=60,
    memory=2048,
)
@modal.fastapi_endpoint(method="POST")
async def lookup_gene(request: "fastapi.Request") -> dict:
    """Return pre-computed Tier A/B/B+/C/D predictions for a gene_id in the 244K CV test set.

    Sized for interactive single-gene queries. Batch of up to 20 gene_ids/call;
    120 calls/hour per IP. For bulk access see https://doi.org/10.5281/zenodo.19639621.
    """
    import pandas as pd

    xff = request.headers.get("x-forwarded-for")
    ip = (xff.split(",")[0].strip() if xff else "unknown")
    ok, _ = _lookup_gene_check_rate(ip)
    if not ok:
        return {
            "error": "rate_limit_exceeded",
            "endpoint": "lookup_gene",
            "cap_per_hour": _LOOKUP_GENE_CAP_PER_HOUR,
            "message": (
                f"You've hit the interactive-use rate limit for lookup_gene "
                f"({_LOOKUP_GENE_CAP_PER_HOUR} requests/hour per IP).\n\n"
                "Running Aiki-XP at scale? We'd love to talk.\n"
                "  → Partnership inquiries: partnerships@aikium.com\n"
                "  → Full 492K corpus (properly citable): "
                "https://doi.org/10.5281/zenodo.19639621\n"
                "  → Local Docker image: ghcr.io/aikium-public/aiki-xp:inference\n"
            ),
            "partnerships_contact": "partnerships@aikium.com",
            "bulk_data_doi": "10.5281/zenodo.19639621",
            "results": [],
        }

    try:
        payload = await request.json()
    except Exception as e:
        return {"error": f"Could not parse JSON body: {e}", "results": []}

    ids = payload.get("gene_ids")
    if ids is None:
        single = payload.get("gene_id")
        ids = [single] if single else []
    if not ids:
        return {"error": "Provide gene_id or gene_ids", "results": []}
    if len(ids) > _LOOKUP_GENE_BATCH_CAP:
        return {
            "error": f"gene_ids list capped at {_LOOKUP_GENE_BATCH_CAP} per request. "
                     f"Got {len(ids)}. For bulk access see "
                     "https://doi.org/10.5281/zenodo.19639621, or email "
                     "partnerships@aikium.com for a high-volume API key.",
            "results": [],
        }

    df = pd.read_parquet("/lookups/tier_predictions_lookup.parquet")
    hits = df[df["gene_id"].isin(ids)]
    missing = [g for g in ids if g not in set(hits["gene_id"])]

    results = []
    for _, row in hits.iterrows():
        results.append({
            "gene_id": row["gene_id"],
            "species": row["species"],
            "is_mega": bool(row["is_mega"]),
            "cv_fold": int(row["cv_fold"]),
            "tier_a": float(row["tier_a_prediction"]),
            "tier_b": float(row["tier_b_prediction"]),
            "tier_b_plus": float(row["tier_b_plus_prediction"]),
            "tier_c": float(row["tier_c_prediction"]),
            "tier_d": float(row["tier_d_prediction"]),
            "true_expression": float(row["true_expression"]),
        })

    return {
        "n_found": len(results),
        "n_missing": len(missing),
        "missing_gene_ids": missing[:20],
        "note": (
            "Held-out 5-fold CV predictions (each gene scored by the fold "
            "where it was held out). `is_mega` = conservation annotation "
            "(True = member of a large cross-species cluster); `species` = "
            "gene's source organism; `cv_fold` = the fold where this gene "
            "was held out. To reproduce manuscript numbers, group by "
            "`cv_fold` and average Spearman across folds — "
            "e.g. Tier D rho_non_mega = 0.5904 +/- 0.0121."
        ),
        "results": results,
    }


# ── Utility endpoint: auto-fill CDS from protein + host ─────────────────────

@app.function(
    image=tier_ab_image,
    volumes={"/genomes": genomes_volume},
    timeout=60,
    memory=4096,
)
@modal.fastapi_endpoint(method="POST")
def cds_for_protein(payload: dict) -> dict:
    """Given {protein, host}, return the CDS.

    Native path: exact protein-sequence match against the host genome's CDS
    features → return the corresponding CDS DNA directly.
    Heterologous fallback: back-translate with the host's most-frequent
    codon per amino acid (computed from the host's own CDS catalog).

    Request:  {"protein": "MKR...", "host": "NC_000913.3"}
    Response: {"cds": "ATG...", "source": "native"|"codon_optimized",
               "matched_gene": "NP_417556.2"|null, "host": "NC_000913.3"}
    """
    import pickle
    import pathlib
    from collections import Counter

    protein = (payload.get("protein") or "").strip().replace(" ", "").replace("\n", "").upper()
    host = (payload.get("host") or "").strip()
    if not protein or not host:
        return {"error": "Provide both 'protein' and 'host' fields"}

    host_path = pathlib.Path("/genomes") / f"{host}.pkl"
    if not host_path.exists():
        return {"error": f"Host genome not found on volume: {host}"}

    import sys
    sys.path.insert(0, "/app")
    from aikixp.genome_lookup import load_genome
    genome = load_genome(host_path)
    genome_seq = str(genome.seq).upper()

    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    def rc(s):
        return "".join(comp.get(c, "N") for c in reversed(s))

    # Pass 1: exact protein match
    for feat in genome.features:
        if feat.type != "CDS" or "translation" not in feat.qualifiers or "pseudo" in feat.qualifiers:
            continue
        if feat.qualifiers["translation"][0] == protein:
            s, e = int(feat.location.start), int(feat.location.end)
            cds = genome_seq[s:e]
            if feat.location.strand == -1:
                cds = rc(cds)
            pid = feat.qualifiers.get("protein_id", [""])[0]
            gene = feat.qualifiers.get("gene", [""])[0]
            return {"cds": cds, "source": "native",
                    "matched_gene": pid or gene or None, "host": host}

    # Pass 2: back-translate using the host's own codon usage
    codon_counts_per_aa: dict = {}
    for feat in genome.features:
        if feat.type != "CDS" or "translation" not in feat.qualifiers or "pseudo" in feat.qualifiers:
            continue
        aa_seq = feat.qualifiers["translation"][0]
        s, e = int(feat.location.start), int(feat.location.end)
        cds_raw = genome_seq[s:e]
        if feat.location.strand == -1:
            cds_raw = rc(cds_raw)
        if len(cds_raw) < len(aa_seq) * 3:
            continue
        for i, aa in enumerate(aa_seq):
            codon = cds_raw[i*3 : i*3 + 3]
            if len(codon) != 3:
                continue
            codon_counts_per_aa.setdefault(aa, Counter())[codon] += 1

    aa_to_top_codon = {aa: cc.most_common(1)[0][0] for aa, cc in codon_counts_per_aa.items()}
    stop_counts_per_aa = codon_counts_per_aa.get("*", Counter(TAA=1))
    stop_codon = stop_counts_per_aa.most_common(1)[0][0] if stop_counts_per_aa else "TAA"
    try:
        cds = "".join(aa_to_top_codon[aa] for aa in protein) + stop_codon
    except KeyError as e:
        return {"error": f"No codon table entry for amino acid {e} (is this a non-standard protein?)"}
    return {"cds": cds, "source": "codon_optimized",
            "matched_gene": None, "host": host,
            "note": f"Back-translated using the most-frequent codon per amino acid observed in {host}'s own CDS catalog."}


# ── HTTP endpoint for end-to-end Tier A from FASTA ──────────────────────────

@app.function(image=tier_ab_image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def predict_fasta(payload: dict) -> dict:
    """POST a FASTA string, get ranked Tier A predictions back.

    Request:  {"fasta": ">gene_1\\nMKT...\\n>gene_2\\nMLE..."}
    Response: {"n_sequences": 2, "tier": "A", "predictions": [...]}
    """
    fasta = payload.get("fasta", "")
    if not fasta.strip().startswith(">"):
        return {"error": "Input must start with a FASTA header (>)", "predictions": []}

    tier_a = AikixpTierA()
    return tier_a.predict.remote(fasta)


# ── Hourly manifest snapshot ──────────────────────────────────────────────────
#
# Point-in-time recovery for the three volatile files on the genomes volume:
#   - user_added_genomes.jsonl (grows as users request genomes)
#   - bulk_added_genomes.jsonl (static post-bulk-upload, but cheap to snapshot)
#   - _request_log.jsonl       (audit trail)
#
# Snapshots land on the same volume at /genomes/backups/{YYYYMMDD-HH}/. This is
# NOT cross-region disaster recovery — if the whole volume is lost, restore
# from the git-tracked `scripts/bulk_upload_genomes.py` (re-run it against the
# local pickle cache to regenerate bulk_added_genomes.jsonl) and accept that
# per-user /request_genome additions since the last bulk push are lost.
# For real DR, add a GCS service-account secret and upload there instead.

@app.function(
    image=tier_ab_image,
    volumes={"/genomes": genomes_volume},
    schedule=modal.Cron("0 * * * *"),   # top of every hour
    timeout=120,
    memory=1024,
)
def snapshot_manifests():
    """Copy the three manifest files to /genomes/backups/{YYYYMMDD-HH}/.

    Idempotent: if a snapshot for this hour already exists, overwrite. Old
    snapshots are pruned after 7 days to keep the volume bounded.
    """
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    src = Path("/genomes")
    now = datetime.now(timezone.utc)
    dest = src / "backups" / now.strftime("%Y%m%d-%H")
    dest.mkdir(parents=True, exist_ok=True)

    targets = [
        "user_added_genomes.jsonl",
        "bulk_added_genomes.jsonl",
        "_request_log.jsonl",
    ]
    copied = []
    for name in targets:
        s = src / name
        if not s.exists():
            continue
        shutil.copy2(s, dest / name)
        copied.append(name)

    # Prune snapshots older than 7 days.
    cutoff = time.time() - 7 * 24 * 3600
    pruned = 0
    backups_dir = src / "backups"
    if backups_dir.exists():
        for sub in backups_dir.iterdir():
            if sub.is_dir() and sub.stat().st_mtime < cutoff:
                shutil.rmtree(sub, ignore_errors=True)
                pruned += 1

    genomes_volume.commit()
    print(f"snapshot_manifests: {len(copied)} files -> {dest}, pruned {pruned} old dirs")


# ── CLI for local testing (runs on Modal but prints locally) ──────────────────

@app.local_entrypoint()
def predict_tier_a(fasta: str):
    """Run Tier A on a local FASTA file; prints predictions to stdout."""
    text = Path(fasta).read_text()
    result = AikixpTierA().predict.remote(text)
    print(f"\n=== {result.get('n_sequences', 0)} predictions (Tier {result.get('tier')}) ===")
    for p in result.get("predictions", []):
        print(f"  {p['gene_id']:40s}  rho_hat={p['predicted_expression']:+.4f}")
    if "error" in result:
        print(f"ERROR: {result['error']}")
