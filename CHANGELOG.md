# Changelog

All notable changes to Aiki-XP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-04-21

### Fixed
- **Tier B+ live inference on Modal disabled.** The paper's Tier B+ recipe
  requires Evo-2 1B embeddings, whose loader depends on NVIDIA Transformer
  Engine; Transformer Engine doesn't build on Modal's current image, and
  A100-40GB has no FP8 hardware in any case. The live endpoint now returns
  a structured refusal (`error: tier_b_plus_live_inference_unavailable`)
  with pointers to the Docker image (`ghcr.io/aikium-public/aiki-xp:full`)
  and to `/lookup_gene` for the paper's cached CV predictions. The
  landing page disables the Tier B+ radio with an inline explanation.
- **`aikixp/inference.py` strict dim enforcement.** Per-modality feature
  dim must now match the adapter's expected dim exactly; a mismatch
  raises `ValueError` instead of slicing. Belt-and-suspenders check in
  `predict()` does the same.
- **Smoke test** (`scripts/prelaunch_smoke.sh`) updated to assert the
  refusal payload rather than a prediction shape.

### Unaffected
- The paper's Tier B+ headline number (ρ_nc = 0.543) is correct and
  reproduces exactly from the held-out CV predictions in
  `tier_predictions_lookup.parquet`, served by `/lookup_gene`. No paper
  numbers change.
- Tiers A, B, C, D on Modal continue to run normally.

## [1.0.0] — 2026-04-21

Public launch release. Paper submitted to Nature Biotechnology, preprint on bioRxiv,
data and model weights on Zenodo, code on GitHub, and interactive demo on Modal.

### Added
- **Paper + data + weights**
  - Paper: "Aiki-XP: leakage-controlled multimodal prediction of within-species relative
    protein expression at pan-bacterial scale" (Tien, Sharma Meda, Shastry, Mysore).
  - Zenodo deposit with 23 files, 28.2 GB, CC-BY 4.0 — includes production parquet,
    model checkpoints, cross-validation predictions, and embedding parquets for every tier.
  - bioRxiv preprint (public screening Tuesday 2026-04-21).
- **Code**
  - Python package `aikixp` (inference + training primitives, Apache 2.0).
  - Docker images on GHCR: `ghcr.io/aikium-public/aiki-xp:inference` (2 GB, CPU-capable
    inference only) and `:full` (10.7 GB, includes training deps).
- **Public Modal deployment**
  - Eight public endpoints covering the five-tier ladder (A, B, B+, C, D), a corpus-lookup
    endpoint, a CDS auto-fill endpoint, and an interactive landing page.
  - Landing page pinned warm (`min_containers=1`) so first-click latency is ~0.5 s.
  - Interactive features: host typeahead with phylogenetic-neighbor fallback,
    per-species calibration scatter, gene-in-corpus callout, live Tier D overlay,
    "Reproduce paper numbers" button, share-by-URL permalinks, "Compare all 5 tiers"
    parallel prediction, limitations panel, compute footprint table.
- **On-demand host genome support** (4,566 pre-cached, expandable to any bacterial chromosome)
  - `POST /request_genome` — users paste an NCBI chromosome accession (RefSeq `NC_*`/`NZ_*`,
    GenBank `CP*`/`CM*`), the backend fetches via Entrez in ~30 s, validates (≥100 valid CDS,
    ≤15 Mb, single SeqRecord), pickles to the Modal volume, and adds the host to the
    typeahead. Rate-limited to 10 req/hr per IP with a 100-genome launch-week cap.
  - `GET /genome_status?acc=` — polls cache and Bacformer-precompute progress.
  - Optional `compute_bacformer: true` spawns a background Tier D extraction so the
    first Tier D request on a new host doesn't cold-start 5–30 min.
  - Bulk import of the full Aiki-XP genome cache (`gs://aikium-data/yotta_display/.../genomes/`)
    via `scripts/bulk_upload_genomes.py`: 2,734 validated chromosomes added to the
    Modal volume on 2026-04-21, raising the typeahead from 1,831 → 4,566 hosts.
  - `/hosts.json` now merges three manifests (baseline / bulk-added / user-added) with
    `volume.reload()` on each request so new additions surface without container churn.
  - UI: inline "Request this genome" control in the host selector when typeahead has
    no match; user-added / bulk-added rows render with distinct badges.
- **Reproducibility story**
  - Colab quickstart notebook — reproduces paper Table 1 non-mega Spearman ρ values in
    ~3 minutes on free Colab CPU runtime.
  - `scripts/prelaunch_smoke.sh` — one-shot health check for all 8 public endpoints.
  - `POST /sample_lookup` — random sample from the 244K held-out CV predictions, so any
    user can recompute per-fold ρ locally.
- **Documentation**
  - `README.md` with zero-install / Docker / pip install paths, full data-pipeline
    description, 9-question troubleshooting FAQ (cold-start timing, tagged sequences,
    live-vs-CV differences, per-point uncertainty calibrated on a 20K sample).
  - `CONTRIBUTING.md` + structured GitHub issue templates (bug report,
    scientific question, host request).

### Scientific headline numbers
- **Spearman ρ** (per-fold mean over 5-fold CV, gene-operon leakage-controlled split):
  - Tier A (ESM-C + ProtT5 protein only): ρ_overall = 0.5825, ρ_non-mega = 0.5182.
  - Tier B (+ HyenaDNA + classical): ρ_overall = 0.5929, ρ_non-mega = 0.5312.
  - Tier B+ (+ Evo-2 7B init-window): ρ_overall = 0.6035, ρ_non-mega = 0.5428.
  - Tier C (+ Evo-2 7B full operon): ρ_overall = 0.6501, ρ_non-mega = 0.5747.
  - Tier D (+ Bacformer-large + operon structure, 5 modalities — XP5 champion): ρ_overall = 0.6675, ρ_non-mega = 0.5904.
- **Training corpus:** 492,026 genes · 385 species · 1,831 bacterial reference genomes.

---

### Link references
- [1.0.0]: https://github.com/aikium-public/aiki-xp/releases/tag/v1.0.0
- Paper DOI: [10.5281/zenodo.19639621](https://doi.org/10.5281/zenodo.19639621)
- Live demo: [aikium--aikixp-tier-a-landing-page.modal.run](https://aikium--aikixp-tier-a-landing-page.modal.run/)
