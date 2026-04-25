# Aiki-XP

**Leakage-controlled multimodal prediction of within-species relative protein expression at pan-bacterial scale.**

- 📄 **Paper:** Tien, Sharma Meda, Shastry & Mysore. *Nature Biotechnology* (2026, under review). Preprint: [bioRxiv 10.64898/2026.04.21.719525](https://doi.org/10.64898/2026.04.21.719525).
- 📦 **Data + weights:** Zenodo DOI [10.5281/zenodo.19639621](https://doi.org/10.5281/zenodo.19639621) (23 files, 28.2 GB, CC-BY 4.0)
- 🐳 **Docker:** `ghcr.io/aikium-public/aiki-xp:inference` (2 GB) and `:full` (10.7 GB)
- 🌐 **Live demo:** https://aikium--aikixp-tier-a-landing-page.modal.run
- 📓 **Colab quickstart:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aikium-public/aiki-xp/blob/main/notebooks/aiki_xp_colab_quickstart.ipynb)
- 💻 **From Aikium Inc.** · Apache 2.0

Aiki-XP integrates **five biological modalities** (genome context, operon architecture, coding-sequence composition, protein identity, per-gene biophysical features) across **492,026 genes from 385 bacterial species** under a leakage-controlled gene-operon split that enforces zero MMseqs2-cluster overlap between train and test. ρ_nc = 0.592 ± 0.012 on non-conserved genes — versus 0.509 for ESM-C 600M alone.

## Deployment tiers

Each tier adds a class of biological input and yields a statistically significant improvement.

| Tier | User provides | ρ<sub>nc</sub> (paper) | Recipe |
|---|---|---:|---|
| **A** | Protein sequence (FASTA) | 0.518 ± 0.011 | `esmc_prott5_seed42` |
| **B** | + coding DNA | 0.531 ± 0.014 | `deploy_protein_cds_features_6mod_seed42` |
| **B+** | + 60-nt init window (≥35 nt up, ≥25 nt down of ATG) | 0.543 ± 0.013 | `tier_b_evo2_init_window_classical_rna_init_prott5_seed42` |
| **C** | + full operon DNA (≥100 nt up, ≥50 nt down) | 0.575 ± 0.011 | `evo2_prott5_seed42` |
| **D** (XP5) | + host genome (Bacformer proteome context) | **0.590 ± 0.012** | `balanced_nonmega_5mod` |

See `configs/deployment_tiers.yaml` for full recipe definitions. The Modal demo
serves live inference for Tier A and Tier D; Tiers B / B+ / C are scientific
control conditions with cached predictions on the [Zenodo deposit](https://doi.org/10.5281/zenodo.19639621).

## Quick start

### Option 1: zero-install via the live demo
Visit https://aikium--aikixp-tier-a-landing-page.modal.run — paste any protein + CDS, pick a host (4,566 bacterial reference genomes pre-cached, or paste an NCBI chromosome accession to fetch one on demand), run Tier A through D in seconds. Includes a live per-species calibration scatter (truth vs held-out CV prediction) for visual trust.

### Option 2: Python SDK (recommended for programmatic use)
```bash
pip install aikixp-client
```
```python
from aikixp_client import Client

client = Client()
# Tier A (CPU, no host needed)
r = client.predict_tier_a("MKTVRQERLKSIVRILERSK...")
print(r["predictions"][0]["predicted_expression"])

# Tier D (GPU, 5 modalities — paper champion)
r = client.predict_tier_d(protein="MKT...", cds="ATG...TAA", host="NC_000913.3", mode="native")

# Look up any gene in the 244K held-out CV corpus
g = client.lookup_gene("Escherichia_coli_K12|NP_417556.2")
```
Full client reference: [`sdk/python/README.md`](sdk/python/README.md).

### Option 3: Docker (reproducible local inference)
```bash
# Pull the inference-only image (2 GB)
docker pull ghcr.io/aikium-public/aiki-xp:inference

# Run Tier A on a FASTA file
docker run -v $(pwd):/work ghcr.io/aikium-public/aiki-xp:inference \
  --tier A --input /work/proteins.fasta --output /work/predictions.csv
```

### Option 4: pip install + Python (full training stack)
```bash
git clone https://github.com/aikium-public/aiki-xp.git
cd aiki-xp
pip install -e .

# Download checkpoints + embeddings from Zenodo
python -m zenodo_get 10.5281/zenodo.19639621 -o data/

# Run inference
python scripts/predict.py --input proteins.fasta --tier A --output predictions.csv
```

## Reproduce the paper's numbers

### From the public API (no install needed)

```python
import requests, numpy as np
from scipy.stats import spearmanr

# Fetch a random sample from the 244,002 held-out CV predictions
r = requests.post(
    "https://aikium--aikixp-tier-a-landing-page.modal.run/sample_lookup",
    json={"n": 5000, "seed": 42},
).json()
rows = r["rows"]

# Per-fold mean Spearman rho for Tier D non-mega genes
rhos = []
for f in range(5):
    g = [r for r in rows if r["cv_fold"] == f and not r["is_mega"]]
    rhos.append(spearmanr([r["true_expression"] for r in g],
                          [r["tier_d"] for r in g]).statistic)
print(f"Tier D rho_nm = {np.mean(rhos):.3f} +/- {np.std(rhos):.3f}")
# Expected: 0.590 +/- 0.012 (matches paper Table 1)
```

Or use the bundled reproducer:

```bash
python validation/reproduce_paper_numbers.py path/to/tier_predictions_lookup.parquet
```

### Full-table reproduction

```bash
python validation/build_enriched_lookup.py   # rebuilds the lookup from GCS
python validation/reproduce_paper_numbers.py # prints per-tier rho matching paper
```

## Adding a new host genome

The live demo ships with 4,566 pre-cached bacterial chromosomes — 1,831
from the training corpus (with held-out CV metadata) and 2,734 from the
full Aiki-XP genome cache (accessible but without CV metadata). If your
organism isn't in the typeahead, the landing page offers a **"Request
this genome"** affordance:

1. Type any query in the host field. When there's no match, a link
   appears: *"Request this genome — paste an NCBI accession"*.
2. Paste a chromosome-level NCBI accession (RefSeq `NC_*`/`NZ_*`/`NT_*`,
   GenBank `CP*`/`CM*`/`AE*`/`AL*`). Assembly-level `GCF_*`/`GCA_*` are
   rejected — paste the primary replicon's accession.
3. Optionally check "pre-compute Bacformer cache" to warm up Tier D
   before your first request (adds 5–30 min in the background).
4. Click *Request genome*. The backend fetches from NCBI in ~30 s,
   validates ≥100 annotated CDS and ≤15 Mb size, pickles the record
   to the Modal volume, and refreshes the host typeahead.

The new host is immediately usable by Tier A, B, B+, C, and D. Tier D
on a freshly-added genome incurs a 5–30 min Bacformer cold-start on the
first request unless you opted in to pre-compute at request time.

**Rate limits:** 10 requests/hour per IP, 100 user-added genomes per
launch window, 15 Mb genome-size cap. Multi-record WGS-merged pickles
are supported via `aikixp.genome_lookup.MergedGenome`, which concatenates
contigs with a 2 kb N-gap and re-indexes features into a single
coordinate space — operon inference never bridges contigs because the
gap is larger than `MAX_INTERGENIC_OPERON_DISTANCE` (150 nt). New
`/request_genome` submissions still only accept single-chromosome
accessions (paste the primary replicon, not an assembly GCF_*).

To bulk-load a directory of genome pickles (e.g. a sibling research
cache) into the Modal volume, see `scripts/bulk_upload_genomes.py`.

## Data pipeline: from raw proteomics to the 492K training table

The manuscript's Methods section describes the pipeline that turned PaXDb v6.0 and the Abele 2025 proteomics atlas into a 492K-gene integrated training table with leakage-controlled splits. At a high level:

1. **Source harmonisation.** Gene-product entries from PaXDb v6.0 (∼235K genes, 253 species) and the Abele atlas (256K genes, 249 species, MassIVE MSV000096603) are unified on a per-species z-score scale. Raw abundance values span >5 orders of magnitude and differ systematically between platforms, so z-scoring removes platform bias and makes within-species rank the prediction target.
2. **Genome resolution.** Every gene is mapped to a BioPython `SeqRecord` via a cascade of match strategies (locus_tag → protein_id → exact translation → k-mer), preferring NCBI RefSeq `NC_*` complete genomes where available and falling back to WGS contigs where not. The 1,831 resolved genome pickles ship with the Zenodo deposit.
3. **Operon resolution.** Each gene's operon is inferred from its host's annotated CDS cluster, using a conservative same-strand, ≤300 nt intergenic-gap heuristic. Singletons (71% of genes) are leak-safe for the multimodal-fusion test; multi-gene operons share a single per-operon expression label by design.
4. **Leakage control.** Gene proteins are clustered with MMseqs2 (30% identity, 80% coverage) into families; the five hard-hybrid splits place entire families on one side of train/test, guaranteeing zero cluster overlap. The conserved "ribosomal" component (41% of genes, one connected component spanning all 385 species) is flagged via `is_mega=True` and reported separately throughout.
5. **Embedding extraction.** All 492K genes are passed through 12 foundation models (protein, DNA, RNA, genome-context). Resulting parquets ship on Zenodo. The champion Tier D uses only 5 of the 12 (ESM-C, HyenaDNA, Evo-2 7B operon, Bacformer-large, 5-block classical).

Source code for every step is in `scripts/` (see `scripts/build_public_dataset.py` for the end-to-end pipeline). Each step is idempotent and reads/writes to local paths so it can be re-run against fresh proteomics data.

## Repository structure

```
aikixp/                    # Core package (pip-installable)
  train.py                 # FusionModel training (single_adapter architecture)
  inference.py             # XP5Ensemble: canonical 5-fold inference with normalization
  sequence_normalization.py # Tag stripping for recombinant proteins
  champion_registry.py     # Recipe configurations
  embedding_registry.py    # SHA256 verification of embedding parquets
  genome_lookup.py         # Gene-to-genome-context resolution (native + heterologous)
  classical_features.py    # 5 biophysical feature blocks (codon, protein, disorder, operon, rna_init)
  extract.py               # ESM-C + ProtT5-XL extraction

scripts/                   # Evaluation, prediction, dataset construction
  predict.py               # End-to-end inference CLI
  build_public_dataset.py  # Raw proteomics -> 492K training table
  build_supplementary_tables.py
  external_validation.py   # E. coli LOSO (Mori, Li, Taniguchi)
  predict_v21_temporal_holdout.py
  reproduce_showcase_results.py
  compute_external_contamination.py
  rescore_on_clean_subset.py
  extract_and_predict.py

validation/                # Reviewer-facing reproduction scripts
  reproduce_paper_numbers.py                     # Prints per-tier rho matching paper
  build_enriched_lookup.py                       # Rebuilds tier_predictions_lookup.parquet
  modal_tier_d_native_ecoli_n50_seed42.{py,json} # 50-gene CDS-only smoke test

figures/                   # Reproduce all manuscript figures
  fig1_dataset.py          # Fig 1: dataset overview
  fig2_platform.py         # Fig 2: platform schematic
  fig3_core_results.py     # Fig 3: core results + all SI figures
  fig5_modality.py         # Fig 5: CKA + LOO importance
  fig6_cross_species.py    # Fig 6: cross-species transfer
  fig_comprehensive_external.py  # ED: external evaluation

configs/
  deployment_tiers.yaml    # Tier definitions and dataset-tier mapping
  norm_stats_492k.json     # Classical feature normalization statistics

web/                       # Aikium-branded landing page (hosted on Modal)
  index.html               # Full demo UI w/ scatter + corpus-match + reproduce-button
  aikium_logo.png
  hosts.json               # 1,831 hosts with species keys and test-gene counts
  concepts/                # Aiki-suite logo concepts (10 SVGs)

Dockerfile                 # Inference-only image
Dockerfile.full            # Raw FASTA -> foundation-model extraction -> prediction
```

## Public live endpoints

The Modal demo serves live inference for Tier A (CPU) and Tier D (GPU, the
paper's XP5 champion). Intermediate-tier (B / B+ / C) cached predictions for
every gene in the 492K corpus live in the Zenodo deposit, not as live endpoints.

| Endpoint | Purpose |
|---|---|
| `https://aikium--aikixp-tier-a-landing-page.modal.run` | Landing page + static assets |
| `POST .../species_scatter` | Per-species calibration data (truth vs paper held-out CV) |
| `POST .../find_in_corpus` | Membership check: is this protein in the 492K corpus? |
| `POST .../sample_lookup` | Random sample from the 244K held-out CV predictions |
| `POST .../cds_for_protein` | Auto-fill CDS from protein + host (native match or codon-optimized) |
| `POST .../request_genome` | Fetch + cache a new NCBI chromosome accession on demand |
| `GET  .../genome_status?acc=` | Poll cache / Bacformer-precompute progress for a requested accession |
| `POST https://aikium--aikixp-tier-a-predict-fasta.modal.run` | Tier A end-to-end (CPU) |
| `POST https://aikium--aikixp-tier-d-predict-tier-d-endpoint.modal.run` | Tier D XP5 (GPU, paper champion) |

## Sequence normalization for tagged proteins

Aiki-XP is trained on native bacterial proteins without recombinant tags. For inference on tagged proteins:

```python
from aikixp.sequence_normalization import normalize_sequence

clean_seq, changes = normalize_sequence(
    seq, strip_tags=True, apply_metap=False, ensure_m=True
)
```

This strips common His, HiBit, FLAG, and other affinity tags before scoring.

## Troubleshooting & FAQ

**My first Tier D request took ~90 seconds. Is it broken?** No — the Modal GPU container scales to zero after ~5 minutes of idle, so the first request after a quiet period cold-starts the image (Evo-2 7B, Bacformer-large, ProtT5, ESM-C, HyenaDNA all load into VRAM). Subsequent requests within the idle window return in 10–20 s.

**My host organism isn't in the 4,566-genome cache.** The landing page's host selector shows a "Request this genome" link when your query returns no match — paste a chromosome-level NCBI accession (e.g. `NC_000913.3`, `NZ_CP007039.1`, `CP158060.1`), and the backend fetches it from NCBI in ~30 seconds, validates it has ≥100 annotated CDS, and adds it to the typeahead. Tier A and Tier D work immediately on the new host; Tier D's Bacformer context computes lazily on the first request (5–30 min) unless you check "pre-compute Bacformer cache" at request time. Assembly-level `GCF_*`/`GCA_*` accessions are rejected for now — paste the primary replicon's accession instead. If that still doesn't cover your case, email [venkatesh@aikium.com](mailto:venkatesh@aikium.com).

**My live Tier D prediction differs from the paper's held-out CV prediction for the same gene.** Three things to check. (1) The live endpoint re-extracts all 5 modalities from scratch using the host and anchor you specify; the CV prediction was computed once against the production genome/operon context. (2) The CV prediction is from the one fold that held the gene out — the model never saw it during training. The live prediction uses the full ensemble trained on all folds, so it has a different bias. (3) The held-out CV is the number the paper reports; the live endpoint is what you'd use for proteins not already in the corpus. Small differences (|Δz| < 0.3) are expected.

**Can I batch-predict 10,000 sequences?** Yes — use the Docker image (`ghcr.io/aikium-public/aiki-xp:inference`) or the pip install path for bulk inference. The Modal demo is sized for interactive use (a few predictions at a time) and has per-user rate limits to keep GPU costs bounded.

**What's the uncertainty on a single prediction?** The species-level Spearman ρ on non-conserved genes is 0.590 ± 0.012 (5-fold CV), but that's a rank-correlation metric over a whole proteome. For a single z-score on a non-conserved gene (n=3,037 from a 20K-gene random sample): |z_pred − z_true| < 0.5 for 53% of genes, < 1.0 for 83%, and < 1.5 for 95%; the median absolute error is 0.47. Do not read a single prediction as a calibrated absolute number; read it as a percentile against the host's proteome.

**My protein has a His-tag / HiBit / FLAG tag — will it score correctly?** Tagged sequences are out of distribution for this model. Strip the tag first using `aikixp.sequence_normalization.normalize_sequence(seq, strip_tags=True)` — see the section above.

**I want to cite a specific tier's number.** Use the Tier label + per-fold ρ ± SD from Table 1 of the paper (or the table at the top of this README). Citing "Aiki-XP Tier D achieves ρ_nc = 0.590 ± 0.012" is the standard form.

**A prediction failed with "HTTP 502 / timeout".** Modal's cold-start path can occasionally time out on the first request; retry after 30 seconds. If it keeps failing, open a GitHub issue with the request payload (minus any proprietary sequences).

**How do I share a specific prediction with a collaborator?** Run the prediction on the [live demo](https://aikium--aikixp-tier-a-landing-page.modal.run), then click "Copy share link" in the results panel — the URL encodes the protein, CDS, host, tier, and mode, so anyone opening it reproduces your exact input.

## Citation

```bibtex
@article{tien2026aikixp,
  title   = {Aiki-XP: leakage-controlled multimodal prediction of within-species relative protein expression at pan-bacterial scale},
  author  = {Tien, Hudson and Sharma Meda, Radheesh and Shastry, Shankar and Mysore, Venkatesh},
  journal = {Nature Biotechnology},
  year    = {2026},
  note    = {Under review. Preprint: bioRxiv \url{https://doi.org/10.64898/2026.04.21.719525}.}
}

@misc{tien2026aikixp_zenodo,
  title        = {Aiki-XP: Data and Model Weights},
  author       = {Tien, Hudson and Sharma Meda, Radheesh and Shastry, Shankar and Mysore, Venkatesh},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19639621}
}
```

## License

Code: Apache 2.0 (see [LICENSE](LICENSE)). Data and model weights on Zenodo: CC-BY 4.0.

## Acknowledgments

Aiki-XP builds on foundation models and datasets released by many groups: [PaXDb v6.0](https://pax-db.org) (Huang et al. 2025), the Abele et al. 2025 bacterial proteomics atlas (Mol. Cell. Proteomics, MassIVE MSV000096603), [ESM-C](https://www.evolutionaryscale.ai) (EvolutionaryScale), [ProtT5-XL](https://github.com/agemagician/ProtTrans) (Elnaggar et al.), [HyenaDNA](https://huggingface.co/LongSafari) (Nguyen et al.), [Bacformer](https://github.com/macwiatrak/Bacformer) (Wiatrak, Weimann, Floto et al.), [Evo-2](https://arcinstitute.org/manuscripts/Evo2) (Brixi, Hsu, Hie et al., Arc Institute), and [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) (Lorenz et al.).

Infrastructure: [Modal Labs](https://modal.com) (serverless GPU) and [Zenodo](https://zenodo.org) (permanent data deposit).
