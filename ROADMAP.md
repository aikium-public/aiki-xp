# Aiki-XP roadmap

Last updated: 2026-04-21 (launch day).

This document tracks what we plan to ship in the near, medium, and long term.
Not every item is committed — near-term is funded work we intend to deliver;
medium-term is design-stage; long-term is where we see the field going. We
welcome feedback, pull requests, and collaboration across all three tiers.

The `CHANGELOG.md` is the authoritative record of what has shipped.

---

## Near-term (weeks 1-4 post-launch)

### Infrastructure
- **[done 2026-04-21]** User-requested genome additions: `POST /request_genome`
  endpoint plus an inline "Request this genome" control in the host selector
  when the typeahead returns no match. Validates NCBI chromosome accessions,
  fetches via Entrez, pickles to the Modal volume, merges into `/hosts.json`.
  Rate-limited (10 req/hr per IP, 100 user-added cap). Optional Bacformer
  pre-compute spawned to Tier D in the background. Also shipped: bulk upload
  of the full Aiki-XP genome cache via `scripts/bulk_upload_genomes.py`
  (2,734 validated chromosomes → typeahead now covers 4,566 hosts).
- **Multi-record WGS/draft-assembly support**: current pipeline's
  `genome_lookup` assumes a single SeqRecord per pickle; ~7,000 WGS-merged
  pickles in the full cache are excluded until a list-of-records code path
  is added. Tracked for post-launch polish.
- **Python SDK on PyPI** (`pip install aikixp-client`): a thin, typed wrapper
  around the public API so biotech developers don't have to write raw
  `requests.post(...)` calls.
- **Custom domain** `demo.aikium.com` fronting the Modal demo for cleaner
  branded links.
- **Restore Tier B+ live inference on the Modal demo.** The paper's cached
  Tier B+ numbers remain available via `/lookup_gene`; live inference
  needs a Transformer-Engine-capable deployment target (see CHANGELOG
  v1.0.1).

### Scientific / UX
- **Precompute Tier D for all 4,566 cached genomes** (~13M proteins, ~3,500
  A100-hours). Turns the "look up my gene's prediction" flow instant for every
  protein in any of our cached hosts.
- **Nearest-neighbour similarity search** in the 492K-gene corpus: paste a
  protein, see the top-5 most similar training genes with their PaxDb truth
  values. The deepest trust indicator we can build on top of the existing data.
- **Per-prediction modality attribution**: bar chart showing how much each
  of the 9 modalities contributed to a Tier D prediction (ablation-based).
- **External-validation dashboard**: static page showing Aiki-XP's performance
  on the MPEPE, Lisowska, Cambray, Hwang, and Price benchmarks that appear in
  the paper's §3.2, updated if we add new external datasets.

### Community
- **Amazon Bio Discovery catalog submission**.
- **Pre-seeded GitHub Discussions** with Q&As for directed evolution,
  tier selection, and common failure modes.

---

## Medium-term (months 1-3)

### Scientific
- **Fill PaxDb / Abele coverage gaps**: many cached genomes have partial
  proteomics coverage; precompute predictions + host proteome calibration
  for genes with no ground truth so users have a dense reference.
- **Uncertainty quantification**: per-prediction confidence intervals via
  ensemble spread or MC-dropout, visible in both the API and the demo.
- **Support for archaeal hosts**: the same architecture and pipeline should
  extend naturally; we need the proteomics data first.
- **Improved host-coverage**: target >5,000 cached bacterial genomes,
  covering >90% of the RefSeq reference collection.

### Model improvements
- **v2 training run** with refreshed Bacformer-large embeddings and
  additional proteomics data (MassIVE releases since the 2026 freeze).
- **Ablation of modality attention weights**: the XP5 ensemble treats all
  modalities equally; let a small network learn per-prediction weights.
- **Cross-species fusion**: explicitly model the distance between the target
  host and the closest training-set species; currently implicit.

---

## Long-term (months 3-12)

### Model scope
- **Eukaryotic hosts** (yeast, CHO, insect, mammalian). Needs a new training
  corpus and almost certainly a different modality mix (splicing, UTRs,
  chromatin context). Separate paper.
- **Cell-free / synthetic expression systems** (PURE, cell-free yeast, TXTL).
  Very different distribution — targeted fine-tuning, possibly specialised
  sub-models.
- **Absolute yield prediction** (µg/mL, not rank): requires calibration data
  paired with the full growth / harvest / purification pipeline. Wet-lab
  collaboration required.

### Design / inverse problems
- **Expression-aware protein design**: given a target protein family and a
  host, suggest sequence edits (synonymous codon choice, operon-neighbour
  swaps, 5' UTR tweaks) predicted to raise the expression z-score.
- **Joint modelling of expression × function**: expression × enzyme activity
  or expression × stability; the target is a design tool that maximises both.

### Ecosystem integration
- **Hook into directed-evolution frameworks**: an Aiki-XP scoring function
  for libraries generated by existing design tools (e.g. ProGen, Chroma).
- **Cloud-lab integrations**: one-click from a ranked candidate to an
  ordered DNA template at a cloud lab.

---

## How to propose a roadmap item

- Open a [GitHub Discussion](https://github.com/aikium-public/aiki-xp/discussions)
  in the "Ideas" category with: motivation, sketched approach, what success
  looks like, and what you're willing to help with.
- For pre-competitive partnerships / sponsored features, email
  [venkatesh@aikium.com](mailto:venkatesh@aikium.com).
