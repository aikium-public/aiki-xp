# Changelog

All notable changes to Aiki-XP are documented in this file.

## [1.0.1] — 2026-04-21

- Tier B+ live inference on the Modal demo is not available; the paper's
  cached ρ<sub>nc</sub> = 0.543 predictions remain accessible via
  `/lookup_gene`, and live Tier B+ inference is available from the
  `ghcr.io/aikium-public/aiki-xp:full` Docker image.
- `aikixp/inference.py`: per-modality feature dim must match the adapter's
  expected dim exactly; a mismatch now raises `ValueError`.

## [1.0.0] — 2026-04-21

Public launch: paper, preprint, data and model weights on Zenodo, Python
package, Docker images, Python SDK, and an interactive Modal demo.

### Scientific headline numbers

Per-fold mean over 5-fold CV, gene-operon leakage-controlled split:

| Tier | Modalities | ρ<sub>overall</sub> | ρ<sub>non-mega</sub> |
|---|---|---:|---:|
| A | ESM-C + ProtT5 | 0.5825 | 0.5182 |
| B | + HyenaDNA + classical | 0.5929 | 0.5312 |
| B+ | + Evo-2 7B init-window | 0.6035 | 0.5428 |
| C | + Evo-2 7B full operon | 0.6501 | 0.5747 |
| **D / XP5** | **+ Bacformer-large + operon structure** | **0.6675** | **0.5904** |

Training corpus: 492,026 genes across 385 bacterial species and 1,831
reference genomes.

---

- Paper DOI: [10.5281/zenodo.19639621](https://doi.org/10.5281/zenodo.19639621)
- Live demo: [aikium--aikixp-tier-a-landing-page.modal.run](https://aikium--aikixp-tier-a-landing-page.modal.run/)
