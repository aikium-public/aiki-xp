# Aiki-XP Data

The training data and pre-extracted embeddings are hosted on Zenodo due to size (~50 GB total).

**DOI:** `10.5281/zenodo.XXXXXXX`

## Download

```bash
# Install zenodo_get if needed
pip install zenodo_get

# Download all files
zenodo_get 10.5281/zenodo.XXXXXXX -o data/
```

## Contents

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `aikixp_492k_v1.parquet` | Training dataset (genes, sequences, expression labels, split assignments) | 492,026 | ~30 |
| `embeddings/` | Pre-extracted foundation model embeddings (one parquet per modality) | 492,026 each | varies |
| `splits/` | Gene-operon and species-cluster split assignments | — | — |

## Schema

See `DATA_SCHEMA.md` for column descriptions, data types, and provenance.

## Expression labels

- **Target column:** `expression_level` (per-species z-scored log2 abundance)
- **Sources:** PaXDb v6.0 (https://pax-db.org) and Abele et al. 2025 (ProteomicsDB Project 4498, MassIVE MSV000096603)
- **Label source column:** `expression_source` identifies PaXDb vs Abele origin (NaN for 36,623 curated-cohort genes with mixed provenance)

## Split reconstruction

The `gene_cluster_id` and `compound_operon_id` columns, together with the split files in `splits/`, allow exact reconstruction of the gene-operon split used in the paper. Zero of 129,078 MMseqs2 clusters span splits.
