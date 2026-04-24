# aikixp-client

Official Python client for the [Aiki-XP](https://github.com/aikium-public/aiki-xp)
bacterial protein-expression prediction API.

Aiki-XP is a leakage-controlled multimodal model trained on 492,026 genes from
385 bacterial species; it predicts within-species relative protein expression
(per-species z-score). See the [paper](https://doi.org/10.5281/zenodo.19639621)
and [main repository](https://github.com/aikium-public/aiki-xp) for background.

## Install

```bash
pip install aikixp-client
```

Python ≥ 3.9. The only dependency is `requests`.

## Quick start

```python
from aikixp_client import Client

client = Client()

# --- Tier A: protein only, no host required (fastest) --------------
r = client.predict_tier_a("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGPGLGLNHVGQIIR")
print(r["predictions"][0]["predicted_expression"])  # e.g. -0.083

# --- Tier D: full 9-modality, A100 (GPU) --------------------------
r = client.predict_tier_d(
    protein="MKT...",
    cds="ATG...TAA",
    host="NC_000913.3",
    mode="native",
)
print(r["predictions"][0]["predicted_expression"])
print(r["modalities_filled"])  # ['evo2_7b_full_operon_pca4096', 'hyenadna_dna_cds', ...]

# --- Look up a gene in the 244K held-out CV corpus ----------------
g = client.lookup_gene("Escherichia_coli_K12|NP_417556.2")
row = g["results"][0]
print(row["tier_d"], row["true_expression"], row["cv_fold"])

# --- Reproduce the paper's numbers on a fresh random sample -------
sample = client.sample_lookup(n=5000, seed=42)

# --- Compare all 5 tiers on the same sequence, in parallel --------
results = client.compare_all_tiers(
    protein="MKT...",
    cds="ATG...TAA",
    host="NC_000913.3",
    mode="native",
)
for tier in ("A", "B", "B+", "C", "D"):
    print(tier, results[tier]["predictions"][0]["predicted_expression"])
```

## Auto-fill a CDS from a protein + host

```python
r = client.cds_for_protein(
    protein="MAKPIL...",
    host="NC_000913.3",
)
print(r["cds"], r["source"])  # "ATG...", "native" or "codon_optimized"
```

## Host catalog

```python
hosts = client.hosts()            # 1,831 bacterial reference genomes
print(f"{len(hosts)} hosts cached")
e_coli = [h for h in hosts if h["acc"] == "NC_000913.3"][0]
print(e_coli["name"], e_coli["n_test"])
```

## Error handling

All HTTP errors raise :class:`AikixpError` with the status code and response body.

```python
from aikixp_client import Client, AikixpError

try:
    client.predict_tier_d(...)
except AikixpError as e:
    if e.status in (502, 504):
        # Cold-start timeout; retry after a short delay
        ...
    else:
        raise
```

## Tiers at a glance

| Tier | Modalities | Hardware | Cold | Warm | ρ_non-conserved |
|---|---|---|---|---|---|
| A | ESM-C + ProtT5 (protein only) | CPU | ~33 s | ~15 s | 0.518 |
| B | + HyenaDNA + classical | A100-40 GB | ~44 s | ~6 s | 0.531 |
| B+ | + Evo-2 7B init-window + RNA init | A100-40 GB | ~79 s | ~10 s | 0.543 |
| C | + Evo-2 7B full operon | A100-40 GB | ~8 s | ~7 s | 0.575 |
| D | + Bacformer-large + operon structure | A100-80 GB | ~6 s | ~5 s | 0.590 |

## Gotchas

- The server scales to zero after ~5 minutes of idle — first calls after a quiet
  period cold-start a fresh GPU container. Default timeouts account for this
  (120 s for CPU endpoints, 1200 s for GPU endpoints).
- Tagged sequences (His6, HiBit, FLAG, etc.) are out-of-distribution. Strip
  tags first using `aikixp.sequence_normalization` from the main repo.
- Native mode requires the user-supplied CDS to exact-substring-match the host
  genome; if it doesn't, the server silently falls back to heterologous mode
  with the lacZ anchor. The response's `predictions[0]["operon_source"]` field
  tells you which path was taken.

## License

Code: Apache 2.0. Aiki-XP model weights + data: CC-BY 4.0 (via
[Zenodo](https://doi.org/10.5281/zenodo.19639621)).

## Citation

```bibtex
@article{tien2026aikixp,
  title = {Aiki-XP: leakage-controlled multimodal prediction of within-species
           relative protein expression at pan-bacterial scale},
  author = {Tien, Hudson and Sharma Meda, Radheesh and Shastry, Shankar and
            Mysore, Venkatesh},
  journal = {Nature Biotechnology},
  year = {2026},
  note = {Under review. Preprint on bioRxiv.}
}
```
