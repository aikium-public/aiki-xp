# Using Aiki-XP for directed-evolution library ranking

**Category:** Q&A
**Tags:** `workflow`, `directed-evolution`, `library-design`
**Title:** What's the right way to use Aiki-XP in a directed-evolution workflow?

---

## The short recipe

1. Generate your library (fold designer, variational model, ancestral reconstruction, whatever).
2. For each variant, get its CDS — either by back-translating with your target host's
   codon usage, or by pulling the exact DNA your oligo pool will synthesise.
3. Score every variant with Tier D, in batches of 50-200 per request.
4. Sort by predicted z. The top 5-10% is your candidate pool for ordering.
5. Order + test. Feed the measurement data back into a per-host calibration check.

## What Aiki-XP is and isn't for in this loop

**Good at:** rank-ordering candidates within the same host. If your library has
one thousand variants of an enzyme for expression in *B. subtilis*, Aiki-XP
will rank them by predicted per-species z with ρ_non-conserved ≈ 0.59 (per the
paper's held-out CV) — useful for culling the bottom 70-80% before wet-lab.

**OK at:** ranking across hosts, when the hosts are both in our 1,831-genome cache.
The per-host calibration can drift by up to ~0.2 z between species (see the
per-species calibration scatter on the demo). Use with care.

**Bad at:** predicting absolute µg/mL yield. Aiki-XP predicts relative rank
within a host's proteome, not how many molecules come out of the fermenter.
Use wet-lab measurement for absolute values.

**Bad at:** fine-grained effect-size prediction for single substitutions. The
model was trained on diverse native sequences, not mutant ladders. For a
point-mutation scan, the delta between wild-type and mutant predictions is often
in the noise (|Δz| < 0.5 for 53% of genes per the FAQ calibration). If you're
doing deep mutational scanning, an expression predictor specialised for single
substitutions will likely outperform Aiki-XP.

## Example: ranking 500 TM domain variants for heterologous expression in *E. coli* K12

```python
from aikixp_client import Client
client = Client()

variants = [...]  # 500 (aa_seq, cds) tuples
preds = client.predict_tier_d(
    protein=[v[0] for v in variants],
    cds=[v[1] for v in variants],
    host="NC_000913.3",
    mode="heterologous",  # since they're foreign to E. coli
)
ranked = sorted(zip(variants, preds["predictions"]),
                key=lambda x: -x[1]["predicted_expression"])
top_50 = ranked[:50]  # order these for synthesis
```

Warm request latency on a batch of 500 is ~20 s; cost ~$0.10.

## Tips

- Use `mode="heterologous"` (the default) when your variants are synthetic
  or from a different organism than the host. `mode="native"` is for cases
  where the CDS you're scoring is already in the host genome exactly.
- If you want a cheap first-pass cull (Tier A on 10,000 variants, then Tier D
  on the top 1,000), the SDK has `predict_tier_a` + `predict_tier_d` with the
  same batching guidance.
- Post-experiment: compare your measured ranks to the Aiki-XP ranks. If the
  correlation is lower than the paper's ρ = 0.59 for non-conserved, that's a
  signal that your library is out-of-distribution in some specific way — tell
  us about it in this discussion thread. We're building a catalogue of
  "where Aiki-XP works and where it doesn't" for v2 training.

## Related

- The paper's §3.3 includes two external directed-evolution benchmarks
  (Cambray 2018, MPEPE) where we report ρ and discuss the cases we lose.
- `ROADMAP.md` item: "Expression-aware protein design" — the long-term plan
  is to close the loop so you can ask Aiki-XP to *propose* variants, not
  just rank yours.
