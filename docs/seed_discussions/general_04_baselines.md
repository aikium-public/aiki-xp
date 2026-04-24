# How does Aiki-XP compare to <other expression predictor>?

**Category:** General
**Tags:** `baselines`, `comparisons`, `benchmarks`
**Title:** How does Aiki-XP compare to <your favourite expression predictor>?

---

## The short answer

The paper's §3 and §3.2 have the full comparison table. The headline:

- On our own leakage-controlled 5-fold CV (1 class-balanced, 385-species split;
  non-conserved genes only): **Aiki-XP Tier D ρ = 0.590 ± 0.012**, ESM-C-600M
  baseline ρ = 0.509, linear regression on codon-adaptation index + GC + length
  ρ = 0.301.
- On the external Cambray 2018 lacZ library: Aiki-XP Tier D ρ = 0.42;
  specialised mRNA-LM (Hwang 2025) ρ = 0.48 — we lose here, and the paper
  discusses why (context-specific codon optimisation signal that our tier ladder
  under-weights).
- On MPEPE (Zhang 2022, recombinant mammalian in *E. coli*): Aiki-XP Tier D
  ρ = 0.35; ESM-C alone ρ = 0.30; specialised MPEPE-tuned tree ρ = 0.52. The
  general-purpose model loses to the task-specific one, as expected.

## Baselines we report in the paper

- **Classical** (CAI, codon usage, protein length, %GC, linear regression).
- **Protein-LM only:** ESM-2 650M, ESM-C 600M, ProtT5-XL, all with a trained
  linear head on top.
- **DNA-LM:** HyenaDNA, Evo-2 (1B, 7B, both checkpoints) individually.
- **Published expression predictors:** Hwang 2025 (mRNA-LM), MPEPE (Zhang 2022),
  Price-NESG, Lisowska.

Aiki-XP beats all general-purpose baselines on the leakage-controlled split
and is competitive with or beats specialised predictors on external benchmarks
with two exceptions (Cambray, MPEPE) where the specialist wins by 0.06-0.17 ρ.

## Want to add a comparison we didn't cover?

1. Open a "Scientific question" issue or reply to this discussion with:
   - The paper and repo of the model you want us to compare to.
   - The benchmark dataset + split that's fair for both.
   - Your own run of both models on that split (ρ + standard error).
2. We'll add it to the post-launch comparison matrix (tracked in `ROADMAP.md`)
   and credit you in the changelog.

## Things to watch out for when comparing

- **Split leakage** is the #1 failure mode. We use gene-operon + MMseqs2-cluster
  holdout; many published predictors use random gene-level splits that let
  homologues leak between train and test. Compare fairly and you'll see Aiki-XP's
  edge narrows on random splits (~0.05 ρ over ESM-C) and widens on leakage-
  controlled splits (~0.08 ρ over ESM-C).
- **Conserved vs non-conserved**: the "mega" cluster genes (abundant
  housekeeping genes with orthologues in many species) are where ANY model
  looks good. We report non-conserved ρ by default because that's the harder
  and more informative number. Make sure your baseline is reported on the
  same subset.
- **Per-species ρ vs pooled ρ**: pooling across species inflates the number by
  ~0.05 because inter-species mean differences are easy to learn. Use the
  per-fold per-species mean for an honest number.

## Related

- Paper §3.2 comparison table (lives in the manuscript repo, reproducible from
  the Zenodo archive).
- `ROADMAP.md`: the external-validation dashboard is a near-term deliverable.
