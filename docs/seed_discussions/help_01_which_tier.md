# Which tier should I use?

**Category:** Q&A
**Tags:** `tiers`, `getting-started`, `performance`
**Title:** Which tier should I use? A / B / B+ / C / D — what changes as you go up the ladder?

---

## The quick answer

| Use case | Recommended tier |
|---|---|
| Scoring a single novel protein, no host in mind, fastest possible | **Tier A** |
| Ranking a small library (~10 candidates) in an arbitrary host | **Tier C** |
| Final scoring of a ranked short-list (~top 20), best accuracy | **Tier D** |
| You want every modality the paper uses, no ambiguity | **Tier D** |
| You're on a budget / CPU only / offline | **Tier A** (Docker `:inference` image) |

## The longer answer

Each tier of Aiki-XP adds one or more modalities on top of the previous tier, in a
monotone ladder. The fusion model at each level is trained separately so that
every step is a statistically significant gain over the previous one (details in
the paper's §2.4 ablations).

- **Tier A** (CPU, ~15 s warm, ρ_nc = 0.518). ESM-C + ProtT5 on the protein alone.
  No host required. This is the fastest, cheapest tier — great for initial
  screening of very large libraries.
- **Tier B** (+ HyenaDNA on the CDS + classical features, GPU, ~6 s warm, ρ_nc = 0.531).
  Adds the CDS-language-model signal. Marginal ρ gain is small; the main reason
  you'd use Tier B over A is if you already have CDS and want to see whether it
  changes the ranking.
- **Tier B+** (+ Evo-2 7B on the init-70nt window + classical RNA-init, ρ_nc = 0.543).
  Adds the 5' init-region genome LM signal. Worth the extra step if your ranking
  depends on subtle translation-initiation differences.
- **Tier C** (+ Evo-2 7B on the full operon DNA, ρ_nc = 0.575). **This is the biggest
  single jump in the ladder** (+0.032 ρ vs B+). The full-operon context captures
  co-regulation and polycistronic organisation.
- **Tier D** (+ Bacformer-large genome-level embedding + operon-structure features,
  ρ_nc = 0.590). **The paper's headline recipe**, 9 modalities. Adds another
  ~0.015 ρ over C and is the recommended tier when accuracy matters more than cost.

## Cost (per warm request, Modal prices)

- Tier A: ~$0.001 / sequence, 15 s
- Tier B: ~$0.004 / sequence, 6 s
- Tier B+: ~$0.006 / sequence, 10 s
- Tier C: ~$0.005 / sequence, 7 s
- Tier D: ~$0.004 / sequence, 5 s

**Batch your requests.** Tier D on a list of 100 sequences in one call takes
~5-10 s total, amortising the GPU overhead. Single-sequence calls waste 95% of
the GPU time on model load + featurisation.

## If you can't decide — use Tier D

If you only care about one number per protein and cost isn't a concern, use
Tier D. It's what we used for every number in the paper. The other tiers exist
because we want the ablation story to be explicit and reproducible, not because
they're normally the right choice for new work.

## The demo's "Compare all 5 tiers" button

The live demo has a "Compare all 5 tiers" button that fires all five in parallel
on the same input and shows you the bar chart of the resulting predictions. If
you're undecided, click it once on your sequence — the shape of the ladder often
makes the choice obvious. A flat ladder means Tier A is enough; a steep one
means the extra context matters.

## See also
- Paper §2 (Deployment ladder) and §2.4 (per-tier ablations)
- `CHANGELOG.md` for the ρ numbers at each tier
- `ROADMAP.md` for planned tier additions (uncertainty quantification in v2)
