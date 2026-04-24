# Can I use Aiki-XP for <my organism>?

**Category:** Q&A
**Tags:** `hosts`, `organisms`, `coverage`
**Title:** Can Aiki-XP handle my organism? How to check, and what to do if not.

---

## Quick answer table

| Organism | Answer |
|---|---|
| *E. coli* K12, *B. subtilis* 168, *P. aeruginosa* PAO1, *M. tuberculosis* H37Rv, *C. diff* 630, *S. aureus* USA300, *V. cholerae* N16961, *Streptomyces coelicolor* A3(2) | Yes — these are in the cache and well-covered by PaXDb/Abele ground truth. |
| Most named bacterial strains with a complete reference genome on NCBI | Probably yes — check the host typeahead on the live demo, or `/hosts.json`, or the SDK's `client.hosts()`. |
| A specific strain not in our 1,831-genome cache | Use the closest related species + **file a host-request issue** (see below). |
| An archaeon | Not yet. Architecturally feasible, needs training data. |
| Any eukaryote (yeast, CHO, plant, mammalian, insect) | **No.** Out of scope for this release. Different scale on the training corpus required. |
| Cell-free / synthetic systems (PURE, TXTL, yeast lysate) | **No.** We train on in-vivo proteomics atlases. |

## How to check

```python
from aikixp_client import Client
c = Client()
hosts = c.hosts()
print(len(hosts), "hosts cached")
print([h for h in hosts if 'coli' in h['name'].lower()][:5])
```

Or use the typeahead on [the live demo](https://aikium--aikixp-tier-a-landing-page.modal.run/) —
it also suggests the closest phylogenetic neighbour when your organism isn't a match.

## I want my organism added

File a **"Request a new host genome"** issue:
[github.com/aikium-public/aiki-xp/issues/new?template=host_request.yml](https://github.com/aikium-public/aiki-xp/issues/new?template=host_request.yml).

Include:
- NCBI chromosome-level assembly accession (NC_*, NZ_*, CP_*, CM_*).
- The organism name + strain.
- One sentence about your use case.

Typical turnaround for a bacterial chromosome-level accession is same-day.
There's also an upcoming self-service path — the request-a-genome textbox
on the live demo (tracked in `ROADMAP.md`) — which will let you add new
organisms without human review for most cases.

## What if I need a phylum we don't cover well?

The paper's §3.4 reports per-phylum held-out performance. Some phyla (Firmicutes,
Gammaproteobacteria) are very well covered because our training corpus is
dense there; others (Chlamydiae, some Spirochaetes) have much less ground
truth and predictions are less well-calibrated. Per-species ρ for those can
dip below 0.4 and occasionally below 0.2 on tiny test sets.

If your phylum is poorly covered, you can still run predictions — just calibrate
against a small wet-lab set before trusting the rank. We'd love to hear about
it in this thread.

## Related

- The `hosts.json` manifest lists every cached genome with `n_test` (held-out
  gene count) and `n_test_nm` (non-mega subset). Genomes with low `n_test` have
  less PaxDb/Abele coverage and noisier per-host calibration.
- The paper's Figure 3 is the per-phylum breakdown.
