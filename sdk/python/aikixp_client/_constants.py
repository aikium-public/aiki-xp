"""Default Modal endpoint URLs for the public Aiki-XP deployment.

These can be overridden via ``Client(base_urls={...})`` if you deploy your own
Modal app (e.g. behind a custom domain).
"""
from __future__ import annotations

DEFAULT_LANDING = "https://aikium--aikixp-tier-a-landing-page.modal.run"

DEFAULT_ENDPOINTS = {
    # Landing ASGI routes
    "hosts":          f"{DEFAULT_LANDING}/hosts.json",
    "sample_lookup":  f"{DEFAULT_LANDING}/sample_lookup",
    "species_scatter": f"{DEFAULT_LANDING}/species_scatter",
    "find_in_corpus":  f"{DEFAULT_LANDING}/find_in_corpus",

    # Separate Modal functions
    "lookup_gene":      "https://aikium--aikixp-tier-a-lookup-gene.modal.run",
    "cds_for_protein":  "https://aikium--aikixp-tier-a-cds-for-protein.modal.run",
    "tier_a":           "https://aikium--aikixp-tier-a-predict-fasta.modal.run",
    "tier_b":           "https://aikium--aikixp-tier-d-predict-tier-b-endpoint.modal.run",
    "tier_b_plus":      "https://aikium--aikixp-tier-d-predict-tier-b-plus-endpoint.modal.run",
    "tier_c":           "https://aikium--aikixp-tier-d-predict-tier-c-endpoint.modal.run",
    "tier_d":           "https://aikium--aikixp-tier-d-predict-tier-d-endpoint.modal.run",
}

# Sensible default timeouts per endpoint class.
# Cold Tier D can take ~90s; warm ~5s. 20-minute cap keeps pathological cases bounded.
DEFAULT_TIMEOUT_S = {
    "cpu":  120,     # landing routes, lookup_gene, cds_for_protein, tier_a
    "gpu":  1200,    # tier B/B+/C/D
}
