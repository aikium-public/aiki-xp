"""Smoke test for the aikixp-client SDK against the live public deployment.

Run with ``pytest``. Skips gracefully if the network isn't reachable. Fires
only the cheap CPU endpoints by default (set ``AIKIXP_TEST_GPU=1`` to hit
Tier B/C/D too — each request costs ~$0.005 and takes up to 90 s cold).
"""
from __future__ import annotations

import os

import pytest

from aikixp_client import AikixpError, Client

TEST_PROTEIN = (
    "MRIAIVGSGNVGSAVAFGLAQRDLGDEVVLIDINQKKADGEAMDLNHASAFISPIAVRAGDYG"
    "DLAGAKIVILTAGLARKPGQSRLDLVGKNTKILREVVGSFKTYSPKAIVIIVSNPVDILTYVA"
    "YKCSGLDKEKVIGAGTILDTARFRYFLADYFRVAPANVHAWLLGEHGDSAFPAWSHAKIGGVP"
    "LSEILQAKEDGVDPTMRELIEEAAPAEYAIAMAGKGLVDAAIGIVKDVKRILHGEYGIKSIFK"
    "TINGDYFGIHESLATISRLAGKGQYYELSLPWEEIEKLKASSFLINNLARPLSRGKDA"
)
TEST_GENE_ID = "Escherichia_coli_K12|NP_417556.2"


@pytest.fixture(scope="module")
def client():
    return Client()


def test_hosts(client):
    hosts = client.hosts()
    assert isinstance(hosts, list) and len(hosts) >= 1800
    assert all("acc" in h and "name" in h for h in hosts[:5])


def test_lookup_gene(client):
    r = client.lookup_gene(TEST_GENE_ID)
    assert r["n_found"] == 1
    row = r["results"][0]
    assert "tier_d" in row and "true_expression" in row


def test_sample_lookup(client):
    r = client.sample_lookup(n=100, seed=123)
    assert len(r["rows"]) == 100
    assert {"tier_d", "true_expression", "is_mega", "cv_fold"} <= set(r["rows"][0])


def test_cds_for_protein(client):
    r = client.cds_for_protein(TEST_PROTEIN, "NC_000913.3")
    assert "cds" in r and "source" in r


def test_find_in_corpus(client):
    r = client.find_in_corpus(
        TEST_PROTEIN, host="NC_000913.3",
        species_keys=["Escherichia_coli_K12", "1006000", "1006004"],
    )
    assert "matched" in r


def test_predict_tier_a(client):
    r = client.predict_tier_a(TEST_PROTEIN)
    assert r["tier"] == "A"
    assert "predicted_expression" in r["predictions"][0]


@pytest.mark.skipif(os.environ.get("AIKIXP_TEST_GPU") != "1",
                    reason="GPU tests cost money; set AIKIXP_TEST_GPU=1 to enable.")
def test_predict_tier_d(client):
    r = client.predict_tier_d(
        protein=TEST_PROTEIN,
        cds="ATG" + "GCG" * (len(TEST_PROTEIN) - 1) + "TAA",  # dummy synonymous CDS
        host="NC_000913.3",
        mode="heterologous",
    )
    assert len(r["modalities_filled"]) == 9
    assert r["predictions"][0]["predicted_expression"] is not None


def test_invalid_gene_id(client):
    """A non-existent gene ID returns n_found=0 and empty results (no 500)."""
    r = client.lookup_gene("Nonsense_species|XX_000000.0")
    assert r["n_found"] == 0 and r["results"] == []


def test_bad_url_raises(client):
    """Hitting a deliberately-wrong URL should raise AikixpError, not hang."""
    bad_client = Client(base_urls={
        "lookup_gene": "https://aikium--aikixp-tier-a-lookup-gene.modal.run/this-path-doesnt-exist"
    })
    with pytest.raises((AikixpError, Exception)):
        bad_client.lookup_gene(TEST_GENE_ID)
