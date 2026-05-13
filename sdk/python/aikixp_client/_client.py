"""Thin wrapper around the Aiki-XP public Modal endpoints.

All methods return the raw decoded JSON. See ``_types.py`` for the expected shapes.
Errors from the server are raised as :class:`AikixpError` with the HTTP status code
and response body for debugging.
"""
from __future__ import annotations

import json as _json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import requests

from ._constants import DEFAULT_ENDPOINTS, DEFAULT_TIMEOUT_S
from ._types import (
    CdsForProteinResponse,
    FindInCorpusResponse,
    PredictionResponse,
    SampleLookupResponse,
    SpeciesScatterResponse,
)


class AikixpError(RuntimeError):
    """Raised when the Aiki-XP API returns a non-2xx response."""

    def __init__(self, status: int, body: Any, url: str) -> None:
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"Aiki-XP {status} on {url}: {body!r}")


class Client:
    """HTTP client for the Aiki-XP public API.

    Parameters
    ----------
    base_urls : dict, optional
        Override specific endpoint URLs. Keys match the names in
        ``_constants.DEFAULT_ENDPOINTS`` (e.g. ``{"tier_d": "https://..."}``).
    timeout : float, optional
        Default per-request timeout in seconds. CPU endpoints honour 120 s and
        GPU endpoints honour 1200 s by default; set this to override both.
    session : requests.Session, optional
        Supply your own session (for retries, custom headers, etc.).
    """

    def __init__(
        self,
        base_urls: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._urls: Dict[str, str] = dict(DEFAULT_ENDPOINTS)
        if base_urls:
            self._urls.update(base_urls)
        self._timeout_override = timeout
        self._s = session or requests.Session()
        self._s.headers.setdefault("User-Agent", "aikixp-client/1.1.0 python-requests")

    # -------------- internals --------------

    def _timeout(self, kind: str) -> float:
        if self._timeout_override is not None:
            return self._timeout_override
        return DEFAULT_TIMEOUT_S[kind]

    def _post(self, url: str, payload: dict, *, kind: str = "cpu") -> Any:
        r = self._s.post(url, json=payload, timeout=self._timeout(kind),
                         allow_redirects=True)
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise AikixpError(r.status_code, body, url)
        return r.json()

    def _get(self, url: str, *, kind: str = "cpu") -> Any:
        r = self._s.get(url, timeout=self._timeout(kind))
        if not r.ok:
            raise AikixpError(r.status_code, r.text, url)
        return r.json()

    # -------------- corpus lookup --------------

    def sample_lookup(self, n: int = 5000, seed: int = 42) -> SampleLookupResponse:
        """Random sample ``n`` rows from the 244K held-out CV predictions.

        Use this to reproduce the manuscript's per-fold Spearman ρ on a fresh subset
        without downloading the full parquet.
        """
        return self._post(self._urls["sample_lookup"], {"n": int(n), "seed": int(seed)})

    def species_scatter(self, species_keys: Sequence[str]) -> SpeciesScatterResponse:
        """Per-species scatter data: truth vs all 5-tier held-out CV predictions.

        ``species_keys`` is a list of underscored species names (``Escherichia_coli_K12``)
        and/or numeric tax IDs (``"1006000"``). Both aliases are typically included for
        a given host — see ``hosts.json`` for the canonical list.
        """
        return self._post(self._urls["species_scatter"], {"species_keys": list(species_keys)})

    def find_in_corpus(
        self,
        protein: str,
        host: str,
        species_keys: Optional[Sequence[str]] = None,
    ) -> FindInCorpusResponse:
        """Check whether a protein sequence is exactly in the training corpus for ``host``.

        If ``species_keys`` is omitted, the server uses the host's default species keys.
        Useful for "is this gene already scored by the paper's held-out CV?" queries.
        """
        payload: Dict[str, Any] = {"protein": protein, "host": host}
        if species_keys is not None:
            payload["species_keys"] = list(species_keys)
        return self._post(self._urls["find_in_corpus"], payload)

    def hosts(self) -> List[dict]:
        """Fetch the full manifest of the 1,831 cached bacterial host genomes."""
        return self._get(self._urls["hosts"])

    # -------------- auto-fill CDS --------------

    def cds_for_protein(self, protein: str, host: str) -> CdsForProteinResponse:
        """Return a CDS for ``protein`` in ``host``.

        Tries a native exact-substring match against the host genome first; falls
        back to codon-optimised back-translation using the host's own codon usage.
        """
        return self._post(self._urls["cds_for_protein"], {"protein": protein, "host": host})

    # -------------- predictions --------------

    def predict_tier_a(
        self,
        proteins: Union[str, Sequence[str]],
        *,
        gene_ids: Optional[Sequence[str]] = None,
    ) -> PredictionResponse:
        """Tier A prediction (ESM-C + ProtT5, CPU, no host required).

        ρ_non-conserved ≈ 0.518 on the paper's leakage-controlled test split.
        """
        seqs = [proteins] if isinstance(proteins, str) else list(proteins)
        ids = list(gene_ids) if gene_ids else [f"query_{i}" for i in range(len(seqs))]
        fasta = "".join(f">{gid}\n{seq}\n" for gid, seq in zip(ids, seqs))
        return self._post(self._urls["tier_a"], {"fasta": fasta}, kind="cpu")

    def _predict_gpu(
        self,
        url_key: str,
        protein: Union[str, Sequence[str]],
        cds: Union[str, Sequence[str]],
        host: str,
        mode: str = "native",
        anchor: str = "lacZ",
    ) -> PredictionResponse:
        proteins = [protein] if isinstance(protein, str) else list(protein)
        cdss = [cds] if isinstance(cds, str) else list(cds)
        payload = {
            "proteins": proteins,
            "cds": cdss,
            "host": host,
            "mode": mode,
            "anchor": anchor,
        }
        return self._post(self._urls[url_key], payload, kind="gpu")

    def predict_tier_d(
        self, protein: Union[str, Sequence[str]], cds: Union[str, Sequence[str]],
        host: str, mode: str = "native", anchor: str = "lacZ",
    ) -> PredictionResponse:
        """Tier D prediction (XP5 ensemble, 5 modalities, Bacformer-large). ρ_nc ≈ 0.590.

        This is the paper's headline recipe. Cold-start ~90 s; warm ~5 s per sequence.
        Batch multiple proteins/cds in one call to amortise the GPU cost.
        """
        return self._predict_gpu("tier_d", protein, cds, host, mode, anchor)

    # -------------- convenience --------------

    def compare_a_vs_d(
        self,
        protein: str,
        cds: str,
        host: str,
        mode: str = "native",
    ) -> Dict[str, PredictionResponse]:
        """Fire Tier A and Tier D in parallel (thread pool) and return a dict keyed by tier.

        Clock time ≈ max(tier latencies). Cold ~90 s, warm ~10 s.
        The intermediate tiers from the paper are scientific control conditions
        whose cached predictions live on the Zenodo deposit
        (https://doi.org/10.5281/zenodo.19639621), not as live endpoints.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _run(tier: str) -> "tuple[str, Union[PredictionResponse, AikixpError]]":
            try:
                if tier == "A":
                    return tier, self.predict_tier_a(protein)
                return tier, self.predict_tier_d(protein, cds, host, mode=mode)
            except AikixpError as e:
                return tier, e

        with ThreadPoolExecutor(max_workers=2) as ex:
            results = dict(ex.map(_run, ["A", "D"]))
        return results
