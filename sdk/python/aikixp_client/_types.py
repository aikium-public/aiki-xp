"""Typed data structures for the Aiki-XP API responses.

Uses ``TypedDict`` so the runtime is zero-cost and type-checkers see proper shapes.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict


class Prediction(TypedDict, total=False):
    predicted_expression: float
    operon_source: str                   # "native" | "heterologous"
    operon_length_nt: int
    cds_start_in_operon: int
    bacformer_available: bool
    gene_id: str


class PredictionResponse(TypedDict, total=False):
    mode: str                            # "native" | "heterologous"
    host: str
    tier: str                            # "A" | "B" | "B+" | "C" | "D"
    recipe: str
    n_sequences: int
    predictions: List[Prediction]
    modalities_filled: List[str]
    modalities_zero_filled: List[str]


class GeneLookup(TypedDict, total=False):
    gene_id: str
    species: str
    is_mega: bool
    cv_fold: int
    true_expression: float
    tier_a: float
    tier_b: float
    tier_b_plus: float
    tier_c: float
    tier_d: float


class LookupGeneResponse(TypedDict):
    n_found: int
    n_missing: int
    missing_gene_ids: List[str]
    note: str
    results: List[GeneLookup]


class CdsForProteinResponse(TypedDict, total=False):
    cds: str
    source: str                          # "native" | "codon_optimized"
    matched_gene: Optional[str]
    host: str
    error: Optional[str]


class SampleLookupResponse(TypedDict):
    n: int
    seed: int
    rows: list  # list of dicts with tier_a..tier_d, true_expression, is_mega, cv_fold, ...


class SpeciesScatterResponse(TypedDict, total=False):
    n: int
    n_nonmega: int
    rho_overall: float
    rho_nonmega: float
    species_keys: List[str]
    points: list


class FindInCorpusResponse(TypedDict, total=False):
    matched: bool
    gene_id: Optional[str]
    truth: Optional[float]
    tier_d_cv: Optional[float]
    cv_fold: Optional[int]
    is_mega: Optional[bool]
    reason: Optional[str]
