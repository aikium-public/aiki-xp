#!/usr/bin/env python3
"""Comprehensive classical (non-LLM) feature extraction for ProtEx.

Computes ~80 features across operon-structural, DNA/codon, RNA/thermodynamic,
and protein layers that capture biology no language model can learn from
sequence alone:

  0. OPERON STRUCTURAL / POSITIONAL METADATA (Tier 0 — no LLM captures this)
     operon_length_nt, operon_num_genes, gene_position_in_operon,
     gene_relative_position, is_first/last/singleton, cds_fraction_of_operon,
     upstream/downstream intergenic distances

  1. HOST-SPECIFIC CODON ADAPTATION (Tier 1 — highest complementarity)
     tAI, CAI, ENC, Fop, CPB, codon ramp ratio, GC1/GC2/GC3

  2. RNA THERMODYNAMICS — extended (Tier 1–2)
     Recomputes MFE/accessibility WITH ViennaRNA, plus:
     SD binding ΔG (thermodynamic), ensemble free energy, ensemble diversity,
     DRACH (m6A) motif density, CpG observed/expected

  3. PROTEIN PHYSICOCHEMISTRY (Tier 2)
     pI, GRAVY, instability index, aliphatic index, MW, aromaticity,
     charge@pH7, cysteine fraction, N-end rule, Boman index

  4. PROTEIN DISORDER & CHARGE PATTERNING (Tier 3)
     metapredict disorder stats, localCIDER FCR/NCPR/κ/Ω,
     hydrophobic patches, PEST motifs, low-complexity fraction

  5. SEQUENCE MOTIFS & REGULATORY SIGNALS
     N-glycosylation (NxS/T), m6A (DRACH), Dam methylation (GATC),
     rare codon clusters

Outputs (all keyed by gene_id):
  classical_operon_structural_features.parquet — 10 features per gene
  classical_codon_features.parquet     — 11 features per gene
  classical_protein_features.parquet   — ~25 features per gene
  classical_rna_thermo_features.parquet — 12+ features per gene (init window)
  classical_rna_junc_features.parquet   — 8 features per gene (junction, DEPRECATED)
  classical_features_combined.parquet   — all features joined
  classical_features.summary.json       — run metadata

Dependencies:
  Required: pandas, numpy, biopython
  Recommended: ViennaRNA (pip install ViennaRNA)  — thermodynamic features
  Recommended: codon-bias (pip install codon-bias) — codon adaptation metrics
  Recommended: peptides (pip install peptides)     — accurate pI/instability
  Recommended: localcider (pip install localcider) — charge patterning
  Recommended: metapredict (pip install metapredict) — disorder prediction

Usage:
  python scripts/protex/featurize_classical.py                  # all features
  python scripts/protex/featurize_classical.py --skip-vienna    # no ViennaRNA
  python scripts/protex/featurize_classical.py --skip-disorder  # no metapredict (slow)
  python scripts/protex/featurize_classical.py --codon-only     # just codon features
  python scripts/protex/featurize_classical.py --protein-only   # just protein features

License notes:
  All libraries used here are MIT / BSD / LGPL — free for commercial use.
  See --license-report flag for per-feature license info.

  SKIPPED (license concern — flag for open-source model):
    - TANGO / Zyggregator (aggregation prediction) — academic-only license
    - IUPred2A (disorder) — academic-only; we use metapredict (MIT) instead
    - NetSurfP (surface accessibility) — academic-only
    - SignalP (signal peptide) — academic-only
    - RBS Calculator v2.1 (Salis Lab) — academic-only; we reimplement SD thermo

References:
  Oyarzún et al. 2025 (NAR): mechanistic features improve OOD generalization
  Boël et al. 2016 (Nature): codon content influences expression
  Kudla et al. 2009 (Science): 5' MFE explains >50% expression variation
  Washburn et al. 2019: protein properties predict heterologous expression
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("featurize_classical")

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "datasets" / "protex_v0.2.1_corrected_20260129"
PROD_TABLE = DATA_DIR / "production_table" / "protex_v0.2.1_production_with_dna_rna_subseq.parquet"
OUTPUT_DIR = DATA_DIR

# ── Optional dependency checks ────────────────────────────────────────────────

def _check_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False

HAS_VIENNA = _check_import("RNA")
HAS_CODONBIAS = _check_import("codonbias")
HAS_PEPTIDES = _check_import("peptides")
HAS_LOCALCIDER = _check_import("localcider")
HAS_METAPREDICT = _check_import("metapredict")
HAS_BIOPYTHON = _check_import("Bio")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: DNA / CODON FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# Standard genetic code — stop codons excluded from adaptation metrics
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Synonymous codon families (standard genetic code)
CODON_TABLE = {
    "F": ["TTT", "TTC"], "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "I": ["ATT", "ATC", "ATA"], "M": ["ATG"], "V": ["GTT", "GTC", "GTA", "GTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "P": ["CCT", "CCC", "CCA", "CCG"], "T": ["ACT", "ACC", "ACA", "ACG"],
    "A": ["GCT", "GCC", "GCA", "GCG"], "Y": ["TAT", "TAC"],
    "H": ["CAT", "CAC"], "Q": ["CAA", "CAG"], "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"], "D": ["GAT", "GAC"], "E": ["GAA", "GAG"],
    "C": ["TGT", "TGC"], "W": ["TGG"], "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
}

# Reverse: codon -> amino acid
CODON_TO_AA = {}
for aa, codons in CODON_TABLE.items():
    for c in codons:
        CODON_TO_AA[c] = aa


def _extract_codons(dna_cds: str) -> List[str]:
    """Extract codons from CDS DNA, trimming to multiple of 3."""
    seq = dna_cds.upper().replace("U", "T")
    n = (len(seq) // 3) * 3
    return [seq[i:i+3] for i in range(0, n, 3)]


def _gc_at_positions(codons: List[str]) -> Tuple[float, float, float]:
    """Compute GC fraction at each codon position (GC1, GC2, GC3)."""
    if not codons:
        return (0.0, 0.0, 0.0)
    gc = [0, 0, 0]
    n = len(codons)
    for codon in codons:
        if len(codon) < 3:
            continue
        for pos in range(3):
            if codon[pos] in "GC":
                gc[pos] += 1
    return (gc[0] / n, gc[1] / n, gc[2] / n)


def _compute_enc_wright(codons: List[str]) -> float:
    """Effective Number of Codons (Wright 1990).

    Reference-free measure of codon usage bias.
    Range: 20 (maximum bias) to 61 (no bias).
    """
    if HAS_CODONBIAS:
        from codonbias.scores import EffectiveNumberOfCodons
        try:
            cds_str = "".join(codons)
            enc = EffectiveNumberOfCodons()
            result = enc.get_score([cds_str])
            if len(result) > 0 and np.isfinite(result[0]):
                return float(result[0])
        except Exception:
            pass

    # Fallback: manual ENC calculation
    codon_counts: Dict[str, Counter] = {}
    for aa, syns in CODON_TABLE.items():
        if len(syns) <= 1:
            continue  # Skip single-codon families (M, W)
        family_counts = Counter()
        for codon in codons:
            if codon in syns:
                family_counts[codon] += 1
        if sum(family_counts.values()) > 0:
            codon_counts[aa] = family_counts

    if not codon_counts:
        return 61.0  # No bias

    # Group families by degeneracy
    groups: Dict[int, List[float]] = {}
    for aa, counts in codon_counts.items():
        deg = len(CODON_TABLE[aa])
        n_total = sum(counts.values())
        if n_total < 2:
            continue
        # F-statistic for this family
        p_vals = [counts[c] / n_total for c in CODON_TABLE[aa]]
        f_val = (n_total * sum(p ** 2 for p in p_vals) - 1) / (n_total - 1)
        if deg not in groups:
            groups[deg] = []
        groups[deg].append(f_val)

    # ENC formula (Wright 1990): Nc = 2 + 9/F̂₂ + 1/F̂₃ + 5/F̂₄ + 3/F̂₆
    # The 2 accounts for Met + Trp (single-codon amino acids, each = 1 effective codon).
    enc = 2.0  # Met and Trp each contribute exactly 1 effective codon
    for deg in [2, 3, 4, 6]:
        if deg in groups and groups[deg]:
            mean_f = np.mean(groups[deg])
            if mean_f > 0:
                n_families = {2: 9, 3: 1, 4: 5, 6: 3}.get(deg, 1)
                enc += n_families / mean_f
            else:
                n_families = {2: 9, 3: 1, 4: 5, 6: 3}.get(deg, 1)
                enc += n_families * deg

    return min(61.0, max(20.0, enc))


def _compute_rscu(codons: List[str]) -> Dict[str, float]:
    """Relative Synonymous Codon Usage (Sharp & Li 1986)."""
    codon_freq = Counter(codons)

    rscu = {}
    for aa, syns in CODON_TABLE.items():
        total = sum(codon_freq.get(c, 0) for c in syns)
        n_syn = len(syns)
        for c in syns:
            if total > 0:
                rscu[c] = (codon_freq.get(c, 0) / total) * n_syn
            else:
                rscu[c] = 1.0  # Equal usage if no data
    return rscu


def _compute_cai(codons: List[str], ref_rscu: Dict[str, float]) -> float:
    """Codon Adaptation Index (Sharp & Li 1987).

    Uses per-species reference RSCU to compute relative adaptiveness.

    NOTE: Our reference set is ALL genes in the species (genome-derived RSCU),
    not the canonical "highly expressed genes" reference (Sharp & Li 1987).
    This makes our CAI a measure of deviation from species-average codon usage
    rather than adaptation toward optimal usage.  This is a common modern
    variant (e.g., used by the CAI Python package when given custom references).
    The metric still captures host-specific codon adaptation signal that LMs miss.
    """
    # Compute relative adaptiveness: w_i = RSCU_i / max(RSCU in family)
    # Floor at 0.5 per Sharp & Li convention for codons absent from reference.
    w = {}
    for aa, syns in CODON_TABLE.items():
        if len(syns) <= 1:
            continue
        max_rscu = max(ref_rscu.get(c, 0.0) for c in syns)
        for c in syns:
            if max_rscu > 0:
                w[c] = max(ref_rscu.get(c, 0.0) / max_rscu, 0.5)  # Floor per Sharp & Li 1987
            else:
                w[c] = 1.0

    # CAI = geometric mean of w values for coding codons
    log_sum = 0.0
    n = 0
    for codon in codons:
        if codon in STOP_CODONS or codon not in CODON_TO_AA:
            continue
        aa = CODON_TO_AA[codon]
        if len(CODON_TABLE.get(aa, [])) <= 1:
            continue  # Skip M, W
        if codon in w:
            log_sum += math.log(w[codon])
            n += 1

    if n == 0:
        return 0.0
    return math.exp(log_sum / n)


def _compute_fop(codons: List[str], ref_rscu: Dict[str, float], threshold: float = 1.5) -> float:
    """Frequency of Optimal Codons (Ikemura 1981).

    Optimal codons are defined as those with RSCU > threshold in the
    per-species reference set.  Ikemura's original definition used tRNA
    abundance data; the RSCU > 1.5 threshold is a practical proxy
    widely used in computational implementations (e.g., Sharp & Li 1987,
    Shields et al. 1988).  RSCU > 1.0 means a codon is used more than
    average; > 1.5 means ≥ 50% above expectation.
    """
    optimal = set()
    for aa, syns in CODON_TABLE.items():
        if len(syns) <= 1:
            continue
        for c in syns:
            if ref_rscu.get(c, 0) > threshold:
                optimal.add(c)

    n_optimal = 0
    n_total = 0
    for codon in codons:
        if codon in STOP_CODONS or codon not in CODON_TO_AA:
            continue
        aa = CODON_TO_AA[codon]
        if len(CODON_TABLE.get(aa, [])) <= 1:
            continue
        n_total += 1
        if codon in optimal:
            n_optimal += 1

    return n_optimal / n_total if n_total > 0 else 0.0


def _build_codon_pair_scores(all_codons_list: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """Build genome-wide Codon Pair Scores (CPS) per Coleman et al. 2008.

    CPS(XY) = ln[ f(XY) / expected(XY) ]
    where expected(XY) = f(X) * f(Y) * f(AB) / (f(A) * f(B))
    and A = aa(X), B = aa(Y).

    This measures how much more/less frequently codon pair XY appears
    relative to what's expected from independent codon usage.

    Args:
        all_codons_list: List of codon lists, one per gene in the species.

    Returns:
        Dict mapping (codon1, codon2) -> CPS score.
    """
    # Count genome-wide frequencies
    codon_count: Counter = Counter()
    pair_count: Counter = Counter()
    aa_count: Counter = Counter()
    aa_pair_count: Counter = Counter()

    for codons in all_codons_list:
        for c in codons:
            if c not in STOP_CODONS and c in CODON_TO_AA:
                codon_count[c] += 1
                aa_count[CODON_TO_AA[c]] += 1
        for i in range(len(codons) - 1):
            c1, c2 = codons[i], codons[i + 1]
            if c1 in CODON_TO_AA and c2 in CODON_TO_AA and c1 not in STOP_CODONS and c2 not in STOP_CODONS:
                pair_count[(c1, c2)] += 1
                aa_pair_count[(CODON_TO_AA[c1], CODON_TO_AA[c2])] += 1

    total_codons = sum(codon_count.values())
    total_pairs = sum(pair_count.values())
    if total_codons == 0 or total_pairs == 0:
        return {}

    total_aa = sum(aa_count.values())
    total_aa_pairs = sum(aa_pair_count.values())

    # Compute CPS for each observed codon pair
    cps: Dict[Tuple[str, str], float] = {}
    for (c1, c2), obs in pair_count.items():
        a1, a2 = CODON_TO_AA[c1], CODON_TO_AA[c2]
        f_pair = obs / total_pairs
        f_c1 = codon_count[c1] / total_codons
        f_c2 = codon_count[c2] / total_codons
        f_a1 = aa_count[a1] / total_aa
        f_a2 = aa_count[a2] / total_aa
        f_aa_pair = aa_pair_count[(a1, a2)] / total_aa_pairs
        expected = f_c1 * f_c2 * f_aa_pair / (f_a1 * f_a2) if (f_a1 * f_a2) > 0 else 0
        if expected > 0 and f_pair > 0:
            cps[(c1, c2)] = math.log(f_pair / expected)

    return cps


def _compute_cpb(codons: List[str], cps_table: Dict[Tuple[str, str], float]) -> float:
    """Codon Pair Bias (Coleman et al. 2008, Science 320:1784).

    CPB for a gene = mean CPS over all consecutive codon pairs.
    Positive CPB = gene uses codon pairs more frequently than expected.
    Negative CPB = gene uses disfavored codon pairs.
    Centered near 0 for average genes.

    Args:
        codons: List of codons for this gene.
        cps_table: Species-wide Codon Pair Scores from _build_codon_pair_scores().
    """
    if len(codons) < 2 or not cps_table:
        return np.nan

    cpb_sum = 0.0
    n = 0
    for i in range(len(codons) - 1):
        pair = (codons[i], codons[i + 1])
        if pair in cps_table:
            cpb_sum += cps_table[pair]
            n += 1

    return cpb_sum / n if n > 0 else np.nan


def _compute_codon_ramp_ratio(codons: List[str], ref_rscu: Dict[str, float],
                               ramp_codons: int = 50) -> float:
    """Codon ramp: ratio of CAI for first N codons vs. rest (Tuller 2010).

    A ratio < 1.0 indicates a slow ramp (beneficial for proper folding).
    """
    if len(codons) <= ramp_codons:
        return 1.0

    head = codons[:ramp_codons]
    tail = codons[ramp_codons:]

    cai_head = _compute_cai(head, ref_rscu)
    cai_tail = _compute_cai(tail, ref_rscu)

    if cai_tail > 0:
        return cai_head / cai_tail
    return 1.0


def _count_rare_codon_clusters(codons: List[str], ref_rscu: Dict[str, float],
                                 window: int = 10, threshold: float = 0.5) -> int:
    """Count windows with mean relative adaptiveness below threshold.

    Concept from Clarke & Clark 2008 (rare codon clusters affect
    co-translational folding).  Window size of 10 codons and threshold
    of 0.5 relative adaptiveness are practical parameters.
    """
    if len(codons) < window:
        return 0

    # Compute per-codon relative adaptiveness (same w_i as CAI)
    w = {}
    for aa, syns in CODON_TABLE.items():
        max_rscu = max(ref_rscu.get(c, 0.0) for c in syns) if len(syns) > 1 else 1.0
        for c in syns:
            w[c] = ref_rscu.get(c, 0.0) / max_rscu if max_rscu > 0 else 1.0

    # N-containing or ambiguous codons get neutral adaptiveness
    wa = [w.get(c, 1.0) for c in codons]

    clusters = 0
    for i in range(len(wa) - window + 1):
        if np.mean(wa[i:i + window]) < threshold:
            clusters += 1
    return clusters


def compute_operon_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute operon-level structural/positional metadata features.

    These are trivially derivable from columns already in the table but
    capture information no embedding model can infer:
      - Operon total length and gene count
      - Gene position within the operon (ordinal and fractional)
      - Whether the gene is first, last, or in a singleton operon
      - CDS fraction of the operon footprint
      - Upstream and downstream intergenic distances

    Requires columns: gene_id, operon_source, taxid, operon_id,
    full_operon_length, cds_start_in_operon, gene_end_rel, dna_cds_len.
    No external libraries needed.
    """
    log.info("Computing operon structural features for %d genes", len(df))

    group_cols = ["operon_source", "taxid", "operon_id"]

    # Ensure numeric CDS start for sorting
    cds_start = pd.to_numeric(df["cds_start_in_operon"], errors="coerce").fillna(-1).astype(int)

    # Number of genes per operon
    operon_num_genes = df.groupby(group_cols, dropna=False)["gene_id"].transform("count")

    # Sort within each operon by position and compute rank
    sort_idx = cds_start.values
    df_sorted = df.assign(_sort_key=sort_idx).sort_values(group_cols + ["_sort_key"])
    position = df_sorted.groupby(group_cols, dropna=False).cumcount() + 1
    # Map back to original index
    position = position.reindex(df.index)

    # Compute upstream and downstream intergenic distances within operon
    # These require knowing the previous/next gene's coordinates
    upstream_dist = pd.Series(np.nan, index=df.index, dtype=float)
    downstream_dist = pd.Series(np.nan, index=df.index, dtype=float)

    cds_end = pd.to_numeric(df["gene_end_rel"], errors="coerce")

    for _, grp_idx in df_sorted.groupby(group_cols, dropna=False).groups.items():
        grp = df_sorted.loc[grp_idx]
        if len(grp) < 2:
            continue
        starts = pd.to_numeric(grp["cds_start_in_operon"], errors="coerce").values
        ends = pd.to_numeric(grp["gene_end_rel"], errors="coerce").values
        idx_list = grp.index.tolist()
        for j in range(len(grp)):
            if j > 0:
                # Distance from previous gene's end to this gene's start
                upstream_dist.loc[idx_list[j]] = starts[j] - ends[j - 1]
            if j < len(grp) - 1:
                # Distance from this gene's end to next gene's start
                downstream_dist.loc[idx_list[j]] = starts[j + 1] - ends[j]

    operon_len = pd.to_numeric(df["full_operon_length"], errors="coerce")
    cds_len = pd.to_numeric(df.get("dna_cds_len", pd.Series(dtype=float)), errors="coerce")
    # Fallback: compute CDS length from coordinates if column missing
    if cds_len.isna().all():
        cds_len = cds_end - cds_start
        cds_len = cds_len.clip(lower=0)

    result = pd.DataFrame({
        "gene_id": df["gene_id"],
        "operon_length_nt": operon_len,
        "operon_num_genes": operon_num_genes.astype(int),
        "gene_position_in_operon": position.astype(int),
        "gene_relative_position": np.where(
            operon_num_genes > 1,
            (position - 1) / (operon_num_genes - 1),  # 0.0 = first, 1.0 = last
            0.5,  # singleton gets 0.5
        ),
        "is_first_gene": (position == 1).astype(int),
        "is_last_gene": (position == operon_num_genes).astype(int),
        "is_singleton": (operon_num_genes == 1).astype(int),
        "cds_fraction_of_operon": np.where(
            operon_len > 0, cds_len / operon_len, np.nan,
        ),
        "upstream_intergenic_dist_nt": upstream_dist.fillna(0),
        "downstream_intergenic_dist_nt": downstream_dist.fillna(0),
    })

    log.info("  Operon size distribution: min=%d, median=%d, max=%d",
             result["operon_num_genes"].min(),
             int(result["operon_num_genes"].median()),
             result["operon_num_genes"].max())
    log.info("  Singletons: %d (%.1f%%)",
             result["is_singleton"].sum(),
             100 * result["is_singleton"].mean())
    log.info("  Genes with upstream neighbor: %d",
             result["upstream_intergenic_dist_nt"].notna().sum())

    return result


def compute_codon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute host-specific codon adaptation features for all genes.

    Groups by species, builds per-species reference RSCU from all CDS
    in that species, then scores each gene against its host's reference.
    """
    log.info("Computing codon features for %d genes across %d species",
             len(df), df["species"].nunique())

    results = []

    for species, group in df.groupby("species"):
        log.info("  %s: %d genes", species, len(group))

        # Build per-species reference RSCU from all CDS in this species
        all_codons_species = []
        gene_codons = {}
        all_codons_lists = []  # For CPS computation

        for _, row in group.iterrows():
            cds = row["dna_cds_seq"]
            if pd.isna(cds) or len(cds) < 30:
                gene_codons[row["gene_id"]] = []
                continue
            codons = _extract_codons(cds)
            gene_codons[row["gene_id"]] = codons
            all_codons_species.extend(codons)
            all_codons_lists.append(codons)

        ref_rscu = _compute_rscu(all_codons_species)

        # Build species-wide Codon Pair Scores (Coleman et al. 2008)
        cps_table = _build_codon_pair_scores(all_codons_lists)

        # Score each gene
        for _, row in group.iterrows():
            gid = row["gene_id"]
            codons = gene_codons.get(gid, [])
            cds = row["dna_cds_seq"] if not pd.isna(row.get("dna_cds_seq")) else ""

            if not codons:
                actual_len = len(cds) if cds else 0
                results.append({"gene_id": gid, "cai": np.nan, "enc": np.nan,
                                "fop": np.nan, "cpb": np.nan, "codon_ramp_ratio": np.nan,
                                "gc1": np.nan, "gc2": np.nan, "gc3": np.nan,
                                "gc_cds": np.nan, "cds_length_nt": actual_len,
                                "rare_codon_clusters": 0})
                continue

            gc1, gc2, gc3 = _gc_at_positions(codons)
            gc_cds = sum(1 for c in cds.upper() if c in "GC") / len(cds) if cds else 0.0

            results.append({
                "gene_id": gid,
                "cai": _compute_cai(codons, ref_rscu),
                "enc": _compute_enc_wright(codons),
                "fop": _compute_fop(codons, ref_rscu),
                "cpb": _compute_cpb(codons, cps_table),
                "codon_ramp_ratio": _compute_codon_ramp_ratio(codons, ref_rscu),
                "gc1": gc1,
                "gc2": gc2,
                "gc3": gc3,
                "gc_cds": gc_cds,
                "cds_length_nt": len(cds),
                "rare_codon_clusters": _count_rare_codon_clusters(codons, ref_rscu),
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: RNA THERMODYNAMIC FEATURES (extended)
# ══════════════════════════════════════════════════════════════════════════════

# E. coli 16S rRNA anti-SD sequence for thermodynamic duplex calculation
ANTI_SD_SEQ = "ACCUCCUUA"  # 3'→5' of 16S rRNA tail

# SD regex motifs — heuristic scoring by substring length/strength.
# These weights are NOT from a specific publication; they are a practical
# heuristic for ordering SD motif variants by expected binding strength.
# The full-consensus AGGAGG (Shine & Dalgarno 1974) scores 1.0; partial
# matches score proportionally lower.  For rigorous thermodynamic SD
# binding energy, use sd_binding_dg (ViennaRNA duplexfold) instead.
SD_MOTIFS = [
    ("AGGAGG", 1.0), ("GGAGG", 0.85), ("AGGAG", 0.85),
    ("GAGG", 0.6), ("AGGA", 0.6), ("GGAG", 0.6),
    ("GGA", 0.3), ("GAG", 0.3), ("AGG", 0.3),
]

# m6A DRACH consensus: [D=AGU][R=AG]AC[H=ACU] (in DNA: [AGT][AG]AC[ACT])
DRACH_PATTERN = re.compile(r"[AGT][AG]AC[ACT]", re.IGNORECASE)

# Dam methylation
DAM_PATTERN = re.compile(r"GATC", re.IGNORECASE)

# CpG dinucleotide
CPG_PATTERN = re.compile(r"CG", re.IGNORECASE)


def _compute_mfe(seq: str) -> float:
    """MFE via ViennaRNA, NaN if unavailable."""
    if not HAS_VIENNA:
        return np.nan
    import RNA
    seq_clean = re.sub(r"[^AUGCaugc]", "N", seq.upper().replace("T", "U"))
    try:
        _, mfe = RNA.fold(seq_clean)
        return float(mfe)
    except Exception:
        return np.nan


def _compute_ensemble_energy(seq: str) -> Tuple[float, float]:
    """Ensemble free energy and ensemble diversity via ViennaRNA partition function.

    Returns (ensemble_energy, ensemble_diversity).
    """
    if not HAS_VIENNA:
        return np.nan, np.nan
    import RNA
    seq_clean = re.sub(r"[^AUGCaugc]", "N", seq.upper().replace("T", "U"))
    try:
        fc = RNA.fold_compound(seq_clean)
        _, pf_energy = fc.pf()
        centroid_struct, centroid_dist = fc.centroid()
        return float(pf_energy), float(centroid_dist)
    except Exception:
        return np.nan, np.nan


def _compute_accessibility(seq: str, start: int, end: int) -> float:
    """Unpaired probability for positions [start:end] via partition function."""
    if not HAS_VIENNA:
        return np.nan
    import RNA
    seq_clean = re.sub(r"[^AUGCaugc]", "N", seq.upper().replace("T", "U"))
    try:
        fc = RNA.fold_compound(seq_clean)
        fc.pf()
        bpp = fc.bpp()
        seq_len = len(seq_clean)
        unpaired_probs = []
        for pos in range(max(1, start + 1), min(seq_len, end + 1)):
            paired_prob = 0.0
            for j in range(1, seq_len + 1):
                if j != pos:
                    i_idx, j_idx = min(pos, j), max(pos, j)
                    if i_idx < len(bpp) and j_idx < len(bpp[i_idx]):
                        paired_prob += bpp[i_idx][j_idx]
            unpaired_probs.append(1.0 - min(paired_prob, 1.0))
        return float(np.mean(unpaired_probs)) if unpaired_probs else np.nan
    except Exception:
        return np.nan


def _compute_sd_duplex_dg(upstream_seq: str) -> float:
    """Thermodynamic SD:anti-SD binding energy via ViennaRNA duplexfold.

    More accurate than regex-based SD scoring — captures full base-pairing
    thermodynamics between the mRNA upstream region and 16S rRNA tail.
    Returns ΔG in kcal/mol (more negative = stronger SD).
    """
    if not HAS_VIENNA:
        return np.nan
    import RNA
    rna_upstream = upstream_seq.upper().replace("T", "U")
    rna_upstream = re.sub(r"[^AUGC]", "N", rna_upstream)
    if len(rna_upstream) < 3:
        return np.nan
    try:
        result = RNA.duplexfold(rna_upstream, ANTI_SD_SEQ)
        return float(result.energy)
    except Exception:
        return np.nan


def _find_best_sd(upstream_seq: str) -> Tuple[float, int, str]:
    """Find strongest SD motif (regex-based, from existing pipeline)."""
    seq = upstream_seq.upper()
    best_score, best_spacing, best_motif = 0.0, -1, ""
    for motif, weight in SD_MOTIFS:
        pos = seq.rfind(motif)
        if pos >= 0:
            spacing = len(seq) - (pos + len(motif))
            if weight > best_score:
                best_score, best_spacing, best_motif = weight, spacing, motif
    return best_score, best_spacing, best_motif


def _sd_spacing_penalty(spacing: int) -> float:
    """Exponential penalty for non-optimal SD spacing (5-10 nt optimal).

    Optimal SD-to-start spacing is 5-10 nt (Chen et al. 1994, NAR 22:4953).
    Penalty decays exponentially outside this range.  The decay constant of
    0.3 nt⁻¹ is a heuristic inspired by the Salis RBS Calculator
    (Salis et al. 2009, Nature Biotechnology) which uses a Gaussian spacing
    penalty.  For full thermodynamic analysis, use sd_binding_dg instead.
    """
    if spacing < 0:
        return 0.0
    if 5 <= spacing <= 10:
        return 1.0
    delta = (5 - spacing) if spacing < 5 else (spacing - 10)
    return math.exp(-0.3 * delta)


def _drach_density(seq: str) -> float:
    """DRACH (m6A methylation consensus) motif density."""
    if not seq:
        return 0.0
    n_drach = len(DRACH_PATTERN.findall(seq))
    return n_drach / (len(seq) / 1000)  # Per kb


def _cpg_observed_expected(seq: str) -> float:
    """CpG observed/expected ratio."""
    seq = seq.upper()
    n = len(seq)
    if n < 2:
        return 0.0
    n_c = seq.count("C")
    n_g = seq.count("G")
    n_cg = len(CPG_PATTERN.findall(seq))
    expected = (n_c * n_g) / n if n > 0 else 0
    return (n_cg * n) / (n_c * n_g) if (n_c * n_g) > 0 else 0.0


def _dam_density(seq: str) -> float:
    """Dam methylation (GATC) motif density per kb."""
    if not seq:
        return 0.0
    return len(DAM_PATTERN.findall(seq)) / (len(seq) / 1000)


def compute_rna_thermo_features(df: pd.DataFrame, use_vienna: bool = True) -> pd.DataFrame:
    """Compute extended RNA thermodynamic features from initiation windows.

    Reads directly from the production table's rna_init_window_seq column.
    Recomputes everything the old featurize_biophysical.py did, PLUS:
    - SD binding ΔG (thermodynamic)
    - Ensemble free energy
    - Ensemble diversity (centroid distance)
    - DRACH motif density on CDS
    - CpG O/E on CDS
    - Dam methylation density
    """
    log.info("Computing RNA thermodynamic features for %d genes (ViennaRNA=%s)",
             len(df), HAS_VIENNA and use_vienna)

    results = []
    n_ok = 0
    n_fail = 0
    n_total = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 5000 == 0:
            log.info("  ...%d/%d init windows (vienna_ok=%d)", i + 1, n_total, n_ok)
        gid = row["gene_id"]
        init_seq = row.get("rna_init_window_seq", "")
        cds_dna = row.get("dna_cds_seq", "")
        cds_start = row.get("cds_start_in_operon", 0)
        win_start = row.get("rna_init_window_start", 0)

        if pd.isna(init_seq) or not init_seq:
            results.append({"gene_id": gid})
            continue

        init_seq = str(init_seq)
        seq_len = len(init_seq)

        # Position of CDS start within this window
        cds_offset = cds_start - win_start if not pd.isna(cds_start) and not pd.isna(win_start) else seq_len // 2
        cds_offset = max(0, min(seq_len, cds_offset))

        upstream = init_seq[:cds_offset] if cds_offset > 0 else ""
        downstream = init_seq[cds_offset:] if cds_offset < seq_len else ""

        feat: Dict[str, Any] = {"gene_id": gid}

        # ── Sequence features (no external deps) ─────────────────────────
        gc_all = sum(1 for c in init_seq.upper() if c in "GC") / seq_len if seq_len else 0
        gc_up = sum(1 for c in upstream.upper() if c in "GC") / len(upstream) if upstream else 0
        gc_dn = sum(1 for c in downstream.upper() if c in "GC") / len(downstream) if downstream else 0
        au_early = sum(1 for c in downstream[:30].upper() if c in "AUT") / min(30, len(downstream)) if downstream else 0

        feat["gc_content"] = gc_all
        feat["gc_content_upstream"] = gc_up
        feat["gc_content_downstream"] = gc_dn
        feat["au_richness_init"] = au_early

        # SD motif (regex)
        sd_score, sd_sp, sd_motif = _find_best_sd(upstream)
        feat["sd_score"] = sd_score
        feat["sd_spacing"] = sd_sp
        feat["sd_spacing_penalty"] = _sd_spacing_penalty(sd_sp)

        # ── Structure features (ViennaRNA) ────────────────────────────────
        if use_vienna and HAS_VIENNA:
            try:
                feat["mfe_full"] = _compute_mfe(init_seq)

                rbs_start = max(0, cds_offset - 20)
                rbs_end = min(seq_len, cds_offset + 13)
                rbs_region = init_seq[rbs_start:rbs_end]
                feat["mfe_rbs"] = _compute_mfe(rbs_region) if rbs_region else np.nan

                acc_start = max(0, cds_offset - 2)
                acc_end = min(seq_len, cds_offset + 4)
                feat["accessibility_start"] = _compute_accessibility(init_seq, acc_start, acc_end)

                # NEW: thermodynamic SD binding
                feat["sd_binding_dg"] = _compute_sd_duplex_dg(upstream) if upstream else np.nan

                # NEW: ensemble properties
                ens_energy, ens_diversity = _compute_ensemble_energy(init_seq)
                feat["ensemble_energy"] = ens_energy
                feat["ensemble_diversity"] = ens_diversity

                n_ok += 1
            except Exception as e:
                log.debug("ViennaRNA error for %s: %s", gid, e)
                for k in ["mfe_full", "mfe_rbs", "accessibility_start",
                           "sd_binding_dg", "ensemble_energy", "ensemble_diversity"]:
                    feat[k] = np.nan
                n_fail += 1
        else:
            for k in ["mfe_full", "mfe_rbs", "accessibility_start",
                       "sd_binding_dg", "ensemble_energy", "ensemble_diversity"]:
                feat[k] = np.nan

        # ── CDS-level motif features ─────────────────────────────────────
        if not pd.isna(cds_dna) and cds_dna:
            feat["drach_density_cds"] = _drach_density(cds_dna)
            feat["cpg_oe_cds"] = _cpg_observed_expected(cds_dna)
            feat["dam_density_cds"] = _dam_density(cds_dna)
        else:
            feat["drach_density_cds"] = np.nan
            feat["cpg_oe_cds"] = np.nan
            feat["dam_density_cds"] = np.nan

        results.append(feat)

    if use_vienna and HAS_VIENNA:
        log.info("  ViennaRNA: %d succeeded, %d failed", n_ok, n_fail)

    return pd.DataFrame(results)


def compute_rna_junc_features(df: pd.DataFrame, use_vienna: bool = True) -> pd.DataFrame:
    """Compute junction window features (only for genes with junctions)."""
    junc_mask = df["has_rna_junc_window"] == True
    junc_df = df[junc_mask].copy()
    log.info("Computing junction features for %d genes", len(junc_df))

    results = []
    for _, row in junc_df.iterrows():
        gid = row["gene_id"]
        junc_seq = row.get("rna_junc_window_seq", "")
        if pd.isna(junc_seq) or not junc_seq:
            results.append({"gene_id": gid})
            continue

        junc_seq = str(junc_seq)
        feat: Dict[str, Any] = {"gene_id": gid}

        # GC content
        gc = sum(1 for c in junc_seq.upper() if c in "GC") / len(junc_seq) if junc_seq else 0
        feat["junc_gc_content"] = gc

        # Intergenic distance
        feat["junc_intergenic_distance"] = row.get("rna_junc_intergenic_distance", np.nan)

        # Overlap motifs (AUGA, UGAUG)
        seq_upper = junc_seq.upper()
        feat["junc_overlap_motif"] = int(any(m in seq_upper for m in ["AUGA", "UGAUG"]))

        # SD downstream
        downstream_start = max(0, len(junc_seq) - 50)
        downstream_region = junc_seq[downstream_start:]
        sd_sc, sd_sp, _ = _find_best_sd(downstream_region)
        feat["junc_sd_score_downstream"] = sd_sc
        feat["junc_sd_spacing_downstream"] = sd_sp

        # MFE
        if use_vienna and HAS_VIENNA:
            feat["junc_mfe_full"] = _compute_mfe(junc_seq)
            # SD duplex for downstream gene
            feat["junc_sd_binding_dg"] = _compute_sd_duplex_dg(downstream_region)
        else:
            feat["junc_mfe_full"] = np.nan
            feat["junc_sd_binding_dg"] = np.nan

        results.append(feat)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: PROTEIN PHYSICOCHEMICAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# N-end rule: N-terminal amino acid → half-life class in E. coli
# Based on Tobias et al. 1991 (Science 254:1374) and Varshavsky 1996.
#
# IMPORTANT: In bacteria, nearly all proteins start with Met.  Methionine
# aminopeptidase (MetAP) cleaves the initial Met if the second residue has
# a small side chain (Hirel et al. 1989, PNAS 86:8247).
#
# MetAP-cleavable 2nd residues: G, A, C, S, T, V, P
# If Met is cleaved, the EXPOSED second residue determines half-life.
# If Met is retained (large 2nd residue), the protein is stable (Met = class 1).
#
# Classification of exposed N-terminal residue (Tobias et al. 1991):
#   Class 1 (stable, >600 min): M, G, A, V, S, T, C, I, P
#   Class 2 (secondary destabilizing, ~10 min): D (arginylated by Aat),
#            N (deamidated → D → Aat), Q (deamidated → E, variable in E. coli)
#   Class 3 (primary destabilizing, ~2 min): L, F, W, Y (ClpS/ClpAP),
#            R, K (ClpAP directly)
N_END_RULE_ECOLI = {
    "M": 1, "V": 1, "G": 1, "A": 1, "S": 1, "T": 1, "C": 1, "P": 1,
    "I": 1,  # Stable per Tobias et al. 1991
    "E": 1, "H": 1, "Q": 1,  # Not directly targeted in E. coli
    "D": 2, "N": 2,  # Secondary/tertiary destabilizing via Aat pathway
    "L": 3, "F": 3, "W": 3, "Y": 3,  # Primary type 1 (ClpS → ClpAP)
    "K": 3, "R": 3,  # Primary type 2 (ClpAP directly)
}

# MetAP cleaves N-terminal Met if second residue is small (Hirel et al. 1989)
METAP_CLEAVABLE = set("GACSTVP")

# PEST motif: region between K/R/H that is enriched in P, E, D, S, T
PEST_PATTERN = re.compile(r"[KRH]([^KRH]{12,60})[KRH]")

# Kyte-Doolittle hydropathy
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

HYDROPHOBIC_AAS = set("AILMFVWY")
CHARGED_AAS = set("DEKRH")
AROMATIC_AAS = set("FWY")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _check_pest(region: str) -> bool:
    """Check if a region between basic residues qualifies as PEST.

    Simplified from the original PEST scoring algorithm (Rechsteiner &
    Rogers 1996, Trends Biochem Sci 21:267).  The original uses a weighted
    score based on mole percent of specific residues and a hydrophobic
    correction.  This implementation uses a threshold of >50% P/E/D/S/T
    residues, which captures the essential enrichment criterion without
    the hydrophobic correction term.
    """
    if len(region) < 12:
        return False
    pest_aa = sum(1 for c in region if c in "PEDST")
    return (pest_aa / len(region)) > 0.5


def _count_pest_motifs(seq: str) -> int:
    """Count PEST signal motifs in a protein sequence."""
    count = 0
    for m in PEST_PATTERN.finditer(seq):
        if _check_pest(m.group(1)):
            count += 1
    return count


def _low_complexity_fraction(seq: str, window: int = 20, threshold: float = 0.7) -> float:
    """Fraction of sequence in low-complexity regions (Shannon entropy).

    Inspired by the SEG algorithm (Wootton & Federhen 1993, Comput Chem 17:149).
    A sliding window of 20 AA is scored by Shannon entropy normalized by the
    maximum possible entropy (log₂ of the alphabet size).  Windows with
    normalized entropy < 0.7 are classified as low-complexity.  These
    parameters are practical heuristics, not the exact SEG K1/K2 thresholds.
    """
    if len(seq) < window:
        return 0.0
    lc_positions = 0
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        counts = Counter(w)
        entropy = -sum((c / window) * math.log2(c / window) for c in counts.values() if c > 0)
        max_entropy = math.log2(min(20, window))
        if max_entropy > 0 and (entropy / max_entropy) < threshold:
            lc_positions += 1
    return lc_positions / (len(seq) - window + 1)


def _count_hydrophobic_patches(seq: str, min_len: int = 5) -> Tuple[int, float]:
    """Count consecutive hydrophobic stretches >= min_len.

    Returns (n_patches, fraction_in_patches).
    """
    patches = 0
    patch_residues = 0
    current = 0
    for c in seq:
        if c in HYDROPHOBIC_AAS:
            current += 1
        else:
            if current >= min_len:
                patches += 1
                patch_residues += current
            current = 0
    if current >= min_len:
        patches += 1
        patch_residues += current
    return patches, patch_residues / len(seq) if seq else 0.0


def compute_protein_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute protein physicochemical features from protein_sequence_paxdb."""
    log.info("Computing protein features for %d genes", len(df))

    results = []

    for idx, row in df.iterrows():
        gid = row["gene_id"]
        seq = row.get("protein_sequence_paxdb", "")
        if pd.isna(seq) or not seq:
            results.append({"gene_id": gid})
            continue

        seq = str(seq).upper().replace("*", "")
        seq_clean = "".join(c for c in seq if c in STANDARD_AA)
        L = len(seq_clean)

        if L == 0:
            results.append({"gene_id": gid})
            continue

        feat: Dict[str, Any] = {"gene_id": gid}

        # ── Core physicochemical (peptides library if available) ──────────
        if HAS_PEPTIDES:
            from peptides import Peptide
            try:
                pep = Peptide(seq_clean)
                feat["pI"] = pep.isoelectric_point()
                feat["mw"] = pep.molecular_weight()
                feat["gravy"] = pep.hydrophobicity("KyteDoolittle")
                feat["instability_index"] = pep.instability_index()
                feat["aliphatic_index"] = pep.aliphatic_index()
                feat["aromaticity"] = sum(1 for c in seq_clean if c in AROMATIC_AAS) / L
                feat["charge_ph7"] = pep.charge(pH=7.0)
                feat["boman_index"] = pep.boman()
            except Exception:
                # Fallback below
                feat.update(_manual_physicochemical(seq_clean))
        else:
            feat.update(_manual_physicochemical(seq_clean))

        # ── Composition ──────────────────────────────────────────────────
        feat["protein_length"] = L
        feat["log_protein_length"] = math.log(L)
        feat["cysteine_fraction"] = seq_clean.count("C") / L
        feat["hydrophobic_fraction"] = sum(1 for c in seq_clean if c in HYDROPHOBIC_AAS) / L
        feat["charged_fraction"] = sum(1 for c in seq_clean if c in CHARGED_AAS) / L
        # NOTE: aromatic_fraction removed — identical to aromaticity (both = |F,W,Y| / L)
        feat["polar_fraction"] = sum(1 for c in seq_clean if c in "STNQCY") / L
        feat["tiny_fraction"] = sum(1 for c in seq_clean if c in "AGS") / L
        feat["proline_fraction"] = seq_clean.count("P") / L

        # ── MetAP processing and N-terminal identity ──────────────────
        # Nearly all bacterial proteins start with Met (confirmed: 100% in
        # our dataset).  Methionine aminopeptidase (MetAP) cleaves the
        # initial Met if the 2nd residue has a small side chain
        # (Hirel et al. 1989, PNAS 86:8247; Frottin et al. 2006, MCP 5:2085).
        #
        # MetAP cleavage affects protein half-life, folding, and N-terminal
        # acetylation.  ~47% of E. coli proteins are predicted MetAP-cleaved.
        #
        # NOTE: The classical N-end rule half-life classification (Tobias
        # et al. 1991) produces ZERO VARIANCE for native bacterial proteins:
        # MetAP only cleaves when the 2nd residue is small, and all small
        # residues are in the "stabilizing" class.  When Met is retained,
        # Met itself is also stabilizing.  So the N-end rule class is
        # always 1 (stable) for native proteins — uninformative for ML.
        # We therefore report metap_cleaved (binary) which HAS variance.
        if L >= 2 and seq_clean[0] == "M" and seq_clean[1] in METAP_CLEAVABLE:
            feat["metap_cleaved"] = 1
        else:
            feat["metap_cleaved"] = 0

        # ── Charge-related ───────────────────────────────────────────────
        n_pos = sum(1 for c in seq_clean if c in "KR")
        n_neg = sum(1 for c in seq_clean if c in "DE")
        # NOTE: net_charge_per_res removed — identical to localCIDER NCPR (= (K+R−D−E)/L).
        # NCPR is the canonical term (Das & Pappu 2013). A manual fallback is in
        # compute_disorder_features for when localCIDER is unavailable.
        feat["abs_charge_per_res"] = abs(n_pos - n_neg) / L
        feat["charge_balance"] = min(n_pos, n_neg) / max(n_pos, n_neg) if max(n_pos, n_neg) > 0 else 1.0

        # ── Stability & aggregation signals ──────────────────────────────
        n_glyco = len(re.findall(r"N[^P][ST]", seq_clean))
        feat["n_glycosylation_motifs"] = n_glyco
        feat["pest_motif_count"] = _count_pest_motifs(seq_clean)
        n_patches, patch_frac = _count_hydrophobic_patches(seq_clean)
        feat["hydrophobic_patch_count"] = n_patches
        feat["hydrophobic_patch_fraction"] = patch_frac
        feat["low_complexity_fraction"] = _low_complexity_fraction(seq_clean)

        results.append(feat)

    return pd.DataFrame(results)


def _manual_physicochemical(seq: str) -> Dict[str, float]:
    """Minimal physicochemical calculations (fallback when `peptides` unavailable).

    WARNING: This fallback can only compute a subset of features exactly.
    Features that require iterative pKa-based calculations (pI, charge, Boman)
    or published dipeptide weight matrices (instability index) are returned as
    NaN rather than using ad hoc approximations.

    For production use, install the `peptides` package (Apache-2.0 license).
    """
    L = len(seq)
    if L == 0:
        return {"pI": np.nan, "mw": np.nan, "gravy": np.nan, "instability_index": np.nan,
                "aliphatic_index": np.nan, "aromaticity": 0.0, "charge_ph7": np.nan,
                "boman_index": np.nan}

    # GRAVY: mean Kyte-Doolittle hydropathy — exact calculation (Kyte & Doolittle 1982)
    gravy = sum(KD_SCALE.get(c, 0) for c in seq) / L

    # MW: sum of residue weights minus water (standard biochemistry)
    MW_TABLE = {
        "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15,
        "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
        "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
        "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
    }
    mw = sum(MW_TABLE.get(c, 110) for c in seq) - (L - 1) * 18.015

    # Aliphatic index: exact formula (Ikai 1980)
    ai = 100 * (seq.count("A") / L + 2.9 * seq.count("V") / L +
                3.9 * (seq.count("I") / L + seq.count("L") / L))

    return {
        "pI": np.nan,                # Requires Henderson-Hasselbalch iteration — cannot approximate
        "mw": mw,                    # Exact
        "gravy": gravy,              # Exact (Kyte & Doolittle 1982)
        "instability_index": np.nan, # Requires Guruprasad DIWV matrix — cannot approximate
        "aliphatic_index": ai,       # Exact (Ikai 1980)
        "aromaticity": sum(1 for c in seq if c in "FWY") / L,  # Exact
        "charge_ph7": np.nan,        # Requires pKa-based calculation — cannot approximate
        "boman_index": np.nan,       # Requires Boman (2003) solvation scale — cannot approximate
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: DISORDER & CHARGE PATTERNING
# ══════════════════════════════════════════════════════════════════════════════

def compute_disorder_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intrinsic disorder features using metapredict + localCIDER.

    Performance notes:
      - metapredict: ~12ms/protein (fast)
      - localCIDER FCR/NCPR: <1ms (instant)
      - localCIDER kappa: O(n²), ~100ms for 300aa, ~1s for 1000aa
      - localCIDER omega: O(n²), ~750ms for 300aa — SKIPPED (too slow, low value)
      - For kappa: cap at 1000 residues to keep total time reasonable
    """
    log.info("Computing disorder & charge patterning for %d genes", len(df))

    # Import once outside the loop
    meta_mod = None
    if HAS_METAPREDICT:
        import metapredict as meta_mod

    sp_cls = None
    if HAS_LOCALCIDER:
        from localcider.sequenceParameters import SequenceParameters as sp_cls

    results = []
    t_meta = 0.0
    t_cider = 0.0
    KAPPA_MAX_LEN = 1000  # Skip kappa for very long proteins (O(n²))

    for idx, row in df.iterrows():
        gid = row["gene_id"]
        seq = row.get("protein_sequence_paxdb", "")
        if pd.isna(seq) or not seq:
            results.append({"gene_id": gid})
            continue

        seq = str(seq).upper().replace("*", "")
        seq_clean = "".join(c for c in seq if c in STANDARD_AA)
        L = len(seq_clean)
        if L < 5:
            results.append({"gene_id": gid})
            continue

        feat: Dict[str, Any] = {"gene_id": gid}

        # ── metapredict disorder ─────────────────────────────────────────
        if meta_mod is not None:
            try:
                _t0 = time.time()
                scores = meta_mod.predict_disorder(seq_clean)
                t_meta += time.time() - _t0
                scores_arr = np.array(scores)
                feat["disorder_mean"] = float(np.mean(scores_arr))
                feat["disorder_fraction"] = float(np.mean(scores_arr > 0.5))
                # Longest IDR
                disordered = scores_arr > 0.5
                max_idr = 0
                current = 0
                for d in disordered:
                    if d:
                        current += 1
                        max_idr = max(max_idr, current)
                    else:
                        current = 0
                feat["longest_idr"] = max_idr
                # N-term and C-term disorder (first/last 30 residues)
                n_term = min(30, L)
                feat["disorder_nterm"] = float(np.mean(scores_arr[:n_term]))
                feat["disorder_cterm"] = float(np.mean(scores_arr[-n_term:]))
            except Exception:
                for k in ["disorder_mean", "disorder_fraction", "longest_idr",
                           "disorder_nterm", "disorder_cterm"]:
                    feat[k] = np.nan
        else:
            for k in ["disorder_mean", "disorder_fraction", "longest_idr",
                       "disorder_nterm", "disorder_cterm"]:
                feat[k] = np.nan

        # ── localCIDER charge patterning ─────────────────────────────────
        if sp_cls is not None:
            try:
                _t0 = time.time()
                sp = sp_cls(seq_clean)
                feat["fcr"] = sp.get_FCR()     # Fraction Charged Residues
                feat["ncpr"] = sp.get_NCPR()    # Net Charge Per Residue
                # NOTE: mean_hydropathy removed — it equals gravy + 4.5 exactly
                # (both are mean Kyte-Doolittle, just on different scales).
                # kappa: charge patterning (O(n²) — cap length)
                if L <= KAPPA_MAX_LEN:
                    try:
                        kappa_val = sp.get_kappa()
                        # localCIDER returns -1 sentinel for sequences with
                        # <5 charged residues (kappa undefined per Das & Pappu 2013).
                        feat["kappa"] = kappa_val if kappa_val >= 0 else np.nan
                    except Exception:
                        feat["kappa"] = np.nan
                else:
                    feat["kappa"] = np.nan
                t_cider += time.time() - _t0
                # omega skipped — O(n²) and too slow for 39k proteins
                # at median length 299aa. Proline patterning is low-value
                # for expression prediction.
            except Exception:
                # Manual fallback for FCR/NCPR when localCIDER fails
                n_pos_d = sum(1 for c in seq_clean if c in "KR")
                n_neg_d = sum(1 for c in seq_clean if c in "DE")
                feat["fcr"] = (n_pos_d + n_neg_d) / L
                feat["ncpr"] = (n_pos_d - n_neg_d) / L
                feat["kappa"] = np.nan
        else:
            # Manual fallback when localCIDER not installed
            n_pos_d = sum(1 for c in seq_clean if c in "KR")
            n_neg_d = sum(1 for c in seq_clean if c in "DE")
            feat["fcr"] = (n_pos_d + n_neg_d) / L
            feat["ncpr"] = (n_pos_d - n_neg_d) / L
            feat["kappa"] = np.nan

        results.append(feat)

        if (len(results) % 5000) == 0:
            log.info("  ...%d/%d genes (metapredict=%.0fs, localCIDER=%.0fs)",
                     len(results), len(df), t_meta, t_cider)

    log.info("  Timing: metapredict=%.1fs, localCIDER=%.1fs", t_meta, t_cider)
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# LICENSE REPORT
# ══════════════════════════════════════════════════════════════════════════════

LICENSE_REPORT = """
=== FEATURE LICENSE REPORT ===

INCLUDED (all commercially free):
  ViennaRNA         — LGPL-2.1 (free for commercial use via Python bindings)
  codon-bias        — MIT
  peptides          — Apache 2.0
  localCIDER        — MIT
  metapredict       — MIT
  Biopython         — Biopython License (BSD-like, free for commercial)
  All manual impls  — Internal (no dependency)

SKIPPED (academic/restrictive license — candidates for open-source model):
  TANGO             — Academic-only — predicts β-aggregation propensity
  Zyggregator       — Academic-only — aggregation prediction
  IUPred2A          — Academic-only — disorder (we use metapredict instead)
  NetSurfP 3.0      — Academic-only — surface accessibility
  SignalP 6.0       — Academic-only — signal peptide detection
  RBS Calculator v2 — Academic-only — full thermodynamic translation initiation model
  CamSol            — Free for academic; requires license for commercial
  DisEMBL           — Academic-only — disorder
  FoldX             — Academic-only — stability ΔΔG
  Rosetta           — Commercial license EXPIRED (March 2026); disabled in codebase

POSSIBLE FUTURE ADDITIONS (commercial-OK):
  pyhmmer           — BSD-3 — Pfam domain search (useful but Bacformer covers)
  forgi             — GPL-3 — RNA structure decomposition (copyleft concern)
  seqfold           — MIT — pure-Python RNA folding (backup for ViennaRNA)
  modlAMP           — BSD-3 — peptide descriptors (26+ scales)
  iFeature          — GPL-3 — 53 protein descriptors (copyleft concern)
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute comprehensive classical features for ProtEx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prod-table", type=Path, default=PROD_TABLE,
                        help="Path to production parquet table")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for feature parquets")
    parser.add_argument("--skip-vienna", action="store_true",
                        help="Skip ViennaRNA thermodynamic features")
    parser.add_argument("--skip-disorder", action="store_true",
                        help="Skip metapredict disorder (can be slow)")
    parser.add_argument("--codon-only", action="store_true",
                        help="Only compute codon features")
    parser.add_argument("--protein-only", action="store_true",
                        help="Only compute protein features")
    parser.add_argument("--rna-only", action="store_true",
                        help="Only compute RNA thermo features")
    parser.add_argument("--license-report", action="store_true",
                        help="Print license report and exit")
    args = parser.parse_args()

    if args.license_report:
        print(LICENSE_REPORT)
        return

    # ── Load production table ─────────────────────────────────────────────
    prod_path = args.prod_table if args.prod_table.is_absolute() else PROJECT_ROOT / args.prod_table
    log.info("Loading production table: %s", prod_path)
    df = pd.read_parquet(prod_path)
    log.info("Loaded %d genes, %d species", len(df), df["species"].nunique())

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    use_vienna = not args.skip_vienna
    compute_all = not (args.codon_only or args.protein_only or args.rna_only)

    t0 = time.time()
    summary: Dict[str, Any] = {
        "n_genes": len(df),
        "n_species": df["species"].nunique(),
        "species_list": sorted(df["species"].unique().tolist()),
        "vienna_available": HAS_VIENNA and use_vienna,
        "codonbias_available": HAS_CODONBIAS,
        "peptides_available": HAS_PEPTIDES,
        "localcider_available": HAS_LOCALCIDER,
        "metapredict_available": HAS_METAPREDICT,
    }

    all_features = df[["gene_id"]].copy()

    # ── 0. Operon structural features ──────────────────────────────────────
    if compute_all:
        log.info("=" * 70)
        log.info("PHASE 0: Operon structural / positional features")
        log.info("=" * 70)
        t0a = time.time()
        struct_df = compute_operon_structural_features(df)
        struct_path = output_dir / "classical_operon_structural_features.parquet"
        struct_df.to_parquet(struct_path, index=False)
        log.info("Saved %d rows × %d cols → %s (%.1fs)",
                 len(struct_df), len(struct_df.columns) - 1, struct_path.name, time.time() - t0a)
        _log_feature_stats(struct_df)
        summary["operon_structural"] = {
            "n_features": len(struct_df.columns) - 1,
            "features": [c for c in struct_df.columns if c != "gene_id"],
            "elapsed_s": round(time.time() - t0a, 1),
        }
        all_features = all_features.merge(struct_df, on="gene_id", how="left")

    # ── 1. Codon features ─────────────────────────────────────────────────
    if compute_all or args.codon_only:
        log.info("=" * 70)
        log.info("PHASE 1: Host-specific codon adaptation features")
        log.info("=" * 70)
        t1 = time.time()
        codon_df = compute_codon_features(df)
        codon_path = output_dir / "classical_codon_features.parquet"
        codon_df.to_parquet(codon_path, index=False)
        log.info("Saved %d rows × %d cols → %s (%.1fs)",
                 len(codon_df), len(codon_df.columns) - 1, codon_path.name, time.time() - t1)
        _log_feature_stats(codon_df)
        summary["codon"] = {
            "n_features": len(codon_df.columns) - 1,
            "features": [c for c in codon_df.columns if c != "gene_id"],
            "elapsed_s": round(time.time() - t1, 1),
        }
        all_features = all_features.merge(codon_df, on="gene_id", how="left")

    # ── 2. RNA thermo features ────────────────────────────────────────────
    if compute_all or args.rna_only:
        log.info("=" * 70)
        log.info("PHASE 2: Extended RNA thermodynamic features (init windows)")
        log.info("=" * 70)
        t2 = time.time()
        rna_df = compute_rna_thermo_features(df, use_vienna=use_vienna)
        rna_path = output_dir / "classical_rna_thermo_features.parquet"
        rna_df.to_parquet(rna_path, index=False)
        log.info("Saved %d rows × %d cols → %s (%.1fs)",
                 len(rna_df), len(rna_df.columns) - 1, rna_path.name, time.time() - t2)
        _log_feature_stats(rna_df)
        summary["rna_init"] = {
            "n_features": len(rna_df.columns) - 1,
            "features": [c for c in rna_df.columns if c != "gene_id"],
            "elapsed_s": round(time.time() - t2, 1),
        }
        all_features = all_features.merge(rna_df, on="gene_id", how="left")

        # Junction features
        log.info("=" * 70)
        log.info("PHASE 2b: Junction window features")
        log.info("=" * 70)
        t2b = time.time()
        junc_df = compute_rna_junc_features(df, use_vienna=use_vienna)
        junc_path = output_dir / "classical_rna_junc_features.parquet"
        junc_df.to_parquet(junc_path, index=False)
        log.info("Saved %d rows × %d cols → %s (%.1fs)",
                 len(junc_df), len(junc_df.columns) - 1, junc_path.name, time.time() - t2b)
        _log_feature_stats(junc_df)
        summary["rna_junc"] = {
            "n_features": len(junc_df.columns) - 1,
            "features": [c for c in junc_df.columns if c != "gene_id"],
            "n_genes_with_junctions": len(junc_df),
            "elapsed_s": round(time.time() - t2b, 1),
        }
        if len(junc_df) > 0 and "gene_id" in junc_df.columns:
            all_features = all_features.merge(junc_df, on="gene_id", how="left")

    # ── 3. Protein features ───────────────────────────────────────────────
    if compute_all or args.protein_only:
        log.info("=" * 70)
        log.info("PHASE 3: Protein physicochemical features")
        log.info("=" * 70)
        t3 = time.time()
        prot_df = compute_protein_features(df)
        prot_path = output_dir / "classical_protein_features.parquet"
        prot_df.to_parquet(prot_path, index=False)
        log.info("Saved %d rows × %d cols → %s (%.1fs)",
                 len(prot_df), len(prot_df.columns) - 1, prot_path.name, time.time() - t3)
        _log_feature_stats(prot_df)
        summary["protein"] = {
            "n_features": len(prot_df.columns) - 1,
            "features": [c for c in prot_df.columns if c != "gene_id"],
            "elapsed_s": round(time.time() - t3, 1),
        }
        all_features = all_features.merge(prot_df, on="gene_id", how="left")

    # ── 4. Disorder & charge patterning ───────────────────────────────────
    if (compute_all or args.protein_only) and not args.skip_disorder:
        log.info("=" * 70)
        log.info("PHASE 4: Disorder & charge patterning")
        log.info("=" * 70)
        t4 = time.time()
        dis_df = compute_disorder_features(df)
        dis_path = output_dir / "classical_disorder_features.parquet"
        dis_df.to_parquet(dis_path, index=False)
        log.info("Saved %d rows × %d cols → %s (%.1fs)",
                 len(dis_df), len(dis_df.columns) - 1, dis_path.name, time.time() - t4)
        _log_feature_stats(dis_df)
        summary["disorder"] = {
            "n_features": len(dis_df.columns) - 1,
            "features": [c for c in dis_df.columns if c != "gene_id"],
            "elapsed_s": round(time.time() - t4, 1),
        }
        all_features = all_features.merge(dis_df, on="gene_id", how="left")

    # ── 5. Combined output ────────────────────────────────────────────────
    combined_path = output_dir / "classical_features_combined.parquet"
    all_features.to_parquet(combined_path, index=False)
    n_feat_cols = len([c for c in all_features.columns if c != "gene_id"])
    log.info("=" * 70)
    log.info("Combined: %d genes × %d features → %s", len(all_features), n_feat_cols, combined_path.name)

    elapsed = time.time() - t0
    summary["total_features"] = n_feat_cols
    summary["elapsed_total_s"] = round(elapsed, 1)
    summary["output_files"] = {
        "combined": str(combined_path),
    }

    # ── Summary ───────────────────────────────────────────────────────────
    summary_path = output_dir / "classical_features.summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 70)
    log.info("DONE in %.1fs — %d features across %d genes", elapsed, n_feat_cols, len(all_features))
    log.info("Summary → %s", summary_path)
    log.info("=" * 70)

    # Quick sanity check
    non_null_frac = all_features.drop(columns=["gene_id"]).notna().mean()
    low_coverage = non_null_frac[non_null_frac < 0.5]
    if len(low_coverage) > 0:
        log.warning("Features with <50%% coverage (expected for junction-only):")
        for col, frac in low_coverage.items():
            log.warning("  %s: %.1f%% non-null", col, frac * 100)


def _log_feature_stats(df: pd.DataFrame) -> None:
    """Log summary statistics for numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        vals = numeric[col].dropna()
        if len(vals) > 0:
            log.info("  %-30s mean=%8.3f  std=%8.3f  [%.3f, %.3f]  nan=%d/%d",
                     col, vals.mean(), vals.std(), vals.min(), vals.max(),
                     numeric[col].isna().sum(), len(df))


if __name__ == "__main__":
    main()
