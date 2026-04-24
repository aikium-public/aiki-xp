"""
Central module for protein sequence normalization.

This is the SINGLE SOURCE OF TRUTH for:
- Expression tag detection and removal
- MetAP (methionine aminopeptidase) cleavage prediction
- Sequence canonicalization

All code that needs to normalize sequences should import from here.
DO NOT reimplement these functions elsewhere.

Key Functions:
    normalize_sequence()       - Full pipeline (tags + MetAP)
    detect_tag()              - Identify expression tags
    strip_tag()               - Remove expression tags
    predict_metap_cleavage()  - Apply MetAP biological rules
    ensure_initiator_m()      - Add M if missing

Usage:
    from protein_filter.sequence_normalization import normalize_sequence
    
    # Full normalization
    normalized, mods = normalize_sequence("MHHHHHHGSKPILOT")
    # normalized = "SKPILOT" (His-tag stripped, M cleaved by MetAP)
    # mods = {'had_tag': True, 'tag_type': 'His', 'metap_status': 'cleaved', ...}

References:
    - MetAP rules: Frottin et al., MCP 2006
    - "The proteomics of N-terminal methionine cleavage"

Author: Aikium Protein Engineering
Created: 2026-01-26
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =============================================================================
# MetAP CLEAVAGE RULES (Frottin et al., MCP 2006)
# =============================================================================
# 
# The initiator Met is cleaved by methionine aminopeptidase (MetAP) based on
# the 2nd (P1') and 3rd (P2') residues. ~1/3 of proteins RETAIN their Met.
#
# | P1' (2nd)           | P2' (3rd)        | Met removed?           |
# |---------------------|------------------|------------------------|
# | G, A, S, C, P       | Any except Pro   | YES (complete)         |
# | V, T                | Not D, E, or P   | Variable (50-95%)      |
# | V, T                | D, E, or P       | NO                     |
# | Any                 | P                | NO (blocks cleavage)   |
# | L,I,F,Y,W,M,K,R,H,D,E,N,Q | Any       | NO                     |
#
# IMPORTANT: Recombinant expression can cause Met retention even when rules
# say cleave (MetAP saturation, metal cofactor depletion).

METAP_DEFINITE_CLEAVE = frozenset('GASCP')  # P1' that leads to complete cleavage
METAP_DEFINITE_RETAIN = frozenset('LIFYWMKRHDNEQ')  # P1' that always retains Met
METAP_VARIABLE = frozenset('VT')  # P1' with variable cleavage (50-95%)
METAP_BLOCKING_P2 = frozenset('DEP')  # P2' that blocks cleavage when P1' is V/T


# =============================================================================
# TAG DETECTION PATTERNS
# =============================================================================

# N-terminal tags
TAG_PATTERNS_N = {
    'His': re.compile(r'^M?H{5,10}'),
    # Broader His-tag: 6+ consecutive H anywhere in the first 20 residues,
    # followed by a known protease cleavage site. Catches common constructs like
    # pET-His6 (MRGSHHHHHHTDPALRA), SUMO-His, and other vector-derived N-terminal
    # His-tags with short leader sequences. Requires both 6+ H AND a cleavage site
    # to avoid matching natural His-rich repeats in the training data.
    'His_broad': re.compile(
        r'^.{1,15}H{6,10}'
        r'(TDPALRA|ENLYFQ|LEVLFQ|LVPRGS?|IEGR|DDDDK|SSGLVPRGS?|SSG|EN)'
    ),
    'FLAG': re.compile(r'^M?DYKDDDDK'),
    'Strep': re.compile(r'^M?WSHPQFEK'),
    'Strep_II': re.compile(r'^M?SAWSHPQFEK'),
    'GST_linker': re.compile(r'^M?GPLGS'),
    'MBP_linker': re.compile(r'^M?NSSSN'),
    'SUMO': re.compile(r'^M?MGSSHHHHHHSSGLVPR'),
    'Thioredoxin': re.compile(r'^M?SDKIIHLTDDSFDTDVLK'),
    # HiBit (split-luciferase peptide). Often appears at N-terminus with
    # optional GGGGS linkers on either side. Added 2026-04-13 per
    # Extended tag stripping: covers His-tags, HiBit, FLAG, and common linkers.
    # HiBit and the canonical module previously missed all of them.
    'HiBit': re.compile(r'^M?(SGGGG){0,3}VSGWRLFKKIS(GGGGS){0,3}'),
}

# C-terminal tags
TAG_PATTERNS_C = {
    # Allow up to 5 trailing residues after the H-run: many constructs end
    # with `...HHHHHHLE*` or `...HHHHHHGS` where the trailing 1-5 aa
    # prevent an anchored `H{5,10}$` match. Added 2026-04-13.
    'His_C': re.compile(r'(GGGGS|GS){0,3}H{5,10}[A-Z*]{0,5}$'),
    'FLAG_C': re.compile(r'DYKDDDDK$'),
    'Strep_C': re.compile(r'WSHPQFEK$'),
    'HiBit_C': re.compile(r'(SGGGG){0,3}VSGWRLFKKIS(GGGGS){0,3}[A-Z*]{0,5}$'),
}

# Common linkers (can appear at N or C terminus)
LINKER_PATTERNS = {
    'GGGGS': re.compile(r'(GGGGS){2,}'),
    'GS': re.compile(r'^(GS){2,}|^G{2,}S|^GGG+'),
    'EAAAK': re.compile(r'(EAAAK){2,}'),
}

# Protease cleavage sites (often left after tag removal)
CLEAVAGE_SITE_PATTERNS = {
    'TEV': re.compile(r'^ENLYFQ[GS]?'),
    'PreScission': re.compile(r'^LEVLFQ[GP]?'),
    'Thrombin': re.compile(r'^LVPRGS?'),
    'Thrombin_pET': re.compile(r'^TDPALRA'),  # pET vector thrombin site (e.g. progsol eSOL)
    'Factor_Xa': re.compile(r'^IEGR'),
    'Enterokinase': re.compile(r'^DDDDK'),
}


@dataclass
class MetAPResult:
    """Result of MetAP cleavage prediction."""
    sequence: str
    status: str  # 'cleaved', 'retained', 'cleaved_uncertain', 'no_change'
    confidence: float
    p1_prime: str  # 2nd residue
    p2_prime: str  # 3rd residue


@dataclass
class NormalizationResult:
    """Result of full sequence normalization."""
    sequence: str
    original_length: int
    final_length: int
    had_tag: bool
    tag_type: Optional[str]
    tag_position: Optional[str]  # 'N', 'C', or 'both'
    added_m: bool
    metap_status: str
    metap_confidence: float


# =============================================================================
# TAG DETECTION
# =============================================================================

def detect_tag(seq: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Detect expression tag in sequence.
    
    Args:
        seq: Protein sequence
        
    Returns:
        (tag_type, position, match_end)
        - tag_type: Name of detected tag or None
        - position: 'N' (N-terminal), 'C' (C-terminal), or None
        - match_end: End position of match (for N-term) or start (for C-term)
    """
    if not seq:
        return None, None, None
    
    # Check N-terminal tags first (more common)
    for tag_name, pattern in TAG_PATTERNS_N.items():
        match = pattern.match(seq)
        if match:
            return tag_name, 'N', match.end()
    
    # Check C-terminal tags
    for tag_name, pattern in TAG_PATTERNS_C.items():
        match = pattern.search(seq)
        if match:
            return tag_name, 'C', match.start()
    
    return None, None, None


def detect_all_tags(seq: str) -> Dict[str, any]:
    """
    Detect all tags and linkers in sequence.
    
    Returns dict with:
        - n_terminal_tag: (type, end_pos) or None
        - c_terminal_tag: (type, start_pos) or None
        - linkers: list of (type, start, end)
        - cleavage_sites: list of (type, start, end)
    """
    result = {
        'n_terminal_tag': None,
        'c_terminal_tag': None,
        'linkers': [],
        'cleavage_sites': [],
    }
    
    if not seq:
        return result
    
    # N-terminal tags
    for tag_name, pattern in TAG_PATTERNS_N.items():
        match = pattern.match(seq)
        if match:
            result['n_terminal_tag'] = (tag_name, match.end())
            break
    
    # C-terminal tags
    for tag_name, pattern in TAG_PATTERNS_C.items():
        match = pattern.search(seq)
        if match:
            result['c_terminal_tag'] = (tag_name, match.start())
            break
    
    # Linkers
    for linker_name, pattern in LINKER_PATTERNS.items():
        for match in pattern.finditer(seq):
            result['linkers'].append((linker_name, match.start(), match.end()))
    
    # Cleavage sites (usually after N-terminal tag removal)
    for site_name, pattern in CLEAVAGE_SITE_PATTERNS.items():
        match = pattern.match(seq)
        if match:
            result['cleavage_sites'].append((site_name, match.start(), match.end()))
    
    return result


# =============================================================================
# TAG STRIPPING
# =============================================================================

def strip_tag(seq: str, strip_linkers: bool = True) -> str:
    """
    Strip expression tags and optionally linkers from sequence.
    
    Handles:
    - N-terminal His-tags with optional linkers
    - C-terminal His-tags
    - FLAG, Strep tags
    - Common linkers (GGGGS, GS repeats)
    - Protease cleavage site remnants
    
    Args:
        seq: Input sequence
        strip_linkers: Also remove common linkers (default True)
        
    Returns:
        Sequence with tags removed
    """
    if not seq:
        return seq
    
    original = seq
    
    # === N-terminal tag removal ===
    
    # His-tag with optional linker/cleavage site
    # Matches: MHHHHHH, HHHHHHGS, HHHHHHLEVLFQG, etc.
    his_n_pattern = re.compile(
        r'^M?H{5,10}'
        r'(GGGGS|GGSGG|GS{1,3}|GSGS|LEVLFQ[GS]?|ENLYFQ[GS]?|LVPRGS?)?'
    )
    seq = his_n_pattern.sub('', seq)

    # Broad His-tag: 6+ H preceded by short leader (e.g. pET-His6 MRGSHHHHHHTDPALRA).
    # Requires 6+ H AND a known cleavage site immediately after to avoid matching
    # natural His-rich repeats (e.g. THISSCKSLEHHHHHHTHISSCKSLEHH...).
    his_broad_pattern = re.compile(
        r'^.{1,15}H{6,10}'
        r'(TDPALRA|ENLYFQ[GS]?|LEVLFQ[GP]?|LVPRGS?|IEGR|DDDDK|SSGLVPRGS?|SSG|EN)'
    )
    seq = his_broad_pattern.sub('', seq)
    
    # FLAG tag
    flag_pattern = re.compile(r'^M?DYKDDDDK(GS)?')
    seq = flag_pattern.sub('', seq)
    
    # Strep tag
    strep_pattern = re.compile(r'^M?(SA)?WSHPQFEK(GS)?')
    seq = strep_pattern.sub('', seq)
    
    # GST linker
    gst_pattern = re.compile(r'^M?GPLGS')
    seq = gst_pattern.sub('', seq)
    
    # MBP linker
    mbp_pattern = re.compile(r'^M?NSSSN')
    seq = mbp_pattern.sub('', seq)

    # HiBit (N-terminal, with optional GGGGS linker, pre- or post-)
    # Added 2026-04-13. Handles MVSGWRLFKKIS..., GGGGSVSGWRLFKKIS..., etc.
    hibit_n_pattern = re.compile(r'^M?(SGGGG){0,3}VSGWRLFKKIS(GGGGS){0,3}')
    seq = hibit_n_pattern.sub('', seq)

    # === C-terminal tag removal ===

    # His-tag with optional linker and up to 5 trailing residues.
    # Aikium scaffolds frequently end with `...GSHHHHHHLE*` where the
    # trailing 1-5 aa prevent an anchored H{5,10}$ match. The bounded
    # [A-Z*]{0,5}$ permits the common `LE`, `GS`, `*` terminators.
    his_c_pattern = re.compile(r'(GGGGS|GS){0,3}H{5,10}[A-Z*]{0,5}$')
    seq = his_c_pattern.sub('', seq)

    # HiBit at C-terminus (with optional linker + trailing residues)
    hibit_c_pattern = re.compile(
        r'(SGGGG){0,3}VSGWRLFKKIS(GGGGS){0,3}[A-Z*]{0,5}$')
    seq = hibit_c_pattern.sub('', seq)

    # FLAG
    seq = re.sub(r'DYKDDDDK$', '', seq)

    # Strep
    seq = re.sub(r'WSHPQFEK$', '', seq)
    
    # === Linker removal (if enabled) ===
    
    if strip_linkers:
        # N-terminal linkers
        seq = re.sub(r'^(GGGGS){1,3}', '', seq)
        seq = re.sub(r'^(GS){2,4}', '', seq)
        seq = re.sub(r'^GGG+', '', seq)
        
        # C-terminal linkers
        seq = re.sub(r'(GGGGS){1,3}$', '', seq)
        seq = re.sub(r'(GS){2,4}$', '', seq)
    
    # === Cleavage site remnants ===
    
    # These are sometimes left after protease cleavage
    seq = re.sub(r'^ENLYFQ[GS]?', '', seq)  # TEV
    seq = re.sub(r'^LEVLFQ[GP]?', '', seq)  # PreScission
    seq = re.sub(r'^LVPRGS?', '', seq)  # Thrombin
    
    return seq


# =============================================================================
# METAP PREDICTION
# =============================================================================

def predict_metap_cleavage(
    seq: str,
    is_recombinant: bool = True,
) -> MetAPResult:
    """
    Predict MetAP cleavage using Frottin et al. (MCP 2006) rules.
    
    Args:
        seq: Input protein sequence
        is_recombinant: Whether protein was recombinantly expressed
            (recombinant = lower confidence due to MetAP saturation)
    
    Returns:
        MetAPResult with processed sequence and confidence
    
    Reference:
        Frottin et al., "The proteomics of N-terminal methionine cleavage"
        Mol Cell Proteomics, 2006
    """
    if not seq or not seq.startswith('M'):
        return MetAPResult(
            sequence=seq,
            status='no_change',
            confidence=1.0,
            p1_prime='',
            p2_prime='',
        )
    
    if len(seq) < 2:
        return MetAPResult(
            sequence=seq,
            status='retained',
            confidence=1.0,
            p1_prime='',
            p2_prime='',
        )
    
    p1_prime = seq[1].upper()
    p2_prime = seq[2].upper() if len(seq) > 2 else ''
    
    # Rule 1: P2' is Pro - blocks cleavage regardless of P1'
    if p2_prime == 'P':
        return MetAPResult(
            sequence=seq,
            status='retained',
            confidence=0.95,
            p1_prime=p1_prime,
            p2_prime=p2_prime,
        )
    
    # Rule 2: Definite cleavage (G, A, S, C, P at P1')
    if p1_prime in METAP_DEFINITE_CLEAVE:
        # Recombinant expression can cause retention due to MetAP saturation
        confidence = 0.70 if is_recombinant else 0.95
        return MetAPResult(
            sequence=seq[1:],
            status='cleaved',
            confidence=confidence,
            p1_prime=p1_prime,
            p2_prime=p2_prime,
        )
    
    # Rule 3: Definite retention (L, I, F, Y, W, M, K, R, H, D, E, N, Q at P1')
    if p1_prime in METAP_DEFINITE_RETAIN:
        return MetAPResult(
            sequence=seq,
            status='retained',
            confidence=0.90,
            p1_prime=p1_prime,
            p2_prime=p2_prime,
        )
    
    # Rule 4: Variable (V, T at P1')
    if p1_prime in METAP_VARIABLE:
        if p2_prime in METAP_BLOCKING_P2:
            # D, E, or P at P2' blocks cleavage
            return MetAPResult(
                sequence=seq,
                status='retained',
                confidence=0.80,
                p1_prime=p1_prime,
                p2_prime=p2_prime,
            )
        else:
            # Variable - 50-95% cleavage
            confidence = 0.50 if is_recombinant else 0.70
            return MetAPResult(
                sequence=seq[1:],
                status='cleaved_uncertain',
                confidence=confidence,
                p1_prime=p1_prime,
                p2_prime=p2_prime,
            )
    
    # Fallback - unknown P1'
    return MetAPResult(
        sequence=seq,
        status='retained',
        confidence=0.50,
        p1_prime=p1_prime,
        p2_prime=p2_prime,
    )


# =============================================================================
# INITIATOR METHIONINE
# =============================================================================

def ensure_initiator_m(seq: str) -> str:
    """
    Ensure sequence starts with initiator methionine.
    
    Some databases report the mature form (after MetAP processing),
    so we add M back for consistent processing.
    
    Args:
        seq: Input sequence
        
    Returns:
        Sequence with M at start (added if missing)
    """
    if not seq:
        return seq
    
    if not seq.startswith('M'):
        return 'M' + seq
    
    return seq


def normalize_leading_m(seq: str, mode: str = 'metap') -> str:
    """
    Normalize the leading methionine.
    
    Args:
        seq: Input sequence (should start with M)
        mode: 
            - 'metap': Apply MetAP biological rules (recommended)
            - 'always_strip': Always remove leading M
            - 'always_keep': Never remove leading M
            
    Returns:
        Normalized sequence
    """
    if not seq:
        return seq
    
    if mode == 'always_strip':
        return seq[1:] if seq.startswith('M') else seq
    elif mode == 'always_keep':
        return seq
    elif mode == 'metap':
        result = predict_metap_cleavage(seq, is_recombinant=False)
        return result.sequence
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# FULL NORMALIZATION PIPELINE
# =============================================================================

def normalize_sequence(
    seq: str,
    strip_tags: bool = True,
    strip_linkers: bool = True,
    apply_metap: bool = True,
    is_recombinant: bool = False,
    ensure_m: bool = True,
) -> Tuple[str, NormalizationResult]:
    """
    Full sequence normalization pipeline.
    
    Pipeline:
    1. Strip expression tags (if enabled)
    2. Strip linkers (if enabled)
    3. Ensure initiator M is present (if enabled and biologically plausible)
    4. Apply MetAP biological rules (if enabled)
    
    Args:
        seq: Input protein sequence
        strip_tags: Remove expression tags (His, FLAG, etc.)
        strip_linkers: Remove common linkers (GGGGS, etc.)
        apply_metap: Apply MetAP cleavage rules
        is_recombinant: Use recombinant expression rules for MetAP
            (lower confidence due to potential MetAP saturation)
        ensure_m: Add initiator M if missing. Only adds M if biologically
            plausible (i.e., first residue could have been MetAP-cleaved).
            Set to False for clustering where you want exact sequence matching.
    
    Returns:
        (normalized_sequence, NormalizationResult)
    """
    if not seq:
        return seq, NormalizationResult(
            sequence=seq,
            original_length=0,
            final_length=0,
            had_tag=False,
            tag_type=None,
            tag_position=None,
            added_m=False,
            metap_status='no_change',
            metap_confidence=1.0,
        )
    
    original_length = len(seq)
    had_tag = False
    tag_type = None
    tag_position = None
    added_m = False
    metap_status = 'no_change'
    metap_confidence = 1.0
    
    # Step 1: Detect and strip tags
    if strip_tags:
        detected_tag, detected_pos, _ = detect_tag(seq)
        if detected_tag:
            had_tag = True
            tag_type = detected_tag
            tag_position = detected_pos
            seq = strip_tag(seq, strip_linkers=strip_linkers)
    elif strip_linkers:
        # Strip linkers even if not stripping tags
        seq = strip_tag(seq, strip_linkers=True)
    
    # Step 2: Ensure M is present (only if biologically plausible)
    # Only prepend M if the first residue could have been MetAP-cleaved
    # MetAP cleaves when P1' is G, A, S, C, P, V, T - so these could be missing M
    # If P1' is H, D, E, K, R, etc., MetAP wouldn't cleave, so M was never there
    if ensure_m and seq and not seq.startswith('M'):
        first_residue = seq[0]
        # Only add M if first residue is MetAP-cleavable (could have lost its M)
        if first_residue in METAP_DEFINITE_CLEAVE or first_residue in METAP_VARIABLE:
            added_m = True
            seq = 'M' + seq
        # Don't add M if first residue wouldn't have been MetAP-cleaved
        # (sequence likely naturally starts with this residue, or M was never there)
    
    # Step 3: Apply MetAP rules
    if apply_metap and seq:
        result = predict_metap_cleavage(seq, is_recombinant=is_recombinant)
        seq = result.sequence
        metap_status = result.status
        metap_confidence = result.confidence
    
    # Validation: warn if normalization resulted in very short sequence
    # (but not during testing - suppress with PYTHONWARNINGS=ignore)
    if seq and len(seq) < 10 and original_length > 20:
        import warnings
        warnings.warn(
            f"Normalization produced very short sequence ({len(seq)} AA from {original_length} AA). "
            f"Input may have been mostly tag/linker."
        )
    
    return seq, NormalizationResult(
        sequence=seq,
        original_length=original_length,
        final_length=len(seq) if seq else 0,
        had_tag=had_tag,
        tag_type=tag_type,
        tag_position=tag_position,
        added_m=added_m,
        metap_status=metap_status,
        metap_confidence=metap_confidence,
    )


def normalize_sequence_simple(seq: str) -> str:
    """
    Simplified normalization - just return the canonical sequence.
    
    For when you don't need the detailed modifications info.
    
    Args:
        seq: Input sequence
        
    Returns:
        Normalized sequence
    """
    normalized, _ = normalize_sequence(seq)
    return normalized


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def normalize_sequences_batch(
    sequences: list,
    strip_tags: bool = True,
    strip_linkers: bool = True,
    apply_metap: bool = True,
    is_recombinant: bool = False,
) -> Tuple[list, Dict]:
    """
    Normalize a batch of sequences with aggregate statistics.
    
    Args:
        sequences: List of sequences
        strip_tags, strip_linkers, apply_metap, is_recombinant: See normalize_sequence
        
    Returns:
        (normalized_sequences, stats_dict)
    """
    normalized = []
    results = []
    
    for seq in sequences:
        norm_seq, result = normalize_sequence(
            seq,
            strip_tags=strip_tags,
            strip_linkers=strip_linkers,
            apply_metap=apply_metap,
            is_recombinant=is_recombinant,
        )
        normalized.append(norm_seq)
        results.append(result)
    
    # Compute statistics
    n_total = len(sequences)
    n_had_tag = sum(1 for r in results if r.had_tag)
    n_added_m = sum(1 for r in results if r.added_m)
    n_metap_cleaved = sum(1 for r in results if r.metap_status == 'cleaved')
    n_metap_uncertain = sum(1 for r in results if r.metap_status == 'cleaved_uncertain')
    
    # Tag type breakdown
    tag_counts = {}
    for r in results:
        if r.tag_type:
            tag_counts[r.tag_type] = tag_counts.get(r.tag_type, 0) + 1
    
    stats = {
        'total': n_total,
        'had_expression_tag': n_had_tag,
        'had_expression_tag_pct': n_had_tag / n_total * 100 if n_total else 0,
        'added_initiator_m': n_added_m,
        'added_initiator_m_pct': n_added_m / n_total * 100 if n_total else 0,
        'metap_cleaved': n_metap_cleaved,
        'metap_cleaved_pct': n_metap_cleaved / n_total * 100 if n_total else 0,
        'metap_uncertain': n_metap_uncertain,
        'metap_retained': n_total - n_metap_cleaved - n_metap_uncertain,
        'tag_type_counts': tag_counts,
    }
    
    return normalized, stats


# =============================================================================
# TESTS
# =============================================================================

def _test_normalization():
    """Self-test for normalization functions."""
    
    # Test tag detection
    assert detect_tag("MHHHHHHGSKPILOT")[0] == 'His'
    assert detect_tag("MDYKDDDDKPILOT")[0] == 'FLAG'
    assert detect_tag("PILOTHHHHHHH")[0] == 'His_C'
    assert detect_tag("MPILOT")[0] is None
    
    # Test tag stripping
    assert strip_tag("MHHHHHHGSKPILOT") == "KPILOT"
    assert strip_tag("MHHHHHHLEVLFQGPILOT") == "PILOT"
    assert strip_tag("PILOTHHHHHH") == "PILOT"
    assert strip_tag("MPILOT") == "MPILOT"
    
    # Test MetAP
    # G at P1' -> cleaved
    result = predict_metap_cleavage("MGTEST", is_recombinant=False)
    assert result.status == 'cleaved'
    assert result.sequence == "GTEST"
    
    # L at P1' -> retained
    result = predict_metap_cleavage("MLTEST", is_recombinant=False)
    assert result.status == 'retained'
    assert result.sequence == "MLTEST"
    
    # P at P2' -> retained regardless of P1'
    result = predict_metap_cleavage("MGPEST", is_recombinant=False)
    assert result.status == 'retained'
    
    # Test full normalization
    # His-tag + MetAP cleavage
    norm, result = normalize_sequence("MHHHHHHGSKPILOT")
    assert result.had_tag == True
    assert result.tag_type == 'His'
    # After stripping His+GS linker: "KPILOT". ensure_m is selective —
    # it only prepends M when the first residue is MetAP-cleavable
    # (G/A/S/C/P or V/T). K is not cleavable, so M is NOT added.
    # Final: KPILOT.
    assert norm == "KPILOT", f"Expected KPILOT, got {norm}"
    
    # No tag, M retained (L at P1' -> retained)
    norm, result = normalize_sequence("MLTEST")
    assert result.had_tag == False
    assert norm == "MLTEST"
    
    # Missing M, added, then MetAP cleaved (G at P1')
    norm, result = normalize_sequence("GTEST")
    assert result.added_m == True
    # GTEST -> add M -> MGTEST -> MetAP cleaves -> GTEST
    assert norm == "GTEST", f"Expected GTEST, got {norm}"
    
    # Edge case: sequence that is ONLY tag (results in empty - this is expected!)
    norm, result = normalize_sequence("MHHHHHHGS")
    # His-tag + GS linker stripped -> nothing left -> empty
    # This is correct behavior - the input was entirely tag
    assert norm == "", f"Expected empty for tag-only input, got: {norm}"
    assert result.had_tag == True
    
    # Edge case: empty input
    norm, result = normalize_sequence("")
    assert norm == ""
    assert result.original_length == 0

    # === 2026-04-13 additions: HiBit + C-terminal trailing-residue His ===

    # HiBit at N-terminus (plain)
    norm, r = normalize_sequence("MVSGWRLFKKISKPILOT", ensure_m=False, apply_metap=False)
    assert r.had_tag and r.tag_type == 'HiBit', f"expected HiBit tag, got {r.tag_type}"
    assert norm == "KPILOT", f"HiBit N-term strip failed: got {norm}"

    # HiBit with GGGGS linker on both sides at N-terminus
    norm, r = normalize_sequence(
        "MSGGGGVSGWRLFKKISGGGGSKPILOT", ensure_m=False, apply_metap=False)
    assert norm == "KPILOT", f"HiBit+linker N-term strip failed: got {norm}"

    # HiBit at C-terminus with trailing residues
    # First strip with standalone strip_tag to bypass MetAP/ensure_m side effects
    stripped = strip_tag("KPILOTSGGGGVSGWRLFKKISGGGGSLE*")
    assert stripped == "KPILOT", f"HiBit C-term with trailing residues failed: got {stripped}"

    # C-terminal His with trailing LE (the Aikium scaffold pattern)
    stripped = strip_tag("KPILOTGGGGSHHHHHHLE*")
    assert stripped == "KPILOT", f"His C-term with trailing LE failed: got {stripped}"

    # Internal HHHHHH must NOT be stripped (scaffold H-rich body preservation)
    internal = "MIKEEHVIIQAEFYLNPDQSGEFHHHHHHRECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRP"
    stripped = strip_tag(internal)
    assert stripped == internal, \
        f"Internal HHHHHH must not be stripped: input={internal!r}, got={stripped!r}"

    # Idempotence: stripping twice == stripping once
    seq_with_both = "MHHHHHHGSMIKEEHVIIQAEFYLNPDQSGEFSGGGGVSGWRLFKKISGGGGSLE*"
    once = strip_tag(seq_with_both)
    twice = strip_tag(once)
    assert once == twice, f"strip_tag not idempotent: once={once!r}, twice={twice!r}"

    print("All normalization tests passed!")


if __name__ == "__main__":
    _test_normalization()
