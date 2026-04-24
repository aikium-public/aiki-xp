"""Find a CDS in a bacterial genome and extract operon/flanking context.

Given:
  - A BioPython SeqRecord genome (loaded from pickle)
  - A CDS DNA sequence (the user's gene)
  - A mode: "native" (CDS exists in genome) or "heterologous" (synthesize context)

Return:
  - full_operon_dna: polycistronic unit containing the CDS
  - cds_start_in_operon: position of CDS start within the operon
  - upstream_context_seq, downstream_context_seq: 100 nt up / 50 nt down of CDS
  - rna_init_window_seq: 60 nt (35 up + 25 down of ATG) with T→U substitution
  - genome_cds_index: for Bacformer-large lookup (None in heterologous mode)
  - strand: "+" or "-"

Matches the columns of aikixp_492k_v1.parquet for downstream featurization.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Nieuwkoop 2023 asymmetric window — matches init60_extract_base.py
INIT_UPSTREAM = 35
INIT_DOWNSTREAM = 25

# Intergenic distance heuristic for operon boundary detection (matches MicrobesOnline-style calls)
MAX_INTERGENIC_OPERON_DISTANCE = 150  # nt

# Context extraction sizes (match upstream_context_length_nt / downstream_context_length_nt in training)
UPSTREAM_CONTEXT_NT = 100
DOWNSTREAM_CONTEXT_NT = 50

# Contig gap for merged WGS pickles. Much larger than MAX_INTERGENIC_OPERON_DISTANCE
# so operon detection never bridges two contigs, and large enough that no real
# CDS substring will accidentally hit into the gap.
_CONTIG_GAP_LEN = 2000


def load_genome(path: str | Path):
    """Open a pickle and return a SeqRecord-compatible object.

    Legacy WGS merged pickles store a `list[SeqRecord]` (one record per
    contig). The rest of the pipeline expects a single SeqRecord with
    `.seq` and `.features`. When we see a list, we wrap it in a
    :class:`MergedGenome` that presents the whole assembly as one
    concatenated SeqRecord with features re-indexed into the merged
    coordinate space.

    Set ``AIKIXP_MERGE_WGS=0`` to disable the wrapper and raise on lists
    instead — useful if the wrapper ever produces unexpected behaviour
    in production.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        return obj
    if os.environ.get("AIKIXP_MERGE_WGS", "1") == "0":
        raise ValueError(
            f"Genome pickle {path} is a multi-record list (len={len(obj)}); "
            "AIKIXP_MERGE_WGS=0 disables the merge wrapper."
        )
    if len(obj) == 1:
        return obj[0]
    return MergedGenome(obj)


class MergedGenome:
    """Virtual SeqRecord built from a list of SeqRecord contigs.

    Presents `.seq`, `.features`, `.description`, and `.annotations` so the
    existing `find_cds_in_genome` / `_find_operon` / Bacformer enumeration
    paths can consume it unchanged. The concatenated sequence places each
    contig end-to-end with a 2 kb N-gap between contigs; this guarantees
    operon detection (which requires ≤150 nt intergenic distance) never
    bridges contigs.

    Features are copied with locations shifted into the merged coordinate
    space. The filtered CDS list (`CDS & translation & non-pseudo`) sorted
    by `location.start` therefore yields a deterministic, contig-ordered
    enumeration that Bacformer extraction can use as its canonical index
    space (CLAUDE.md Rule 9).
    """

    __slots__ = ("id", "description", "annotations", "seq", "features")

    def __init__(self, records: list):
        from Bio.Seq import Seq
        from Bio.SeqFeature import FeatureLocation, SeqFeature

        seq_parts: list[str] = []
        offsets: list[int] = []
        cursor = 0
        gap = "N" * _CONTIG_GAP_LEN
        for idx, rec in enumerate(records):
            offsets.append(cursor)
            seq_parts.append(str(rec.seq))
            cursor += len(rec.seq)
            if idx < len(records) - 1:
                seq_parts.append(gap)
                cursor += _CONTIG_GAP_LEN
        self.seq = Seq("".join(seq_parts))

        merged_feats: list = []
        for rec, offset in zip(records, offsets):
            for feat in rec.features:
                try:
                    start = int(feat.location.start) + offset
                    end = int(feat.location.end) + offset
                    strand = feat.location.strand
                except Exception:
                    continue
                new_loc = FeatureLocation(start, end, strand=strand)
                merged_feats.append(
                    SeqFeature(
                        location=new_loc,
                        type=feat.type,
                        qualifiers=feat.qualifiers,
                    )
                )
        self.features = merged_feats

        first = records[0]
        self.id = getattr(first, "id", "") or ""
        self.description = getattr(first, "description", "") or ""
        self.annotations = dict(getattr(first, "annotations", {}) or {})
        self.annotations.setdefault("wgs_contigs", [getattr(r, "id", "") for r in records])


@dataclass
class GeneContext:
    """All the fields the Tier D featurization needs for one gene."""
    protein_sequence: str
    dna_cds_seq: str
    full_operon_dna: str
    cds_start_in_operon: int
    upstream_context_seq: str
    downstream_context_seq: str
    rna_init_window_seq: str
    strand: str
    mode: str  # "native" or "heterologous"
    genome_cds_index: Optional[int]  # for Bacformer; None in heterologous
    locus_tag: Optional[str]
    num_genes_in_operon: int


def _reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in seq.upper()[::-1])


def find_cds_in_genome(
    genome,  # BioPython SeqRecord
    cds_dna: str,
) -> Optional[tuple[int, int, int, str, str]]:
    """Exact-substring search for a user's CDS in a genome record.

    Returns (start, end, cds_feature_index, strand, locus_tag) or None.
    """
    cds_dna = cds_dna.upper().replace("U", "T")
    genome_seq = str(genome.seq).upper()

    # Forward strand
    idx = genome_seq.find(cds_dna)
    if idx >= 0:
        # Find which CDS feature this is
        cds_feats = [f for f in genome.features
                     if f.type == "CDS" and "translation" in f.qualifiers and "pseudo" not in f.qualifiers]
        for i, feat in enumerate(cds_feats):
            if int(feat.location.start) == idx and feat.location.strand == 1:
                lt = feat.qualifiers.get("locus_tag", [""])[0]
                return (idx, idx + len(cds_dna), i, "+", lt)
        return (idx, idx + len(cds_dna), -1, "+", "")

    # Reverse strand
    rc = _reverse_complement(cds_dna)
    idx = genome_seq.find(rc)
    if idx >= 0:
        cds_feats = [f for f in genome.features
                     if f.type == "CDS" and "translation" in f.qualifiers and "pseudo" not in f.qualifiers]
        for i, feat in enumerate(cds_feats):
            if int(feat.location.end) == idx + len(rc) and feat.location.strand == -1:
                lt = feat.qualifiers.get("locus_tag", [""])[0]
                return (idx, idx + len(rc), i, "-", lt)
        return (idx, idx + len(rc), -1, "-", "")

    return None


def _find_operon(
    genome,
    cds_start: int,
    cds_end: int,
    strand: str,
    max_gap: int = MAX_INTERGENIC_OPERON_DISTANCE,
) -> tuple[int, int, int, list]:
    """Find operon boundaries by extending through co-directional genes with small intergenic gaps.

    Returns (operon_start, operon_end, num_genes, list_of_cds_features).
    """
    # All CDS features on the same strand, sorted by position
    cds_feats = sorted(
        [f for f in genome.features
         if f.type == "CDS"
         and "translation" in f.qualifiers
         and "pseudo" not in f.qualifiers
         and ((strand == "+" and f.location.strand == 1)
              or (strand == "-" and f.location.strand == -1))],
        key=lambda f: int(f.location.start),
    )

    # Find the anchor gene
    anchor_idx = None
    for i, f in enumerate(cds_feats):
        if int(f.location.start) == cds_start and int(f.location.end) == cds_end:
            anchor_idx = i
            break

    if anchor_idx is None:
        # CDS not found exactly — return just this gene as a singleton operon
        return (cds_start, cds_end, 1, [])

    # Extend upstream (toward lower positions for +, higher for -)
    # For simplicity, always treat genome coordinates as forward-strand positions
    operon_members = [cds_feats[anchor_idx]]
    i = anchor_idx - 1
    while i >= 0:
        prev = cds_feats[i]
        curr = operon_members[0]
        gap = int(curr.location.start) - int(prev.location.end)
        if gap > max_gap:
            break
        operon_members.insert(0, prev)
        i -= 1

    i = anchor_idx + 1
    while i < len(cds_feats):
        nxt = cds_feats[i]
        curr = operon_members[-1]
        gap = int(nxt.location.start) - int(curr.location.end)
        if gap > max_gap:
            break
        operon_members.append(nxt)
        i += 1

    op_start = int(operon_members[0].location.start)
    op_end = int(operon_members[-1].location.end)
    return (op_start, op_end, len(operon_members), operon_members)


def lookup_native_gene(
    genome,
    cds_dna: str,
    protein_sequence: Optional[str] = None,
) -> Optional[GeneContext]:
    """Native gene: CDS is in the genome. Extract all Tier D fields."""
    hit = find_cds_in_genome(genome, cds_dna)
    if hit is None:
        return None
    cds_start_g, cds_end_g, cds_feat_idx, strand, locus_tag = hit

    op_start, op_end, n_genes, _ = _find_operon(genome, cds_start_g, cds_end_g, strand)
    genome_seq = str(genome.seq).upper()

    # Full operon DNA (reverse-complemented if on minus strand)
    operon_raw = genome_seq[op_start:op_end]
    if strand == "-":
        full_operon_dna = _reverse_complement(operon_raw)
        cds_start_in_operon = op_end - cds_end_g
    else:
        full_operon_dna = operon_raw
        cds_start_in_operon = cds_start_g - op_start

    # Upstream / downstream context (100 / 50 nt, strand-aware)
    if strand == "+":
        ups = genome_seq[max(0, cds_start_g - UPSTREAM_CONTEXT_NT) : cds_start_g]
        dns = genome_seq[cds_end_g : cds_end_g + DOWNSTREAM_CONTEXT_NT]
    else:
        ups = _reverse_complement(genome_seq[cds_end_g : cds_end_g + UPSTREAM_CONTEXT_NT])
        dns = _reverse_complement(genome_seq[max(0, cds_start_g - DOWNSTREAM_CONTEXT_NT) : cds_start_g])

    # RNA init window (60 nt: 35 up + 25 down of ATG) — from full_operon_dna + cds_start_in_operon
    win_start = max(0, cds_start_in_operon - INIT_UPSTREAM)
    win_end = min(len(full_operon_dna), cds_start_in_operon + INIT_DOWNSTREAM)
    init_win_dna = full_operon_dna[win_start:win_end]
    rna_init_window = init_win_dna.replace("T", "U").replace("t", "u")

    # Derive protein sequence if not provided
    if not protein_sequence:
        protein_sequence = _translate(cds_dna)

    return GeneContext(
        protein_sequence=protein_sequence,
        dna_cds_seq=cds_dna.upper().replace("U", "T"),
        full_operon_dna=full_operon_dna,
        cds_start_in_operon=cds_start_in_operon,
        upstream_context_seq=ups,
        downstream_context_seq=dns,
        rna_init_window_seq=rna_init_window,
        strand=strand,
        mode="native",
        genome_cds_index=cds_feat_idx if cds_feat_idx >= 0 else None,
        locus_tag=locus_tag,
        num_genes_in_operon=n_genes,
    )


def synthesize_heterologous_context(
    genome,
    cds_dna: str,
    host_anchor_locus_tag: str = "lacZ",
    protein_sequence: Optional[str] = None,
) -> GeneContext:
    """Heterologous expression: wrap user's CDS in the host's {anchor} context.

    Default anchor: lacZ (the paper's K12 pseudo-operon protocol).
    Uses the anchor's chromosomal upstream 100 nt + user CDS + anchor downstream 50 nt.
    """
    # Find the anchor gene in the host genome
    anchor_feats = []
    for f in genome.features:
        if f.type != "CDS":
            continue
        if "pseudo" in f.qualifiers:
            continue
        gene_name = f.qualifiers.get("gene", [""])[0].lower()
        lt = f.qualifiers.get("locus_tag", [""])[0].lower()
        if host_anchor_locus_tag.lower() in (gene_name, lt):
            anchor_feats.append(f)

    if not anchor_feats:
        # Fall back: use the first CDS as anchor (still gives a real promoter context)
        cds_list = [f for f in genome.features if f.type == "CDS" and "pseudo" not in f.qualifiers]
        if not cds_list:
            raise RuntimeError(f"No CDS features found in genome for heterologous fallback")
        anchor = cds_list[0]
    else:
        anchor = anchor_feats[0]

    # Find the anchor's index in the Bacformer-filtered CDS list
    # (same filter Bacformer's extractor uses: CDS + translation + not pseudo)
    anchor_cds_index = None
    filt_cds = [f for f in genome.features
                if f.type == "CDS" and "translation" in f.qualifiers and "pseudo" not in f.qualifiers]
    filt_cds.sort(key=lambda f: int(f.location.start))
    for i, f in enumerate(filt_cds):
        if int(f.location.start) == int(anchor.location.start) and f.location.strand == anchor.location.strand:
            anchor_cds_index = i
            break

    genome_seq = str(genome.seq).upper()
    a_start = int(anchor.location.start)
    a_end = int(anchor.location.end)
    a_strand = "+" if anchor.location.strand == 1 else "-"

    if a_strand == "+":
        anchor_ups = genome_seq[max(0, a_start - UPSTREAM_CONTEXT_NT) : a_start]
        anchor_dns = genome_seq[a_end : a_end + DOWNSTREAM_CONTEXT_NT]
    else:
        anchor_ups = _reverse_complement(genome_seq[a_end : a_end + UPSTREAM_CONTEXT_NT])
        anchor_dns = _reverse_complement(genome_seq[max(0, a_start - DOWNSTREAM_CONTEXT_NT) : a_start])

    cds_clean = cds_dna.upper().replace("U", "T")
    full_operon_dna = anchor_ups + cds_clean + anchor_dns
    cds_start_in_operon = len(anchor_ups)

    win_start = max(0, cds_start_in_operon - INIT_UPSTREAM)
    win_end = min(len(full_operon_dna), cds_start_in_operon + INIT_DOWNSTREAM)
    init_win_dna = full_operon_dna[win_start:win_end]
    rna_init_window = init_win_dna.replace("T", "U")

    if not protein_sequence:
        protein_sequence = _translate(cds_clean)

    return GeneContext(
        protein_sequence=protein_sequence,
        dna_cds_seq=cds_clean,
        full_operon_dna=full_operon_dna,
        cds_start_in_operon=cds_start_in_operon,
        upstream_context_seq=anchor_ups,
        downstream_context_seq=anchor_dns,
        rna_init_window_seq=rna_init_window,
        strand=a_strand,
        mode="heterologous",
        # Use the anchor's CDS index so Bacformer-large returns the anchor's
        # genome-neighborhood embedding — the user's CDS inherits that context.
        genome_cds_index=anchor_cds_index,
        locus_tag=host_anchor_locus_tag,
        num_genes_in_operon=1,
    )


# ── Minimal codon table for back-translation fallback ──────────────────────────

_STANDARD_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate(cds: str) -> str:
    cds = cds.upper().replace("U", "T")
    aa = []
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i:i + 3]
        if "N" in codon:
            aa.append("X")
            continue
        a = _STANDARD_CODE.get(codon, "X")
        if a == "*":
            break
        aa.append(a)
    return "".join(aa)
