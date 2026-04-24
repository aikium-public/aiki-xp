"""Download and cache new bacterial genomes from NCBI for Aiki-XP.

Given an NCBI chromosome-level accession (NC_*, NZ_*, CP*, CM*), fetch
the GenBank record via Entrez, validate it, and write a pickle to the
target directory matching the format expected by genome_lookup.py and
the Modal endpoints — a single BioPython SeqRecord with .seq and .features.

Public-repo port of the NCBI fetch logic in
scripts/protex/download_missing_bacformer_genomes.py (private repo).

CLAUDE.md rules that apply:
  - Rule 9: pseudogene filtering. Any Bacformer cache built from the
    returned pickle MUST enumerate CDS in the filtered space:
      [f for f in rec.features
       if f.type == "CDS"
       and "translation" in f.qualifiers
       and "pseudo" not in f.qualifiers]
    This module does not build the Bacformer cache itself; it only
    emits pickles that downstream extractors can filter. See
    genome_lookup.py for the canonical filter.
  - Rule 11: never overwrite the 1,831 baseline pickles. The caller
    (the /request_genome endpoint) checks file existence BEFORE calling
    and returns `already_cached` without touching disk.
  - Rule 12: no silent fallback. Every failure raises
    GenomeDownloadError with a clear message.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Validation policy ────────────────────────────────────────────────────────

# Loose accession regex — backend does the real check by attempting a fetch.
# RefSeq prefixes: NC_, NZ_, NT_, NW_. GenBank direct submission: CP, CM, AE, AL.
_ACCESSION_RE = re.compile(r"^(NC_|NZ_|NT_|NW_|CP|CM|AE|AL|GCA_|GCF_)[\w.]+$")
_ASSEMBLY_PREFIXES = ("GCA_", "GCF_")

# Upper bound on genome size. Bacterial chromosomes up to ~15 Mb exist
# (Sorangium cellulosum). Above that, assume eukaryotic / out-of-distribution.
MAX_GENOME_BP = 15_000_000

# Minimum valid (non-pseudo, translated) CDS for a usable bacterial host.
MIN_VALID_CDS = 100

# Pseudogene fractions above this flag the genome as a possible draft assembly.
PSEUDO_WARN_FRACTION = 0.30

# Address NCBI wants for polite-use tracking. Overridable via NCBI_EMAIL env
# var (the aikium-ncbi-api-key Modal secret populates this in production).
DEFAULT_NCBI_EMAIL = "aikixp-public@aikium.com"


class GenomeDownloadError(ValueError):
    """Raised when an NCBI fetch or validation fails. No silent fallback."""


@dataclass
class GenomeStats:
    accession: str
    name: str
    n_cds: int
    n_pseudo: int
    genome_bp: int
    pseudogene_fraction: float
    sha256: str
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["pseudogene_fraction"] = round(d["pseudogene_fraction"], 4)
        return d


# ── Public API ───────────────────────────────────────────────────────────────


def validate_accession_format(accession: str) -> str:
    """Normalize and format-check an accession. Raises on anything suspicious."""
    accession = (accession or "").strip()
    if not accession:
        raise GenomeDownloadError("Missing accession")
    if len(accession) > 40:
        raise GenomeDownloadError(
            f"Accession too long ({len(accession)} chars): {accession[:30]}..."
        )
    if not _ACCESSION_RE.match(accession):
        raise GenomeDownloadError(
            f"Unrecognized accession format: {accession!r}. "
            "Expect an NCBI RefSeq or GenBank chromosome accession "
            "such as NC_000913.3, NZ_CP007039.1, or CP158060.1."
        )
    if accession.startswith(_ASSEMBLY_PREFIXES):
        raise GenomeDownloadError(
            f"Assembly-level accessions ({accession}) are not supported yet. "
            "Please paste the chromosome-level accession (NC_*/NZ_*/CP*/CM*) "
            "of the primary replicon."
        )
    return accession


def fetch_genbank(accession: str, api_key: Optional[str] = None) -> list:
    """Fetch and parse a GenBank record via NCBI Entrez.

    Returns a list of BioPython SeqRecord (usually length 1 for a
    chromosome accession; WGS assemblies may return multiple contigs).

    Raises GenomeDownloadError on any failure — never returns [].
    """
    from Bio import Entrez, SeqIO

    Entrez.email = os.environ.get("NCBI_EMAIL") or DEFAULT_NCBI_EMAIL
    if api_key:
        Entrez.api_key = api_key

    try:
        handle = Entrez.efetch(
            db="nuccore",
            id=accession,
            rettype="gbwithparts",
            retmode="text",
        )
        text = handle.read()
        handle.close()
    except Exception as e:
        raise GenomeDownloadError(
            f"NCBI fetch failed for {accession}: {type(e).__name__}: {e}"
        ) from e

    if not text or not text.strip():
        raise GenomeDownloadError(f"NCBI returned empty body for {accession}")

    try:
        records = list(SeqIO.parse(StringIO(text), "genbank"))
    except Exception as e:
        raise GenomeDownloadError(
            f"Could not parse GenBank response for {accession}: "
            f"{type(e).__name__}: {e}"
        ) from e

    if not records:
        raise GenomeDownloadError(
            f"No GenBank records parsed for {accession} "
            "(the accession may exist but lack annotations)"
        )
    return records


def _count_cds(record) -> tuple[int, int]:
    """Return (n_valid_cds, n_pseudogene_cds).

    Filter matches genome_lookup.py and extract_proteome_for_bacformer
    — see CLAUDE.md Rule 9.
    """
    n_valid = 0
    n_pseudo = 0
    for feat in record.features:
        if feat.type != "CDS":
            continue
        if "pseudo" in feat.qualifiers:
            n_pseudo += 1
        elif "translation" in feat.qualifiers:
            n_valid += 1
    return n_valid, n_pseudo


def _organism_name(record) -> str:
    annotations = getattr(record, "annotations", {}) or {}
    name = annotations.get("organism") or record.description or record.id
    return str(name).strip().rstrip(".")


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def validate_and_pickle(record, accession: str, pkl_path: Path) -> GenomeStats:
    """Validate a single SeqRecord and write it to pkl_path. Raises on reject."""
    genome_bp = len(record.seq) if record.seq is not None else 0
    if genome_bp == 0:
        raise GenomeDownloadError(
            f"{accession} has no genome sequence (features-only record)."
        )
    if genome_bp > MAX_GENOME_BP:
        raise GenomeDownloadError(
            f"{accession} is {genome_bp / 1e6:.1f} Mb — too large. "
            f"Aiki-XP is pan-bacterial only (cap: {MAX_GENOME_BP / 1e6:.0f} Mb)."
        )

    n_valid, n_pseudo = _count_cds(record)
    if n_valid < MIN_VALID_CDS:
        raise GenomeDownloadError(
            f"{accession} has only {n_valid} valid CDS "
            f"(pseudogene count: {n_pseudo}). "
            f"Need at least {MIN_VALID_CDS} — is this a partial record?"
        )

    pseudo_frac = n_pseudo / max(1, n_pseudo + n_valid)
    warnings: list[str] = []
    if pseudo_frac > PSEUDO_WARN_FRACTION:
        warnings.append(
            f"High pseudogene fraction ({pseudo_frac:.1%}) — may be an "
            "incomplete draft assembly. Predictions may be less reliable."
        )

    blob = pickle.dumps(record)
    sha = _sha256_bytes(blob)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    pkl_path.write_bytes(blob)

    return GenomeStats(
        accession=accession,
        name=_organism_name(record),
        n_cds=n_valid,
        n_pseudo=n_pseudo,
        genome_bp=genome_bp,
        pseudogene_fraction=pseudo_frac,
        sha256=sha,
        warnings=warnings,
    )


def download_and_cache_genome(
    accession: str,
    target_dir: Path | str,
    api_key: Optional[str] = None,
    overwrite: bool = False,
) -> GenomeStats:
    """Fetch `accession` from NCBI, validate, pickle to target_dir/{acc}.pkl.

    Args:
        accession: NCBI chromosome-level accession (NC_*/NZ_*/CP*/CM*).
        target_dir: Where to write the .pkl (Modal volume mount in prod).
        api_key: Optional NCBI API key (from env or a Modal secret).
        overwrite: If False, refuses to overwrite an existing pickle.
            Callers SHOULD check existence first to distinguish the
            "already cached" vs "actually downloaded" UX states.

    Returns:
        GenomeStats describing the cached genome.

    Raises:
        GenomeDownloadError on any validation failure.
    """
    accession = validate_accession_format(accession)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = target_dir / f"{accession}.pkl"
    if pkl_path.exists() and not overwrite:
        raise GenomeDownloadError(
            f"Cache already contains {accession} — refusing to overwrite "
            "(baseline genomes are load-bearing; see CLAUDE.md Rule 11)."
        )

    records = fetch_genbank(accession, api_key=api_key)
    if len(records) > 1:
        raise GenomeDownloadError(
            f"{accession} returned {len(records)} records. "
            "Multi-record (WGS/draft) assemblies are not yet supported — "
            "please paste a single-chromosome accession."
        )

    stats = validate_and_pickle(records[0], accession, pkl_path)
    log.info(
        "cached %s: %s (%d bp, %d CDS, %d pseudogenes, sha=%s)",
        accession,
        stats.name,
        stats.genome_bp,
        stats.n_cds,
        stats.n_pseudo,
        stats.sha256[:12],
    )
    return stats
