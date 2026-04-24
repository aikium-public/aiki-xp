#!/usr/bin/env python3
"""Backfill species_keys / n_test / n_test_nm into bulk_added_genomes.jsonl.

For each bulk-added pickle we extract the NCBI taxid from the SeqRecord's
source feature. If that taxid matches a baseline host (via its
species_keys), we copy the baseline's species_keys + n_test + n_test_nm
onto the bulk entry. That unlocks the per-species scatter for hosts that
are the same species as a training-set species but represented by a
different chromosome accession.

Writes the enriched manifest to /tmp/bulk_added_genomes.enriched.jsonl.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

BULK_IN = Path("/tmp/bulk_added_genomes.jsonl")
BULK_OUT = Path("/tmp/bulk_added_genomes.enriched.jsonl")
BASELINE = Path("/home/venkatesh/HBG/aiki-xp/web/hosts.json")
PICKLE_DIR = Path("/home/venkatesh/HBG/scripts/binding_model/cache/protex/genomes")


def extract_taxid(pkl_path: Path) -> str | None:
    try:
        with open(pkl_path, "rb") as f:
            rec = pickle.load(f)
    except Exception:
        return None
    if isinstance(rec, list):
        # Multi-record WGS pickles are excluded from the bulk set so we
        # should never hit this branch; keep the guard anyway.
        rec = rec[0]
    for feat in rec.features:
        if feat.type != "source":
            continue
        for db in feat.qualifiers.get("db_xref", []):
            if db.startswith("taxon:"):
                return db.split(":", 1)[1]
    return None


def main() -> int:
    baseline = json.loads(BASELINE.read_text())

    # Build taxid -> list of (species_keys, n_test, n_test_nm) from baseline.
    # A taxid may map to multiple baseline entries (different representative
    # chromosomes of the same species); we take the first since n_test is
    # species-level and identical across them.
    taxid_to_info: dict[str, dict] = {}
    for h in baseline:
        sk = h.get("species_keys") or []
        nt = h.get("n_test")
        ntn = h.get("n_test_nm")
        for key in sk:
            # Only numeric keys are taxids. The baseline also uses species-
            # name strings like "Escherichia_coli_K12".
            if key.isdigit() and key not in taxid_to_info:
                taxid_to_info[key] = {
                    "species_keys": sk,
                    "n_test": nt,
                    "n_test_nm": ntn,
                }

    with open(BULK_IN) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    enriched = 0
    no_taxid = 0
    no_match = 0
    out_entries = []
    for e in entries:
        pkl = PICKLE_DIR / f"{e['acc']}.pkl"
        taxid = extract_taxid(pkl)
        if taxid is None:
            no_taxid += 1
            out_entries.append(e)
            continue
        info = taxid_to_info.get(taxid)
        if info is None:
            no_match += 1
            out_entries.append(e)
            continue
        # Enrich. Keep a copy of the original empty list under `species_keys`
        # replaced, and record taxid for traceability.
        enriched_entry = dict(e)
        enriched_entry["species_keys"] = info["species_keys"]
        if info.get("n_test") is not None:
            enriched_entry["n_test"] = info["n_test"]
        if info.get("n_test_nm") is not None:
            enriched_entry["n_test_nm"] = info["n_test_nm"]
        enriched_entry["taxid"] = taxid
        out_entries.append(enriched_entry)
        enriched += 1

    with open(BULK_OUT, "w") as f:
        for e in out_entries:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")

    print(f"total bulk entries: {len(entries)}")
    print(f"  enriched (taxid matched baseline): {enriched}")
    print(f"  taxid not found in pickle:         {no_taxid}")
    print(f"  taxid found but no baseline match: {no_match}")
    print(f"wrote {BULK_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
