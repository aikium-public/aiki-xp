#!/usr/bin/env python3
"""Bulk-upload the full Aiki-XP genome cache (~11.5K pickles, 11.5 GB) to the
Modal `aikixp-genomes` volume and extend the public host manifest.

Inputs:
  --src DIR     Local directory of `.pkl` genome pickles (default:
                /home/venkatesh/HBG/scripts/binding_model/cache/protex/genomes)
  --manifest-out PATH  Write a JSONL of new host entries (default:
                /tmp/bulk_added_genomes.jsonl)
  --min-cds N   Skip genomes with fewer than N valid (non-pseudo) CDS (default: 100)
  --max-bp N    Skip genomes longer than N bp (default: 15_000_000)
  --dry-run     Don't upload; just validate + manifest-write

Steps:
  1) Walk the source directory; for each .pkl, unpickle the SeqRecord and
     apply the same Rule-9 CDS filter as genome_lookup.py.
  2) Skip genomes already on the Modal volume (by filename). Skip pickles
     that fail validation (reason logged to stderr).
  3) Write a JSONL manifest of passing genomes with {acc, name, n,
     genome_bp, n_pseudo, sha256, bulk_added: true, added_at}.
  4) `modal volume put aikixp-genomes <local_path> /<filename>` for each
     new pickle (batched).
  5) `modal volume put aikixp-genomes <manifest> /bulk_added_genomes.jsonl`.

The landing ASGI /hosts.json then merges baseline + user-added + bulk-added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SRC = Path("/home/venkatesh/HBG/scripts/binding_model/cache/protex/genomes")
DEFAULT_MANIFEST = Path("/tmp/bulk_added_genomes.jsonl")
MODAL_VOLUME = "aikixp-genomes"

MIN_CDS = 100
MAX_BP = 15_000_000


def count_valid_cds(record) -> tuple[int, int]:
    """Same filter as aikixp/genome_lookup.py (CLAUDE.md Rule 9)."""
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


def organism_name(record) -> str:
    annotations = getattr(record, "annotations", {}) or {}
    name = annotations.get("organism") or record.description or record.id
    return str(name).strip().rstrip(".")


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_pickle(p: Path, min_cds: int, max_bp: int) -> dict | None:
    """Return a manifest entry dict, or None if the pickle is invalid.

    Accepts both single-SeqRecord pickles and list-of-SeqRecord (WGS merged)
    pickles. The Modal landing endpoints consume either format transparently
    via `aikixp.genome_lookup.load_genome`, which wraps lists in a
    MergedGenome that exposes the concatenated `.seq` and a re-indexed
    `.features` list.
    """
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"SKIP {p.name}: unpickle failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None

    multi_record = isinstance(obj, list)
    if multi_record and len(obj) == 1:
        obj = obj[0]
        multi_record = False

    if multi_record:
        n_records = len(obj)
        if n_records == 0:
            print(f"SKIP {p.name}: empty list pickle", file=sys.stderr)
            return None
        seq_len = 0
        n_valid = 0
        n_pseudo = 0
        for rec in obj:
            try:
                seq_len += len(rec.seq) if rec.seq is not None else 0
            except Exception as e:
                print(f"SKIP {p.name}: contig unreadable: {type(e).__name__}", file=sys.stderr)
                return None
            v, ps = count_valid_cds(rec)
            n_valid += v
            n_pseudo += ps
        name = organism_name(obj[0])
    else:
        try:
            seq_len = len(obj.seq) if obj.seq is not None else 0
            _ = obj.features
        except Exception as e:
            print(f"SKIP {p.name}: not a SeqRecord ({type(obj).__name__}): {e}", file=sys.stderr)
            return None
        n_valid, n_pseudo = count_valid_cds(obj)
        name = organism_name(obj)
        n_records = 1

    if seq_len == 0:
        print(f"SKIP {p.name}: empty genome seq", file=sys.stderr)
        return None
    if seq_len > max_bp:
        print(f"SKIP {p.name}: {seq_len / 1e6:.1f} Mb > {max_bp / 1e6:.0f} Mb cap", file=sys.stderr)
        return None
    if n_valid < min_cds:
        print(f"SKIP {p.name}: only {n_valid} valid CDS (<{min_cds})", file=sys.stderr)
        return None

    acc = p.stem
    entry: dict = {
        "acc": acc,
        "name": name,
        "n": n_valid,
        "genome_bp": seq_len,
        "n_pseudo": n_pseudo,
        "species_keys": [],
        "bulk_added": True,
        "added_at": int(time.time()),
        "sha256": sha256_path(p),
        "_local_path": str(p),   # consumed by the uploader; stripped before writing
    }
    if n_records > 1:
        entry["n_contigs"] = n_records
    return entry


def modal_volume_list() -> set[str]:
    """Return the set of .pkl filenames already present on the Modal volume."""
    print("Listing existing entries on Modal volume...", flush=True)
    try:
        out = subprocess.run(
            ["modal", "volume", "ls", MODAL_VOLUME, "/"],
            check=True, capture_output=True, text=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"modal volume ls failed: {e.stderr}", file=sys.stderr)
        return set()
    existing = set()
    for line in out.splitlines():
        parts = line.split()
        for part in parts:
            if part.endswith(".pkl"):
                existing.add(part.lstrip("/").strip())
    print(f"  found {len(existing)} existing .pkl files on volume")
    return existing


def modal_volume_put(src: Path, dst: str, force: bool = False) -> bool:
    cmd = ["modal", "volume", "put", MODAL_VOLUME, str(src), dst]
    if force:
        cmd.append("--force")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  upload failed for {src}: {e.stderr[:400]}", file=sys.stderr)
        return False


def modal_volume_put_dir(src_dir: Path, dst: str = "/") -> bool:
    """Upload a directory in one `modal volume put` call. No --force so the
    baseline 1,831 pickles on the volume are never overwritten (Rule 11)."""
    cmd = ["modal", "volume", "put", MODAL_VOLUME, str(src_dir), dst]
    print(f"  running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  directory upload failed: exit {e.returncode}", file=sys.stderr)
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--min-cds", type=int, default=MIN_CDS)
    p.add_argument("--max-bp", type=int, default=MAX_BP)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="Upload at most N new pickles (0 = no cap). Useful for a canary batch.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Validate + write manifest but skip the modal volume put step.")
    args = p.parse_args()

    src = args.src
    if not src.is_dir():
        print(f"--src not a directory: {src}", file=sys.stderr)
        return 2
    pickles = sorted(src.glob("*.pkl"))
    print(f"Found {len(pickles)} local pickles under {src}")

    existing = set() if args.dry_run else modal_volume_list()

    entries: list[dict] = []
    to_upload: list[Path] = []
    for pkl in pickles:
        fname = pkl.name
        if fname in existing:
            continue  # already on volume — don't re-upload or re-register
        entry = validate_pickle(pkl, args.min_cds, args.max_bp)
        if entry is None:
            continue
        entries.append(entry)
        to_upload.append(pkl)
        if args.limit and len(to_upload) >= args.limit:
            break

    print(f"Validated {len(entries)} genomes ready for upload "
          f"(skipped {len(pickles) - len(entries) - len(existing)} on disk, "
          f"{len(existing)} already on volume).")

    # Write manifest (strip the internal _local_path key).
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        for entry in entries:
            clean = {k: v for k, v in entry.items() if not k.startswith("_")}
            f.write(json.dumps(clean, separators=(",", ":")) + "\n")
    print(f"Wrote manifest: {args.manifest_out} ({len(entries)} entries)")

    if args.dry_run or args.skip_upload:
        print("dry-run / skip-upload: not uploading to Modal volume.")
        return 0

    # Hardlink validated pickles into a staging dir so `modal volume put`
    # uploads only the subset we want (some pickles in --src fail validation
    # and shouldn't get registered or uploaded). Hardlinks are O(0) bytes and
    # avoid duplicating 11 GB of data. No --force on upload = existing
    # pickles on the volume (baseline 1,831) are untouched (Rule 11).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="aikixp_bulk_") as tmp:
        stage = Path(tmp)
        print(f"Hardlinking {len(to_upload)} validated pickles into {stage}...")
        n_linked = 0
        for pkl in to_upload:
            dst = stage / pkl.name
            try:
                dst.hardlink_to(pkl)
                n_linked += 1
            except OSError as e:
                # Fall back to a copy if hardlinks aren't allowed (e.g. across FS).
                import shutil
                shutil.copy2(pkl, dst)
                n_linked += 1
        print(f"Staged {n_linked} pickles.")

        t0 = time.time()
        print(f"Uploading directory to Modal volume {MODAL_VOLUME}...")
        ok = modal_volume_put_dir(stage, "/")
        elapsed = time.time() - t0
        if not ok:
            print("Directory upload failed; manifest NOT uploaded.")
            return 1
        print(f"Directory upload done in {elapsed:.0f}s "
              f"({n_linked / max(1e-3, elapsed):.1f} files/s).")

    # Upload the manifest itself — the landing ASGI reads it verbatim.
    modal_volume_put(args.manifest_out, "/bulk_added_genomes.jsonl", force=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
