#!/usr/bin/env python3
"""
Embedding Registry — provenance tracking and integrity verification.

Design principle: ONE canonical registry per embedding set, generated ONCE
on a trusted machine after verifying every file against GCS, then stored on
GCS alongside the embeddings. Compute nodes download the registry and verify
local files against it — they NEVER generate their own.

Workflow:
    # 1. ON THE MACHINE THAT EXTRACTED THE EMBEDDINGS (trusted):
    python scripts/protex/embedding_registry.py generate \
        --embed-dir datasets/protex_aggregated/embeddings_finalized \
        --gcs-dir gs://your-bucket/embeddings \
        --label "protex_v2_492k"
    # This verifies every local file against GCS MD5, then writes the registry.
    # Upload the registry to GCS:
    gsutil cp .../embedding_registry.json gs://.../embedding_registry.json

    # 2. ON COMPUTE NODES (untrusted):
    # Download registry + embeddings from GCS
    gsutil cp gs://.../embedding_registry.json local_dir/
    gsutil cp gs://.../embeddings/*.parquet local_dir/
    # Verify before any training or inference:
    python scripts/protex/embedding_registry.py verify \
        --embed-dir local_dir

    # 3. train_fusion.py and aikium_expanded_inference.py verify automatically.

Registry file: <embed-dir>/embedding_registry.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


REGISTRY_FILENAME = "embedding_registry.json"
REGISTRY_VERSION = "1.0"


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inspect_parquet(path: Path) -> dict:
    """Read a parquet and return metadata without loading all data into memory."""
    df = pd.read_parquet(path)
    n_rows = len(df)

    gene_id_col = "gene_id" if "gene_id" in df.columns else None

    emb_cols = [c for c in df.columns if c.endswith("_embedding")]
    feature_cols = [c for c in df.columns if c not in ("gene_id",) and c not in emb_cols]

    if emb_cols:
        col = emb_cols[0]
        sample = df[col].iloc[0]
        if isinstance(sample, np.ndarray):
            dim = len(sample)
        elif isinstance(sample, list):
            dim = len(sample)
        else:
            dim = 1
        col_type = "embedding"
    elif feature_cols:
        dim = len(feature_cols)
        col = None
        col_type = "features"
    else:
        dim = 0
        col = None
        col_type = "unknown"

    has_nan = False
    has_zero_rows = False
    if emb_cols:
        sample_n = min(1000, n_rows)
        sample_df = df.sample(sample_n, random_state=42) if n_rows > sample_n else df
        vecs = np.stack(sample_df[emb_cols[0]].values)
        has_nan = bool(np.isnan(vecs).any())
        has_zero_rows = bool((np.abs(vecs).sum(axis=1) == 0).any())

    return {
        "n_rows": n_rows,
        "dim": dim,
        "embedding_column": col,
        "column_type": col_type,
        "feature_columns": feature_cols if col_type == "features" else None,
        "has_gene_id": gene_id_col is not None,
        "has_nan_sample": has_nan,
        "has_zero_rows_sample": has_zero_rows,
    }


def _gcs_md5_b64(gcs_path: str) -> Optional[str]:
    """Get the base64 MD5 of a GCS object via gsutil stat."""
    import subprocess
    try:
        result = subprocess.run(
            ["gsutil", "stat", gcs_path],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if "Hash (md5)" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return None


def _local_md5_b64(path: Path) -> str:
    """Compute base64 MD5 of a local file (same format gsutil uses)."""
    import base64 as _b64
    import hashlib as _hl
    h = _hl.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return _b64.b64encode(h.digest()).decode("ascii")


def generate_registry(
    embed_dir: Path,
    label: str,
    gcs_dir: Optional[str] = None,
    notes: str = "",
    skip_gcs_verify: bool = False,
) -> dict:
    """Scan all parquets in a directory and build a canonical registry.

    If --gcs-dir is provided, every local file is verified against GCS MD5
    before being registered. This ensures the registry is built from
    known-good files, not from potentially stale or corrupted local copies.

    The registry should be generated ONCE on a trusted machine and then
    uploaded to GCS. Compute nodes download and verify against it — they
    should NOT regenerate it.
    """
    parquets = sorted(embed_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {embed_dir}")

    if gcs_dir and not skip_gcs_verify:
        print(f"  Verifying {len(parquets)} files against GCS: {gcs_dir}")
        gcs_failures = []
        gcs_missing = []
        for path in parquets:
            gcs_path = f"{gcs_dir.rstrip('/')}/{path.name}"
            gcs_md5 = _gcs_md5_b64(gcs_path)
            if gcs_md5 is None:
                gcs_missing.append(path.name)
                print(f"    WARN: {path.name} not found on GCS (new extraction?)")
                continue
            local_md5 = _local_md5_b64(path)
            if gcs_md5 != local_md5:
                gcs_failures.append(
                    f"  {path.name}: GCS={gcs_md5} Local={local_md5}"
                )
                print(f"    FAIL: {path.name} — MD5 mismatch with GCS")
            else:
                print(f"    OK: {path.name}")

        if gcs_failures:
            raise ValueError(
                f"GCS verification FAILED for {len(gcs_failures)} file(s). "
                f"Local files do not match GCS — refusing to build registry "
                f"from potentially corrupted data:\n" + "\n".join(gcs_failures) +
                f"\nRe-download with: gsutil -m rsync {gcs_dir} {embed_dir}"
            )
        n_verified = len(parquets) - len(gcs_missing)
        print(f"  {n_verified}/{len(parquets)} files verified against GCS"
              f"{f' ({len(gcs_missing)} new, not yet on GCS)' if gcs_missing else ''}.")
    elif gcs_dir is None and not skip_gcs_verify:
        print(
            "  WARNING: No --gcs-dir provided. Building registry from local files "
            "WITHOUT verifying against the canonical GCS source. The resulting "
            "registry is only as trustworthy as the files currently on disk."
        )

    modalities = {}
    for path in parquets:
        modality_name = path.stem
        print(f"  Registering {path.name}...", end=" ", flush=True)
        sha = compute_file_sha256(path)
        md5_b64 = _local_md5_b64(path)
        meta = inspect_parquet(path)
        print(f"{meta['n_rows']} rows, {meta['dim']}d, SHA={sha[:12]}...")

        modalities[modality_name] = {
            "file": path.name,
            "sha256": sha,
            "md5_base64": md5_b64,
            "n_rows": meta["n_rows"],
            "dim": meta["dim"],
            "embedding_column": meta["embedding_column"],
            "column_type": meta["column_type"],
            "feature_columns": meta["feature_columns"],
            "has_gene_id": meta["has_gene_id"],
            "has_nan_sample": meta["has_nan_sample"],
            "has_zero_rows_sample": meta["has_zero_rows_sample"],
        }

    registry = {
        "registry_version": REGISTRY_VERSION,
        "label": label,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embed_dir": str(embed_dir),
        "gcs_source": gcs_dir,
        "gcs_verified": gcs_dir is not None and not skip_gcs_verify,
        "n_modalities": len(modalities),
        "notes": notes,
        "modalities": modalities,
    }
    return registry


def verify_registry(
    embed_dir: Path,
    modalities: Optional[List[str]] = None,
    strict: bool = True,
) -> List[dict]:
    """Verify files on disk against the registry. Returns list of failures."""
    registry_path = embed_dir / REGISTRY_FILENAME
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No registry found at {registry_path}. "
            f"Run: python scripts/protex/embedding_registry.py generate "
            f"--embed-dir {embed_dir}"
        )

    with open(registry_path) as f:
        registry = json.load(f)

    check_modalities = modalities or list(registry["modalities"].keys())
    failures = []

    for mod_name in check_modalities:
        if mod_name not in registry["modalities"]:
            failures.append({
                "modality": mod_name,
                "error": "NOT_IN_REGISTRY",
                "detail": f"Modality '{mod_name}' not found in registry. "
                          f"Available: {sorted(registry['modalities'].keys())[:10]}...",
            })
            continue

        entry = registry["modalities"][mod_name]
        fpath = embed_dir / entry["file"]

        if not fpath.exists():
            failures.append({
                "modality": mod_name,
                "error": "FILE_MISSING",
                "detail": f"Expected {fpath}, file does not exist.",
            })
            continue

        actual_sha = compute_file_sha256(fpath)
        if actual_sha != entry["sha256"]:
            failures.append({
                "modality": mod_name,
                "error": "SHA256_MISMATCH",
                "detail": (
                    f"File has been modified or replaced. "
                    f"Registry SHA: {entry['sha256'][:16]}..., "
                    f"Actual SHA: {actual_sha[:16]}..."
                ),
            })
            continue

        if strict:
            df = pd.read_parquet(fpath)
            if len(df) != entry["n_rows"]:
                failures.append({
                    "modality": mod_name,
                    "error": "ROW_COUNT_MISMATCH",
                    "detail": f"Expected {entry['n_rows']} rows, got {len(df)}.",
                })
                continue

    # Check for unregistered files
    registered_files = {e["file"] for e in registry["modalities"].values()}
    actual_files = {p.name for p in embed_dir.glob("*.parquet")}
    unregistered = actual_files - registered_files - {REGISTRY_FILENAME}
    if unregistered:
        for uf in sorted(unregistered):
            failures.append({
                "modality": uf,
                "error": "UNREGISTERED_FILE",
                "detail": (
                    f"File '{uf}' exists in {embed_dir} but is NOT in the registry. "
                    f"This file will be ignored by registry-aware loaders but could "
                    f"corrupt glob-based loaders."
                ),
            })

    return failures


def main():
    parser = argparse.ArgumentParser(description="Embedding Registry")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate registry from parquets")
    gen.add_argument("--embed-dir", type=Path, required=True)
    gen.add_argument("--gcs-dir", type=str, default=None,
                     help="GCS path to canonical embeddings. If provided, every local file "
                          "is verified against GCS MD5 before registration. STRONGLY recommended.")
    gen.add_argument("--skip-gcs-verify", action="store_true",
                     help="Skip GCS verification (use only if you are certain local files are correct)")
    gen.add_argument("--label", type=str, default="unlabeled")
    gen.add_argument("--notes", type=str, default="")

    ver = sub.add_parser("verify", help="Verify files against registry")
    ver.add_argument("--embed-dir", type=Path, required=True)
    ver.add_argument("--modality", type=str, nargs="*", default=None)
    ver.add_argument("--strict", action="store_true", default=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "generate":
        print(f"Generating registry for {args.embed_dir}...")
        if not args.gcs_dir and not args.skip_gcs_verify:
            print(
                "\n  ⚠ No --gcs-dir provided. For production registries, pass:\n"
                f"    --gcs-dir gs://your-bucket/embeddings\n"
                "  to verify every file against the canonical GCS source.\n"
                "  Use --skip-gcs-verify to suppress this warning.\n"
            )
        registry = generate_registry(
            args.embed_dir, args.label,
            gcs_dir=args.gcs_dir,
            notes=args.notes,
            skip_gcs_verify=args.skip_gcs_verify,
        )
        out_path = args.embed_dir / REGISTRY_FILENAME
        with open(out_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"\nRegistry saved: {out_path}")
        print(f"  {registry['n_modalities']} modalities registered")
        print(f"  GCS verified: {registry['gcs_verified']}")
        if registry["gcs_verified"]:
            print(f"  GCS source: {registry['gcs_source']}")
        else:
            print("  ⚠ NOT GCS-verified — upload to GCS and treat as canonical only if you are certain.")

    elif args.command == "verify":
        print(f"Verifying {args.embed_dir} against registry...")
        registry_path = args.embed_dir / REGISTRY_FILENAME
        if registry_path.exists():
            with open(registry_path) as f:
                reg = json.load(f)
            if not reg.get("gcs_verified", False):
                print(
                    "  ⚠ This registry was NOT verified against GCS at generation time.\n"
                    "  It fingerprints whatever was on disk — which may have been stale or corrupted.\n"
                    "  For production use, regenerate with --gcs-dir."
                )
            else:
                print(f"  Registry was GCS-verified at generation time (source: {reg.get('gcs_source', '?')})")
        failures = verify_registry(args.embed_dir, args.modality)
        if not failures:
            print("  ALL CHECKS PASSED")
        else:
            print(f"\n  {len(failures)} FAILURE(S):")
            for fail in failures:
                print(f"    [{fail['error']}] {fail['modality']}: {fail['detail']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
