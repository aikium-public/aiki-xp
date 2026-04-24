#!/usr/bin/env python3
"""Phase 0.1 — External-benchmark contamination filter .

Computes the overlap between an external protein-expression / solubility benchmark
and the 492K protein training set used by Aiki-XP. The "high bar" is 30% sequence
identity AND 80% query coverage — anything above is treated as a homologous hit
that may inflate published predictors' performance through training-set memorization.

For every query gene we record:
    - max_pident_to_train ∈ [0, 1]
    - best_hit_train_gene
    - is_contaminated  := (max_pident >= 0.30 AND query_cov >= 0.80)

Outputs:
    results/landscape_2026/contamination/<benchmark>__vs_protex492k.json
    results/landscape_2026/contamination/<benchmark>__vs_protex492k.parquet
        (per-gene table with the contamination flag)
    results/landscape_2026/contamination/<benchmark>__clean_gene_ids.txt
        (newline-delimited gene_ids that survive the filter — used by downstream
         landscape baselines as the "clean" subset)

The same script can be pointed at an EXTERNAL TOOL'S TRAINING SET (Phase 2) by
passing --query-fasta directly instead of --benchmark-parquet.

Usage:
    # Phase 0.1: contaminate-check an external benchmark catalog
    python scripts/protex/compute_external_contamination.py \\
        --benchmark boel_2016 \\
        --benchmark-parquet datasets/external_validation/boel_2016_gene_catalog.parquet \\
        --threads 12

    # Phase 2: contaminate-check an external tool's training FASTA
    python scripts/protex/compute_external_contamination.py \\
        --benchmark netsolp_train \\
        --query-fasta /tmp/netsolp_train.fasta \\
        --threads 12

The target (our 492K training proteins) is loaded once from
    datasets/protex_aggregated/protex_aggregated_v1.1_final_freeze.parquet
using `protein_sequence` with fallback to `protein_sequence_paxdb` for v1_gold rows.

"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("contamination")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_PARQUET = (
    PROJECT_ROOT
    / "datasets"
    / "protex_aggregated"
    / "protex_aggregated_v1.1_final_freeze.parquet"
)
OUT_DIR = PROJECT_ROOT / "results" / "landscape_2026" / "contamination"

DEFAULT_PIDENT = 0.30
DEFAULT_COVERAGE = 0.80


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────


def _coalesce_protein_sequence(df: pd.DataFrame) -> pd.Series:
    """Return per-row protein sequence, coalescing protein_sequence with paxdb."""
    if "protein_sequence" not in df.columns:
        raise ValueError("production parquet missing protein_sequence column")
    seq = df["protein_sequence"].copy()
    if "protein_sequence_paxdb" in df.columns:
        seq = seq.fillna(df["protein_sequence_paxdb"])
    return seq


def load_target_492k() -> tuple[pd.Series, pd.Series]:
    """Load the 492K production training proteins. Returns (gene_ids, sequences)."""
    log.info(f"loading target 492K from {PROD_PARQUET}")
    cols = ["gene_id", "protein_sequence", "protein_sequence_paxdb"]
    df = pd.read_parquet(PROD_PARQUET, columns=cols)
    seq = _coalesce_protein_sequence(df)

    # Rule 1: explicit NaN check, do not silently drop.
    n_nan = seq.isna().sum()
    if n_nan > 0:
        raise ValueError(
            f"target 492K has {n_nan} NaN protein sequences after coalesce; "
            "fix the data source — silent dropna is forbidden"
        )

    log.info(f"target proteins: {len(df):,}")
    return df["gene_id"].astype(str), seq.astype(str)


def load_benchmark(benchmark_parquet: Path) -> tuple[pd.Series, pd.Series]:
    """Load an external benchmark catalog. Returns (gene_ids, sequences)."""
    log.info(f"loading benchmark from {benchmark_parquet}")
    df = pd.read_parquet(benchmark_parquet)
    if "gene_id" not in df.columns or "protein_sequence" not in df.columns:
        raise ValueError(
            f"{benchmark_parquet} must have gene_id + protein_sequence columns; "
            f"saw {df.columns.tolist()[:10]}"
        )
    seq = df["protein_sequence"].copy()
    if "protein_sequence_paxdb" in df.columns:
        seq = seq.fillna(df["protein_sequence_paxdb"])

    n_nan = seq.isna().sum()
    if n_nan > 0:
        raise ValueError(
            f"benchmark has {n_nan} NaN protein sequences — fix the catalog"
        )
    log.info(f"benchmark proteins: {len(df):,}")
    return df["gene_id"].astype(str), seq.astype(str)


def load_query_fasta(fasta_path: Path) -> tuple[pd.Series, pd.Series]:
    """Load query sequences directly from a FASTA file (for Phase 2)."""
    log.info(f"loading query FASTA from {fasta_path}")
    ids: list[str] = []
    seqs: list[str] = []
    cur_id, cur_seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur_seq))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            ids.append(cur_id)
            seqs.append("".join(cur_seq))
    log.info(f"query proteins: {len(ids):,}")
    return pd.Series(ids), pd.Series(seqs)


def write_fasta(gene_ids: pd.Series, sequences: pd.Series, fasta_path: Path) -> int:
    n = 0
    with open(fasta_path, "w") as f:
        for gid, seq in zip(gene_ids, sequences):
            seq = str(seq).strip().replace("*", "")
            if len(seq) < 10:
                continue
            f.write(f">{gid}\n{seq}\n")
            n += 1
    log.info(f"wrote {n:,} sequences to {fasta_path}")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# MMseqs2 search
# ──────────────────────────────────────────────────────────────────────────────


def run_mmseqs_search(
    query_fasta: Path,
    target_fasta: Path,
    output_tsv: Path,
    *,
    min_pident: float,
    min_coverage: float,
    threads: int,
    sensitivity: float = 7.5,
) -> pd.DataFrame:
    """Run MMseqs2 easy-search query→target with the given thresholds.

    Returns a DataFrame with columns
        [query, target, pident, alnlen, qstart, qend, tstart, tend, evalue, bits, qcov, tcov]
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="mmseqs2_contam_"))
    log.info(
        f"mmseqs easy-search query→target  pident≥{min_pident:.2f}  "
        f"qcov≥{min_coverage:.2f}  threads={threads}  sens={sensitivity}"
    )
    cmd = [
        "mmseqs",
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tsv),
        str(tmpdir),
        "--min-seq-id",
        f"{min_pident:.4f}",
        "-c",
        f"{min_coverage:.4f}",
        "--cov-mode",
        "0",  # bidirectional coverage
        "--alignment-mode",
        "3",  # full local alignment
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
        "--max-seqs",
        "50",
        "--format-output",
        "query,target,pident,alnlen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov",
    ]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    log.info(f"mmseqs search done in {time.time() - t0:.1f}s")

    if not output_tsv.exists() or output_tsv.stat().st_size == 0:
        log.warning("no MMseqs2 hits at the chosen thresholds — clean subset == full")
        return pd.DataFrame(
            columns=[
                "query",
                "target",
                "pident",
                "alnlen",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "evalue",
                "bits",
                "qcov",
                "tcov",
            ]
        )

    hits = pd.read_csv(
        output_tsv,
        sep="\t",
        header=None,
        names=[
            "query",
            "target",
            "pident",
            "alnlen",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bits",
            "qcov",
            "tcov",
        ],
    )
    log.info(f"raw hits at threshold: {len(hits):,}")
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────


def aggregate_contamination(
    query_ids: pd.Series,
    hits: pd.DataFrame,
    *,
    pident_threshold: float,
    coverage_threshold: float,
) -> pd.DataFrame:
    """Per-query: max pident, best hit, is_contaminated flag."""
    if len(hits) == 0:
        out = pd.DataFrame({"gene_id": query_ids.astype(str)})
        out["max_pident"] = 0.0
        out["best_hit_train_gene"] = ""
        out["best_qcov"] = 0.0
        out["is_contaminated"] = False
        return out

    hits = hits.copy()
    hits["query"] = hits["query"].astype(str)
    hits["target"] = hits["target"].astype(str)
    hits["pident"] = hits["pident"].astype(float)
    if hits["pident"].max() > 1.5:  # MMseqs2 returns 0..1, but be safe
        hits["pident"] = hits["pident"] / 100.0

    # Apply both thresholds inside the agg (mmseqs --min-seq-id + -c already filter,
    # but be defensive in case the CLI flags shift in future versions).
    valid = hits[(hits["pident"] >= pident_threshold) & (hits["qcov"] >= coverage_threshold)]
    log.info(
        f"hits passing pident≥{pident_threshold:.2f} ∧ qcov≥{coverage_threshold:.2f}: "
        f"{len(valid):,}"
    )

    # Best hit per query by max pident.
    if len(valid):
        idx = valid.groupby("query")["pident"].idxmax()
        best = valid.loc[idx, ["query", "target", "pident", "qcov"]].rename(
            columns={
                "query": "gene_id",
                "target": "best_hit_train_gene",
                "pident": "max_pident",
                "qcov": "best_qcov",
            }
        )
    else:
        best = pd.DataFrame(
            columns=["gene_id", "best_hit_train_gene", "max_pident", "best_qcov"]
        )

    out = pd.DataFrame({"gene_id": query_ids.astype(str)})
    out = out.merge(best, on="gene_id", how="left")
    out["max_pident"] = out["max_pident"].fillna(0.0)
    out["best_hit_train_gene"] = out["best_hit_train_gene"].fillna("")
    out["best_qcov"] = out["best_qcov"].fillna(0.0)
    out["is_contaminated"] = (out["max_pident"] >= pident_threshold) & (
        out["best_qcov"] >= coverage_threshold
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, help="Benchmark name used for output filenames"
    )
    parser.add_argument(
        "--benchmark-parquet",
        type=Path,
        default=None,
        help="Path to benchmark gene catalog parquet (Phase 0.1 mode).",
    )
    parser.add_argument(
        "--query-fasta",
        type=Path,
        default=None,
        help="Path to query FASTA (Phase 2 mode — published tool training sets).",
    )
    parser.add_argument(
        "--pident", type=float, default=DEFAULT_PIDENT, help="Identity threshold (0..1)"
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=DEFAULT_COVERAGE,
        help="Query-coverage threshold (0..1)",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory (default: results/landscape_2026/contamination/)",
    )
    parser.add_argument(
        "--target-fasta-cache",
        type=Path,
        default=None,
        help=(
            "Optional cached FASTA of the 492K target. If not given, a fresh one "
            "is written to a tempfile each run. Cache to disk for repeated runs."
        ),
    )
    args = parser.parse_args()

    if (args.benchmark_parquet is None) == (args.query_fasta is None):
        parser.error("provide exactly one of --benchmark-parquet OR --query-fasta")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.benchmark}__vs_protex492k.json"
    out_parquet = args.out_dir / f"{args.benchmark}__vs_protex492k.parquet"
    out_clean_ids = args.out_dir / f"{args.benchmark}__clean_gene_ids.txt"

    t0 = time.time()

    # Load query
    if args.benchmark_parquet is not None:
        query_ids, query_seqs = load_benchmark(args.benchmark_parquet)
    else:
        query_ids, query_seqs = load_query_fasta(args.query_fasta)

    # Load / prepare target FASTA
    work_dir = Path(tempfile.mkdtemp(prefix="contam_work_"))
    try:
        if args.target_fasta_cache and args.target_fasta_cache.exists():
            target_fasta = args.target_fasta_cache
            log.info(f"using cached target FASTA at {target_fasta}")
        else:
            target_ids, target_seqs = load_target_492k()
            target_fasta = (
                args.target_fasta_cache
                if args.target_fasta_cache
                else (work_dir / "target_492k.fasta")
            )
            if args.target_fasta_cache:
                args.target_fasta_cache.parent.mkdir(parents=True, exist_ok=True)
            write_fasta(target_ids, target_seqs, target_fasta)

        query_fasta = work_dir / "query.fasta"
        n_query_written = write_fasta(query_ids, query_seqs, query_fasta)
        if n_query_written == 0:
            raise ValueError("no valid query sequences written; check input")

        # Search
        hits_tsv = work_dir / "hits.tsv"
        hits = run_mmseqs_search(
            query_fasta,
            target_fasta,
            hits_tsv,
            min_pident=args.pident,
            min_coverage=args.coverage,
            threads=args.threads,
        )

        # Aggregate
        per_gene = aggregate_contamination(
            query_ids,
            hits,
            pident_threshold=args.pident,
            coverage_threshold=args.coverage,
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Save outputs
    per_gene.to_parquet(out_parquet, index=False)

    n_total = len(per_gene)
    n_contam = int(per_gene["is_contaminated"].sum())
    n_clean = n_total - n_contam
    pct_contam = 100.0 * n_contam / max(n_total, 1)

    clean_ids = per_gene.loc[~per_gene["is_contaminated"], "gene_id"].tolist()
    with open(out_clean_ids, "w") as f:
        f.write("\n".join(clean_ids))
        f.write("\n")

    summary = {
        "benchmark": args.benchmark,
        "target_parquet": str(PROD_PARQUET),
        "n_query": int(n_total),
        "n_target": 492026,
        "threshold": {"pident": args.pident, "coverage": args.coverage},
        "n_contaminated": n_contam,
        "n_clean": n_clean,
        "pct_contaminated": round(pct_contam, 4),
        "max_pident_distribution": {
            "p10": float(np.percentile(per_gene["max_pident"], 10)),
            "p50": float(np.percentile(per_gene["max_pident"], 50)),
            "p90": float(np.percentile(per_gene["max_pident"], 90)),
            "p99": float(np.percentile(per_gene["max_pident"], 99)),
            "max": float(per_gene["max_pident"].max()),
        },
        "wall_seconds": round(time.time() - t0, 2),
        "outputs": {
            "per_gene_parquet": str(out_parquet),
            "clean_gene_ids_txt": str(out_clean_ids),
        },
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("─" * 60)
    log.info(f"benchmark              : {args.benchmark}")
    log.info(f"N query                : {n_total:,}")
    log.info(f"N contaminated (≥30%id): {n_contam:,}  ({pct_contam:.2f}%)")
    log.info(f"N clean                : {n_clean:,}")
    log.info(f"per-gene parquet       : {out_parquet}")
    log.info(f"clean gene id list     : {out_clean_ids}")
    log.info(f"summary json           : {out_json}")
    log.info("─" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
