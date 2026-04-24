#!/usr/bin/env python3
"""Full end-to-end: FASTA -> embeddings -> predictions.

Usage:
    python scripts/extract_and_predict.py \
        --fasta proteins.fasta \
        --tier A \
        --ckpt-dir checkpoints/ \
        --output predictions.csv \
        --cache-dir /tmp/embeddings
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aikixp.extract import extract_tier_a_embeddings, parse_fasta
from aikixp.inference import XP5Ensemble
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TIER_TO_RECIPE = {
    "A": "esmc_prott5_seed42",
}


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and predict")
    parser.add_argument("--fasta", required=True, type=Path, help="Input FASTA")
    parser.add_argument("--tier", default="A", choices=["A"],
                        help="Deployment tier (only A supported in end-to-end mode currently)")
    parser.add_argument("--ckpt-dir", type=Path, help="Checkpoint directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/aikixp_cache"),
                        help="Where to save extracted embeddings")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()

    if args.ckpt_dir:
        import os
        os.environ["AIKIXP_CKPT_DIR"] = str(args.ckpt_dir)

    log.info("Reading %s...", args.fasta)
    gene_ids, sequences = parse_fasta(args.fasta)
    log.info("  %d sequences", len(gene_ids))

    log.info("Extracting Tier %s embeddings (device=%s)...", args.tier, args.device)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = extract_tier_a_embeddings(sequences, gene_ids, args.cache_dir, device=args.device)
    log.info("  Cached at %s", args.cache_dir)

    log.info("Loading XP5Ensemble...")
    model = XP5Ensemble(TIER_TO_RECIPE[args.tier], device=args.device)

    log.info("Loading embeddings for inference...")
    arrays = {}
    for mod, path in paths.items():
        df = pd.read_parquet(path).set_index("gene_id").reindex(gene_ids)
        col = [c for c in df.columns if c != "gene_id"][0]
        arr = np.stack(df[col].values).astype(np.float32)
        arrays[mod] = arr

    predictions = model.predict(arrays)

    out_df = pd.DataFrame({"gene_id": gene_ids, "predicted_expression": predictions})
    out_df = out_df.sort_values("predicted_expression", ascending=False)
    out_df["rank"] = range(1, len(out_df) + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    log.info("Wrote %d predictions to %s", len(out_df), args.output)
    log.info("  Range: [%.3f, %.3f], std=%.3f",
             predictions.min(), predictions.max(), predictions.std())


if __name__ == "__main__":
    main()
