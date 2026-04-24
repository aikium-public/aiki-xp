"""Extract ESM-C and ProtT5-XL embeddings from protein sequences.

Produces the same embedding format as the pre-extracted parquets on Zenodo,
so the output can feed directly into predict.py.

Usage:
    from aikixp.extract import extract_tier_a_embeddings
    extract_tier_a_embeddings(
        sequences=["MKTVRQERLK", "MNIFEMLRID"],
        gene_ids=["gene_1", "gene_2"],
        output_dir="output/embeddings/",
        device="cpu",
    )
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def _save_parquet(gene_ids: List[str], embeddings: np.ndarray, name: str, out_dir: Path) -> Path:
    """Save embeddings in the same format as the Zenodo parquets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_embeddings.parquet"
    df = pd.DataFrame({
        "gene_id": gene_ids,
        f"{name}_embedding": [e for e in embeddings],
    })
    df.to_parquet(path, index=False)
    log.info("Saved %d x %dd embeddings -> %s", len(gene_ids), embeddings.shape[1], path.name)
    return path


def extract_esmc(
    sequences: List[str],
    gene_ids: List[str],
    out_dir: Path,
    device: str = "cpu",
    batch_size: int = 8,
) -> Path:
    """Extract ESM-C 600M embeddings (1152d, mean-pooled)."""
    log.info("ESM-C: loading model (600M params)...")
    try:
        from esm.models.esmc import ESMC
    except ImportError:
        raise ImportError(
            "ESM-C not installed. pip install esm (EvolutionaryScale package)"
        )

    torch_device = torch.device(device)
    model = ESMC.from_pretrained("esmc_600m", device=torch_device).eval()
    tokenizer = model.tokenizer

    embs = np.zeros((len(sequences), 1152), dtype=np.float32)
    log.info("ESM-C: extracting %d sequences (batch=%d)...", len(sequences), batch_size)

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            toks = tokenizer.batch_encode_plus(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=2048,
            )
            input_ids = toks["input_ids"].to(torch_device)
            attn = toks["attention_mask"].clone().to(torch_device)
            attn[:, 0] = 0
            attn[input_ids == 2] = 0

            out = model(input_ids)
            hidden = out.embeddings
            mask = attn.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs[i:i + batch_size] = pooled.float().cpu().numpy()

            if ((i // batch_size) + 1) % 20 == 0 or i + batch_size >= len(sequences):
                log.info("  ESM-C: %d/%d", min(i + batch_size, len(sequences)), len(sequences))

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return _save_parquet(gene_ids, embs, "esmc_protein", out_dir)


def extract_prott5(
    sequences: List[str],
    gene_ids: List[str],
    out_dir: Path,
    device: str = "cpu",
    batch_size: int = 4,
) -> Path:
    """Extract ProtT5-XL embeddings (1024d, mean-pooled)."""
    log.info("ProtT5-XL: loading model (3B params)...")
    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        raise ImportError("transformers not installed. pip install transformers sentencepiece")

    torch_device = torch.device(device)
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False, legacy=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(torch_device).eval()
    if device == "cuda":
        model = model.half()

    embs = np.zeros((len(sequences), 1024), dtype=np.float32)
    log.info("ProtT5-XL: extracting %d sequences (batch=%d)...", len(sequences), batch_size)

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            # ProtT5 expects space-separated residues, rare AAs → X
            spaced = [
                " ".join(list(s.replace("U", "X").replace("Z", "X").replace("O", "X").replace("*", "")))
                for s in batch
            ]
            ids = tokenizer.batch_encode_plus(
                spaced, add_special_tokens=True, padding="longest", return_tensors="pt"
            ).to(torch_device)
            out = model(input_ids=ids.input_ids, attention_mask=ids.attention_mask)
            hidden = out.last_hidden_state
            mask = ids.attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs[i:i + batch_size] = pooled.float().cpu().numpy()

            if ((i // batch_size) + 1) % 20 == 0 or i + batch_size >= len(sequences):
                log.info("  ProtT5: %d/%d", min(i + batch_size, len(sequences)), len(sequences))

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return _save_parquet(gene_ids, embs, "prot_t5_xl_protein", out_dir)


def extract_tier_a_embeddings(
    sequences: List[str],
    gene_ids: List[str],
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Path]:
    """Extract both ESM-C and ProtT5 embeddings for Tier A inference."""
    from .sequence_normalization import normalize_sequence

    log.info("Normalizing %d sequences (strip tags, ensure M)...", len(sequences))
    normed = []
    for s in sequences:
        clean, _ = normalize_sequence(s, strip_tags=True, apply_metap=False, ensure_m=True)
        normed.append(clean)

    esmc_path = extract_esmc(normed, gene_ids, output_dir, device)
    prott5_path = extract_prott5(normed, gene_ids, output_dir, device)
    return {"esmc_protein": esmc_path, "prot_t5_xl_protein": prott5_path}


def parse_fasta(path: Path) -> tuple[List[str], List[str]]:
    """Parse a FASTA file into (gene_ids, sequences)."""
    gene_ids, sequences = [], []
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    gene_ids.append(current_id)
                    sequences.append("".join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            gene_ids.append(current_id)
            sequences.append("".join(current_seq))
    return gene_ids, sequences
