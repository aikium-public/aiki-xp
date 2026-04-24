#!/usr/bin/env python3
"""
XP5 Canonical Inference — THE ONLY CORRECT WAY to run XP5 on new data.

═══════════════════════════════════════════════════════════════════════════
WARNING: Do NOT load FusionModel checkpoints directly in eval scripts.
Use this module instead. It handles:
  1. Z-score normalization of classical features (REQUIRED for models with classical features)
  2. Feature range validation (errors on impossible values)
  3. Sequence form consistency checking
  4. Training-set sanity verification
  5. Tier → recipe mapping

If you bypass this module, your results will be WRONG for any model
that uses classical features (codon, rna_init, protein, disorder, operon).
═══════════════════════════════════════════════════════════════════════════

Usage:
    from aikixp.inference import XP5Ensemble

    model = XP5Ensemble("balanced_nonmega_5mod", device="cpu")
    predictions = model.predict(embedding_dict)
    # embedding_dict: {modality_name: np.ndarray(N, dim)}
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

log = logging.getLogger(__name__)

CKPT_DIR = Path(os.environ.get("AIKIXP_CKPT_DIR", REPO_ROOT / "checkpoints"))
NORM_STATS_PATH = Path(os.environ.get("AIKIXP_NORM_STATS", REPO_ROOT / "configs" / "norm_stats_492k.json"))
TIER_CONFIG_PATH = Path(os.environ.get("AIKIXP_TIER_CONFIG", REPO_ROOT / "configs" / "deployment_tiers.yaml"))


def get_tier_recipe(dataset_name: str) -> str:
    """Look up the correct recipe for a dataset from the tier config.

    Raises ValueError if dataset is not in the config — forces explicit
    tier assignment, preventing ad-hoc model selection (Failure 4).
    """
    import yaml
    if not TIER_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Tier config not found: {TIER_CONFIG_PATH}")
    config = yaml.safe_load(TIER_CONFIG_PATH.read_text())
    dataset_tiers = config.get("dataset_tiers", {})
    if dataset_name not in dataset_tiers:
        raise ValueError(
            f"Dataset '{dataset_name}' not in deployment_tiers.yaml. "
            f"Add it to configs/protex/deployment_tiers.yaml before evaluating. "
            f"Available datasets: {list(dataset_tiers.keys())}"
        )
    tier_name = dataset_tiers[dataset_name]
    tier = config["tiers"][tier_name]
    recipe = tier["recipe"]
    log.info("Dataset '%s' → tier %s → recipe '%s'", dataset_name, tier_name, recipe)
    return recipe


def validate_sequence_form(sequences: list[str]) -> None:
    """Check that sequences are in gene-product form (with M, no MetAP).

    Raises ValueError if sequences appear MetAP-processed or still tagged.
    """
    if not sequences:
        return

    sample = sequences[:1000]
    n_starts_m = sum(1 for s in sample if s and s[0] == "M")
    pct_m = n_starts_m / len(sample)

    # In gene-product form, ~100% should start with M
    # (the few that don't are MetAP-cleaved in vivo, but we keep M for consistency)
    if pct_m < 0.85:
        raise ValueError(
            f"Only {pct_m:.0%} of sequences start with M. "
            f"Expected ~100% for gene-product form. "
            f"Did you apply MetAP cleavage? Use normalize_sequence(seq, apply_metap=False, ensure_m=True). "
            f"See configs/protex/deployment_tiers.yaml §sequence_form."
        )

    # Check for His-tags still present
    n_his = sum(1 for s in sample if "HHHHHH" in s[:30])
    if n_his > len(sample) * 0.05:
        raise ValueError(
            f"{n_his}/{len(sample)} sequences still have His-tags. "
            f"Strip tags before embedding extraction. "
            f"Use normalize_sequence(seq, strip_tags=True, apply_metap=False, ensure_m=True)."
        )

# ═══════════════════════════════════════════════════════════════════════════
# CLASSICAL FEATURE EXPECTED RANGES (from 492K training set)
# If a feature is outside these ranges, the input is likely wrong.
# ═══════════════════════════════════════════════════════════════════════════

CLASSICAL_FEATURE_RANGES = {
    # (modality, feature_index): (min, max, description)
    # Protein biophysical (24d)
    ("classical_protein", 0): (2.0, 14.0, "pI (isoelectric point)"),
    ("classical_protein", 1): (500, 600000, "molecular_weight"),
    ("classical_protein", 2): (-3.0, 3.0, "GRAVY"),
    ("classical_protein", 3): (-100, 200, "instability_index"),
    ("classical_protein", 8): (10, 5000, "protein_length"),
    # Codon features (11d)
    ("classical_codon", 0): (0.0, 1.0, "CAI"),
    ("classical_codon", 1): (20, 62, "ENC (effective number of codons)"),
    ("classical_codon", 8): (0.15, 0.85, "GC content of CDS"),
    # RNA-init features (16d)
    ("classical_rna_init", 7): (-80, 0, "MFE full junction"),
    # Operon struct (10d)
    ("classical_operon_struct", 1): (1, 50, "num_genes_in_operon"),
}

# Modalities that require z-score normalization
CLASSICAL_MODALITIES = {
    "classical_codon", "classical_rna_init", "classical_protein",
    "classical_disorder", "classical_operon_struct",
}


class XP5Ensemble:
    """5-fold ensemble inference with built-in normalization and validation.

    Parameters
    ----------
    recipe : str
        Recipe name (e.g., "balanced_nonmega_5mod").
    device : str
        "cuda" or "cpu".
    norm_stats_path : Path, optional
        Path to norm_stats JSON. Default: results/recipe_5fold_cv/norm_stats_492k.json.
    skip_norm_check : bool
        If True, skip norm_stats requirement for models without classical features.
    """

    def __init__(
        self,
        recipe: str,
        device: str = "cpu",
        norm_stats_path: Optional[Path] = None,
        skip_norm_check: bool = False,
    ):
        self.recipe = recipe
        self.device = device
        self.models = []
        self.input_dims = None
        self.norm_stats = {}
        self.has_classical = False

        # Load checkpoints + per-fold norm_stats
        self.fold_norm_stats = []  # per-fold norm_stats
        for fold in range(5):
            ckpt_path = CKPT_DIR / f"{recipe}_fold{fold}_checkpoint.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if self.input_dims is None:
                self.input_dims = ckpt["input_dims"]
                self.has_classical = bool(
                    set(self.input_dims.keys()) & CLASSICAL_MODALITIES
                )

            from aikixp.train import FusionModel, TrainConfig
            config = ckpt["config"]
            tc = TrainConfig(
                fusion_type=ckpt.get("fusion_type", "single_adapter"),
                latent_dim=config["latent_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config.get("dropout", 0.1),
                num_layers=config.get("num_layers", 2),
            )
            model = FusionModel(tc, self.input_dims)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            self.models.append(model)

            # Extract per-fold norm_stats from checkpoint (GUARDRAIL G1)
            fold_ns = {}
            if "norm_stats" in ckpt and ckpt["norm_stats"]:
                for mod_name, stats in ckpt["norm_stats"].items():
                    if mod_name in self.input_dims:
                        fold_ns[mod_name] = (
                            np.array(stats["mean"], dtype=np.float32),
                            np.array(stats["std"], dtype=np.float32),
                        )
            self.fold_norm_stats.append(fold_ns)

        if not self.models:
            raise FileNotFoundError(
                f"No checkpoints found for recipe '{recipe}' in {CKPT_DIR}"
            )
        log.info("Loaded %d folds for %s (%d modalities, classical=%s)",
                 len(self.models), recipe, len(self.input_dims), self.has_classical)

        # Check norm_stats availability
        if self.has_classical:
            # Prefer per-fold norm_stats from checkpoints (GUARDRAIL G1)
            has_per_fold = all(bool(ns) for ns in self.fold_norm_stats)

            if has_per_fold:
                log.info("  Using per-fold norm_stats from checkpoints (correct)")
            else:
                # Fall back to global norm_stats file (approximate — warns)
                ns_path = norm_stats_path or NORM_STATS_PATH
                if not ns_path.exists():
                    raise FileNotFoundError(
                        f"norm_stats required for {recipe} (has classical features) "
                        f"but not found in checkpoints or at {ns_path}.\n"
                        f"Options:\n"
                        f"  1. Re-train with updated train_fusion.py (saves norm_stats in checkpoint)\n"
                        f"  2. Run: python scripts/protex/compute_norm_stats.py (approximate — uses all 492K)"
                    )
                log.warning(
                    "  Checkpoints lack per-fold norm_stats (trained before GUARDRAIL G1). "
                    "Falling back to global norm_stats from %s. "
                    "This is APPROXIMATE — each fold's training set has slightly different "
                    "mean/std. Re-train to get exact per-fold stats.", ns_path
                )
                raw = json.loads(ns_path.read_text())
                global_ns = {}
                for mod_name, stats in raw.items():
                    if mod_name in self.input_dims:
                        global_ns[mod_name] = (
                            np.array(stats["mean"], dtype=np.float32),
                            np.array(stats["std"], dtype=np.float32),
                        )
                # Use global stats for all folds
                self.fold_norm_stats = [global_ns] * len(self.models)
                self.norm_stats = global_ns
                log.info("  Loaded global norm_stats for: %s", list(global_ns.keys()))
        elif not skip_norm_check:
            log.info("  No classical features — norm_stats not needed")

    def _validate_features(self, mod_arrays: Dict[str, np.ndarray], n: int):
        """Validate feature ranges. Raises ValueError on impossible values."""
        for mod_name, arr in mod_arrays.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"{mod_name}: expected {n} rows, got {arr.shape[0]}"
                )

            expected_dim = self.input_dims.get(mod_name)
            if expected_dim and arr.shape[1] != expected_dim:
                # Strict equality. Prior behaviour used `< expected_dim` here
                # and silently truncated larger inputs via `t[:, :dim]`
                # downstream (see `predict` below). That allowed a mismatched
                # extractor (e.g. Evo-2 7B 11264-d fed into a 1B 5120-d
                # adapter) to produce scientifically invalid predictions
                # labeled as valid. Rule 12 forbids silent fallbacks in
                # metric/data paths; we raise instead.
                raise ValueError(
                    f"{mod_name}: expected dim {expected_dim}, got {arr.shape[1]}. "
                    f"A dim mismatch means the feature extractor disagrees with "
                    f"the checkpoint's adapter. Fix the extractor — do NOT rely "
                    f"on implicit slice/pad."
                )

            # Check for NaN/Inf
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            if n_nan > 0:
                raise ValueError(
                    f"{mod_name}: contains {n_nan} NaN values. "
                    f"Fix the feature computation — do not fill with zeros."
                )
            if n_inf > 0:
                raise ValueError(
                    f"{mod_name}: contains {n_inf} Inf values."
                )

            # Check specific feature ranges for classical modalities
            for (check_mod, feat_idx), (vmin, vmax, desc) in CLASSICAL_FEATURE_RANGES.items():
                if mod_name == check_mod and feat_idx < arr.shape[1]:
                    col = arr[:, feat_idx]
                    col_nonzero = col[col != 0]  # ignore zero-filled
                    if len(col_nonzero) == 0:
                        continue
                    actual_min = col_nonzero.min()
                    actual_max = col_nonzero.max()
                    # Symmetric extremity check: "value is further from zero
                    # than 2× the expected extremum." Works for any signed
                    # range. The previous formula (`vmin * 0.5`) had wrong
                    # semantics for negative vmin — e.g. for GRAVY [-3, 3]
                    # it fired on any value below -1.5, which includes
                    # perfectly normal hydrophilic proteins. See audit
                    # §A.AIKIUM_REEVAL_2026-04-15 for the incident that
                    # surfaced this on dataset 3 (188/1249 scaffolds with
                    # GRAVY < -1.5 tripped the check).
                    lower_bound = vmin - abs(vmin)
                    upper_bound = vmax + abs(vmax)
                    if actual_min < lower_bound or actual_max > upper_bound:
                        raise ValueError(
                            f"{mod_name}[{feat_idx}] ({desc}): "
                            f"range [{actual_min:.2f}, {actual_max:.2f}] "
                            f"outside expected [{vmin}, {vmax}] "
                            f"(extremity bounds [{lower_bound}, {upper_bound}]). "
                            f"Features are likely NOT z-scored or computed wrong. "
                            f"This check runs BEFORE normalization — raw ranges expected."
                        )

    def predict(
        self,
        mod_arrays: Dict[str, np.ndarray],
        batch_size: int = 512,
    ) -> np.ndarray:
        """Run 5-fold ensemble prediction.

        Parameters
        ----------
        mod_arrays : dict
            {modality_name: np.ndarray(N, dim)} for available modalities.
            Missing modalities are zero-filled automatically.
            Classical features must be in RAW form (not pre-z-scored).
        batch_size : int
            Inference batch size.

        Returns
        -------
        np.ndarray of shape (N,) — ensemble-averaged predictions.
        """
        n = next(iter(mod_arrays.values())).shape[0]

        # Check all arrays have the same N
        for mod_name, arr in mod_arrays.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"Array size mismatch: first array has {n} rows, "
                    f"but {mod_name} has {arr.shape[0]} rows"
                )

        # Check for unknown modality names (catch typos)
        unknown = set(mod_arrays.keys()) - set(self.input_dims.keys())
        if unknown:
            raise ValueError(
                f"Unknown modality names: {unknown}. "
                f"Valid modalities for {self.recipe}: {list(self.input_dims.keys())}. "
                f"These would be silently ignored — likely a typo."
            )

        # Validate raw features BEFORE normalization
        self._validate_features(mod_arrays, n)

        # Log what's filled vs zero-filled
        filled = [m for m in self.input_dims if m in mod_arrays]
        zeroed = [m for m in self.input_dims if m not in mod_arrays]
        if zeroed:
            log.info("  Filled: %s", filled)
            log.info("  Zero-filled: %s", zeroed)

        # Per-fold prediction with per-fold normalization
        all_preds = []
        for fold_i, model in enumerate(self.models):
            # Apply THIS fold's norm_stats to classical features
            fold_ns = self.fold_norm_stats[fold_i] if fold_i < len(self.fold_norm_stats) else {}
            normed = {}
            for mod_name, arr in mod_arrays.items():
                if mod_name in fold_ns:
                    mean, std = fold_ns[mod_name]
                    std_safe = np.maximum(std, 1e-8)
                    normed[mod_name] = ((arr - mean) / std_safe).astype(np.float32)
                else:
                    normed[mod_name] = arr

            # Build tensors (zero-fill missing modalities).
            # Strict-equality invariant: dim mismatch should already have
            # raised in `_validate_features`; the belt-and-suspenders check
            # here catches any path that skips validation (defensive). Prior
            # behaviour silently truncated/padded, which let a mismatched
            # extractor produce invalid predictions.
            tensors = {}
            for mod_name, dim in self.input_dims.items():
                if mod_name in normed:
                    t = torch.tensor(normed[mod_name], dtype=torch.float32)
                    if t.shape[1] != dim:
                        raise ValueError(
                            f"{mod_name}: tensor dim {t.shape[1]} != checkpoint "
                            f"expected {dim}. This should have been caught in "
                            f"_validate_features — investigate the calling path."
                        )
                    tensors[mod_name] = t
                else:
                    tensors[mod_name] = torch.zeros(n, dim)

            preds = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    j = min(i + batch_size, n)
                    batch = {
                        k: tensors[k][i:j].to(self.device)
                        for k in self.input_dims
                    }
                    out = model(batch).cpu().numpy().flatten()
                    preds.extend(out)
            all_preds.append(np.array(preds))

        ensemble = np.mean(all_preds, axis=0)

        # Sanity: predictions should be in a reasonable range
        # Training labels are raw expression levels (typically 0-20 range)
        pred_min, pred_max = ensemble.min(), ensemble.max()
        pred_std = ensemble.std()
        if len(ensemble) > 1 and pred_std < 1e-6:
            raise ValueError(
                f"All predictions are identical ({ensemble[0]:.4f}). "
                f"The model is likely receiving garbage input (wrong normalization?)."
            )
        if pred_min < -50 or pred_max > 50:
            raise ValueError(
                f"Predictions wildly out of range: [{pred_min:.2f}, {pred_max:.2f}]. "
                f"This almost certainly indicates a normalization failure. "
                f"Check that classical features are in RAW form (not pre-z-scored) "
                f"and that norm_stats match the training distribution."
            )
        if pred_min < -10 or pred_max > 10:
            log.warning(
                "Predictions outside typical range [-10, 10]: [%.2f, %.2f]. "
                "Check normalization if unexpected.", pred_min, pred_max,
            )

        return ensemble
