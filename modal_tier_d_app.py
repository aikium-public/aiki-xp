"""Aiki-XP Tier D deployment on Modal — A100 container for Evo-2 7B + Bacformer-large.

This app is separate from modal_app.py (Tier A CPU) so that:
  (a) GPU containers scale independently from CPU endpoints
  (b) GPU cold-starts don't block fast Tier A requests
  (c) We can iterate on GPU image without redeploying Tier A

Deploy (Monday post-submission):
    modal deploy modal_tier_d_app.py

Status (2026-04-18 evening):
  * Genome cache upload in progress
  * Evo-2 7B image spec drafted (below)
  * Deployment: Monday 2026-04-20 (not Sunday 2026-04-19 submission day)

Canonical Evo-2 recipe source:
  docs/protex/EVO2_ENVIRONMENT_SETUP.md (private repo)
  scripts/protex/setup_evo2_py312.sh (private repo)
  scripts/protex/evo2_7b_multiwindow_extract.py (private repo)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

# ── Evo-2 7B image ────────────────────────────────────────────────────────────
# Pinned versions per docs/protex/EVO2_ENVIRONMENT_SETUP.md:
#   Python 3.12, torch 2.5.0+cu124, flash-attn 2.7.0.post2 prebuilt wheel,
#   evo2 latest, fp8 patch applied for A100 compute capability 8.0

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-"
    "cp312-cp312-linux_x86_64.whl"
)

evo2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "wget")
    # Torch first (specific CUDA version)
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Core deps
    .pip_install(
        "numpy<2.0",
        "pandas>=2.0",
        "scipy>=1.11",
        "biopython>=1.83",
        "tqdm",
        "pyarrow",
        "pyyaml",
        "joblib",
        "scikit-learn>=1.3",
    )
    # Flash-attn prebuilt wheel (source compile fails on Modal's GLIBC)
    .pip_install(FLASH_ATTN_WHEEL)
    # Evo-2 package (pulls vortex / StripedHyena deps)
    .pip_install("evo2")
    # NOTE: Evo-2 1B requires a real Transformer Engine install (vortex calls
    # TE APIs like fixup_te_workspace internally, not just imports). TE's
    # source build fails on Modal's build environment, and stubs only get
    # you past the first import gate. A100-40GB (compute cap 8.0) has no
    # FP8 hardware anyway. Result: Tier B+ — which legitimately needs
    # Evo-2 1B to match the paper's recipe (`tier_b_evo2_init_window_...`)
    # — can't be served cleanly on this Modal deployment. It's disabled on
    # the landing page, and users who need live Tier B+ inference on novel
    # proteins are pointed at the Docker image where TE can be installed
    # on a host with FP8-capable GPUs (H100) or bf16 fallback via a full
    # TE install. Cached Tier B+ CV predictions for the 492K corpus genes
    # remain available via /lookup_gene.
    # FastAPI + transformers for ProtT5/HyenaDNA (NO esm — it upgrades torch and breaks flash-attn)
    # NO bacformer package — it pulls torch 2.11 + transformers 5.x, incompatible with Evo-2
    .pip_install("fastapi[standard]", "sentencepiece", "transformers>=4.36", "accelerate", "einops")
    # Classical features deps: peptides (protein physicochemistry), metapredict (disorder), codon-bias
    .pip_install("peptides", "metapredict", "codon-bias", "localcider")
    # ViennaRNA for classical_rna_init (pip wheel works on linux x86_64)
    .pip_install("ViennaRNA")
    # Final torch re-pin to 2.5.0+cu124 — defensive against any transitive upgrade
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
        force_build=False,
    )
    # ViennaRNA python bindings for classical_rna_init (apt-installed above provides python3-rna)
    # FP8 patch applied at runtime in @modal.enter (transformer_engine may not be present here)
    # Package code
    .add_local_python_source("aikixp")
    .add_local_dir("configs", "/app/configs")
)


# ── Volumes ──────────────────────────────────────────────────────────────────

checkpoints_vol = modal.Volume.from_name("aikixp-checkpoints", create_if_missing=False)
genomes_vol = modal.Volume.from_name("aikixp-genomes", create_if_missing=True)
pca_vol = modal.Volume.from_name("aikixp-pca", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("aikixp-hf-cache", create_if_missing=True)


# ── ESM-C + Bacformer image (separate torch pin per canonical Aikium recipe) ──
# See docs/protex/ESMC_ENVIRONMENT_SETUP.md (private repo).
# Key constraints documented in scripts/setup_esmc_base_env.sh:
#   torch==2.4.0+cu124 (NOT 2.5.0 — SDPA/cuDNN bug with ESM-C on A100)
#   pip install esm --no-deps (prevents torch upgrade to cu130)
#   transformers==4.48.1 (for ESM-2 via HuggingFace)

esmc_bacformer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # ESM-C via --no-deps (canonical Aikium recipe)
    .run_commands("pip install --no-deps esm")
    # ESM-C's transitive deps that we install manually (no torch upgrade)
    .pip_install(
        "einops", "zstd", "biotite", "cloudpathlib", "msgpack_numpy", "biopython",
        "httpx", "tenacity", "attrs", "msgpack", "pillow", "brotli",
    )
    .pip_install("transformers==4.48.1")
    .pip_install(
        "numpy<2.0", "pandas>=2.0", "scipy>=1.11", "scikit-learn>=1.3",
        "joblib", "pyarrow", "pyyaml", "tqdm", "accelerate",
        "fastapi[standard]",
        # Bacformer's bacformer.pp module imports these at module-load time
        "datasets", "safetensors", "dataclasses_json",
        "lightning", "omegaconf", "hydra-core",
    )
    # Bacformer — pin older version that doesn't force torch upgrade
    .run_commands("pip install --no-deps bacformer==0.1.1 || pip install --no-deps bacformer")
    # Defensive torch re-pin in case anything else tried to upgrade
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("aikixp")
    .add_local_dir("configs", "/app/configs")
)


# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App("aikixp-tier-d")


# ── ESM-C + Bacformer class (A100-40GB, smaller models than Evo-2) ──────────

@app.cls(
    image=esmc_bacformer_image,
    volumes={"/genomes": genomes_vol, "/models": hf_cache_vol},
    gpu="A100-40GB",
    memory=32 * 1024,
    scaledown_window=120,
    timeout=600,
)
class AikixpEmbeddings:
    """Complementary extraction: ESM-C (1152d protein) + Bacformer-large (960d genome).

    Runs in a separate container from Evo-2 because the two foundation models have
    irreconcilable torch version requirements (Aikium's documented two-env pattern).
    """

    @modal.enter()
    def startup(self):
        import os, sys
        os.environ["HF_HOME"] = "/models"
        sys.path.insert(0, "/app")

        import torch
        self.device = torch.device("cuda")
        print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")

        # Monkey-patch torch._dynamo.config for bacformer compatibility on torch < 2.6
        # bacformer sets `torch._dynamo.config.recompile_limit = 16` but the attr
        # only exists in torch >= 2.6. We patch the class-level __setattr__ to fall
        # back to a plain dict write when the guarded setter rejects a new attr.
        try:
            from torch._dynamo import config as _dc
            _cm_cls = type(_dc)
            if not getattr(_cm_cls, "_aikixp_patched", False):
                _orig = _cm_cls.__setattr__
                def _safe_setattr(self, name, value, _orig=_orig):
                    try:
                        _orig(self, name, value)
                    except AttributeError:
                        object.__setattr__(self, name, value)
                _cm_cls.__setattr__ = _safe_setattr
                _cm_cls._aikixp_patched = True
                # Smoke test
                _dc.recompile_limit = 16
                print(f"  monkey-patched {_cm_cls.__module__}.{_cm_cls.__name__}.__setattr__ "
                      f"(recompile_limit={_dc.recompile_limit})")
        except Exception as _e:
            print(f"  torch._dynamo patch skipped: {type(_e).__name__}: {_e}")

        # Load ESM-C once per container
        print("Loading ESM-C 600M...")
        from esm.models.esmc import ESMC
        self.esmc = ESMC.from_pretrained("esmc_600m", device=self.device).eval()
        self.esmc_tokenizer = self.esmc.tokenizer
        print(f"  ESM-C ready on {torch.cuda.get_device_name(0)}")

        # Try to load Bacformer-large (may fail if package has bugs)
        self.bacformer = None
        self.bacformer_tokenizer = None
        try:
            from transformers import AutoModel, AutoTokenizer
            model_id = "macwiatrak/bacformer-large-masked-complete-genomes"
            print(f"Loading Bacformer-large from {model_id}...")
            self.bacformer = (
                AutoModel.from_pretrained(model_id, trust_remote_code=True)
                .to(self.device).eval()
            )
            print("  Bacformer-large loaded")
        except Exception as e:
            print(f"  Bacformer load failed: {type(e).__name__}: {e}")

    @modal.method()
    def extract_esmc(self, sequences: list[str]) -> list[list[float]]:
        """ESM-C 600M mean-pooled embeddings, 1152d per sequence."""
        import numpy as np
        import torch

        embs = np.zeros((len(sequences), 1152), dtype=np.float32)
        batch = 8
        with torch.no_grad():
            for i in range(0, len(sequences), batch):
                seqs = sequences[i:i+batch]
                toks = self.esmc_tokenizer.batch_encode_plus(
                    seqs, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                )
                input_ids = toks["input_ids"].to(self.device)
                attn = toks["attention_mask"].clone().to(self.device)
                attn[:, 0] = 0
                attn[input_ids == 2] = 0
                out = self.esmc(input_ids)
                hidden = out.embeddings
                mask = attn.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                embs[i:i+batch] = pooled.float().cpu().numpy()
        return embs.tolist()

    def _bacformer_progress_path(self, host_accession: str):
        import pathlib
        d = pathlib.Path("/genomes/_bacformer_progress")
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{host_accession}.json"

    def _write_bacformer_progress(self, host_accession: str, **fields) -> None:
        """Write a small JSON progress file that the landing /genome_status
        route polls while a Bacformer cache is being built. Kept
        volume-resident so it survives container churn.
        """
        import json, time
        path = self._bacformer_progress_path(host_accession)
        payload = {"ts": time.time(), **fields}
        try:
            path.write_text(json.dumps(payload))
        except OSError as e:
            print(f"bacformer progress write failed: {e}")

    def _build_bacformer_cache(self, host_accession: str):
        """Build the per-proteome Bacformer cache for host_accession.

        Writes /genomes/_bacformer_cache/{acc}.npy and emits progress JSON
        to /genomes/_bacformer_progress/{acc}.json. Returns the numpy array
        on success; raises on failure (caller translates to response JSON).
        """
        import json, pickle, pathlib, traceback
        import numpy as np
        import torch

        cache_dir = pathlib.Path("/genomes/_bacformer_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{host_accession}.npy"

        host_path = pathlib.Path("/genomes") / f"{host_accession}.pkl"
        if not host_path.exists():
            raise FileNotFoundError(f"genome pickle missing: {host_accession}")

        import sys
        sys.path.insert(0, "/app")
        from aikixp.genome_lookup import load_genome
        genome = load_genome(host_path)

        cds_feats = sorted(
            [f for f in genome.features
             if f.type == "CDS" and "translation" in f.qualifiers and "pseudo" not in f.qualifiers],
            key=lambda f: int(f.location.start),
        )
        proteins = [f.qualifiers["translation"][0] for f in cds_feats][:6000]
        n_proteins = len(proteins)
        print(f"BACFORMER: extracting proteome of {n_proteins} proteins for {host_accession}")
        self._write_bacformer_progress(
            host_accession,
            status="running", phase="forward_pass", n_proteins=n_proteins,
        )

        try:
            import bacformer.pp as bpp
            if not hasattr(bpp, "protein_seqs_to_bacformer_inputs"):
                helpers = [a for a in dir(bpp)
                           if not a.startswith("_") and callable(getattr(bpp, a, None))]
                raise RuntimeError(
                    f"bacformer.pp missing protein_seqs_to_bacformer_inputs; available: {helpers}"
                )
            inputs = bpp.protein_seqs_to_bacformer_inputs(
                [proteins],
                bacformer_model_type="large",
                max_n_proteins=6000,
                device="cuda",
            )
            with torch.no_grad():
                out = self.bacformer(**inputs)
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            else:
                hidden = out[0] if isinstance(out, tuple) else out
            embs = hidden[0].float().cpu().numpy()
            if embs.shape[0] > n_proteins:
                embs = embs[1:1 + n_proteins]
            np.save(cache_path, embs)
            print(f"BACFORMER: cached shape={embs.shape} -> {cache_path}")
            # Commit so the landing ASGI sees the .npy on its next /genome_status read.
            try:
                genomes_vol.commit()
            except Exception as e:
                print(f"BACFORMER: volume commit warning: {e}")
            self._write_bacformer_progress(
                host_accession,
                status="done", n_proteins=n_proteins,
                shape=list(embs.shape),
            )
            return embs
        except Exception as e:
            tb = traceback.format_exc()
            print(f"BACFORMER: extraction failed\n{tb}")
            self._write_bacformer_progress(
                host_accession,
                status="failed", error=f"{type(e).__name__}: {e}",
            )
            raise

    @modal.method()
    def extract_bacformer(self, host_accession: str, cds_indices: list) -> dict:
        """Run Bacformer-large on host genome's proteome, return per-gene 960d vectors."""
        import pathlib
        import numpy as np

        if self.bacformer is None:
            return {"embeddings": [[0.0] * 960 for _ in cds_indices], "n_available": 0,
                    "error": "Bacformer model not loaded"}

        cache_path = pathlib.Path(f"/genomes/_bacformer_cache/{host_accession}.npy")
        if cache_path.exists():
            proteome_embs = np.load(cache_path)
            print(f"BACFORMER: cache hit {cache_path.name}, shape={proteome_embs.shape}")
        else:
            try:
                proteome_embs = self._build_bacformer_cache(host_accession)
            except Exception as e:
                return {"embeddings": [[0.0] * 960 for _ in cds_indices], "n_available": 0,
                        "error": f"{type(e).__name__}: {e}"}

        result_embs = []
        n_available = 0
        for idx in cds_indices:
            if idx is not None and idx < len(proteome_embs):
                result_embs.append(proteome_embs[idx].tolist())
                n_available += 1
            else:
                result_embs.append([0.0] * 960)
        print(f"BACFORMER: returning {n_available}/{len(cds_indices)} available")
        return {"embeddings": result_embs, "n_available": n_available}

    @modal.method()
    def precompute_bacformer(self, host_accession: str) -> dict:
        """Build the Bacformer-large cache for a newly-added genome.

        Spawned in the background by /request_genome when the user opts in to
        pre-compute. Writes /genomes/_bacformer_cache/{acc}.npy and a progress
        JSON file the landing ASGI exposes via /genome_status.
        """
        import pathlib
        cache_path = pathlib.Path(f"/genomes/_bacformer_cache/{host_accession}.npy")
        if cache_path.exists():
            self._write_bacformer_progress(
                host_accession, status="done", phase="already_cached",
            )
            return {"status": "already_cached", "acc": host_accession}

        if self.bacformer is None:
            self._write_bacformer_progress(
                host_accession, status="failed", error="Bacformer model not loaded",
            )
            return {"status": "failed", "acc": host_accession,
                    "error": "Bacformer model not loaded in container"}

        try:
            embs = self._build_bacformer_cache(host_accession)
        except Exception as e:
            return {"status": "failed", "acc": host_accession,
                    "error": f"{type(e).__name__}: {e}"}
        return {"status": "done", "acc": host_accession,
                "n_proteins": int(embs.shape[0]), "dim": int(embs.shape[1])}


@app.cls(
    image=evo2_image,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/genomes": genomes_vol,
        "/pca": pca_vol,
    },
    gpu="A100-80GB",      # Evo-2 7B needs 80GB for long operons up to 64 kb
    memory=64 * 1024,     # 64 GB RAM for genome manipulation
    scaledown_window=120, # 2 min idle before spin-down (GPU is expensive)
    timeout=1200,         # 20 min per request max
)
class AikixpTierD:
    """Tier D GPU inference: operon DNA + genome context → champion prediction."""

    EVO2_MODEL_NAME = "evo2_7b"
    EVO2_LAYER = "blocks.28.mlp.l1"
    EVO2_HIDDEN_DIM = 11264

    # Evo-2 1B — canonical source for the `evo2_init_window` modality
    # (the paper's Tier B+ recipe was trained on Evo-2 1B, 5120-d, per
    # CLAUDE.md Rule 18 and configs/deployment_tiers.yaml). Kept separate
    # from the 7B config so there's no silent substitution ever again.
    EVO2_1B_MODEL_NAME = "evo2_1b_base"
    EVO2_1B_LAYER = "blocks.16.mlp.l1"
    EVO2_1B_HIDDEN_DIM = 5120
    EVO2_MAX_LEN = 65536

    @modal.enter()
    def startup(self):
        """Lightweight container startup.

        Evo-2 7B + ProtT5 + PCA are lazily loaded on first Tier C/D call
        (via `_ensure_evo2_ready()`), so Tier B requests never incur the
        ~7 GB HuggingFace download cost. This drops Tier B cold-start
        from ~10 min to ~30-60 s.
        """
        import os, sys
        os.environ["AIKIXP_CKPT_DIR"] = "/checkpoints"
        os.environ["AIKIXP_TIER_CONFIG"] = "/app/configs/deployment_tiers.yaml"
        os.environ["AIKIXP_NORM_STATS"] = "/checkpoints/norm_stats_492k.json"
        sys.path.insert(0, "/app")
        import torch
        self.device = torch.device("cuda")
        self._evo2_ready = False
        print(f"AikixpTierD minimal startup on {torch.cuda.get_device_name(0)}")

    def _ensure_evo2_ready(self):
        """Lazy-load Evo-2 7B + PCA on first Tier C/D call."""
        if self._evo2_ready:
            return
        import joblib
        import torch
        from evo2 import Evo2

        # FP8 autocast patch for A100 compute capability 8.0 (fp8 requires 8.9+)
        try:
            import transformer_engine.pytorch as tep
            from contextlib import contextmanager
            @contextmanager
            def _no_fp8(*args, **kwargs):
                yield
            tep.fp8_autocast = _no_fp8
            if hasattr(tep, "quantization"):
                tep.quantization.fp8_autocast = _no_fp8
                if hasattr(tep.quantization, "autocast"):
                    tep.quantization.autocast = _no_fp8
            print("  fp8_autocast patched for A100")
        except ImportError:
            print("  transformer_engine not present — fp8 patch skipped (bf16 path)")

        print(f"Lazy-loading Evo-2 7B on {torch.cuda.get_device_name(0)}...")
        self.evo2 = Evo2(self.EVO2_MODEL_NAME)
        self.tokenizer = self.evo2.tokenizer
        test_ids = torch.tensor(
            self.tokenizer.tokenize("ATGCATGC" * 10), dtype=torch.int
        ).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, out = self.evo2(
                test_ids, return_embeddings=True, layer_names=[self.EVO2_LAYER]
            )
        probe = out[self.EVO2_LAYER][0]
        assert probe.shape[-1] == self.EVO2_HIDDEN_DIM, (
            f"Evo-2 7B layer {self.EVO2_LAYER} hidden_dim={probe.shape[-1]}, "
            f"expected {self.EVO2_HIDDEN_DIM} — wrong checkpoint?"
        )
        print(f"  Evo-2 7B ready: layer={self.EVO2_LAYER}, dim={self.EVO2_HIDDEN_DIM}")
        self.pca = joblib.load("/pca/evo2_7b_full_operon_pca4096.pkl")
        print(f"  PCA (full_operon) loaded: {self.pca.n_components_} components")
        self._evo2_ready = True

    def _ensure_init_pca_ready(self):
        """DEPRECATED: the 7B init_70nt PCA is from the retired Tier B+ R2
        contaminated recipe. The paper's clean Tier B+ uses raw Evo-2 1B
        5120-d init-window, no PCA. Kept only so old callers don't crash."""
        if getattr(self, "_init_pca", None) is not None:
            return
        import joblib
        self._init_pca = joblib.load("/pca/evo2_7b_init_70nt_pca4096.pkl")
        print(f"  PCA (init_70nt, retired-recipe) loaded: {self._init_pca.n_components_} components")

    # Evo-2 1B loading is not possible on the current Modal image (TE
    # source build fails; stubs only pass the first-import gate). Any call
    # that reaches `_evo2_embed_init_window` raises; the landing page UI
    # disables Tier B+ to avoid triggering it. See the comment block near
    # the image builder at the top of this file.

    def _evo2_embed_raw(self, sequences):
        """Internal: raw (pre-PCA) Evo-2 7B embedding extraction. Triggers lazy model load."""
        self._ensure_evo2_ready()
        import numpy as np
        import torch
        import re

        sanitized = [
            re.sub(r"[^ATGCN]", "N", s.upper().replace("U", "T"))
            for s in sequences
        ]

        raw = []
        for seq in sanitized:
            if not seq:
                raw.append(np.zeros(self.EVO2_HIDDEN_DIM, dtype=np.float32))
                continue
            token_ids = self.tokenizer.tokenize(seq)
            if len(token_ids) > self.EVO2_MAX_LEN:
                token_ids = token_ids[: self.EVO2_MAX_LEN]
            ids_tensor = torch.tensor(token_ids, dtype=torch.int).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _, out = self.evo2(
                    ids_tensor, return_embeddings=True, layer_names=[self.EVO2_LAYER]
                )
            hidden = out[self.EVO2_LAYER][0]
            raw.append(hidden.mean(dim=0).float().cpu().numpy())
        return np.vstack(raw)

    def _evo2_embed(self, sequences):
        """Evo-2 7B on full-operon DNA, PCA-reduced to 4096d."""
        import numpy as np
        raw = self._evo2_embed_raw(sequences)
        return self.pca.transform(raw).astype(np.float32)

    def _evo2_embed_init_window(self, sequences):
        """Disabled on this Modal deployment. Raises with a clear message.

        The paper's Tier B+ recipe (`tier_b_evo2_init_window_...`) requires
        Evo-2 1B 5120-d raw features. Evo-2 1B's loader hard-depends on
        Transformer Engine (vortex's model internals call TE APIs at init
        time, not just `import`). TE's source build fails inside Modal's
        image build environment, and stubs only pass the first import gate
        before vortex calls TE functions directly.

        For Tier B+ on novel proteins, install Transformer Engine via the
        Docker image (`ghcr.io/aikium-public/aiki-xp:full`) or a local H100
        + TE setup and run the extractor there.

        The prior implementation fed Evo-2 7B 11264-d raw, which XP5Ensemble
        silently truncated to the first 5120 channels — producing invalid
        predictions. Fixed to RAISE rather than silently mis-extract.
        """
        raise RuntimeError(
            "Tier B+ live inference is not available on this Modal deployment — "
            "Evo-2 1B requires Transformer Engine, which doesn't build cleanly "
            "on the current Modal image. For novel proteins, use the Docker "
            "image (ghcr.io/aikium-public/aiki-xp:full) or the Zenodo archive. "
            "Cached Tier B+ CV predictions for the 492K corpus genes remain "
            "available via /lookup_gene."
        )

    @modal.method()
    def extract_evo2_7b_operon(self, operon_sequences: list[str]) -> list[list[float]]:
        """Extract PCA-reduced Evo-2 7B embeddings for operon DNA. Returns 4096-d vectors."""
        return self._evo2_embed(operon_sequences).tolist()

    @modal.method()
    def predict_tier_c(
        self,
        protein_sequences: list[str],
        cds_sequences: list[str],
        host_accession: str,
        mode: str = "heterologous",
        anchor_locus_tag: str = "lacZ",
    ) -> dict:
        """End-to-end Tier C prediction for protein + CDS in a host context.

        Tier C recipe (evo2_prott5_seed42): Evo-2 7B full_operon + ProtT5-XL.
        Achieves rho_nc=0.576 on non-conserved genes (paper).

        For native mode: attempts to find CDS in the host genome, uses real operon.
        For heterologous mode: wraps CDS in host's anchor (lacZ) chromosomal context.
        """
        import numpy as np
        import pickle
        import pandas as pd
        import sys
        sys.path.insert(0, "/app")
        from aikixp.genome_lookup import lookup_native_gene, synthesize_heterologous_context

        # 1. Load host genome
        import pathlib
        host_path = pathlib.Path("/genomes") / f"{host_accession}.pkl"
        if not host_path.exists():
            return {"error": f"Host genome not found: {host_accession}"}
        from aikixp.genome_lookup import load_genome
        genome = load_genome(host_path)

        # 2. Build gene contexts (operon DNA + everything else)
        contexts = []
        for prot, cds in zip(protein_sequences, cds_sequences):
            if mode == "native":
                ctx = lookup_native_gene(genome, cds, protein_sequence=prot)
                if ctx is None:
                    ctx = synthesize_heterologous_context(
                        genome, cds, anchor_locus_tag, protein_sequence=prot
                    )
            else:
                ctx = synthesize_heterologous_context(
                    genome, cds, anchor_locus_tag, protein_sequence=prot
                )
            contexts.append(ctx)

        # 3. Extract Evo-2 7B on operon DNA (expensive GPU call)
        operon_seqs = [c.full_operon_dna for c in contexts]
        evo2_embs = self._evo2_embed(operon_seqs)  # (N, 4096)

        # 4. Extract ProtT5-XL on the CPU (it's loaded in the startup of another class;
        #    for this function we just zero-fill protein modality and rely on operon signal,
        #    OR call the sibling CPU app). For simplicity in v1, call HF transformers inline.
        protein_embs = self._extract_prott5_inline(protein_sequences)  # (N, 1024)

        # 5. XP5 inference with Tier C recipe
        from aikixp.inference import XP5Ensemble
        tier_c = XP5Ensemble("evo2_prott5_seed42", device="cpu")
        predictions = tier_c.predict({
            "evo2_7b_full_operon_pca4096": evo2_embs,
            "prot_t5_xl_protein": protein_embs,
        })

        return {
            "mode": mode,
            "host": host_accession,
            "tier": "C",
            "recipe": "evo2_prott5_seed42",
            "n_sequences": len(protein_sequences),
            "predictions": [
                {
                    "predicted_expression": float(p),
                    "operon_source": c.mode,
                    "operon_length_nt": len(c.full_operon_dna),
                    "cds_start_in_operon": c.cds_start_in_operon,
                }
                for p, c in zip(predictions, contexts)
            ],
        }

    # ── Tier B+: ProtT5 + Evo-2 init-70nt window + classical_rna_init ──

    @modal.method()
    def predict_tier_b_plus(
        self,
        protein_sequences: list[str],
        cds_sequences: list[str],
        host_accession: str,
        mode: str = "heterologous",
        anchor_locus_tag: str = "lacZ",
    ) -> dict:
        """End-to-end Tier B+ prediction: ProtT5 + Evo-2 init-window + classical_rna_init.

        Recipe: tier_b_evo2_init_window_classical_rna_init_prott5_seed42
        (paper rho_nc=0.555 on non-conserved genes). Adds a 60-nt window
        around the ATG to Tier B's protein-only pipeline.
        """
        import numpy as np
        import pickle
        import pathlib
        import sys
        sys.path.insert(0, "/app")
        from aikixp.genome_lookup import lookup_native_gene, synthesize_heterologous_context

        host_path = pathlib.Path("/genomes") / f"{host_accession}.pkl"
        if not host_path.exists():
            return {"error": f"Host genome not found: {host_accession}"}
        from aikixp.genome_lookup import load_genome
        genome = load_genome(host_path)

        contexts = []
        for prot, cds in zip(protein_sequences, cds_sequences):
            if mode == "native":
                ctx = lookup_native_gene(genome, cds, protein_sequence=prot)
                if ctx is None:
                    ctx = synthesize_heterologous_context(
                        genome, cds, anchor_locus_tag, protein_sequence=prot
                    )
            else:
                ctx = synthesize_heterologous_context(
                    genome, cds, anchor_locus_tag, protein_sequence=prot
                )
            contexts.append(ctx)

        # Evo-2 on the init window (60 nt). rna_init_window_seq is stored as RNA
        # by genome_lookup; convert U -> T for Evo-2.
        init_dna = [ctx.rna_init_window_seq.replace("U", "T") for ctx in contexts]
        evo2_init = self._evo2_embed_init_window(init_dna)    # (N, 4096)

        # ProtT5 on protein
        protein_embs = self._extract_prott5_inline(protein_sequences)  # (N, 1024)

        # classical_rna_init via the existing helper (it computes all 5 classical blocks;
        # we pick only rna_init for this recipe).
        classical = self._compute_classical_features(contexts)

        feed = {
            "prot_t5_xl_protein": protein_embs,
            "evo2_init_window": evo2_init,
        }
        if "classical_rna_init" in classical and classical["classical_rna_init"] is not None:
            feed["classical_rna_init"] = classical["classical_rna_init"]

        from aikixp.inference import XP5Ensemble
        tier_b_plus = XP5Ensemble(
            "tier_b_evo2_init_window_classical_rna_init_prott5_seed42", device="cpu"
        )
        predictions = tier_b_plus.predict(feed)

        return {
            "mode": mode,
            "host": host_accession,
            "tier": "B+",
            "recipe": "tier_b_evo2_init_window_classical_rna_init_prott5_seed42",
            "modalities_filled": list(feed.keys()),
            "n_sequences": len(protein_sequences),
            "predictions": [
                {
                    "predicted_expression": float(p),
                    "operon_source": c.mode,
                    "operon_length_nt": len(c.full_operon_dna),
                    "cds_start_in_operon": c.cds_start_in_operon,
                    "init_window_nt": len(c.rna_init_window_seq),
                }
                for p, c in zip(predictions, contexts)
            ],
        }

    # ── Tier B: ProtT5 + 5 classical biophysical feature blocks (no operon DNA) ──

    @modal.method()
    def predict_tier_b(
        self,
        protein_sequences: list[str],
        cds_sequences: list[str],
        host_accession: str,
        mode: str = "heterologous",
        anchor_locus_tag: str = "lacZ",
    ) -> dict:
        """End-to-end Tier B prediction: ProtT5 + 5 classical biophysical blocks.

        Recipe: deploy_protein_cds_features_6mod_seed42 (paper rho_nc=0.531).
        No Evo-2, no Bacformer — faster than Tier C/D but requires the CDS
        and the host genome (for operon-structural features).
        """
        import numpy as np
        import pickle
        import pathlib
        import sys
        sys.path.insert(0, "/app")
        from aikixp.genome_lookup import lookup_native_gene, synthesize_heterologous_context

        host_path = pathlib.Path("/genomes") / f"{host_accession}.pkl"
        if not host_path.exists():
            return {"error": f"Host genome not found: {host_accession}"}
        from aikixp.genome_lookup import load_genome
        genome = load_genome(host_path)

        contexts = []
        for prot, cds in zip(protein_sequences, cds_sequences):
            if mode == "native":
                ctx = lookup_native_gene(genome, cds, protein_sequence=prot)
                if ctx is None:
                    ctx = synthesize_heterologous_context(
                        genome, cds, anchor_locus_tag, protein_sequence=prot
                    )
            else:
                ctx = synthesize_heterologous_context(
                    genome, cds, anchor_locus_tag, protein_sequence=prot
                )
            contexts.append(ctx)

        # Tier B recipe expects: esmc_protein, hyenadna_dna_cds, codonfm_cds,
        # classical_{codon, protein, disorder}. CodonFM isn't on Modal yet —
        # XP5Ensemble will zero-fill it. ESM-C must be extracted in the sibling
        # container (different torch version pin).
        import numpy as np
        hyenadna_embs = self._extract_hyenadna_inline([c.dna_cds_seq for c in contexts])
        classical = self._compute_classical_features(contexts)
        try:
            sibling = AikixpEmbeddings()
            esmc_result = sibling.extract_esmc.remote(protein_sequences)
            esmc_embs = np.array(esmc_result, dtype=np.float32)
        except Exception as e:
            print(f"  ESM-C sibling call failed: {type(e).__name__}: {e}")
            esmc_embs = None

        feed = {}
        if esmc_embs is not None:
            feed["esmc_protein"] = esmc_embs
        if hyenadna_embs is not None:
            feed["hyenadna_dna_cds"] = hyenadna_embs
        for k in ("classical_codon", "classical_protein", "classical_disorder"):
            if k in classical and classical[k] is not None:
                feed[k] = classical[k]

        from aikixp.inference import XP5Ensemble
        tier_b = XP5Ensemble("deploy_protein_cds_features_6mod_seed42", device="cpu")
        predictions = tier_b.predict(feed)

        return {
            "mode": mode,
            "host": host_accession,
            "tier": "B",
            "recipe": "deploy_protein_cds_features_6mod_seed42",
            "modalities_filled": list(feed.keys()),
            "n_sequences": len(protein_sequences),
            "predictions": [
                {
                    "predicted_expression": float(p),
                    "operon_source": c.mode,
                    "operon_length_nt": len(c.full_operon_dna),
                    "cds_start_in_operon": c.cds_start_in_operon,
                }
                for p, c in zip(predictions, contexts)
            ],
        }

    # ── Tier D champion: Evo-2 7B + Bacformer + ESM-C + (others zero-filled) ──

    @modal.method()
    def predict_tier_d(
        self,
        protein_sequences: list[str],
        cds_sequences: list[str],
        host_accession: str,
        mode: str = "heterologous",
        anchor_locus_tag: str = "lacZ",
        return_attribution: bool = False,
    ) -> dict:
        """End-to-end Tier D champion prediction.

        Uses balanced_nonmega_5mod checkpoint with modality dropout ('nm5_moddropout_0.2_seed42')
        which tolerates zero-filling of modalities not fully extracted here.

        REAL modalities extracted: evo2_7b_full_operon_pca4096, bacformer_large, esmc_protein
        ZERO-FILLED modalities: hyenadna_dna_cds, classical_codon, classical_rna_init,
                                classical_protein, classical_disorder, classical_operon_struct

        Paper ρ_nc champion: 0.592 (full 9 modalities)
        This partial-modality version: expected ρ_nc in the 0.55-0.58 range based on
        paper's modality-dropout ablations.
        """
        import numpy as np
        import pickle
        import sys
        import pathlib
        sys.path.insert(0, "/app")
        from aikixp.genome_lookup import lookup_native_gene, synthesize_heterologous_context

        # 1. Load host genome
        host_path = pathlib.Path("/genomes") / f"{host_accession}.pkl"
        if not host_path.exists():
            return {"error": f"Host genome not found: {host_accession}"}
        from aikixp.genome_lookup import load_genome
        genome = load_genome(host_path)

        # 2. Build gene contexts
        contexts = []
        cds_indices = []
        for prot, cds in zip(protein_sequences, cds_sequences):
            if mode == "native":
                ctx = lookup_native_gene(genome, cds, protein_sequence=prot)
                if ctx is None:
                    ctx = synthesize_heterologous_context(
                        genome, cds, anchor_locus_tag, protein_sequence=prot
                    )
            else:
                ctx = synthesize_heterologous_context(
                    genome, cds, anchor_locus_tag, protein_sequence=prot
                )
            contexts.append(ctx)
            cds_indices.append(ctx.genome_cds_index)

        # 3. Evo-2 7B on operon DNA
        operon_seqs = [c.full_operon_dna for c in contexts]
        evo2_embs = self._evo2_embed(operon_seqs)

        # 4. HyenaDNA on CDS (small model, ~25s cold, ~1s warm)
        hyenadna_embs = self._extract_hyenadna_inline([c.dna_cds_seq for c in contexts])

        # 5. Classical features (codon 11d, protein 24d, disorder 8d, operon_struct 10d, rna_init 16d)
        classical_embs = self._compute_classical_features(contexts)

        # 6. ESM-C + Bacformer in the sibling container (torch 2.4 + esm --no-deps).
        #    Two separate Modal classes because of documented torch version conflict.
        esmc_embs = None
        bacformer_embs = None
        try:
            sibling = AikixpEmbeddings()
            esmc_result = sibling.extract_esmc.remote(protein_sequences)
            esmc_embs = np.array(esmc_result, dtype=np.float32)
            cds_idx_list = [c.genome_cds_index for c in contexts]
            bf_result = sibling.extract_bacformer.remote(host_accession, cds_idx_list)
            if bf_result and "embeddings" in bf_result:
                bacformer_embs = np.array(bf_result["embeddings"], dtype=np.float32)
                if not bacformer_embs.any():
                    bacformer_embs = None  # treat as unavailable
        except Exception as e:
            print(f"  AikixpEmbeddings sibling call failed: {type(e).__name__}: {e}")

        # 7. XP5 inference with dropout-robust Tier D recipe
        from aikixp.inference import XP5Ensemble
        tier_d = XP5Ensemble("balanced_nonmega_5mod", device="cpu")
        feed = {"evo2_7b_full_operon_pca4096": evo2_embs}
        modalities_filled = ["evo2_7b_full_operon_pca4096"]
        if hyenadna_embs is not None:
            feed["hyenadna_dna_cds"] = hyenadna_embs
            modalities_filled.append("hyenadna_dna_cds")
        if esmc_embs is not None:
            feed["esmc_protein"] = esmc_embs
            modalities_filled.append("esmc_protein")
        if bacformer_embs is not None and bacformer_embs.any():
            feed["bacformer_large"] = bacformer_embs
            modalities_filled.append("bacformer_large")
        for mod_name, arr in classical_embs.items():
            if arr is not None:
                feed[mod_name] = arr
                modalities_filled.append(mod_name)
        predictions = tier_d.predict(feed)

        # Optional per-modality attribution via leave-one-out ablation.
        # Cheap: re-runs only the CPU fusion head (cheap ~100ms), not the
        # expensive GPU extractors. Total added cost for 5 modalities
        # filled = ~0.5s. We interpret each score as "how much this modality
        # added to the full prediction" = z_full - z_without_this_modality.
        attribution = None
        if return_attribution and len(modalities_filled) >= 2:
            attribution = []
            full_z = [float(p) for p in predictions]
            for i_seq in range(len(protein_sequences)):
                attrs = {}
                for m in modalities_filled:
                    reduced_feed = {k: v for k, v in feed.items() if k != m}
                    if not reduced_feed:
                        attrs[m] = full_z[i_seq]  # sole modality
                        continue
                    reduced_preds = tier_d.predict(reduced_feed)
                    attrs[m] = float(full_z[i_seq] - float(reduced_preds[i_seq]))
                attribution.append(attrs)

        return {
            "mode": mode,
            "host": host_accession,
            "tier": "D",
            "recipe": "balanced_nonmega_5mod",
            "modalities_filled": modalities_filled,
            "attribution": attribution,
            "modalities_zero_filled": [
                m for m in [
                    "esmc_protein", "bacformer_large", "classical_rna_init",
                    "hyenadna_dna_cds", "classical_codon", "classical_protein",
                    "classical_disorder", "classical_operon_struct",
                ] if m not in modalities_filled
            ],
            "n_sequences": len(protein_sequences),
            "predictions": [
                {
                    "predicted_expression": float(p),
                    "operon_source": c.mode,
                    "operon_length_nt": len(c.full_operon_dna),
                    "cds_start_in_operon": c.cds_start_in_operon,
                    "bacformer_available": c.genome_cds_index is not None,
                }
                for p, c in zip(predictions, contexts)
            ],
        }

    def _extract_hyenadna_inline(self, dna_sequences):
        """HyenaDNA-large-1M extraction on CDS DNA. 256-d per sequence."""
        import re
        import numpy as np
        import torch

        if not hasattr(self, "_hyenadna"):
            print("Loading HyenaDNA-large-1M (first call)...")
            try:
                from transformers import AutoModel, AutoTokenizer
                model_id = "LongSafari/hyenadna-large-1m-seqlen-hf"
                self._hyenadna_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self._hyenadna = (
                    AutoModel.from_pretrained(model_id, trust_remote_code=True)
                    .to(self.device).eval()
                )
                print("  HyenaDNA ready")
            except Exception as e:
                print(f"  HyenaDNA load failed: {type(e).__name__}: {e}")
                return None

        iupac_re = re.compile(r"[^ATGCN]")
        sanitized = [iupac_re.sub("N", s.upper().replace("U", "T")) for s in dna_sequences]

        embs = np.zeros((len(sanitized), 256), dtype=np.float32)
        try:
            with torch.no_grad():
                for i, seq in enumerate(sanitized):
                    if not seq:
                        continue
                    toks = self._hyenadna_tok(
                        seq, return_tensors="pt", padding=True, truncation=True, max_length=32000,
                    ).to(self.device)
                    out = self._hyenadna(**toks)
                    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    attn = toks.get("attention_mask")
                    if attn is not None:
                        mask = attn.unsqueeze(-1).float()
                        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    else:
                        pooled = hidden.mean(dim=1)
                    embs[i] = pooled.float().cpu().numpy()[0]
            return embs
        except Exception as e:
            print(f"  HyenaDNA extraction failed: {type(e).__name__}: {e}")
            return None

    def _compute_classical_features(self, contexts):
        """Compute the 4 classical feature blocks that don't need ViennaRNA.

        Returns a dict of modality_name -> np.ndarray.
        Uses aikixp.classical_features compute functions, built on one-row dataframes.
        """
        import numpy as np
        import pandas as pd
        import sys
        sys.path.insert(0, "/app")

        try:
            from aikixp import classical_features as cf
        except Exception as e:
            print(f"  Could not load classical_features module: {e}")
            return {}

        # Build a dataframe with the columns each feature computer expects
        rows = []
        for ctx in contexts:
            rows.append({
                "gene_id": f"g{len(rows)}",
                "species": "Escherichia_coli_K12",  # codon-bias defaults to E. coli
                "taxid": 83333,
                "protein_sequence": ctx.protein_sequence,
                "protein_sequence_paxdb": ctx.protein_sequence,  # legacy column name used by classical_features
                "dna_cds_seq": ctx.dna_cds_seq,
                "full_operon_dna": ctx.full_operon_dna,
                "rna_init_window_seq": ctx.rna_init_window_seq,
                "cds_start_in_operon": ctx.cds_start_in_operon,
                "gene_end_rel": ctx.cds_start_in_operon + len(ctx.dna_cds_seq),
                "dna_cds_len": len(ctx.dna_cds_seq),
                "full_operon_length": len(ctx.full_operon_dna),
                "operon_source": "demo",
                "operon_id": "demo_operon",
            })
        df = pd.DataFrame(rows)

        results = {}

        def _run(name, fn, expected_dim):
            try:
                out_df = fn(df)
                feature_cols = [c for c in out_df.columns if c != "gene_id"]
                print(f"  {name}: output columns = {feature_cols[:5]}... ({len(feature_cols)} total)")
                if len(feature_cols) < expected_dim:
                    print(f"  {name}: SKIP — got {len(feature_cols)} cols, need {expected_dim}")
                    return None
                arr = out_df[feature_cols[:expected_dim]].to_numpy(dtype=np.float32)
                # Replace NaN/Inf with 0 (classical features occasionally produce NaN for edge cases)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if arr.shape[1] != expected_dim:
                    print(f"  {name}: SKIP — array shape {arr.shape} != expected {expected_dim}")
                    return None
                print(f"  {name}: OK shape={arr.shape}")
                return arr
            except Exception as e:
                import traceback
                print(f"  {name} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                return None

        arr = _run("classical_codon", cf.compute_codon_features, 11)
        if arr is not None:
            results["classical_codon"] = arr
        arr = _run("classical_protein", cf.compute_protein_features, 24)
        if arr is not None:
            results["classical_protein"] = arr
        arr = _run("classical_disorder", cf.compute_disorder_features, 8)
        if arr is not None:
            results["classical_disorder"] = arr
        arr = _run("classical_operon_struct", cf.compute_operon_structural_features, 10)
        if arr is not None:
            results["classical_operon_struct"] = arr
        # rna_init needs ViennaRNA — compute_rna_thermo_features is the right one
        try:
            arr = _run(
                "classical_rna_init",
                lambda df_: cf.compute_rna_thermo_features(df_, use_vienna=True),
                16,
            )
            if arr is not None:
                results["classical_rna_init"] = arr
        except Exception as e:
            print(f"  classical_rna_init skipped: {e}")

        return results

    def _extract_esmc_inline(self, sequences):
        """ESM-C 600M extraction. Cached model after first call."""
        import numpy as np
        import torch

        if not hasattr(self, "_esmc"):
            print("Loading ESM-C 600M (first call)...")
            from esm.models.esmc import ESMC
            self._esmc = ESMC.from_pretrained("esmc_600m", device=self.device).eval()
            self._esmc_tokenizer = self._esmc.tokenizer

        embs = np.zeros((len(sequences), 1152), dtype=np.float32)
        batch = 8
        with torch.no_grad():
            for i in range(0, len(sequences), batch):
                seqs = sequences[i : i + batch]
                toks = self._esmc_tokenizer.batch_encode_plus(
                    seqs, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                )
                input_ids = toks["input_ids"].to(self.device)
                attn = toks["attention_mask"].clone().to(self.device)
                attn[:, 0] = 0
                attn[input_ids == 2] = 0
                out = self._esmc(input_ids)
                hidden = out.embeddings
                mask = attn.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                embs[i : i + batch] = pooled.float().cpu().numpy()
        return embs

    def _extract_prott5_inline(self, sequences):
        """Lightweight ProtT5-XL extraction inside the GPU container.

        Loads ProtT5 on first call and caches; 1024-d embeddings.
        """
        import numpy as np
        import torch
        from transformers import T5Tokenizer, T5EncoderModel

        if not hasattr(self, "_prott5"):
            print("Loading ProtT5-XL (first call)...")
            self._prott5_tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50", do_lower_case=False, legacy=False
            )
            # Force safetensors to avoid torch 2.5 torch.load security restriction
            self._prott5 = (
                T5EncoderModel.from_pretrained(
                    "Rostlab/prot_t5_xl_uniref50",
                    use_safetensors=True,
                )
                .to(self.device).half().eval()
            )

        embs = np.zeros((len(sequences), 1024), dtype=np.float32)
        batch = 4
        with torch.no_grad():
            for i in range(0, len(sequences), batch):
                seqs = sequences[i : i + batch]
                spaced = [
                    " ".join(list(s.replace("U", "X").replace("Z", "X")
                                   .replace("O", "X").replace("*", "")))
                    for s in seqs
                ]
                ids = self._prott5_tokenizer(
                    spaced, add_special_tokens=True, padding="longest", return_tensors="pt"
                ).to(self.device)
                out = self._prott5(input_ids=ids.input_ids, attention_mask=ids.attention_mask)
                hidden = out.last_hidden_state
                mask = ids.attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                embs[i : i + batch] = pooled.float().cpu().numpy()

        return embs


# ── CLI entrypoint for testing ────────────────────────────────────────────────

@app.local_entrypoint()
def test_evo2_extraction():
    """Spin up the A100 container and verify Evo-2 7B extraction works.

    Usage:  modal run modal_tier_d_app.py::test_evo2_extraction
    Cost:   ~$1-2 (A100 warmup + 30 s of extraction)
    """
    # Three test operons of varying lengths to exercise the pipeline
    test_operons = [
        "ATGAAACGCATTGGATCTGCA" * 50,   # short, ~1 kb
        "GCATGCATGCATGCATGCAT" * 250,   # medium, ~5 kb
        "ATCGATCGATCGATCGATCG" * 1000,  # longer, ~20 kb (tests batching)
    ]
    lengths = [len(s) for s in test_operons]
    print(f"Testing with {len(test_operons)} operons of lengths {lengths}")

    tier_d = AikixpTierD()
    import time
    t0 = time.time()
    embeddings = tier_d.extract_evo2_7b_operon.remote(test_operons)
    elapsed = time.time() - t0

    print(f"\n=== RESULTS ===")
    print(f"Total wall time: {elapsed:.1f}s (includes A100 cold start)")
    print(f"Returned: {len(embeddings)} embeddings")
    if embeddings:
        import numpy as np
        arr = np.array(embeddings)
        print(f"Shape: {arr.shape} (expected {len(test_operons)} x 4096)")
        print(f"Value range: [{arr.min():.3f}, {arr.max():.3f}], mean={arr.mean():.3f}")
        print(f"Non-zero rows: {(arr != 0).any(axis=1).sum()}")


# ── HTTP endpoint for Tier C ─────────────────────────────────────────────────

import fastapi


@app.function(image=evo2_image, timeout=1200)
@modal.fastapi_endpoint(method="POST")
async def predict_tier_d_endpoint(request: "fastapi.Request") -> dict:
    """POST: protein FASTA + CDS FASTA + host accession → Tier D champion prediction.

    Body JSON (same schema as Tier C, invokes dropout-robust Tier D recipe):
      {
        "proteins": ["MKT...", "MLE..."],
        "cds": ["ATG...", "ATG..."],
        "host": "NC_000913.3",
        "mode": "heterologous" (default) or "native",
        "anchor": "lacZ" (default)
      }
    """
    payload = await request.json()
    proteins = payload.get("proteins", [])
    cds = payload.get("cds", [])
    host = payload.get("host", "NC_000913.3")
    mode = payload.get("mode", "heterologous")
    anchor = payload.get("anchor", "lacZ")
    return_attribution = bool(payload.get("return_attribution", False))

    if not proteins or not cds or len(proteins) != len(cds):
        return {"error": f"Provide matched 'proteins' and 'cds' lists ({len(proteins)} vs {len(cds)})"}

    tier_d = AikixpTierD()
    return tier_d.predict_tier_d.remote(proteins, cds, host, mode, anchor, return_attribution)


@app.function(image=evo2_image, timeout=1200)
@modal.fastapi_endpoint(method="POST")
async def predict_tier_c_endpoint(request: "fastapi.Request") -> dict:
    """POST: protein FASTA + CDS FASTA (matched order) + host accession.

    Body JSON:
      {
        "proteins": ["MKT...", "MLE..."],
        "cds": ["ATG...", "ATG..."],
        "host": "NC_000913.3",
        "mode": "heterologous" (default) or "native",
        "anchor": "lacZ" (default, used for heterologous)
      }
    """
    payload = await request.json()
    proteins = payload.get("proteins", [])
    cds = payload.get("cds", [])
    host = payload.get("host", "NC_000913.3")  # default E. coli K12
    mode = payload.get("mode", "heterologous")
    anchor = payload.get("anchor", "lacZ")

    if not proteins or not cds:
        return {"error": "Provide 'proteins' and 'cds' as matched lists"}
    if len(proteins) != len(cds):
        return {"error": f"Mismatch: {len(proteins)} proteins vs {len(cds)} CDS"}

    tier_d = AikixpTierD()
    return tier_d.predict_tier_c.remote(proteins, cds, host, mode, anchor)


@app.function(image=evo2_image, timeout=1200)
@modal.fastapi_endpoint(method="POST")
async def predict_tier_b_endpoint(request: "fastapi.Request") -> dict:
    """POST: protein FASTA + CDS (matched order) + host accession.

    Tier B recipe (deploy_protein_cds_features_6mod_seed42): ProtT5 + 5
    classical biophysical blocks. No Evo-2, no Bacformer.
    Paper rho_nc = 0.531 on non-conserved genes.

    Body JSON:
      {
        "proteins": ["MKT...", ...],
        "cds": ["ATG...", ...],
        "host": "NC_000913.3",
        "mode": "heterologous" (default) or "native",
        "anchor": "lacZ" (default)
      }
    """
    payload = await request.json()
    proteins = payload.get("proteins", [])
    cds = payload.get("cds", [])
    host = payload.get("host", "NC_000913.3")
    mode = payload.get("mode", "heterologous")
    anchor = payload.get("anchor", "lacZ")

    if not proteins or not cds or len(proteins) != len(cds):
        return {"error": f"Provide matched 'proteins' and 'cds' lists ({len(proteins)} vs {len(cds)})"}

    tier_d = AikixpTierD()
    return tier_d.predict_tier_b.remote(proteins, cds, host, mode, anchor)


@app.function(image=evo2_image, timeout=1200)
@modal.fastapi_endpoint(method="POST")
async def predict_tier_b_plus_endpoint(request: "fastapi.Request") -> dict:
    """POST: Tier B+ live-inference endpoint — currently returns a clean
    refusal with a redirect to the Docker / Zenodo path.

    The paper's Tier B+ recipe
    (`tier_b_evo2_init_window_classical_rna_init_prott5_seed42`, ρ_nc=0.543)
    was trained on Evo-2 1B 5120-d raw init-window embeddings. Evo-2 1B
    requires Transformer Engine for loading, which doesn't build cleanly
    on Modal's image; and A100-40GB has no FP8 hardware anyway. Rather
    than serve a silent dim-substituted prediction labelled as Tier B+,
    this endpoint refuses and points users at the valid paths.

    Cached held-out CV Tier B+ predictions for the 492K corpus genes
    remain available via /lookup_gene.
    """
    return {
        "error": "tier_b_plus_live_inference_unavailable",
        "message": (
            "Tier B+ live inference is not available on this Modal deployment. "
            "The paper's Tier B+ recipe requires Evo-2 1B with Transformer Engine, "
            "which doesn't build cleanly on the current Modal image. For live "
            "Tier B+ inference on novel proteins, use the Docker image "
            "(ghcr.io/aikium-public/aiki-xp:full) with a TE install on an "
            "H100 or a bf16-capable setup. Cached held-out CV Tier B+ "
            "predictions for the 492K corpus genes remain available via "
            "/lookup_gene."
        ),
        "alternatives": {
            "corpus_cv_lookup": "https://aikium--aikixp-tier-a-lookup-gene.modal.run",
            "docker_image": "ghcr.io/aikium-public/aiki-xp:full",
            "zenodo_archive": "https://doi.org/10.5281/zenodo.19639621",
            "paper_recipe_ref": "tier_b_evo2_init_window_classical_rna_init_prott5_seed42",
        },
        "predictions": [],
    }


# ── Placeholder endpoints — to be filled out Monday ──────────────────────────
# TODO (Monday):
#   - POST /extract_tier_d: genome_accession + cds + host → full Tier D prediction
#   - POST /extract_bacformer: genome pickle + CDS index → 960-d Bacformer embedding
#   - Classical feature computation (ViennaRNA in image or separate CPU class)
#   - Orchestrator that assembles all 9 modalities and calls XP5Ensemble
#
# For now, AikixpTierD.extract_evo2_7b_operon is the Evo-2 7B primitive.
# All other primitives are pending.
