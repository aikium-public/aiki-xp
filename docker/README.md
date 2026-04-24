# Aiki-XP Docker

## Pre-built images (GitHub Container Registry)

```bash
# Pull (no build step)
docker pull ghcr.io/aikium-public/aiki-xp:inference
docker pull ghcr.io/aikium-public/aiki-xp:full
```

Versioned tags: `:inference-0.1.0`, `:full-0.1.0`. The `:inference` and
`:full` tags always point to the latest release.

## Quick start (inference only)

```bash
# Run Tier A on pre-extracted embeddings
docker run -v /path/to/checkpoints:/app/checkpoints \
           -v /path/to/embeddings:/app/embeddings \
           -v /path/to/output:/app/output \
           ghcr.io/aikium-public/aiki-xp:inference \
           --tier A \
           --embed-dir /app/embeddings \
           --ckpt-dir /app/checkpoints \
           --output /app/output/predictions.csv
```

Or build locally: `docker build -t aikixp:inference .` from the repo root.

## Mount points

| Path | Contents |
|------|----------|
| `/app/checkpoints` | Model checkpoints (from Zenodo) |
| `/app/embeddings` | Pre-extracted embedding parquets (from Zenodo) |
| `/app/output` | Prediction output directory |

## Image sizes

| Image | Size | Use case |
|-------|------|----------|
| `ghcr.io/aikium-public/aiki-xp:inference` | ~2 GB | Pre-extracted embeddings → predictions (CPU-friendly) |
| `ghcr.io/aikium-public/aiki-xp:full` | ~10.7 GB | Raw FASTA → foundation-model embeddings → predictions (GPU required for Tier D) |
