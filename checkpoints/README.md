# Aiki-XP Model Checkpoints

Trained model checkpoints for all deployment tiers are hosted on Zenodo.

**DOI:** `10.5281/zenodo.XXXXXXX`

## Download

```bash
zenodo_get 10.5281/zenodo.XXXXXXX -o checkpoints/
```

## Available checkpoints

Each recipe has 5 fold checkpoints (`*_fold{0..4}_checkpoint.pt`).

| Tier | Recipe | Modalities | In-dist. rho_nc |
|------|--------|------------|-----------------|
| A | `esmc_prott5_seed42` | 2 | 0.518 |
| B | `deploy_protein_cds_features_6mod_seed42` | 6 | 0.530 |
| B+ | `tier_b_evo2_init_window_classical_rna_init_prott5_seed42` | 9 | 0.543 |
| C | `evo2_prott5_seed42` | 2 | 0.576 |
| D | `balanced_nonmega_5mod` | 9 | 0.592 |

## Usage

```python
from aikixp.inference import XP5Ensemble

model = XP5Ensemble("esmc_prott5_seed42", device="cpu")
predictions = model.predict({"esmc_protein": esmc_arr, "prot_t5_xl_protein": prott5_arr})
```

## Checkpoint contents

Each `.pt` file contains:
- `model_state_dict`: trained weights
- `config`: architecture hyperparameters (latent_dim, hidden_dim, dropout, num_layers)
- `input_dims`: {modality_name: dimension} mapping
- `norm_stats`: per-fold z-score normalization statistics for classical features
- `fusion_type`: architecture type (single_adapter)
