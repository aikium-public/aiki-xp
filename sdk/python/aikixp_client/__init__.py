"""Official Python client for the Aiki-XP public API.

Quick start
-----------
>>> from aikixp_client import Client
>>> client = Client()
>>> # Tier A: protein-only, no host required (fastest, CPU)
>>> r = client.predict_tier_a("MKTVRQERLKSIVRILERSK...")
>>> print(r["predictions"][0]["predicted_expression"])
-0.083

>>> # Tier D: full 5-modality prediction, the paper's XP5 champion (GPU, ~5-90 s)
>>> r = client.predict_tier_d(
...     protein="MKT...",
...     cds="ATG...TAA",
...     host="NC_000913.3",
...     mode="native",
... )

>>> # Pull a random sample of held-out CV predictions for reproducing paper numbers
>>> sample = client.sample_lookup(n=5000, seed=42)
>>> import pandas as pd
>>> df = pd.DataFrame(sample["rows"])

The paper's full per-tier predictions for all 244,002 corpus genes are on
Zenodo (https://doi.org/10.5281/zenodo.19639621). The intermediate tiers
B / B+ / C from the paper are scientific control conditions; their cached
predictions live in the Zenodo deposit, not as live endpoints.

See the paper preprint (https://doi.org/10.64898/2026.04.21.719525) and
the GitHub README (https://github.com/aikium-public/aiki-xp) for full
details, benchmarks, and limitations.
"""
from ._client import AikixpError, Client
from ._types import Prediction, PredictionResponse

__all__ = ["Client", "AikixpError", "Prediction", "PredictionResponse"]
__version__ = "1.1.0"
