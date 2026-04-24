"""Official Python client for the Aiki-XP public API.

Quick start
-----------
>>> from aikixp_client import Client
>>> client = Client()
>>> # Tier A: protein-only, no host required (fastest)
>>> r = client.predict_tier_a("MKTVRQERLKSIVRILERSK...")
>>> print(r["predictions"][0]["predicted_expression"])
-0.083

>>> # Tier D: full 9-modality prediction (GPU, ~5-90 s)
>>> r = client.predict_tier_d(
...     protein="MKT...",
...     cds="ATG...TAA",
...     host="NC_000913.3",
...     mode="native",
... )

>>> # Look up a gene already in the 492K held-out CV corpus
>>> g = client.lookup_gene("Escherichia_coli_K12|NP_417556.2")
>>> print(g["tier_d"], g["true_expression"])
-0.393 -0.141

>>> # Pull a random sample for reproducing paper numbers
>>> sample = client.sample_lookup(n=5000, seed=42)
>>> import pandas as pd
>>> df = pd.DataFrame(sample["rows"])

See the paper (https://doi.org/10.5281/zenodo.19639621) and
the GitHub README (https://github.com/aikium-public/aiki-xp)
for full details, benchmarks, and limitations.
"""
from ._client import AikixpError, Client
from ._types import GeneLookup, Prediction, PredictionResponse

__all__ = ["Client", "AikixpError", "Prediction", "PredictionResponse", "GeneLookup"]
__version__ = "1.0.0"
