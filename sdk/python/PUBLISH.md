# How to publish `aikixp-client` to PyPI

Not automated in CI yet — do this from a workstation with PyPI credentials
configured via `~/.pypirc` or the `TWINE_USERNAME` / `TWINE_PASSWORD` env vars
(token-based auth recommended: `TWINE_USERNAME=__token__`,
`TWINE_PASSWORD=pypi-<your-token>`).

## First-time setup

1. Reserve the package name on PyPI and TestPyPI by uploading an initial
   release there. The name `aikixp-client` is currently unclaimed.
2. Create an API token scoped to the `aikixp-client` project (after the
   first upload lets you scope it, a global token is fine for the initial one).

## Release recipe

```bash
cd sdk/python

# 1. bump the version in pyproject.toml and aikixp_client/__init__.py
# 2. update CHANGELOG.md
# 3. clean + rebuild
rm -rf dist/ build/ *.egg-info/
python -m build

# 4. test on TestPyPI first
python -m twine upload -r testpypi dist/*
python -m pip install -i https://test.pypi.org/simple/ aikixp-client==X.Y.Z

# 5. if that works, upload to the real PyPI
python -m twine upload dist/*

# 6. tag the release in git
git tag -a aikixp-client-vX.Y.Z -m "aikixp-client vX.Y.Z"
git push origin aikixp-client-vX.Y.Z
```

## Version policy

- The SDK version tracks the public API version, not the paper version.
- Breaking API changes bump the major.
- New endpoints or optional parameters bump the minor.
- Bug fixes and docs bump the patch.

## Verifying a release

After `pip install aikixp-client==X.Y.Z`:

```python
from aikixp_client import Client
c = Client()
assert c.lookup_gene("Escherichia_coli_K12|NP_417556.2")["n_found"] == 1
assert len(c.hosts()) >= 1800
```
