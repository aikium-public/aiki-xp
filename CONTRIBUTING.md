# Contributing to Aiki-XP

Thanks for your interest. Aiki-XP is actively maintained by [Aikium](https://aikium.com);
we welcome community contributions across three lanes.

## Ways to contribute

### 1. File a useful issue
The [issue templates](.github/ISSUE_TEMPLATE/) cover three cases:
- **Bug report** — Aiki-XP returned a wrong result or crashed.
- **Scientific question** — about the model, training, benchmarks, or an interpretation.
- **Host request** — your bacterial organism isn't in our 1,831-genome cache; we'll add it.

Before filing, please scan the [FAQ](README.md#troubleshooting--faq). Many questions are
already answered there.

### 2. Send a pull request
Good first PRs:
- Documentation improvements (typos, clarifications, code-comment fixes).
- Extra example notebooks beyond the Colab quickstart.
- Additional unit tests.
- Host-specific quality-of-life scripts (e.g. converters for other genome formats).

PRs that touch the scientific path — training, featurization, inference, metrics, or data
loading — will be reviewed carefully. In particular:
- No silent fallbacks on metric computation, data selection, or model selection.
- No overwriting frozen artifacts in place — use a new versioned path.
- Keep any new CDS-index mappings aligned with the extractor's filtered CDS space (pseudogenes excluded).

### 3. Cite and build on the work
If Aiki-XP helps your research, please cite the paper (see
[README.md#citation](README.md#citation)) and tell us what you built at
[venkatesh@aikium.com](mailto:venkatesh@aikium.com). We maintain an informal list of
downstream users and will link back where appropriate.

## Development workflow

### Local setup
```bash
git clone https://github.com/aikium-public/aiki-xp.git
cd aiki-xp
pip install -e .
```

### Running the smoke test
The pre-launch health-check covers every public endpoint:
```bash
bash scripts/prelaunch_smoke.sh
```

### Reproducing the headline numbers
See [README.md#reproduce-the-papers-numbers](README.md#reproduce-the-papers-numbers).
The Colab quickstart reproduces the per-tier non-mega Spearman ρ against the manuscript's
headline values in about 3 minutes on a free Colab CPU runtime.

## Ground rules

- **Code license:** Apache 2.0. Contributions accepted under the same license.
- **Data / model-weight license:** CC-BY 4.0 via the
  [Zenodo deposit](https://doi.org/10.5281/zenodo.19639621). Contributions that
  add new data must clarify licensing up front.
- **Sign-off:** use `git commit -s` or include `Signed-off-by: Your Name <email>`
  to acknowledge the [Developer Certificate of Origin](https://developercertificate.org/).
- **Be kind.** We enforce the [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
  as a code of conduct. Bad-faith contributions and harassment earn a ban.

## Where to ask a question
- Scientific / use-case questions → [GitHub Discussions](https://github.com/aikium-public/aiki-xp/discussions)
- Bug reports → [GitHub Issues](https://github.com/aikium-public/aiki-xp/issues) (use the bug-report template)
- Commercial / partnership inquiries → [venkatesh@aikium.com](mailto:venkatesh@aikium.com)
- Security issues → **do not** open a public issue; email [security@aikium.com](mailto:security@aikium.com).
