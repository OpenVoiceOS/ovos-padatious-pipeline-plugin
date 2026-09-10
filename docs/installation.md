# Installation

Padatious is pure Python (numpy). It needs no native libraries or compilers.

## Python Package

Install from PyPI:

```bash
pip install ovos-padatious-pipeline-plugin
```

Or install directly from source:

```bash
git clone https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin
cd ovos-padatious-pipeline-plugin
pip install -e .
```

## Optional Dependencies

| Package | Purpose |
|---|---|
| `snowballstemmer` | Word stemming for better multilingual matching |
| `langcodes` | Language tag normalization and closest-match lookup |

These are pulled in automatically as transitive dependencies when used inside OVOS.

## Verifying the Installation

```python
from ovos_padatious import IntentContainer
container = IntentContainer()
container.add_intent('test', ['hello world'])
container.train()
print(container.calc_intent('hello world'))
```

If no exceptions are raised and a `MatchData` object is printed, the installation is working correctly.

---
[Home](README.md) · [Intent Format →](intent_format.md)
