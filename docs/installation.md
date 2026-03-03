# Installation

## System Dependencies

Padatious requires native libraries before the Python package can be installed.

### Ubuntu / Debian

```bash
sudo apt-get install libfann-dev python3-dev python3-pip swig libfann-dev python3-fann2
```

### Arch Linux

```bash
sudo pacman -S fann swig
```

### Fedora / RHEL

```bash
sudo dnf install fann-devel python3-devel swig
```

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
