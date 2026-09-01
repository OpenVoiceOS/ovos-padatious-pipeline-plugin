[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

# Padatious

Padatious is a neural network intent parser, implemented in pure numpy with a [FANN](https://github.com/libfann/fann)-compatible model format. This repository packages it as an [OpenVoiceOS](https://openvoiceos.org/) (OVOS) pipeline plugin and bundles a maintained fork of the original [padatious](https://github.com/MycroftAI/padatious) from Mycroft AI.

## Features

- Intents are easy to create from a handful of example sentences.
- Each intent trains its own small network, independent of the others.
- Fast training on a small amount of data.
- Entity extraction from a matched sentence (for example, `Find the nearest {place}` matches "Find the nearest gas station" and extracts `place: gas station`).

## Installing

Padatious is pure Python (numpy). It needs no native libraries or compilers.

Install from PyPI:

```bash
pip install ovos-padatious-pipeline-plugin
```

## Direct Usage

```python
from ovos_padatious import IntentContainer

container = IntentContainer('intent_cache')
container.add_intent('hello', ['Hi there!', 'Hello.'])
container.add_intent('goodbye', ['See you!', 'Goodbye!'])
container.add_intent('search', ['Search for {query} (using|on) {engine}.'])
container.train()

print(container.calc_intent('Hello there!'))
print(container.calc_intent('Search for cats on CatTube.'))

container.remove_intent('goodbye')
```

Inside OVOS, the plugin is discovered automatically through its `opm.pipeline` entry point. See [docs/](docs/README.md) for installation details, the intent file syntax, the full Python API, pipeline configuration, and the matching algorithm.

Inside OVOS, training and compiling always run on a background worker, never on the thread that registered or queried something (including the very first pass); a test or tool that registers an intent and needs to query it right away should call `PadatiousPipeline.wait_until_trained()` (see [docs/ovos_pipeline.md](docs/ovos_pipeline.md#training-is-asynchronous)) rather than polling or sleeping.

## Related projects

- [OpenVoiceOS/ovos-spec-tools](https://github.com/OpenVoiceOS/ovos-spec-tools): the reference implementation of the OVOS architecture specifications, used here for sentence-template expansion and language tag handling.
- [OpenVoiceOS/architecture](https://github.com/OpenVoiceOS/architecture): the OpenVoiceOS architecture specifications.
- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): the plugin and entry-point system that loads this pipeline into OVOS.

## License

Licensed under the Apache 2.0 license.
