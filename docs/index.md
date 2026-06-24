# ovos-padatious-pipeline-plugin

An efficient neural network intent parser for the [OpenVoiceOS](https://openvoiceos.org/) (OVOS) ecosystem, powered by [FANN](https://github.com/libfann/fann).

This repository bundles a fork of the original [padatious](https://github.com/MycroftAI/padatious) from Mycroft AI and exposes it as an OVOS pipeline plugin.

## Documentation

| Document | Description |
|---|---|
| [Installation](installation.md) | How to install dependencies and the package |
| [Intent Format](intent_format.md) | Syntax for writing `.intent` and `.entity` files |
| [API Reference](api_reference.md) | Python API for `IntentContainer` and `DomainIntentContainer` |
| [OVOS Pipeline](ovos_pipeline.md) | Using the plugin inside OVOS / configuration options |
| [Architecture](architecture.md) | Internal architecture and data flow |
| [Theory](theory.md) | Algorithm design, decisions, and limitations |

## Spec conformance

Sentence-template expansion and language tag handling are delegated to
[`ovos-spec-tools`](https://github.com/OpenVoiceOS/ovos-spec-tools), the
reference implementation of the
[OpenVoiceOS architecture specifications](https://github.com/OpenVoiceOS/architecture).
The bracket grammar (`(a|b)`, `[optional]`, `{slot}`) is expanded by
`ovos_spec_tools.expand` per **OVOS-INTENT-1**, and language tags are
normalised/matched with `standardize_lang`/`closest_lang`. The plugin's
own `bracket_expansion` helpers remain as deprecated shims that forward
to `ovos_spec_tools.expand`.

## Quick Example

```python
from ovos_padatious import IntentContainer

container = IntentContainer('intent_cache')
container.add_intent('hello', ['Hi there!', 'Hello.'])
container.add_intent('search', ['Search for {query} (using|on) {engine}.'])
container.train()

result = container.calc_intent('Search for cats on CatTube.')
print(result.name)    # 'search'
print(result.matches) # {'query': 'cats', 'engine': 'CatTube'}
```

## License

Licensed under the **Apache 2.0** license.

> **Note:** This plugin is an exception to the [OVOS universal donor policy](https://openvoiceos.github.io/ovos-technical-manual/license/).
> It depends on `fann2`, which is licensed under the LGPL. See the [license compatibility note](https://softwareengineering.stackexchange.com/questions/119436/what-does-gpl-with-classpath-exception-mean-in-practice/326325#326325) for details.
