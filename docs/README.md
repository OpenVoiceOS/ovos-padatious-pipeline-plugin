# ovos-padatious-pipeline-plugin

An efficient neural network intent parser for the [OpenVoiceOS](https://openvoiceos.org/) (OVOS) ecosystem, implemented in pure numpy with a [FANN](https://github.com/libfann/fann)-compatible model format.

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
