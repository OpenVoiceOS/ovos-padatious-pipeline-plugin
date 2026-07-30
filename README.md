[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md) 
# Padatious

An efficient and agile neural network intent parser, implemented in pure numpy with a [FANN](https://github.com/libfann/fann)-compatible model format.

This repository contains a OVOS pipeline plugin and bundles a fork of the original [padatious](https://github.com/MycroftAI/padatious) from the defunct MycroftAI

## Features

 - Intents are easy to create
 - Requires a relatively small amount of data
 - Intents run independent of each other
 - Easily extract entities (ie. Find the nearest *gas station* -> `place: gas station`)
 - Fast training with a modular approach to neural networks

## OPM Pipeline Entry Points

Two pipeline entry points are registered:

| Entry point | Backed by | When to use |
|---|---|---|
| `ovos-padatious-pipeline-plugin` | `IntentContainer` | Flat intent registry (default). |
| `ovos-padatious-domain-pipeline-plugin` | `DomainIntentContainer` | Group intents by `skill_id` (domain); every sub-domain scores the query in parallel and the global argmax wins. |

The legacy `domain_engine: true` config flag on the flat pipeline still works but is **deprecated** — prefer selecting `ovos-padatious-domain-pipeline-plugin` at the pipeline level. See [`docs/ovos_pipeline.md`](docs/ovos_pipeline.md) for full details.


### Installing

Padatious is pure Python — no native packages or compilers required.

Install via `pip3`:

```
pip3 install padatious
```
Padatious also works in Python 2 if you are unable to upgrade.

### Direct Usage

Here's a simple example of how to use Padatious:

```Python
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

### License

Licensed under the Apache 2 license.
