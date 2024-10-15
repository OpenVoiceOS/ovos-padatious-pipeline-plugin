[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md) 
# Padatious

An efficient and agile neural network intent parser powered by [fann](https://github.com/libfann/fann).

This repository contains a OVOS pipeline plugin and bundles a fork of the original [padatious](https://github.com/MycroftAI/padatious) from the defunct MycroftAI

## Features

 - Intents are easy to create
 - Requires a relatively small amount of data
 - Intents run independent of each other
 - Easily extract entities (ie. Find the nearest *gas station* -> `place: gas station`)
 - Fast training with a modular approach to neural networks


### Installing

Padatious requires the following native packages to be installed:

 - [`FANN`][fann] (with dev headers)
 - Python development headers
 - `pip3`
 - `swig`

Ubuntu:

```
sudo apt-get install libfann-dev python3-dev python3-pip swig libfann-dev python3-fann2
```

Next, install Padatious via `pip3`:

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

> **NOTE**: This plugin is an exception to [OVOS universal donor policy](https://openvoiceos.github.io/ovos-technical-manual/license/)

It is licensed under the Apache 2 license, however it depends on fann2 which is licensed under the LGPL. [Why is this an issue?](https://softwareengineering.stackexchange.com/questions/119436/what-does-gpl-with-classpath-exception-mean-in-practice/326325#326325)
