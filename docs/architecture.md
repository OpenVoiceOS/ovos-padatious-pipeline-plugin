# Architecture

## Overview

The plugin is composed of two loosely-coupled layers:

1. **Core engine** (`ovos_padatious/`) — a pure-Python intent matching library that can be used standalone.
2. **OVOS plugin** (`ovos_padatious/opm.py`) — wraps the core engine as an OVOS pipeline plugin.

```
 User utterance
       │
       ▼
 PadatiousPipeline          (opm.py)
  ├─ normalize utterance
  ├─ select language container
  └─ calc_intent()
        │
        ├─► padaos.IntentContainer  ──► regex match  (conf = 1.0)
        │       (padaos.py)
        │
        └─► IntentContainer         ──► neural network match
                (intent_container.py)
                 ├─ IntentManager   ──► SimpleIntent × N
                 └─ EntityManager   ──► Entity × M
```

## Core Engine Components

### `IntentContainer` (`intent_container.py`)

The top-level API class. It:

- Holds an `IntentManager` and an `EntityManager`.
- Optionally holds a `padaos.IntentContainer` for fast exact matching.
- Tracks a `must_train` flag. Calling any `add_*` / `load_*` / `remove_*` method sets it; calling `train()` clears it.
- Serializes all registration calls in `serialized_args` so the container can be replicated via `get_training_args` / `apply_training_args`.

### `DomainIntentContainer` (`domain_container.py`)

A wrapper that adds a **domain classification** layer on top of `IntentContainer`.

- Maintains a **domain engine** (an `IntentContainer`) trained on the union of all samples per domain.
- Maintains one **child `IntentContainer` per domain** for fine-grained matching.
- On `calc_intent(query)`:
  1. The domain engine selects the most likely domain.
  2. The child container for that domain returns the best intent.

This two-pass strategy reduces cross-skill interference when many intents are registered.

### `padaos.IntentContainer` (`padaos.py`)

A lightweight, regex-based exact matcher that runs **before** the neural network. It converts intent templates to regular expressions at compile time and returns confidence `1.0` for any full regex match. This provides:

- Zero training time for exact patterns.
- Deterministic, predictable results for common utterances.
- Optional disabling via `disable_padaos=True`.

Regex compilation is protected by a `threading.Lock` for thread safety.

### `IntentManager` / `EntityManager` (`intent_manager.py`, `entity_manager.py`)

Manage collections of `SimpleIntent` / `Entity` objects respectively. Responsible for:

- Hash-based cache invalidation (only re-train when training data changes).
- Loading/saving neural network models to disk.
- Delegating `calc_intents(query, entities)` across all registered intents.

### `SimpleIntent` (`simple_intent.py`)

One neural network per intent, implemented in numpy (`ovos_padatious.fann`):

- **Vectorization**: maps tokens to a binary input vector. Unknown tokens are counted and included as a ratio. Sentence length is encoded as four fractional features.
- **Architecture**: `[input_size, 10, 1]` feed-forward network with `SIGMOID_SYMMETRIC_STEPWISE` activation.
- **Training data augmentation**:
  - Positive samples (the intent's own sentences) → target `1.0`
  - Per-word weight samples → fractional targets (longer words get higher weight)
  - "Polluted" samples (extra unknown tokens) → target `0.6` (lenience)
  - Negative samples (other intents' sentences) → target `0.0`
  - Samples with entity slots replaced by `:null:` → target `0.0`
- Training repeats up to 10 times until bit-fail reaches 0.

### `Entity` / `EntityEdge` (`entity.py`, `entity_edge.py`)

`Entity` is a trainable object that models which text can fill a `{slot}`. It uses a similar FANN network but trained to recognise entity boundaries (start/end positions).

Entity names must be wrapped in `{...}` internally. `Entity.verify_name` enforces lowercase-letters-and-underscores only. `Entity.wrap_name` / `Entity.unwrap_name` handle the braces.

### `TrainData` / `TrainingManager` (`train_data.py`, `training_manager.py`)

`TrainData` aggregates all registered intents/entities and provides `my_sents(name)` (positive samples) and `other_sents(name)` (negative samples) to drive contrastive training.

`TrainingManager` currently runs training sequentially over the queued objects, using a snapshot of training data to avoid in-flight mutation issues.
### `BracketExpansion` / `SentenceTreeParser` (`bracket_expansion.py`, `simple_intent.py`)

The `(a|b|c)` template syntax is expanded into concrete sentences before training. The parser builds a `Fragment` tree (`Word`, `Sentence`, `Options`) and calls `expand()` to enumerate all combinations.

### `IdManager` (`id_manager.py`)

Maps tokens to fixed indices for the neural network input vector. Backed by a persistent file so the same mapping is used across training and inference.

### `util.py`

- `tokenize(sent)` — lowercases and splits on whitespace/punctuation boundaries.
- `resolve_conflicts(inputs, outputs)` — handles cases where the same input vector maps to conflicting target outputs during training.

## Threading Model

- Training is performed synchronously in `PadatiousPipeline.train()`.
- A `threading.RLock` (`self.lock`) guards the training path against concurrent invocations.
- A `threading.Event` (`finished_training_event`) lets callers wait for training to complete before matching.
- `padaos.IntentContainer._compile()` uses a `threading.Lock` for thread-safe regex compilation.
- `_calc_padatious_intent` is decorated with `@lru_cache(maxsize=3)` so repeated calls for the same utterance at different confidence levels reuse the result.

## Cache Layout

```
<intent_cache>/
  <lang>/
    <intent_name>.intent        # raw training samples
    <intent_name>.intent.net    # serialised FANN network
    <intent_name>.hash          # SHA hash of training data
    {<entity_name>}.entity      # raw entity values
    {<entity_name>}.entity.net  # serialised FANN network
    {<entity_name>}.hash
```

The hash files allow `instantiate_from_disk()` to skip re-training unchanged intents on startup.
