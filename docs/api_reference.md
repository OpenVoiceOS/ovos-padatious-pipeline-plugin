# API Reference

## `ovos_padatious.IntentContainer`

The primary interface for training and querying intents.

```python
from ovos_padatious import IntentContainer
```

### Constructor

```python
IntentContainer(cache_dir: str = None, disable_padaos: bool = False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cache_dir` | `str` | XDG data home | Directory used to cache trained neural network models and hash files. Created automatically if it does not exist. |
| `disable_padaos` | `bool` | `False` | Disable the fast regex-based exact matcher (padaos). Useful when only the neural network matcher is desired. |

### Methods

#### `add_intent`

```python
add_intent(name: str, lines: List[str],
           reload_cache: bool = False,
           must_train: bool = True,
           blacklisted_words: List[str] = None) -> None
```

Register an intent from a list of sample utterances.

| Parameter | Description |
|---|---|
| `name` | Unique intent name. |
| `lines` | Sample utterances (may use the intent format syntax). |
| `reload_cache` | Ignore cached model and force re-training. |
| `must_train` | Mark the container as needing training. |
| `blacklisted_words` | Suppress matches that contain any of these words. |

#### `load_intent`

```python
load_intent(name: str, file_name: str,
            reload_cache: bool = False,
            must_train: bool = True) -> None
```

Register an intent by loading samples from a `.intent` file.

#### `add_entity`

```python
add_entity(name: str, lines: List[str],
           reload_cache: bool = False,
           must_train: bool = True) -> None
```

Register a named entity with example values.

```python
container.add_intent('weather', ['will it rain on {weekday}?'])
container.add_entity('weekday', ['monday', 'tuesday', 'wednesday'])
```

#### `load_entity`

```python
load_entity(name: str, file_name: str,
            reload_cache: bool = False,
            must_train: bool = True) -> None
```

Register an entity by loading values from a `.entity` file.

#### `remove_intent`

```python
remove_intent(name: str) -> None
```

Remove a registered intent and mark the container as needing re-training.

#### `remove_entity`

```python
remove_entity(name: str) -> None
```

Remove a registered entity.

#### `train`

```python
train(debug: bool = True, force: bool = False) -> bool
```

Train all intents and entities that have changed since the last training run. Returns `True` on success.

Hash-based caching means only intents/entities whose training data has changed are actually re-trained.

| Parameter | Description |
|---|---|
| `debug` | Print a message for each intent being trained. |
| `force` | Train even if nothing has changed. |

> **Note:** `single_thread` and `timeout` parameters are deprecated and ignored.

#### `calc_intent`

```python
calc_intent(query: str) -> MatchData
```

Return the single best-matching intent for the given query. Automatically trains if needed.

#### `calc_intents`

```python
calc_intents(query: str) -> List[MatchData]
```

Return all intents scored against the query, sorted by confidence. Useful for debugging or handling ambiguous input.

#### `instantiate_from_disk`

```python
instantiate_from_disk() -> None
```

Reload cached models from `cache_dir` without re-training. Call this after constructing a container that should resume from a previous session.

#### `clear`

```python
clear() -> None
```

Reset the container, discarding all registered intents, entities, and cached data.

#### `get_training_args` / `apply_training_args`

```python
get_training_args() -> List[Dict[str, Any]]
apply_training_args(data: List[Dict[str, Any]]) -> None
```

Serialize and replay the sequence of `add_intent` / `add_entity` calls. Useful for transferring container state.

---

## `ovos_padatious.DomainIntentContainer`

A hierarchical engine that first classifies a query into a **domain** and then matches the intent within that domain. This reduces interference between intents from different skills.

```python
from ovos_padatious import DomainIntentContainer
```

### Constructor

```python
DomainIntentContainer(cache_dir: str = None, disable_padaos: bool = False)
```

Same parameters as `IntentContainer`.

### Methods

#### `add_domain_intent`

```python
add_domain_intent(domain_name: str, intent_name: str,
                  intent_samples: List[str],
                  blacklisted_words: List[str] = None) -> None
```

Register an intent within a named domain.

#### `add_domain_entity`

```python
add_domain_entity(domain_name: str, entity_name: str,
                  entity_samples: List[str]) -> None
```

Register an entity scoped to a domain.

#### `remove_domain_intent` / `remove_domain_entity`

```python
remove_domain_intent(domain_name: str, intent_name: str) -> None
remove_domain_entity(domain_name: str, entity_name: str) -> None
```

Remove an intent or entity from a domain.

#### `remove_domain`

```python
remove_domain(domain_name: str) -> None
```

Remove an entire domain and all its intents and entities.

#### `calc_intent`

```python
calc_intent(query: str, domain: str = None) -> MatchData
```

Return the best intent match. If `domain` is not provided the engine first determines the best-matching domain automatically.

#### `calc_intents`

```python
calc_intents(query: str, domain: str = None, top_k_domains: int = 2) -> List[MatchData]
```

Return matching intents from the top `top_k_domains` domains (or a specific domain).

#### `calc_domain` / `calc_domains`

```python
calc_domain(query: str) -> MatchData
calc_domains(query: str) -> List[MatchData]
```

Return the best (or all) domain matches for a query without resolving the final intent.

#### `train`

```python
train() -> None
```

Train the domain classifier and all per-domain intent containers.

---

## `ovos_padatious.MatchData`

Returned by `calc_intent` and `calc_intents`.

```python
from ovos_padatious import MatchData
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Name of the matched intent, or `''` for no match. |
| `sent` | `str` | The query string (after entity extraction). |
| `conf` | `float` | Confidence score from `0.0` (no match) to `1.0` (perfect match). |
| `matches` | `dict` | Extracted entity values, keyed by slot name. |

### Usage

```python
result = container.calc_intent('search for cats on CatTube')

print(result.name)        # 'search'
print(result.conf)        # e.g. 0.98
print(result.matches)     # {'query': 'cats', 'engine': 'CatTube'}
print(result['query'])    # 'cats'        (dict-style access)
print('query' in result)  # True
print(result.get('engine', 'google'))  # 'CatTube'
```

### `detokenize()`

```python
result.detokenize()
```

Converts `sent` and `matches` from token lists back to human-readable strings, handling apostrophes correctly. Called automatically by the OVOS pipeline; typically not needed in direct usage.
