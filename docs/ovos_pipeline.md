# OVOS Pipeline Plugin

`PadatiousPipeline` integrates Padatious as a confidence-based intent matcher in the OVOS pipeline system.

## Entry Point

The plugin is registered via `setup.py` / `setup.cfg` entry points and discovered automatically by OVOS.

```
ovos.pipeline.padatious = ovos_padatious.opm:PadatiousPipeline
```

## Configuration

Place configuration under `"intent_boxes"` → `"ovos-padatious-pipeline-plugin"` in your OVOS `mycroft.conf`:

```json
{
  "intent_boxes": {
    "ovos-padatious-pipeline-plugin": {
      "conf_high": 0.95,
      "conf_med": 0.8,
      "conf_low": 0.5,
      "domain_engine": false,
      "instant_train": false,
      "intent_cache": "~/.local/share/mycroft/intent_cache",
      "inference_workers": 4,
      "disable_padaos": false,
      "cast_to_ascii": false,
      "stem": false
    }
  }
}
```

### Configuration Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `conf_high` | `float` | `0.95` | Minimum confidence for `match_high`. |
| `conf_med` | `float` | `0.80` | Minimum confidence for `match_medium`. |
| `conf_low` | `float` | `0.50` | Minimum confidence for `match_low`. |
| `domain_engine` | `bool` | `false` | Use `DomainIntentContainer` instead of `IntentContainer`. Groups intents by skill for faster disambiguation. |
| `instant_train` | `bool` | `false` | Trigger training immediately after each intent registration instead of waiting for the `mycroft.skills.train` bus event. |
| `intent_cache` | `str` | XDG data home | Override the directory where trained models are cached. |
| `inference_workers` | `int` | Python default | Maximum reusable worker threads per language for neural intent matching. Set this explicitly to bound CPU contention under concurrent load. |
| `disable_padaos` | `bool` | `false` | Disable the fast regex exact-match layer (padaos). Only the neural network is used. |
| `cast_to_ascii` | `bool` | `false` | Strip accented characters and punctuation from utterances before matching. |
| `stem` | `bool` | `false` | Apply Snowball stemming to utterances and training samples. Improves recall for inflected languages. |

### Cache directory suffixes

The actual cache path is modified automatically based on active options:

| Option active | Suffix appended |
|---|---|
| `domain_engine: true` | `_domain` |
| `stem: true` | `_stemmer` |
| `cast_to_ascii: true` | `_normalized` |

This allows switching between configurations without invalidating unrelated caches.

## Messagebus Events

The plugin subscribes to and emits the following events on the OVOS messagebus.

### Subscribed (incoming)

| Event | Handler | Description |
|---|---|---|
| `padatious:register_intent` | `register_intent` | Register a new intent from a skill. |
| `padatious:register_entity` | `register_entity` | Register a new entity from a skill. |
| `detach_intent` | `handle_detach_intent` | Remove a specific intent. |
| `detach_skill` | `handle_detach_skill` | Remove all intents belonging to a skill. |
| `intent.service.padatious.get` | `handle_get_padatious` | Parse an utterance and return the best intent. |
| `intent.service.padatious.manifest.get` | `handle_padatious_manifest` | Return a list of all registered intent names. |
| `intent.service.padatious.entities.manifest.get` | `handle_entity_manifest` | Return a list of all registered entity definitions. |
| `mycroft.skills.train` | `train` | Trigger training of all pending intents. |

### Emitted (outgoing)

| Event | When |
|---|---|
| `mycroft.skills.trained` | After training completes (or when there is nothing to train). |
| `intent.service.padatious.reply` | In response to `intent.service.padatious.get`. |
| `intent.service.padatious.manifest` | In response to `intent.service.padatious.manifest.get`. |
| `intent.service.padatious.entities.manifest` | In response to `intent.service.padatious.entities.manifest.get`. |

## Multilingual Support

The pipeline maintains a separate `IntentContainer` per language. Languages are taken from the OVOS `lang` and `secondary_langs` core configuration keys.

When matching, the pipeline finds the closest registered language using `langcodes` distance scoring (distance < 10 is accepted).

```json
{
  "lang": "en-US",
  "secondary_langs": ["de-DE", "fr-FR"]
}
```

## Stemming Support

When `stem: true` is set, a `Stemmer` wrapping [Snowball](https://snowballstem.org/) is created per language. Supported language codes:

`ar`, `eu`, `ca`, `da`, `nl`, `en`, `fi`, `fr`, `de`, `el`, `hi`, `hu`, `id`, `ga`, `it`, `lt`, `ne`, `no`, `pt`, `ro`, `ru`, `sr`, `es`, `sv`, `ta`, `tr`

Utterances for unsupported languages are passed through without stemming.

## Confidence Tiers

The OVOS pipeline calls matchers at three confidence levels in sequence:

1. `match_high` (≥ `conf_high`, default 0.95): very confident matches
2. `match_medium` (≥ `conf_med`, default 0.8): good matches
3. `match_low` (≥ `conf_low`, default 0.5): lower-confidence fallback

Exact regex matches from the padaos layer always receive a confidence of `1.0` and will therefore always satisfy `match_high`.

## Word Limit

Utterances longer than **50 words** are silently skipped to avoid excessive computation. This limit is not configurable.

---
[← API Reference](api_reference.md) · [Home](README.md) · [Architecture →](architecture.md)
