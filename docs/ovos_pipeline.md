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
| `mycroft.skills.train` | `train` | Trigger training of all pending intents (asynchronous - see below). |

### Emitted (outgoing)

| Event | When |
|---|---|
| `mycroft.skills.trained` | After training completes (or when there is nothing to train). |
| `intent.service.padatious.reply` | In response to `intent.service.padatious.get`. |
| `intent.service.padatious.manifest` | In response to `intent.service.padatious.manifest.get`. |
| `intent.service.padatious.entities.manifest` | In response to `intent.service.padatious.entities.manifest.get`. |

## Training Is Asynchronous

Training and compiling always happen on a single background worker, never
on the thread that made a query or a registration — including a
container's very first pass ever, `mycroft.skills.train` (`train`, the
same method register handlers call), and `domain_engine: true`
(`DomainIntentContainer` mirrors the same rule for its cross-domain
container and every per-domain sub-container, including their shared
debounce/quiet-window logic). The one exception is `instant_train` mode:
an explicit, opt-in config flag that promises a registration is fully
trained by the time the triggering call returns, at the cost of blocking
that call for the full compile+train duration.

Outside `instant_train`, the `intent.service.padatious.get` getter and
every other match-path entry point NEVER wait for a compile - they answer
immediately from whatever state already exists:

- Before a container has EVER compiled, the padaos exact-match layer has
  nothing to contribute and answers no match at all, but the neural tier
  can still answer from a hash-cache-hit object it already loaded
  synchronously at registration time (see `TrainingManager.add`) - so a
  query in this window may get a real, if lower-confidence, match rather
  than a hard `None`.
- Once a compile has landed at least once, a **removal or a disable**
  (`detach_intent`/`detach_skill`, OVOS-INTENT-4 §8.5 enable/disable) is
  reflected immediately - a removed intent is dropped from the served
  state right away, and a disabled intent is filtered at match time by
  name - never waiting on a recompile to forget it.
- An **addition or a replacement** (a new intent/entity, or re-registering
  an existing name with different samples) only becomes matchable once the
  background pass actually compiles it. A replacement's OLD compiled
  template stops matching immediately, though - dropped the moment the
  new registration lands, the same as an outright removal - so a query in
  between never gets a stale match, only no match until the new content
  compiles. This holds at the pipeline's own query cache too: registering
  invalidates it right away, so a query answered from the old definition
  moments earlier does not keep echoing that answer until the next
  compile.
- A match answered "no match" before a container's first compile pass
  landed is not cached forever: the pass, however it was triggered
  (including one kicked off by the query itself rather than by
  `mycroft.skills.train`), invalidates that cached answer so the same
  utterance is re-evaluated fresh once the pass completes.

A lang whose compile keeps raising is retried with exponential backoff
(starting at 2 seconds, doubling, capped at 5 minutes) instead of at the
worker's normal debounce cadence. After enough consecutive failures for a
given language, that language stops retrying until a registration change
touches it again; `mycroft.skills.trained` is only ever emitted for a pass
that actually trained something, never for one that raised.

A test or tool that registers something and needs to query it
deterministically right afterward — without polling internal container
flags — can call `PadatiousPipeline.wait_until_trained(timeout=...)`. It
joins the background worker (spawning one if none is running, then
waiting on it) and never trains on the calling thread itself, so the
`timeout` is honoured even while a pass is already in flight; it returns
`False` rather than hanging past the deadline. This is a synchronization
helper for tests and tooling; production skills should rely on
`mycroft.skills.trained` instead.

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
