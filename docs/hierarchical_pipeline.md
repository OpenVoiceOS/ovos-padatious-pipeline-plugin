# Hierarchical Pipeline Plugin

`HierarchicalPadatiousPipeline` is a two-stage variant of the Padatious pipeline. Intents are grouped into domains; a top-level classifier first selects a single domain, and only that domain's intents are scored to resolve the final intent.

## Intent-Matching Variants

The plugin ships three OPM pipeline entry points, all discovered automatically by OVOS:

| Entry point | Container | Strategy |
|---|---|---|
| `ovos-padatious-pipeline-plugin` | `IntentContainer` | **flat** — every intent is scored globally |
| `ovos-padatious-domain-pipeline-plugin` | `DomainIntentContainer` | **Domain (parallel)** — all domains are scored, global argmax wins |
| `ovos-padatious-hierarchical-pipeline-plugin` | `HierarchicalIntentContainer` | **Hierarchical (two-stage)** — a classifier picks one domain, then only that domain is scored |

In every variant the `skill_id` of a registered intent becomes its domain.

## How Two-Stage Routing Works

1. **Domain classification** — a top-level `IntentContainer` is trained with each domain name as a label over the union of that domain's intent samples. At query time it predicts the most likely domain.
2. **Intent resolution** — only the predicted domain's sub-container scores the utterance; its best intent is returned.

The top-level classifier is rebuilt lazily: registering intents only marks a domain stale, and the classifier is retrained on the next `train()` call. This keeps bulk registration linear.

A domain can also be supplied explicitly to `calc_intent`, which bypasses the classifier and the threshold gate.

## Configuration

Configuration lives under `intents.ovos-padatious-hierarchical-pipeline-plugin` (or `padatious_hierarchical`):

```json
{
  "intents": {
    "ovos-padatious-hierarchical-pipeline-plugin": {
      "conf_high": 0.95,
      "conf_med": 0.8,
      "conf_low": 0.5,
      "domain_threshold": 0.0,
      "instant_train": false,
      "intent_cache": "~/.local/share/mycroft/intent_cache",
      "disable_padaos": false,
      "cast_to_ascii": false,
      "stem": false
    }
  }
}
```

The pipeline accepts every key the flat pipeline does (see [OVOS Pipeline](ovos_pipeline.md)), plus:

| Key | Type | Default | Description |
|---|---|---|---|
| `domain_threshold` | `float` | `0.0` | Minimum confidence the top-level classifier must reach for a query to be routed. When the best domain scores below this, the query is rejected before any sub-container runs. `0.0` disables the gate. |

The cache path is suffixed with `_hierarchical` so this variant's trained models do not collide with the flat or Domain caches.

## When to Use

- **Flat** — small intent sets where global scoring is cheap.
- **Domain (parallel)** — many domains, but per-domain confidences are comparable and a global argmax is acceptable.
- **Hierarchical (two-stage)** — large intent sets where a fast top-level classifier narrows the search, and `domain_threshold` can reject off-topic utterances early.

## Messagebus Events

The hierarchical pipeline subscribes to and emits the same messagebus events as the flat pipeline. See [OVOS Pipeline](ovos_pipeline.md#messagebus-events).
