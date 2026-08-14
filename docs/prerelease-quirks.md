# Prerelease quirks

This page tracks user-visible behavior changes since the last stable release,
`1.4.3`. Newest first. Package name on PyPI is `ovos-padatious`; this repo is
`ovos-padatious-pipeline-plugin`. Resets to empty at the next stable release.

## 2.0.5a1 — entity training bounded; listed values score exactly 1.0

Entity value sets of any size now train in roughly constant time. The
per-entity neural net trains on a deterministic, evenly-strided subset of at
most 128 positive and 128 negative sentences; past a few hundred diverse
samples the net stopped converging, so every training restart burned its
full epoch budget and a pair of ~2200-value entities took over ten minutes
to train (breaking service-ready timeouts on cold boots). Engines own
handling unbounded entity data; this is the subset choice.

The full value set is kept for an exact-match fast path: a listed value now
scores exactly `1.0` through `Entity.match`, deterministically, instead of
the ~0.91 the net gave it. Behavior change: an in-list slot value scores the
same as if no entity were attached — on a default (high-only) pipeline this
restores routing for utterances that registering an entity had silently
pushed below `conf_high` (e.g. 0.9318 back to ~0.9547). Out-of-list values
keep the floor-ramped hint band and still rank below listed ones.

The training-cache hash is salted with a format version, so the whole cache
— intents included — retrains once on first boot after upgrade and
regenerates with the new `.samples` sidecar. Containers restored through
`instantiate_from_disk` bypass that check: a pre-sidecar cache loaded that
way keeps net scoring for listed values (exactly the old behavior) until
its entities are registered again.

## 2.0.4a1 — `tokenize()` no longer splits underscore/digit slot names

`{thing_name}` used to tokenize as `['{thing', '_', 'name}']` instead of one
token, because `tokenize()` treated `_` as a break character. The same bug
hit digits: `{thing_1}` split into three pieces. With the neural path unable
to see the slot at all, only the padaos exact-regex layer (which parses
`{thing_name}` correctly on its own) scored the match, and it only knows
listed values — so an out-of-list value for an underscore/digit slot name
matched at **no confidence** instead of the mid-confidence hint band
OVOS-INTENT-1 §5.4 requires. Slot names without underscores or digits never
hit this path and always degraded gracefully, which is why it went unnoticed.

Fixed in two parts, both scoped to `{...}` placeholders:

- `tokenize()`'s alpha-like character class now includes `_`, so
  `{thing_name}` tokenizes as one token, same as `{thing}` already did.
- Inside a brace span, digits fold into the same token instead of breaking
  on them, so `{thing_1}` tokenizes as one token too.

Two side effects to know about:

- `tokenize()` is shared by template lines and plain runtime utterances, so
  the underscore change also affects how a literal underscore in user text
  tokenizes: `foo_bar` now stays one token instead of splitting into
  `['foo', '_', 'bar']`.
- The digit change is scoped to brace spans only. Outside braces, digits
  still split from letters exactly as before (`one1` -> `['one', '1']`),
  because `IdManager.adj_token()` relies on isolated pure-digit tokens to
  canonicalize numbers to `#` placeholders for the neural net.

## 2.0.3a1 — entity value sets bias confidence, they do not close the vocabulary

Per OVOS-INTENT-1 §5.4, an `.entity` file (or `add_entity()` call) is a set
of **training hints** — example values that bias scoring toward the expected
shape of a slot — not a closed vocabulary. An engine may use the set to
*score* a slot, but must not treat a value outside the set as unmatchable.

Before this fix, a registered entity value set behaved like a closed
vocabulary: an out-of-list slot value did not match. Skills on `<=2.0.2a1`
that relied on that closed-vocabulary behavior (rejecting anything not in
the `.entity` file) will see it eliminated on upgrade — a slot value outside
the registered set now matches at a real, if lower, confidence band instead
of being rejected outright. `Intent.match` passes the intent's own trained
template vocabulary (`self.simple_intent.ids`) into `PosIntent.match` so it
can tell a genuine slot value apart from template words that leaked into the
matched span.

## 2.0.0a1 — pure numpy neural network backend, `fann2` dependency dropped

`fann2` (the FANN Python bindings) is no longer a dependency. The neural
network backend is now pure numpy, reading and writing the same
FANN-compatible model file format. No `.intent`/`.entity` file format change;
existing trained model caches are not compatible across the swap and will
retrain the first time they are loaded (see the cache-clear fix below).

## Since 1.4.3 — other user-visible fixes

- **Stale intent cache no longer survives train/detach (#67).** Training or
  detaching an intent used to leave a stale cached match behind; the cache
  now clears correctly, and an end-to-end ovoscope suite was added to guard
  the fix.
- **`ovos-workshop` floor widened to allow 9.x (#73)**, and an unhashable
  `Session` object was dropped from an `lru_cache` key so caching works
  again against `ovos-bus-client` 2.x sessions (#75).
- **`ovos-spec-tools` upper bound lifted** to allow `ovos-spec-tools` 1.x
  (#78).
- **OVOS-CONTEXT-1 `requires_context`/`excludes_context` gating** is now
  enforced at match time, and slot fill draws on live context per CONTEXT-1
  §7, alongside the INTENT-2 §4.3 slot blacklist (#82, #80).
- **OVOS-INTENT-4 template registration** is consumed alongside the legacy
  format (#72), and dual-registered intent aliases collapse at registration
  time instead of double-registering (#89, #95).
- **`blacklisted_words` forwarded to `add_intent`** by keyword, fixing a
  parameter that was silently dropped in some call shapes (#83).

## Training determinism changed with the 2.0.0a1 backend swap

The old `fann2`/libfann backend seeded its network weights from the clock, so
two training runs on identical `.intent`/`.entity` data could produce
different trained models and slightly different confidence scores. The pure
numpy backend introduced at 2.0.0a1 (`ovos_padatious/fann.py`) seeds instead
from a CRC32 of the training data itself (`training_data.seed`, combined with
a per-attempt counter and the layer shape), so identical training data now
produces an identical initial network and, absent floating-point summation
order differences, an identical trained model. Do not rely on this for
bit-exact confidence values across machines/numpy versions — pin on the
matched intent name and slot values, or use a tolerance band on confidence —
but repeated same-machine retrains on unchanged data should now be stable,
which they were not before 2.0.0a1.

See [docs/theory.md](theory.md) for the matching algorithm and
[docs/intent_format.md](intent_format.md) for `.intent`/`.entity` file
syntax, including the §5.4 hint-vs-vocabulary distinction.
