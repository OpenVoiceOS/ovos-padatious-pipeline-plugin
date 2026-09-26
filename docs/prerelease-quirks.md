# Prerelease quirks

This page tracks user-visible behavior changes since the last stable release,
`1.4.3`. Newest first. Package name on PyPI is `ovos-padatious`; this repo is
`ovos-padatious-pipeline-plugin`. Resets to empty at the next stable release.

## 2.0.13a3 — re-registration cache invalidation and bounded compile retries

Two more gaps in the 2.0.13a2 entry below, found by a further adversarial
pass:

- The `compiled_generation`-keyed `_calc_padatious_intent` lru_cache
  (2.0.13a2) is only bumped by an actual compile. `register_intent`/
  `register_entity` - and everything that funnels through them, including
  the OVOS-INTENT-4 spec template-registration handlers and the §8.5
  enable handler - never cleared that cache themselves, unlike every
  removal/disable path. So the "OLD compiled entry is dropped immediately"
  claim in the 2.0.13a2 entry held at the padaos layer but not at the
  pipeline's own query cache: a query answered before a re-registration
  could keep matching the retired template at confidence 1.0 until the
  next compile landed. Registration now clears that cache immediately,
  symmetric with removal/disable.
- A compile that keeps raising every pass used to retry forever at the
  background worker's normal debounce cadence, logging a full traceback
  and emitting `mycroft.skills.trained` on every failed attempt. Each lang
  now backs off exponentially (2 seconds, doubling, capped at 5 minutes)
  and, after enough consecutive failures, that lang stops retrying until a
  registration change touches it again. `mycroft.skills.trained` is now
  only emitted for a pass that trained something successfully.

## 2.0.13a2 — padaos never compiles on the match path anymore

A registration whose content exactly matched what was already cached on
disk (a hash-cache hit) never dirtied the container's `must_train` flag,
by design - but `padaos.add_intent`/`add_entity` have no such cache-aware
skip and always marked padaos' own regex container dirty regardless. Every
gate that decided whether to (background-)train checked `must_train` only,
so a boot that replayed nothing but cache hits (e.g. every skill's intents
already cached from an earlier boot) never trained at all: padaos stayed
dirty with nothing to clear it until the first live query forced a full
compile synchronously on the bus message thread - tens of seconds on a
large skill set, stalling that query and every other bus message behind
it. Those gates now also account for a padaos-only dirty container.

On top of that, `padaos.calc_intents` - the match path itself - no longer
compiles inline under any circumstances; it serves whatever was compiled
last (an empty result if nothing ever compiled yet) and leaves compiling
to the background worker.

Removal and disable/enable are runtime gates, not compile products, and
still take effect immediately despite the match path no longer compiling:
`remove_intent` now drops the already-compiled padaos entry for that name
right away instead of only marking the container dirty, and a disabled
intent (OVOS-INTENT-4 Sec8.5) is filtered by name at match time rather than
by waiting for a recompile to forget it. Re-registering an existing name
with different samples is treated the same way: the OLD compiled entry is
dropped immediately (`padaos.add_intent`), so a query in between never
gets a stale match from the retired template - it just gets no match at
all until the new content compiles. Only genuinely new or changed content
is visible after the background pass runs, and that now holds without
exception: a container that has never trained at all (a fresh boot
replaying a 100% hash-cache-hit registration set, with nothing yet
dirtying `must_train`) used to still call `train()` synchronously on the
querying thread the moment its very first query arrived - the same defect
one call frame up from where this entry started, in
`IntentContainer._train_in_background`'s "no previously-trained state"
special case - and `domain_engine: true` kept its own separate inline
`if self.must_train: self.train()` on every `calc_domain(s)`/`calc_intent(s)`
call regardless. Both are gone: a container (or `DomainIntentContainer`
and its per-domain sub-containers) that has never compiled is served empty
until the background worker's first pass actually lands, exactly like
every other pending-compile case.

A test or tool that registers something and needs to query it
deterministically right after should call
`PadatiousPipeline.wait_until_trained(timeout=...)` rather than sleeping or
polling internal container flags - see `docs/ovos_pipeline.md`. It joins
the background worker and never trains on the calling thread itself, so
`timeout` is honoured even while a pass is already in flight.

Four more gaps in that "never on the calling thread" rule, found by a
second adversarial pass:

- `mycroft.skills.train` (`PadatiousPipeline.train`, the same method every
  register handler calls) still ran the first-ever pass synchronously on
  the bus-message thread handling that event, blocking it for the full
  compile duration - the ~85s ser9 field figure, one layer up from where
  this whole entry started. Only `instant_train` mode may still train
  synchronously now; it is an explicit, documented, opt-in trade-off.
- A query answered "no match" before a container's very first compile
  pass landed stayed cached that way forever: `_calc_padatious_intent`'s
  `lru_cache` was only ever invalidated from `PadatiousPipeline._train_sync`,
  which never runs for a pass that a QUERY itself triggered (see
  `IntentContainer._train_in_background`). The cache key now includes each
  container's own `compiled_generation` counter (bumped every time a real
  compile pass finishes), so a query made in a different compile
  generation is a distinct cache entry rather than a stale hit.
- `domain_engine: true` had no `DomainIntentContainer._wait_for_quiet`:
  the background worker's debounce step called it unconditionally on
  every dirty container and died with `AttributeError` for a domain
  engine, silently killing the worker thread. `mycroft.skills.trained`
  was never emitted and `wait_until_trained` timed out; only a query
  forcing training itself ever worked. `DomainIntentContainer` now
  delegates quiescence to its cross-domain engine and every dirty
  per-domain sub-container.
- A training pass whose compile raised left `finished_training_event`
  cleared forever, hanging every later pass (and `wait_until_trained`) on
  an untimed wait. `_train_sync` now always sets that event and logs the
  exception; the failed container stays dirty so the background worker's
  own retry loop picks it back up on its next debounced pass.

## 2.0.12a1 — literal intent-line alternation groups are now bounded too

A literal `(a|b|c|...)` alternation typed directly into an intent line
(rather than referencing a registered entity) went through padaos'
regex-rewrite pipeline with no bound at all: the 64-branch cap added in
2.0.9a2 only guarded entity inlining. A generated or pathological line
with hundreds or thousands of branches, especially repeated across many
lines of the same intent, could reproduce the same compile blowup the
entity cap fixed - the expensive part is the rewrite pipeline itself, not
just the final `re.compile`, so the group is now capped on the raw line
before that pipeline runs. Groups at or under 64 branches (the common
case for a hand-written or migrated `.intent` file) are unaffected and
compile identically to before.

A group over the cap is dropped from padaos entirely for that one line,
the same way a malformed line already is: padaos keeps compiling this
intent's other lines, and the neural tier still trains on this line's
own expanded samples independently. An earlier revision instead degraded
the over-cap group to the same wildcard capture used for an over-cap
entity; that is unsafe here specifically because, unlike a capped entity
slot (whose match can still be checked against that entity's own sample
list), a literal line group has no registered entity behind it to verify
against, so the wildcard made the whole line match almost any utterance
sharing its surrounding words. Padatious' own e2e suite caught exactly
that: an unrelated utterance falsely matched an intent whose only line
held an over-cap group.

## 2.0.11a1 — background training moved off the bus-message thread

`PadatiousPipeline.train()`, the entry point `register_intent`/`register_entity`
and friends actually call from a bus message handler, used to retrain
synchronously on every call once the container had trained once before.
`MessageBusClient.on_message` dispatches every incoming bus message to its
handlers synchronously on the connection's single receive thread, so a slow
retrain (a large registration burst, or an oversized entity — see 2.0.9a2)
blocked that thread for the full compile+train duration: every other message
on the same connection, including a later skill's own registration and the
`intent.service.padatious.manifest.get` / `intent.service.padatious.get`
getters other services poll with a bounded timeout, queued behind it and
could time out. Retraining after the first pass now runs on a single
background worker, mirroring the query-triggered background training added
in 2.0.9a2; `instant_train` mode is unaffected and still retrains
synchronously on every registration, as it always has.

The padaos slow-compile warning now reports that its measured duration
covers the whole container (every entity and intent line), not just the
named "largest entity" — a large entity past the inline cap (2.0.9a2) is
skipped from inlining and therefore cheap on its own, so a slow compile with
a capped entity named as "largest" was actually coming from elsewhere in the
container; the warning says so instead of pointing at the capped entity. The
warning also no longer misfires with a literal `'None'` entity name on a
container with no entities registered at all.

## 2.0.9a2 — bounded padaos entity alternations, training off the query thread

`padaos` (the exact-template matcher) no longer inlines the full value list
of a large entity into every intent line that references it. Entities with
more than 64 values fall back to the same wildcard capture already used for
a `{slot}` with no registered entity at all; exact in-list scoring for
those values is still guaranteed end-to-end through the neural tier's
`Entity.samples` exact-match path, so a listed value still scores 1.0
overall, it just no longer gets padaos's own instant exact match. Without
this cap, an auto-registered entity with a couple thousand multi-word
values (ovos-workshop can emit exactly that), referenced from a few dozen
intent lines, produced megabyte regex sources that took minutes to compile.

Training triggered by a query (`calc_intent`/`calc_intents`) now runs on a
background worker instead of the calling thread, except for the very first
training pass on a fresh container, which still blocks since there is no
previously trained state to answer from. A query issued while a retrain is
in flight is answered from the last trained state instead of paying for the
retrain inline. Compiling `padaos`'s regexes now logs a warning naming the
largest entity when it takes longer than a second.

## 2.0.8a2 — inline '#' digit wildcard deprecated

The inline `#` digit wildcard in `.intent`/`.entity` template lines (e.g.
`count to #`) is deprecated: registering an intent or entity with an
unescaped `#` in a line now logs a one-time deprecation warning naming the
intent/entity and the offending line. `#` is a padatious-only extension
(`ovos_padatious.padaos` compiles it to a digit-class regex, and the neural
net side canonicalizes literal digits to `#` internally) that no other OVOS
intent engine understands, collides with the `#`-as-comment-marker
convention used elsewhere, and assumes the matched entity is spoken/ASR'd
as a literal digit string. Migrate to a `{slot}` placeholder with
skill-side number parsing instead. Matching behavior is unchanged this
cycle - `#` still matches digits exactly as before - and escaping it
(`\#`) or using it only as a leading comment marker does not warn. Removal
is planned for the next major version.

## 2.0.7a2 — malformed template lines are skipped, not fatal

A single malformed template line in a `.intent`/`.entity` file or a bus
registration (e.g. a single-branch group like `"cansad(e)"`, which the intent
template spec rejects) no longer aborts registration of the whole
intent/entity. The offending line is logged as a warning and contributes no
training sample; every other line in the same file or registration still
trains normally. If every line in an intent/entity turns out malformed, the
intent/entity is refused outright (an ERROR is logged naming it) instead of
being registered empty and silently unmatchable.

## 2.0.7a1 — intent suppression: no exact-match bypass, word boundaries

`blacklisted_words` suppression now applies to padaos exact template
matches too: an utterance that perfectly matched a template previously
scored 1.0 and bypassed the blacklist entirely (only neural candidates were
filtered). And the word check now matches at word boundaries instead of raw
substring: "install" suppresses "install firefox" but no longer suppresses
"what is an installment loan".

## 2.0.6a1 — slot blacklists match by whole-value equality

An INTENT-2 §4.3 slot blacklist now drops a bound value only when the whole
captured value equals a blacklisted entry (case-insensitive, whitespace
collapsed). It previously matched by word-subsequence containment, which
made every multi-word value containing a blacklisted token unmatchable — a
pronoun blacklist collaterally blocked real titles like "her majesty",
"the it crowd" or "it takes two". Bare blacklisted values ("it", "he") are
dropped exactly as before.

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
