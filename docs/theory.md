# Theory: How Padatious Works

## Design Philosophy

Padatious was designed around a specific constraint: a voice assistant skill developer should be able to define intents with a small number of example sentences and get reliable matching without tuning hyperparameters or understanding machine learning.

Two guiding principles follow from this:

**Intents are independent.** Each intent trains its own tiny neural network in isolation. There is no shared encoder, no joint training, and no competition at training time. This means adding a new intent never degrades existing ones — each network only needs to learn a binary signal (match / no match).

**Small data is the norm.** A network with 5–20 training samples should work well. The system achieves this by aggressive data augmentation and by keeping each network's input space small: only the tokens that actually appear in that intent's samples are given dedicated features.

---

## The Two-Layer Matching System

Every query passes through two independent matchers. Their results are merged before the best intent is selected.

### Layer 1 — padaos (regex exact match)

Before any neural network is consulted, the intent template syntax is compiled into regular expressions. This happens once at train time and is stored in memory.

The regex compiler (`padaos.py`) transforms template lines through a chain of substitutions:

- Parenthesised alternatives `(a|b)` become non-capturing regex groups `(?:a|b)`.
- Slot references `{name}` become named capture groups with a greedy `.*?\w.*?` pattern, or a constrained pattern if the entity has been registered.
- `#` becomes `\d`, `:0` becomes `\w+`.
- Spacing between words is made flexible (`\W+` between words, `\W*` elsewhere).
- The full pattern is anchored with `^...$` and compiled case-insensitively.

When a query fully matches a regex, the intent is returned with confidence `1.0` and extracted entities taken directly from the named capture groups. No neural network runs for that query.

This layer is fast and deterministic. Its limitation is that it only fires on exact structural matches — minor rephrasing, extra words, or typos cause it to miss.

### Layer 2 — FANN neural networks (fuzzy match)

For queries that don't produce an exact regex match (or when padaos is disabled), every registered intent's neural network is scored. The best-scoring intent above the confidence threshold wins.

The two layers are complementary: padaos catches the common exact cases with zero inference cost; FANN handles paraphrase and partial matches.

---

## Feature Engineering (Vectorization)

Each intent's neural network operates on a bag-of-tokens feature vector. The vocabulary is determined at training time from the intent's own sample sentences — tokens that never appeared in training are not given dedicated indices.

For a given input sentence, the vector is constructed as follows:

1. **Known tokens**: for each token in the sentence that exists in the intent's vocabulary, set the corresponding element to `1.0`.
2. **Unknown token ratio**: `unknown_count / sentence_length` is stored in a dedicated slot (`:0`). This encodes how "foreign" the sentence looks to this intent.
3. **Length features**: four slots store `len / 1`, `len / 2`, `len / 3`, `len / 4`. These give the network a sense of sentence length without committing to a fixed-size encoding.

All digits in a token are replaced with `#` before lookup, so `"42"` and `"99"` both map to the same `"##"` entry.

This representation is order-insensitive (bag of words) and grows only with the distinct tokens in each intent's training data. A typical intent has tens to low hundreds of vocabulary entries, keeping the network small.

---

## Intent Classification Network (`SimpleIntent`)

Each intent gets one FANN feed-forward network with the architecture:

```
[vocab_size]  →  [10]  →  [1]
```

The hidden layer uses `SIGMOID_SYMMETRIC_STEPWISE` (maps to `[-1, 1]`); the output uses the same. The output is clamped to `[0, 1]` by `max(0, raw_output)`. A score near `1.0` means "this sentence matches this intent"; near `0.0` means it does not.

### Training data augmentation

Raw training samples are not enough on their own. The trainer constructs several categories of examples:

| Example type | Target | Purpose |
|---|---|---|
| Positive samples (the intent's own sentences) | `1.0` | Teach the network what matches |
| Per-word weight samples (each word alone) | `word_len³ / sum(word_len³)` | Make the network weight longer, more distinctive words more heavily |
| Polluted samples (intent sentences + extra `:null:` tokens at start/end) | `0.6` (lenience) | Teach partial tolerance for sentences with extra words |
| Negative samples (other intents' sentences) | `0.0` | Teach the network to reject other intents' patterns |
| Entity-slot-removed samples (slot tokens replaced by `:null:`) | `0.0` | Prevent the network from matching utterances that have no entity content where a slot is expected |
| Empty sentence / `:null:` alone | `0.0` | Anchor the negative end of the output range |

The lenience constant (`0.6`) is deliberately below the default confidence thresholds, so polluted matches contribute positively but do not reach `match_high` on their own.

When two training examples produce the same feature vector (because two different sentences tokenize identically), `resolve_conflicts` deduplicates and takes the **maximum** target value. This prevents the network from receiving contradictory gradient signals.

Training runs up to 10 restarts. Each restart reinitialises the weights and trains for up to 1 000 epochs. The loop exits early if the bit-fail metric reaches zero, meaning the network correctly classifies all training examples within the `0.1` tolerance band.

---

## Entity Extraction (`PosIntent` + `EntityEdge`)

Slot extraction is a separate problem from intent detection. Once an intent fires with sufficient confidence, its entity slots must be located in the sentence.

### The boundary detection approach

For each `{slot}` in an intent, a `PosIntent` is created. It holds two `EntityEdge` networks: one for the **left boundary** (direction = -1) and one for the **right boundary** (direction = +1).

Each `EntityEdge` is another small FANN network:

```
[context_vocab_size]  →  [3]  →  [1]
```

The input to an edge network is a position-weighted context vector. For a candidate position `p`, the vector encodes the tokens to the left (for the left-edge network) or to the right (for the right-edge network), weighted by inverse distance: a token 1 step away contributes `1.0`, 2 steps away `0.5`, etc. The distance to the sentence boundary is always included as a feature.

An edge network outputs a score near `1.0` if the given position looks like the boundary of the entity, and near `0.0` otherwise.

### Candidate enumeration

For a query sentence of length `N`, both edge networks score every position `0..N-1`. Valid `(left_pos, right_pos)` pairs must satisfy:
- Both scores are at least `0.2`.
- `left_pos ≤ right_pos`.
- No other slot token appears between the two positions.

For each valid pair the extracted text is `sent[left_pos : right_pos+1]`.

### Confidence composition

Three signals are combined for each candidate extraction:

1. **Positional confidence**: `(left_score - 0.5 + right_score - 0.5) / 2 + 0.5`
2. **Entity value confidence**: if the slot has a registered entity, the entity's own network scores the candidate text. If no entity is registered, this defaults to `1.0`.
3. **Combined adjustment**: `sqrt(pos_conf × ent_conf) - 0.5` is **added** to the intent's base confidence score.

The geometric mean (via `sqrt(a × b)`) ensures both signals must be positive for the extraction to help. A strong position but a poor entity value (or vice versa) yields a smaller bonus than both being strong.

The final `MatchData` for each candidate is ranked by total confidence. The pipeline returns the best-confidence result with the **fewest total characters** in its extractions — preferring tight, precise matches over sprawling ones.

### Edge training augmentation

Edge networks also receive pollution: for each positive boundary position, additional samples are generated by inserting `:0` tokens adjacent to neighbouring slots. This simulates multi-token entities next to each other and prevents the edge networks from relying on fixed inter-entity spacing.

---

## Domain Classification (`DomainIntentContainer`)

When many intents from many skills are registered, the full pairwise scoring can become noisy — an intent from an unrelated skill may produce a spuriously high score simply because the vocabulary overlaps.

The domain engine addresses this with a two-pass strategy:

**Pass 1 — domain selection.** A top-level `IntentContainer` is trained on the union of all samples per domain (one sample set per skill). This network classifies "what domain does this query belong to?" and returns the top-k scoring domains.

**Pass 2 — intent matching.** Only the per-domain containers for the selected domains are scored. These containers have much smaller vocabularies and fewer intents, reducing cross-skill interference.

The trade-off is an extra training step and a slightly longer inference path. The benefit is better isolation between unrelated skills.

---

## Caching and Incremental Training

Training FANN networks is not cheap relative to the inference latency. Padatious avoids unnecessary re-training through xxhash-based cache invalidation.

When an intent is added or loaded, the sorted training lines are hashed with `xxh32`. The hash is stored as a `.hash` file in the cache directory alongside the `.net` (FANN model) and `.ids` (vocabulary mapping) files.

At startup, `instantiate_from_disk` reads the cache directory and reconstructs all intents whose `.hash`, `.net`, and `.ids` files are present. When `train()` is called, only intents whose training data hash has changed since the last run are actually re-trained.

This means a typical restart after adding one new skill only trains the new skill's intents; all other networks are loaded directly from disk.

---

## Design Decisions and Trade-offs

### One network per intent

The choice to give each intent its own tiny network rather than using a single shared model is the central design decision. It means:

- **Training is embarrassingly parallel.** Intents do not interfere with each other during training.
- **Incremental updates are cheap.** Only changed intents need re-training.
- **No catastrophic forgetting.** Adding a new intent never touches existing network weights.
- **The vocabulary is minimal.** Each network only allocates features for tokens it has actually seen, so even 200 intents do not collectively produce a huge feature space.

The cost is that each network makes its decision in isolation — there is no global context shared across intents.

### FANN over modern frameworks

FANN (`libfann`) was chosen for its extremely low runtime overhead — it is a C library with minimal Python binding overhead, requires no GPU, and produces models that load in microseconds. This matters for a voice assistant where the full intent pipeline must complete in tens of milliseconds.

The cost is that FANN is a legacy library (unmaintained), has a known tendency to segfault under certain conditions (hence `faulthandler.enable()` at startup), and imposes a hard dependency on native shared libraries that complicates packaging.

### Bag-of-words feature representation

Ignoring word order makes the vectorization simple and fast, and means the network is robust to paraphrasing within the same vocabulary. It also means the network cannot distinguish "dog bites man" from "man bites dog". In practice, for the short, constrained utterances typical of voice commands, word order ambiguity is rare.

### Contrastive negative sampling

Rather than using a softmax over all intents, each binary network is trained with the other intents' sentences as negatives. This forces each network to learn what makes its intent *different* from others, not just what matches it. The downside is that training cost grows with the number of intents: more intents means more negative samples for each individual network.

---

## Limitations

**Closed vocabulary.** The network for each intent can only represent tokens it saw during training. An unseen word contributes to the unknown-token ratio but carries no semantic information. There is no subword decomposition, no word embeddings, and no transfer from a pretrained language model.

**Word-order blindness.** The bag-of-words input means grammatically different sentences with the same words receive identical feature vectors. This is usually acceptable for short commands but can produce false matches for intents with overlapping vocabularies.

**Training data size.** Contrastive training works well when all intents have comparable sample counts. A very large intent can dominate the negative sample pool and make it harder for small intents to learn.

**Fixed network topology.** Every `SimpleIntent` uses a `[vocab, 10, 1]` network regardless of intent complexity. A very simple intent wastes capacity; a complex one may underfit. There is no automatic capacity tuning.

**FANN stability.** The `fann2` library can segfault under certain conditions (documented in comments in `opm.py`). This is a known risk when training is run in the main process — a crash in the C library brings down the whole OVOS core. The long-term mitigation is to move training to a subprocess.

**Tokenization is language-agnostic.** The tokenizer splits on character-class boundaries (alpha / digit / punctuation). This works for space-separated European languages but is unsuitable for logographic scripts (Chinese, Japanese, Korean) or languages where morphology is encoded by affixes rather than separate tokens (Finnish, Turkish). The optional Snowball stemmer helps somewhat for inflected languages but does not address tokenization.

**Entity extraction is O(N²).** For each slot, every pair of positions in the sentence is evaluated by the edge networks. Long sentences are thus quadratically expensive in entity extraction, which is why utterances exceeding 50 words are rejected entirely.

**Regex exact matching requires exact templates.** The padaos layer only fires when the query matches the template structure precisely. It provides no tolerance for punctuation variation, extra articles, or different word forms not explicitly listed as alternatives.
