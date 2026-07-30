# Benchmark

`ovos-padatious-pipeline-plugin` ships a comparative accuracy and speed benchmark in `benchmark/compare.py`. It runs on the OpenVoiceOS `intents-for-eval` evaluation dataset and reports the padatious engine — in all three variants (flat, per-domain parallel, hierarchical two-stage) — alongside a fixed set of external baselines, so results are directly comparable across the OVOS intent-engine family.

---

## Headline results — `intents-for-eval`

50 intents across 10 domains, 1700 labelled test cases, 50 off-topic (`far_ood`) cases.

| Engine | def F_0.5 | **opt F_0.5** | opt thr | opt FP | **Rec @ P≥99%** |
|---|---|---|---|---|---|
| **padatious flat** | 0.895 | **0.917** | 0.16 | 9 | **70.2%** |
| padatious domain (parallel) | 0.898 | 0.900 | 0.38 | 16 | 65.4% |
| padatious hierarchical (two-stage) | 0.887 | 0.892 | 0.39 | 4 | 63.2% |
| nebulento `damerau` | 0.909 | 0.918 | 0.43 | 26 | 62.0% |
| padaos (regex) | 0.662 (F1) | 0.662 (F1) | 0.50 | 1 | 0.0% (recall ≤ 50%) |

**Padatious flat reaches F_0.5 = 0.917 once its threshold is calibrated — essentially tied with nebulento (0.918) and pulling ahead at the strict precision floor (R@P99 = 70.2% vs nebulento's 62.0%).** Among sample-matching engines on this dataset, padatious holds the highest recall while keeping precision at or above 99%.

The shipped default threshold (0.5) is already close to F_0.5-optimal for every padatious variant: the calibration sweep moves F_0.5 by ≤ 0.03 points. This is the inverse of the Markov engine — padatious's defaults are well-chosen out of the box.

---

## Why F_0.5 and not F1

A voice assistant's two failure modes are not symmetric:

- **False positive** — the wrong intent fires, the skill executes the wrong action, the assistant says the wrong thing. The user has to notice, abort, and re-ask. There is no recovery layer above the intent service that can catch this.
- **False negative** — no intent fires. OVOS hands the utterance to its fallback chain: common-query, the LLM fallback, online search. These exist precisely to handle "I don't know what you meant." Worst case the user re-phrases; best case the LLM nails it.

The cost ratio is roughly 5–10× in favour of false negatives. F1 (which weights precision and recall equally) is the wrong summary metric. F_β with β=0.5 weights precision twice as recall and is the right summary for OVOS.

We also report **Rec@P≥99%** — the recall achievable once the engine's threshold is tuned to keep precision at or above 99%. This is the operating point a maintainer actually picks: "give me the most coverage you can while letting through at most 1% wrong matches." Padatious flat at **70.2%** is the strongest sample-matching engine on this floor in the OVOS family today.

---

## Datasets

The dataset is loaded from the Hugging Face Hub by `benchmark/dataset.py`. It has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances).

| Name | Repo | Intents | Domains | Test cases |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 10 | 1750 |

`intents-for-eval` test splits:

| Split | Cases | Tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

### Slots and entities

Every `{slot}` placeholder in the templates ships with a list of example values. `benchmark/dataset.py` collects them into `Bundle.entities`.

- **padaos**, **padatious** and **nebulento** register the slot values as named entities natively (`add_entity` / `add_domain_entity`). Padatious's neural matcher then learns to associate the slot position with the entity vocabulary, so test queries containing real values (e.g. an actual song name in `play {song}`) still match.
- For padatious domain and hierarchical, each entity is registered into every domain whose intents reference it — the benchmark walks `intent["entities"]` per intent to build the `domain → entities` map.

---

## Engines

The three `padatious` rows are the **subject** of this benchmark. `padaos` and `nebulento` are **fixed baselines** — the same engines and settings as in every OVOS intent-engine benchmark.

| Engine | Role | Notes |
|---|---|---|
| `padaos` | baseline | regex-based exact matcher, no confidence knob |
| `nebulento` | baseline | fuzzy string matcher, `DAMERAU_LEVENSHTEIN_SIMILARITY` |
| `padatious flat` | subject | one neural `IntentContainer` over all 50 intents |
| `padatious domain` (parallel) | subject | `DomainIntentContainer` — one neural net per domain, scored in parallel, global argmax |
| `padatious hierarchical` (two-stage) | subject | `HierarchicalIntentContainer` — top-level domain classifier routes to the matching per-domain net |

All three padatious variants require a `train()` step. Training time and per-query latency are recorded in each per-engine report.

---

## Deep-dive: tuning padatious

### 1. Defaults are well-tuned

The most striking finding: padatious's shipped default threshold (0.5) and default neural configuration are essentially F_0.5-optimal on this dataset.

| Variant | def F_0.5 | opt F_0.5 | Δ |
|---|---|---|---|
| padatious flat | 0.895 | 0.917 | +0.022 |
| padatious domain | 0.898 | 0.900 | +0.002 |
| padatious hierarchical | 0.887 | 0.892 | +0.005 |

Compared to the Markov engine on the same dataset (def F_0.5 = 0.665, opt F_0.5 = 0.909 — a +0.244 swing from calibration alone), padatious arrives operationally ready. The neural network produces well-calibrated confidences: the conf distribution has a clear separation between true matches and noise, and 0.5 sits roughly at that boundary.

Flat does benefit slightly from a lower threshold (0.16 vs 0.50): the network's confidence for correct matches sometimes dips below 0.5 on paraphrases and ASR-noise inputs, and the false-positive surface remains small even when the threshold drops. Domain and hierarchical's optimal thresholds stay near the default (0.38–0.39) because their per-domain nets are more confident inside their own scope.

### 2. Per-domain training: does it help?

Yes — marginally on F_0.5, but with operational benefits beyond pure accuracy.

| Variant | opt F_0.5 | opt FP | R@P99 | Train ms | Median ms |
|---|---|---|---|---|---|
| padatious flat | **0.917** | 9 | **70.2%** | 13 158 | 6.17 |
| padatious domain | 0.900 | 16 | 65.4% | 3 688 | 14.14 |
| padatious hierarchical | 0.892 | 4 | 63.2% | 58 908 | 13.19 |

Each domain in `DomainIntentContainer` trains its own neural net against per-domain negatives (the other intents in that domain). This is fundamentally different from flat training, where a single net must learn to discriminate all 50 intents simultaneously. Two consequences:

- **Domain trains 3.6× faster than flat** (3.7 s vs 13.2 s). Each per-domain net only has ~5 intents to discriminate, so each net's training surface is much smaller; even running 10 nets, total wall-time wins.
- **Domain loses 1.7 points of opt F_0.5 vs flat** (0.900 vs 0.917). When confidences from 10 independent nets are compared, calibration drift across nets shows up: a net trained on a low-variance domain (e.g. `timers_alarms`) produces sharper, higher confidences than one trained on a noisy domain (e.g. `search_qa`), so the global argmax is noisier than a single flat net's argmax.

Domain's FP count rises (16 vs flat's 9) because each net votes inside its own domain regardless of how off-topic the utterance is — there is no shared "this isn't anything" signal.

### 3. Hierarchical: routing cost vs precision

`HierarchicalIntentContainer` adds a top-level domain classifier on top of the per-domain nets. On this dataset it is the weakest variant by F_0.5 (0.892), but it produces the **lowest FP count** (4 — tied with flat at default threshold) and pays the highest training cost (58.9 s — 4.5× the flat baseline, 16× the domain baseline).

The trade-off:

- **Why FP is so low**: an off-topic utterance has to first pass the domain classifier, *then* clear the per-domain intent threshold. Two filters compound. Each fires for a different reason, so false positives that slip through one rarely slip through both.
- **Why recall (and F_0.5) drops**: the domain classifier is itself a sample matcher. If it picks the wrong domain for a borderline utterance (e.g. routing "set my morning alarm to 7" to `calendar` instead of `timers_alarms`), the correct intent is never even scored. This routing cost shows up as roughly 2 points of opt F_0.5 vs flat.
- **Why training is so slow**: the domain classifier itself needs to be trained on the union of every intent's templates as labelled-by-domain examples — that's the full 50-intent training surface again, on top of the 10 per-domain nets. So hierarchical pays the flat training cost *plus* the domain training cost.

Hierarchical is the right pick when (a) you need the strictest possible FP floor and (b) per-skill modular retraining matters more than throughput. It is not the right default.

### 4. Where padatious wins

- **R@P99 = 70.2% (flat).** No other sample-matching engine in the OVOS family currently exceeds this on `intents-for-eval`. Padatious's neural calibration means the confidence threshold can be tightened to keep precision ≥ 99% while still recovering 70% of legitimate matches. Nebulento tops out at 62.0%, the Markov engine at 59.5%. For a production OVOS install where "wrong action" is the dominant failure cost, padatious flat is the safest choice.
- **Zero or very low FP at default threshold.** Flat returns 4 FP at threshold 0.5; hierarchical returns 4; domain returns 6. This is consistently among the lowest in the engine family before any tuning.
- **Precision is ≥ 99% across all three variants** at the default threshold (99.5%–99.6%). The neural net is conservative by construction: it returns no match when no learned template is close enough.

### 5. Where padatious loses

- **Ceiling: recall caps near 64%.** At default threshold, padatious flat recovers only 63.7% of match cases. Looking at the per-intent breakdown (60+ intents with recall below 75%), the bottleneck is template coverage: with ~10–20 training templates per intent, the neural net has not seen enough lexical variation to handle the full `paraphrase` and `asr_noise` splits. Nebulento's edit-distance scoring is more forgiving of minor word swaps, which is why it edges flat at opt F_0.5 (0.918 vs 0.917).
- **Latency: 6–14 ms per query.** Padaos runs at 0.30 ms — 20× faster. For low-power devices or high-QPS routing, the regex baseline still wins on speed.
- **Training time: 13–59 seconds.** Padaos compiles in 305 ms. If your deployment retrains on every skill load, padatious is in another order of magnitude. The domain variant softens this (3.7 s) at a small accuracy cost.

---

## Flat vs Domain vs Hierarchical (for padatious)

After running each variant at its F_0.5-optimal threshold:

| Variant | opt F_0.5 | opt FP | R@P99 | Train s | Median ms |
|---|---|---|---|---|---|
| **padatious flat** | **0.917** | 9 | **70.2%** | 13.2 | 6.17 |
| padatious domain (parallel) | 0.900 | 16 | 65.4% | **3.7** | 14.14 |
| padatious hierarchical (two-stage) | 0.892 | 4 | 63.2% | 58.9 | 13.19 |

Unlike a pure sample-matcher (where flat/domain/hierarchical are essentially the same model partitioned differently), padatious legitimately ships all three because each variant trains a *different* neural model:

- **Flat** trains one net against all 50 intents as competing classes — maximum cross-intent discrimination, slowest single-net training.
- **Domain (parallel)** trains 10 small nets, each discriminating only the 5-or-so intents in its domain. Faster training, smaller per-net surface, looser cross-domain calibration.
- **Hierarchical** adds a routing net on top. Lowest FP, highest training cost, lowest peak F_0.5 due to routing errors.

**Recommendation for OVOS deployments:**

- **Default**: `padatious flat`. Highest F_0.5, highest R@P99, moderate training time, lowest median latency.
- **Modular / per-skill retraining**: `padatious domain`. Pay 1.7 points of F_0.5 to get 3.6× faster training and the ability to retrain a single domain in isolation.
- **Strictest FP floor**: `padatious hierarchical`. Pay 2.5 points of F_0.5 (and 4.5× training time) for the two-stage filter.

For most installs the flat engine is the sensible default; domain and hierarchical are worth picking when modularity or precision dominance outweigh the F_0.5 cost.

---

## Reproducing

```bash
pip install ovos-padatious-pipeline-plugin[benchmark]
python benchmark/compare.py intents-for-eval   # ~2 minutes
```

The first run downloads the dataset from the Hugging Face Hub (cached afterwards).

To re-run the calibration from scratch:

```bash
rm -rf ~/.cache/huggingface/datasets/OpenVoiceOS___intents-for-eval \
       ~/.cache/huggingface/hub/datasets--OpenVoiceOS--intents-for-eval
python benchmark/compare.py intents-for-eval
```

The script prints a per-engine report, a summary table, and a calibration table sweeping the threshold from 0 to 1 in 0.01 steps to find the F_0.5-optimum and the Rec@P≥99% operating point for every engine that exposes a confidence score.

---

## How metrics are calculated

Source: `compute_metrics`, `calibrate_threshold`, `fbeta`, `recall_at_precision` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2·P·R / (P + R)
- **F_0.5** = 1.25·P·R / (0.25·P + R) — weights precision 2× recall (default summary metric for OVOS)
- **Rec@P≥99%** = max recall achievable by sweeping the threshold while keeping precision ≥ 99%
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf ≥ threshold`. A no-match case is correct only when the engine returns `None` or a confidence below threshold.

The calibration sweep re-applies the threshold to the engine's raw `(label, conf)` outputs rather than re-running the engine — every engine's runner returns the unthresholded `raw` list so calibration is a pure post-processing step.
