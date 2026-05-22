# Benchmark

`ovos-padatious-pipeline-plugin` ships a comparative accuracy and speed benchmark in `benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and reports every engine side by side.

Every OVOS intent engine ships the same benchmark — evaluated on the same two shared datasets against the same fixed baselines — so results are directly comparable across the whole engine family.

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`. Each has a `<lang>-templates` config (training templates) and a `<lang>-test` config (labelled evaluation utterances). Every engine in this benchmark is a template / sample matcher, so it trains on `-templates` and is evaluated on `-test`.

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits, including a `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of the MASSIVE corpus; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | What it tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

`massive` has a single labelled `test` split and **no no-match cases** — so on `massive` every engine has zero false positives by construction, and accuracy equals recall.

### Entities

Each `{slot}` placeholder ships with example values. `benchmark/dataset.py` collects them into an entities map and every engine registers them (the equivalent of a padatious `.entity` file) before matching.

---

## Engines Compared

The benchmark separates fixed external **baselines** from the **subject** engines under test.

| Engine | Role | Description |
|---|---|---|
| `padaos` | baseline | Regex-based exact matcher (no fuzzy) |
| `nebulento` | baseline | Fuzzy string matching engine, default `damerau-levenshtein` strategy |
| `padatious flat` | subject | This repo's flat neural `IntentContainer` |
| `padatious domain` | subject | This repo's `DomainIntentContainer` — one container per domain, parallel argmax over domains |
| `padatious hierarchical` | subject | This repo's `HierarchicalIntentContainer` — a top-level domain classifier routes the query, then the chosen domain's container resolves the intent |

The three `padatious` rows are the subject of this benchmark; `padaos` and `nebulento` are the fixed baselines shared across every OVOS intent engine. padatious is itself one of the standard baselines — here it doubles as the subject, which is expected.

---

## Results — `intents-for-eval`

1750 cases (1700 match, 50 no-match), 50 intents across 10 domains.

Run `python benchmark/compare.py intents-for-eval` to produce this table. The full run trains five engines, three of which require a neural training pass — it is slow. Results are not committed here; fill the table from a local run.

| Engine | Accuracy | Precision | Recall | F1 | FP / 50 | Median lat |
|---|---|---|---|---|---|---|
| padaos | _run_ | _run_ | _run_ | _run_ | _run_ | _run_ |
| nebulento | _run_ | _run_ | _run_ | _run_ | _run_ | _run_ |
| padatious flat | _run_ | _run_ | _run_ | _run_ | _run_ | _run_ |
| padatious domain | _run_ | _run_ | _run_ | _run_ | _run_ | _run_ |
| padatious hierarchical | _run_ | _run_ | _run_ | _run_ | _run_ | _run_ |

FP = false positives on the 50 `far_ood` no-match utterances. Latency varies run-to-run.

---

## Results — `massive`

2974 cases, 60 intents across 18 domains. The corpus has no no-match cases, so false positives are zero for every engine and accuracy equals recall — this dataset measures recall on a broad, diverse intent set.

Run it with `python benchmark/compare.py massive`. The summary table has the same columns as above; because `massive` has no off-topic split, the `FP` column is zero for every engine and `Accuracy` equals `Recall`.

---

## How to Run

Install benchmark dependencies:

```bash
pip install ovos-padatious[benchmark]
# installs: padaos, nebulento, datasets, fann2==1.0.7
```

Run both datasets:

```bash
python benchmark/compare.py
```

Or one at a time:

```bash
python benchmark/compare.py intents-for-eval
python benchmark/compare.py massive
```

The first run downloads each dataset from the Hugging Face Hub (cached afterwards). The three padatious engines each require a training pass; `padaos` and `nebulento` start immediately. Pass `--ci` for collapsible Markdown output suitable for a CI job summary.

---

## How Metrics Are Calculated

Source: `compute_metrics` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2 × precision × recall / (precision + recall)
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the expected intent and `conf >= threshold` (0.5). A no-match case is correct only when the engine returns `None` or a confidence below threshold.

---

## The Three padatious Engines

- **`padatious flat`** — the standard `IntentContainer`. Every intent is scored in a single flat namespace.
- **`padatious domain`** — `DomainIntentContainer` gives each domain its own `IntentContainer` and takes the global argmax over all domains' top-1 matches. There is no router; parallel scoring keeps confidences comparable.
- **`padatious hierarchical`** — `HierarchicalIntentContainer` is two-stage: a top-level classifier picks one domain, then only that domain's container resolves the intent. With `domain_threshold` raised above `0.0`, the classifier also gates off-topic queries before any intent is scored. The benchmark runs it with `domain_threshold=0.0` (no gate) to isolate the cost of misrouting.
