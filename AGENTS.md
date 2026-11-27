# AGENTS.md

Conventions for AI coding agents (internal and community) working in this
repository.

## What this repo is

`ovos-padatious-pipeline-plugin` (importable as `ovos_padatious`) is an OVOS
pipeline plugin (`opm.pipeline` entry point `PadatiousPipeline`) that matches
utterances to intents using a small neural network trained per-intent. It
bundles a maintained fork of the original MycroftAI `padatious`, including a
pure-numpy reimplementation of the FANN neural-network library
(`ovos_padatious/fann.py`) that reads and writes the same `FANN_FLO_2.1`
model format the original C library used, so old and newly trained models
stay cross-loadable. It plugs into `ovos-core`'s intent pipeline via
`ovos-plugin-manager` and depends on `ovos_bus_client` for the pipeline
runtime contract.

## Ground rules

- Work on a feature branch. Never push to `dev` or `master` directly.
- Open pull requests against `dev` as **drafts** until CI is green and the
  change is ready for review.
- One commit per PR. Squash before pushing if history accumulates.
- Use conventional commit prefixes (`fix:`, `feat:`, `refactor:`, `docs:`,
  `test:`, `chore:`). Reserve `feat:` for changes a user or downstream
  consumer can actually observe.
- Never hand-edit `ovos_padatious/version.py`. CI computes and bumps the
  version from conventional commit history.
- Every PR description and issue you write or edit carries an AI-authorship
  disclosure at the top, naming the exact model used, and states the text is
  not human-reviewed.

## Dependencies

- Use `uv`, never `pip`, for installing and resolving dependencies.
- Pin floors only, and always allow prereleases: `>=X.Y.Za1`, matching the
  existing pattern in `pyproject.toml` (`ovos-plugin-manager>=2.3.0a1,<3.0.0`,
  `ovos-bus-client>=2.5.1a1,<3.0.0` in the `test` extra).
- All dependency and metadata declarations live in `pyproject.toml`.
- Never install a dependency from a git URL. Publish an alpha to PyPI and
  depend on that. The prerelease floor on `ovos-plugin-manager` is
  deliberately set to let `pip install .[test]` resolve without `--pre`.
  Keep that comment's reasoning in mind before tightening the pin.

## Testing

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[test]"
pytest tests/
```

Note the test directory is `tests/` (plural), unlike some sibling OVOS
repos that use `test/`. The suite includes end-to-end tests
(`tests/end2end/`, `test_ovoscope_e2e.py`, `test_intent_4.py`) that train and
run real intent containers, not just unit-level mocks.

A regression test for a bug must be shown to fail against the code before the
fix and pass after it. A test that passes against unfixed code proves
nothing and does not satisfy this gate.

## Docs discipline

Any change that touches observable behavior updates `README.md` and the
relevant file under `docs/` in the same PR. Also add a version-stamped entry
at the top of `docs/prerelease-quirks.md` describing the change (create the
file if it does not exist yet), newest entry first.

## Repo-specific notes

- `ovos_padatious/fann.py` is a from-scratch pure-numpy trainer, not a
  binding to the C `libfann` library. Its RNG is deliberately seeded from
  the training data itself (`crc32` of the intent's expected outputs,
  combined with the training attempt number and layer shape) rather than
  from the system clock. This makes training reproducible given the same
  intent samples. Do not swap this for an unseeded `np.random` call. That
  would make trained models non-reproducible and break any test that
  compares trained output.

- Intent and entity training data live in `.intent`/`.entity` resource files
  consumed via `ovos-workshop`'s resource loading. As in `ovos-workshop`,
  `.entity` values are training hints for the neural matcher, not a closed
  vocabulary. The trained network can and does match entity values it
  never saw verbatim.
- Entities train the neural net on at most `ENTITY_NET_TRAINING_CAP`
  (128) positive and 128 negative sentences, chosen as an evenly strided,
  deterministic subset (`ovos_padatious/entity.py`). Every listed value is
  still matched exactly through `Entity.samples`, persisted next to the
  trained model as a `.samples` sidecar. The cache hash is salted with
  `"format2"` (`ovos_padatious/training_manager.py`) so caches written
  before the sidecar existed retrain instead of loading half-populated.
  Raising the cap makes large entities (~2000 values) train for minutes
  instead of seconds and does not improve matching of listed values.
- `IntentContainer` defaults its `cache_dir` to
  `f"{xdg_data_home()}/{get_xdg_base()}/intent_cache"` (see
  `ovos_padatious/intent_container.py` and `ovos_padatious/opm.py`) when no
  explicit cache directory is passed. A locally trained cache under a real
  user's XDG data directory can leak into a test run and produce
  stale-model false passes or false failures. Tests and throwaway venvs
  should pass an explicit `cache_dir` (or set `XDG_DATA_HOME` to a temp
  path) rather than relying on the default.
