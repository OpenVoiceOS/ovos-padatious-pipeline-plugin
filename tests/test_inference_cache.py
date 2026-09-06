# Copyright 2026 OpenVoiceOS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ovos_padatious._metrics import (
    EXACT_MATCH,
    NEURAL_MATCH,
    performance_metrics,
)
from ovos_padatious.match_data import MatchData
from ovos_padatious.opm import _calc_padatious_intent


def _value(counter):
    return counter.snapshot()["value"]


def test_confidence_retry_cache_keeps_interleaved_utterances():
    """Interleaved clients must not evict confidence-tier retry results."""

    class Container:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def calc_exact_intents(_utterance):
            return []

        def calc_intents(self, utterance):
            self.calls += 1
            return [MatchData(
                name="test-skill:test-intent",
                sent=utterance,
                matches={},
                conf=0.9,
            )]

    container = Container()
    utterances = [f"query {index}" for index in range(8)]
    _calc_padatious_intent.cache_clear()
    try:
        for utterance in utterances:
            _calc_padatious_intent(utterance, container)
        for utterance in utterances:
            _calc_padatious_intent(utterance, container)
    finally:
        _calc_padatious_intent.cache_clear()

    # the second pass is served entirely from the cache; maxsize=3 would have
    # evicted every one of these before it came back around
    assert container.calls == len(utterances)
    assert _calc_padatious_intent.cache_info().maxsize == 128


def test_exact_tier_answers_without_neural_inference():
    """A padaos match must not pay for the neural pass."""

    class Container:
        def __init__(self):
            self.neural_calls = 0

        @staticmethod
        def calc_exact_intents(utterance):
            if utterance == "exact":
                return [MatchData(
                    name="test-skill:exact",
                    sent=utterance,
                    matches={},
                    conf=1.0,
                )]
            return []

        def calc_intents(self, utterance):
            self.neural_calls += 1
            return [MatchData(
                name="test-skill:neural",
                sent=utterance,
                matches={},
                conf=0.9,
            )]

    container = Container()
    _calc_padatious_intent.cache_clear()
    try:
        assert _calc_padatious_intent("exact", container).name == "test-skill:exact"
        assert container.neural_calls == 0
        assert _calc_padatious_intent("fuzzy", container).name == "test-skill:neural"
        assert container.neural_calls == 1
    finally:
        _calc_padatious_intent.cache_clear()


def test_match_path_counters_name_the_resolving_tier():
    class Container:
        @staticmethod
        def calc_exact_intents(utterance):
            if utterance == "exact":
                return [MatchData(
                    name="test-skill:exact",
                    sent=utterance,
                    matches={},
                    conf=1.0,
                )]
            return []

        @staticmethod
        def calc_intents(utterance):
            if utterance == "nothing":
                return []
            return [MatchData(
                name="test-skill:neural",
                sent=utterance,
                matches={},
                conf=0.9,
            )]

    container = Container()
    before = {"exact": _value(EXACT_MATCH), "neural": _value(NEURAL_MATCH)}
    _calc_padatious_intent.cache_clear()
    try:
        _calc_padatious_intent("exact", container)
        _calc_padatious_intent("fuzzy", container)
        _calc_padatious_intent("nothing", container)
    finally:
        _calc_padatious_intent.cache_clear()

    assert _value(EXACT_MATCH) - before["exact"] == 1
    assert _value(NEURAL_MATCH) - before["neural"] == 1
    snapshots = performance_metrics()
    assert snapshots["ovos_padatious_exact_match_total"]["type"] == "counter"
    assert snapshots["ovos_padatious_neural_match_total"]["type"] == "counter"
