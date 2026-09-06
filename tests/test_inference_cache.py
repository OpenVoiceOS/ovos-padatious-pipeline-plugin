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
from ovos_padatious.match_data import MatchData
from ovos_padatious.opm import _calc_padatious_intent, _INTENT_CACHE_SIZE


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

    # the second pass is served entirely from the cache
    assert container.calls == len(utterances)
    assert _calc_padatious_intent.cache_info().maxsize == _INTENT_CACHE_SIZE


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


def test_cached_match_is_not_shared_between_callers():
    """A caller mutates the match; the next one must not see that.

    PadatiousPipeline.calc_intent fills declared slots from the session's
    intent_context (_fill_context_slots). Handing out the cached object
    means a value one session filled is still filled when the next session
    matches the same utterance.
    """

    class Container:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def calc_exact_intents(_utterance):
            return []

        def calc_intents(self, utterance):
            self.calls += 1
            return [MatchData(
                name="test-skill:call",
                sent=utterance,
                matches={"who": ""},
                conf=0.9,
            )]

    container = Container()
    _calc_padatious_intent.cache_clear()
    try:
        first = _calc_padatious_intent("call them", container)
        first.matches["who"] = "alice"          # session 1 fills its slot

        second = _calc_padatious_intent("call them", container)

        assert second is not first
        assert second.matches == {"who": ""}, "context leaked across sessions"
        # still a cache hit: the container was only consulted once
        assert container.calls == 1
    finally:
        _calc_padatious_intent.cache_clear()


def test_cache_controls_stay_on_the_public_name():
    """Callers manage the cache through _calc_padatious_intent."""
    assert hasattr(_calc_padatious_intent, "cache_clear")
    assert _calc_padatious_intent.cache_info().maxsize == _INTENT_CACHE_SIZE
