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
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

from ovos_padatious.match_data import MatchData
from ovos_padatious.opm import _calc_padatious_intent


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

    assert container.calls == len(utterances)
    assert _calc_padatious_intent.cache_info().maxsize == 128


def test_identical_concurrent_cache_misses_are_coalesced():
    """Only one caller should calculate an identical in-flight match."""

    class Container:
        def __init__(self):
            self.calls = 0
            self.lock = Lock()

        def calc_exact_intents(self, utterance):
            with self.lock:
                self.calls += 1
            time.sleep(0.05)
            return [MatchData(
                name="test-skill:test-intent",
                sent=utterance,
                matches={},
                conf=1.0,
            )]

    callers = 16
    barrier = Barrier(callers)
    container = Container()
    _calc_padatious_intent.cache_clear()

    def calculate():
        barrier.wait()
        return _calc_padatious_intent("same query", container)

    try:
        with ThreadPoolExecutor(max_workers=callers) as executor:
            matches = list(executor.map(lambda _: calculate(), range(callers)))
    finally:
        _calc_padatious_intent.cache_clear()

    assert container.calls == 1
    assert all(match.name == "test-skill:test-intent" for match in matches)
    assert len({id(match) for match in matches}) == callers
    matches[0].matches["session-slot"] = "private"
    assert all("session-slot" not in match.matches for match in matches[1:])


def test_distinct_concurrent_cache_misses_are_not_serialized():
    """Singleflight must not block unrelated utterance keys."""

    class Container:
        def __init__(self):
            self.barrier = Barrier(2)

        def calc_exact_intents(self, utterance):
            self.barrier.wait(timeout=1)
            return [MatchData(
                name=f"test-skill:{utterance}",
                sent=utterance,
                matches={},
                conf=1.0,
            )]

    container = Container()
    _calc_padatious_intent.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            matches = list(executor.map(
                lambda utterance: _calc_padatious_intent(utterance, container),
                ("first", "second"),
            ))
    finally:
        _calc_padatious_intent.cache_clear()

    assert {match.name for match in matches} == {
        "test-skill:first", "test-skill:second"}


def test_invalidation_does_not_revive_an_old_inflight_result():
    """A pre-invalidation flight must not populate the new generation."""

    class Container:
        def __init__(self):
            self.calls = 0
            self.started = Event()
            self.release = Event()

        def calc_exact_intents(self, utterance):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(timeout=1)
            return [MatchData(
                name=f"test-skill:version-{self.calls}",
                sent=utterance,
                matches={},
                conf=1.0,
            )]

    container = Container()
    _calc_padatious_intent.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            old_future = executor.submit(
                _calc_padatious_intent, "same query", container)
            assert container.started.wait(timeout=1)
            _calc_padatious_intent.cache_clear()
            container.release.set()
            old_match = old_future.result(timeout=1)

        new_match = _calc_padatious_intent("same query", container)
    finally:
        _calc_padatious_intent.cache_clear()

    assert old_match.name == "test-skill:version-1"
    assert new_match.name == "test-skill:version-2"
    assert container.calls == 2
