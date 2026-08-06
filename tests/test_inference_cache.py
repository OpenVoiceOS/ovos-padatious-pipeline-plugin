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
from ovos_padatious.opm import _calc_padatious_intent


def test_confidence_retry_cache_keeps_interleaved_utterances():
    class Container:
        def __init__(self):
            self.calls = 0

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
