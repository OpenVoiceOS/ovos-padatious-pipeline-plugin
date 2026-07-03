# Copyright 2020 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Engine-level ``blacklisted_words`` registration.

A registered intent carries an optional ``blacklisted_words`` list. The
pipeline must forward it to ``IntentContainer.add_intent`` by keyword so it
reaches the container's per-intent blacklist; an utterance containing a
blacklisted word is then dropped at match time.
"""
from unittest import TestCase, mock

from ovos_bus_client.message import Message

from ovos_padatious.opm import PadatiousPipeline

SKILL = "blacklist.skill"
NAME = f"{SKILL}:forbidden"
LANG = "en-US"
BLACKLIST = ["bad", "404"]


def register_msg(blacklisted_words):
    data = {"skill_id": SKILL, "name": NAME, "lang": LANG,
            "samples": ["this is a test", "another test"],
            "blacklisted_words": blacklisted_words}
    return Message("padatious:register_intent", data, {"skill_id": SKILL})


class TestRegisterBlacklist(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def test_blacklisted_words_reach_container(self):
        """The list must land in the container's blacklist, not reload_cache."""
        self.pipeline.register_intent(register_msg(BLACKLIST))
        container = self.pipeline.containers[LANG]
        self.assertEqual(container.blacklisted_words[NAME], BLACKLIST)

    def test_blacklisted_utterance_dropped(self):
        """A trained intent does not match an utterance with a blacklisted word."""
        self.pipeline.register_intent(register_msg(BLACKLIST))
        container = self.pipeline.containers[LANG]
        container.train(single_thread=True, timeout=120)
        matches = container.calc_intents("this is a bad test")
        self.assertEqual([m for m in matches if m.name == NAME], [])

    def test_clean_utterance_still_matches(self):
        """Without a blacklisted word the same intent still matches."""
        self.pipeline.register_intent(register_msg(BLACKLIST))
        container = self.pipeline.containers[LANG]
        container.train(single_thread=True, timeout=120)
        self.assertEqual(container.calc_intent("this is another test").name, NAME)
