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
"""OVOS-INTENT-4 §3.2: a §§5-8 message acts on its payload ``skill_id``.

The spec handlers take the target from the payload and never substitute
``context.skill_id`` for it, so a message that names another skill acts on
that skill. The legacy handlers keep the context, which is the only identity
the legacy wire carries.
"""
from unittest import TestCase, mock

from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage

from ovos_padatious.opm import PadatiousPipeline, LOG


def _warnings(mock_warning):
    return [call.args[0] if call.args else "" for call in mock_warning.call_args_list]


class TestRegisterIntentTakesSkillIdFromContext(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def test_matching_payload_registers_as_before(self):
        msg = Message("padatious:register_intent",
                      {"name": "music.skill:play", "lang": "en-US",
                       "skill_id": "music.skill", "samples": ["play {query}"]},
                      {"skill_id": "music.skill"})
        self.pipeline.register_intent(msg)
        self.assertIn("music.skill:play", self.pipeline.registered_intents)

    def test_absent_payload_skill_id_registers_from_context(self):
        msg = Message("padatious:register_intent",
                      {"name": "music.skill:play", "lang": "en-US",
                       "samples": ["play {query}"]},
                      {"skill_id": "music.skill"})
        self.pipeline.register_intent(msg)
        self.assertIn("music.skill:play", self.pipeline.registered_intents)

    def test_differing_payload_skill_id_is_ignored_with_warning(self):
        """A payload skill_id that disagrees with the context must not
        change the outcome: the context value is used and the mismatch is
        only logged."""
        msg = Message("padatious:register_intent",
                      {"name": "music.skill:play", "lang": "en-US",
                       "skill_id": "attacker.skill", "samples": ["play {query}"]},
                      {"skill_id": "music.skill"})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.register_intent(msg)
        self.assertIn("music.skill:play", self.pipeline.registered_intents)
        self.assertNotIn("attacker.skill:play", self.pipeline.registered_intents)
        self.assertTrue(any("differs from" in w for w in _warnings(warn)))

    def test_missing_context_skill_id_is_dropped_with_warning(self):
        msg = Message("padatious:register_intent",
                      {"name": "music.skill:play", "lang": "en-US",
                       "skill_id": "music.skill", "samples": ["play {query}"]},
                      {})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.register_intent(msg)
        self.assertNotIn("music.skill:play", self.pipeline.registered_intents)
        self.assertTrue(any("missing" in w for w in _warnings(warn)))


class TestDetachSkillTakesSkillIdFromContext(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())
        self.pipeline.register_intent(Message(
            "padatious:register_intent",
            {"name": "music.skill:play", "lang": "en-US",
             "skill_id": "music.skill", "samples": ["play {query}"]},
            {"skill_id": "music.skill"}))

    def test_differing_payload_skill_id_is_ignored(self):
        msg = Message("detach_intent",
                      {"skill_id": "attacker.skill"},
                      {"skill_id": "music.skill"})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_detach_skill(msg)
        self.assertNotIn("music.skill:play", self.pipeline.registered_intents)
        self.assertTrue(any("differs from" in w for w in _warnings(warn)))

    def test_missing_context_skill_id_is_dropped(self):
        msg = Message("detach_intent", {"skill_id": "music.skill"}, {})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_detach_skill(msg)
        self.assertIn("music.skill:play", self.pipeline.registered_intents)
        self.assertTrue(any("missing" in w for w in _warnings(warn)))


class TestSpecHandlersTakeSkillIdFromPayload(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def _register(self, skill_id, intent_name, sample, sender=None):
        self.pipeline.handle_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE,
            {"skill_id": skill_id, "intent_name": intent_name,
             "lang": "en-US", "samples": [sample]},
            {"skill_id": sender or skill_id}))

    def test_register_template_indexes_under_the_payload_skill(self):
        # a provisioning tool registering on another skill's behalf
        self._register("music.skill", "play_music", "play {query}",
                       sender="admin.skill")
        self.assertIn("music.skill:play_music", self.pipeline.registered_intents)
        self.assertNotIn("admin.skill:play_music",
                         self.pipeline.registered_intents)

    def test_register_template_missing_context_registers(self):
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE,
                      {"skill_id": "music.skill", "intent_name": "play_music",
                       "lang": "en-US", "samples": ["play {query}"]},
                      {})
        self.pipeline.handle_register_template(msg)
        self.assertIn("music.skill:play_music", self.pipeline.registered_intents)

    def test_register_template_missing_payload_is_dropped(self):
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE,
                      {"intent_name": "play_music", "lang": "en-US",
                       "samples": ["play {query}"]},
                      {"skill_id": "admin.skill"})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_register_template(msg)
        self.assertNotIn("admin.skill:play_music",
                         self.pipeline.registered_intents)
        self.assertTrue(any("missing skill_id" in w for w in _warnings(warn)))

    def test_skill_deregister_removes_the_named_skill_not_the_sender(self):
        self._register("music.skill", "play_music", "play {query}")
        self._register("admin.skill", "shutdown", "shut down")
        self.pipeline.handle_deregister_skill_spec(Message(
            SpecMessage.SKILL_DEREGISTER,
            {"skill_id": "music.skill"},
            {"skill_id": "admin.skill"}))
        self.assertNotIn("music.skill:play_music", self.pipeline.registered_intents)
        self.assertIn("admin.skill:shutdown", self.pipeline.registered_intents)

    def test_skill_deregister_missing_context_removes_the_payload_skill(self):
        self._register("music.skill", "play_music", "play {query}")
        self.pipeline.handle_deregister_skill_spec(Message(
            SpecMessage.SKILL_DEREGISTER, {"skill_id": "music.skill"}, {}))
        self.assertNotIn("music.skill:play_music", self.pipeline.registered_intents)

    def test_skill_deregister_missing_payload_is_dropped(self):
        self._register("music.skill", "play_music", "play {query}")
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_deregister_skill_spec(Message(
                SpecMessage.SKILL_DEREGISTER, {}, {"skill_id": "music.skill"}))
        self.assertIn("music.skill:play_music", self.pipeline.registered_intents)
        self.assertTrue(any("missing skill_id" in w for w in _warnings(warn)))
