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
"""OVOS-INTENT-4 §3.2: ``message.context["skill_id"]`` is the authoritative
attribution of the producing component. Every registration/deregistration
handler must take the skill id from the context, never from a payload that
differs from it, and must drop the request when the context carries none.
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


class TestSpecHandlersTakeSkillIdFromContext(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def test_register_template_differing_payload_uses_context(self):
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE,
                      {"skill_id": "attacker.skill", "intent_name": "play_music",
                       "lang": "en-US", "samples": ["play {query}"]},
                      {"skill_id": "music.skill"})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_register_template(msg)
        self.assertIn("music.skill:play_music", self.pipeline.registered_intents)
        self.assertNotIn("attacker.skill:play_music",
                         self.pipeline.registered_intents)
        self.assertTrue(any("differs from" in w for w in _warnings(warn)))

    def test_register_template_missing_context_is_dropped(self):
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE,
                      {"skill_id": "music.skill", "intent_name": "play_music",
                       "lang": "en-US", "samples": ["play {query}"]},
                      {})
        self.pipeline.handle_register_template(msg)
        self.assertNotIn("music.skill:play_music", self.pipeline.registered_intents)

    def test_skill_deregister_differing_payload_uses_context(self):
        self.pipeline.handle_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE,
            {"skill_id": "music.skill", "intent_name": "play_music",
             "lang": "en-US", "samples": ["play {query}"]},
            {"skill_id": "music.skill"}))
        msg = Message(SpecMessage.SKILL_DEREGISTER,
                      {"skill_id": "attacker.skill"},
                      {"skill_id": "music.skill"})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_deregister_skill_spec(msg)
        self.assertNotIn("music.skill:play_music", self.pipeline.registered_intents)
        self.assertTrue(any("differs from" in w for w in _warnings(warn)))

    def test_skill_deregister_missing_context_is_dropped(self):
        self.pipeline.handle_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE,
            {"skill_id": "music.skill", "intent_name": "play_music",
             "lang": "en-US", "samples": ["play {query}"]},
            {"skill_id": "music.skill"}))
        msg = Message(SpecMessage.SKILL_DEREGISTER, {"skill_id": "music.skill"}, {})
        with mock.patch.object(LOG, "warning") as warn:
            self.pipeline.handle_deregister_skill_spec(msg)
        self.assertIn("music.skill:play_music", self.pipeline.registered_intents)
        self.assertTrue(any("missing" in w for w in _warnings(warn)))
