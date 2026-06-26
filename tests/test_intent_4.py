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
"""Tests for the OVOS-INTENT-4 spec registration contract.

Padatious consumes the new ``ovos.intent.register.template`` /
``ovos.entity.register`` / deregister / enable / disable topics *in
addition to* the legacy ``padatious:register_intent`` family. These tests
verify the spec payloads land in the same internal container.

Matching requires a trained fann2/libfann model. When the native library
is unavailable the match assertions are skipped, but registration
acceptance (the spec->padatious mapping) is still asserted.
"""
from unittest import TestCase, mock

from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage

from ovos_padatious.opm import PadatiousPipeline


def _fann2_available():
    try:
        import fann2  # noqa: F401
        from fann2 import libfann  # noqa: F401
        return True
    except Exception:
        return False


FANN2 = _fann2_available()


def template_msg(skill_id, intent_name, samples, blacklist=None, lang="en-US"):
    """Build an ovos.intent.register.template payload (INTENT-4 §6.1)."""
    data = {"skill_id": skill_id, "intent_name": intent_name,
            "lang": lang, "samples": samples}
    if blacklist is not None:
        data["blacklist"] = blacklist
    return Message(SpecMessage.INTENT_REGISTER_TEMPLATE, data,
                   {"skill_id": skill_id})


def entity_msg(skill_id, entity_name, samples, lang="en-US"):
    """Build an ovos.entity.register payload (INTENT-4 §7.1)."""
    return Message(SpecMessage.ENTITY_REGISTER,
                   {"skill_id": skill_id, "entity_name": entity_name,
                    "lang": lang, "samples": samples},
                   {"skill_id": skill_id})


class TestIntent4Registration(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    # ---- §6 template registration ------------------------------------ #

    def test_register_template_indexed(self):
        """A template registration maps onto the internal padatious name."""
        msg = template_msg("music.skill", "play_music",
                            ["(play|put on) {query}",
                             "i want to listen to {query}"])
        self.pipeline.handle_register_template(msg)
        # internal name is <skill_id>:<intent_name>
        self.assertIn("music.skill:play_music",
                      self.pipeline.registered_intents)
        self.assertIn("music.skill:play_music",
                      self.pipeline._skill2intent["music.skill"])

    def test_register_template_blacklist_mapped(self):
        """§6 'blacklist' is forwarded as padatious 'blacklisted_words'."""
        msg = template_msg("music.skill", "play_music",
                            ["play {query}"], blacklist=["trailer"])
        with mock.patch.object(self.pipeline, "register_intent") as reg:
            self.pipeline.handle_register_template(msg)
        reg.assert_called_once()
        forwarded = reg.call_args[0][0]
        self.assertEqual(forwarded.data["blacklisted_words"], ["trailer"])
        self.assertEqual(forwarded.data["name"], "music.skill:play_music")

    def test_register_template_empty_samples_rejected(self):
        """§6.3 malformed: empty samples is not indexed."""
        msg = template_msg("music.skill", "bad_intent", [])
        self.pipeline.handle_register_template(msg)
        self.assertNotIn("music.skill:bad_intent",
                         self.pipeline.registered_intents)

    def test_register_template_missing_identity_rejected(self):
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE,
                      {"samples": ["hello"]}, {})
        self.pipeline.handle_register_template(msg)
        self.assertEqual(self.pipeline.registered_intents, [])

    # ---- §7 entity registration -------------------------------------- #

    def test_register_entity_indexed(self):
        msg = entity_msg("music.skill", "engine",
                         ["spotify", "youtube music"])
        self.pipeline.handle_register_entity_spec(msg)
        names = [e["name"] for e in self.pipeline.registered_entities]
        self.assertIn("music.skill:engine", names)

    def test_register_entity_empty_rejected(self):
        msg = entity_msg("music.skill", "engine", [])
        self.pipeline.handle_register_entity_spec(msg)
        self.assertEqual(self.pipeline.registered_entities, [])

    # ---- §8 deregister / enable / disable ---------------------------- #

    def test_deregister_intent(self):
        self.pipeline.handle_register_template(
            template_msg("music.skill", "play_music", ["play {query}"]))
        self.pipeline.handle_deregister_intent_spec(
            Message(SpecMessage.INTENT_DEREGISTER,
                    {"skill_id": "music.skill", "intent_name": "play_music",
                     "lang": "en-US"}, {"skill_id": "music.skill"}))
        self.assertNotIn("music.skill:play_music",
                         self.pipeline.registered_intents)

    def test_skill_deregister(self):
        self.pipeline.handle_register_template(
            template_msg("music.skill", "play_music", ["play {query}"]))
        self.pipeline.handle_register_entity_spec(
            entity_msg("music.skill", "engine", ["spotify"]))
        self.pipeline.handle_deregister_skill_spec(
            Message(SpecMessage.SKILL_DEREGISTER,
                    {"skill_id": "music.skill"}, {"skill_id": "music.skill"}))
        self.assertNotIn("music.skill:play_music",
                         self.pipeline.registered_intents)
        self.assertEqual(self.pipeline.registered_entities, [])

    def test_disable_then_enable(self):
        self.pipeline.handle_register_template(
            template_msg("music.skill", "play_music", ["play {query}"]))
        disable = Message(SpecMessage.INTENT_DISABLE,
                          {"skill_id": "music.skill",
                           "intent_name": "play_music", "lang": "en-US"},
                          {"skill_id": "music.skill"})
        self.pipeline.handle_disable_intent_spec(disable)
        self.assertNotIn("music.skill:play_music",
                         self.pipeline.registered_intents)
        self.assertIn("music.skill:play_music",
                      self.pipeline._disabled_intents)

        enable = Message(SpecMessage.INTENT_ENABLE,
                         {"skill_id": "music.skill",
                          "intent_name": "play_music", "lang": "en-US"},
                         {"skill_id": "music.skill"})
        self.pipeline.handle_enable_intent_spec(enable)
        self.assertIn("music.skill:play_music",
                      self.pipeline.registered_intents)
        self.assertNotIn("music.skill:play_music",
                         self.pipeline._disabled_intents)

    # ---- match (requires fann2/libfann) ------------------------------ #

    @mock.patch("ovos_padatious.opm.PadatiousPipeline.train")
    def test_template_intent_matches_utterance(self, _mock_train):
        """End-to-end: a template registered via the spec topic matches.

        Skipped when fann2/libfann is unavailable; registration acceptance
        is covered by the tests above regardless.
        """
        if not FANN2:
            self.skipTest("fann2/libfann unavailable; match not exercised")

        pipeline = PadatiousPipeline(
            mock.Mock(), config={"instant_train": True})
        pipeline.first_train.set()
        pipeline.handle_register_template(
            template_msg("hello.skill", "greet",
                         ["hello there", "hi there", "good morning"]))
        # train for real now that an intent exists
        with mock.patch.object(pipeline, "bus"):
            pipeline.containers["en-US"].train()
        match = pipeline.calc_intent("hello there", lang="en-US")
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "hello.skill:greet")
