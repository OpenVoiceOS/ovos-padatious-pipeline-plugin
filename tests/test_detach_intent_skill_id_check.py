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
attribution of the producing component. ``detach_intent`` is a legacy topic
whose payload carries only ``intent_name`` (built by the producer as
``"<skill_id>:<name>"``); a producer that forges the skill-id prefix of that
name while running under its own context must not be able to detach another
skill's intent.
"""
import shutil
import tempfile
from unittest import TestCase, mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.opm import PadatiousPipeline, LOG

VICTIM = "victim.skill"
ATTACKER = "attacker.skill"


class TestDetachIntentChecksSkillIdAgainstContext(TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.bus = FakeBus()
        self.pipeline = PadatiousPipeline(
            self.bus, config={"intent_cache": self.cache_dir})
        self.lang = self.pipeline.lang
        self.bus.emit(Message("padatious:register_intent", {
            "name": f"{VICTIM}:on", "samples": ["turn on the {thing}"],
            "lang": self.lang, "skill_id": VICTIM,
        }, {"skill_id": VICTIM}))

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_mismatched_context_cannot_detach_another_skills_intent(self):
        """An attacker running under its own context must not be able to
        detach an intent whose name it forges with the victim's prefix."""
        with mock.patch.object(LOG, "warning") as warn:
            self.bus.emit(Message("detach_intent",
                                   {"intent_name": f"{VICTIM}:on"},
                                   {"skill_id": ATTACKER}))
        self.assertIn(f"{VICTIM}:on",
                      self.pipeline.containers[self.lang].intent_names)
        self.assertTrue(any("differs from" in (c.args[0] if c.args else "")
                             for c in warn.call_args_list))

    def test_matching_context_can_detach_own_intent(self):
        self.bus.emit(Message("detach_intent",
                               {"intent_name": f"{VICTIM}:on"},
                               {"skill_id": VICTIM}))
        self.assertNotIn(f"{VICTIM}:on",
                         self.pipeline.containers[self.lang].intent_names)

    def test_missing_context_skill_id_keeps_legacy_behaviour(self):
        """Legacy producers that never set a context skill_id must still be
        able to detach by intent_name alone."""
        self.bus.emit(Message("detach_intent",
                               {"intent_name": f"{VICTIM}:on"}, {}))
        self.assertNotIn(f"{VICTIM}:on",
                         self.pipeline.containers[self.lang].intent_names)
