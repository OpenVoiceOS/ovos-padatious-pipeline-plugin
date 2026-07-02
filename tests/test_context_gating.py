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
"""Tests for OVOS-CONTEXT-1 requires_context / excludes_context gating.

The padatious pipeline stores the optional gating declarations at
registration time (keyed by the internal ``<skill_id>:<name>``) and drops
gated candidates at match time via the shared ``gate_satisfied`` helper.

These tests exercise the gate in isolation by stubbing the container match
(``_calc_padatious_intent``), so no trained fann2 model is required.
"""
from unittest import TestCase, mock

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_spec_tools import SpecMessage

import ovos_padatious.opm as opm
from ovos_padatious.match_data import MatchData
from ovos_padatious.opm import PadatiousPipeline


SKILL = "context.skill"
INTENT = "guarded"
FULL = f"{SKILL}:{INTENT}"


def template_msg(requires=None, excludes=None):
    """Build an ovos.intent.register.template payload with optional gates."""
    data = {"skill_id": SKILL, "intent_name": INTENT, "lang": "en-US",
            "samples": ["do the guarded thing"]}
    if requires is not None:
        data["requires_context"] = requires
    if excludes is not None:
        data["excludes_context"] = excludes
    return Message(SpecMessage.INTENT_REGISTER_TEMPLATE, data,
                   {"skill_id": SKILL})


def utter_msg(intent_context):
    """An utterance message carrying a session with the given intent_context."""
    sess = Session("sess-ctx")
    sess.intent_context = dict(intent_context)
    return Message("recognizer_loop:utterance", {},
                   {"session": sess.serialize()})


class TestContextGating(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    # ---- storage ----------------------------------------------------- #

    def test_requires_context_stored_on_register(self):
        """A template registration retains its requires_context declaration."""
        self.pipeline.handle_register_template(template_msg(requires=["mode"]))
        self.assertIn(FULL, self.pipeline._intent_context_gates)
        requires, excludes = self.pipeline._intent_context_gates[FULL]
        self.assertEqual(requires, ["mode"])
        self.assertIsNone(excludes)

    def test_no_gate_not_stored(self):
        """An ungated registration adds no gate entry (unchanged behavior)."""
        self.pipeline.handle_register_template(template_msg())
        self.assertNotIn(FULL, self.pipeline._intent_context_gates)

    def test_gate_dropped_on_deregister(self):
        """Deregistering an intent drops its retained gate."""
        self.pipeline.handle_register_template(template_msg(requires=["mode"]))
        self.pipeline.handle_deregister_intent_spec(
            Message(SpecMessage.INTENT_DEREGISTER,
                    {"skill_id": SKILL, "intent_name": INTENT},
                    {"skill_id": SKILL}))
        self.assertNotIn(FULL, self.pipeline._intent_context_gates)

    # ---- enforcement (container match stubbed) ----------------------- #

    def _stub_match(self, conf=0.9):
        """Patch the container match so calc_intent yields our candidate."""
        candidate = MatchData(FULL, "do the guarded thing", matches={}, conf=conf)
        patcher = mock.patch.object(opm, "_calc_padatious_intent",
                                    return_value=candidate)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_requires_context_present_matches(self):
        """requires_context satisfied by a live private entry => candidate kept."""
        self.pipeline.handle_register_template(template_msg(requires=["mode"]))
        self._stub_match()
        # private scope default => stored key is "<skill_id>:mode"
        msg = utter_msg({f"{SKILL}:mode": {"value": True}})
        result = self.pipeline.calc_intent("do the guarded thing", "en-US", msg)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, FULL)

    def test_requires_context_absent_drops(self):
        """requires_context with no matching entry => candidate dropped."""
        self.pipeline.handle_register_template(template_msg(requires=["mode"]))
        self._stub_match()
        msg = utter_msg({})  # empty context, gate unsatisfied
        result = self.pipeline.calc_intent("do the guarded thing", "en-US", msg)
        self.assertIsNone(result)

    def test_excludes_context_present_drops(self):
        """excludes_context present as a live entry => candidate dropped."""
        self.pipeline.handle_register_template(template_msg(excludes=["busy"]))
        self._stub_match()
        msg = utter_msg({f"{SKILL}:busy": {"value": True}})
        result = self.pipeline.calc_intent("do the guarded thing", "en-US", msg)
        self.assertIsNone(result)

    def test_excludes_context_absent_matches(self):
        """excludes_context absent => candidate kept."""
        self.pipeline.handle_register_template(template_msg(excludes=["busy"]))
        self._stub_match()
        msg = utter_msg({})
        result = self.pipeline.calc_intent("do the guarded thing", "en-US", msg)
        self.assertIsNotNone(result)

    def test_ungated_intent_unaffected(self):
        """An intent without gates matches regardless of context."""
        self.pipeline.handle_register_template(template_msg())
        self._stub_match()
        msg = utter_msg({})
        result = self.pipeline.calc_intent("do the guarded thing", "en-US", msg)
        self.assertIsNotNone(result)
