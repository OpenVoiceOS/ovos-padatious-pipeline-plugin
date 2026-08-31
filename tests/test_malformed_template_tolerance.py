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
"""Regression coverage for a production benchmarking report (7 intents
affected): a single-branch group like ``"cansad(e)"`` makes
``ovos_spec_tools.expansion.expand`` raise ``MalformedTemplate``. That
strictness is deliberate spec-side behaviour and must NOT be relaxed - but a
single malformed training line in one intent/entity must not abort
registration of the whole intent or drop the remaining lines. The malformed
line itself contributes no training sample: a literal line with unbalanced
parenthesis syntax is not a real utterance and would only train garbage.

Sibling fix: OpenVoiceOS/nebulento PR #43 (``nebulento/container.py``
``_expand_or_skip``), fixed the same defect for the nebulento engine. This
mirrors that approach for padatious.
"""
import shutil
import tempfile
from unittest import TestCase, mock

from ovos_bus_client.message import Message

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline
from ovos_padatious.util import expand_lines, expand_or_skip

SKILL = "mood.skill"
NAME = f"{SKILL}:mood"
LANG = "en-US"

MIXED_LINES = [
    "estou (feliz|triste)",
    "estou cansad(e)",
    "sinto-me (bem|mal)",
]

ALL_MALFORMED_LINES = [
    "estou cansad(e)",
    "sinto-me triste(e)",
]


def register_intent_msg(samples):
    data = {"skill_id": SKILL, "name": NAME, "lang": LANG, "samples": samples}
    return Message("padatious:register_intent", data, {"skill_id": SKILL})


def register_entity_msg(entity_name, samples):
    data = {"skill_id": SKILL, "name": entity_name, "lang": LANG, "samples": samples}
    return Message("padatious:register_entity", data, {"skill_id": SKILL})


class TestExpandOrSkip(TestCase):
    def test_expand_or_skip_returns_variants_for_well_formed_line(self):
        self.assertEqual(
            sorted(expand_or_skip("estou (feliz|triste)")),
            sorted(["estou feliz", "estou triste"]),
        )

    def test_expand_or_skip_returns_empty_list_for_malformed_line(self):
        self.assertEqual(expand_or_skip("estou cansad(e)"), [])

    def test_expand_or_skip_logs_warning_naming_the_line(self):
        with mock.patch("ovos_padatious.util.LOG") as fake_log:
            expand_or_skip("estou cansad(e)", context="intent 'mood.skill:mood'")
        fake_log.warning.assert_called_once()
        logged = " ".join(str(a) for a in fake_log.warning.call_args[0])
        self.assertIn("estou cansad(e)", logged)
        self.assertIn("mood.skill:mood", logged)


class TestMalformedTemplateToleranceIntent(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def test_register_intent_does_not_raise_on_malformed_line(self):
        """The bug: a single malformed line must not abort registration."""
        try:
            self.pipeline.register_intent(register_intent_msg(MIXED_LINES))
        except Exception as e:  # pragma: no cover - the assertion is the point
            self.fail(f"register_intent raised on malformed template line: {e!r}")

    def test_unpack_object_expands_valid_lines_and_skips_malformed_line(self):
        """Direct check of the fixed call site: valid lines still fully
        expand, and the malformed line contributes no sample instead of
        aborting the whole batch."""
        lang, skill_id, name, samples, _ = self.pipeline._unpack_object(
            register_intent_msg(MIXED_LINES))
        self.assertIn("estou feliz", samples)
        self.assertIn("estou triste", samples)
        self.assertIn("sinto-me bem", samples)
        self.assertIn("sinto-me mal", samples)
        # the malformed line must not survive, literally or otherwise
        self.assertNotIn("estou cansad(e", samples)
        self.assertNotIn("estou cansad(e)", samples)
        self.assertFalse(any("cansad" in s for s in samples))

    def test_malformed_line_contributes_no_training_data(self):
        """After registration, the malformed line is nowhere in the
        container's intent training data."""
        self.pipeline.register_intent(register_intent_msg(MIXED_LINES))
        container = self.pipeline.containers[LANG]
        sents = container.intents.train_data.sent_lists[NAME]
        flat_tokens = {tok for sent in sents for tok in sent}
        self.assertFalse(any("cansad" in tok for tok in flat_tokens))

    def test_well_formed_utterances_still_match_the_intent(self):
        """The valid lines survive registration and remain the best match
        for their own utterances - never pinned to a fixed confidence."""
        self.pipeline.register_intent(register_intent_msg(MIXED_LINES))
        other_name = f"{SKILL}:greeting"
        other_msg = register_intent_msg(["hello there", "hi there"])
        other_msg.data["name"] = other_name
        self.pipeline.register_intent(other_msg)
        container = self.pipeline.containers[LANG]
        container.train(single_thread=True, timeout=120)
        for utterance in ("estou feliz", "estou triste", "sinto-me bem", "sinto-me mal"):
            match = container.calc_intent(utterance)
            self.assertEqual(match.name, NAME, utterance)

    def test_other_intents_still_register_after_a_malformed_one(self):
        """A malformed intent must not poison sibling intents registered
        afterwards (no shared corrupted state)."""
        self.pipeline.register_intent(register_intent_msg(MIXED_LINES))
        other_name = f"{SKILL}:greeting"
        other_msg = register_intent_msg(["hello there", "hi there"])
        other_msg.data["name"] = other_name
        self.pipeline.register_intent(other_msg)
        container = self.pipeline.containers[LANG]
        container.train(single_thread=True, timeout=120)
        match = container.calc_intent("hello there")
        self.assertEqual(match.name, other_name)


class TestMalformedTemplateToleranceEntity(TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def test_register_entity_does_not_raise_on_malformed_line(self):
        entity_lines = ["feliz", "triste", "cansad(e)"]
        try:
            self.pipeline.register_entity(register_entity_msg("mood_word", entity_lines))
        except Exception as e:  # pragma: no cover
            self.fail(f"register_entity raised on malformed template line: {e!r}")

    def test_entity_unpack_skips_malformed_line(self):
        entity_lines = ["feliz", "triste", "cansad(e)"]
        lang, skill_id, name, samples, _ = self.pipeline._unpack_object(
            register_entity_msg("mood_word", entity_lines))
        self.assertIn("feliz", samples)
        self.assertIn("triste", samples)
        self.assertFalse(any("cansad" in s for s in samples))


class TestExpandLinesToleratesMalformedTemplate(TestCase):
    """``ovos_padatious.util.expand_lines`` is the second call site (used for
    file-based / offline training), independent of the bus handler path."""

    def test_expand_lines_does_not_raise_on_malformed_line(self):
        try:
            result = expand_lines(MIXED_LINES)
        except Exception as e:  # pragma: no cover
            self.fail(f"expand_lines raised on malformed template line: {e!r}")
        self.assertIn("estou feliz", result)
        self.assertIn("estou triste", result)
        self.assertIn("sinto-me bem", result)
        self.assertIn("sinto-me mal", result)
        self.assertFalse(any("cansad" in s for s in result))


class TestAllMalformedIntentNotRegistered(TestCase):
    """When every line of an intent/entity is malformed, expand_or_skip
    empties the sample list entirely: registering it anyway would create a
    dead intent that can never match (conf 0.0 forever) while only logging
    a warning. Such a registration must be refused outright, loudly."""

    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())

    def test_all_malformed_intent_is_not_registered(self):
        with mock.patch("ovos_padatious.opm.LOG") as fake_log:
            self.pipeline.register_intent(register_intent_msg(ALL_MALFORMED_LINES))
            fake_log.error.assert_called()
        container = self.pipeline.containers[LANG]
        self.assertNotIn(NAME, container.intents.train_data.sent_lists)

    def test_all_malformed_intent_error_logged_names_intent_and_skill(self):
        with mock.patch("ovos_padatious.opm.LOG") as fake_log:
            self.pipeline.register_intent(register_intent_msg(ALL_MALFORMED_LINES))
        fake_log.error.assert_called()
        logged = " ".join(str(a) for a in fake_log.error.call_args[0])
        self.assertIn(NAME, logged)
        self.assertIn(SKILL, logged)

    def test_all_malformed_intent_never_matches(self):
        self.pipeline.register_intent(register_intent_msg(ALL_MALFORMED_LINES))
        other_name = f"{SKILL}:greeting"
        other_msg = register_intent_msg(["hello there", "hi there"])
        other_msg.data["name"] = other_name
        self.pipeline.register_intent(other_msg)
        container = self.pipeline.containers[LANG]
        container.train(single_thread=True, timeout=120)
        match = container.calc_intent("hello there")
        self.assertEqual(match.name, other_name)

    def test_all_malformed_entity_is_not_registered(self):
        entity_lines = ["cansad(e)", "triste(e)"]
        with mock.patch("ovos_padatious.opm.LOG") as fake_log:
            self.pipeline.register_entity(register_entity_msg("mood_word", entity_lines))
            fake_log.error.assert_called()
        container = self.pipeline.containers[LANG]
        self.assertNotIn("{mood_word}", container.entities.train_data.sent_lists)


class TestAllMalformedFilePathNotRegistered(TestCase):
    """Same guarantee for the file-based / direct API path (expand_lines
    consumed via TrainData.add_lines), independent of the bus handler."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.container = IntentContainer(self.cache_dir)

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_all_malformed_intent_lines_not_added(self):
        with mock.patch("ovos_padatious.train_data.LOG") as fake_log:
            self.container.add_intent("all_bad", ALL_MALFORMED_LINES)
            fake_log.error.assert_called()
        self.assertNotIn("all_bad", self.container.intents.train_data.sent_lists)

    def test_mixed_intent_lines_still_added(self):
        self.container.add_intent("mixed", MIXED_LINES)
        self.assertIn("mixed", self.container.intents.train_data.sent_lists)
        sents = self.container.intents.train_data.sent_lists["mixed"]
        self.assertTrue(len(sents) > 0)
