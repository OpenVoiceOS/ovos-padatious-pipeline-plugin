"""A registration the plugin already applied is not applied again.

ovos-workshop sends every intent twice, on the legacy
``padatious:register_intent`` topic and on OVOS-INTENT-4's
``ovos.intent.register.template``, with the same samples.
"""
import shutil
import tempfile
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "once.skill"
LANG = "en-US"
SAMPLES = ["what time is it", "tell me the time", "time (please|now)"]


def legacy_msg(samples=SAMPLES):
    return Message("padatious:register_intent", {
        "skill_id": SKILL_ID, "name": f"{SKILL_ID}:time.intent", "lang": LANG,
        "samples": list(samples),
    }, {"skill_id": SKILL_ID})


def spec_msg(samples=SAMPLES):
    return Message("ovos.intent.register.template", {
        "skill_id": SKILL_ID, "intent_name": "time", "lang": LANG,
        "samples": list(samples), "blacklist": [],
    }, {"skill_id": SKILL_ID})


class _Pipeline(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache, ignore_errors=True)

    def settle(self):
        self.assertTrue(self.pipeline.wait_until_trained(timeout=30))


class TestDualEmittedIntentAppliedOnce(_Pipeline):
    def test_second_wire_contract_copy_does_not_reach_the_container(self):
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.register_intent(legacy_msg())
            self.pipeline.handle_register_template(spec_msg())
        self.assertEqual(add.call_count, 1)
        self.settle()
        match = self.pipeline.calc_intent("what time is it", lang=LANG)
        self.assertEqual(match.name, f"{SKILL_ID}:time")
    def test_metadata_still_recorded_from_every_copy(self):
        self.pipeline.register_intent(legacy_msg())
        msg = spec_msg()
        msg.data["requires_context"] = [{"key": "topic"}]
        self.pipeline.handle_register_template(msg)
        self.assertIn(f"{SKILL_ID}:time", self.pipeline._intent_context_gates)
    def test_changed_samples_are_applied(self):
        self.pipeline.register_intent(legacy_msg())
        self.settle()
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.handle_register_template(spec_msg(["what hour is it"]))
        self.assertEqual(add.call_count, 1)
    def test_detach_then_identical_registration_applies_again(self):
        self.pipeline.register_intent(legacy_msg())
        self.settle()
        self.pipeline.handle_detach_intent(Message(
            "detach_intent", {"intent_name": f"{SKILL_ID}:time.intent"}))
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.register_intent(legacy_msg())
        self.assertEqual(add.call_count, 1)
    def test_rejected_registration_does_not_shadow_a_later_one(self):
        with mock.patch.object(PadatiousPipeline, "_unpack_object", return_value=None):
            self.pipeline.register_intent(legacy_msg())
        with mock.patch.object(IntentContainer, "add_intent",
                               autospec=True,
                               side_effect=IntentContainer.add_intent) as add:
            self.pipeline.handle_register_template(spec_msg())
        self.assertEqual(add.call_count, 1)


class TestFailedApplicationRetried(unittest.TestCase):
    """Every configured language is registered and compiled the same way, as
    soon as its registrations go quiet; the plugin waits for no event."""
    def setUp(self):
        self.cache = tempfile.mkdtemp()
        debounce = mock.patch.object(IntentContainer, "_TRAIN_DEBOUNCE_S", 0.05)
        debounce.start()
        self.addCleanup(debounce.stop)
        self.pipeline = self.new_pipeline()
    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache, ignore_errors=True)
    def new_pipeline(self):
        conf = {"lang": "en-US", "secondary_langs": ["fr-FR"]}
        with mock.patch("ovos_padatious.opm.Configuration", return_value=conf):
            return PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})
    def register(self, lang, name="time", samples=SAMPLES):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "skill_id": SKILL_ID, "name": f"{SKILL_ID}:{name}", "lang": lang,
            "samples": list(samples)}, {"skill_id": SKILL_ID}))
    def wait(self, condition, timeout=20.0):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.05)
        return condition()
    def test_a_failed_application_is_retried_for_the_other_copy(self):
        en = self.pipeline.containers["en-US"]
        real_add = en.add_intent
        calls = []

        def flaky(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 1:
                raise OSError("transient")
            return real_add(*args, **kwargs)

        with mock.patch.object(en, "add_intent", side_effect=flaky):
            self.register("en-US")
            self.register("en-US")  # the other wire contract's copy
        self.assertEqual(len(calls), 2)
        self.assertIn(f"{SKILL_ID}:time", en.intent_names)
