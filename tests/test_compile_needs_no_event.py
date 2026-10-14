"""The training worker waits for no outside event."""
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


class TestCompilesWithoutAnyEvent(unittest.TestCase):
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
    def test_every_language_compiles_without_any_event(self):
        self.register("en-US")
        self.register("fr-FR")
        en, fr = self.pipeline.containers["en-US"], self.pipeline.containers["fr-FR"]
        self.assertTrue(self.wait(lambda: not en.needs_compile and not fr.needs_compile))
        self.assertEqual(fr.intent_names, [f"{SKILL_ID}:time"])
        self.assertFalse(self.pipeline.bus.ee.listeners("mycroft.ready"))
        self.assertEqual(self.pipeline.calc_intent("what time is it", "fr-FR").name,
                         f"{SKILL_ID}:time")


class TestQuietWindow(unittest.TestCase):
    def test_an_already_quiet_container_does_not_wait(self):
        container = IntentContainer(cache_dir=tempfile.mkdtemp())
        container.add_intent("a", ["hello there"])
        time.sleep(0.12)
        started = time.monotonic()
        container._wait_for_quiet(0.1)
        self.assertLess(time.monotonic() - started, 0.05)

    def test_a_fresh_registration_waits_only_the_rest_of_the_window(self):
        container = IntentContainer(cache_dir=tempfile.mkdtemp())
        container.add_intent("a", ["hello there"])
        time.sleep(0.2)
        started = time.monotonic()
        container._wait_for_quiet(0.4)
        elapsed = time.monotonic() - started
        self.assertGreater(elapsed, 0.1)
        self.assertLess(elapsed, 0.35)
