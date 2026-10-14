"""A pipeline announces ovos.pipeline.loading when it starts compiling what was
registered and ovos.pipeline.loaded when nothing is left to compile."""
import tempfile
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import (PADATIOUS_PIPELINE_ID, PIPELINE_LOADED,
                                PIPELINE_LOADING, PadatiousPipeline)

SKILL_ID = "load.skill"


class TestLoadSignal(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.events = []
        self.bus.on(PIPELINE_LOADING, self.events.append)
        self.bus.on(PIPELINE_LOADED, self.events.append)

    def _pipeline(self):
        pipeline = PadatiousPipeline(
            self.bus, config={"intent_cache": tempfile.mkdtemp(), "disable_padaos": False})
        self.addCleanup(pipeline.shutdown)
        return pipeline

    def _register(self, pipeline, name, sample):
        pipeline.register_intent(Message("padatious:register_intent", {
            "name": f"{SKILL_ID}:{name}", "samples": [sample],
            "lang": pipeline.lang, "skill_id": SKILL_ID}, {"skill_id": SKILL_ID}))

    def test_nothing_announced_before_a_registration(self):
        self._pipeline()
        self.assertEqual(self.events, [])

    def test_loading_then_loaded_around_a_compile(self):
        pipeline = self._pipeline()
        self._register(pipeline, "greet", "hello")
        done = self.bus.wait_for_message(PIPELINE_LOADED, timeout=10)
        self.assertIsNotNone(done, "loaded was never announced")
        self.assertEqual([e.msg_type for e in self.events],
                         [PIPELINE_LOADING, PIPELINE_LOADED])
        self.assertEqual(self.events[0].data["pipeline_id"], PADATIOUS_PIPELINE_ID)

    def test_a_later_registration_loads_again(self):
        pipeline = self._pipeline()
        self._register(pipeline, "greet", "hello")
        self.bus.wait_for_message(PIPELINE_LOADED, timeout=10)
        self._register(pipeline, "bye", "goodbye")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(self.events) < 4:
            time.sleep(0.05)
        self.assertEqual([e.msg_type for e in self.events],
                         [PIPELINE_LOADING, PIPELINE_LOADED] * 2)

    def test_a_language_that_gave_up_does_not_keep_the_pipeline_loading(self):
        pipeline = self._pipeline()

        def always_boom(self, *args, **kwargs):
            raise RuntimeError("compile exploded")

        with mock.patch.object(IntentContainer, "train", always_boom), \
                mock.patch.object(PadatiousPipeline, "_COMPILE_MAX_CONSECUTIVE_FAILURES", 1):
            self._register(pipeline, "hello", "hello")
            done = self.bus.wait_for_message(PIPELINE_LOADED, timeout=30)

        self.assertIsNotNone(done)
        self.assertIn(pipeline.lang, pipeline._compile_giveup)
        self.assertEqual(self.events[-1].msg_type, PIPELINE_LOADED)


if __name__ == "__main__":
    unittest.main()
