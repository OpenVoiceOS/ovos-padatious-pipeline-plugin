"""OVOS-READY-1 (proposed, OpenVoiceOS/architecture#285): a pipeline's own
warm/cold transition on the bus, so an orchestrator can aggregate "every
configured pipeline warm" instead of gating on process status alone
(OpenVoiceOS/ovos-core#1027 -- a misroute landed well after
``IntentService.status`` reported ready, while padatious was still
"compiling in background, serving last compiled state in the meantime")."""
import tempfile
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import (PADATIOUS_PIPELINE_ID, PIPELINE_WARM_STATE,
                                PadatiousPipeline)

SKILL_ID = "warm.skill"


class TestWarmStateSignal(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.events = []
        self.bus.on(PIPELINE_WARM_STATE, self.events.append)

    def _pipeline(self):
        return PadatiousPipeline(
            self.bus, config={"intent_cache": tempfile.mkdtemp(), "disable_padaos": False})

    def test_cold_announced_at_construction(self):
        pipeline = self._pipeline()
        self.addCleanup(pipeline.shutdown)
        cold = [e for e in self.events if e.data == {
            "pipeline_id": PADATIOUS_PIPELINE_ID, "warm": False}]
        self.assertEqual(len(cold), 1,
                         "exactly one cold announcement, before any "
                         "registration can arrive")

    def test_warm_announced_once_every_language_compiled(self):
        pipeline = self._pipeline()
        self.addCleanup(pipeline.shutdown)
        lang = pipeline.lang
        pipeline.register_intent(Message("padatious:register_intent", {
            "name": "warm.skill:greet", "samples": ["hello"],
            "lang": lang, "skill_id": SKILL_ID}, {"skill_id": SKILL_ID}))
        # mycroft.skills.trained fires per completed registration and can
        # land before the background worker's own next poll notices every
        # container is clean; wait on the signal under test directly.
        done = self.bus.wait_for_message(PIPELINE_WARM_STATE, timeout=10)
        self.assertIsNotNone(done, "warm was never announced")
        self.assertTrue(done.data.get("warm"),
                        "the only other event on this topic is the cold "
                        "one at construction, already past by now")

        warm = [e for e in self.events if e.data.get("warm") is True]
        self.assertEqual(len(warm), 1,
                         "exactly one warm announcement, once every "
                         "language's first compile is done")
        self.assertEqual(warm[0].data["pipeline_id"], PADATIOUS_PIPELINE_ID)

    def test_a_later_registration_does_not_announce_cold_again(self):
        """A container that needs to retrain after the first warm-up
        already has a usable compiled state to serve -- the one case this
        signal exists to distinguish FROM cold. It must not flip warm back
        to False."""
        pipeline = self._pipeline()
        self.addCleanup(pipeline.shutdown)
        lang = pipeline.lang
        pipeline.register_intent(Message("padatious:register_intent", {
            "name": "warm.skill:greet", "samples": ["hello"],
            "lang": lang, "skill_id": SKILL_ID}, {"skill_id": SKILL_ID}))
        self.bus.wait_for_message('mycroft.skills.trained', timeout=10)

        pipeline.register_intent(Message("padatious:register_intent", {
            "name": "warm.skill:bye", "samples": ["goodbye"],
            "lang": lang, "skill_id": SKILL_ID}, {"skill_id": SKILL_ID}))
        self.bus.wait_for_message('mycroft.skills.trained', timeout=10)

        self.assertFalse(
            any(e.data.get("warm") is False for e in self.events[1:]),
            "no cold announcement after the first one at construction")

    def test_no_warm_announcement_while_a_language_has_given_up(self):
        """`pending` excludes a lang in `_compile_giveup` by design, so a
        container whose only dirty languages have all given up looks like
        'nothing left to compile' -- it must not be reported warm, since a
        permanently-failing language never produced a usable state."""
        pipeline = self._pipeline()
        self.addCleanup(pipeline.shutdown)
        lang = pipeline.lang

        def always_boom(self, *args, **kwargs):
            raise RuntimeError("compile exploded")

        with mock.patch.object(IntentContainer, "train", always_boom):
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": f"{SKILL_ID}:hello", "samples": ["hello", "hi"],
                "lang": lang, "skill_id": SKILL_ID}, {"skill_id": SKILL_ID}))
            pipeline.train()
            # 5 consecutive failures at 2,4,8,16,32s backoff before giveup
            time.sleep(40)

        self.assertIn(lang, pipeline._compile_giveup, "the lang must have "
                     "given up for this test to mean anything")
        self.assertFalse(
            any(e.data.get("warm") is True for e in self.events),
            "warm must never be announced while a language has given up "
            "without ever producing a compiled state")


if __name__ == "__main__":
    unittest.main()
