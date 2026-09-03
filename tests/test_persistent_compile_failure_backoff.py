"""A persistently raising compile must not be retried at the background
worker's tight ``_wait_for_quiet``/poll cadence forever: that spams an
ERROR traceback and a spurious ``mycroft.skills.trained`` on every pass
(a field trace measured 9 passes in 20s). ``_train_sync``/``_train_worker``
back off exponentially per lang and give up retrying a lang after enough
consecutive failures, resuming only once a registration touches that lang
again; ``mycroft.skills.trained`` is only ever emitted for a pass that
actually trained something.

The existing transient-failure test
(``test_train_sync_survives_a_raising_compile.py``) already covers healing
after a single failure; this test is the adversarial persistent case.
"""
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "always_broken.skill"


class TestPersistentCompileFailureBackoff(unittest.TestCase):
    def test_bounded_attempts_no_trained_event_and_rearm_on_registration(self):
        pipeline = PadatiousPipeline(FakeBus(), config={})
        lang = pipeline.lang
        trained_events = []
        pipeline.bus.on('mycroft.skills.trained', lambda m: trained_events.append(1))

        calls = {"n": 0}

        def always_boom(self, *args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("compile exploded")

        try:
            with mock.patch.object(IntentContainer, "train", always_boom):
                pipeline.register_intent(Message("padatious:register_intent", {
                    "name": f"{SKILL_ID}:hello", "samples": ["hello", "hi there"],
                    "lang": lang, "skill_id": SKILL_ID,
                }))
                pipeline.train()
                # exponential backoff (2s, 4s, 8s...) bounds this to a
                # handful of attempts within a short window, never a retry
                # every ~0.5s for the whole window
                time.sleep(20)
                self.assertLessEqual(
                    calls["n"], 5,
                    "a persistently raising compile must back off, not "
                    "retry at the pre-fix ~0.5Hz cadence")
                self.assertGreaterEqual(calls["n"], 1)
                self.assertEqual(
                    len(trained_events), 0,
                    "mycroft.skills.trained must never be emitted for a "
                    "pass that trained nothing successfully")

            # a subsequent registration change must re-arm training even
            # after the container was given up on
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": f"{SKILL_ID}:bye", "samples": ["bye", "goodbye"],
                "lang": lang, "skill_id": SKILL_ID,
            }))
            self.assertTrue(pipeline.wait_until_trained(timeout=15.0))
            match = pipeline.calc_intent(["hello"], lang)
            self.assertIsNotNone(match)
        finally:
            pipeline.shutdown()


if __name__ == '__main__':
    unittest.main()
