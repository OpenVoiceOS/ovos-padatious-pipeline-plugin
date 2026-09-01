"""A raising ``_compile()``/``.train()`` inside ``PadatiousPipeline._train_sync``
must never leave ``finished_training_event`` cleared - every later
``_train_sync`` call (including the untimed ``.wait()`` at the top of that
method, which every subsequent pass and ``wait_until_trained``'s own
polling loop depends on indirectly) would otherwise block forever. The
failed container stays dirty (``needs_compile`` True) so the background
worker's own retry loop (``_train_worker``) picks it back up on its next
debounced pass instead of silently giving up.
"""
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious import padaos
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "flaky_compile.skill"


class TestTrainSyncSurvivesARaisingCompile(unittest.TestCase):
    def test_a_failed_pass_does_not_hang_and_the_retry_succeeds(self):
        pipeline = PadatiousPipeline(FakeBus(), config={})
        pipeline.first_train.set()  # steady state: registration goes to the background worker
        lang = pipeline.lang

        real_compile = padaos.IntentContainer._compile
        calls = []

        def flaky_compile(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom: simulated compile failure")
            return real_compile(self, *args, **kwargs)

        try:
            with mock.patch.object(padaos.IntentContainer, "_compile", flaky_compile):
                pipeline.register_intent(Message("padatious:register_intent", {
                    "name": f"{SKILL_ID}:hello", "samples": ["hello", "hi there"],
                    "lang": lang, "skill_id": SKILL_ID,
                }))
                ok = pipeline.wait_until_trained(timeout=15.0)

            self.assertGreaterEqual(
                len(calls), 2,
                "the background worker must retry after a raising compile, "
                "not give up after the first failure")
            self.assertTrue(
                ok, "a failed pass must not hang wait_until_trained/leave "
                    "finished_training_event stuck cleared")

            match = pipeline.calc_intent(["hello"], lang)
            self.assertIsNotNone(match, "the intent must match after the retry")
            self.assertEqual(match.name, f"{SKILL_ID}:hello")
        finally:
            pipeline.shutdown()


if __name__ == '__main__':
    unittest.main()
