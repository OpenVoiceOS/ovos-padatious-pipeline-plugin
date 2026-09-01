"""``domain_engine: true`` (``DomainIntentContainer``) must be able to
train off the query thread just like the plain engine.

Adversarial review of PR #124 found that
``PadatiousPipeline._train_worker`` calls ``engine._wait_for_quiet()`` on
every dirty container - a method ``DomainIntentContainer`` never
implemented. That raised ``AttributeError`` inside the background worker
thread, silently killing it: ``mycroft.skills.trained`` was never emitted,
``wait_until_trained`` could never observe completion (timing out even
with a generous deadline), and the ONLY way an intent ever became
matchable was a query forcing training itself via
``DomainIntentContainer._train_in_background``.
"""
import unittest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "domain_engine.skill"


class TestDomainEngineBackgroundWorkerCompletes(unittest.TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(
            FakeBus(), config={"domain_engine": True})
        self.pipeline.first_train.set()  # steady state: use the background path
        self.lang = self.pipeline.lang

    def tearDown(self):
        self.pipeline.shutdown()

    def test_wait_until_trained_completes_and_intent_matches(self):
        trained_emitted = []
        self.pipeline.bus.on("mycroft.skills.trained",
                             lambda m: trained_emitted.append(m))

        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": f"{SKILL_ID}:hello", "samples": ["hello", "hi there"],
            "lang": self.lang, "skill_id": SKILL_ID,
        }))

        self.assertTrue(
            self.pipeline.wait_until_trained(timeout=10.0),
            "wait_until_trained must complete without a query forcing "
            "training itself - the background worker died silently on "
            "DomainIntentContainer._wait_for_quiet if this times out")
        self.assertTrue(trained_emitted,
                         "mycroft.skills.trained was never emitted")

        match = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"{SKILL_ID}:hello")


if __name__ == '__main__':
    unittest.main()
