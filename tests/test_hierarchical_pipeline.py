"""Tests for the HierarchicalPadatiousPipeline OPM entry point.

Exercises:
  - the pipeline constructs a HierarchicalIntentContainer.
  - skill_id is used as the domain when registering intents.
  - detach via remove_domain_intent (intent detach) and skill detach.
  - end-to-end two-stage intent matching.
"""
import tempfile
import unittest

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.hierarchical_container import HierarchicalIntentContainer
from ovos_padatious.opm import HierarchicalPadatiousPipeline


def _make_pipeline(cache_dir, **extra):
    config = {
        "intent_cache": cache_dir,
        "single_thread": True,
        "instant_train": True,
        "conf_low": 0.5,
    }
    config.update(extra)
    return HierarchicalPadatiousPipeline(FakeBus(), config=config)


class TestHierarchicalPadatiousPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name
        self.pipeline = _make_pipeline(self.cache_dir)

    def tearDown(self):
        self.pipeline.shutdown()
        self._tmp.cleanup()

    def test_engine_class_is_hierarchical_container(self):
        self.assertIs(self.pipeline.engine_class, HierarchicalIntentContainer)
        for c in self.pipeline.containers.values():
            self.assertIsInstance(c, HierarchicalIntentContainer)

    def test_domain_threshold_propagated(self):
        pipeline = _make_pipeline(self._tmp.name, domain_threshold=0.4)
        try:
            for c in pipeline.containers.values():
                self.assertEqual(c.domain_threshold, 0.4)
        finally:
            pipeline.shutdown()

    def test_skill_id_is_domain_on_register(self):
        skill_id = "weather.skill"
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": f"{skill_id}:current",
            "samples": ["what is the weather", "tell me the weather"],
            "lang": "en-US",
            "skill_id": skill_id,
        }))
        container = next(iter(self.pipeline.containers.values()))
        self.assertIn(skill_id, container.domains)
        self.assertIn(f"{skill_id}:current",
                      container.domains[skill_id].intent_names)

    def test_intent_matches_via_two_stage_routing(self):
        for skill_id, name, samples in [
            ("hello.skill", "hello", ["hello", "hi", "hey there", "good morning"]),
            ("iot.skill", "lights_on", ["turn on the lights", "lights on",
                                        "switch on the lights"]),
        ]:
            self.pipeline.register_intent(Message("padatious:register_intent", {
                "name": f"{skill_id}:{name}",
                "samples": samples,
                "lang": "en-US",
                "skill_id": skill_id,
            }))
        self.pipeline.train()
        intent = self.pipeline.calc_intent("turn on the lights", "en-US")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.name, "iot.skill:lights_on")

    def test_detach_intent_uses_remove_domain_intent(self):
        skill_id = "iot.skill"
        intent_name = f"{skill_id}:lights_on"
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": intent_name,
            "samples": ["turn on the lights", "lights on"],
            "lang": "en-US",
            "skill_id": skill_id,
        }))
        self.pipeline.handle_detach_intent(Message("detach_intent", {
            "intent_name": intent_name,
        }))
        container = next(iter(self.pipeline.containers.values()))
        if skill_id in container.domains:
            self.assertNotIn(intent_name,
                             container.domains[skill_id].intent_names)
        self.assertNotIn(intent_name, self.pipeline.registered_intents)

    def test_detach_skill_removes_all_intents_for_domain(self):
        skill_id = "chatty.skill"
        for n, samples in [("hello", ["hello", "hi"]),
                           ("bye", ["bye", "goodbye"])]:
            self.pipeline.register_intent(Message("padatious:register_intent", {
                "name": f"{skill_id}:{n}",
                "samples": samples,
                "lang": "en-US",
                "skill_id": skill_id,
            }))
        self.pipeline.handle_detach_skill(Message("detach_skill", {
            "skill_id": skill_id,
        }))
        for intent in (f"{skill_id}:hello", f"{skill_id}:bye"):
            self.assertNotIn(intent, self.pipeline.registered_intents)


if __name__ == "__main__":
    unittest.main()
