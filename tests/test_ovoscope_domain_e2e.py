"""End-to-end tests for DomainPadatiousPipeline using ovoscope.

Mirrors the structure of test_ovoscope_e2e.py but targets the new
``ovos-padatious-domain-pipeline-plugin`` entry point, which selects
the DomainIntentContainer-backed pipeline. The skill_id used at
registration time becomes the domain.
"""
import unittest

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402

from ovoscope import E2EPipelineHarness  # noqa: E402

from ovos_padatious.domain_container import DomainIntentContainer  # noqa: E402
from ovos_padatious.opm import DomainPadatiousPipeline  # noqa: E402

PIPELINE_ID = "ovos-padatious-domain-pipeline-plugin"
CONFIG_KEY = "padatious_domain"

_HELLO_SAMPLES = ["hello", "hi", "hey", "greetings", "good morning"]
_BYE_SAMPLES = ["goodbye", "bye", "see you later", "farewell", "take care"]
_LIGHTS_ON_SAMPLES = ["turn on the lights", "switch on lights", "lights on please"]


class _DomainPadatiousHarness(E2EPipelineHarness):
    PIPELINE_ID = PIPELINE_ID
    CONFIG_KEY = CONFIG_KEY
    PLUGIN_CONFIG = {"instant_train": True, "conf_low": 0.6}
    SKILL_ID = "test_skill_padatious_domain"

    pipeline: DomainPadatiousPipeline  # type: ignore[assignment]

    def _register_intent(self, name, samples, lang="en-US"):
        skill_id = name.split(":", 1)[0]
        self.bus.emit(Message("padatious:register_intent", {
            "name": name, "samples": samples, "lang": lang,
            "skill_id": skill_id,
        }))
        import time
        time.sleep(0.1)


class TestDomainPipelineEntryPoint(_DomainPadatiousHarness):
    def test_engine_is_domain_container(self):
        for c in self.pipeline.containers.values():
            self.assertIsInstance(c, DomainIntentContainer)

    def test_exact_utterance_dispatches_intent(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture(
            "hello", expected_types=[f"{self.SKILL_ID}:hello"], timeout=10.0
        )
        self.assertIsNotNone(msg, "expected intent match on bus")
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:hello")

    def test_intents_routed_via_skill_id_domain(self):
        # Register two skills (= two domains) with disjoint intents.
        skill_a = "domain_a_skill"
        skill_b = "domain_b_skill"
        self._register_intent(f"{skill_a}:hello", _HELLO_SAMPLES)
        self._register_intent(f"{skill_b}:lights_on", _LIGHTS_ON_SAMPLES)

        container = next(iter(self.pipeline.containers.values()))
        self.assertIn(skill_a, container.domains)
        self.assertIn(skill_b, container.domains)

        msg = self.send_and_capture(
            "turn on the lights",
            expected_types=[f"{skill_b}:lights_on"],
            timeout=10.0,
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, f"{skill_b}:lights_on")


if __name__ == "__main__":
    unittest.main()
