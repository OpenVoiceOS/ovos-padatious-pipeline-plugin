"""OVOS-INTENT-4 *consumer* end-to-end tests for the Padatious pipeline.

``tests/test_ovoscope_e2e.py`` proves padatious matches intents registered via
the legacy ``padatious:register_intent`` event. This suite proves padatious
*consumes the INTENT-4 spec registration topics* (``ovos-intent-4.md``) and then
matches.

Padatious is a **template** engine: it consumes ``ovos.intent.register.template``
(§6) and not ``ovos.intent.register.keyword`` (§11). The suite boots a real
``MiniCroft`` pinned to the padatious pipeline, emits the spec registration on
the wire, sends a matching utterance, and asserts the intent dispatches
``<skill_id>:<intent_name>`` — proving spec-topic consumption.

The pipeline runs with ``instant_train: True`` so every registration trains
synchronously and matches are deterministic. All assertions live in one
``TestCase`` (one MiniCroft for the file) — padatious's per-boot training is
heavy, so a single long-lived engine with per-test ``detach_skill`` isolation
is far more reliable than re-booting between classes.
"""
import time
import unittest

import pytest

ovoscope = pytest.importorskip(
    "ovoscope", reason="ovoscope not installed; skipping E2E tests"
)

from ovoscope import E2EPipelineHarness  # noqa: E402
from ovos_bus_client.message import Message  # noqa: E402
from ovos_spec_tools import SpecMessage  # noqa: E402

from ovos_padatious.opm import PadatiousPipeline  # noqa: E402

PIPELINE_ID = "ovos-padatious-pipeline-plugin"
CONFIG_KEY = "padatious"

REGISTER_TEMPLATE = str(SpecMessage.INTENT_REGISTER_TEMPLATE)
REGISTER_KEYWORD = str(SpecMessage.INTENT_REGISTER_KEYWORD)
INTENT_DEREGISTER = str(SpecMessage.INTENT_DEREGISTER)
SKILL_DEREGISTER = str(SpecMessage.SKILL_DEREGISTER)
INTENT_DISABLE = str(SpecMessage.INTENT_DISABLE)
INTENT_ENABLE = str(SpecMessage.INTENT_ENABLE)

_HELLO = ["hello", "hi there", "hey", "greetings", "good morning"]
_BYE = ["goodbye", "bye bye", "see you later", "farewell"]

_MATCH_TIMEOUT = 12.0


class TestIntent4Consume(E2EPipelineHarness):
    """All OVOS-INTENT-4 consumer assertions for padatious on one MiniCroft."""

    PIPELINE_ID = PIPELINE_ID
    CONFIG_KEY = CONFIG_KEY
    # instant_train -> each registration trains synchronously so a following
    # utterance can match deterministically; conf_low keeps short greetings in.
    PLUGIN_CONFIG = {"instant_train": True, "conf_low": 0.6}
    SKILL_ID = "intent4_padatious.skill"

    pipeline: PadatiousPipeline  # type: ignore[assignment]

    # -- helpers --------------------------------------------------------

    def _register_template(self, intent_name, samples, *, blacklist=None,
                           lang="en-US", settle=2.0):
        payload = {
            "skill_id": self.SKILL_ID,
            "intent_name": intent_name,
            "lang": lang,
            "samples": samples,
        }
        if blacklist is not None:
            payload["blacklist"] = blacklist
        self.bus.emit(Message(REGISTER_TEMPLATE, payload,
                              {"skill_id": self.SKILL_ID}))
        time.sleep(settle)
        # probe with a verbatim sample so the readiness check is exact.
        self._wait_trained(f"{self.SKILL_ID}:{intent_name}",
                           probe=samples[0], lang=lang)

    def _wait_trained(self, intent_name, *, probe=None, lang="en-US",
                      timeout=20.0):
        """Block until the intent is registered, trained, and actually matches.

        padatious trains on a threadpool; even with ``instant_train`` the
        synchronous flag only guarantees a train was *kicked off*. Polling the
        engine's own ``calc_intent`` against a probe utterance is the true
        readiness signal — it removes the cold-train race deterministically.
        """
        probe = probe or intent_name.split(":")[-1]
        deadline = time.time() + timeout
        while time.time() < deadline:
            registered = intent_name in self.pipeline.registered_intents
            container = self.pipeline.containers.get(lang)
            trained = container is not None and not getattr(
                container, "must_train", False)
            if registered and trained:
                try:
                    match = self.pipeline.calc_intent([probe], lang)
                    if match is not None and getattr(match, "name", None) == intent_name:
                        return
                except Exception:
                    pass
            time.sleep(0.25)

    def _wait_gone(self, intent_name, *, probe=None, lang="en-US", timeout=10.0):
        """Block until the intent no longer matches — padatious detaches then
        re-trains the model asynchronously, so the no-match assertion must wait
        for the removed intent to drop out of the (re-trained) container."""
        probe = probe or intent_name.split(":")[-1]
        deadline = time.time() + timeout
        while time.time() < deadline:
            if intent_name not in self.pipeline.registered_intents:
                try:
                    match = self.pipeline.calc_intent([probe], lang)
                    if match is None or getattr(match, "name", None) != intent_name:
                        return
                except Exception:
                    return
            time.sleep(0.25)

    def _capture_match(self, utterance, intent_name, timeout=_MATCH_TIMEOUT,
                       attempts=4):
        expected = [f"{self.SKILL_ID}:{intent_name}"]
        for _ in range(attempts):
            msg = self.send_and_capture(utterance, expected_types=expected,
                                        timeout=timeout)
            if msg is not None:
                return msg
            time.sleep(0.5)
        return None

    def _emit(self, topic, intent_name=None, settle=1.5, **extra):
        data = {"skill_id": self.SKILL_ID, "lang": "en-US"}
        if intent_name is not None:
            data["intent_name"] = intent_name
        data.update(extra)
        self.bus.emit(Message(topic, data, {"skill_id": self.SKILL_ID}))
        time.sleep(settle)

    # -- §6 spec template registration is matchable ---------------------

    def test_spec_template_registration_is_matchable(self):
        self._register_template("hello", _HELLO)
        msg = self._capture_match("hello", "hello")
        self.assertIsNotNone(msg, "expected intent match from spec registration")
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:hello")

    def test_spec_template_second_intent_matchable(self):
        self._register_template("bye", _BYE)
        msg = self._capture_match("goodbye", "bye")
        self.assertIsNotNone(msg, "second template should match")

    # -- back-compat: legacy registration still matches -----------------

    def test_legacy_template_registration_still_matches(self):
        self.bus.emit(Message("padatious:register_intent", {
            "name": f"{self.SKILL_ID}:hello", "samples": _HELLO, "lang": "en-US",
        }, {"skill_id": self.SKILL_ID}))
        time.sleep(0.5)
        self._wait_trained(f"{self.SKILL_ID}:hello", probe="hello")
        msg = self._capture_match("hello", "hello")
        self.assertIsNotNone(msg, "legacy registration must still match")

    # -- §8.2 / §8.4 deregistration -------------------------------------

    def test_spec_deregister_removes_intent(self):
        self._register_template("greet_d", _HELLO)
        self.assertIsNotNone(
            self._capture_match("hello", "greet_d"),
            "sanity: intent should match before deregister",
        )
        self._emit(INTENT_DEREGISTER, "greet_d")
        self._wait_gone(f"{self.SKILL_ID}:greet_d", probe="hello")
        self.expect_no_match("hello", timeout=4.0)

    def test_spec_skill_deregister_removes_intent(self):
        self._register_template("greet_s", _HELLO)
        self._emit(SKILL_DEREGISTER)
        self._wait_gone(f"{self.SKILL_ID}:greet_s", probe="hello")
        self.expect_no_match("hello", timeout=4.0)

    # -- §8.5 disable / enable ------------------------------------------

    def test_spec_disable_suppresses_intent(self):
        """Padatious has no native suppression flag; disable detaches the intent
        from the container (retaining its definition) — registration-scoped
        suppression per §8.5."""
        self._register_template("greet_x", _HELLO)
        self._emit(INTENT_DISABLE, "greet_x")
        self._wait_gone(f"{self.SKILL_ID}:greet_x", probe="hello")
        self.expect_no_match("hello", timeout=4.0)

    def test_spec_enable_rearms_intent(self):
        """Enable re-registers from the retained definition (§8.5)."""
        self._register_template("greet_e", _HELLO)
        self._emit(INTENT_DISABLE, "greet_e")
        self._emit(INTENT_ENABLE, "greet_e")
        self._wait_trained(f"{self.SKILL_ID}:greet_e", probe="hello")
        msg = self._capture_match("hello", "greet_e")
        self.assertIsNotNone(msg, "intent should match again after enable")

    # -- §11 negative: template engine ignores the keyword topic --------

    def test_keyword_topic_does_not_match_on_template_engine(self):
        self.bus.emit(Message(REGISTER_KEYWORD, {
            "skill_id": self.SKILL_ID,
            "intent_name": "lights_off",
            "lang": "en-US",
            "required": [{"name": "TurnOff", "samples": ["off"]},
                         {"name": "Light", "samples": ["lights"]}],
            "optional": [], "one_of": [], "excluded": [],
        }, {"skill_id": self.SKILL_ID}))
        time.sleep(1.0)
        self.expect_no_match("turn off the lights", timeout=4.0)


if __name__ == "__main__":
    unittest.main()
