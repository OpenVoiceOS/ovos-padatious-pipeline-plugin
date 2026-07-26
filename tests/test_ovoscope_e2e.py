"""End-to-end tests for PadatiousPipeline using ovoscope.

Built on top of ovoscope's reusable :class:`E2EPipelineHarness` so this
file only contains padatious-specific concerns (sample-based intent +
entity registration via the `padatious:register_*` bus events).

The pipeline is configured with ``instant_train: true`` so that each
``register_intent`` triggers a synchronous train, making assertions
deterministic without sleeping or listening for ``mycroft.skills.trained``.
"""
import unittest

import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402

from ovoscope import (  # noqa: E402
    E2EPipelineHarness,
    detach_intent,
    detach_skill,
    make_session,
    register_padatious_entity,
)

from ovos_padatious.opm import PadatiousPipeline  # noqa: E402

PIPELINE_ID = "ovos-padatious-pipeline-plugin"
CONFIG_KEY = "padatious"

_HELLO_SAMPLES = ["hello", "hi", "hey", "greetings", "good morning"]
_BYE_SAMPLES = ["goodbye", "bye", "see you later", "farewell", "take care"]
_LIGHTS_ON_SAMPLES = ["turn on the lights", "switch on lights", "lights on please"]


class _PadatiousHarness(E2EPipelineHarness):
    PIPELINE_ID = PIPELINE_ID
    CONFIG_KEY = CONFIG_KEY
    # instant_train avoids the async train flow — every register_intent
    # call retrains synchronously so the test does not race the trainer.
    # conf_low default is permissive enough that unrelated utterances
    # match against trained intents; raise it to a stricter threshold so
    # 'set a timer for five minutes' does not match a 'hello' intent.
    PLUGIN_CONFIG = {"instant_train": True, "conf_low": 0.6}
    SKILL_ID = "test_skill_padatious"

    pipeline: PadatiousPipeline  # type: ignore[assignment]

    def _register_intent(self, name, samples, lang="en-US"):
        # Padatious tracks intent -> skill_id in an internal _skill2intent
        # map populated at register time; without it, detach_skill cannot
        # remove the intent. Inject skill_id (the prefix before ":" in the
        # intent name) so the ovoscope harness's per-test cleanup works.
        skill_id = name.split(":", 1)[0]
        self.bus.emit(Message("padatious:register_intent", {
            "name": name, "samples": samples, "lang": lang,
            "skill_id": skill_id,
        }))
        import time
        time.sleep(0.1)

    def _register_entity(self, name, samples):
        register_padatious_entity(self.bus, name, samples)


class TestRegisteredIntentMatch(_PadatiousHarness):
    def test_exact_utterance_dispatches_intent(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture(
            "hello", expected_types=[f"{self.SKILL_ID}:hello"], timeout=10.0
        )
        self.assertIsNotNone(msg, "expected intent match on bus")
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:hello")
        self.assertEqual(msg.data.get("utterance"), "hello")

    def test_no_match_when_no_intents_registered(self):
        self.expect_no_match("hello")

    def test_no_match_unrelated_utterance(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        self.expect_no_match("set a timer for five minutes")

    def test_best_intent_selected_among_multiple(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        self._register_intent(f"{self.SKILL_ID}:bye", _BYE_SAMPLES)
        msg = self.send_and_capture(
            "goodbye", expected_types=[f"{self.SKILL_ID}:bye"], timeout=10.0
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:bye")


class TestEntityExtraction(_PadatiousHarness):
    def test_entity_slot_captured_in_match(self):
        self._register_entity("item", ["milk", "bread", "eggs", "cheese"])
        self._register_intent(
            f"{self.SKILL_ID}:buy",
            ["buy {item}", "get {item}", "purchase {item}"],
        )
        msg = self.send_and_capture(
            "buy milk", expected_types=[f"{self.SKILL_ID}:buy"], timeout=10.0
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, f"{self.SKILL_ID}:buy")


class TestDetach(_PadatiousHarness):
    def test_detach_intent_prevents_match(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        msg = self.send_and_capture(
            "hello", expected_types=[f"{self.SKILL_ID}:hello"], timeout=10.0
        )
        self.assertIsNotNone(msg)

        detach_intent(self.bus, f"{self.SKILL_ID}:hello")
        self.expect_no_match("hello")

    def test_detach_skill_removes_all_its_intents(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        self._register_intent(f"{self.SKILL_ID}:bye", _BYE_SAMPLES)
        self._register_intent("skill_b_padatious:lights_on", _LIGHTS_ON_SAMPLES)

        detach_skill(self.bus, self.SKILL_ID)

        self.expect_no_match("hello")
        self.expect_no_match("goodbye")
        msg = self.send_and_capture(
            "turn on the lights",
            expected_types=["skill_b_padatious:lights_on"],
            timeout=10.0,
        )
        self.assertIsNotNone(msg, "skill_b intent should survive skill_a detach")
        detach_skill(self.bus, "skill_b_padatious")


class TestSessionBlacklist(_PadatiousHarness):
    def test_blacklisted_intent_is_skipped(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        sess = make_session(
            "bl-intent-test",
            blacklisted_intents=[f"{self.SKILL_ID}:hello"],
        )
        self.expect_no_match("hello", session=sess, timeout=3.0)

    def test_blacklisted_skill_is_skipped(self):
        self._register_intent(f"{self.SKILL_ID}:hello", _HELLO_SAMPLES)
        sess = make_session(
            "bl-skill-test",
            blacklisted_skills=[self.SKILL_ID],
        )
        self.expect_no_match("hello", session=sess, timeout=3.0)


class TestSessionBlacklistAlias(_PadatiousHarness):
    """ovos-workshop >= 9.3 dual-registers each ``.intent`` file under both
    the legacy ``<skill_id>:<file>.intent`` id and the OVOS-INTENT-4
    canonical ``<skill_id>:<file>`` id (ovos-core#831). The plugin collapses
    that alias onto one canonical engine entry at REGISTRATION time (see
    ``PadatiousPipeline.register_intent``/``handle_register_template``), so
    a session blacklist entry naming either alias must suppress the single
    canonical match, per OVOS-PIPELINE-1 §5.4. Here both messages go through
    the legacy topic with different names to exercise the blacklist
    canonicalization path directly; ``tests/test_registration_collapse.py``
    covers the registration-time collapse across both wire topics.
    """

    LEGACY_NAME = f"{_PadatiousHarness.SKILL_ID}:hello.intent"
    NEW_NAME = f"{_PadatiousHarness.SKILL_ID}:hello"

    def _register_both_aliases(self):
        self._register_intent(self.LEGACY_NAME, _HELLO_SAMPLES)
        self._register_intent(self.NEW_NAME, _HELLO_SAMPLES)

    def test_blacklisting_legacy_id_suppresses_new_alias(self):
        self._register_both_aliases()
        sess = make_session(
            "bl-alias-legacy-test",
            blacklisted_intents=[self.LEGACY_NAME],
        )
        self.expect_no_match("hello", session=sess, timeout=3.0)

    def test_blacklisting_new_id_suppresses_legacy_alias(self):
        self._register_both_aliases()
        sess = make_session(
            "bl-alias-new-test",
            blacklisted_intents=[self.NEW_NAME],
        )
        self.expect_no_match("hello", session=sess, timeout=3.0)

    def test_non_blacklisted_intent_still_matches(self):
        self._register_both_aliases()
        self._register_intent(f"{self.SKILL_ID}:bye", _BYE_SAMPLES)
        sess = make_session(
            "bl-alias-unrelated-test",
            blacklisted_intents=[f"{self.SKILL_ID}:bye"],
        )
        msg = self.send_and_capture(
            "hello",
            expected_types=[self.LEGACY_NAME, self.NEW_NAME],
            session=sess,
            timeout=10.0,
        )
        self.assertIsNotNone(msg)


if __name__ == "__main__":
    unittest.main()
