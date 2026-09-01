"""End-to-end tests for PadatiousPipeline using ovoscope.

Built on top of ovoscope's reusable :class:`E2EPipelineHarness` so this
file only contains padatious-specific concerns (sample-based intent +
entity registration via the `padatious:register_*` bus events).

The pipeline is configured with ``instant_train: true`` so that each
``register_intent`` triggers a synchronous train, making assertions
deterministic without sleeping or listening for ``mycroft.skills.trained``.
"""
import time
import unittest
import uuid
from unittest import mock

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

from ovos_padatious.intent_container import IntentContainer  # noqa: E402
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
        # A registration is only visible for matching once its (possibly
        # background/debounced) compile pass has actually run - see
        # PadatiousPipeline.wait_until_trained. This is a test-only sync
        # point; skills never need it.
        self.pipeline.wait_until_trained(timeout=10.0)

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


class _NonInstantHarness(E2EPipelineHarness):
    """Steady-state harness with NO ``instant_train``: ovoscope's boot
    already leaves ``first_train`` set (the initial empty pass has run),
    so - unlike ``_PadatiousHarness`` above - a subsequent
    ``register_intent`` retrains on the background worker instead of
    blocking the registration call itself. This is what actually exercises
    ``wait_until_trained``'s join, as opposed to ``instant_train`` mode
    where the registration call is already fully synchronous on its own.

    ``SKILL_ID`` carries a per-import random suffix rather than a fixed
    name: this harness's ``PLUGIN_CONFIG`` (unlike the ovoscope config
    that would normally set an isolated ``intent_cache``) is not reliably
    propagated into the pipeline in this environment, so a fixed
    skill/intent name would eventually collect a REAL on-disk hash cache
    (the plugin's default ``~/.local/share/mycroft/intent_cache``) across
    repeated local runs, turning this into a hash-cache-hit replay - whose
    neural object loads synchronously regardless of any training pass, see
    ``test_never_train_on_calling_thread.py`` - and silently defeating the
    entire point of this test."""
    PIPELINE_ID = PIPELINE_ID
    CONFIG_KEY = CONFIG_KEY
    PLUGIN_CONFIG = {"conf_low": 0.6}
    SKILL_ID = f"wait_until_trained_skill_{uuid.uuid4().hex[:8]}"

    pipeline: PadatiousPipeline  # type: ignore[assignment]


class TestWaitUntilTrainedGenuinelyGatesTheQuery(_NonInstantHarness):
    """Pins the dependency end-to-end: a caller that emits a registration
    and then calls ``wait_until_trained`` must not see a match before the
    background pass actually lands, and must see it right after. Adversarial
    review of PR #124 found that gutting ``wait_until_trained`` to
    ``return True`` left all e2e tests in this module green - none of them
    exercised a slow enough training pass, under a harness where
    registration is NOT already synchronous via ``instant_train``, for the
    difference to show up before generous polling elsewhere absorbed it."""

    def test_wait_until_trained_actually_waits_for_a_slow_background_pass(self):
        self.assertFalse(self.pipeline.config.get("instant_train", False))
        # the boot's own initial (empty) pass no longer trains synchronously
        # either (see docs/ovos_pipeline.md); join it deterministically
        # before exercising the actual scenario below.
        self.assertTrue(self.pipeline.wait_until_trained(timeout=10.0),
                         "boot's initial pass never completed")
        self.assertTrue(self.pipeline.first_train.is_set(),
                         "harness precondition: boot's initial pass already ran")

        real_train = IntentContainer.train

        def slow_train(self, *args, **kwargs):
            time.sleep(0.5)
            return real_train(self, *args, **kwargs)

        with mock.patch.object(IntentContainer, "train", slow_train):
            self.bus.emit(Message("padatious:register_intent", {
                "name": f"{self.SKILL_ID}:hello", "samples": _HELLO_SAMPLES,
                "lang": "en-US", "skill_id": self.SKILL_ID,
            }))
            # register/query race window: a genuinely fresh, never-compiled
            # registration must not be matchable before its background pass
            # has run
            self.assertIsNone(
                self.pipeline.calc_intent(["hello"], "en-US"),
                "a fresh registration must not be matchable before its "
                "background pass has run")

            self.assertTrue(self.pipeline.wait_until_trained(timeout=10.0))
            # a single direct query, no retry loop: if wait_until_trained
            # had not genuinely waited for the (artificially slowed)
            # background pass, this would very likely still be None
            match = self.pipeline.calc_intent(["hello"], "en-US")
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"{self.SKILL_ID}:hello")


if __name__ == "__main__":
    unittest.main()
