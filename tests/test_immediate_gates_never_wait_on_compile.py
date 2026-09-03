"""Removal and disable/enable are runtime GATES, not compile products, and
must take effect immediately - even though additions only become matchable
once padaos's background compile (see
``test_padaos_replay_background_compile.py``) actually runs.

Round-8 amendment: ovoscope e2e caught that making the match path never
compile inline (``padaos.calc_intents``) broke ``detach_intent``/
``detach_skill``/OVOS-INTENT-4 disable, all of which relied on the very
next query's inline compile to forget a removed/disabled intent. Removal
now drops the padaos-compiled entry for that name immediately
(``padaos.remove_intent``); disable is filtered by name at match time
(``PadatiousPipeline.calc_intent`` folds ``_disabled_intents`` into the
blacklist) instead of waiting on a recompile. A new
``PadatiousPipeline.wait_until_trained`` gives tests/tools a deterministic
way to wait for an ADDITION to become visible without polling internals.
"""
import time
import unittest
from threading import Event, Thread
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus
from ovos_spec_tools import SpecMessage

from ovos_padatious import padaos
from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "immediate.skill"
LANG = "en-US"

INTENT_DISABLE = str(SpecMessage.INTENT_DISABLE)
INTENT_ENABLE = str(SpecMessage.INTENT_ENABLE)
INTENT_REGISTER_TEMPLATE = str(SpecMessage.INTENT_REGISTER_TEMPLATE)


class TestPadaosRemoveIntentIsImmediate(unittest.TestCase):
    """No compile ever needs to run for a removal to take effect - the
    already-compiled entry is dropped from padaos's served state right
    away, on the same thread that called ``remove_intent``."""

    def test_removed_intent_never_matches_again_without_a_compile(self):
        c = padaos.IntentContainer()
        c.add_intent("hello", ["hello", "hi there"])
        c.compile()
        self.assertEqual(len(list(c.calc_intents(" hello "))), 1)

        real_compile = padaos.IntentContainer._compile
        calls = []

        def counting_compile(self, *a, **kw):
            calls.append(1)
            return real_compile(self, *a, **kw)

        with mock.patch.object(padaos.IntentContainer, "_compile", counting_compile):
            c.remove_intent("hello")
            # no compile happened, and the match path never triggers one -
            # yet the removed intent must be gone from the served state
            self.assertEqual(list(c.calc_intents(" hello ")), [])
        self.assertEqual(calls, [], "removal must not require a recompile")


class _PipelineCase(unittest.TestCase):
    def setUp(self):
        self.pipeline = PadatiousPipeline(FakeBus(), config={"instant_train": True})
        self.lang = self.pipeline.lang

    def tearDown(self):
        self.pipeline.shutdown()

    def _register(self, name, samples):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": name, "samples": samples, "lang": self.lang, "skill_id": SKILL_ID,
        }))


class TestDisableSuppressesImmediately(_PipelineCase):
    def test_disable_blocks_the_very_next_query_before_any_recompile_starts(self):
        """Suppression is a pure name-membership check independent of
        compile state - proven here by disabling and querying BEFORE the
        instant_train retrain triggered by disable has even run, via a
        blocked ``_compile``."""
        self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self.assertIsNotNone(self.pipeline.calc_intent(["hello"], self.lang))

        from threading import Event
        started = Event()
        release = Event()
        real_compile = padaos.IntentContainer._compile

        def blocking_compile(self):
            started.set()
            release.wait(timeout=5.0)
            return real_compile(self)

        with mock.patch.object(padaos.IntentContainer, "_compile", blocking_compile):
            from threading import Thread
            t = Thread(
                target=self.pipeline.handle_disable_intent_spec,
                args=(Message(INTENT_DISABLE, {
                    "skill_id": SKILL_ID, "intent_name": "hello",
                }),),
                daemon=True,
            )
            t.start()
            self.assertTrue(started.wait(timeout=5.0),
                             "disable's instant_train retrain never started")
            try:
                # the retrain triggered by disable is stuck mid-compile
                # RIGHT NOW; suppression must already be in effect
                match = self.pipeline.calc_intent(["hello"], self.lang)
                self.assertIsNone(
                    match, "a disabled intent must stop matching immediately, "
                           "before its own retrain even finishes")
            finally:
                release.set()
                t.join(timeout=5.0)

    def test_disable_suppresses_via_blacklist_even_if_removal_is_a_noop(self):
        """Isolates the NAME-BASED suppression path
        (``PadatiousPipeline.calc_intent`` folding ``_disabled_intents``
        into the match-time blacklist) from padaos's own immediate
        removal (covered separately by
        ``TestPadaosRemoveIntentIsImmediate`` and
        ``test_disable_blocks_the_very_next_query_before_any_recompile_starts``).
        With both containers' ``remove_intent`` patched to a no-op, the
        ONLY thing that can stop a disabled intent from matching is the
        blacklist - this fails if that mechanism is missing/broken even
        though the compiled regex and the trained neural object are both
        still fully intact."""
        self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self.assertIsNotNone(self.pipeline.calc_intent(["hello"], self.lang))

        with mock.patch.object(padaos.IntentContainer, "remove_intent", lambda self, name: None), \
             mock.patch.object(IntentContainer, "remove_intent", lambda self, name: None):
            self.pipeline.handle_disable_intent_spec(Message(INTENT_DISABLE, {
                "skill_id": SKILL_ID, "intent_name": "hello",
            }))
            match = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNone(
            match, "disable must suppress via the blacklist even when the "
                   "underlying removal is a no-op")

    def test_enable_rearms_after_the_background_pass_lands(self):
        self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self.pipeline.handle_disable_intent_spec(Message(INTENT_DISABLE, {
            "skill_id": SKILL_ID, "intent_name": "hello",
        }))
        self.assertIsNone(self.pipeline.calc_intent(["hello"], self.lang))

        self.pipeline.handle_enable_intent_spec(Message(INTENT_ENABLE, {
            "skill_id": SKILL_ID, "intent_name": "hello",
        }))
        # enable is effectively an ADDITION (re-registers from the retained
        # definition) - it is fine, and expected, for this to need the
        # deterministic sync helper rather than being instantaneous
        self.assertTrue(self.pipeline.wait_until_trained(timeout=10.0))
        match = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"{SKILL_ID}:hello")


class TestWaitUntilTrained(_PipelineCase):
    def test_wait_until_trained_blocks_until_addition_is_matchable(self):
        self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self.assertTrue(self.pipeline.wait_until_trained(timeout=10.0))
        self.assertFalse(self.pipeline.containers[self.lang].needs_compile)
        match = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNotNone(match)

    def test_wait_until_trained_returns_false_on_timeout(self):
        # a container that can never finish compiling (patched _compile is
        # a no-op that never clears must_compile) must make
        # wait_until_trained give up at the deadline rather than hang
        with mock.patch.object(padaos.IntentContainer, "_compile", lambda self: None):
            self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
            start = time.monotonic()
            result = self.pipeline.wait_until_trained(timeout=0.3)
            elapsed = time.monotonic() - start
        self.assertFalse(result)
        self.assertLess(elapsed, 5.0)


class TestWaitUntilTrainedHonoursTimeoutWhilePassInFlight(unittest.TestCase):
    """``wait_until_trained`` must give up at its deadline even while a
    pass is ALREADY in flight - the previous implementation called
    ``_train_sync`` in a loop, whose untimed
    ``finished_training_event.wait()`` made ``timeout`` a no-op whenever a
    pass was in progress (probe: ``wait_until_trained(2.0)`` had not
    returned after 8s)."""

    def setUp(self):
        # steady state (no instant_train): a registration's own retrain
        # goes to the background worker instead of blocking register_intent
        # itself, so this test can get a pass genuinely in flight before it
        # ever calls wait_until_trained.
        self.pipeline = PadatiousPipeline(FakeBus(), config={})
        self.pipeline.first_train.set()
        self.lang = self.pipeline.lang

    def tearDown(self):
        self.pipeline.shutdown()

    def test_wait_until_trained_gives_up_at_deadline_while_blocked(self):
        started = Event()
        release = Event()
        real_compile = padaos.IntentContainer._compile

        def blocking_compile(self):
            started.set()
            release.wait(timeout=10.0)
            return real_compile(self)

        try:
            with mock.patch.object(padaos.IntentContainer, "_compile", blocking_compile):
                self.pipeline.register_intent(Message("padatious:register_intent", {
                    "name": f"{SKILL_ID}:hello", "samples": ["hello", "hi there"],
                    "lang": self.lang, "skill_id": SKILL_ID,
                }))
                self.assertTrue(started.wait(timeout=5.0),
                                 "background pass never started")

                start = time.monotonic()
                result = self.pipeline.wait_until_trained(timeout=0.5)
                elapsed = time.monotonic() - start

                self.assertFalse(
                    result, "wait_until_trained must return False once its "
                            "deadline elapses, even mid-pass")
                self.assertLess(
                    elapsed, 2.0,
                    f"wait_until_trained ignored its timeout, blocked for "
                    f"{elapsed:.2f}s while a pass was in flight")
        finally:
            release.set()

        # once released, the pass completes on its own and a real wait
        # (no pass blocked) succeeds
        self.assertTrue(self.pipeline.wait_until_trained(timeout=10.0))
        match = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNotNone(match)


if __name__ == '__main__':
    unittest.main()
