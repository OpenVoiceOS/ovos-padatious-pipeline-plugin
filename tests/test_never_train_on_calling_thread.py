"""Nothing in this plugin may ever train/compile on the thread that made a
query or a registration - not even a container's very first pass ever, and
not even under `domain_engine: true`. A container (or DomainIntentContainer
and its per-domain sub-containers) that has never compiled is served empty
(no match) until the background worker's first pass actually lands, the
same "serve stale/empty while a pass is in flight" rule everywhere else in
this file already follows.

Adversarial review of PR #124 (round 8) reproduced the defect one call
frame up from where that PR's fix stopped: ``IntentContainer.calc_intents``
(the match path) itself never compiles anymore, but it calls
``_train_in_background()`` first, and that method's own "no
previously-trained state" special case called ``self.train()``
*synchronously* - so a fresh boot whose registrations are all hash-cache
hits (never dirtying ``must_train``, but always dirtying
``padaos.must_compile`` - see ``test_padaos_replay_background_compile.py``)
paid the full padaos compile on the bus/query thread the moment its first
live query arrived, reproducing the exact ser9 field defect this whole
series of fixes targets. ``domain_engine: true`` kept the same defect
wholesale via its own separate, always-inline
``if self.must_train: self.train()`` on every ``calc_domain(s)``/
``calc_intent(s)`` call.
"""
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious import padaos
from ovos_padatious.domain_container import DomainIntentContainer
from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline


class TestNeverTrainedContainerNeverCompilesOnCallerThread(unittest.TestCase):
    """The round-8 warm-cache no-op-replay scenario: a FRESH container
    instance (``_ever_trained`` False) replays registrations whose
    ``.hash`` sidecars already match on disk."""

    def test_warm_cache_replay_first_query_never_trains_on_caller_thread(self):
        cache = tempfile.mkdtemp()
        try:
            first = IntentContainer(cache)
            first.add_intent("hello", ["hello", "hi there"])
            first.train(False)  # writes the .hash sidecar for "hello"

            # a brand new process/container instance against the SAME
            # cache_dir: this registration is a pure hash-cache hit
            second = IntentContainer(cache)
            second.add_intent("hello", ["hello", "hi there"])
            self.assertFalse(second.must_train,
                              "a hash-cache hit must never dirty must_train")
            self.assertFalse(second._ever_trained)

            caller_thread = threading.current_thread()
            real_train = IntentContainer.train
            real_padaos_compile = padaos.IntentContainer._compile
            violations = []

            def guard_train(self, *a, **kw):
                if threading.current_thread() is caller_thread:
                    violations.append("train() ran on the calling/query thread")
                return real_train(self, *a, **kw)

            def guard_padaos_compile(self, *a, **kw):
                if threading.current_thread() is caller_thread:
                    violations.append("padaos._compile() ran on the "
                                       "calling/query thread")
                return real_padaos_compile(self, *a, **kw)

            with mock.patch.object(IntentContainer, "train", guard_train), \
                 mock.patch.object(padaos.IntentContainer, "_compile", guard_padaos_compile):
                # the match path itself must never call train()/compile()
                # on the caller. Note the neural tier's cache-hit object
                # was already loaded from disk synchronously at
                # registration time (see TrainingManager.add), so a match
                # CAN legitimately come back from it without any train()
                # call at all - only train()/padaos-compile running on
                # THIS thread is the defect under test.
                start = time.monotonic()
                second.calc_intent("hello")
                elapsed = time.monotonic() - start

                self.assertEqual(violations, [],
                                  f"{violations}")
                # A container that has never published a generation waits
                # (bounded, on an Event) for its first pass rather than
                # answering from empty structures, so this first query is
                # expected to take about the debounce window - and no
                # longer. What must never happen is the compile running
                # HERE, which is the `violations` assertion above.
                self.assertLess(elapsed,
                                 IntentContainer._TRAIN_DEBOUNCE_S + 1.0,
                                 "calc_intent took longer than the debounce "
                                 "window the wait exists to cover")
                self.assertFalse(second.needs_compile,
                                  "the query returned before the pass it "
                                  "waited for had actually landed")

                # the background pass must still land on its own
                deadline = time.monotonic() + 10.0
                while second.needs_compile and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(violations, [])
            self.assertFalse(second.needs_compile,
                              "background training never completed")
            match = second.calc_intent("hello")
            self.assertEqual(match.name, "hello")
            self.assertEqual(match.conf, 1.0,
                              "padaos' exact match should apply once compiled")
        finally:
            shutil.rmtree(cache, ignore_errors=True)


class TestDomainEngineNeverCompilesOnCallerThread(unittest.TestCase):
    """Mirrors the above for ``domain_engine: true``
    (``DomainIntentContainer``): add_domain_intent always dirties
    ``must_train`` unconditionally (no hash-aware skip at all at this
    layer), so this is an even more aggressive version of the same
    scenario."""

    def test_warm_cache_replay_never_trains_domain_container_on_caller_thread(self):
        cache = tempfile.mkdtemp()
        try:
            first = DomainIntentContainer(cache_dir=cache)
            first.add_domain_intent("greetings", "hello", ["hello", "hi there"])
            first.train()

            second = DomainIntentContainer(cache_dir=cache)
            second.add_domain_intent("greetings", "hello", ["hello", "hi there"])
            self.assertTrue(second.needs_compile)

            caller_thread = threading.current_thread()
            real_train = DomainIntentContainer.train
            real_padaos_compile = padaos.IntentContainer._compile
            violations = []

            def guard_train(self, *a, **kw):
                if threading.current_thread() is caller_thread:
                    violations.append("DomainIntentContainer.train() ran on "
                                       "the calling/query thread")
                return real_train(self, *a, **kw)

            def guard_padaos_compile(self, *a, **kw):
                if threading.current_thread() is caller_thread:
                    violations.append("padaos._compile() ran on the "
                                       "calling/query thread")
                return real_padaos_compile(self, *a, **kw)

            with mock.patch.object(DomainIntentContainer, "train", guard_train), \
                 mock.patch.object(padaos.IntentContainer, "_compile", guard_padaos_compile):
                # instantiate_from_disk() proactively loads every
                # already-cached neural object synchronously (both here
                # and for the per-domain sub-container), so - exactly as
                # in the plain-IntentContainer case above - a match CAN
                # legitimately come back from the neural tier alone
                # without any train() call. What must never happen is
                # DomainIntentContainer.train() or padaos._compile()
                # running on the caller thread.
                start = time.monotonic()
                second.calc_intent("hello")
                elapsed = time.monotonic() - start

                self.assertEqual(violations, [], f"{violations}")
                self.assertLess(elapsed, 0.5,
                                 "calc_intent blocked on an inline compile")

                deadline = time.monotonic() + 10.0
                while second.needs_compile and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(violations, [])
            self.assertFalse(second.needs_compile,
                              "background training never completed")
            match = second.calc_intent("hello", domain="greetings")
            self.assertEqual(match.name, "hello")
            self.assertEqual(match.conf, 1.0,
                              "padaos' exact match should apply once compiled")
        finally:
            shutil.rmtree(cache, ignore_errors=True)



class TestMycroftSkillsTrainNeverBlocksCallerThread(unittest.TestCase):
    """The ``mycroft.skills.train`` bus handler (``PadatiousPipeline.train``)
    kept its own separate "very first pass blocks the caller" special case
    even after ``IntentContainer._train_in_background`` was fixed to never
    do that: ``not self.first_train.is_set()`` forced a synchronous
    ``_train_sync()`` call for a pipeline's very first training pass,
    stalling the bus-message thread handling ``mycroft.skills.train`` for
    the full compile+train duration - the ~85s ser9 field figure. Only
    ``instant_train`` mode (an explicit, documented, opt-in exception) may
    still train synchronously."""

    def test_first_ever_train_call_never_trains_on_caller_thread(self):
        pipeline = PadatiousPipeline(FakeBus(), config={})
        try:
            lang = pipeline.lang
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": "s:hello", "samples": ["hello", "hi there"],
                "lang": lang, "skill_id": "s",
            }))
            self.assertFalse(pipeline.first_train.is_set(),
                              "test precondition: this must be the very "
                              "first training pass")

            caller_thread = threading.current_thread()
            real_train = IntentContainer.train
            violations = []

            def guard_train(self, *a, **kw):
                if threading.current_thread() is caller_thread:
                    violations.append(
                        "IntentContainer.train() ran on the "
                        "mycroft.skills.train handler thread")
                return real_train(self, *a, **kw)

            with mock.patch.object(IntentContainer, "train", guard_train):
                start = time.monotonic()
                pipeline.train()  # simulates the mycroft.skills.train handler
                elapsed = time.monotonic() - start

                self.assertEqual(violations, [])
                self.assertLess(
                    elapsed, 0.5,
                    "mycroft.skills.train blocked the bus thread on the "
                    "first-ever compile pass")

                self.assertTrue(pipeline.wait_until_trained(timeout=10.0))
                self.assertEqual(violations, [])

            match = pipeline.calc_intent(["hello"], lang)
            self.assertIsNotNone(match)
            self.assertEqual(match.name, "s:hello")
        finally:
            pipeline.shutdown()

    def test_instant_train_is_the_documented_exception(self):
        """instant_train explicitly promises synchronous training; that
        opt-in trade-off is unaffected by the fix above."""
        pipeline = PadatiousPipeline(FakeBus(), config={"instant_train": True})
        try:
            lang = pipeline.lang
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": "s:hello", "samples": ["hello", "hi there"],
                "lang": lang, "skill_id": "s",
            }))
            match = pipeline.calc_intent(["hello"], lang)
            self.assertIsNotNone(match)
            self.assertEqual(match.name, "s:hello")
        finally:
            pipeline.shutdown()


if __name__ == '__main__':
    unittest.main()
