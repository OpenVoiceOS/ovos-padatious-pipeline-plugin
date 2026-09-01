"""Identical re-registration must be a no-op; a genuine burst of
registrations must coalesce into one retrain; getters must keep answering
while a train pass is deliberately blocked.

Field evidence (ser9, ovos-core 2.0.11a1): ovos-core's skill intent
registration reconciliation re-emits every already-registered, UNCHANGED
intent on a roughly 81s cadence. Before this fix, ``IntentContainer.add_intent``
/``add_entity`` called ``self._set_must_train(must_train)`` using the
*requested* ``must_train`` flag (always ``True`` from the registration path)
instead of whether ``TrainingManager.add`` actually found the content
changed - so every replay of an unchanged intent re-dirtied the container
even though ``TrainingManager.add`` already knew, via its on-disk content
hash, that nothing needed retraining. Each re-dirty spawned another full
compile+train pass; at this skill-set size a pass costs ~70-78s, which is
longer than the 81s re-registration cadence, so the background trainer
(``IntentContainer._background_train_loop``) never went idle: CPU pinned
92-99%, back-to-back retrains observed indefinitely, and padatious never
answered a match again post-ready.
"""
import shutil
import tempfile
import time
import unittest
from threading import Event
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline
import ovos_padatious.training_manager as training_manager

SKILL_ID = "reregistration.skill"
LANG = "en-US"


def _register_msg(name, samples):
    return Message("padatious:register_intent", {
        "name": name, "samples": samples, "lang": LANG, "skill_id": SKILL_ID,
    })


class TestIdenticalReplayIsANoOp(unittest.TestCase):
    """Replaying the exact same registration must not touch ``must_train``
    or trigger a retrain - the ser9 reconciliation-loop scenario."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.tmp})
        self.pipeline.first_train.set()

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _register_and_settle(self, name, samples):
        self.pipeline.register_intent(_register_msg(name, samples))
        deadline = time.monotonic() + 5.0
        while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_replaying_identical_registrations_never_retrains(self):
        self._register_and_settle(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self._register_and_settle(f"{SKILL_ID}:bye", ["bye", "see you"])

        train_calls = []
        real_train = IntentContainer.train

        def counting_train(self, *args, **kwargs):
            train_calls.append(1)
            return real_train(self, *args, **kwargs)

        with mock.patch.object(IntentContainer, "train", counting_train):
            # replay the exact same two registrations twice, back to back -
            # mirrors two reconciliation passes seeing no actual changes
            for _ in range(2):
                self.pipeline.register_intent(_register_msg(f"{SKILL_ID}:hello", ["hello", "hi there"]))
                self.pipeline.register_intent(_register_msg(f"{SKILL_ID}:bye", ["bye", "see you"]))
                self.assertFalse(
                    self.pipeline.containers[LANG].must_train,
                    "an identical re-registration must never dirty the container")

            time.sleep(0.2)  # give any (wrongly) spawned background worker a chance to run
            self.assertEqual(train_calls, [],
                              "identical replays must never trigger a retrain")

    def test_one_changed_intent_among_a_replay_triggers_exactly_one_retrain(self):
        self._register_and_settle(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self._register_and_settle(f"{SKILL_ID}:bye", ["bye", "see you"])

        train_calls = []
        real_train = IntentContainer.train

        def counting_train(self, *args, **kwargs):
            train_calls.append(1)
            return real_train(self, *args, **kwargs)

        with mock.patch.object(IntentContainer, "train", counting_train):
            self.pipeline.register_intent(_register_msg(f"{SKILL_ID}:hello", ["hello", "hi there"]))
            # genuinely new sample line for "bye"
            self.pipeline.register_intent(
                _register_msg(f"{SKILL_ID}:bye", ["bye", "see you", "farewell"]))

            deadline = time.monotonic() + 5.0
            while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
                time.sleep(0.02)

        self.assertEqual(len(train_calls), 1,
                          f"expected exactly one retrain, got {len(train_calls)}")
        match = self.pipeline.calc_intent("farewell", lang=LANG)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"{SKILL_ID}:bye")


class TestRegistrationBurstCoalesces(unittest.TestCase):
    """A burst of distinct new registrations arriving faster than a single
    train pass completes must coalesce into (close to) one retrain, not
    one retrain per registration."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.tmp})
        self.pipeline.first_train.set()

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_burst_of_registrations_coalesces_into_one_retrain(self):
        # shrink the debounce window so the burst below (fired well within
        # it) coalesces quickly instead of the test waiting out the real
        # (much larger) production window
        with mock.patch.object(IntentContainer, "_TRAIN_DEBOUNCE_S", 0.05), \
             mock.patch.object(IntentContainer, "_TRAIN_MAX_DEFER_S", 2.0):
            train_calls = []
            real_train = IntentContainer.train

            def counting_train(self, *args, **kwargs):
                train_calls.append(1)
                return real_train(self, *args, **kwargs)

            with mock.patch.object(IntentContainer, "train", counting_train):
                words = ["zero", "one", "two", "three", "four", "five"]
                for i, word in enumerate(words):
                    self.pipeline.register_intent(
                        _register_msg(f"{SKILL_ID}:burst{i}", [f"tell me about {word} please"]))

                deadline = time.monotonic() + 5.0
                while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
                    time.sleep(0.02)

        self.assertFalse(self.pipeline.containers[LANG].must_train)
        # 6 registrations fired near-instantly must not cost 6 separate
        # passes; allow slack for timing but this must be well under 6
        self.assertLessEqual(len(train_calls), 3,
                              f"expected a coalesced retrain, got {len(train_calls)} passes")
        words = ["zero", "one", "two", "three", "four", "five"]
        for i, word in enumerate(words):
            match = self.pipeline.calc_intent(f"tell me about {word} please", lang=LANG)
            self.assertIsNotNone(match)
            self.assertEqual(match.name, f"{SKILL_ID}:burst{i}")


class TestGettersRespondWhileTrainIsBlocked(unittest.TestCase):
    """The manifest getters must keep answering even while a train pass is
    stuck mid-compile (Event-blocked ``_train_and_save``, mirrors #118)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.tmp})
        self.pipeline.first_train.set()

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_manifest_get_responds_while_train_blocked_mid_pass(self):
        started = Event()
        release = Event()
        real_train_and_save = training_manager._train_and_save

        def blocking_train_and_save(*args, **kwargs):
            started.set()
            release.wait(timeout=5.0)
            return real_train_and_save(*args, **kwargs)

        training_manager._train_and_save = blocking_train_and_save
        try:
            self.pipeline.register_intent(
                _register_msg(f"{SKILL_ID}:hello", ["hello", "hi there"]))
            self.assertTrue(started.wait(timeout=5.0),
                             "background train pass never started")

            replies = []
            self.pipeline.bus.on("intent.service.padatious.manifest",
                                 lambda m: replies.append(m))
            start = time.monotonic()
            self.pipeline.handle_padatious_manifest(
                Message("intent.service.padatious.manifest.get", {}))
            elapsed = time.monotonic() - start

            self.assertLess(elapsed, 2.0,
                             f"manifest.get blocked for {elapsed:.2f}s on a stuck train pass")
            self.assertEqual(len(replies), 1)
            self.assertIn(f"{SKILL_ID}:hello", replies[0].data["intents"])
        finally:
            release.set()
            training_manager._train_and_save = real_train_and_save
            deadline = time.monotonic() + 5.0
            while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
                time.sleep(0.02)


class TestMatchingServesPreviousGenerationDuringRetrain(unittest.TestCase):
    """A CHANGED, already-live intent must keep matching against its
    previous, already-trained content for the entire compile window of its
    own retrain - not go dark the instant the registration lands.

    Before this fix, ``TrainingManager.add`` evicted the live, trained
    object for a name up front (a blanket ``self.remove(name)``) the moment
    a registration for that name landed, and only re-added a (new) object
    once training for the whole batch finished. So any query for that
    intent arriving between "registration lands" and "batch pass
    completes" - which, at ser9's boot scale, is a window of tens of
    seconds to minutes for the whole registration drain - saw the intent as
    simply gone."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # padaos (the regex/exact-match sibling index) is updated
        # immediately and unconditionally at registration time regardless
        # of training state, so it would mask a neural-only regression;
        # disable it here to isolate the neural (fann) matching path this
        # test targets.
        self.pipeline = PadatiousPipeline(
            FakeBus(), config={"intent_cache": self.tmp, "disable_padaos": True})
        self.pipeline.first_train.set()

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_query_matches_previous_content_while_its_own_retrain_is_in_flight(self):
        name = f"{SKILL_ID}:hello"
        self.pipeline.register_intent(_register_msg(name, ["hello", "hi there"]))
        deadline = time.monotonic() + 5.0
        while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
            time.sleep(0.02)
        match = self.pipeline.calc_intent("hello", lang=LANG)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, name)

        started = Event()
        release = Event()
        real_train_and_save = training_manager._train_and_save

        def blocking_train_and_save(*args, **kwargs):
            started.set()
            release.wait(timeout=5.0)
            return real_train_and_save(*args, **kwargs)

        training_manager._train_and_save = blocking_train_and_save
        try:
            # genuinely changed content for the SAME name - forces a real
            # retrain of an intent that was already live and matchable
            self.pipeline.register_intent(
                _register_msg(name, ["hello", "hi there", "greetings"]))
            self.assertTrue(started.wait(timeout=5.0),
                             "retrain of the changed intent never started")

            # the compile is stuck mid-pass right now; a query against the
            # OLD, still-valid sample must still succeed
            # a fresh, previously-unqueried utterance - the lru_cache in
            # ovos_padatious.opm._calc_padatious_intent (cleared only once
            # a train pass fully completes) would otherwise mask a
            # regression here by replaying an earlier cached result
            match = self.pipeline.calc_intent("hi there", lang=LANG)
            self.assertIsNotNone(
                match, "the previous generation must keep answering matches "
                       "while its own retrain is still in flight")
            self.assertEqual(match.name, name)
        finally:
            release.set()
            training_manager._train_and_save = real_train_and_save
            deadline = time.monotonic() + 5.0
            while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
                time.sleep(0.02)

        # once the retrain lands, the NEW sample must also match
        match = self.pipeline.calc_intent("greetings", lang=LANG)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, name)


if __name__ == '__main__':
    unittest.main()
