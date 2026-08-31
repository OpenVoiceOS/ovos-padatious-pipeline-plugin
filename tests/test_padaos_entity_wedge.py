"""padaos entity alternations must stay bounded and training must never
block the query thread.

Field stack (ser9 production wedge): handle_utterance -> calc_intent ->
IntentContainer.calc_intents -> train() [must_train] -> padaos compile ->
_create_regex -> re.compile stuck for minutes. padaos inlines every value of
an entity into a ``(a|b|c|...)`` alternation and substitutes that verbatim
into EVERY intent line referencing the slot; an auto-registered entity with
thousands of multi-word values, referenced from a few dozen intent lines,
produces megabyte regex sources whose compilation dominates wall time. On
top of that, training used to run inline on the utterance thread the moment
``must_train`` was set, so the first query after a registration wave paid
the full compile cost.

These tests pin:
* an entity under ``PADAOS_ENTITY_INLINE_CAP`` compiles byte-identically to
  the unbounded implementation (no regression for the common case);
* an entity over the cap falls back to the plain wildcard capture instead
  of being inlined, and compiles in a small fraction of the time;
* a listed value of an over-cap entity still scores exactly 1.0 end-to-end
  through the neural tier's exact-sample path, while an unlisted value
  scores strictly lower (comparative only, never an absolute band);
* training triggered by a query never blocks that query once the container
  has trained once before.
"""
import shutil
import threading
import tempfile
import time
import unittest

from ovos_padatious import padaos, IntentContainer
import ovos_padatious.training_manager as training_manager
from ovos_padatious.padaos import PADAOS_ENTITY_INLINE_CAP


class TestPadaosEntityCapUnderCap(unittest.TestCase):
    """Small entities must behave exactly as before the cap was introduced."""

    def test_under_cap_produces_identical_regex(self):
        values = ["london", "porto", "lisbon", "madrid"]
        self.assertLess(len(values), PADAOS_ENTITY_INLINE_CAP)

        c = padaos.IntentContainer()
        c.add_entity("city", values)
        c.add_intent("go", ["travel to {city}"])
        c.compile()

        expected = r'({})'.format('|'.join(
            c._create_pattern(line) for line in values
        ))
        self.assertEqual(c.entities["city"], expected)

        # matching behaves as an exact inline alternation: a listed city
        # matches, an unlisted one does not.
        matches = list(c.calc_intents(" travel to lisbon "))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['entities'].get('city'), 'lisbon')

        no_match = list(c.calc_intents(" travel to nowhereville "))
        self.assertEqual(no_match, [])


class TestPadaosEntityCapOverCap(unittest.TestCase):
    """Large entities must not be inlined; padaos falls back to wildcard."""

    def setUp(self):
        self.n = PADAOS_ENTITY_INLINE_CAP + 20
        self.values = [f"thing number {i:05d} extra words here" for i in range(self.n)]
        self.c = padaos.IntentContainer()
        self.c.add_entity("item", self.values)
        self.c.add_intent("get", ["fetch {item} now"])
        self.c.compile()

    def test_over_cap_entity_not_inlined(self):
        self.assertNotIn("item", self.c.entities)

    def test_over_cap_falls_back_to_wildcard_match(self):
        # the wildcard capture still matches *some* value (any value), it
        # just no longer discriminates listed vs unlisted at the padaos tier
        matches = list(self.c.calc_intents(" fetch " + self.values[0] + " now "))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['entities'].get('item'), self.values[0])

        matches_unlisted = list(self.c.calc_intents(" fetch something else entirely now "))
        self.assertEqual(len(matches_unlisted), 1)


class TestExactScoringSurvivesCapping(unittest.TestCase):
    """End-to-end via IntentContainer: listed values still score exactly
    1.0 through Entity.samples regardless of padaos capping the value list."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.n = PADAOS_ENTITY_INLINE_CAP + 20
        self.values = [f"thing number {i:05d} extra words here" for i in range(self.n)]
        self.container = IntentContainer(self.tmp)
        self.container.add_entity("item", self.values)
        self.container.add_intent("get", ["fetch {item} now", "grab {item}"])
        self.container.train(debug=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_in_list_value_scores_one(self):
        matches = self.container.calc_intents("fetch " + self.values[3] + " now")
        best = max(matches, key=lambda m: m.conf)
        self.assertEqual(best.conf, 1.0)

    def test_out_of_list_scores_strictly_below_in_list(self):
        in_list = self.container.calc_intents("fetch " + self.values[3] + " now")
        in_list_conf = max(m.conf for m in in_list)

        out_of_list = self.container.calc_intents("fetch something totally unlisted now")
        out_of_list_conf = max((m.conf for m in out_of_list), default=0.0)

        self.assertEqual(in_list_conf, 1.0)
        self.assertLess(out_of_list_conf, in_list_conf)


class TestPadaosCompileTimeScalesDown(unittest.TestCase):
    """Regression for the wedge itself: capping keeps compile fast even
    when many intent lines reference a very large entity."""

    def test_many_referencing_lines_over_two_large_entities_compile_fast(self):
        n = 2200
        values = [f"thing number {i:05d} extra words here" for i in range(n)]
        values2 = [f"other value {i:05d} more words too" for i in range(n)]
        c = padaos.IntentContainer()
        c.add_entity("item", values)
        c.add_entity("item2", values2)
        lines = [f"fetch {{item}} and {{item2}} now variant {j}" for j in range(260)]
        c.add_intent("get", lines)

        start = time.monotonic()
        c.compile()
        duration = time.monotonic() - start

        # unbounded, this configuration takes >30s (verified against dev);
        # capped, it should complete in a couple of seconds.
        self.assertLess(duration, 5.0)


class TestTrainingOffUtteranceThread(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_query_after_registration_burst_is_not_blocked_by_retrain(self):
        container = IntentContainer(self.tmp)
        container.add_intent("hello", ["hello there", "hi there"])
        container.train(debug=False)  # first-ever train, blocks once

        # simulate a registration burst that sets must_train again
        container.add_intent("bye", ["goodbye now", "see you later"])
        self.assertTrue(container.must_train)

        start = time.monotonic()
        matches = container.calc_intents("hello there")
        duration = time.monotonic() - start

        # answered from the previously trained state, must not pay for a
        # retrain inline
        self.assertLess(duration, 1.0)
        self.assertTrue(any(m.name == "hello" for m in matches))

        # the background retrain is now running (or has just finished);
        # join it explicitly and confirm the new intent becomes visible
        trainer = container._background_trainer
        self.assertIsNotNone(trainer)
        trainer.join(timeout=30)
        self.assertFalse(container.must_train)

        matches = container.calc_intents("goodbye now")
        self.assertTrue(any(m.name == "bye" for m in matches))


class TestTrainGenerationRace(unittest.TestCase):
    """A registration landing after a background train's snapshot but
    before it finishes must never be stranded.

    ``must_train`` used to be both the dirty bit and the field
    unconditionally cleared by ``train()`` on completion: ``TrainingManager
    .train()`` takes a snapshot copy of ``objects_to_train`` before doing
    any work, so a registration landing after that copy but before the
    pass finishes is never part of it - yet the pass's ``train()`` caller
    still cleared ``must_train`` unconditionally on completion, silently
    stranding the registration forever. ``_train_generation`` closes the
    window: a training pass only clears ``must_train`` if the generation
    is still the one it started with, otherwise the worker loops and
    retrains immediately instead of stranding the registration.

    The window is opened deterministically (not via a timing hammer) by
    monkeypatching the low-level per-object training step
    (``ovos_padatious.training_manager._train_and_save``, called strictly
    after the ``objects_to_train`` snapshot copy) to block on an ``Event``
    until the test has landed the late registration precisely inside it.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _burst_plus_inflight_registration_trial(self):
        trial_dir = tempfile.mkdtemp(dir=self.tmp)
        # padaos has its own independent must_compile dirty bit and would
        # mask the neural-side race by lazily recompiling on the next
        # query regardless of the container's must_train state; disabling
        # it isolates the actual race under test
        container = IntentContainer(trial_dir, disable_padaos=True)
        container.add_intent("hello", ["hello there", "hi there"])
        container.train(debug=False)  # first-ever train, blocks

        container.add_intent("early", ["good morning", "morning to you"])

        started = threading.Event()
        release = threading.Event()
        real_train_and_save = training_manager._train_and_save

        def blocking_train_and_save(*args, **kwargs):
            started.set()
            release.wait(timeout=10)
            return real_train_and_save(*args, **kwargs)

        training_manager._train_and_save = blocking_train_and_save
        try:
            # a query triggers the background worker; its TrainingManager
            # .train() call has already copied objects_to_train by the
            # time the first _train_and_save call blocks
            container.calc_intents("hello there")
            self.assertTrue(started.wait(timeout=10),
                             "background train never started")

            # land the late registration strictly inside the snapshotted,
            # in-flight training pass
            container.add_intent("late", ["goodbye now", "see you later"])
            release.set()

            trainer = container._background_trainer
            self.assertIsNotNone(trainer)
            trainer.join(timeout=10)
            self.assertFalse(trainer.is_alive(),
                              "background worker never settled")
        finally:
            training_manager._train_and_save = real_train_and_save

        # drain any further pass the generation bump queued
        for _ in range(50):
            if not container.must_train:
                break
            container.calc_intents("late")
            trainer = container._background_trainer
            if trainer is not None:
                trainer.join(timeout=10)
        return container

    def test_late_intent_always_matchable_after_settling(self):
        failures = []
        for trial in range(10):
            container = self._burst_plus_inflight_registration_trial()
            self.assertFalse(container.must_train,
                              f"trial {trial}: container never settled")
            matches = container.calc_intents("goodbye now")
            if not any(m.name == "late" for m in matches):
                failures.append(trial)
        self.assertEqual(failures, [],
                          f"'late' intent unmatchable in trials {failures}")


if __name__ == "__main__":
    unittest.main()
