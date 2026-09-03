"""``PadatiousPipeline.calc_intent``'s own ``_calc_padatious_intent``
lru_cache (maxsize=3) must never keep serving a stale "no match" answer
once a compile pass that was triggered OUTSIDE ``_train_sync`` (the only
place that used to call ``.cache_clear()``) actually lands.

Adversarial review of PR #124 found this: a query that reaches
``IntentContainer.calc_intents`` for a container that has never compiled
triggers training via ``IntentContainer._train_in_background`` directly
(see ``test_never_train_on_calling_thread.py``) rather than through
``PadatiousPipeline.train()``/``_train_sync`` - which is the ONLY place
that cleared this cache. So the very first query for an utterance, made
before any pass has run, got cached as "no match" and kept returning that
same cached miss forever afterwards - even once the background pass
completed - until three other distinct utterances happened to evict the
entry (maxsize=3).

Fixed by keying the cache on ``IntentContainer.compiled_generation`` /
``DomainIntentContainer.compiled_generation`` (bumped every time a real
compile pass actually finishes), so a query made in a different compile
generation is simply a different cache key rather than a stale hit.

This test MUST NOT call ``PadatiousPipeline.wait_until_trained()``:
that method calls ``_spawn_background_trainer()``, which starts the
PIPELINE's own background worker/``_train_sync`` alongside the
container's own ``_train_in_background`` thread that this scenario
actually exercises, and ``_train_sync`` calls
``_calc_padatious_intent.cache_clear()`` on ANY successful pass - which
would mask a broken ``compiled_generation`` key (mutating it back to a
constant still passed under the original version of this test). Waiting
by polling ``containers[lang].needs_compile`` directly, with no call to
``wait_until_trained``/``train``/``_train_sync``, isolates the
generation-key fix itself.
"""
import shutil
import tempfile
import time
import unittest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "cache_gen.skill"


class TestCalcIntentCacheInvalidatesAfterBackgroundCompile(unittest.TestCase):
    def setUp(self):
        # A fresh, unique cache_dir: the neural tier can otherwise answer a
        # query with no training at all by loading an already-cached object
        # synchronously at registration time (TrainingManager.add's
        # hash-cache-hit path - see test_never_train_on_calling_thread.py),
        # which would make the "no match before the pass" assertion below
        # depend on host-disk state left behind by an earlier run rather
        # than on the compile-cache-invalidation bug this test targets.
        self.cache_dir = tempfile.mkdtemp()
        # NOT instant_train, and first_train left UNSET: register_intent's
        # own gate (`if instant_train or first_train.is_set(): self.train()`)
        # then never calls PadatiousPipeline.train()/_train_sync at all, so
        # the ONLY thing that ever compiles this container is the query
        # itself, via IntentContainer._train_in_background - exactly the
        # path whose completion opm.py's cache-clearing code never saw.
        self.pipeline = PadatiousPipeline(
            FakeBus(), config={"intent_cache": self.cache_dir})
        self.lang = self.pipeline.lang

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_same_utterance_matches_after_the_triggering_querys_own_pass_completes(self):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": f"{SKILL_ID}:hello", "samples": ["hello", "hi there"],
            "lang": self.lang, "skill_id": SKILL_ID,
        }))
        self.assertFalse(self.pipeline.first_train.is_set(),
                          "test precondition: register_intent must not have "
                          "triggered PadatiousPipeline.train() itself")
        container = self.pipeline.containers[self.lang]

        # A never-compiled container waits (bounded) for its first pass
        # instead of answering from empty structures, so an intent
        # registered moments ago is matchable on the very first query.
        first = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNotNone(first)

        # Poll the CONTAINER's own compile flag directly - never call
        # wait_until_trained()/train()/_train_sync() here, see module
        # docstring: those clear the lru_cache themselves on any successful
        # pass and would mask a broken compiled_generation key.
        deadline = time.monotonic() + 10.0
        while container.needs_compile and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(container.needs_compile,
                          "container never finished its own background compile")

        # the SAME utterance, now that the pass it triggered has landed,
        # must match - not keep replaying the cached pre-compile miss
        second = self.pipeline.calc_intent(["hello"], self.lang)
        self.assertIsNotNone(
            second, "the lru_cache kept serving the stale pre-compile "
                    "'no match' after the background pass completed")
        self.assertEqual(second.name, f"{SKILL_ID}:hello")


if __name__ == '__main__':
    unittest.main()
