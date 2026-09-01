"""padaos must compile in the background after a registration burst - even
when every registration in that burst is a hash-cache no-op - and must
NEVER compile synchronously on the match path.

Field evidence (ser9, 2.0.13a1, host lang pt-pt, issue #116): with .hash
sidecars present from earlier boots, a fresh boot's registrations were all
hash no-ops - ONE trivial 2.26s compile, zero full compiles for 4h15m -
then the FIRST live query ran an 85.49s FULL padaos compile synchronously
inside the match path:
    handle_get_padatious (opm.py:979)
    -> calc_intent (opm.py:878)
    -> _calc_padatious_intent (opm.py:1107)
    -> IntentContainer.calc_intents (intent_container.py:430)
    -> padaos.calc_intents (padaos.py:274)
    -> padaos.compile (padaos.py:216)
    -> _compile (padaos.py:235)

Root cause: ``padaos.add_intent``/``add_entity`` unconditionally mark the
regex container dirty (``must_compile = True``) even when the registration
is a hash-cache hit that never dirties the wrapping ``IntentContainer``
(``must_train`` stays False - see ``TrainingManager.add``). Every gate that
decided whether to (background-)train - ``opm.py``'s ``train()``,
``_train_worker()`` and ``_train_sync()``, and
``IntentContainer.train()``/``_train_in_background()`` - checked only
``must_train``, so a boot that replays nothing but cache hits never trained
at all, leaving ``padaos.must_compile`` stuck True until the first live
query forced ``padaos.calc_intents`` to compile inline on the bus thread.
"""
import os
import shutil
import tempfile
import time
import unittest
from threading import Event
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious import padaos
from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "replay.skill"


def _register_msg(name, samples, lang):
    return Message("padatious:register_intent", {
        "name": name, "samples": samples, "lang": lang, "skill_id": SKILL_ID,
    })


class _XdgIsolated(unittest.TestCase):
    """Best-effort isolation of XDG_CONFIG_HOME/XDG_DATA_HOME so a stray
    default cache_dir can never touch the host's real state.

    ``ovos_config.config.Configuration`` resolves its user-config search
    paths once, at import time (a plain class attribute, not something
    re-derived per read), so patching these env vars after ``ovos_config``
    has already been imported anywhere in the process - e.g. by another
    test module collected first, or by this host's own real config (lang
    pt-pt, per issue #116) - cannot retroactively change which files it
    reads. Rather than fight that import-order dependency, every test here
    reads the pipeline's OWN resolved ``lang`` back (``pipeline.lang``)
    instead of assuming a literal "en-US", so the tests hold regardless of
    which language the host happens to resolve to.
    """

    def setUp(self):
        self._xdg_tmp = tempfile.mkdtemp()
        self._env_patch = mock.patch.dict(os.environ, {
            "XDG_CONFIG_HOME": os.path.join(self._xdg_tmp, "config"),
            "XDG_DATA_HOME": os.path.join(self._xdg_tmp, "data"),
        })
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        shutil.rmtree(self._xdg_tmp, ignore_errors=True)


class TestReplayWithSidecarsCompilesInBackground(_XdgIsolated):
    """The ser9 scenario: a registration whose .hash sidecar already
    matches on disk (simulated here by re-registering already-cached
    intents against the SAME long-running pipeline, exactly like
    ovos-core's periodic registration reconciliation re-emitting every
    unchanged intent - see ``test_reregistration_no_retrain.py``) must
    still end up recompiling padaos via the background worker with NO
    query, and the next query must not compile on its own thread."""

    def setUp(self):
        super().setUp()
        self.cache = tempfile.mkdtemp()
        self.pipeline = PadatiousPipeline(FakeBus(), config={"intent_cache": self.cache})
        # simulate steady state: the initial full-skill-load pass has
        # already completed, so every registration from here on takes the
        # background-worker path instead of the (expected, documented)
        # synchronous very-first-ever pass
        self.pipeline.first_train.set()
        self.lang = self.pipeline.lang

    def tearDown(self):
        self.pipeline.shutdown()
        shutil.rmtree(self.cache, ignore_errors=True)
        super().tearDown()

    def _register(self, name, samples):
        self.pipeline.register_intent(_register_msg(name, samples, self.lang))

    def test_background_compile_runs_after_noop_replay_with_no_query(self):
        self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
        self._register(f"{SKILL_ID}:bye", ["bye", "see you"])
        deadline = time.monotonic() + 5.0
        while self.pipeline.containers[self.lang].must_train and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.pipeline.containers[self.lang].must_train)
        self.assertFalse(self.pipeline.containers[self.lang].padaos.must_compile)

        with mock.patch.object(IntentContainer, "_TRAIN_DEBOUNCE_S", 0.05), \
             mock.patch.object(IntentContainer, "_TRAIN_MAX_DEFER_S", 2.0):
            compile_calls = []
            real_compile = padaos.IntentContainer._compile

            def counting_compile(self, *a, **kw):
                compile_calls.append(1)
                return real_compile(self, *a, **kw)

            with mock.patch.object(padaos.IntentContainer, "_compile", counting_compile):
                # identical replay: TrainingManager.add reports changed=False
                # for both (hash-cache hit), so must_train must never be set -
                # but padaos.add_intent has no cache-aware skip of its own and
                # marks padaos dirty regardless (see IntentContainer.needs_compile)
                self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
                self._register(f"{SKILL_ID}:bye", ["bye", "see you"])

                self.assertFalse(
                    self.pipeline.containers[self.lang].must_train,
                    "an identical re-registration must never dirty must_train")

                # NO query has happened; the background worker must still
                # compile padaos on its own within the debounce window
                deadline = time.monotonic() + 5.0
                while (self.pipeline.containers[self.lang].padaos.must_compile
                       and time.monotonic() < deadline):
                    time.sleep(0.02)

                self.assertFalse(
                    self.pipeline.containers[self.lang].padaos.must_compile,
                    "padaos was never compiled in the background despite "
                    "every registration being a hash no-op")
                self.assertGreaterEqual(
                    len(compile_calls), 1,
                    "no background compile pass ran for the no-op replay")

    def test_first_query_after_noop_replay_does_not_compile_on_caller_thread(self):
        self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])
        deadline = time.monotonic() + 5.0
        while self.pipeline.containers[self.lang].must_train and time.monotonic() < deadline:
            time.sleep(0.02)

        with mock.patch.object(IntentContainer, "_TRAIN_DEBOUNCE_S", 0.05), \
             mock.patch.object(IntentContainer, "_TRAIN_MAX_DEFER_S", 2.0):
            self._register(f"{SKILL_ID}:hello", ["hello", "hi there"])  # no-op replay

            # give the background worker its debounce window to compile,
            # mirroring the 4h+ of idle time before ser9's first live query
            deadline = time.monotonic() + 5.0
            while (self.pipeline.containers[self.lang].padaos.must_compile
                   and time.monotonic() < deadline):
                time.sleep(0.02)

        import threading
        caller_thread = threading.current_thread()
        real_compile = padaos.IntentContainer._compile

        def assert_not_caller_thread(self, *a, **kw):
            assert threading.current_thread() is not caller_thread, (
                "padaos compiled synchronously on the query/caller "
                "thread - the exact ser9 field defect")
            return real_compile(self, *a, **kw)

        with mock.patch.object(padaos.IntentContainer, "_compile", assert_not_caller_thread):
            match = self.pipeline.calc_intent("hello", lang=self.lang)
            self.assertIsNotNone(match)
            self.assertEqual(match.name, f"{SKILL_ID}:hello")


class TestMatchPathNeverCompilesSynchronously(_XdgIsolated):
    """Even a container that has NEVER been compiled at all must not block
    a query while padaos compiles - the query is served (possibly with no
    padaos match yet) and a background compile is scheduled."""

    def setUp(self):
        super().setUp()
        self.container = padaos.IntentContainer()
        self.container.add_intent("hello", ["hello", "hi there"])

    def test_query_during_pending_compile_returns_promptly(self):
        # must_compile is True and nothing has ever been compiled; calling
        # calc_intents directly (the match path) must return immediately
        # instead of compiling inline
        start = time.monotonic()
        result = list(self.container.calc_intents(" hello "))
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.5,
                         "padaos.calc_intents compiled inline on the match path")
        self.assertEqual(result, [])
        self.assertTrue(self.container.must_compile)

    def test_query_blocked_on_a_pending_compile_does_not_block_the_caller(self):
        """Mirrors #118/#122's race style: an Event-blocked compile running
        on another thread must not stall a concurrent query."""
        started = Event()
        release = Event()
        real_compile = padaos.IntentContainer._compile

        def blocking_compile(self):
            started.set()
            release.wait(timeout=5.0)
            return real_compile(self)

        with mock.patch.object(padaos.IntentContainer, "_compile", blocking_compile):
            import threading
            t = threading.Thread(target=self.container.compile, daemon=True)
            t.start()
            self.assertTrue(started.wait(timeout=5.0),
                             "background compile never started")
            try:
                start = time.monotonic()
                result = list(self.container.calc_intents(" hello "))
                elapsed = time.monotonic() - start
                self.assertLess(elapsed, 0.5,
                                 f"query blocked for {elapsed:.2f}s on a "
                                 f"pending compile running elsewhere")
                self.assertEqual(result, [])
            finally:
                release.set()
                t.join(timeout=5.0)


class TestMustCompileIsClearedAndIdempotent(_XdgIsolated):
    """``must_compile`` must be cleared after a successful compile, and a
    second compile with nothing new to do must be a cheap no-op."""

    def setUp(self):
        super().setUp()
        self.container = padaos.IntentContainer()
        self.container.add_intent("hello", ["hello", "hi there"])

    def test_must_compile_false_after_compile(self):
        self.assertTrue(self.container.must_compile)
        self.container.compile()
        self.assertFalse(self.container.must_compile)

    def test_second_compile_call_is_a_noop(self):
        self.container.compile()
        intents_before = self.container.intents

        real_compile = padaos.IntentContainer._compile
        calls = []

        def counting_compile(self, *a, **kw):
            calls.append(1)
            return real_compile(self, *a, **kw)

        with mock.patch.object(padaos.IntentContainer, "_compile", counting_compile):
            # calling calc_intents again must not recompile: must_compile
            # is already False, and the match path must never compile
            # regardless
            list(self.container.calc_intents(" hello "))
        self.assertEqual(calls, [])
        self.assertIs(self.container.intents, intents_before)

    def test_query_answers_fast_after_a_compile_has_completed(self):
        """skills-QA probe: once compiled, a second query answers fast -
        asserted via the mechanism (no recompile), not just a wall-clock
        bound, since a generous bound alone can hide a reintroduced
        synchronous compile on a fast machine."""
        self.container.compile()
        real_compile = padaos.IntentContainer._compile
        calls = []

        def counting_compile(self, *a, **kw):
            calls.append(1)
            return real_compile(self, *a, **kw)

        with mock.patch.object(padaos.IntentContainer, "_compile", counting_compile):
            start = time.monotonic()
            result = list(self.container.calc_intents(" hi there "))
            elapsed = time.monotonic() - start
        self.assertEqual(calls, [], "a fully compiled container must never recompile on query")
        self.assertLess(elapsed, 2.0)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
