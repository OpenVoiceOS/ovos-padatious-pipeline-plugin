"""An intent is matchable as soon as it is registered.

Registration alone used to schedule nothing. ``register_intent`` trains only
under ``instant_train`` or once ``first_train`` is set, and neither holds
during boot, so the dirty container waited for some later live query to
drive training through ``IntentContainer.calc_intents`` itself. Every intent
registered at boot was therefore unmatchable until an unrelated utterance
happened to arrive and prime the worker - the first real utterance after
boot got "no match" for an intent that was registered and trainable.

Two independent guards, tested here. The load-bearing one is that a
registration arms the background worker, so the compile is tied to the
registration that caused it. The second is that a query against a container
which has never published any compiled state waits (bounded) for the first
pass instead of being answered from empty structures.

Neither reintroduces compiling on the match path or on a registration
thread: the pass still runs on the worker's own thread, the wait is on an
Event and never a lock, and once any generation has been published the
match path never waits again.
"""
import tempfile
import threading
import time
import unittest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline


class TestRegistrationArmsTheCompile(unittest.TestCase):

    def test_registering_an_intent_schedules_a_compile_without_a_query(self):
        bus = FakeBus()
        pipeline = PadatiousPipeline(bus, {"modules": {"padatious": {}}})
        try:
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": "skill:speak", "lang": "en-US", "skill_id": "skill",
                "samples": ["say {words}", "repeat {words}"]}))

            self.assertFalse(pipeline.first_train.is_set(),
                             "boot registration must not have trained inline")
            container = pipeline.containers["en-US"]
            deadline = time.monotonic() + 30
            while container.needs_compile and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(container.needs_compile,
                             "registration did not schedule a compile")
        finally:
            pipeline.shutdown()

    def test_registration_burst_still_coalesces_into_few_passes(self):
        bus = FakeBus()
        pipeline = PadatiousPipeline(bus, {"modules": {"padatious": {}}})
        try:
            container = pipeline.containers["en-US"]
            passes = []
            real_train = container.train

            def counting_train(*args, **kwargs):
                passes.append(1)
                return real_train(*args, **kwargs)

            container.train = counting_train

            for i in range(8):
                pipeline.register_intent(Message("padatious:register_intent", {
                    "name": f"skill:intent{i}", "lang": "en-US",
                    "skill_id": "skill", "samples": [f"do thing {i}"]}))
                time.sleep(0.1)

            deadline = time.monotonic() + 30
            while container.needs_compile and time.monotonic() < deadline:
                time.sleep(0.05)

            self.assertFalse(container.needs_compile)
            self.assertLessEqual(len(passes), 3,
                                 f"burst was not debounced: {len(passes)} passes "
                                 f"for 8 registrations")
        finally:
            pipeline.shutdown()


class TestFirstQueryWaitsForFirstCompileOnly(unittest.TestCase):

    def test_query_matches_an_intent_registered_moments_earlier(self):
        container = IntentContainer(tempfile.mkdtemp())
        container.add_intent("speak", ["say {words}", "repeat {words}"])

        match = container.calc_intent("say hello world")

        self.assertEqual(match.name, "speak")
        self.assertEqual(match.conf, 1.0)

    def test_the_wait_is_bounded_and_gives_up(self):
        container = IntentContainer(tempfile.mkdtemp())
        container.FIRST_COMPILE_WAIT_S = 0.2
        container.add_intent("speak", ["say {words}"])
        # nothing will ever publish a generation
        container._train_in_background = lambda: None

        start = time.monotonic()
        container.calc_intent("say hello world")
        elapsed = time.monotonic() - start

        self.assertGreaterEqual(elapsed, 0.2, "the query did not wait at all")
        self.assertLess(elapsed, 5.0, "the query was not bounded by "
                                      "FIRST_COMPILE_WAIT_S")

    def test_a_compiled_container_never_waits_again(self):
        container = IntentContainer(tempfile.mkdtemp())
        container.add_intent("speak", ["say {words}"])
        container.train(False)
        self.assertTrue(container._first_compile_done.is_set())

        # a later registration leaves the container dirty, but a generation
        # exists, so the match path serves it immediately (#126) instead of
        # blocking on the pending pass
        container.add_intent("other", ["something else entirely"])
        container.FIRST_COMPILE_WAIT_S = 30.0

        start = time.monotonic()
        match = container.calc_intent("say hello world")
        elapsed = time.monotonic() - start

        self.assertEqual(match.name, "speak")
        self.assertLess(elapsed, 2.0,
                        "a dirty but already-compiled container blocked the "
                        "match path on the pending compile")

    def test_the_wait_never_compiles_on_the_calling_thread(self):
        container = IntentContainer(tempfile.mkdtemp())
        container.add_intent("speak", ["say {words}"])
        caller = threading.current_thread()
        threads = []
        real_compile = container.padaos.compile

        def recording_compile(*args, **kwargs):
            threads.append(threading.current_thread())
            return real_compile(*args, **kwargs)

        container.padaos.compile = recording_compile

        container.calc_intent("say hello world")

        self.assertTrue(threads, "no compile ran at all")
        for t in threads:
            self.assertIsNot(t, caller,
                             "compile ran on the querying thread")


class TestTheWaitNeverOutlivesItsUsefulness(unittest.TestCase):
    """A query may only wait when waiting can actually produce an answer."""

    def test_a_container_with_nothing_to_compile_never_waits(self):
        # an unregistered language's container, and one built with
        # disable_padaos and no intents: nothing will ever publish a
        # generation, so there is nothing for a query to wait for
        for container in (IntentContainer(tempfile.mkdtemp()),
                          IntentContainer(tempfile.mkdtemp(), disable_padaos=True)):
            start = time.monotonic()
            container.calc_intent("hello world")
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.5,
                            "a container with nothing registered waited for a "
                            "compile that can never happen")

    def test_a_raising_pass_releases_waiters_instead_of_parking_them(self):
        container = IntentContainer(tempfile.mkdtemp())
        container.add_intent("speak", ["say {words}"])

        attempts = []

        def blows_up(*args, **kwargs):
            attempts.append(1)
            raise RuntimeError("compile blew up")

        container.intents.train = blows_up

        elapsed = []
        for _ in range(3):
            start = time.monotonic()
            container.calc_intent("say hello")
            elapsed.append(time.monotonic() - start)

        self.assertTrue(attempts, "the pass never ran")
        # the first query waits out the pass that then raises; every query
        # after it is answered immediately rather than re-paying the cap
        for e in elapsed[1:]:
            self.assertLess(e, 0.5,
                            f"a raising pass kept parking queries: {elapsed}")

    def test_the_cap_is_sized_to_the_debounce_window(self):
        # a registration storm re-arms _wait_for_quiet indefinitely; the
        # query must give up quickly rather than pay a long cap for an
        # answer it is not going to get
        self.assertLessEqual(IntentContainer.FIRST_COMPILE_WAIT_S,
                             IntentContainer._TRAIN_DEBOUNCE_S + 1.5)


class TestWorkerRetirementIsAtomic(unittest.TestCase):
    """``_spawn_background_trainer`` is the only thing that schedules a
    pass, so a worker deciding to exit must not leave a registration that
    lands at the same moment with nothing to compile it."""

    def test_a_retired_worker_releases_its_handle(self):
        bus = FakeBus()
        pipeline = PadatiousPipeline(bus, {"modules": {"padatious": {}}})
        try:
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": "skill:one", "lang": "en-US", "skill_id": "skill",
                "samples": ["say one"]}))
            container = pipeline.containers["en-US"]
            deadline = time.monotonic() + 30
            while container.needs_compile and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(container.needs_compile)

            # the worker has nothing left to do and retires; leaving its
            # handle in place would make the next spawn a no-op
            deadline = time.monotonic() + 10
            while pipeline._background_trainer is not None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNone(pipeline._background_trainer,
                              "a retired worker stayed recorded as current")
        finally:
            pipeline.shutdown()

    def test_a_registration_after_the_worker_retires_still_compiles(self):
        bus = FakeBus()
        pipeline = PadatiousPipeline(bus, {"modules": {"padatious": {}}})
        try:
            container = pipeline.containers["en-US"]
            pipeline.register_intent(Message("padatious:register_intent", {
                "name": "skill:one", "lang": "en-US", "skill_id": "skill",
                "samples": ["say one"]}))
            deadline = time.monotonic() + 30
            while container.needs_compile and time.monotonic() < deadline:
                time.sleep(0.05)

            pipeline.register_intent(Message("padatious:register_intent", {
                "name": "skill:two", "lang": "en-US", "skill_id": "skill",
                "samples": ["say two"]}))
            deadline = time.monotonic() + 30
            while container.needs_compile and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(container.needs_compile,
                             "a registration arriving after the worker "
                             "retired was stranded")
        finally:
            pipeline.shutdown()

    def test_a_raising_worker_releases_its_handle(self):
        bus = FakeBus()
        pipeline = PadatiousPipeline(bus, {"modules": {"padatious": {}}})
        try:
            def boom():
                raise RuntimeError("worker blew up")

            pipeline._train_worker = boom
            pipeline._spawn_background_trainer()
            deadline = time.monotonic() + 10
            while pipeline._background_trainer is not None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNone(pipeline._background_trainer,
                              "a worker that raised stayed recorded as current")
        finally:
            pipeline.shutdown()
