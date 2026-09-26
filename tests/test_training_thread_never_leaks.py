"""No exception may escape a training thread.

The suite already made two training passes raise on purpose, to prove that
a raising pass releases its waiters and that a raising worker releases its
handle. Neither code path caught the exception at the thread boundary, so
it escaped to `threading.excepthook`. Under pytest that surfaces as a
`PytestUnhandledThreadExceptionWarning` attributed to whatever test runs
NEXT, which is why the defect read as an unrelated failure in
`test_the_cap_is_sized_to_the_debounce_window` while the test that caused
it passed green:

    RuntimeError: compile blew up
    tests/test_registration_arms_first_compile.py:181
    ... ERROR at setup of
        TestTheWaitNeverOutlivesItsUsefulness.test_the_cap_is_sized_to_the_debounce_window

Outside pytest it is worse, not better: `threading.excepthook` prints to
stderr and nothing a running service reads is told anything, so a compile
that keeps failing looks exactly like a compile that never ran.

These tests watch `threading.excepthook` itself rather than the log,
because that hook is the thing the defect fires and OVOS's `LOG` does not
reach the root logger, so a log capture can read empty while the thread is
still leaking.
"""
import sys
import tempfile
import threading
import time
import unittest

from ovos_utils.fakebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline


class _RecordingExceptHook:
    """Collect every exception that escapes a thread while installed."""

    def __init__(self):
        self.escaped = []
        self._previous = None

    def __enter__(self):
        self._previous = threading.excepthook
        threading.excepthook = self._record
        return self

    def _record(self, args):
        self.escaped.append(args)

    def __exit__(self, *exc):
        threading.excepthook = self._previous
        return False

    def names(self):
        return [f"{a.exc_type.__name__}: {a.exc_value}" for a in self.escaped]


class TestTheContainerLoopSwallowsNothingSilently(unittest.TestCase):

    def test_a_raising_pass_does_not_escape_the_thread(self):
        container = IntentContainer(tempfile.mkdtemp())
        container.add_intent("speak", ["say {words}"])

        def blows_up(*args, **kwargs):
            raise RuntimeError("compile blew up")

        container.intents.train = blows_up

        with _RecordingExceptHook() as hook:
            container.calc_intent("say hello")
            # the worker is a daemon thread; give it room to finish the
            # pass and to raise, so a pass still in flight is not read as
            # a pass that did not leak
            deadline = time.monotonic() + 10
            trainer = container._background_trainer
            while trainer is not None and trainer.is_alive() \
                    and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.2)

        self.assertEqual(hook.escaped, [],
                         f"an exception escaped the training thread: "
                         f"{hook.names()}")

    def test_the_container_still_answers_after_a_raising_pass(self):
        # the catch must not turn a failed compile into a hang: the query
        # is still answered, and the container is still marked dirty so a
        # later pass can retry
        container = IntentContainer(tempfile.mkdtemp())
        container.add_intent("speak", ["say {words}"])
        container.intents.train = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("compile blew up"))
        start = time.monotonic()
        container.calc_intent("say hello")
        self.assertLess(time.monotonic() - start, 10)
        self.assertTrue(container.needs_compile,
                        "a failed compile marked the container clean")


class TestThePipelineWorkerSwallowsNothingSilently(unittest.TestCase):

    def test_a_raising_worker_does_not_escape_the_thread(self):
        bus = FakeBus()
        pipeline = PadatiousPipeline(bus, {"modules": {"padatious": {}}})
        try:
            def boom():
                raise RuntimeError("worker blew up")

            pipeline._train_worker = boom
            with _RecordingExceptHook() as hook:
                pipeline._spawn_background_trainer()
                deadline = time.monotonic() + 10
                while pipeline._background_trainer is not None \
                        and time.monotonic() < deadline:
                    time.sleep(0.05)
                time.sleep(0.2)

            self.assertEqual(hook.escaped, [],
                             f"an exception escaped the worker thread: "
                             f"{hook.names()}")
            # the handle is still released, which is what the existing test
            # on this path asserts; the catch must not change that
            self.assertIsNone(pipeline._background_trainer)
        finally:
            pipeline.shutdown()


class TestTheRecorderItselfWorks(unittest.TestCase):
    """The control.

    A recorder that never fires would make both tests above pass on the
    unfixed code as well, so it is fired deliberately here.
    """

    def test_the_recorder_sees_a_thread_that_really_leaks(self):
        def leaks():
            raise RuntimeError("deliberate leak")

        with _RecordingExceptHook() as hook:
            thread = threading.Thread(target=leaks, daemon=True)
            thread.start()
            thread.join(10)
            time.sleep(0.2)

        self.assertEqual(len(hook.escaped), 1,
                         "the recorder did not see a thread that leaked, so "
                         "it proves nothing about the ones that did not")
        self.assertIn("deliberate leak", hook.names()[0])


if __name__ == "__main__":
    unittest.main()
