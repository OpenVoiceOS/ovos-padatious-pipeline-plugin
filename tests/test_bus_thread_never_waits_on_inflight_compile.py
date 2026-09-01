"""Queries and registrations must never wait on an in-flight padaos compile.

``PadatiousPipeline`` answers ``intent.service.padatious.get`` on the same
bus thread that delivers ``padatious:register_intent``, so anything that
blocks a registration handler also delays every query queued behind it.
A padaos compile that ran for tens of seconds while holding
``compile_lock`` did exactly that: the registration handler waited for the
whole compile, and the query behind it was answered only once the backlog
drained.

The compile therefore runs against a private snapshot with no lock held,
and takes ``compile_lock`` only to publish the result. Both tests below
hold a compile open on a background thread and assert the bus thread comes
back while that compile is still in flight, rather than measuring wall
clock.
"""
import time
import unittest
from threading import Event, Thread

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious import padaos
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "inflight.skill"
LANG = "en-US"

# generous enough that a blocked caller cannot plausibly come back
# "early" by luck, short enough not to drag the suite out
COMPILE_HOLD_S = 3.0


class _HeldCompile:
    """Patch ``padaos.IntentContainer._compile`` so the next compile blocks
    inside its own body until released, with the real compile still running
    afterwards."""

    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.finished = Event()
        self._real = padaos.IntentContainer._compile

    def __enter__(self):
        real, entered, release, finished = (self._real, self.entered,
                                            self.release, self.finished)

        def held_compile(container):
            entered.set()
            release.wait(COMPILE_HOLD_S)
            try:
                return real(container)
            finally:
                finished.set()

        padaos.IntentContainer._compile = held_compile
        return self

    def __exit__(self, *exc):
        self.release.set()
        padaos.IntentContainer._compile = self._real

    @property
    def in_flight(self):
        return self.entered.is_set() and not self.finished.is_set()


class TestBusThreadNeverWaitsOnInflightCompile(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.pipeline = PadatiousPipeline(self.bus, {"intent_cache": None})
        self.container = self.pipeline.containers[self.pipeline.lang]

    def _register(self, name, samples):
        self.bus.emit(Message("padatious:register_intent",
                              {"lang": LANG, "name": f"{SKILL_ID}:{name}",
                               "samples": samples}))

    def _start_held_compile(self, held):
        """Dirty the container and let a background pass reach the held
        compile, so the bus thread's next call races a real in-flight one."""
        self.container.add_intent(f"{SKILL_ID}:filler", ["say something else"])
        worker = Thread(target=self.container.train, daemon=True)
        worker.start()
        self.assertTrue(held.entered.wait(10), "compile never started")
        return worker

    def test_registration_returns_while_a_compile_is_in_flight(self):
        self.container.add_intent(f"{SKILL_ID}:hello", ["hello there"])
        self.container.train()

        with _HeldCompile() as held:
            worker = self._start_held_compile(held)
            self._register("goodbye", ["goodbye now"])
            self.assertTrue(held.in_flight,
                            "registration only returned after the compile "
                            "finished - the bus thread waited on it")
            held.release.set()
            worker.join(10)

    def test_query_is_answered_while_a_compile_is_in_flight(self):
        self.container.add_intent(f"{SKILL_ID}:hello", ["hello there"])
        self.container.train()

        replies = []
        self.bus.on("intent.service.padatious.reply", replies.append)

        with _HeldCompile() as held:
            worker = self._start_held_compile(held)
            # the bus thread handles a registration and the query back to
            # back, exactly as ovos-core delivers them
            self._register("goodbye", ["goodbye now"])
            self.bus.emit(Message("intent.service.padatious.get",
                                  {"utterance": "hello there", "lang": LANG}))
            self.assertTrue(held.in_flight,
                            "the query was answered only after the compile "
                            "finished")
            held.release.set()
            worker.join(10)

        self.assertEqual(len(replies), 1)
        self.assertIsNotNone(replies[0].data["intent"],
                             "the previously compiled state must keep serving "
                             "matches while a compile is in flight")
        self.assertEqual(replies[0].data["intent"]["name"], f"{SKILL_ID}:hello")


class TestServedStateDuringARecompileWindow(unittest.TestCase):
    """What a container serves between a registration and the pass that
    compiles it: everything the registration did not touch keeps matching
    from the last compiled snapshot, the replaced entry's retired template
    stops matching at once, and the replacement becomes matchable when the
    pass lands. The window covers the debounce quiet period as well as the
    compile itself, so the compile is held open for the duration.
    """

    def setUp(self):
        self.bus = FakeBus()
        self.pipeline = PadatiousPipeline(self.bus, {"intent_cache": None})
        self.container = self.pipeline.containers[self.pipeline.lang]

    def _conf(self, utterance):
        match = self.pipeline.calc_intent(utterance)
        return (match.name, match.conf) if match is not None else (None, 0.0)

    def test_untouched_intents_survive_a_replacement_window(self):
        self.container.add_intent(f"{SKILL_ID}:light", ["turn on the light"])
        self.container.add_intent(f"{SKILL_ID}:music", ["play some music"])
        self.container.train()
        self.assertEqual(self._conf("turn on the light"),
                         (f"{SKILL_ID}:light", 1.0))

        with _HeldCompile() as held:
            self.bus.emit(Message("padatious:register_intent",
                                  {"lang": LANG, "name": f"{SKILL_ID}:music",
                                   "samples": ["play a song now"]}))
            worker = Thread(target=self.container.train, daemon=True)
            worker.start()
            self.assertTrue(held.entered.wait(10), "compile never started")

            name, conf = self._conf("turn on the light")
            self.assertEqual(name, f"{SKILL_ID}:light")
            self.assertEqual(conf, 1.0,
                             "an intent the registration never touched lost "
                             "its compiled match while an unrelated intent "
                             "was recompiling")
            self.assertLess(self._conf("play some music")[1], 1.0,
                            "the replaced intent's retired template still "
                            "matches at compiled confidence")
            self.assertLess(self._conf("play a song now")[1], 1.0,
                            "the replacement matched before its pass landed")

            held.release.set()
            worker.join(10)

        self.assertTrue(self.pipeline.wait_until_trained(30))
        self.assertEqual(self._conf("play a song now"),
                         (f"{SKILL_ID}:music", 1.0))
        self.assertEqual(self._conf("turn on the light"),
                         (f"{SKILL_ID}:light", 1.0))


if __name__ == "__main__":
    unittest.main()
