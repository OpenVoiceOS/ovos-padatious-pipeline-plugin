"""``PadatiousPipeline.register_intent`` must never block the calling thread
past the first training pass.

Field evidence (ser9, ~128 registrations at boot): a compile+train pass that
takes tens of seconds (see ``test_padaos_entity_wedge.py`` for why a single
oversized entity can do that) ran synchronously inside ``register_intent``
for every registration after the very first one. ``MessageBusClient.on_message``
dispatches every incoming bus message to its handlers synchronously on the
single websocket-receive thread (``self.emitter.emit(...)`` in
``ovos_bus_client.client.client``), so a slow ``register_intent`` handler
starves every other message on that connection - including a skill's own
later registration (which then never goes live until the stalled handler
returns) and the ``intent.service.padatious.manifest.get`` /
``intent.service.padatious.get`` getters other services poll with a bounded
timeout.

``IntentContainer`` already hands retraining to a single background worker
(see ``intent_container.py``) so a query is never blocked by a stale
``must_train`` flag; these tests pin that ``PadatiousPipeline.train()`` -
the entry point actually invoked from the bus-handler thread - honors that
same contract instead of calling the container's blocking ``train()``
directly.
"""
import time
import unittest
from threading import Event
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "background.train.skill"
LANG = "en-US"


def _register_msg(name, samples):
    return Message("padatious:register_intent", {
        "name": name, "samples": samples, "lang": LANG, "skill_id": SKILL_ID,
    })


def _slow_train(sleep_seconds):
    """Wraps the real ``IntentContainer.train`` so a pass takes at least
    ``sleep_seconds``, standing in for a real oversized-entity compile
    without actually building one."""
    real_train = IntentContainer.train

    def wrapper(self, *args, **kwargs):
        time.sleep(sleep_seconds)
        return real_train(self, *args, **kwargs)

    return wrapper


class TestRegisterIntentDoesNotBlockBusThread(unittest.TestCase):
    """Red -> green: register_intent must return promptly once the
    container has trained once before, even while a slow retrain runs."""

    def setUp(self):
        self.pipeline = PadatiousPipeline(FakeBus(), config={})
        # simulate ovos-core's initial full-skill-load training pass having
        # already completed, so we are exercising the steady-state path
        self.pipeline.first_train.set()

    def tearDown(self):
        self.pipeline.shutdown()

    def test_register_intent_returns_promptly_during_slow_train(self):
        with mock.patch.object(IntentContainer, "train", _slow_train(1.5)):
            start = time.monotonic()
            self.pipeline.register_intent(
                _register_msg(f"{SKILL_ID}:hello", ["hello", "hi there"]))
            elapsed = time.monotonic() - start

        # the call must return long before the (patched) 1.5s train pass
        # completes; on unfixed code it blocks for the full duration
        self.assertLess(elapsed, 0.5,
                         f"register_intent blocked the caller for {elapsed:.2f}s")

        # the background pass must still land: poll for it to finish and
        # confirm the intent is actually matchable afterwards
        deadline = time.monotonic() + 5.0
        while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.pipeline.containers[LANG].must_train,
                          "background training never completed")

        match = self.pipeline.calc_intent("hello there", lang=LANG)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"{SKILL_ID}:hello")

    def test_late_registration_becomes_visible_after_a_slow_earlier_pass(self):
        """A second, unrelated intent registered while a slow retrain for
        the first one is still in flight (the 'Spell' skill landing behind
        a slow pokepedia-sized registration in the ser9 trace) must still
        end up trained and matchable, not silently stranded."""
        with mock.patch.object(IntentContainer, "train", _slow_train(1.0)):
            self.pipeline.register_intent(
                _register_msg(f"{SKILL_ID}:hello", ["hello", "hi there"]))
            # arrives while the background worker spawned above is still
            # running its (patched) slow pass
            self.pipeline.register_intent(
                _register_msg(f"{SKILL_ID}:spell", ["spell {word}", "how do you spell {word}"]))
            self.pipeline.register_entity(Message("padatious:register_entity", {
                "name": f"{SKILL_ID}:word", "samples": ["hello"], "lang": LANG,
                "skill_id": SKILL_ID,
            }))

            deadline = time.monotonic() + 10.0
            while self.pipeline.containers[LANG].must_train and time.monotonic() < deadline:
                time.sleep(0.05)

        self.assertFalse(self.pipeline.containers[LANG].must_train,
                          "background training never converged")
        match = self.pipeline.calc_intent("how do you spell hello", lang=LANG)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"{SKILL_ID}:spell")


class TestGettersRespondWithoutWaitingOnTraining(unittest.TestCase):
    """The manifest/get bus getters must answer from already-published
    state and never be routed through a path that waits on the trainer."""

    def setUp(self):
        self.pipeline = PadatiousPipeline(FakeBus(), config={})
        self.pipeline.first_train.set()

    def tearDown(self):
        self.pipeline.shutdown()

    def test_manifest_get_responds_immediately_during_background_train(self):
        with mock.patch.object(IntentContainer, "train", _slow_train(1.5)):
            self.pipeline.register_intent(
                _register_msg(f"{SKILL_ID}:hello", ["hello", "hi there"]))

            replies = []
            self.pipeline.bus.on("intent.service.padatious.manifest",
                                 lambda m: replies.append(m))
            start = time.monotonic()
            self.pipeline.handle_padatious_manifest(
                Message("intent.service.padatious.manifest.get", {}))
            elapsed = time.monotonic() - start

        self.assertLess(elapsed, 0.5)
        self.assertEqual(len(replies), 1)
        self.assertIn(f"{SKILL_ID}:hello", replies[0].data["intents"])


if __name__ == '__main__':
    unittest.main()
