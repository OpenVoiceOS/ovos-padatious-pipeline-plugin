"""``mycroft.skills.trained`` must be a single readiness signal covering
BOTH the neural model and padaos, for every container it claims to speak
for - never emitted while any container it just (attempted to) train
still reports ``needs_compile``.

``_train_sync`` used to gate the emission purely on whether
``container.train()`` raised (``any_success``), not on whether the
container was actually left clean. A registration landing while
``train()`` is mid-pass bumps the container's generation counters
(``_train_generation`` for the neural side, ``_mutation_gen`` for padaos -
see ``IntentContainer.train``/``padaos.IntentContainer._compile``) so that
pass correctly does NOT clear ``must_train``/``must_compile`` for the
newly-landed registration - ``train()`` still returns normally (no
exception), so it counted as a success and fired ``mycroft.skills.trained``
regardless. padaos is far more likely to be the one left dirty: every
``padaos.add_intent``/``add_entity`` call marks it dirty unconditionally,
with no cache-aware skip of its own (see
``test_padaos_replay_background_compile.py``), so a registration trickling
in during a compile leaves padaos as a second, independent readiness
signal the neural-only ``any_success`` check never saw. A caller that
waits for ``mycroft.skills.trained`` and then queries immediately (the
ovos-core contract) could get an intent whose slot/return regex predates
that registration.
"""
import tempfile
import time
import unittest
from threading import Event
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "readiness.skill"


class TestTrainedSignalWaitsOnFullReadiness(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.pipeline = PadatiousPipeline(
            self.bus, config={"intent_cache": tempfile.mkdtemp(), "disable_padaos": False})
        self.lang = self.pipeline.lang
        self.trained_events = []
        self.bus.on('mycroft.skills.trained',
                    lambda m: self.trained_events.append(time.monotonic()))

    def tearDown(self):
        self.pipeline.shutdown()

    def _register(self, name, samples):
        self.pipeline.register_intent(Message("padatious:register_intent", {
            "name": name, "samples": samples, "lang": self.lang, "skill_id": SKILL_ID,
        }))

    def test_trained_never_fires_while_container_still_needs_compile(self):
        """Deterministic repro: force ``container.train()`` to land a fresh
        registration (bumping both generation counters) strictly AFTER its
        own snapshot but BEFORE it returns, so it returns successfully
        while the container is still dirty - and ``mycroft.skills.trained``
        must not have fired yet."""
        self._register(f"{SKILL_ID}:hello", ["hello there", "hi there"])
        self.assertTrue(self.pipeline.wait_until_trained(timeout=15.0))
        self.trained_events.clear()

        container = self.pipeline.containers[self.lang]
        real_train = IntentContainer.train
        late_registration_landed = Event()

        def train_that_races_a_late_registration(self_container, *a, **kw):
            result = real_train(self_container, *a, **kw)
            if self_container is container and not late_registration_landed.is_set():
                late_registration_landed.set()
                # simulate a registration landing after this pass's
                # snapshot was taken but before train() (and therefore
                # _train_sync) observes the outcome
                self_container.add_intent(f"{SKILL_ID}:bye", ["goodbye now"])
            return result

        with mock.patch.object(IntentContainer, "train", train_that_races_a_late_registration):
            self._register(f"{SKILL_ID}:hello", ["hello there", "hi there", "hey there"])
            deadline = time.monotonic() + 10.0
            while not self.trained_events and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertTrue(self.trained_events, "trained never fired at all")
        self.assertFalse(
            container.needs_compile,
            "mycroft.skills.trained fired while the container it just "
            "trained still needs a compile - a late-landing registration "
            "was silently left unpublished behind a false-positive "
            "readiness signal")


if __name__ == "__main__":
    unittest.main()
