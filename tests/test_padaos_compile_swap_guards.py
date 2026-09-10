"""The padaos compile publishes its result under the lock but builds it
off-lock, so registrations land *while* a pass is in flight. These tests pin
what the swap is allowed to publish in that window.

Each one gates the compile from the inside - the pass has already taken its
snapshot and is part way through building regexes - then performs the
registration and lets the pass finish, so the assertions are about the swap
itself rather than about timing.
"""
import unittest
from threading import Event, Thread
from unittest import mock

from ovos_padatious import padaos


class _GatedCompile:
    """Block the first ``create_regexes`` call of a compile until released.

    That point is after ``_compile`` snapshots ``intent_lines`` and before it
    publishes, which is exactly the window a mid-compile registration lands
    in.
    """

    def __init__(self):
        self.entered = Event()
        self.release = Event()

    def __enter__(self):
        real = padaos.IntentContainer.create_regexes
        gate = self

        def gated(container, *args, **kwargs):
            if not gate.entered.is_set():
                gate.entered.set()
                gate.release.wait(10)
            return real(container, *args, **kwargs)

        self._patch = mock.patch.object(padaos.IntentContainer,
                                        "create_regexes", gated)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self.release.set()
        self._patch.stop()

    def run_compile(self, container):
        """Start a compile and block until it is mid-flight."""
        worker = Thread(target=container.compile, daemon=True)
        worker.start()
        assert self.entered.wait(10), "compile never started"
        return worker

    def finish(self, worker):
        self.release.set()
        worker.join(10)


class TestCompileSwapGuards(unittest.TestCase):

    def _matches(self, container, utterance):
        return [m['name'] for m in container.calc_intents(f" {utterance} ")]

    def test_replacement_mid_compile_does_not_republish_the_retired_template(self):
        """A name re-registered while a pass was building must not have that
        pass's now-stale regexes published over it - the retired template
        would start matching again at conf 1.0, undoing the immediate
        retirement ``add_intent`` performs."""
        c = padaos.IntentContainer()
        c.add_intent("greet", ["old phrasing here"])
        c.compile()
        self.assertEqual(self._matches(c, "old phrasing here"), ["greet"])

        with _GatedCompile() as gate:
            c.add_intent("greet", ["fresh phrasing here"])
            worker = gate.run_compile(c)
            # the pass is mid-flight and its snapshot still holds the OLD
            # lines; replace again so what it is compiling is stale
            c.add_intent("greet", ["newest phrasing here"])
            gate.finish(worker)

        self.assertEqual(self._matches(c, "old phrasing here"), [],
                         "the swap republished the retired template")
        self.assertEqual(self._matches(c, "fresh phrasing here"), [],
                         "the swap published a superseded registration")
        self.assertTrue(c.must_compile,
                        "a registration landed mid-pass but the container "
                        "was marked clean")

        c.compile()
        self.assertEqual(self._matches(c, "newest phrasing here"), ["greet"])

    def test_registration_mid_compile_leaves_the_container_dirty(self):
        """``must_compile`` is the only dirty bit padaos has, and for a
        registration the hash cache considers unchanged it is the only dirty
        bit anywhere - clearing it after a pass that never saw that
        registration strands it until something else happens to dirty the
        container again."""
        c = padaos.IntentContainer()
        c.add_intent("greet", ["hello there"])
        c.compile()
        self.assertFalse(c.must_compile)

        with _GatedCompile() as gate:
            c.add_entity("colour", ["red"])
            worker = gate.run_compile(c)
            # entities are published wholesale, so this isolates the dirty
            # bit from the per-name identity filter
            c.add_entity("size", ["large"])
            gate.finish(worker)

        self.assertTrue(c.must_compile,
                        "the swap cleared must_compile even though a "
                        "registration arrived after its snapshot")
        c.compile()
        self.assertIn("size", c.entities)
        self.assertFalse(c.must_compile)

    def test_removal_mid_compile_is_not_resurrected_by_the_swap(self):
        """Removal is a runtime gate that takes effect immediately; a pass
        that snapshotted the intent before it was removed must not publish
        it back."""
        c = padaos.IntentContainer()
        c.add_intent("greet", ["hello there"])
        c.add_intent("bye", ["goodbye now"])
        c.compile()
        self.assertEqual(self._matches(c, "goodbye now"), ["bye"])

        with _GatedCompile() as gate:
            c.add_intent("greet", ["hello there again"])
            worker = gate.run_compile(c)
            c.remove_intent("bye")
            gate.finish(worker)

        self.assertEqual(self._matches(c, "goodbye now"), [],
                         "the swap resurrected an intent removed mid-pass")
        self.assertNotIn("bye", c.intents)
        # the surviving intent is unaffected
        c.compile()
        self.assertEqual(self._matches(c, "hello there again"), ["greet"])
        self.assertEqual(self._matches(c, "goodbye now"), [])


if __name__ == "__main__":
    unittest.main()
