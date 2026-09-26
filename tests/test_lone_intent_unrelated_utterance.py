"""A lone intent must not claim an utterance that shares no word with it.

With one registered intent the training set holds no real negative: only the
``:null:`` pollution rows, which train to ``LENIENCE`` (0.6). An utterance made
of unknown words then scores near 0.6, and the exact value depends on the
trained weights, which differ across numpy builds (0.525 on one box, above
0.6 on the CI runner). ``tests/test_ovoscope_e2e.py::test_no_match_unrelated_utterance``
gates at ``conf_low`` 0.6 and was decided by that noise. The match is now
decided by vocabulary: no shared word and no ``{slot}`` in the templates means
no template of the intent can produce the utterance, so the score is 0."""
import tempfile
import unittest

from ovos_padatious.intent_container import IntentContainer

HELLO = ["hello", "hi", "hey", "greetings", "good morning"]


class TestLoneIntentUnrelatedUtterance(unittest.TestCase):
    def _container(self, samples, name="hello"):
        c = IntentContainer(tempfile.mkdtemp())
        c.add_intent(name, samples)
        c.train()
        return c

    def test_all_unknown_words_score_zero(self):
        c = self._container(HELLO)
        for utt in ("set a timer for five minutes", "set a timer for 5 minutes"):
            self.assertEqual(c.calc_intent(utt).conf, 0.0, utt)

    def test_a_known_word_still_matches(self):
        c = self._container(HELLO)
        self.assertEqual(c.calc_intent("hello").name, "hello")
        self.assertGreaterEqual(c.calc_intent("hello").conf, 0.9)
        self.assertGreater(c.calc_intent("good afternoon").conf, 0.0,
                           "a shared word keeps the net's verdict")

    def test_a_slot_intent_keeps_the_net_verdict(self):
        c = self._container(["buy {item}", "get {item}"], name="buy")
        self.assertGreater(c.calc_intent("purchase cheese").conf, 0.0,
                           "a {slot} may stand for words never seen")

    def test_the_guard_survives_the_cache(self):
        cache = tempfile.mkdtemp()
        c = IntentContainer(cache)
        c.add_intent("hello", HELLO)
        c.train()
        c2 = IntentContainer(cache)
        c2.add_intent("hello", HELLO)
        c2.train()  # loads the cached net and ids
        self.assertEqual(c2.calc_intent("set a timer for five minutes").conf, 0.0)
