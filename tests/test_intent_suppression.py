"""Intent suppression (blacklisted_words): word-boundary matching, and the
padaos exact-template tier honors it identically to the neural tier."""
import tempfile
import unittest

from ovos_padatious import IntentContainer


class TestIntentSuppression(unittest.TestCase):
    def _container(self):
        c = IntentContainer(tempfile.mkdtemp())
        c.add_intent('wiki', ['tell me about {thing} on wikipedia',
                              'look up {thing} on wikipedia'])
        c.blacklisted_words['wiki'] += ['weather', 'install']
        c.train(debug=False)
        return c

    def test_exact_template_match_is_suppressed(self):
        c = self._container()
        # exact template shape: padaos would score this 1.0; the blacklist
        # must apply to that tier too, not just the neural candidates
        matches = c.calc_intents('tell me about the weather on wikipedia')
        self.assertEqual([m for m in matches if m.name == 'wiki'], [])

    def test_word_boundary_not_substring(self):
        c = self._container()
        m = c.calc_intent('tell me about an installment loan on wikipedia')
        self.assertEqual(m.name, 'wiki')
        matches = c.calc_intents('look up how to install firefox on wikipedia')
        self.assertEqual([x for x in matches if x.name == 'wiki'], [])

    def test_unrelated_utterance_unaffected(self):
        c = self._container()
        m = c.calc_intent('tell me about python on wikipedia')
        self.assertEqual(m.name, 'wiki')
        self.assertGreater(m.conf, 0.8)


if __name__ == "__main__":
    unittest.main()
