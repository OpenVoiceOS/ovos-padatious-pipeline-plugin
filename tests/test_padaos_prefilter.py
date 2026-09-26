"""The per-intent prefilter is a fast reject only: it must never change
which intents match, which entities are extracted, or which line wins the
least-greedy tie-break that ``calc_intents`` applies across an intent's
lines.

Each ``self.intents[name]`` carries its ``prefilter`` on the regex list
itself; these tests pin both the equivalence to a brute-force per-line scan
and that the prefilter tracks its regexes across add/remove/recompile.
"""
import unittest

from ovos_padatious import padaos


def _brute_force(container, query):
    """Reference implementation: scan every line of every intent, ignore
    the prefilter entirely, and apply the same least-greedy rule."""
    q = ' ' + query + ' '
    out = {}
    for name, regexes in container.intents.items():
        best, best_len = None, None
        for rx in regexes:
            m = rx.match(q)
            if m is None:
                continue
            ent = {k.rsplit('__', 1)[0].replace('__colon__', ':'): v.strip()
                   for k, v in m.groupdict().items() if v}
            total = sum(len(v) for v in ent.values())
            if best is None or total < best_len:
                best, best_len = ent, total
        if best is not None:
            out[name] = best
    return out


class TestPrefilterEquivalence(unittest.TestCase):
    def setUp(self):
        self.c = padaos.IntentContainer()
        self.c.add_entity("city", ["porto", "lisbon", "london"])
        self.c.add_intent("go", ["travel to {city}", "go to {city}"])
        self.c.add_intent("greet", ["hello there", "hi"])
        # an intent whose two lines can both match the same query, with
        # different captured-entity lengths, exercising least-greedy
        self.c.add_intent("book", ["book {a}", "book {a} for {b}"])
        self.c.compile()

    def _assert_equiv(self, query):
        got = {m['name']: m['entities']
               for m in self.c.calc_intents(query)}
        self.assertEqual(got, _brute_force(self.c, query),
                         f"prefilter diverged from brute force on {query!r}")

    def test_matches_match(self):
        for q in ["travel to lisbon", "go to london", "hello there", "hi",
                  "book a table", "book a table for two", "nothing here",
                  "travel to nowhere"]:
            self._assert_equiv(q)

    def test_least_greedy_line_wins(self):
        # both "book {a}" and "book {a} for {b}" match; the least-greedy
        # rule (smallest total captured length) must pick the same line
        # the plain per-line scan would.
        got = self.c.calc_intent("book x for y")
        ref = _brute_force(self.c, "book x for y")
        self.assertEqual(got['entities'], ref['book'])


class TestPrefilterTracksRegexes(unittest.TestCase):
    def test_prefilter_tracks_intents_across_mutation(self):
        c = padaos.IntentContainer()
        c.add_intent("a", ["alpha one", "alpha two"])
        c.add_intent("b", ["bravo"])
        c.compile()
        self.assertIsNotNone(c.intents["a"].prefilter)

        # removal drops the whole entry, prefilter with it
        c.remove_intent("b")
        self.assertNotIn("b", c.intents)

        # re-registration retires the stale entry immediately
        c.add_intent("a", ["completely different"])
        self.assertNotIn("a", c.intents)
        c.compile()
        self.assertEqual([m['name'] for m in c.calc_intents(" completely different ")],
                         ["a"])
        self.assertEqual(list(c.calc_intents(" alpha one ")), [])

    def test_single_line_intent_reuses_its_regex_as_prefilter(self):
        c = padaos.IntentContainer()
        c.add_intent("one", ["just one line"])
        c.compile()
        self.assertIs(c.intents["one"].prefilter, c.intents["one"][0])

    def test_empty_intent_has_no_prefilter(self):
        c = padaos.IntentContainer()
        c.add_intent("empty", [""])
        c.compile()
        self.assertEqual(c.intents["empty"], [])
        self.assertIsNone(c.intents["empty"].prefilter)


if __name__ == "__main__":
    unittest.main()
