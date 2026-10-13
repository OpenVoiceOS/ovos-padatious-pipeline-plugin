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
    """Reference implementation: compile a regex for EVERY line of every
    intent (plain lines included, as padaos always did), ignore the
    prefilter and the plain-line shortcut, and apply the same least-greedy
    rule."""
    from itertools import count
    q = ' ' + query + ' '
    out = {}
    for name, lines in container.intent_lines.items():
        regexes = [container._create_regex(line, name, container.entities, count())
                   for line in sorted(lines, key=len, reverse=True) if line.strip()]
        best, best_len = None, None
        for rx in regexes:
            if rx is None:
                continue
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

    def test_plain_lines_match_like_their_regexes(self):
        c = padaos.IntentContainer()
        c.add_entity("city", ["porto", "lisbon"])
        c.add_intent("time", ["what time is it", "quelle heure est-il", "l'heure",
                              "  hoe  laat is het ", "¿qué hora es?", "à quelle heure"])
        c.add_intent("travel", ["travel to {city}", "travel there"])
        c.add_intent("count", ["count to 3", "count now"])
        c.compile()
        self.assertEqual(c.intents["time"].plain,
                         {("what", "time", "is", "it"), ("quelle", "heure", "est", "il"),
                          ("l", "heure"), ("hoe", "laat", "is", "het"),
                          ("qué", "hora", "es"), ("à", "quelle", "heure")})
        for q in ["what time is it", "WHAT TIME IS IT?", "  what   time is it!! ",
                  "what-time is it", "what time is it now", "whatever time is it",
                  "what time is", "Qué hora es", "QUÉ HORA ES", "que hora es",
                  "hoe laat is het", "à quelle heure ?", "quelle heure est-il",
                  "quelle heure est il", "quelle heure estil", "l heure", "lheure",
                  "L'HEURE ?", "qué hora es?", "travel there", "travel to porto",
                  "travel to lisbon", "count now", "count to 3", "count to 4",
                  "what_time is it", "what time is it 2", ""]:
            got = {m['name']: m['entities'] for m in c.calc_intents(q)}
            self.assertEqual(got, _brute_force(c, q), q)

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
        # digits keep these lines on regexes (plain lines have none)
        c.add_intent("a", ["alpha 1", "alpha 2"])
        c.add_intent("b", ["bravo 3"])
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
        self.assertEqual(list(c.calc_intents(" alpha 1 ")), [])

    def test_single_line_intent_reuses_its_regex_as_prefilter(self):
        c = padaos.IntentContainer()
        c.add_intent("one", ["just 1 line"])
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
