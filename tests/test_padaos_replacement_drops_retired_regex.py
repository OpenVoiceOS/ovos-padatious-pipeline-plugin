"""Re-registering an existing intent name with different samples must not
keep serving the RETIRED compiled regex - with its now-stale slot captures
- at conf 1.0 until some future background compile happens to run.
Replacement is symmetric with removal (see
``test_immediate_gates_never_wait_on_compile.py``): the old compiled entry
is dropped the instant the new registration lands (``padaos.add_intent``),
so a query for the OLD phrasing in between gets no match at all, never a
stale one. The NEW phrasing only becomes matchable once the next compile
actually runs, same as any other addition.

Adversarial review of PR #124 reproduced this directly: after
re-registering `s:play` as "play {thing} on the radio", "play jazz on tv"
(the OLD phrasing) still matched at conf 1.0 with slots from the retired
regex.
"""
import unittest

from ovos_padatious import padaos


class TestReregistrationDropsRetiredCompiledEntry(unittest.TestCase):
    def test_old_phrasing_never_matches_after_reregistration(self):
        c = padaos.IntentContainer()
        c.add_intent("s:play", ["play {thing} on the radio"])
        c.add_entity("thing", ["jazz", "rock"])
        c.compile()

        old_matches = list(c.calc_intents(" play jazz on the radio "))
        self.assertEqual(len(old_matches), 1)
        self.assertEqual(old_matches[0]['name'], 's:play')

        # re-register the SAME name with DIFFERENT samples; the match path
        # never compiles (see test_padaos_replay_background_compile.py),
        # so nothing recompiles here
        c.add_intent("s:play", ["play {thing} on tv"])

        # the OLD phrasing's compiled regex must be gone immediately - not
        # served again until some future compile happens to run
        self.assertEqual(
            list(c.calc_intents(" play jazz on the radio ")), [],
            "the retired template kept matching after replacement")

        # the new phrasing is a plain addition: not visible until compiled
        self.assertEqual(list(c.calc_intents(" play jazz on tv ")), [])

        c.compile()
        new_matches = list(c.calc_intents(" play jazz on tv "))
        self.assertEqual(len(new_matches), 1)
        self.assertEqual(new_matches[0]['name'], 's:play')
        self.assertEqual(
            list(c.calc_intents(" play jazz on the radio ")), [],
            "the old phrasing must stay gone once the new content compiles")


if __name__ == '__main__':
    unittest.main()
