"""Literal ``(a|b|c)`` alternation groups written directly in intent lines
must be bounded the same way registered entities already are.

#118 documented that ``PADAOS_ENTITY_INLINE_CAP`` only guards entity
inlining (``entity_lines``): a literal alternation typed straight into an
intent line goes through ``_create_pattern`` unconstrained and can
reproduce the same #115 compile blowup, made hot by an Adapt -> ``.intent``
migration whose generated lines are almost all small (2-6 branch) literal
alternations.

An earlier revision of this fix degraded an over-cap group to the same
wildcard capture used for an over-cap entity. That is unsafe here: unlike
an entity slot, a literal line group has no registered entity/sample list
behind it to check a match against, so the wildcard made the WHOLE LINE
match almost any utterance sharing its surrounding words. An ovoscope e2e
case caught this directly (an unrelated utterance matched an intent whose
only line held an over-cap group). The fix instead treats an over-cap
group the same way ``util.expand_or_skip`` already treats a malformed
line: log a warning and skip that line for padaos entirely - the intent's
other lines still register with padaos, and the neural tier still trains
on this line's expanded samples independently (padaos is only ever a fast
exact-match layer on top of the neural tier, never the sole path).

These tests pin:

* a group at or under the cap compiles byte-identical to the pre-cap
  implementation - the common migration shape must not regress;
* escaped literal parens (``\\(literal\\)``) and a nested group are left
  untouched, never miscounted;
* a group over the cap contributes NO padaos pattern for that line (it is
  dropped, not degraded to a wildcard) while sibling lines of the same
  intent are unaffected;
* an unrelated utterance does not falsely match an intent whose only line
  holds an over-cap group (unit-level pin of the ovoscope e2e regression);
* a container-wide compile that used to take tens of seconds because of
  a large literal line alternation now completes quickly, pinned via the
  degradation MECHANISM rather than an absolute wall-clock band (which
  flakes on shared CI runners).
"""
import time
import unittest
import unittest.mock

import shutil
import tempfile

from ovos_padatious import padaos, IntentContainer
from ovos_padatious.padaos import PADAOS_ENTITY_INLINE_CAP, _LineAlternationCapExceeded


class TestLineAlternationUnderCap(unittest.TestCase):
    """The ~60-intent Adapt migration shape: 2-6 branch literal groups."""

    def test_small_group_is_byte_identical_to_uncapped_pipeline(self):
        line = '(hello|hi|hey) there'
        c = padaos.IntentContainer()
        c.add_intent('greet', [line])
        c.compile()

        expected = '^{}$'.format(c._create_pattern(line))
        self.assertEqual(c.intents['greet'][0].pattern, expected)
        self.assertEqual(c.capped_entities, set())

    def test_small_group_still_matches_every_branch(self):
        c = padaos.IntentContainer()
        c.add_intent('greet', ['(hello|hi|hey) there'])
        c.compile()
        for word in ('hello', 'hi', 'hey'):
            matches = list(c.calc_intents(f' {word} there '))
            self.assertEqual(len(matches), 1)

    def test_escaped_literal_parens_untouched(self):
        line = r'say \(literal\) thing'
        c = padaos.IntentContainer()
        c.add_intent('say', [line])
        c.compile()

        c_ref = padaos.IntentContainer()
        c_ref.add_intent('say', [line])
        expected = '^{}$'.format(c_ref._create_pattern(line))
        self.assertEqual(c.intents['say'][0].pattern, expected)
        self.assertEqual(c.capped_entities, set())

    def test_small_nested_group_untouched(self):
        line = '((a|b)|c) there'
        c = padaos.IntentContainer()
        c.add_intent('nested', [line])
        c.compile()

        c_ref = padaos.IntentContainer()
        expected = '^{}$'.format(c_ref._create_pattern(line))
        self.assertEqual(c.intents['nested'][0].pattern, expected)
        self.assertEqual(c.capped_entities, set())


class TestLineAlternationOverCap(unittest.TestCase):
    """A literal group past the cap must be skipped for padaos entirely,
    never degraded to a wildcard (an over-cap ENTITY slot still may -
    that path has an independent sample list to verify against)."""

    def setUp(self):
        self.n = PADAOS_ENTITY_INLINE_CAP + 20
        self.branches = '|'.join(f'w{i}' for i in range(self.n))

    def test_raw_helper_raises_over_cap(self):
        c = padaos.IntentContainer()
        with self.assertRaises(_LineAlternationCapExceeded) as ctx:
            c._cap_line_alternations(f'({self.branches}) there')
        self.assertEqual(ctx.exception.branch_count, self.n)

    def test_over_cap_line_contributes_no_padaos_pattern(self):
        c = padaos.IntentContainer()
        c.add_intent('go', [f'({self.branches}) there'])
        c.compile()

        # the line was skipped, not degraded: no capped-slot bookkeeping
        # (there is nothing to verify, unlike a capped entity) and no
        # regex registered for it at all
        self.assertEqual(c.capped_entities, set())
        self.assertEqual(c.intents['go'], [])

    def test_over_cap_line_never_matches_anything(self):
        c = padaos.IntentContainer()
        c.add_intent('go', [f'({self.branches}) there'])
        c.compile()

        self.assertEqual(list(c.calc_intents(' w5 there ')), [])
        self.assertEqual(list(c.calc_intents(' anything there ')), [])

    def test_sibling_line_of_same_intent_is_unaffected(self):
        c = padaos.IntentContainer()
        c.add_intent('go', [f'({self.branches}) there', 'a normal line here'])
        c.compile()

        self.assertEqual(len(c.intents['go']), 1)
        matches = list(c.calc_intents(' a normal line here '))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['name'], 'go')

    def test_unrelated_utterance_does_not_match_over_cap_line_intent(self):
        # unit-level pin of the ovoscope e2e regression: an intent whose
        # only line holds an over-cap group must not become an accidental
        # match-anything wildcard for unrelated utterances
        c = padaos.IntentContainer()
        c.add_intent('go', [f'({self.branches}) there'])
        c.compile()
        self.assertEqual(list(c.calc_intents(' set a timer for five minutes ')), [])

    def test_innermost_over_cap_group_in_a_nested_construct_skips_the_line(self):
        # only a paren-free innermost group is ever counted - the outer
        # wrapper (which contains parens) is never itself matched, but an
        # over-cap innermost group still takes the whole line down with it
        c = padaos.IntentContainer()
        line = f'(({self.branches})|other) there'
        c.add_intent('go2', [line])
        c.compile()
        self.assertEqual(c.intents['go2'], [])
        self.assertEqual(list(c.calc_intents(' w5 there ')), [])
        self.assertEqual(list(c.calc_intents(' other there ')), [])


class TestOverCapLineStillTrainsOnTheNeuralSide(unittest.TestCase):
    """padaos is only ever a fast exact-match layer in front of the
    neural tier; dropping an over-cap line for padaos must not drop it
    from training altogether. This is also the unit-level pin of the
    ovoscope e2e regression: an intent whose only line holds an over-cap
    group must not become a match-anything wildcard for an unrelated
    utterance end to end, through the full IntentContainer, not just the
    padaos layer in isolation."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.n = PADAOS_ENTITY_INLINE_CAP + 20
        self.branches = '|'.join(f'thing{i}' for i in range(self.n))
        self.container = IntentContainer(self.tmp)
        self.container.add_intent(
            'go', [f'go to ({self.branches}) now', 'go there right now'])
        self.container.train(debug=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_padaos_only_has_a_pattern_for_the_sibling_line(self):
        # the over-cap line contributed nothing; only the normal sibling
        # line got a padaos pattern
        patterns = self.container.padaos.intents['go']
        self.assertEqual(len(patterns), 1)
        self.assertTrue(patterns[0].match(' go there right now '))

    def test_neural_tier_still_scores_an_in_group_utterance(self):
        # expand_template turned the over-cap group into per-branch
        # training samples same as any other bracket expansion in a
        # .intent line; the neural tier still has them to score against,
        # it just doesn't get padaos' instant exact match for this line
        matches = self.container.calc_intents('go to thing5 now')
        self.assertTrue(any(m.name == 'go' for m in matches))

    def test_sibling_line_still_gets_padaos_exact_match(self):
        matches = self.container.calc_intents('go there right now')
        best = max(matches, key=lambda m: m.conf)
        self.assertEqual(best.name, 'go')
        self.assertEqual(best.conf, 1.0)

    def test_unrelated_utterance_never_gets_padaos_exact_confidence(self):
        # the ovoscope e2e regression, pinned at the unit level: without
        # the fix, a wildcard standing in for the over-cap group made
        # padaos claim an unverified conf=1.0 "perfect match" for almost
        # any utterance sharing the line's surrounding words, regardless
        # of the neural tier's own (much lower) opinion. IntentContainer
        # returns a raw candidate per trained intent regardless of score
        # (confidence thresholding is the pipeline's job, exercised by
        # the ovoscope e2e test), so what must be pinned at this level is
        # that no match reaches padaos' exact conf=1.0 for this utterance.
        matches = self.container.calc_intents('set a timer for five minutes')
        go_confs = [m.conf for m in matches if m.name == 'go']
        self.assertTrue(all(c < 1.0 for c in go_confs))


class TestLineAlternationCompileTimeScalesDown(unittest.TestCase):
    """Regression for the #118 wedge: a large literal alternation typed
    directly into intent lines must not reproduce the #115 compile
    blowup.

    Wall-clock assertions band on shared-runner speed and flake forever
    (a CI run of an earlier revision of this test measured 12.1s against
    a 10s bound on a loaded runner). What must actually be pinned is the
    MECHANISM - the over-cap line is dropped instead of being expanded
    into a giant alternation - plus a very generous ceiling kept only as
    a hang guard, never as a performance claim.
    """

    @staticmethod
    def _build_many_lines_container(n=3000, num_lines=700):
        values = [f'thing number {i:05d} extra words here' for i in range(n)]
        branches = '|'.join(values)
        lines = [f'fetch ({branches}) and other ({branches}) now variant {j}'
                 for j in range(num_lines)]
        c = padaos.IntentContainer()
        c.add_intent('get', lines)
        return c

    def test_large_literal_alternation_across_many_lines_is_dropped(self):
        c = self._build_many_lines_container()
        c.compile()

        # mechanism: every one of the 700 lines held an over-cap group and
        # was dropped for padaos entirely - nothing left to match on
        self.assertEqual(c.intents['get'], [])
        self.assertEqual(c.capped_entities, set())

    def test_large_literal_alternation_across_many_lines_does_not_hang(self):
        # generous smoke ceiling: a pure hang guard, not a perf claim.
        # unbounded, this configuration measured 27-33s on dev hardware
        # and could plausibly run several times slower on a loaded shared
        # runner; 120s leaves wide headroom while still catching a true
        # hang/regression to the unbounded behavior.
        c = self._build_many_lines_container()
        start = time.monotonic()
        c.compile()
        duration = time.monotonic() - start
        self.assertLess(duration, 120.0)

    def test_capping_is_faster_than_no_capping_at_a_shared_scale(self):
        # relative, not absolute: disable capping (simulating the
        # unbounded pre-fix behavior) via the same module-level constant
        # the real cap uses, and compare against the real cap at a scale
        # small enough that both complete quickly and deterministically -
        # this stays meaningful on any machine since it only claims
        # "capped is faster", never a specific duration.
        n, num_lines = 400, 40

        capped = self._build_many_lines_container(n=n, num_lines=num_lines)
        start = time.monotonic()
        capped.compile()
        capped_duration = time.monotonic() - start

        with unittest.mock.patch(
                'ovos_padatious.padaos.PADAOS_ENTITY_INLINE_CAP', n + 1):
            uncapped = self._build_many_lines_container(n=n, num_lines=num_lines)
            start = time.monotonic()
            uncapped.compile()
            uncapped_duration = time.monotonic() - start

        self.assertEqual(capped.intents['get'], [])
        self.assertEqual(len(uncapped.intents['get']), num_lines)
        self.assertLess(capped_duration, uncapped_duration)

    def test_single_large_alternation_line_is_dropped(self):
        n = 2000
        branches = '|'.join(f'word{i:05d}' for i in range(n))
        c = padaos.IntentContainer()
        c.add_intent('x', [f'do the ({branches}) thing'])
        c.compile()

        self.assertEqual(c.intents['x'], [])


if __name__ == '__main__':
    unittest.main()
