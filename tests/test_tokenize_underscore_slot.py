"""Regression for tokenize() splitting underscore slot names.

Bug: tokenize() treated '_' as a break character, so a template slot token
like '{thing_name}' split into ['{thing', '_', 'name}'] instead of staying
one token. Consequences: Intent.train created a bogus PosIntent token for
the '{thing' fragment, EntityManager.find never resolved the entity (it is
registered as 'name}' under wrap_name('{thing_name}')), and the padaos
exact-regex layer (which parses the brace span correctly on its own) ended
up being the only layer that could narrow a match - so it behaved like a
closed vocabulary for any slot with an underscore in its name, an
out-of-list value matched at NO confidence rather than a mid/hint band.
That is a violation of OVOS-INTENT-1 §5.4 (see the MUST-NOT quoted in
ovos_padatious/pos_intent.py:22-30): a registered entity value set is a
hint, never an exhaustive allow-list.

The fix widens tokenize()'s alpha-like character class to include '_', so
'{thing_name}' tokenizes as one token, matching how it already handles
'{thing}'. See the comment in ovos_padatious/util.py for the utterance-side
tokenization tradeoff this implies for plain-text underscores.
"""
import tempfile

import pytest

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.util import tokenize


def test_slot_placeholder_with_underscore_is_one_token():
    """{pokemon_a} must tokenize as a single slot token, not split on '_'."""
    assert tokenize('{pokemon_a}') == ['{pokemon_a}']


def test_slot_placeholder_without_underscore_still_one_token():
    """Control: the underscore-free case already worked - keep it working."""
    assert tokenize('{pokemon}') == ['{pokemon}']


def test_plain_text_underscore_is_a_single_token():
    """Documented decision: tokenize() is shared by template lines and
    runtime utterances, so widening the alpha class also changes plain-text
    underscore handling. 'foo_bar' now tokenizes as one token rather than
    ['foo', '_', 'bar'] - accepted because word_with_underscore already
    reads as one identifier, and it keeps template/utterance tokenization
    consistent for the same input."""
    assert tokenize('foo_bar') == ['foo_bar']


@pytest.fixture()
def container():
    return IntentContainer(tempfile.mkdtemp())


def test_underscore_slot_out_of_list_value_still_matches(container):
    """The §5.4 guarantee, specifically for an underscore slot name.

    Before the fix: {thing_name} split into ['{thing', '_', 'name}'], so
    an out-of-list value for 'thing_name' matched at NO confidence (the
    padaos exact-regex layer alone determined the outcome, behaving like a
    closed vocabulary). After the fix it must match at a real confidence
    band, mirroring how an underscore-free slot already degrades
    gracefully (see tests/test_entity_hints.py).
    """
    container.add_intent('play', ['play {thing_name} now'])
    container.add_entity('thing_name', ['chess', 'poker'])
    container.train(debug=False)

    match = container.calc_intent('play solitaire now')
    assert match.name == 'play'
    assert match.matches.get('thing_name') == 'solitaire'
    # conservative medium-confidence band: must not be zero/None (the bug),
    # and need not reach the near-1.0 in-list band either.
    assert match.conf is not None
    assert 0.5 < match.conf < 0.99


def test_underscore_slot_in_list_value_matches_high(container):
    container.add_intent('play', ['play {thing_name} now'])
    container.add_entity('thing_name', ['chess', 'poker'])
    container.train(debug=False)

    match = container.calc_intent('play chess now')
    assert match.name == 'play'
    assert match.matches.get('thing_name') == 'chess'
    assert match.conf > 0.9


def test_underscore_free_control_slot_unchanged(container):
    """Same shape, no underscore in the slot name: must behave exactly as
    it did before this fix (it was never broken)."""
    container.add_intent('watch', ['watch {thing} now'])
    container.add_entity('thing', ['chess', 'poker'])
    container.train(debug=False)

    out_of_list = container.calc_intent('watch solitaire now')
    assert out_of_list.name == 'watch'
    assert out_of_list.matches.get('thing') == 'solitaire'
    assert out_of_list.conf is not None
    assert 0.5 < out_of_list.conf < 0.99

    in_list = container.calc_intent('watch chess now')
    assert in_list.name == 'watch'
    assert in_list.matches.get('thing') == 'chess'
    assert in_list.conf > 0.9


# --- digit-boundary gap -----------------------------------------------------
#
# The same bug class survives for a slot name that contains a digit, and it
# is WORSE there: tokenize('{slot_1}') split into ['{slot_', '1', '}'] (three
# pieces, not two), so even an IN-list value failed to resolve, not just an
# out-of-list one. Digits get the same brace-scoped merge as underscores:
# ONLY inside a '{...}' span. Outside braces, digits still split from
# letters exactly as before ('one1' -> ['one', '1'], pinned below and in
# tests/test_util.py::test_tokenize) - IdManager.adj_token() canonicalizes
# isolated pure-digit tokens to '#' placeholders and depends on that split.

def test_slot_placeholder_with_digit_is_one_token():
    assert tokenize('{slot_1}') == ['{slot_1}']


def test_plain_text_alpha_then_digit_still_splits():
    """Pin: outside of braces, a digit still breaks away from a preceding
    alpha/underscore run. Load-bearing for IdManager.adj_token()'s '#'
    canonicalization and for the pre-existing test_util.py::test_tokenize
    pin on 'one1 two2'."""
    assert tokenize('a_1') == ['a_', '1']
    assert tokenize('one1') == ['one', '1']


def test_pure_digit_token_still_splits_as_before():
    assert tokenize('5') == ['5']
    assert tokenize('one1 two2') == ['one', '1', 'two', '2']


def test_underscore_digit_slot_in_list_value_matches_high(container):
    """{thing_1} + registered entity: an IN-list value must resolve at all.
    Before the digit-boundary fix this failed even for in-list values
    (matches={}, conf ~0.10) because the placeholder split into three
    pieces and the slot was never recognized as a slot."""
    container.add_intent('play', ['play {thing_1} now'])
    container.add_entity('thing_1', ['chess', 'poker'])
    container.train(debug=False)

    match = container.calc_intent('play chess now')
    assert match.name == 'play'
    assert match.matches.get('thing_1') == 'chess'
    assert match.conf > 0.9


def test_underscore_digit_slot_out_of_list_value_still_matches(container):
    """Same §5.4 guarantee as test_underscore_slot_out_of_list_value_still_matches,
    for a slot name that also contains a digit."""
    container.add_intent('play', ['play {thing_1} now'])
    container.add_entity('thing_1', ['chess', 'poker'])
    container.train(debug=False)

    match = container.calc_intent('play solitaire now')
    assert match.name == 'play'
    assert match.matches.get('thing_1') == 'solitaire'
    assert match.conf is not None
    assert 0.5 < match.conf < 0.99
