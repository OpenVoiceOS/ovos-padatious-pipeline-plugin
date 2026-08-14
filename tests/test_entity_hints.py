"""Entity value sets are training hints, never closed vocabularies.

OVOS-INTENT-1 §5.4: an ``.entity`` file lists *example* values that bias
scoring. A value outside the list must still fill the slot and must still
produce a viable match.
"""
import tempfile

import pytest

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.match_data import MatchData


@pytest.fixture()
def container():
    return IntentContainer(tempfile.mkdtemp())


def test_out_of_list_value_still_matches(container):
    """The zzyzxvania repro: unknown slot value must not close the vocabulary."""
    container.add_intent('time', ['current time in {location}'])
    container.add_entity('location', ['africa', 'london', 'paris'])
    container.train(debug=False)

    match = container.calc_intent('current time in zzyzxvania')
    assert match.name == 'time'
    assert match.matches.get('location') == 'zzyzxvania'
    assert match.conf > 0.8


def test_in_list_value_is_unpenalised(container):
    """Protect the pre-existing in-list behaviour."""
    container.add_intent('time', ['current time in {location}'])
    container.add_entity('location', ['africa', 'london', 'paris'])
    container.train(debug=False)

    match = container.calc_intent('current time in london')
    assert match.name == 'time'
    assert match.matches.get('location') == 'london'
    assert match.conf == pytest.approx(1.0, abs=1e-6)


def test_in_list_outranks_out_of_list(container):
    """The list acts as a score: on ONE utterance, the intent whose value set
    knows the value must win over the one whose set does not.

    This is a *within-span* property. Across different spans a longer
    out-of-list value can still beat a shorter in-list one ("new york city"
    beats "new york"), exactly as it does pre-fix.
    """
    container.add_intent('known', ['weather in {city}'])
    container.add_intent('unknown', ['weather in {town}'])
    container.add_entity('city', ['london'])
    container.add_entity('town', ['paris'])
    container.train(debug=False)

    # same utterance, same template shape: only the value sets differ
    match = container.calc_intent('weather in london')
    assert match.name == 'known'
    assert match.matches.get('city') == 'london'

    # and a value neither set knows still matches, with the slot captured
    other = container.calc_intent('weather in zzyzxvania')
    assert other.conf > 0.8
    assert 'zzyzxvania' in other.matches.values()
    assert other.conf < match.conf


def test_multi_word_value_with_listed_tail_is_not_truncated(container):
    """A rival span may only be dropped when the words it swallows are
    template literals - never merely because its tail is a listed value."""
    container.add_intent('weather', ['weather in {location}'])
    container.add_entity('location', ['london', 'york', 'porto'])
    container.train(debug=False)

    match = container.calc_intent('weather in new london')
    assert match.name == 'weather'
    assert match.matches.get('location') == 'new london'
    assert match.conf > 0.8


def test_multi_word_value_with_listed_tail_mid_utterance(container):
    container.add_intent('play', ['play {song}'])
    container.add_entity('song', ['rain', 'fire'])
    container.train(debug=False)

    match = container.calc_intent('play ring of fire')
    assert match.name == 'play'
    assert match.matches.get('song') == 'ring of fire'
    assert match.conf > 0.8


def test_template_literals_are_still_stripped_from_the_span(container):
    """The one case the discriminator does fire on: the swallowed words are
    literals of the intent's own templates."""
    container.add_intent('timer', ['set a timer for {time} minutes',
                                   'make a {time} minute timer'])
    container.add_entity('time', ['#', '##', '#:##', '##:##'])
    container.train(debug=False)

    match = container.calc_intent('make a timer for 3 minute')
    assert match.name == 'timer'
    assert match.matches == {'time': '3'}


def test_blocked_skill_wave_datetime(container):
    """ovos-skill-date-time#264: location entity without the asked-for city."""
    container.add_intent('current_time', [
        'what time is it in {location}',
        'current time in {location}',
    ])
    container.add_entity('location', ['london', 'paris', 'new york'])
    container.train(debug=False)

    match = container.calc_intent('what time is it in Coimbra')
    assert match.name == 'current_time'
    assert match.matches.get('location') == 'coimbra'
    assert match.conf > 0.8


def test_blocked_skill_wave_mark1_brightness(container):
    """ovos-skill-mark1-ctrl#38: '#' placeholder lines in a brightness entity.

    ``#``/``##`` are digit placeholders, matching padacioso's ``#`` -> ``\\d``
    convention; padatious implements it in ``IdManager.adj_token``, which
    folds any digit run to ``#`` before scoring. So a numeric value is an
    in-list hit here, and any other value is still captured as a hint miss.
    """
    container.add_intent('brightness', ['set eye brightness to {level} percent',
                                        'set eye brightness to {level}'])
    container.add_entity('level', ['full', 'half', '#', '##'])
    container.train(debug=False)

    match = container.calc_intent('set eye brightness to 42 percent')
    assert match.name == 'brightness'
    assert match.matches.get('level') == '42'
    assert match.conf > 0.8

    named = container.calc_intent('set eye brightness to half')
    assert named.matches.get('level') == 'half'
    assert named.conf > 0.8

    # the numeric case above passes even with a hard allow-list, because '42'
    # folds to the in-list '##'. These do not: they are genuinely out-of-list.
    word = container.calc_intent('set eye brightness to dim')
    assert word.name == 'brightness'
    assert word.matches.get('level') == 'dim'
    assert word.conf > 0.8

    phrase = container.calc_intent('set eye brightness to very dim')
    assert phrase.name == 'brightness'
    assert phrase.matches.get('level') == 'very dim'
    assert phrase.conf > 0.8


def test_empty_entity_list(container):
    container.add_intent('time', ['current time in {location}'])
    container.add_entity('location', [])
    container.train(debug=False)

    match = container.calc_intent('current time in london')
    assert match.matches.get('location') == 'london'
    assert match.conf > 0.8


def test_entity_registered_after_training(container):
    container.add_intent('time', ['current time in {location}'])
    container.train(debug=False)
    container.add_entity('location', ['london'])
    container.train(debug=False)

    match = container.calc_intent('current time in zzyzxvania')
    assert match.matches.get('location') == 'zzyzxvania'
    assert match.conf > 0.8


def test_unicode_out_of_list_value(container):
    container.add_intent('time', ['current time in {location}'])
    container.add_entity('location', ['london'])
    container.train(debug=False)

    match = container.calc_intent('current time in Köln')
    assert match.matches.get('location') == 'köln'
    assert match.conf > 0.8


def test_multi_word_out_of_list_value(container):
    container.add_intent('time', ['current time in {location}'])
    container.add_entity('location', ['london'])
    container.train(debug=False)

    match = container.calc_intent('current time in vila nova de gaia')
    assert match.matches.get('location') == 'vila nova de gaia'
    assert match.conf > 0.8


# ---------------------------------------------------------------------------
# Direct PosIntent.match tests.
#
# The discriminator has branches that end-to-end tests cannot reach, because
# the raw scores come from a neural net and cannot be steered from an
# utterance. These drive PosIntent.match with a stub entity so each branch is
# pinned by a test that fails when the branch is removed.
# ---------------------------------------------------------------------------

class _StubEntity:
    """Entity whose score for a span is looked up, not learned."""

    def __init__(self, scores, default=0.0):
        self.scores = scores
        self.default = default

    def match(self, sent):
        return self.scores.get(' '.join(sent), self.default)


def _pos_intent(container, intent_name):
    intent = [o for o in container.intents.objects if o.name == intent_name][0]
    return intent.pos_intents[0], intent.simple_intent.ids


def _spans(results, token):
    return {' '.join(r.matches[token]) for r in results}


@pytest.fixture()
def timer_grammar():
    c = IntentContainer(tempfile.mkdtemp())
    c.add_intent('timer', ['set a timer for {time} minutes',
                           'make a {time} minute timer'])
    c.add_entity('time', ['#', '##'])
    c.train(debug=False)
    return c


def test_weakly_recognised_span_does_not_discard_rivals(timer_grammar):
    """Mutant killed: ENTITY_HINT_RECOGNISED = 0.0.

    A best raw score below the threshold means the value set has not really
    recognised anything, so it may not throw rival spans away - even ones made
    only of template literals.
    """
    pi, literals = _pos_intent(timer_grammar, 'timer')
    sent = ['make', 'a', 'timer', 'for', '3', 'minute']
    orig = MatchData('timer', sent)

    weak = _StubEntity({'3': 0.4})  # in (0, ENTITY_HINT_RECOGNISED)
    kept = _spans(pi.match(orig, weak, literals), '{time}')
    assert '3' in kept
    assert 'timer for 3' in kept, "a weak best score must not discard rivals"

    # and with the same span scored above the threshold, the rival IS dropped
    strong = _StubEntity({'3': 0.9})
    kept = _spans(pi.match(orig, strong, literals), '{time}')
    assert '3' in kept
    assert 'timer for 3' not in kept


def test_no_entity_never_discards_spans(timer_grammar):
    """Pins the contract behind the ``entity is None`` guard.

    With no value set every span scores raw 1.0, so there is no recognised
    span to discriminate around and no span may be dropped: an entity-less
    slot must behave exactly as if the discriminator did not exist.

    Note this test does NOT kill the mutant that deletes ``entity is None``
    from the guard, and cannot: with all raw scores equal, ``max`` picks the
    lexicographically first span, and no other span is then a strict superset
    of it that swallows only template literals. The guard is defensive - it
    makes the intent explicit and stops a future change to the tie-break from
    silently eating spans. This test pins the observable contract instead.
    """
    pi, literals = _pos_intent(timer_grammar, 'timer')
    sent = ['make', 'a', 'timer', 'for', '3', 'minute']
    orig = MatchData('timer', sent)

    guarded = _spans(pi.match(orig, None, literals), '{time}')
    no_discriminator = _spans(pi.match(orig, None, None), '{time}')
    assert guarded == no_discriminator

    # a registered value set, by contrast, does steer the span
    with_entity = _spans(pi.match(orig, _StubEntity({'3': 0.9}), literals),
                         '{time}')
    assert 'timer for 3' in guarded
    assert 'timer for 3' not in with_entity


def test_subset_span_survives(timer_grammar):
    """Mutant killed: turning the strict-superset ``return False`` into a fall
    through.

    A span *contained* in the recognised one swallows no extra tokens, so
    ``all([])`` is vacuously True and it would be discarded by a discriminator
    that forgot to check containment direction.
    """
    pi, literals = _pos_intent(timer_grammar, 'timer')
    sent = ['make', 'a', 'timer', 'for', '3', 'minute']
    orig = MatchData('timer', sent)

    # score the WIDE span as the recognised one; 'for 3' and '3' are subsets
    entity = _StubEntity({'timer for 3': 0.9})
    kept = _spans(pi.match(orig, entity, literals), '{time}')
    assert 'timer for 3' in kept
    assert '3' in kept, "a subset of the recognised span must survive"


def test_in_list_value_scores_as_hintless_baseline():
    """A listed value must score exactly as if the entity were not attached.

    Attaching an ``.entity`` file is a hint, and OVOS-INTENT-1 5.4 says a hint
    biases scoring, it never punishes: for a value the set *knows*, the
    arithmetic must be bit-identical to the same grammar with no entity at
    all (``ent_conf == 1.0`` both ways). The exact-sample path guarantees
    this deterministically — before it, the net scored listed values ~0.91,
    so registering an entity silently LOWERED every listed value's final
    confidence below ``conf_high`` (the finding-38 regression: a default
    high-only pipeline then dropped the utterance entirely).

    Out-of-list values keep the floor-ramped net score, so this does not
    reintroduce the rescaling promotion the hint ramp was designed against:
    unknown values still cannot be lifted by the map.
    """
    grammar = ['what is the weather in {location}',
               'weather in {location}',
               'weather for {location} today']

    hinted = IntentContainer(tempfile.mkdtemp())
    hinted.add_intent('weather', grammar)
    hinted.add_entity('location', ['london', 'york', 'porto', 'the hague'])
    hinted.train(debug=False)

    bare = IntentContainer(tempfile.mkdtemp())
    bare.add_intent('weather', grammar)
    bare.train(debug=False)

    # the hint-less identity is pinned at the unit level (Entity.match on a
    # listed value returns exactly 1.0, and hint_confidence(1.0) is the
    # identity); at container level the bare grammar is not a usable control
    # because padaos' unconstrained {location} wildcard turns any value into
    # a perfect 1.0 match. Pin the restored value instead: with ent_conf 1.0
    # this utterance scores ~0.9547 — back ABOVE conf_high (0.95), which is
    # where it routed on stable releases (no auto-registered entities), and
    # up from the regressed 0.9318 the old pin froze
    utt = 'what is the weather in porto today'
    hinted_match = hinted.calc_intent(utt)
    assert hinted_match.matches.get('location') == 'porto'
    assert hinted_match.conf == pytest.approx(0.9547, abs=5e-3)
    assert hinted_match.conf > 0.95

    # exact template matches are untouched too
    assert hinted.calc_intent('weather in porto').conf == pytest.approx(1.0, abs=1e-6)

    # and an UNLISTED value must still not be promoted to the identity band:
    # the ramp caps its contribution below ENTITY_HINT_IDENTITY
    unlisted = hinted.calc_intent('what is the weather in zanzibar today')
    listed = hinted_match.conf
    assert unlisted.conf < listed, "unknown value must rank below a listed one"


# ---------------------------------------------------------------------------
# Span ranking below the floor.
#
# A flat floor (``max(raw, 0.8)``) is constant on [0, 0.8], so two spans the
# value set recognises only weakly score identically and the tie falls to
# pos_conf, which prefers the shorter span. These pin the strictly increasing
# map that fixes it.
# ---------------------------------------------------------------------------

@pytest.fixture()
def play_grammar():
    c = IntentContainer(tempfile.mkdtemp())
    c.add_intent('play', ['play {song}', 'play the song {song}',
                          'play {song} by {artist}'])
    c.add_entity('song', ['fire', 'rain', 'song', 'wall'])
    c.add_entity('artist', ['adele', 'queen'])
    c.train(debug=False)
    return c


def test_hint_confidence_is_strictly_increasing():
    """The property the whole design rests on."""
    from ovos_padatious.pos_intent import (ENTITY_HINT_FLOOR,
                                           ENTITY_HINT_IDENTITY,
                                           hint_confidence)

    values = [i / 200.0 for i in range(201)]
    scores = [hint_confidence(v) for v in values]
    for lo, hi in zip(scores, scores[1:]):
        assert hi > lo, "hint_confidence must be strictly increasing"

    # never collapses a candidate...
    assert hint_confidence(0.0) == pytest.approx(ENTITY_HINT_FLOOR)
    # ...and never inflates a value the net actually recognises
    for v in (ENTITY_HINT_IDENTITY, 0.93, 0.95, 1.0):
        assert hint_confidence(v) == pytest.approx(v)


def test_partially_recognised_span_beats_shorter_rival(play_grammar):
    """'song' is a listed value, so 'song 2' is partially recognised and must
    outrank the bare '2' that pos_conf prefers. A flat floor scored both 0.8
    and returned '2'."""
    match = play_grammar.calc_intent('play song 2')
    assert match.name == 'play'
    assert match.matches.get('song') == 'song 2'
    assert match.conf > 0.8


def test_out_of_list_span_not_truncated_to_listed_tail(play_grammar):
    """Same shape, fully unrecognised head."""
    match = play_grammar.calc_intent('play ring of fire')
    assert match.matches.get('song') == 'ring of fire'
    assert match.conf > 0.8


def test_weather_multiword_out_of_list_value(container):
    """'city hall' is wholly out of list; dev dropped the candidate entirely."""
    container.add_intent('weather', ['weather in {location}',
                                     'what is the weather in {location}',
                                     'weather for {location} today'])
    container.add_entity('location', ['london', 'york', 'porto', 'the hague'])
    container.train(debug=False)

    match = container.calc_intent('weather in city hall')
    assert match.name == 'weather'
    assert match.matches.get('location') == 'city hall'
    assert match.conf > 0.8


def test_span_choice_matches_dev_on_ambiguous_filler(play_grammar):
    """Parity pin, not a fix: 'some' is not in any template, so the whole tail
    is the slot value. dev extracts 'some song two' here and so do we - the
    only difference is that dev scored it 0.75 and we score it in medium."""
    match = play_grammar.calc_intent('play some song two')
    assert match.matches.get('song') == 'some song two'
    assert match.conf > 0.8
