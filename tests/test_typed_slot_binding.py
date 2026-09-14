"""OVOS-INTENT-1 §5.6 — a typed placeholder binds where the map says it may.

§5.6: "An engine **MAY** use the map to constrain where `{type:name}` matches
— preferring or requiring a span the map lists for that type". The template
counts words; the parser reads the datum. "set a timer for about twenty
minutes" gives the template no reason to drop the hedge, so without the map
`{number:amount}` binds "about twenty" and a skill calling int() on it fails.

`Match.slots[name]` stays the surface string either way (PIPELINE-1 §4.3), so
the assertion is on WHICH surface is bound, never on whether a match happened.
A degrading engine matches the intent just as well and binds the wrong span.
"""
import tempfile
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

LANG = "en-US"
SKILL = "timer.skill"
NAME = f"{SKILL}:set_timer"


def _pipeline():
    from ovos_padatious.opm import PadatiousPipeline
    return PadatiousPipeline(FakeBus(), {"instant_train": True})


def _register(pipeline, samples, slot_types=None):
    data = {"name": NAME, "skill_id": SKILL, "lang": LANG, "samples": samples}
    if slot_types is not None:
        data["slot_types"] = slot_types
    pipeline.register_intent(Message("padatious:register_intent", data,
                                     {"skill_id": SKILL}))
    pipeline.train(Message("mycroft.skills.train", {}, {}))


def _typed(utterance, start, end, value, type_name="number"):
    return {type_name: [{"span": [start, end],
                         "surface": utterance[start:end],
                         "value": value}]}


class TestTypedSlotBinding(unittest.TestCase):
    # the template has no reason to drop "about": it binds every word
    # between "for" and "minutes"
    UTTERANCE = "set a timer for about twenty minutes"
    SAMPLES = ["set a timer for {number:amount} minutes",
               "timer {number:amount} minutes"]

    def test_the_typed_span_wins_over_the_template_guess(self):
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        # the parser read "twenty five" as the number; the span invariant
        # holds, so this entry applies to this utterance
        start = self.UTTERANCE.index("twenty")
        typed = _typed(self.UTTERANCE, start, start + len("twenty"), 20)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match, "the intent must still match")
        self.assertEqual(match.match_data.get("amount"), "twenty",
                         "the hedge 'about' is not part of the number")

    def test_an_absent_map_leaves_the_binding_untouched(self):
        """§5.6 degrade: no map, and the typed placeholder behaves as {name}.

        The template counts words, so without a typed-slot map to correct it
        `{number:amount}` covers every word between `for` and `minutes`.
        """
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("amount"), "about twenty",
                         "without a typed-slot map the slot degrades to {name}")

    def test_a_malformed_map_is_ignored_rather_than_raising(self):
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": {"number": [{"span": [0, 3]}]}}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match, "a bad map must not lose the match")
        self.assertEqual(match.match_data.get("amount"), "about twenty",
                         "a malformed map leaves the template binding")

    def test_an_entry_whose_span_does_not_hold_is_not_applied(self):
        """The invariant is the selector: entries are shared across candidates
        and apply only where `utterance[start:end] == surface`."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        typed = {"number": [{"span": [0, 3], "surface": "ninety", "value": 90}]}
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("amount"), "about twenty",
                         "an entry failing the invariant leaves the template binding")

    def test_an_entry_that_does_not_overlap_the_bound_is_not_applied(self):
        """The fallback must go: no listed span overlaps the template guess.

        §5.6: an engine applies an entry only where the invariant holds and
        SHOULD bind "the entry whose span covers the text it matched". When
        none covers it, the template binding stays untouched.
        """
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        # "set" is a valid entry on this utterance, but it does not overlap
        # the template-bound "about twenty".
        typed = {"number": [{"span": [0, 3], "surface": "set", "value": 1}]}
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("amount"), "about twenty",
                         "a non-overlapping entry must not replace the template binding")

    def test_two_slots_of_the_same_type_do_not_collapse(self):
        """Each entry may be assigned to at most one slot of its type."""
        pipeline = _pipeline()
        samples = ["wake me in {number:a} minutes and again in {number:b} minutes"]
        _register(pipeline, samples)
        utterance = "wake me in twenty minutes and again in ten minutes"
        # Only one number is listed in the map; the second slot must keep its
        # own template binding.
        start = utterance.index("twenty")
        typed = _typed(utterance, start, start + len("twenty"), 20)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [utterance], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([utterance], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("a"), "twenty")
        self.assertEqual(match.match_data.get("b"), "ten",
                         "the only listed number must not be reused for b")

    def test_capitalized_utterance_uses_typed_span(self):
        """The invariant is checked on the original utterance, not intent.sent.

        ``intent.sent`` is lowercased, so a map computed on real ASR output
        must still win for capitalized words.
        """
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        utterance = "Set a timer for About Twenty minutes"
        start = utterance.index("Twenty")
        typed = _typed(utterance, start, start + len("Twenty"), 20)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [utterance], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([utterance], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("amount"), "Twenty",
                         "the typed span on capitalized input must win")

    def test_the_declared_slot_name_is_the_bare_name(self):
        """INTENT-4 §6.1: `{number:amount}` declares the slot `amount`."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        recorded = pipeline._intent_slots.get((LANG, NAME)) or frozenset()
        self.assertIn("amount", recorded)
        self.assertNotIn("number:amount", recorded)
