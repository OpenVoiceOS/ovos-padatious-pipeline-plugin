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
        """§5.6 degrade: no map, and the typed placeholder behaves as {name}."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match)
        self.assertIn("amount", match.match_data)

    def test_a_malformed_map_is_ignored_rather_than_raising(self):
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": {"number": [{"span": [0, 3]}]}}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match, "a bad map must not lose the match")

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
        self.assertNotEqual(match.match_data.get("amount"), "ninety")

    def test_the_declared_slot_name_is_the_bare_name(self):
        """INTENT-4 §6.1: `{number:amount}` declares the slot `amount`."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        recorded = pipeline._intent_slots.get((LANG, NAME)) or frozenset()
        self.assertIn("amount", recorded)
        self.assertNotIn("number:amount", recorded)
