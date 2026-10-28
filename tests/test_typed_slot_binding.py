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

    def test_a_typed_slot_that_binds_no_listed_span_does_not_match(self):
        """§5.6 "requiring a span the map lists for that type": the map places
        the only number at "set", so the template guess "about twenty" holds
        no number and the intent does not match."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        typed = {"number": [{"span": [0, 3], "surface": "set", "value": 1}]}
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": typed}, {})
        self.assertIsNone(pipeline.match_high([self.UTTERANCE], LANG, message),
                          "a typed slot must not bind words the map lists no value in")

    def test_a_type_the_map_does_not_list_leaves_the_slot_to_the_template(self):
        """The transformer omits a type that produced nothing, and also one
        whose parser lacks the language, so a map without numbers says
        nothing about "about twenty"."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        typed = {"date": [{"span": [0, 3], "surface": "set",
                           "value": "2030-01-01T07:00:00+00:00"}]}
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("amount"), "about twenty")

    def test_two_slots_of_the_same_type_do_not_collapse(self):
        """Each entry may be assigned to at most one slot of its type."""
        pipeline = _pipeline()
        samples = ["wake me in {number:a} minutes and again in {number:b} minutes"]
        _register(pipeline, samples)
        utterance = "wake me in twenty minutes and again in ten minutes"
        # both slots overlap only their own entry, and each entry is used once
        twenty, ten = utterance.index("twenty"), utterance.index("ten ")
        typed = {"number": [
            {"span": [ten, ten + 3], "surface": "ten", "value": 10},
            {"span": [twenty, twenty + 6], "surface": "twenty", "value": 20}]}
        message = Message("recognizer_loop:utterance",
                          {"utterances": [utterance], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([utterance], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("a"), "twenty")
        self.assertEqual(match.match_data.get("b"), "ten",
                         "each slot binds its own listed number")

    def test_capitalized_utterance_uses_typed_span(self):
        """The invariant is checked on the original casing of the utterance.

        Padatious matches lowercased text, so a map computed on real ASR
        output must still win for capitalized words.
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

    def test_an_entry_from_another_candidate_is_not_applied(self):
        """§5.6: the map is shared by every candidate, and an entry applies
        only to the candidate the engine matched. Here the entry holds on
        candidate 0, but candidate 1 wins the match."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        noisy = "um so anyway set a timer for ninety nine minutes or so please"
        clean = "set a timer for ninety minutes"
        start = noisy.index("ninety nine")
        typed = _typed(noisy, start, start + len("ninety nine"), 99)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [noisy, clean], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([noisy, clean], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.utterance, clean,
                         "control: the clean candidate must win the match")
        self.assertEqual(match.match_data.get("amount"), "ninety",
                         "an entry of another candidate must not bind")

    def test_an_empty_candidate_list_in_the_message_degrades(self):
        """A message with no `utterances` names no candidate, so no entry
        applies and the template binding stays. It must not raise."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        start = self.UTTERANCE.index("twenty")
        typed = _typed(self.UTTERANCE, start, start + len("twenty"), 20)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("amount"), "about twenty")

    def test_the_declared_slot_name_is_the_bare_name(self):
        """INTENT-4 §6.1: `{number:amount}` declares the slot `amount`."""
        pipeline = _pipeline()
        _register(pipeline, self.SAMPLES)
        recorded = pipeline._intent_slots.get((LANG, NAME)) or frozenset()
        self.assertIn("amount", recorded)
        self.assertNotIn("number:amount", recorded)


class TestClosestTypedEntry(unittest.TestCase):
    """Offsets are code points of the original utterance (§5.6 `span`)."""

    def test_a_length_changing_lowercase_keeps_offsets(self):
        from ovos_padatious.opm import _closest_typed_entry
        # U+0130 lowercases to two code points, which shifts every offset
        # after it in the lowered string.
        utterance = "\u0130\u0130\u0130\u0130\u0130\u0130 ten twenty"
        self.assertNotEqual(len(utterance), len(utterance.lower()),
                            "control: the lowercase must change the length")
        ten = {"span": [7, 10], "surface": "ten", "value": 10}
        twenty = {"span": [11, 17], "surface": "twenty", "value": 20}
        self.assertEqual(utterance[7:10], "ten")
        self.assertEqual(utterance[11:17], "twenty")
        self.assertIs(_closest_typed_entry([ten, twenty], utterance, "ten"), ten)
        self.assertIs(_closest_typed_entry([ten, twenty], utterance, "twenty"),
                      twenty)


class TestSlotTypesFromTheSpecRegistration(unittest.TestCase):
    """ovos-workshop strips the type prefix before it registers (INTENT-1
    §3.4 degrade at the producer) and carries the types in ``slot_types`` on
    ``ovos.intent.register.template`` (INTENT-4 §6.1). The engine must read
    the types from that payload: the samples no longer state them."""
    UTTERANCE = "set the brightness to twenty five please"

    def test_the_typed_span_wins_when_types_arrive_on_the_spec_topic(self):
        pipeline = _pipeline()
        pipeline.handle_register_template(Message(
            "ovos.intent.register.template",
            {"skill_id": SKILL, "intent_name": "set_brightness", "lang": LANG,
             "samples": ["set the brightness to {b}"],
             "slot_types": {"b": "number"}},
            {"skill_id": SKILL}))
        pipeline.train(Message("mycroft.skills.train", {}, {}))
        start = self.UTTERANCE.index("twenty")
        typed = _typed(self.UTTERANCE, start, start + len("twenty five"), 25)
        message = Message("recognizer_loop:utterance",
                          {"utterances": [self.UTTERANCE], "lang": LANG,
                           "typed_slots": typed}, {})
        match = pipeline.match_high([self.UTTERANCE], LANG, message)
        self.assertIsNotNone(match, "the intent must still match")
        self.assertEqual(match.match_data.get("b"), "twenty five",
                         "the trailing 'please' is not part of the number")


class TestDateSlotRejectsNonDates(unittest.TestCase):
    """A ``{date:time}`` slot never binds "play jazz".

    The alerts skill ships both intents below. Without the typed-slot map
    requirement, reschedule_alert can bind ``time="play jazz"``.
    """
    SKILL = "alerts.skill"

    def setUp(self):
        self.pipeline = _pipeline()
        for intent, samples in (
                ("reschedule_alert", ["change my {date:old} alarm to {date:time}",
                                      "move my {date:old} alarm to {date:time}"]),
                ("change_media_properties", ["change my {schedkind} to play {mediakind}",
                                             "make my {schedkind} play {mediakind}"])):
            self.pipeline.register_intent(Message(
                "padatious:register_intent",
                {"name": f"{self.SKILL}:{intent}", "skill_id": self.SKILL,
                 "lang": LANG, "samples": samples}, {"skill_id": self.SKILL}))
        self.pipeline.train(Message("mycroft.skills.train", {}, {}))

    def _match(self, utterance, dates):
        typed = {"date": [{"span": [utterance.index(d), utterance.index(d) + len(d)],
                           "surface": d, "value": value} for d, value in dates.items()]}
        message = Message("recognizer_loop:utterance",
                          {"utterances": [utterance], "lang": LANG,
                           "typed_slots": typed}, {})
        return self.pipeline.match_low([utterance], LANG, message)

    def test_words_that_hold_no_date_are_not_a_time(self):
        match = self._match("change my 7 am alarm to play jazz",
                            {"7 am": "2030-01-01T07:00:00+00:00"})
        self.assertTrue(match is None or match.match_type != f"{self.SKILL}:reschedule_alert",
                        f"reschedule_alert bound {match and match.match_data}")

    def test_a_listed_date_is_a_time(self):
        match = self._match("change my 7 am alarm to 8 am",
                            {"7 am": "2030-01-01T07:00:00+00:00",
                             "8 am": "2030-01-01T08:00:00+00:00"})
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, f"{self.SKILL}:reschedule_alert")
        self.assertEqual(match.match_data.get("time"), "8 am")
