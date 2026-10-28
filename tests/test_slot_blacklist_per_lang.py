"""Regression test for per-language slot blacklists.

OVOS-INTENT-2 §4.3 per-slot value blacklists are registered per language:
``ovos_workshop.skills.ovos.py``'s ``register_intent_file`` loops
``for lang in self.native_langs`` and calls ``register_template`` once per
language, passing that language's own ``word.blacklist`` values. The engine
used to key its blacklist store by intent name alone, so each language's
registration overwrote the previous one and only the last-registered
language's blacklist survived for every language.

Symptom (ovos-skill-spelling#53): adding it-IT/nl-NL/pt-BR/sv-SE locales made
the en-US utterance "spell it" bind the pronoun literally, because the
surviving blacklist was not English's.
"""
import tempfile
from unittest import TestCase, mock

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_spec_tools import SpecMessage

import ovos_padatious.opm as opm
from ovos_padatious.match_data import MatchData
from ovos_padatious.opm import PadatiousPipeline

SKILL = "spelling.skill"
INTENT = "spell"
FULL = f"{SKILL}:{INTENT}"


def register_msg(lang, blacklist_word):
    data = {"skill_id": SKILL, "intent_name": INTENT, "lang": lang,
            "samples": ["spell {word}"],
            "slot_blacklist": {"word": [blacklist_word]}}
    return Message(SpecMessage.INTENT_REGISTER_TEMPLATE, data,
                   {"skill_id": SKILL})


def utter_msg(intent_context):
    sess = Session("test-session")
    sess.intent_context = intent_context
    return Message("recognizer_loop:utterance", {},
                   {"session": sess.serialize()})


class TestSlotBlacklistPerLang(TestCase):
    """OVOS-INTENT-2 §4.3 per-slot value blacklist, keyed by (lang, name)."""

    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())
        # register a pt-PT container so calc_intent("...", lang="pt-PT")
        # resolves to pt-PT itself rather than falling back to en-US
        self.pipeline.containers.setdefault(
            "pt-PT", self.pipeline.engine_class(
                cache_dir=tempfile.mkdtemp(),
                disable_padaos=self.pipeline.config.get("disable_padaos", False)))
        # en-US registers first, pt-PT second, mirroring a multi-lang skill's
        # native_langs registration loop
        self.pipeline.handle_register_template(register_msg("en-US", "it"))
        self.pipeline.handle_register_template(register_msg("pt-PT", "isso"))

    def _stub_match(self, matches, conf=0.9):
        candidate = MatchData(FULL, "spell x", matches=dict(matches), conf=conf)
        patcher = mock.patch.object(opm, "_calc_padatious_intent",
                                    return_value=candidate)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_each_lang_keeps_its_own_blacklist(self):
        # en-US's own blacklisted pronoun ("it") must still be rejected after
        # pt-PT registered later with a different blacklist
        self._stub_match(matches={"word": "it"})
        result = self.pipeline.calc_intent("spell it", "en-US", utter_msg({}))
        self.assertIsNotNone(result)
        self.assertNotIn("word", result.matches)

        # pt-PT's own blacklisted pronoun ("isso") is likewise rejected
        self._stub_match(matches={"word": "isso"})
        result = self.pipeline.calc_intent("spell isso", "pt-PT", utter_msg({}))
        self.assertIsNotNone(result)
        self.assertNotIn("word", result.matches)

        # a value not on en-US's blacklist (pt-PT's word) still binds
        # normally in en-US, proving the two languages' blacklists are
        # independent
        self._stub_match(matches={"word": "isso"})
        result = self.pipeline.calc_intent("spell isso", "en-US", utter_msg({}))
        self.assertIsNotNone(result)
        self.assertEqual(result.matches.get("word"), "isso")
