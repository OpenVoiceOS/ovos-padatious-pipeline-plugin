"""Required keywords (OVOS-INTENT-2 §4.5, OVOS-INTENT-3 §5.6).

A template registration may carry ``required_keywords``. When none of them
occurs in the utterance as a contiguous run of whole words, the intent is
rejected whatever its score. The rejection applies only to that intent.
"""
import tempfile
from unittest import TestCase, mock

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_spec_tools import SpecMessage

from ovos_padatious import IntentContainer
from ovos_padatious.opm import PadatiousPipeline

LANG = "en-US"
WIKIHOW = "wikihow.skill:howto"
QUOTE = "quote.skill:quote"
SIBLING = "other.skill:howto"
REQUIRED = ["wikihow", "wiki how"]
WIKIHOW_SAMPLES = ["how do i {query}",
                   "how do i {query} on wikihow",
                   "search wiki how for {query}"]


def template_msg(full_name, samples, required_keywords=None):
    """An ``ovos.intent.register.template`` payload (INTENT-4 §6.1)."""
    skill_id, intent_name = full_name.split(":")
    data = {"skill_id": skill_id, "intent_name": intent_name,
            "lang": LANG, "samples": samples}
    if required_keywords is not None:
        data["required_keywords"] = required_keywords
    return Message(SpecMessage.INTENT_REGISTER_TEMPLATE, data,
                   {"skill_id": skill_id})


def legacy_msg(full_name, samples, required_keywords=None):
    """A legacy ``padatious:register_intent`` payload."""
    skill_id = full_name.split(":")[0]
    data = {"name": full_name, "lang": LANG, "samples": samples}
    if required_keywords is not None:
        data["required_keywords"] = required_keywords
    return Message("padatious:register_intent", data, {"skill_id": skill_id})


def utter_msg():
    return Message("recognizer_loop:utterance", {},
                   {"session": Session("required-keywords").serialize()})


class _RequiredKeywordsBase:
    """Shared cases; subclasses pick the registration topic."""

    def register(self, full_name, samples, required_keywords=None):
        raise NotImplementedError

    def setUp(self):
        self.pipeline = PadatiousPipeline(mock.Mock())
        self.register(WIKIHOW, WIKIHOW_SAMPLES, REQUIRED)
        self.register(QUOTE, ["tell me a quote", "say a quote"])

    def train(self):
        self.pipeline.containers[LANG].train(single_thread=True, timeout=120)

    def best(self, utterance):
        intent = self.pipeline.calc_intent(utterance, LANG, utter_msg())
        return intent.name if intent else None

    def names(self, utterance):
        return {m.name for m in self.pipeline.containers[LANG].calc_intents(utterance)}

    def test_required_keyword_present_matches(self):
        self.train()
        self.assertEqual(self.best("how do i tie a tie on wikihow"), WIKIHOW)

    def test_no_required_keyword_rejects(self):
        self.train()
        self.assertNotEqual(self.best("how do i tie a tie"), WIKIHOW)
        self.assertNotIn(WIKIHOW, self.names("how do i tie a tie"))

    def test_multi_word_required_keyword_matches(self):
        self.train()
        self.assertEqual(self.best("search wiki how for knots"), WIKIHOW)

    def test_required_keyword_occurs_only_as_whole_words(self):
        self.train()
        for utterance in ("how do i tie a tie on wikihowto",
                          "search wikihowto for knots"):
            self.assertNotEqual(self.best(utterance), WIKIHOW)
            self.assertNotIn(WIKIHOW, self.names(utterance))

    def test_intent_without_required_keywords_unaffected(self):
        self.train()
        self.assertEqual(self.best("tell me a quote"), QUOTE)

    def test_required_keywords_do_not_affect_a_sibling_intent(self):
        self.register(SIBLING, ["how do i {query}"])
        self.train()
        self.assertEqual(self.names("how do i tie a tie") & {WIKIHOW, SIBLING},
                         {SIBLING})
        self.assertEqual(self.best("how do i tie a tie"), SIBLING)


class TestRequiredKeywordsTemplateTopic(_RequiredKeywordsBase, TestCase):
    def register(self, full_name, samples, required_keywords=None):
        self.pipeline.handle_register_template(
            template_msg(full_name, samples, required_keywords))


class TestRequiredKeywordsLegacyTopic(_RequiredKeywordsBase, TestCase):
    def register(self, full_name, samples, required_keywords=None):
        self.pipeline.register_intent(
            legacy_msg(full_name, samples, required_keywords))


class TestRequiredKeywordsNeuralTier(TestCase):
    """With padaos disabled, the neural candidates honour the same rule."""

    def setUp(self):
        self.container = IntentContainer(tempfile.mkdtemp(), disable_padaos=True)
        self.container.add_intent("howto", WIKIHOW_SAMPLES,
                                  required_keywords=REQUIRED)
        self.container.add_intent("quote", ["tell me a quote", "say a quote"])
        self.container.train(single_thread=True, timeout=120)

    def names(self, utterance):
        return {m.name for m in self.container.calc_intents(utterance)}

    def test_no_required_keyword_rejects(self):
        self.assertNotIn("howto", self.names("how do i tie a tie"))
        self.assertIn("quote", self.names("how do i tie a tie"))

    def test_required_keyword_present_matches(self):
        self.assertEqual(self.container.calc_intent(
            "how do i tie a tie on wikihow").name, "howto")
