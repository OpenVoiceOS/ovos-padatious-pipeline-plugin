"""Blacklisted words and required keywords in unspaced scripts.

OVOS-INTENT-2 §4.3: Han, Hiragana, Katakana, Thai, Lao, Khmer and Myanmar do
not separate words with spaces, so each character is a word. A keyword of
those languages occurs inside an unspaced utterance, and a Latin keyword
occurs next to Han text with no space between them.
"""
import tempfile
from unittest import TestCase

from ovos_padatious import IntentContainer

SYNONYMS = "在wordnet中搜索同义词"
WIKI = "打开维基百科"


class TestUnspacedScriptKeywords(TestCase):

    def names(self, intents, utterance):
        container = IntentContainer(tempfile.mkdtemp())
        for name, kwargs in intents.items():
            container.add_intent(name, [kwargs.pop("line")], **kwargs)
        container.train(single_thread=True, timeout=120)
        return {m.name for m in container.calc_intents(utterance)}

    def test_latin_required_keyword_next_to_han_text(self):
        intents = {"synonyms": {"line": SYNONYMS,
                                "required_keywords": ["wordnet"]}}
        self.assertIn("synonyms", self.names(intents, SYNONYMS))

    def test_han_required_keyword_inside_han_text(self):
        intents = {"wiki": {"line": WIKI, "required_keywords": ["维基百科"]}}
        self.assertIn("wiki", self.names(intents, WIKI))

    def test_han_blacklisted_word_inside_han_text(self):
        intents = {"wiki": {"line": WIKI, "blacklisted_words": ["维基百科"]}}
        self.assertNotIn("wiki", self.names(intents, WIKI))

    def test_latin_keyword_that_runs_on_is_a_different_word(self):
        line = "在wordnetx中搜索同义词"
        self.assertIn("synonyms",
                      self.names({"synonyms": {"line": line}}, line))
        intents = {"synonyms": {"line": line,
                                "required_keywords": ["wordnet"]}}
        self.assertNotIn("synonyms", self.names(intents, line))
