"""padaos: a plain line matches the way its regex did, letters that
re.IGNORECASE treats as one included."""
import unittest

from ovos_padatious import padaos


def matches(line, query):
    container = padaos.IntentContainer()
    container.add_intent("skill:intent", [line])
    container.compile()
    return bool(list(container.calc_intents(query)))


class TestPlainLineCaseFolding(unittest.TestCase):
    def test_turkish_dotless_i(self):
        self.assertTrue(matches("tamam bir yardım", "TAMAM BIR YARDIM"))
        self.assertTrue(matches("tamam bir yardım", "tamam bir yardım"))

    def test_long_s(self):
        self.assertTrue(matches("so", "ſo"))
        self.assertTrue(matches("ſo", "SO"))

    def test_micro_sign(self):
        self.assertTrue(matches("µg", "μg"))
        self.assertTrue(matches("μg", "ΜG"))

    def test_final_sigma(self):
        self.assertTrue(matches("οδοσ", "οδος"))
        self.assertTrue(matches("οδος", "ΟΔΟΣ"))

    def test_other_words_still_differ(self):
        self.assertFalse(matches("tamam bir yardım", "TAMAM BIR YARDIMI"))
        self.assertFalse(matches("so", "sa"))


if __name__ == "__main__":
    unittest.main()
