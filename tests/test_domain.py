import unittest
from unittest.mock import MagicMock

from ovos_padatious.domain_container import DomainIntentContainer
from ovos_padatious.match_data import MatchData


class TestDomainIntentContainer(unittest.TestCase):
    def setUp(self):
        self.engine = DomainIntentContainer()

    def test_add_domain_intent_creates_container(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1", "sample2"])
        self.assertIn("domain1", self.engine.domains)
        self.assertIn("intent1", self.engine.domains["domain1"].intent_names)

    def test_remove_domain(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1"])
        self.engine.remove_domain("domain1")
        self.assertNotIn("domain1", self.engine.domains)

    def test_remove_domain_intent(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1"])
        self.engine.remove_domain_intent("domain1", "intent1")
        self.assertNotIn("intent1", self.engine.domains["domain1"].intent_names)

    def test_calc_intent_picks_global_argmax(self):
        self.engine.train = MagicMock()
        dom_a, dom_b = MagicMock(), MagicMock()
        dom_a.calc_intent.return_value = MatchData(name="a", sent="q", matches=None, conf=0.6)
        dom_b.calc_intent.return_value = MatchData(name="b", sent="q", matches=None, conf=0.9)
        self.engine.domains["A"] = dom_a
        self.engine.domains["B"] = dom_b
        result = self.engine.calc_intent("q")
        self.assertEqual(result.name, "b")
        self.assertEqual(result.conf, 0.9)

    def test_calc_intent_restricted_to_domain(self):
        self.engine.train = MagicMock()
        dom_a, dom_b = MagicMock(), MagicMock()
        dom_a.calc_intent.return_value = MatchData(name="a", sent="q", matches=None, conf=0.6)
        dom_b.calc_intent.return_value = MatchData(name="b", sent="q", matches=None, conf=0.9)
        self.engine.domains["A"] = dom_a
        self.engine.domains["B"] = dom_b
        result = self.engine.calc_intent("q", domain="A")
        self.assertEqual(result.name, "a")

    def test_calc_intents_returns_per_domain_best(self):
        self.engine.train = MagicMock()
        dom_a, dom_b = MagicMock(), MagicMock()
        dom_a.calc_intent.return_value = MatchData(name="a", sent="q", matches=None, conf=0.6)
        dom_b.calc_intent.return_value = MatchData(name="b", sent="q", matches=None, conf=0.9)
        self.engine.domains["A"] = dom_a
        self.engine.domains["B"] = dom_b
        result = self.engine.calc_intents("q")
        self.assertEqual([m.name for m in result], ["b", "a"])

    def test_train_calls_each_container(self):
        dom_a, dom_b = MagicMock(), MagicMock()
        self.engine.domains["A"] = dom_a
        self.engine.domains["B"] = dom_b
        self.engine.train()
        dom_a.train.assert_called_once()
        dom_b.train.assert_called_once()
        self.assertFalse(self.engine.must_train)


class TestDomainIntentContainerWithLiveData(unittest.TestCase):
    def setUp(self):
        self.engine = DomainIntentContainer()
        self.training_data = {
            "IOT": {
                "turn_on_device": ["Turn on the lights", "Switch on the fan", "Activate the air conditioner"],
                "turn_off_device": ["Turn off the lights", "Switch off the heater", "Deactivate the air conditioner"],
            },
            "greetings": {
                "say_hello": ["Hello", "Hi there", "Good morning"],
                "say_goodbye": ["Goodbye", "See you later", "Bye"],
            },
            "General Knowledge": {
                "ask_fact": ["Tell me a fact about space", "What is the capital of France?",
                             "Who invented the telephone?"],
            },
            "Question": {
                "ask_question": ["Why is the sky blue?", "What is quantum mechanics?",
                                 "Can you explain photosynthesis?"],
            },
            "Media Playback": {
                "play_music": ["Play some music", "Start the playlist", "Play a song"],
                "stop_music": ["Stop the music", "Pause playback", "Halt the song"],
            },
        }
        for domain, intents in self.training_data.items():
            for intent, samples in intents.items():
                self.engine.add_domain_intent(domain, intent, samples)
        self.engine.train()

    def test_restricted_domain_match(self):
        for query, domain, expected in [
            ("Switch on the fan", "IOT", "turn_on_device"),
            ("Hi there", "greetings", "say_hello"),
            ("What is the capital of France?", "General Knowledge", "ask_fact"),
            ("Why is the sky blue?", "Question", "ask_question"),
            ("Play a song", "Media Playback", "play_music"),
        ]:
            result = self.engine.calc_intent(query, domain=domain)
            self.assertEqual(result.name, expected)
            self.assertGreater(result.conf, 0.8)

    def test_unrestricted_global_argmax(self):
        for query, expected in [
            ("Turn on the lights", "turn_on_device"),
            ("Goodbye", "say_goodbye"),
            ("What is quantum mechanics?", "ask_question"),
        ]:
            result = self.engine.calc_intent(query)
            self.assertEqual(result.name, expected)
            self.assertGreater(result.conf, 0.8)


if __name__ == "__main__":
    unittest.main()
