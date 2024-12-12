import unittest
from unittest.mock import MagicMock

from ovos_padatious.domain_container import DomainIntentContainer  # Replace 'your_module' with the actual module name

from ovos_padatious.match_data import MatchData


class TestDomainIntentEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DomainIntentContainer()

    def test_register_domain_intent(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1", "sample2"])
        self.assertIn("domain1", self.engine.training_data)
        self.assertIn("intent1", self.engine.domains["domain1"].intent_names)

    def test_remove_domain(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1", "sample2"])
        self.engine.remove_domain("domain1")
        self.assertNotIn("domain1", self.engine.training_data)
        self.assertNotIn("domain1", self.engine.domains)

    def test_remove_domain_intent(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1", "sample2"])
        self.engine.remove_domain_intent("domain1", "intent1")
        self.assertNotIn("intent1", self.engine.domains["domain1"].intent_names)

    def test_calc_domains(self):
        self.engine.train = MagicMock()
        self.engine.domain_engine.calc_intents = MagicMock(
            return_value=[MatchData(name="domain1", sent="query", matches=None, conf=0.9)])
        result = self.engine.calc_domains("query")
        self.engine.train.assert_called_once()
        self.assertEqual(result[0].name, "domain1")

    def test_calc_domain(self):
        self.engine.train = MagicMock()
        self.engine.domain_engine.calc_intent = MagicMock(
            return_value=MatchData(name="domain1", sent="query", matches=None, conf=0.9))
        result = self.engine.calc_domain("query")
        self.engine.train.assert_called_once()
        self.assertEqual(result.name, "domain1")

    def test_calc_intent(self):
        self.engine.train = MagicMock()
        mock_domain_container = MagicMock()
        mock_domain_container.calc_intent.return_value = MatchData(name="intent1", sent="query", matches=None, conf=0.9)
        self.engine.domains["domain1"] = mock_domain_container

        self.engine.domain_engine.calc_intent = MagicMock(
            return_value=MatchData(name="domain1", sent="query", matches=None, conf=0.9))
        result = self.engine.calc_intent("query")
        self.assertEqual(result.name, "intent1")

    def test_calc_intents(self):
        self.engine.train = MagicMock()
        mock_domain_container = MagicMock()
        mock_domain_container.calc_intents.return_value = [
            MatchData(name="intent1", sent="query", matches=None, conf=0.9),
            MatchData(name="intent2", sent="query", matches=None, conf=0.8),
        ]
        self.engine.domains["domain1"] = mock_domain_container

        self.engine.domain_engine.calc_intents = MagicMock(
            return_value=[MatchData(name="domain1", sent="query", matches=None, conf=0.9)])
        result = self.engine.calc_intents("query")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "intent1")

    def test_train(self):
        self.engine.training_data["domain1"] = ["sample1", "sample2"]
        self.engine.domain_engine.add_intent = MagicMock()
        self.engine.domain_engine.train = MagicMock()

        mock_domain_container = MagicMock()
        self.engine.domains["domain1"] = mock_domain_container

        self.engine.train()
        self.engine.domain_engine.add_intent.assert_called_with("domain1", ["sample1", "sample2"])
        self.engine.domain_engine.train.assert_called_once()
        mock_domain_container.train.assert_called_once()
        self.assertFalse(self.engine.must_train)


class TestDomainIntentEngineWithLiveData(unittest.TestCase):
    def setUp(self):
        self.engine = DomainIntentContainer()
        # Sample training data
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
        # Register domains and intents
        for domain, intents in self.training_data.items():
            for intent, samples in intents.items():
                self.engine.add_domain_intent(domain, intent, samples)
        self.engine.train()

    def test_live_data_intent_matching(self):
        # Test IOT domain
        query = "Switch on the fan"
        result = self.engine.calc_intent(query, domain="IOT")
        self.assertEqual(result.name, "turn_on_device")
        self.assertGreater(result.conf, 0.8)

        # Test greetings domain
        query = "Hi there"
        result = self.engine.calc_intent(query, domain="greetings")
        self.assertEqual(result.name, "say_hello")
        self.assertGreater(result.conf, 0.8)

        # Test General Knowledge domain
        query = "What is the capital of France?"
        result = self.engine.calc_intent(query, domain="General Knowledge")
        self.assertEqual(result.name, "ask_fact")
        self.assertGreater(result.conf, 0.8)

        # Test Question domain
        query = "Why is the sky blue?"
        result = self.engine.calc_intent(query, domain="Question")
        self.assertEqual(result.name, "ask_question")
        self.assertGreater(result.conf, 0.8)

        # Test Media Playback domain
        query = "Play a song"
        result = self.engine.calc_intent(query, domain="Media Playback")
        self.assertEqual(result.name, "play_music")
        self.assertGreater(result.conf, 0.8)

    def test_live_data_cross_domain_matching(self):
        # Test cross-domain intent matching
        query = "Tell me a fact about space"
        result = self.engine.calc_domain(query)
        self.assertEqual(result.name, "General Knowledge")
        self.assertGreater(result.conf, 0.8)

        # Validate intent from the matched domain
        result = self.engine.calc_intent(query, domain=result.name)
        self.assertEqual(result.name, "ask_fact")
        self.assertGreater(result.conf, 0.8)

    def test_calc_intent_without_domain(self):
        # Test intent calculation without specifying a domain
        query = "Turn on the lights"
        result = self.engine.calc_intent(query)
        self.assertIsNotNone(result.name, "Intent name should not be None")
        self.assertEqual(result.name, "turn_on_device")
        self.assertGreater(result.conf, 0.8)

        query = "Goodbye"
        result = self.engine.calc_intent(query)
        self.assertIsNotNone(result.name, "Intent name should not be None")
        self.assertEqual(result.name, "say_goodbye")
        self.assertGreater(result.conf, 0.8)

        query = "What is quantum mechanics?"
        result = self.engine.calc_intent(query)
        self.assertIsNotNone(result.name, "Intent name should not be None")
        self.assertEqual(result.name, "ask_question")
        self.assertGreater(result.conf, 0.8)


if __name__ == "__main__":
    unittest.main()
