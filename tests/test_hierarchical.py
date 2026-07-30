import tempfile
import unittest
from unittest.mock import MagicMock

from ovos_padatious.hierarchical_container import HierarchicalIntentContainer
from ovos_padatious.match_data import MatchData


class TestHierarchicalIntentContainer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = HierarchicalIntentContainer(cache_dir=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_domain_intent_creates_container(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1", "sample2"])
        self.assertIn("domain1", self.engine.domains)
        self.assertIn("intent1", self.engine.domains["domain1"].intent_names)

    def test_add_domain_intent_feeds_classifier(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample one"])
        self.assertIn("domain1", self.engine._dirty_domains)
        self.engine.train()
        self.assertIn("domain1", self.engine.domain_engine.intent_names)

    def test_remove_domain(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1"])
        self.engine.train()
        self.engine.remove_domain("domain1")
        self.assertNotIn("domain1", self.engine.domains)
        self.assertNotIn("domain1", self.engine.domain_engine.intent_names)

    def test_remove_domain_intent(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1"])
        self.engine.remove_domain_intent("domain1", "intent1")
        self.assertNotIn("intent1", self.engine.domains["domain1"].intent_names)

    def test_calc_intent_routes_to_classified_domain(self):
        self.engine.train = MagicMock()
        self.engine.must_train = False
        self.engine.domain_engine = MagicMock()
        self.engine.domain_engine.calc_intent.return_value = MatchData(
            name="A", sent="q", matches=None, conf=0.9)
        dom_a, dom_b = MagicMock(), MagicMock()
        dom_a.calc_intent.return_value = MatchData(name="a", sent="q", matches=None, conf=0.7)
        dom_b.calc_intent.return_value = MatchData(name="b", sent="q", matches=None, conf=0.95)
        self.engine.domains["A"] = dom_a
        self.engine.domains["B"] = dom_b
        result = self.engine.calc_intent("q")
        # only the classified domain (A) is scored, not the global argmax (B)
        self.assertEqual(result.name, "a")
        dom_b.calc_intent.assert_not_called()

    def test_calc_intent_restricted_to_domain_bypasses_classifier(self):
        self.engine.train = MagicMock()
        self.engine.must_train = False
        self.engine.domain_engine = MagicMock()
        dom_a = MagicMock()
        dom_a.calc_intent.return_value = MatchData(name="a", sent="q", matches=None, conf=0.7)
        self.engine.domains["A"] = dom_a
        result = self.engine.calc_intent("q", domain="A")
        self.assertEqual(result.name, "a")
        self.engine.domain_engine.calc_intent.assert_not_called()

    def test_domain_threshold_rejects_low_confidence(self):
        self.engine.train = MagicMock()
        self.engine.must_train = False
        self.engine.domain_threshold = 0.5
        self.engine.domain_engine = MagicMock()
        self.engine.domain_engine.calc_intent.return_value = MatchData(
            name="A", sent="q", matches=None, conf=0.2)
        dom_a = MagicMock()
        self.engine.domains["A"] = dom_a
        result = self.engine.calc_intent("q")
        self.assertIsNone(result.name)
        dom_a.calc_intent.assert_not_called()

    def test_calc_intents_returns_routed_domain_only(self):
        self.engine.train = MagicMock()
        self.engine.must_train = False
        self.engine.domain_engine = MagicMock()
        self.engine.domain_engine.calc_intent.return_value = MatchData(
            name="A", sent="q", matches=None, conf=0.9)
        dom_a = MagicMock()
        dom_a.calc_intents.return_value = [
            MatchData(name="a", sent="q", matches=None, conf=0.7)]
        self.engine.domains["A"] = dom_a
        result = self.engine.calc_intents("q")
        self.assertEqual([m.name for m in result], ["a"])

    def test_train_clears_must_train(self):
        self.engine.add_domain_intent("domain1", "intent1", ["sample1"])
        self.assertTrue(self.engine.must_train)
        self.engine.train()
        self.assertFalse(self.engine.must_train)


class TestHierarchicalIntentContainerWithLiveData(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = HierarchicalIntentContainer(cache_dir=self._tmp.name)
        self.training_data = {
            "IOT": {
                "turn_on_device": ["Turn on the lights", "Switch on the fan",
                                    "Activate the air conditioner"],
                "turn_off_device": ["Turn off the lights", "Switch off the heater",
                                     "Deactivate the air conditioner"],
            },
            "greetings": {
                "say_hello": ["Hello", "Hi there", "Good morning"],
                "say_goodbye": ["Goodbye", "See you later", "Bye"],
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

    def tearDown(self):
        self._tmp.cleanup()

    def test_restricted_domain_match(self):
        for query, domain, expected in [
            ("Switch on the fan", "IOT", "turn_on_device"),
            ("Hi there", "greetings", "say_hello"),
            ("Play a song", "Media Playback", "play_music"),
        ]:
            result = self.engine.calc_intent(query, domain=domain)
            self.assertEqual(result.name, expected)
            self.assertGreater(result.conf, 0.8)

    def test_two_stage_routing(self):
        for query, expected in [
            ("Turn on the lights", "turn_on_device"),
            ("Goodbye", "say_goodbye"),
            ("Play some music", "play_music"),
        ]:
            result = self.engine.calc_intent(query)
            self.assertEqual(result.name, expected)
            self.assertGreater(result.conf, 0.8)

    def test_calc_domain(self):
        result = self.engine.calc_domain("turn off the heater")
        self.assertEqual(result.name, "IOT")


if __name__ == "__main__":
    unittest.main()
