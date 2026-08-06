import unittest
from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.match_data import MatchData
from ovos_padatious.opm import (
    PadatiousIntentContainer as IntentContainer,
    PadatiousPipeline as PadatiousService,
    _calc_padatious_intent,
)


class UtteranceIntentMatchingTest(unittest.TestCase):
    def get_service(self):
        intent_service = PadatiousService(FakeBus(),
                                          {"intent_cache": "~/.local/share/mycroft/intent_cache",
                                           "train_delay": 1,
                                           "single_thread": True,
                                           "inference_workers": 2,
                                           })
        self.addCleanup(intent_service.shutdown)
        # register test intents
        filename = "/tmp/test.intent"
        with open(filename, "w") as f:
            f.write("this is a test\ntest the intent\nexecute test")
        rxfilename = "/tmp/test2.intent"
        with open(rxfilename, "w") as f:
            f.write("tell me about {thing}\nwhat is {thing}")
        data = {'file_name': filename, 'lang': 'en-US', 'name': 'test'}
        intent_service.register_intent(Message("padatious:register_intent", data))
        data = {'file_name': rxfilename, 'lang': 'en-US', 'name': 'test2'}
        intent_service.register_intent(Message("padatious:register_intent", data))
        intent_service.train()

        return intent_service

    def test_padatious_intent(self):
        intent_service = self.get_service()

        # assert padatious is loaded
        for container in intent_service.containers.values():
            self.assertIsInstance(container, IntentContainer)
            self.assertEqual(container.inference_workers, 2)

        # exact match
        intent = intent_service.calc_intent("this is a test", "en-US")
        self.assertEqual(intent.name, "test")

        # fuzzy match
        intent = intent_service.calc_intent("this test", "en-US")
        self.assertEqual(intent.name, "test")
        self.assertTrue(intent.conf <= 0.8)

        # regex match
        intent = intent_service.calc_intent("tell me about Mycroft", "en-US")
        self.assertEqual(intent.name, "test2")
        self.assertEqual(intent.matches, {'thing': 'mycroft'})

        # fuzzy regex match - success
        utterance = "tell me everything about Mycroft"
        intent = intent_service.calc_intent(utterance, "en-US")
        self.assertEqual(intent.name, "test2")
        self.assertEqual(intent.matches, {'thing': 'mycroft'})
        self.assertEqual(intent.sent, utterance)
        self.assertTrue(intent.conf <= 0.9)

    def test_exact_match_bypasses_neural_candidates(self):
        container = MagicMock()
        exact = MatchData(
            name="weather.skill:current", sent="weather", matches={}, conf=1.0)
        container.calc_exact_intents.return_value = [exact]
        self.addCleanup(_calc_padatious_intent.cache_clear)
        _calc_padatious_intent.cache_clear()

        match = _calc_padatious_intent("weather", container)

        self.assertIsNot(match, exact)
        self.assertEqual(match.__dict__, exact.__dict__)
        container.calc_intents.assert_not_called()

    def test_blocked_exact_match_falls_back_to_neural_candidates(self):
        container = MagicMock()
        blocked = MatchData(
            name="blocked.skill:current", sent="weather", matches={}, conf=1.0)
        allowed = MatchData(
            name="weather.skill:current", sent="weather", matches={}, conf=0.9)
        container.calc_exact_intents.return_value = [blocked]
        container.calc_intents.return_value = [blocked, allowed]
        self.addCleanup(_calc_padatious_intent.cache_clear)
        _calc_padatious_intent.cache_clear()

        match = _calc_padatious_intent(
            "weather", container,
            blacklisted_intents=frozenset({"blocked.skill:current"}))

        self.assertIsNot(match, allowed)
        self.assertEqual(match.__dict__, allowed.__dict__)
        container.calc_intents.assert_called_once_with("weather")
