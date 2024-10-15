import unittest

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.opm import PadatiousIntentContainer as IntentContainer, \
    PadatiousPipeline as PadatiousService


class UtteranceIntentMatchingTest(unittest.TestCase):
    def get_service(self):
        intent_service = PadatiousService(FakeBus(),
                                          {"intent_cache": "~/.local/share/mycroft/intent_cache",
                                           "train_delay": 1,
                                           "single_thread": True,
                                           })
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
        self.assertEqual(intent.matches, {'thing': 'Mycroft'})

        # fuzzy regex match - success
        utterance = "tell me everything about Mycroft"
        intent = intent_service.calc_intent(utterance, "en-US")
        self.assertEqual(intent.name, "test2")

        # case depends on padaos vs padatious matching internally
        # padaos (exact matches only) -> keep case
        # padacioso -> keep case
        # padatious -> lower case
        self.assertEqual(intent.matches, {'thing': 'mycroft'})
        self.assertEqual(intent.sent, utterance)
        self.assertTrue(intent.conf <= 0.9)
