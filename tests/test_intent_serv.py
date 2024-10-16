# Copyright 2017 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from unittest import TestCase, mock

from ovos_bus_client.message import Message

from ovos_padatious.opm import PadatiousPipeline


def create_intent_msg(keyword, value):
    """Create a message for registering a padatious intent."""
    return Message('padatious:register_intent',
                   {'name': f"skill:{keyword}", 'samples': [value]},
                   {"skill_id": "skill"})


def create_entity_msg(keyword, value):
    """Create a message for registering a padatious entity."""
    return Message('padatious:register_entity',
                   {'name': f"skill:{keyword}", 'samples': [value]},
                   {"skill_id": "skill"})


def get_last_message(bus):
    """Get last sent message on mock bus."""
    last = bus.emit.call_args
    return last[0][0]


class TestIntentServiceApi(TestCase):
    def setUp(self):
        self.intent_service = PadatiousPipeline(mock.Mock())
        self.setup_simple_padatious_intent()

    def setup_simple_padatious_intent(self,
                                      msg=create_intent_msg('testIntent', 'test'),
                                      msg2=create_entity_msg('testEntity', 'enty')):
        self.intent_service.register_intent(msg)
        self.intent_service.register_entity(msg2)

    def test_get_padatious_intent(self):
        # Check that the intent is returned
        msg = Message('intent.service.padatious.get',
                      data={'utterance': 'test'})
        self.intent_service.handle_get_padatious(msg)

        reply = get_last_message(self.intent_service.bus)

        self.assertEqual(reply.data['intent']['name'],
                         'skill:testIntent')

    def test_get_padatious_intent_no_match(self):
        """Check that if the intent doesn't match at all None is returned."""
        # Check that no intent is matched
        msg = Message('intent.service.padatious.get',
                      data={'utterance': 'five'})
        self.intent_service.handle_get_padatious(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertLess(reply.data["intent"]['conf'], 0.4)

    def test_get_padatious_intent_manifest(self):
        """Make sure the manifest returns a list of Intent Parser objects."""
        msg = Message('intent.service.padatious.manifest.get')
        self.intent_service.handle_padatious_manifest(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intents'][0], 'skill:testIntent')

    def test_get_padatious_vocab_manifest(self):
        msg = Message('intent.service.padatious.entities.manifest.get')
        self.intent_service.handle_entity_manifest(msg)
        reply = get_last_message(self.intent_service.bus)
        value = reply.data["entities"][0]['name']
        keyword = reply.data["entities"][0]['samples'][0]
        self.assertEqual(value, 'skill:testEntity')
        self.assertEqual(keyword, 'enty')

    def test_get_no_match_after_detach(self):
        """Check that a removed intent doesn't match."""
        # Check that no intent is matched
        msg = Message('detach_intent',
                      data={'intent_name': 'skill:testIntent'})
        self.intent_service.handle_detach_intent(msg)
        msg = Message('intent.service.padatious.get', data={'utterance': 'test'})
        self.intent_service.handle_get_padatious(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)

    def test_get_no_match_after_detach_skill(self):
        """Check that a removed skill's intent doesn't match."""
        # Check that no intent is matched
        msg = Message('detach_intent',
                      data={'skill_id': 'skill'})
        self.intent_service.handle_detach_skill(msg)
        msg = Message('intent.service.padatious.get', data={'utterance': 'test'})
        self.intent_service.handle_get_padatious(msg)
        reply = get_last_message(self.intent_service.bus)
        self.assertEqual(reply.data['intent'], None)
