# Copyright 2017 Mycroft AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from threading import Event, Lock
from time import monotonic
from unittest.mock import MagicMock

import pytest

from ovos_padatious.intent_container import IntentContainer


class TestFromDisk(unittest.TestCase):
    test_lines = ['this is a test\n', 'another test\n']
    other_lines = ['something else\n', 'this is a different thing\n']
    test_lines_with_entities = ['this is a {test}\n', 'another {test}\n']
    other_lines_with_entities = [
        'something {other}\n',
        'this is a {other} thing\n']
    test_entities = ['test\n', 'assessment\n']
    other_entities = ['else\n', 'different\n']

    def setUp(self):
        self.cont = IntentContainer('/tmp/cache2')

    def _add_intent(self):
        self.cont.add_intent('test', self.test_lines)
        self.cont.add_intent('other', self.other_lines)
        self.cont.add_entity('test', self.test_entities)
        self.cont.add_entity('other', self.other_entities)
        self.cont.train()
        self._write_train_data()

    def _write_train_data(self):
        fn1 = join('/tmp/cache2', 'test.intent')
        with open(fn1, 'w') as f:
            f.writelines(self.test_lines_with_entities)

        fn2 = join('/tmp/cache2', 'other.intent')
        with open(fn2, 'w') as f:
            f.writelines(self.other_lines_with_entities)

        fn1 = join('/tmp/cache2', 'test.entity')
        with open(fn1, 'w') as f:
            f.writelines(self.test_entities)

        fn2 = join('/tmp/cache2', 'other.entity')
        with open(fn2, 'w') as f:
            f.writelines(self.other_entities)

    def test_instantiate_from_disk(self):
        # train and cache (i.e. persist)
        self._add_intent()

        # instantiate from disk (load cached files)
        cont = IntentContainer('/tmp/cache2')
        cont.instantiate_from_disk()

        assert len(cont.intents.train_data.sent_lists) == 0
        assert len(cont.intents.objects_to_train) == 0
        assert len(cont.intents.objects) == 2

        result = cont.calc_intent('something different')
        assert result.matches['other'] == 'different'


class TestIntentContainer(unittest.TestCase):
    test_lines = ['this is a test\n', 'another test\n']
    other_lines = ['something else\n', 'this is a different thing\n']
    test_lines_with_entities = ['this is a {test}\n', 'another {test}\n']
    other_lines_with_entities = [
        'something {other}\n',
        'this is a {other} thing\n']
    test_entities = ['test\n', 'assessment\n']
    other_entities = ['else\n', 'different\n']
    blist = ["bad", "404"]

    def setUp(self):
        self.cont = IntentContainer('/tmp/cache')

    def _add_intent(self, blacklist=False):
        self.cont.add_intent('test', self.test_lines, blacklisted_words=self.blist if blacklist else [])
        self.cont.add_intent('other', self.other_lines, blacklisted_words=self.blist if blacklist else [])

    def test_load_intent(self):
        fn1 = join('/tmp', 'test.txt')
        with open(fn1, 'w') as f:
            f.writelines(self.test_lines)

        fn2 = join('/tmp', 'other.txt')
        with open(fn2, 'w') as f:
            f.writelines(self.other_lines)

        self.cont.load_intent('test', fn1)
        self.cont.load_intent('other', fn1)
        assert len(self.cont.intents.train_data.sent_lists) == 2

    def test_train(self):
        def test(a, b):
            self._add_intent()
            self.cont.train(a, b)

        test(False, False)
        test(True, True)

    def test_concurrent_calculation_joins_active_training(self):
        cont = IntentContainer('/tmp/cache-concurrent-training',
                               disable_padaos=True)
        training_started = Event()
        release_training = Event()
        count_lock = Lock()
        training_calls = 0

        def slow_train(debug=True):
            nonlocal training_calls
            with count_lock:
                training_calls += 1
            training_started.set()
            assert release_training.wait(2)

        cont.must_train = True
        cont.intents.train = MagicMock(side_effect=slow_train)
        cont.intents.calc_intents = MagicMock(return_value=[])
        cont.entities = MagicMock()

        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(cont.calc_intents, 'test')
                           for _ in range(8)]
                assert training_started.wait(1)
                release_training.set()
                assert [future.result(timeout=2) for future in futures] == [
                    [] for _ in futures]
        finally:
            release_training.set()
            cont.shutdown()

        assert training_calls == 1
        cont.entities.train.assert_called_once_with(debug=True)
        cont.entities.calc_ent_dict.assert_called_once_with()

    def _create_large_intent(self, depth):
        if depth == 0:
            return '(a|b|)'
        return '{0} {0}'.format(self._create_large_intent(depth - 1))

    def test_calc_intents(self):
        self._add_intent()
        self.cont.train(False)

        intents = self.cont.calc_intents('this is another test')
        assert (
                       intents[0].conf > intents[1].conf) == (
                       intents[0].name == 'test')
        assert self.cont.calc_intent('this is another test').name == 'test'

        # this one will fail in next test, should pass here
        intents = self.cont.calc_intents('this is a bad test')
        self.assertEqual(intents[0].name, 'test')

    def test_calc_exact_intents_does_not_schedule_neural_inference(self):
        self.cont.must_train = False
        self.cont.padaos = MagicMock()
        self.cont.padaos.calc_intents.return_value = [{
            'name': 'test',
            'entities': {'thing': 'mycroft'},
        }]
        self.cont.intents.calc_intents = MagicMock()

        matches = self.cont.calc_exact_intents('tell me about mycroft')

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, 'test')
        self.assertEqual(matches[0].conf, 1.0)
        self.assertEqual(matches[0].matches, {'thing': 'mycroft'})
        self.cont.intents.calc_intents.assert_not_called()

    def test_calc_intents_keeps_neural_and_exact_candidates(self):
        self.cont.must_train = False
        neural = MagicMock(name='neural', conf=0.8)
        neural.name = 'neural'
        self.cont.intents.calc_intents = MagicMock(return_value=[neural])
        self.cont.padaos = MagicMock()
        self.cont.padaos.calc_intents.return_value = [{
            'name': 'exact',
            'entities': {},
        }]

        matches = self.cont.calc_intents('query')

        self.assertEqual({match.name for match in matches},
                         {'neural', 'exact'})

    def test_blacklist(self):
        self._add_intent(blacklist=True)
        self.cont.train(False)
        # matched in previous test
        # no match here due to "bad" being in blacklist
        intents = self.cont.calc_intents('this is a bad test')
        self.assertEqual(intents, [])

    def test_empty(self):
        self.cont.train(False)
        self.cont.calc_intent('hello')

    def test_clear_replaces_and_stops_inference_executor(self):
        old_manager = self.cont.intents
        old_manager.shutdown = MagicMock()

        self.cont.clear()

        self.assertIsNot(self.cont.intents, old_manager)
        old_manager.shutdown.assert_called_once_with(wait=False)

    def _test_entities(self, namespace):
        self.cont.add_intent(namespace + 'intent', [
            'test {ent}'
        ])
        self.cont.add_entity(namespace + 'ent', [
            'one'
        ])
        self.cont.train(False)
        data = self.cont.calc_intent('test one')
        high_conf = data.conf
        assert data.conf > 0.5
        assert data['ent'] == 'one'

        data = self.cont.calc_intent('test two')
        assert high_conf > data.conf
        assert 'ent' not in data

    def test_regular_entities(self):
        self._test_entities('')

    def test_namespaced_entities(self):
        self._test_entities('SkillName:')

    def test_remove(self):
        self._add_intent()
        self.cont.train(False)
        assert self.cont.calc_intent('This is a test').conf == 1.0
        self.cont.remove_intent('test')
        assert self.cont.calc_intent('This is a test').conf < 0.5
        self.cont.add_intent('thing', ['A {thing}'])
        self.cont.add_entity('thing', ['thing'])
        self.cont.train(False)
        assert self.cont.calc_intent('A dog').conf < 0.5
        assert self.cont.calc_intent('A thing').conf == 1.0
        self.cont.remove_entity('thing')
        assert self.cont.calc_intent('A dog').conf == 1.0

    def test_overlap(self):
        self.cont.add_intent('song', ['play {song}'])
        self.cont.add_intent('news', ['play the news'])
        self.cont.train(False)
        assert self.cont.calc_intent('play the news').name == 'news'

    def test_overlap_backwards(self):
        self.cont.add_intent('song', ['play {song}'])
        self.cont.add_intent('news', ['play the news'])
        self.cont.train(False)
        assert self.cont.calc_intent('play the news').name == 'news'

    def test_generalize(self):
        self.cont.add_intent('timer', [
            'set a timer for {time} minutes',
            'make a {time} minute timer'
        ])
        self.cont.add_entity('time', [
            '#', '##', '#:##', '##:##'
        ])
        self.cont.train(False)
        intent = self.cont.calc_intent('make a timer for 3 minute')
        assert intent.name == 'timer'
        assert intent.matches == {'time': '3'}
