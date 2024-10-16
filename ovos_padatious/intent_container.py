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
import inspect
import os
from functools import wraps
from typing import List, Dict, Any, Optional

from ovos_utils.log import LOG

from ovos_padatious import padaos
from ovos_padatious.entity import Entity
from ovos_padatious.entity_manager import EntityManager
from ovos_padatious.intent_manager import IntentManager
from ovos_padatious.match_data import MatchData
from ovos_padatious.util import tokenize


def _save_args(func):
    """
    Decorator that saves the arguments passed to the function in the serialized_args attribute of the class.

    Args:
        func (function): The function to be decorated.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        kwargs = bound_args.arguments
        kwargs['__name__'] = func.__name__
        kwargs.pop('self').serialized_args.append(kwargs)

    return wrapper


class IntentContainer:
    """
    Creates an IntentContainer object used to load and match intents

    Args:
        cache_dir (str): Directory for caching the neural network models and intent/entity files.
    """

    def __init__(self, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir: str = cache_dir
        self.must_train: bool = False
        self.intents: IntentManager = IntentManager(cache_dir)
        self.entities: EntityManager = EntityManager(cache_dir)
        self.padaos: padaos.IntentContainer = padaos.IntentContainer()
        self.train_thread: Optional[Any] = None  # deprecated
        self.serialized_args: List[Dict[str, Any]] = []  # Serialized calls for training intents/entities

    def clear(self) -> None:
        """
        Clears the current intent and entity managers and resets the container.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        self.must_train = False
        self.intents = IntentManager(self.cache_dir)
        self.entities = EntityManager(self.cache_dir)
        self.padaos = padaos.IntentContainer()
        self.serialized_args = []

    def instantiate_from_disk(self) -> None:
        """
        Instantiates the necessary (internal) data structures when loading persisted model from disk.
        This is done via injecting entities and intents back from cached file versions.
        """
        entity_traindata: Dict[str, List[str]] = {}
        intent_traindata: Dict[str, List[str]] = {}

        # workaround: load training data for both entities and intents since
        # padaos regex needs it for (re)compilation until TODO is cleared
        for f in os.listdir(self.cache_dir):
            if f.endswith('.entity'):
                entity_name = f[0:f.find('.entity')]
                with open(os.path.join(self.cache_dir, f), 'r') as d:
                    entity_traindata[entity_name] = [line.strip()
                                                     for line in d]

            elif f.endswith('.intent'):
                intent_name = f[0:f.find('.intent')]
                with open(os.path.join(self.cache_dir, f), 'r') as d:
                    intent_traindata[intent_name] = [line.strip()
                                                     for line in d]

        # TODO: padaos.compile (regex compilation) is redone when loading: find
        # a way to persist regex, as well!
        for f in os.listdir(self.cache_dir):
            if f.startswith('{') and f.endswith('}.hash'):
                entity_name = f[1:f.find('}.hash')]
                if entity_name in entity_traindata:
                    self.add_entity(
                        name=entity_name,
                        lines=entity_traindata[entity_name],
                        reload_cache=False,
                        must_train=False
                    )
            elif not f.startswith('{') and f.endswith('.hash'):
                intent_name = f[0:f.find('.hash')]
                if intent_name in intent_traindata:
                    self.add_intent(
                        name=intent_name,
                        lines=intent_traindata[intent_name],
                        reload_cache=False,
                        must_train=False
                    )

    @_save_args
    def add_intent(self, name: str, lines: List[str], reload_cache: bool = False, must_train: bool = True) -> None:
        """
        Creates a new intent, optionally checking the cache first

        Args:
            name (str): Name of the intent.
            lines (List[str]): Sentences that will activate the intent.
            reload_cache (bool): Whether to ignore cached intent.
            must_train (bool): Whether the model needs training after adding the intent.
        """
        self.intents.add(name, lines, reload_cache, must_train)
        self.padaos.add_intent(name, lines)
        self.must_train = must_train

    @_save_args
    def add_entity(self, name: str, lines: List[str], reload_cache: bool = False, must_train: bool = True) -> None:
        """
        Adds an entity that matches the given lines.

        Example:
            self.add_intent('weather', ['will it rain on {weekday}?'])
            self.add_entity('weekday', ['monday', 'tuesday', 'wednesday'])  # ...

        Args:
            name (str): Name of the entity.
            lines (List[str]): Example extracted entities.
            reload_cache (bool): Whether to refresh the cache.
            must_train (bool): Whether the model needs training after adding the entity.
        """
        Entity.verify_name(name)
        self.entities.add(
            Entity.wrap_name(name),
            lines,
            reload_cache,
            must_train)
        self.padaos.add_entity(name, lines)
        self.must_train = must_train

    @_save_args
    def load_entity(self, name: str, file_name: str, reload_cache: bool = False, must_train: bool = True) -> None:
        """
       Loads an entity, optionally checking the cache first

       Args:
           name (str): The associated name of the entity
           file_name (str): The location of the entity file
           reload_cache (bool): Whether to refresh all of cache
            must_train (bool): Whether the model needs training after loading the entity.
        """
        Entity.verify_name(name)
        self.entities.load(Entity.wrap_name(name), file_name, reload_cache)
        with open(file_name) as f:
            self.padaos.add_entity(name, f.read().split('\n'))
        self.must_train = must_train

    @_save_args
    def load_file(self, *args, **kwargs):
        """Legacy. Use load_intent instead"""
        self.load_intent(*args, **kwargs)

    @_save_args
    def load_intent(self, name: str, file_name: str, reload_cache: bool = False, must_train: bool = True) -> None:
        """
        Loads an intent, optionally checking the cache first

        Args:
            name (str): The associated name of the intent
            file_name (str): The location of the intent file
            reload_cache (bool): Whether to refresh all of cache
            must_train (bool): Whether the model needs training after loading the intent.
        """
        self.intents.load(name, file_name, reload_cache)
        with open(file_name) as f:
            self.padaos.add_intent(name, f.read().split('\n'))
        self.must_train = must_train

    @_save_args
    def remove_intent(self, name: str) -> None:
        """
        Removes an intent by its name.

        Args:
            name (str): Name of the intent to remove.
        """
        self.intents.remove(name)
        self.padaos.remove_intent(name)
        self.must_train = True

    @_save_args
    def remove_entity(self, name: str) -> None:
        """
        Removes an entity by its name.

        Args:
            name (str): Name of the entity to remove.
        """
        self.entities.remove(name)
        self.padaos.remove_entity(name)

    def train(self, debug: bool = True, force: bool = False, single_thread: Optional[bool] = None,
              timeout: Optional[float] = None) -> bool:
        """
        Trains all the loaded intents that need to be updated
        If a cache file exists with the same hash as the intent file,
        the intent will not be trained and just loaded from file

        Args:
            debug (bool): Whether to print a message to stdout each time a new intent is trained
            force (bool): Whether to force training if already finished
            single_thread (bool): DEPRECATED
            timeout (float): DEPRECATED
        Returns:
            bool: True if training succeeded
        """
        if single_thread is not None:
            LOG.warning("'single_thread' argument is deprecated and will be ignored")
        if timeout is not None:
            LOG.warning("'timeout' argument is deprecated and will be ignored")
        if not self.must_train and not force:
            return True
        self.padaos.compile()

        # Train intents and entities
        self.intents.train(debug=debug)
        self.entities.train(debug=debug)

        self.entities.calc_ent_dict()

        self.must_train = False
        return True

    def calc_intents(self, query: str) -> List[MatchData]:
        """
        Tests all the intents against the query and returns
        data on how well each one matched against the query

        Args:
            query (str): Input sentence to test against intents.

        Returns:
            List[MatchData]: A list of all intent matches with confidence scores.
        """
        if self.must_train:
            self.train()
        intents = {i.name: i for i in self.intents.calc_intents(query, self.entities)}
        sent = tokenize(query)
        for perfect_match in self.padaos.calc_intents(query):
            name = perfect_match['name']
            intents[name] = MatchData(name, sent, matches=perfect_match['entities'], conf=1.0)
        return list(intents.values())

    def calc_intent(self, query: str) -> MatchData:
        """
        Returns the best intent match for the given query.

        Args:
            query (str): Input sentence to test against intents.

        Returns:
            MatchData: The best matching intent.
        """
        matches = self.calc_intents(query)
        if not matches:
            return MatchData('', '')
        best_match = max(matches, key=lambda x: x.conf)
        best_matches = [match for match in matches if match.conf == best_match.conf]
        return min(best_matches, key=lambda x: sum(map(len, x.matches.values())))

    def get_training_args(self) -> List[Dict[str, Any]]:
        """
        Returns all serialized arguments used for training intents and entities.

        Returns:
            List[Dict[str, Any]]: List of serialized arguments for training.
        """
        return self.serialized_args

    def apply_training_args(self, data):
        for params in data:
            func_name = params.pop('__name__')
            getattr(self, func_name)(**params)
