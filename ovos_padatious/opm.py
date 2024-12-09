# Copyright 2020 Mycroft AI Inc.
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
"""Intent service wrapping padatious."""
from functools import lru_cache
from os.path import expanduser, isfile
from threading import Event, RLock
from typing import Optional, Dict, List, Union

from langcodes import closest_match
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager, Session
from ovos_config.config import Configuration
from ovos_config.meta import get_xdg_base
from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch, IntentMatch
from ovos_utils import flatten_list
from ovos_utils.fakebus import FakeBus
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG, deprecated, log_deprecation
from ovos_utils.xdg_utils import xdg_data_home

from ovos_padatious import IntentContainer as PadatiousIntentContainer
from ovos_padatious.match_data import MatchData as PadatiousIntent


class PadatiousMatcher:
    """Matcher class to avoid redundancy in padatious intent matching."""

    @deprecated("PadatiousMatcher class is deprecated!", "2.0.0")
    def __init__(self, service: 'PadatiousPipeline'):
        self.service = service

    def _match_level(self, utterances, limit, lang=None, message: Optional[Message] = None) -> Optional[IntentMatch]:
        """Match intent and make sure a certain level of confidence is reached.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
            limit (float): required confidence level.
        """
        m: IntentHandlerMatch = self.service._match_level(utterances, limit, lang, message)
        if not m:
            return None
        return IntentMatch("Padatious", m.match_type, m.match_data, m.skill_id, m.utterance)

    def match_high(self, utterances, lang=None, message=None) -> Optional[IntentMatch]:
        """Intent matcher for high confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.service.conf_high, lang, message)

    def match_medium(self, utterances, lang=None, message=None) -> Optional[IntentMatch]:
        """Intent matcher for medium confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.service.conf_med, lang, message)

    def match_low(self, utterances, lang=None, message=None) -> Optional[IntentMatch]:
        """Intent matcher for low confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.service.conf_low, lang, message)


class PadatiousPipeline(ConfidenceMatcherPipeline):
    """Service class for padatious intent matching."""

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):

        super().__init__(bus, config)
        self.lock = RLock()
        core_config = Configuration()
        self.lang = standardize_lang_tag(core_config.get("lang", "en-US"))
        langs = core_config.get('secondary_langs') or []
        langs = [standardize_lang_tag(l) for l in langs]
        if self.lang not in langs:
            langs.append(self.lang)

        self.conf_high = self.config.get("conf_high") or 0.95
        self.conf_med = self.config.get("conf_med") or 0.8
        self.conf_low = self.config.get("conf_low") or 0.5

        intent_cache = expanduser(self.config.get('intent_cache') or
                                  f"{xdg_data_home()}/{get_xdg_base()}/intent_cache")
        self.containers = {lang: PadatiousIntentContainer(f"{intent_cache}/{lang}")
                           for lang in langs}

        self.finished_training_event = Event()  # DEPRECATED
        self.finished_initial_train = False

        self.registered_intents = []
        self.registered_entities = []
        self.max_words = 50  # if an utterance contains more words than this, don't attempt to match

        self.bus.on('padatious:register_intent', self.register_intent)
        self.bus.on('padatious:register_entity', self.register_entity)
        self.bus.on('detach_intent', self.handle_detach_intent)
        self.bus.on('detach_skill', self.handle_detach_skill)
        self.bus.on('intent.service.padatious.get', self.handle_get_padatious)
        self.bus.on('intent.service.padatious.manifest.get', self.handle_padatious_manifest)
        self.bus.on('intent.service.padatious.entities.manifest.get', self.handle_entity_manifest)
        self.bus.on('mycroft.skills.train', self.train)

        LOG.debug('Loaded Padatious intent pipeline')

    @property
    def padatious_config(self) -> Dict:
        log_deprecation("self.padatious_config is deprecated, access self.config directly instead", "2.0.0")
        return self.config

    @padatious_config.setter
    def padatious_config(self, val):
        log_deprecation("self.padatious_config is deprecated, access self.config directly instead", "2.0.0")
        self.config = val

    def _match_level(self, utterances, limit, lang=None, message: Optional[Message] = None) -> Optional[
        IntentHandlerMatch]:
        """Match intent and make sure a certain level of confidence is reached.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
            limit (float): required confidence level.
        """
        LOG.debug(f'Padatious Matching confidence > {limit}')
        # call flatten in case someone is sending the old style list of tuples
        utterances = flatten_list(utterances)
        lang = standardize_lang_tag(lang or self.lang)
        padatious_intent = self.calc_intent(utterances, lang, message)
        if padatious_intent is not None and padatious_intent.conf > limit:
            skill_id = padatious_intent.name.split(':')[0]
            return IntentHandlerMatch(
                match_type=padatious_intent.name,
                match_data=padatious_intent.matches,
                skill_id=skill_id,
                utterance=padatious_intent.sent)

    def match_high(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """Intent matcher for high confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.conf_high, lang, message)

    def match_medium(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """Intent matcher for medium confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.conf_med, lang, message)

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """Intent matcher for low confidence.

        Args:
            utterances (list of tuples): Utterances to parse, originals paired
                                         with optional normalized version.
        """
        return self._match_level(utterances, self.conf_low, lang, message)

    def train(self, message=None):
        """Perform padatious training.

        Args:
            message (Message): optional triggering message
        """
        LOG.debug("Padatious training start")
        if not any(engine.must_train for engine in self.containers.values()):
            LOG.debug(f"Nothing new to train for padatious")
            # inform the rest of the system to not wait for training finish
            self.bus.emit(Message('mycroft.skills.trained'))
            return

        with self.lock:
            for lang in self.containers:
                if self.containers[lang].must_train:
                    LOG.debug(f"Training padatious for lang '{lang}'")
                    self.containers[lang].train()

            LOG.debug(f"Training complete for padatious!")
            if not self.finished_initial_train:
                self.finished_initial_train = True

        # inform the rest of the system to stop waiting for training finish
        self.bus.emit(Message('mycroft.skills.trained'))
        LOG.debug("Padatious training end")

    @deprecated("'wait_and_train' has been deprecated, use 'train' directly", "2.0.0")
    def wait_and_train(self):
        """Wait for minimum time between training and start training."""
        self.train()

    def __detach_intent(self, intent_name):
        """ Remove an intent if it has been registered.

        Args:
            intent_name (str): intent identifier
        """
        if intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            for lang in self.containers:
                self.containers[lang].remove_intent(intent_name)

    def handle_detach_intent(self, message):
        """Messagebus handler for detaching padatious intent.

        Args:
            message (Message): message triggering action
        """
        self.__detach_intent(message.data.get('intent_name'))

    def handle_detach_skill(self, message):
        """Messagebus handler for detaching all intents for skill.

        Args:
            message (Message): message triggering action
        """
        skill_id = message.data['skill_id']
        remove_list = [i for i in self.registered_intents if skill_id in i]
        for i in remove_list:
            self.__detach_intent(i)

    def _register_object(self, message, object_name, register_func):
        """Generic method for registering a padatious object.

        Args:
            message (Message): trigger for action
            object_name (str): type of entry to register
            register_func (callable): function to call for registration
        """
        file_name = message.data.get('file_name')
        samples = message.data.get("samples")
        name = message.data['name']

        LOG.debug('Registering Padatious ' + object_name + ': ' + name)

        if (not file_name or not isfile(file_name)) and not samples:
            LOG.error('Could not find file ' + file_name)
            return

        if not samples and isfile(file_name):
            with open(file_name) as f:
                samples = [line.strip() for line in f.readlines()]

        register_func(name, samples)

        self.finished_initial_train = False
        if self.config.get("instant_train", True):
            self.train(message)

    def register_intent(self, message):
        """Messagebus handler for registering intents.

        Args:
            message (Message): message triggering action
        """
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang_tag(lang)
        if lang in self.containers:
            self.registered_intents.append(message.data['name'])
            self._register_object(message, 'intent', self.containers[lang].add_intent)

    def register_entity(self, message):
        """Messagebus handler for registering entities.

        Args:
            message (Message): message triggering action
        """
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang_tag(lang)
        if lang in self.containers:
            self.registered_entities.append(message.data)
            self._register_object(message, 'entity',
                                  self.containers[lang].add_entity)

    def calc_intent(self, utterances: Union[str, List[str]], lang: Optional[str] = None,
                    message: Optional[Message] = None) -> Optional[PadatiousIntent]:
        """
        Get the best intent match for the given list of utterances. Utilizes a
        thread pool for overall faster execution. Note that this method is NOT
        compatible with Padatious, but is compatible with Padacioso.
        @param utterances: list of string utterances to get an intent for
        @param lang: language of utterances
        @return:
        """
        if isinstance(utterances, str):
            utterances = [utterances]  # backwards compat when arg was a single string
        utterances = [u for u in utterances if len(u.split()) < self.max_words]
        if not utterances:
            LOG.error(f"utterance exceeds max size of {self.max_words} words, skipping padatious match")
            return None

        lang = lang or self.lang

        lang = self._get_closest_lang(lang)
        if lang is None:  # no intents registered for this lang
            return None

        sess = SessionManager.get(message)

        intent_container = self.containers.get(lang)
        intents = [_calc_padatious_intent(utt, intent_container, sess)
                   for utt in utterances]
        intents = [i for i in intents if i is not None]
        # select best
        if intents:
            return max(intents, key=lambda k: k.conf)

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        if self.containers:
            lang = standardize_lang_tag(lang)
            closest, score = closest_match(lang, list(self.containers.keys()))
            # https://langcodes-hickford.readthedocs.io/en/sphinx/index.html#distance-values
            # 0 -> These codes represent the same language, possibly after filling in values and normalizing.
            # 1- 3 -> These codes indicate a minor regional difference.
            # 4 - 10 -> These codes indicate a significant but unproblematic regional difference.
            if score < 10:
                return closest
        return None

    def shutdown(self):
        self.bus.remove('padatious:register_intent', self.register_intent)
        self.bus.remove('padatious:register_entity', self.register_entity)
        self.bus.remove('intent.service.padatious.get', self.handle_get_padatious)
        self.bus.remove('intent.service.padatious.manifest.get', self.handle_padatious_manifest)
        self.bus.remove('intent.service.padatious.entities.manifest.get', self.handle_entity_manifest)
        self.bus.remove('detach_intent', self.handle_detach_intent)
        self.bus.remove('detach_skill', self.handle_detach_skill)

    def handle_get_padatious(self, message):
        """messagebus handler for perfoming padatious parsing.

        Args:
            message (Message): message triggering the method
        """
        utterance = message.data["utterance"]
        lang = message.data.get("lang", self.lang)
        intent = self.calc_intent(utterance, lang=lang)
        if intent:
            intent = intent.__dict__
        self.bus.emit(message.reply("intent.service.padatious.reply",
                                    {"intent": intent}))

    def handle_padatious_manifest(self, message):
        """Messagebus handler returning the registered padatious intents.

        Args:
            message (Message): message triggering the method
        """
        self.bus.emit(message.reply(
            "intent.service.padatious.manifest",
            {"intents": self.registered_intents}))

    def handle_entity_manifest(self, message):
        """Messagebus handler returning the registered padatious entities.

        Args:
            message (Message): message triggering the method
        """
        self.bus.emit(message.reply(
            "intent.service.padatious.entities.manifest",
            {"entities": self.registered_entities}))


@lru_cache(maxsize=3)  # repeat calls under different conf levels wont re-run code
def _calc_padatious_intent(utt: str,
                           intent_container: PadatiousIntentContainer,
                           sess: Session) -> Optional[PadatiousIntent]:
    """
    Try to match an utterance to an intent in an intent_container
    @param utt: str - text to match intent against

    @return: matched PadatiousIntent
    """
    try:
        matches = [m for m in intent_container.calc_intents(utt)
                   if m.name not in sess.blacklisted_intents
                   and m.name.split(":")[0] not in sess.blacklisted_skills]
        if len(matches) == 0:
            return None
        best_match = max(matches, key=lambda x: x.conf)
        best_matches = (
            match for match in matches if match.conf == best_match.conf)
        intent = min(best_matches, key=lambda x: sum(map(len, x.matches.values())))
        intent.sent = utt
        return intent
    except Exception as e:
        LOG.error(e)
