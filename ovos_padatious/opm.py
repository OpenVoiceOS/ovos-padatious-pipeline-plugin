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
import re
import string
import unicodedata
from collections import defaultdict
from functools import lru_cache
from os.path import expanduser, isfile
from threading import Event, RLock
from typing import Optional, Dict, List, Union, Type

import snowballstemmer
from langcodes import closest_match
from ovos_config.config import Configuration
from ovos_config.meta import get_xdg_base

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager, Session
from ovos_padatious import IntentContainer
from ovos_padatious.domain_container import DomainIntentContainer
from ovos_padatious.match_data import MatchData as PadatiousIntent
from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch, IntentMatch
from ovos_utils import flatten_list
from ovos_utils.bracket_expansion import expand_template
from ovos_utils.fakebus import FakeBus
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG, deprecated, log_deprecation
from ovos_utils.xdg_utils import xdg_data_home

PadatiousIntentContainer = IntentContainer  # backwards compat

# for easy typing
PadatiousEngine = Union[Type[IntentContainer],
Type[DomainIntentContainer]]


# TODO - move to ovos-utils
@lru_cache()
def remove_accents_and_punct(input_str: str) -> str:
    """
    Normalize the input string by removing accents and punctuation (except for '{' and '}').

    Args:
        input_str (str): The input string to be processed.

    Returns:
        str: The processed string with accents and punctuation removed.
    """
    rm_chars = [c for c in string.punctuation if c not in ("{", "}")]
    # Normalize to NFD (Normalization Form Decomposed), which separates characters and diacritical marks
    nfkd_form = unicodedata.normalize('NFD', input_str)
    # Remove characters that are not ASCII letters or punctuation we want to keep
    return ''.join([char for char in nfkd_form
                    if unicodedata.category(char) != 'Mn' and char not in rm_chars])


# TODO - move to ovos-utils
def deduplicate_list(seq: List[str], keep_order: bool = True) -> List[str]:
    """
    Deduplicate a list while optionally maintaining the original order.

    Args:
        seq (List[str]): The list to deduplicate.
        keep_order (bool): Whether to preserve the order of elements. Default is True.

    Returns:
        List[str]: The deduplicated list.

    Notes:
        If `keep_order` is False, the function uses a set for faster deduplication.
    """
    if not keep_order:
        return list(set(seq))
    else:
        return list(dict.fromkeys(seq))


def normalize_utterances(utterances: List[str], lang: str, cast_to_ascii: bool = True,
                         keep_order: bool = True, stemmer: Optional['Stemmer'] = None) -> List[str]:
    """
    Normalize a list of utterances by collapsing whitespaces, removing accents and punctuation,
    and optionally stemming and deduplicating.

    Args:
        utterances (List[str]): The list of utterances to normalize.
        lang (str): The language code for stemming support.
        cast_to_ascii (bool): Whether to remove accented characters and punctuation. Default is True.
        keep_order (bool): Whether to preserve the order of utterances. Default is True.
        stemmer (Optional[Stemmer]): A stemmer object to stem the utterances (default is None).

    Returns:
        List[str]: The normalized list of utterances.
    """
    # Flatten the list if it's in old style tuple format
    utterances = flatten_list(utterances)  # Assuming flatten_list is defined elsewhere
    # Collapse multiple whitespaces into a single space
    utterances = [re.sub(r'\s+', ' ', u) for u in utterances]
    # Replace accented characters and punctuation if needed
    if cast_to_ascii:
        utterances = [remove_accents_and_punct(u) for u in utterances]
    # strip punctuation marks, that just causes duplicate training data
    utterances = [u.rstrip(string.punctuation) for u in utterances]
    # Stem words if stemmer is provided
    if stemmer is not None:
        utterances = stemmer.stem_sentences(utterances)
    # Deduplicate the list
    utterances = deduplicate_list(utterances, keep_order=keep_order)
    return utterances


class Stemmer:
    """
    A simple wrapper around the Snowball stemmer for various languages.

    Attributes:
        LANGS (dict): A dictionary mapping language codes to Snowball stemmer language names.
    """
    LANGS = {'ar': 'arabic', 'eu': 'basque', 'ca': 'catalan', 'da': 'danish', 'nl': 'dutch', 'en': 'english',
             'fi': 'finnish', 'fr': 'french', 'de': 'german', 'el': 'greek', 'hi': 'hindi', 'hu': 'hungarian',
             'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'lt': 'lithuanian', 'ne': 'nepali',
             'no': 'norwegian', 'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian', 'sr': 'serbian',
             'es': 'spanish', 'sv': 'swedish', 'ta': 'tamil', 'tr': 'turkish'}

    def __init__(self, lang: str):
        """
        Initialize the stemmer for a given language.

        Args:
            lang (str): The language code for stemming.

        Raises:
            ValueError: If the language is unsupported.
        """
        lang2 = closest_match(lang, list(self.LANGS))[0]
        if lang2 == "und":
            raise ValueError(f"unsupported language: {lang}")
        self.snowball = snowballstemmer.stemmer(self.LANGS[lang2])

    @classmethod
    def supports_lang(cls, lang: str) -> bool:
        """
        Check if the given language is supported by the stemmer.

        Args:
            lang (str): The language code to check.

        Returns:
            bool: True if the language is supported, False otherwise.
        """
        lang2 = closest_match(lang, list(cls.LANGS))[0]
        return lang2 != "und"

    def stem_sentence(self, sentence: str) -> str:
        """
        Stem a single sentence.

        Args:
            sentence (str): The sentence to stem.

        Returns:
            str: The stemmed sentence.
        """
        return _cached_stem_sentence(self.snowball, sentence)

    def stem_sentences(self, sentences: List[str]) -> List[str]:
        """
        Stem a list of sentences.

        Args:
            sentences (List[str]): The list of sentences to stem.

        Returns:
            List[str]: The list of stemmed sentences.
        """
        return [self.stem_sentence(s) for s in sentences]


@lru_cache()
def _cached_stem_sentence(stemmer, sentence: str) -> str:
    """
    Cache the stemming of a single sentence to optimize repeated calls.

    Args:
        stemmer: The stemmer instance to use.
        sentence (str): The sentence to stem.

    Returns:
        str: The stemmed sentence.
    """
    stems = stemmer.stemWords(sentence.split())
    return " ".join(stems)


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
                 config: Optional[Dict] = None,
                 engine_class: Optional[PadatiousEngine] = None):

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

        engine_class = engine_class or DomainIntentContainer if self.config.get("domain_engine") else IntentContainer
        LOG.info(f"Padatious class: {engine_class.__name__}")

        self.remove_punct = self.config.get("cast_to_ascii", False)
        use_stemmer = self.config.get("stem", False)
        self.engine_class = engine_class or IntentContainer
        intent_cache = expanduser(self.config.get('intent_cache') or
                                  f"{xdg_data_home()}/{get_xdg_base()}/intent_cache")
        if self.engine_class == DomainIntentContainer:
            # allow user to switch back and forth without retraining
            # cache is cheap, training isn't
            intent_cache += "_domain"
        if use_stemmer:
            intent_cache += "_stemmer"
        if self.remove_punct:
            intent_cache += "_normalized"
        self.containers = {lang: self.engine_class(cache_dir=f"{intent_cache}/{lang}",
                                                   disable_padaos=self.config.get("disable_padaos", False))
                           for lang in langs}

        # pre-load any cached intents
        for container in self.containers.values():
            try:
                container.instantiate_from_disk()
            except Exception as e:
                LOG.error(f"Failed to pre-load cached intents: {str(e)}")

        if use_stemmer:
            self.stemmers = {lang: Stemmer(lang)
                             for lang in langs if Stemmer.supports_lang(lang)}
        else:
            self.stemmers = {}

        self.first_train = Event()
        self.finished_training_event = Event()
        self.finished_training_event.set()  # is cleared when training starts

        self.registered_intents = []
        self.registered_entities = []
        self._skill2intent = defaultdict(list)
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
        lang = standardize_lang_tag(lang or self.lang)

        if lang in self.stemmers:
            stemmer = self.stemmers[lang]
        else:
            stemmer = None
        utterances = normalize_utterances(utterances, lang,
                                          stemmer=stemmer,
                                          keep_order=True,
                                          cast_to_ascii=self.remove_punct)
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
        # wait for any already ongoing training
        # padatious doesnt like threads
        if not self.finished_training_event.is_set():
            self.finished_training_event.wait()
        with self.lock:
            if not any(engine.must_train for engine in self.containers.values()):
                # LOG.debug(f"Nothing new to train for padatious")
                # inform the rest of the system to not wait for training finish
                self.bus.emit(Message('mycroft.skills.trained'))
                self.finished_training_event.set()
                return
            self.finished_training_event.clear()
            # TODO - run this in subprocess?, sometimes fann2 segfaults and kills ovos-core...
            for lang in self.containers:
                if self.containers[lang].must_train:
                    #LOG.debug(f"Training padatious for lang '{lang}'")
                    self.containers[lang].train()

            # inform the rest of the system to stop waiting for training finish
            self.bus.emit(Message('mycroft.skills.trained'))
            self.finished_training_event.set()

        if not self.first_train.is_set():
            self.first_train.set()

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
                for skill_id, intents in self._skill2intent.items():
                    if intent_name in intents:
                        try:
                            if isinstance(self.containers[lang], DomainIntentContainer):
                                self.containers[lang].remove_domain_intent(skill_id, intent_name)
                            else:
                                self.containers[lang].remove_intent(intent_name)
                        except Exception as e:
                            LOG.error(f"Failed to remove intent {intent_name} for skill {skill_id}: {str(e)}")

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
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning("Skill ID is missing. Detaching all anonymous intents")
            skill_id = "anonymous_skill"
        for i in self._skill2intent[skill_id]:
            self.__detach_intent(i)

    def _register_object(self, message, object_name, register_func):
        """Generic method for registering a padatious object.

        Args:
            message (Message): trigger for action
            object_name (str): type of entry to register
            register_func (callable): function to call for registration
        """
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning("Skill ID is missing. Registering under 'anonymous_skill'")
            skill_id = "anonymous_skill"
        file_name = message.data.get('file_name')
        samples = message.data.get("samples")
        name = message.data['name']
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang_tag(lang)

        LOG.debug('Registering Padatious ' + object_name + ': ' + name)

        if (not file_name or not isfile(file_name)) and not samples:
            LOG.error('Could not find file ' + file_name)
            return

        if not samples and isfile(file_name):
            with open(file_name) as f:
                samples = [line.strip() for line in f.readlines()]

        samples = deduplicate_list(flatten_list([expand_template(s) for s in samples]))
        if lang in self.stemmers:
            stemmer = self.stemmers[lang]
        else:
            stemmer = None
        samples = normalize_utterances(samples, lang,
                                       stemmer=stemmer,
                                       keep_order=False,
                                       cast_to_ascii=self.remove_punct)

        if self.engine_class == DomainIntentContainer:
            register_func(skill_id, name, samples)
        else:
            register_func(name, samples)

        if self.config.get("instant_train", False) or self.first_train.is_set():
            self.train(message)

    def register_intent(self, message):
        """Messagebus handler for registering intents.

        Args:
            message (Message): message triggering action
        """
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning("Skill ID is missing. Registering under 'anonymous_skill'")
            skill_id = message.data["skill_id"] = "anonymous_skill"

        self._skill2intent[skill_id].append(message.data['name'])

        lang = message.data.get('lang', self.lang)
        lang = standardize_lang_tag(lang)
        if lang in self.containers:
            self.registered_intents.append(message.data['name'])
            if isinstance(self.containers[lang], DomainIntentContainer):
                self._register_object(message, 'intent', self.containers[lang].add_domain_intent)
            else:
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
            if isinstance(self.containers[lang], DomainIntentContainer):
                self._register_object(message, 'entity', self.containers[lang].add_domain_entity)
            else:
                self._register_object(message, 'entity', self.containers[lang].add_entity)

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
                           intent_container: Union[IntentContainer, DomainIntentContainer],
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
