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
from collections import defaultdict
from functools import lru_cache
from os.path import expanduser, isfile
from threading import Event, RLock
from typing import Optional, Dict, List, Union, Type

import snowballstemmer
from ovos_config.config import Configuration
from ovos_config.meta import get_xdg_base

from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager, Session
from ovos_padatious import IntentContainer
from ovos_padatious.domain_container import DomainIntentContainer
from ovos_padatious.match_data import MatchData as PadatiousIntent
from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch
from ovos_spec_tools import closest_lang, expand as expand_template, standardize_lang
from ovos_spec_tools import SpecMessage
from ovos_spec_tools import gate_satisfied, context_slot_candidates
from ovos_utils import flatten_list
from ovos_utils.fakebus import FakeBus
from ovos_utils.list_utils import deduplicate_list
from ovos_utils.log import LOG, deprecated, log_deprecation
from ovos_utils.text_utils import remove_accents_and_punct
from ovos_utils.xdg_utils import xdg_data_home
import faulthandler

PadatiousIntentContainer = IntentContainer  # backwards compat

# for easy typing
PadatiousEngine = Union[Type[IntentContainer], Type[DomainIntentContainer]]


# OVOS-INTENT-1: a template slot is written ``{entity_name}`` in a sample; the
# padaos parser recognises the same lowercase/underscore/colon name form.
_SLOT_RE = re.compile(r"{([a-z_:]+)}")



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
    # Normalize case: OVOS-INTENT-1 §2 normalizes input to lowercase for matching,
    # so training samples and extracted slot values stay case-insensitive
    # (ovos_spec_tools.expand preserves case; the prior expander lowercased).
    utterances = [u.lower() for u in utterances]
    # Collapse multiple whitespaces into a single space
    utterances = [re.sub(r'\s+', ' ', u) for u in utterances]
    # Replace accented characters and punctuation if needed
    if cast_to_ascii:
        utterances = [remove_accents_and_punct(u) for u in utterances]
    # strip trailing punctuation, that just causes duplicate training data —
    # but preserve the slot/vocabulary metacharacters {} <> so a template
    # ending in a slot ({name}) keeps its closing brace (OVOS-INTENT-1 §3)
    _trailing_punct = ''.join(c for c in string.punctuation if c not in '{}<>')
    utterances = [u.rstrip(_trailing_punct) for u in utterances]
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
        lang2 = closest_lang(lang, list(self.LANGS))
        if lang2 is None:
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
        return closest_lang(lang, list(cls.LANGS)) is not None

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


class PadatiousPipeline(ConfidenceMatcherPipeline):
    """Service class for padatious intent matching."""

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None,
                 engine_class: Optional[PadatiousEngine] = None):
        intent_config = Configuration().get('intents', {})
        config = config or intent_config.get("ovos-padatious-pipeline-plugin") or intent_config.get("padatious") or dict()
        super().__init__(bus, config)
        try:
            faulthandler.enable()  # Enables crash logging
        except Exception:
            pass # happens in unittests and such
        self.lock = RLock()
        core_config = Configuration()
        self.lang = standardize_lang(core_config.get("lang", "en-US"))
        langs = core_config.get('secondary_langs') or []
        langs = [standardize_lang(l) for l in langs]
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

        # OVOS-INTENT-4 §8.5 enable/disable: padatious has no native
        # suppression flag, so disable detaches the intent and enable
        # re-registers it. _intent_definitions retains the register Message
        # of every registered intent (full name -> Message); _disabled_intents
        # holds the subset currently suppressed.
        self._intent_definitions = {}
        self._disabled_intents = {}

        # OVOS-CONTEXT-1 §6/§6.1 requires_context / excludes_context gating.
        # Registration MAY carry these declarations; they are stored per
        # registered intent (keyed by the internal ``<skill_id>:<name>``) and
        # evaluated at match time via the shared ``gate_satisfied`` helper.
        # Retained across the disable/enable lifecycle (mirrors
        # _intent_definitions); dropped only on deregister.
        self._intent_context_gates = {}

        # OVOS-CONTEXT-1 §7 uniform slot fill: the declared template slots of
        # each registered intent, keyed by the internal ``<skill_id>:<name>``.
        # Any declared slot the utterance leaves unresolved is filled from a
        # live ``session.intent_context`` entry, independent of requires_context.
        self._intent_slots = {}

        # INTENT-2 §4.3 per-slot value blacklist: ``{slot: [values]}`` carried
        # in the registration payload. A slot the utterance binds to a
        # blacklisted value (whole-word-sequence) is treated as UNRESOLVED so
        # the §7 context candidate fills it. Anaphoric pronouns are supplied
        # here as a locale resource rather than hardcoded.
        self._intent_slot_blacklists = {}

        # legacy registration contract (kept for back-compat)
        self.bus.on('padatious:register_intent', self.register_intent)
        self.bus.on('padatious:register_entity', self.register_entity)
        self.bus.on('detach_intent', self.handle_detach_intent)
        self.bus.on('detach_skill', self.handle_detach_skill)
        self.bus.on('intent.service.padatious.get', self.handle_get_padatious)
        self.bus.on('intent.service.padatious.manifest.get', self.handle_padatious_manifest)
        self.bus.on('intent.service.padatious.entities.manifest.get', self.handle_entity_manifest)
        self.bus.on('mycroft.skills.train', self.train)

        # OVOS-INTENT-4 spec registration contract (in addition to legacy).
        # Padatious is a TEMPLATE engine, so register.template is its primary
        # consumed topic; keyword registrations are ignored by design (§11).
        self.bus.on(SpecMessage.INTENT_REGISTER_TEMPLATE, self.handle_register_template)
        self.bus.on(SpecMessage.ENTITY_REGISTER, self.handle_register_entity_spec)
        self.bus.on(SpecMessage.INTENT_DEREGISTER, self.handle_deregister_intent_spec)
        self.bus.on(SpecMessage.ENTITY_DEREGISTER, self.handle_deregister_entity_spec)
        self.bus.on(SpecMessage.SKILL_DEREGISTER, self.handle_deregister_skill_spec)
        self.bus.on(SpecMessage.INTENT_ENABLE, self.handle_enable_intent_spec)
        self.bus.on(SpecMessage.INTENT_DISABLE, self.handle_disable_intent_spec)

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
        lang = standardize_lang(lang or self.lang)

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

        # Training changes the model; stale LRU cache entries must be evicted
        # so that the next call to calc_intent reflects the updated state.
        _calc_padatious_intent.cache_clear()

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
        # Intent roster changed; evict stale cache so next match reflects removal.
        _calc_padatious_intent.cache_clear()
        # In instant_train mode, retrain immediately so the model also
        # forgets the intent — otherwise the cleared cache repopulates from
        # the still-trained model on the next match.
        if self.config.get("instant_train", False):
            self.train(message)

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
        # Intent roster changed; evict stale cache so next match reflects removal.
        _calc_padatious_intent.cache_clear()
        # See handle_detach_intent — retrain in instant_train mode so the
        # underlying model state matches the registered_intents list.
        if self.config.get("instant_train", False):
            self.train(message)

    def _unpack_object(self, message):
        """convert message to training data"""
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning("Skill ID is missing. Registering under 'anonymous_skill'")
            skill_id = "anonymous_skill"
        file_name = message.data.get('file_name')
        samples = message.data.get("samples")
        name = message.data['name']
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang(lang)
        blacklisted_words = message.data.get('blacklisted_words', [])
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
        return lang, skill_id, name, samples, blacklisted_words

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
        # retain the registration so an INTENT-4 enable (§8.5) can re-train
        # the intent after a disable detached it
        self._intent_definitions[message.data['name']] = message

        # OVOS-CONTEXT-1 §6: retain any requires/excludes gating declarations
        # keyed by the internal intent name. Only stored when present so
        # intents without a gate keep unchanged (ungated) behavior.
        requires = message.data.get("requires_context")
        excludes = message.data.get("excludes_context")
        if requires or excludes:
            self._intent_context_gates[message.data['name']] = (requires, excludes)

        # OVOS-CONTEXT-1 §7: record the declared template slots so an
        # unresolved slot can be filled from context at match time.
        slots = set()
        for sample in message.data.get('samples', []):
            slots.update(_SLOT_RE.findall(sample))
        if slots:
            self._intent_slots[message.data['name']] = frozenset(slots)

        # INTENT-2 §4.3: a per-slot value blacklist rides in the payload keyed
        # by slot name. Accept ``slot_blacklist`` or a dict-valued ``blacklist``
        # (a list-valued ``blacklist`` is the template-method suppression
        # vocabulary and is left untouched).
        slot_blacklist = message.data.get('slot_blacklist')
        if slot_blacklist is None and isinstance(message.data.get('blacklist'), dict):
            slot_blacklist = message.data.get('blacklist')
        if slot_blacklist:
            self._intent_slot_blacklists[message.data['name']] = {
                slot: [str(v) for v in values]
                for slot, values in slot_blacklist.items()}

        lang = message.data.get('lang', self.lang)
        lang = standardize_lang(lang)
        if lang in self.containers:
            self.registered_intents.append(message.data['name'])
            LOG.debug('Registering Padatious intent: ' + message.data['name'])
            lang, skill_id, name, samples, blacklisted_words = self._unpack_object(message)
            if self.engine_class == DomainIntentContainer:
                self.containers[lang].add_domain_intent(skill_id, name, samples, blacklisted_words)
            else:
                self.containers[lang].add_intent(name, samples, blacklisted_words)

        if self.config.get("instant_train", False) or self.first_train.is_set():
            self.train(message)

    def register_entity(self, message):
        """Messagebus handler for registering entities.

        Args:
            message (Message): message triggering action
        """
        lang = message.data.get('lang', self.lang)
        lang = standardize_lang(lang)
        if lang in self.containers:
            self.registered_entities.append(message.data)
            lang, skill_id, name, samples, _ = self._unpack_object(message)
            LOG.debug('Registering Padatious entity: ' + message.data['name'])
            if self.engine_class == DomainIntentContainer:
                self.containers[lang].add_domain_entity(skill_id, name, samples)
            else:
                self.containers[lang].add_entity(name, samples)

    # ------------------------------------------------------------------ #
    # OVOS-INTENT-4 spec registration handlers                           #
    #                                                                    #
    # These translate the spec payloads (§§6-8) into the same internal   #
    # padatious registration calls the legacy handlers use, so both wire #
    # contracts feed one container. The internal padatious intent/entity #
    # name is the colon-joined ``<skill_id>:<name>`` the legacy contract #
    # already used as ``data['name']``.                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spec_identity(message, name_field):
        """Pull (skill_id, name) from a §3.2 spec payload.

        Returns (skill_id, name, full_name) where ``full_name`` is the
        ``<skill_id>:<name>`` key padatious uses internally, or
        (None, None, None) when identity is missing.
        """
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        name = message.data.get(name_field)
        if not skill_id or not name:
            return None, None, None
        # already-namespaced names are passed through unchanged
        full = name if name.startswith(f"{skill_id}:") else f"{skill_id}:{name}"
        return skill_id, name, full

    def handle_register_template(self, message):
        """Consume ``ovos.intent.register.template`` (OVOS-INTENT-4 §6).

        Maps the spec payload (skill_id, intent_name, lang, samples,
        blacklist) onto the legacy padatious registration via
        :meth:`register_intent`.
        """
        skill_id, intent_name, full = self._spec_identity(message, "intent_name")
        if full is None:
            LOG.warning(f"[{SpecMessage.INTENT_REGISTER_TEMPLATE}] rejected: "
                        f"missing skill_id/intent_name")
            return
        samples = message.data.get("samples")
        if not samples:  # §6.3 malformed: samples missing/empty
            LOG.warning(f"[{SpecMessage.INTENT_REGISTER_TEMPLATE}] rejected "
                        f"skill_id={skill_id} intent_name={intent_name} "
                        f"lang={message.data.get('lang')}: empty samples")
            return
        lang = standardize_lang(message.data.get("lang", self.lang))
        # §6 'blacklist' is the template-method suppression vocabulary;
        # padatious calls this 'blacklisted_words'.
        legacy = Message(
            "padatious:register_intent",
            data={"name": full, "samples": list(samples), "lang": lang,
                  "skill_id": skill_id,
                  "blacklisted_words": message.data.get("blacklist", []),
                  # OVOS-CONTEXT-1 §6: forward the optional gating declarations
                  # onto the internal registration so they are stored per intent.
                  "requires_context": message.data.get("requires_context"),
                  "excludes_context": message.data.get("excludes_context"),
                  # INTENT-2 §4.3: per-slot value blacklist keyed by slot name.
                  "slot_blacklist": message.data.get("slot_blacklist")},
            context=dict(message.context, skill_id=skill_id))
        self.register_intent(legacy)

    def handle_register_entity_spec(self, message):
        """Consume ``ovos.entity.register`` (OVOS-INTENT-4 §7)."""
        skill_id, entity_name, full = self._spec_identity(message, "entity_name")
        if full is None:
            LOG.warning(f"[{SpecMessage.ENTITY_REGISTER}] rejected: "
                        f"missing skill_id/entity_name")
            return
        samples = message.data.get("samples")
        if not samples:  # §7.2 malformed: samples missing/empty
            LOG.warning(f"[{SpecMessage.ENTITY_REGISTER}] rejected "
                        f"skill_id={skill_id} entity_name={entity_name} "
                        f"lang={message.data.get('lang')}: empty samples")
            return
        lang = standardize_lang(message.data.get("lang", self.lang))
        legacy = Message(
            "padatious:register_entity",
            data={"name": full, "samples": list(samples), "lang": lang,
                  "skill_id": skill_id},
            context=dict(message.context, skill_id=skill_id))
        self.register_entity(legacy)

    def _spec_intent_names(self, message):
        """Resolve the full padatious intent name(s) targeted by a §8 payload.

        Returns the list of ``<skill_id>:<intent_name>`` keys to act on
        (``lang`` is ignored: padatious keys intents by name, training data
        is shared across the per-lang containers).
        """
        skill_id, intent_name, full = self._spec_identity(message, "intent_name")
        if full is None:
            return []
        return [full]

    def handle_deregister_intent_spec(self, message):
        """Consume ``ovos.intent.deregister`` (OVOS-INTENT-4 §8.2)."""
        for full in self._spec_intent_names(message):
            self.__detach_intent(full)
            self._disabled_intents.pop(full, None)
            self._intent_context_gates.pop(full, None)
            self._intent_slots.pop(full, None)
            self._intent_slot_blacklists.pop(full, None)
        _calc_padatious_intent.cache_clear()
        if self.config.get("instant_train", False):
            self.train(message)

    def handle_deregister_entity_spec(self, message):
        """Consume ``ovos.entity.deregister`` (OVOS-INTENT-4 §8.3)."""
        skill_id, entity_name, full = self._spec_identity(message, "entity_name")
        if full is None:
            return
        for lang in self.containers:
            try:
                self.containers[lang].remove_entity(full)
            except Exception as e:
                LOG.debug(f"entity {full} not present in {lang}: {e}")
        self.registered_entities = [e for e in self.registered_entities
                                    if e.get("name") != full]
        _calc_padatious_intent.cache_clear()
        if self.config.get("instant_train", False):
            self.train(message)

    def handle_deregister_skill_spec(self, message):
        """Consume ``ovos.skill.deregister`` (OVOS-INTENT-4 §8.4).

        Removes every intent and entity owned by the skill.
        """
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if not skill_id:
            LOG.warning(f"[{SpecMessage.SKILL_DEREGISTER}] rejected: missing skill_id")
            return
        for full in list(self._skill2intent.get(skill_id, [])):
            self.__detach_intent(full)
            self._disabled_intents.pop(full, None)
            self._intent_context_gates.pop(full, None)
            self._intent_slots.pop(full, None)
            self._intent_slot_blacklists.pop(full, None)
        # drop the skill's entities too
        prefix = f"{skill_id}:"
        for lang in self.containers:
            for ent in [e.get("name") for e in self.registered_entities
                        if str(e.get("name", "")).startswith(prefix)]:
                try:
                    self.containers[lang].remove_entity(ent)
                except Exception as e:
                    LOG.debug(f"entity {ent} not present in {lang}: {e}")
        self.registered_entities = [e for e in self.registered_entities
                                    if not str(e.get("name", "")).startswith(prefix)]
        _calc_padatious_intent.cache_clear()
        if self.config.get("instant_train", False):
            self.train(message)

    def handle_disable_intent_spec(self, message):
        """Consume ``ovos.intent.disable`` (OVOS-INTENT-4 §8.5).

        Padatious has no native suppression flag; disabling detaches the
        intent from the container while retaining its definition so a later
        enable can re-train it.
        """
        for full in self._spec_intent_names(message):
            if full in self._disabled_intents:
                continue  # already disabled, no-op
            definition = self._intent_definitions.get(full)
            if definition is None:
                LOG.warning(f"[{SpecMessage.INTENT_DISABLE}] no registered "
                            f"definition for {full}; nothing to disable")
                continue
            self._disabled_intents[full] = definition
            self.__detach_intent(full)
        _calc_padatious_intent.cache_clear()
        if self.config.get("instant_train", False):
            self.train(message)

    def handle_enable_intent_spec(self, message):
        """Consume ``ovos.intent.enable`` (OVOS-INTENT-4 §8.5).

        Re-registers a previously disabled intent from its retained
        definition.
        """
        for full in self._spec_intent_names(message):
            definition = self._disabled_intents.pop(full, None)
            if definition is None:
                continue  # already enabled / never disabled -> no-op
            self.register_intent(definition)

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
        # Session is unhashable under ovos-bus-client 2.x, so it cannot be an
        # lru_cache key; pass the blacklists it carries as frozensets instead.
        blacklisted_intents = frozenset(sess.blacklisted_intents or [])
        blacklisted_skills = frozenset(sess.blacklisted_skills or [])

        intent_container = self.containers.get(lang)
        intents = [_calc_padatious_intent(utt, intent_container,
                                          blacklisted_intents, blacklisted_skills)
                   for utt in utterances]
        intents = [i for i in intents if i is not None]
        # OVOS-CONTEXT-1 §6/§6.1: drop any candidate whose requires/excludes
        # gating is not satisfied against the session's intent_context. The
        # shared helper handles liveness/scope/decay; owner_id is the intent's
        # skill_id (the private-scope default owner). Ungated intents pass.
        if intents and self._intent_context_gates:
            intent_context = getattr(sess, "intent_context", None) or {}
            kept = []
            for i in intents:
                gate = self._intent_context_gates.get(i.name)
                if gate is not None:
                    requires, excludes = gate
                    owner_id = i.name.split(":")[0]
                    if not gate_satisfied(intent_context, requires, excludes,
                                          owner_id=owner_id):
                        LOG.debug(f"Padatious intent '{i.name}' dropped: "
                                  f"OVOS-CONTEXT-1 gating not satisfied")
                        continue
                kept.append(i)
            intents = kept
        # select best
        if intents:
            best = max(intents, key=lambda k: k.conf)
            self._fill_context_slots(best, sess)
            return best

    @staticmethod
    def _word_seq_in(needle: str, haystack: str) -> bool:
        """True when ``needle`` occurs in ``haystack`` as a whole-word sequence.

        Comparison is case-insensitive and whitespace-collapsed; the needle's
        tokens must appear as a contiguous run of whole words in the haystack,
        so ``he`` matches ``he`` but not ``the`` or ``header``.
        """
        n = str(needle).lower().split()
        h = str(haystack).lower().split()
        if not n or len(n) > len(h):
            return False
        for i in range(len(h) - len(n) + 1):
            if h[i:i + len(n)] == n:
                return True
        return False

    def _fill_context_slots(self, intent: PadatiousIntent, sess: Session) -> None:
        """OVOS-CONTEXT-1 §7 — uniform context slot fill.

        For EVERY declared template slot of the matched intent, if a live
        non-null ``session.intent_context`` entry exists (private
        ``<skill_id>:name`` precedence over shared bare ``name``), fill the
        slot when the utterance left it unresolved. This is independent of
        requires_context, which gates only the presence flags.

        INTENT-2 §4.3: before the fill, a slot the utterance bound to a value
        listed in that slot's blacklist (e.g. an anaphoric pronoun) is dropped
        so it counts as unresolved and the context candidate takes over.
        """
        slot_names = self._intent_slots.get(intent.name)
        if not slot_names:
            return
        matches = dict(intent.matches or {})

        # INTENT-2 §4.3: unresolve blacklisted slot values.
        for slot, values in self._intent_slot_blacklists.get(intent.name, {}).items():
            bound = matches.get(slot)
            if bound is not None and any(self._word_seq_in(v, bound) for v in values):
                LOG.debug(f"Padatious slot '{slot}'='{bound}' blacklisted "
                          f"(INTENT-2 §4.3): treating as unresolved")
                matches.pop(slot, None)

        intent_context = getattr(sess, "intent_context", None) or {}
        owner_id = intent.name.split(":")[0]
        candidates = context_slot_candidates(intent_context, list(slot_names),
                                             owner_id)
        for slot, value in candidates.items():
            # a value the utterance itself produced wins over the candidate
            if not matches.get(slot):
                LOG.debug(f"Padatious slot '{slot}' filled from context "
                          f"(OVOS-CONTEXT-1 §7): '{value}'")
                matches[slot] = value
        intent.matches = matches

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        if self.containers:
            return closest_lang(standardize_lang(lang), list(self.containers.keys()))
        return None

    def shutdown(self):
        self.bus.remove('padatious:register_intent', self.register_intent)
        self.bus.remove('padatious:register_entity', self.register_entity)
        self.bus.remove('intent.service.padatious.get', self.handle_get_padatious)
        self.bus.remove('intent.service.padatious.manifest.get', self.handle_padatious_manifest)
        self.bus.remove('intent.service.padatious.entities.manifest.get', self.handle_entity_manifest)
        self.bus.remove('detach_intent', self.handle_detach_intent)
        self.bus.remove('detach_skill', self.handle_detach_skill)
        self.bus.remove(SpecMessage.INTENT_REGISTER_TEMPLATE, self.handle_register_template)
        self.bus.remove(SpecMessage.ENTITY_REGISTER, self.handle_register_entity_spec)
        self.bus.remove(SpecMessage.INTENT_DEREGISTER, self.handle_deregister_intent_spec)
        self.bus.remove(SpecMessage.ENTITY_DEREGISTER, self.handle_deregister_entity_spec)
        self.bus.remove(SpecMessage.SKILL_DEREGISTER, self.handle_deregister_skill_spec)
        self.bus.remove(SpecMessage.INTENT_ENABLE, self.handle_enable_intent_spec)
        self.bus.remove(SpecMessage.INTENT_DISABLE, self.handle_disable_intent_spec)

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
                           blacklisted_intents: frozenset = frozenset(),
                           blacklisted_skills: frozenset = frozenset()) -> Optional[PadatiousIntent]:
    """
    Try to match an utterance to an intent in an intent_container
    @param utt: str - text to match intent against

    The session blacklists are passed as hashable frozensets so this stays
    ``lru_cache``-able (Session is unhashable under ovos-bus-client>=2.4.0a1).
    @return: matched PadatiousIntent
    """
    try:
        # OVOS-INTENT-1 §2: match against the lowercase-normalized input so slot
        # values are case-insensitive, but report the original utterance as `sent`.
        matches = [m for m in intent_container.calc_intents(utt.lower())
                   if m.name not in blacklisted_intents
                   and m.name.split(":")[0] not in blacklisted_skills]
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
