from collections import defaultdict
from typing import Dict, List, Optional
from ovos_utils.log import LOG
from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.match_data import MatchData


class DomainIntentContainer:
    """
    A domain-aware intent recognition engine that organizes intents and entities
    into specific domains, providing flexible and hierarchical intent matching.
    """

    def __init__(self, cache_dir: Optional[str] = None, disable_padaos: bool = False):
        """
        Initialize the DomainIntentEngine.

        Attributes:
            domain_engine (IntentContainer): A top-level intent container for cross-domain calculations.
            domains (Dict[str, IntentContainer]): A mapping of domain names to their respective intent containers.
            training_data (Dict[str, List[str]]): A mapping of domain names to their associated training samples.
        """
        self.cache_dir = cache_dir
        self.disable_padaos = disable_padaos
        self.domain_engine = IntentContainer(cache_dir=cache_dir,
                                             disable_padaos=disable_padaos)
        self.domains: Dict[str, IntentContainer] = {}
        self.training_data: Dict[str, List[str]] = defaultdict(list)
        self.instantiate_from_disk()
        self.must_train = True

    def instantiate_from_disk(self) -> None:
        """
        Instantiates the necessary (internal) data structures when loading persisted model from disk.
        This is done via injecting entities and intents back from cached file versions.
        """
        self.domain_engine.instantiate_from_disk()
        for engine in self.domains.values():
            engine.instantiate_from_disk()

    def remove_domain(self, domain_name: str):
        """
        Remove a domain and its associated intents and training data.

        Args:
            domain_name (str): The name of the domain to remove.
        """
        if domain_name in self.training_data:
            self.training_data.pop(domain_name)
        if domain_name in self.domains:
            self.domains.pop(domain_name)
        if domain_name in self.domain_engine.intent_names:
            self.domain_engine.remove_intent(domain_name)

    def add_domain_intent(self, domain_name: str, intent_name: str, intent_samples: List[str],
                          blacklisted_words: Optional[List[str]] = None):
        """
        Register an intent within a specific domain.

        Args:
            domain_name (str): The name of the domain.
            intent_name (str): The name of the intent to register.
            intent_samples (List[str]): A list of sample sentences for the intent.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(cache_dir=self.cache_dir,
                                                        disable_padaos=self.disable_padaos)
            self.domains[domain_name].instantiate_from_disk()

        self.domains[domain_name].add_intent(intent_name, intent_samples,
                                             blacklisted_words=blacklisted_words)
        self.training_data[domain_name] += intent_samples
        self.must_train = True

    def remove_domain_intent(self, domain_name: str, intent_name: str):
        """
        Remove a specific intent from a domain.

        Args:
            domain_name (str): The name of the domain.
            intent_name (str): The name of the intent to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)

    def add_domain_entity(self, domain_name: str, entity_name: str, entity_samples: List[str]):
        """
        Register an entity within a specific domain.

        Args:
            domain_name (str): The name of the domain.
            entity_name (str): The name of the entity to register.
            entity_samples (List[str]): A list of sample phrases for the entity.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(cache_dir=self.cache_dir,
                                                        disable_padaos=self.disable_padaos)
        self.domains[domain_name].add_entity(entity_name, entity_samples)

    def remove_domain_entity(self, domain_name: str, entity_name: str):
        """
        Remove a specific entity from a domain.

        Args:
            domain_name (str): The name of the domain.
            entity_name (str): The name of the entity to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name)

    def calc_domains(self, query: str) -> List[MatchData]:
        """
        Calculate the matching domains for a query.

        Args:
            query (str): The input query.

        Returns:
            List[MatchData]: A list of MatchData objects representing matching domains.
        """
        if self.must_train:
            self.train()

        return self.domain_engine.calc_intents(query)

    def calc_domain(self, query: str) -> MatchData:
        """
        Calculate the best matching domain for a query.

        Args:
            query (str): The input query.

        Returns:
            MatchData: The best matching domain.
        """
        if self.must_train:
            self.train()
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str, domain: Optional[str] = None) -> MatchData:
        """
        Calculate the best matching intent for a query.

        By default this follows the adapt-style parallel-argmax pattern:
        every sub-domain container scores the query and the single highest
        confidence intent wins globally. The legacy two-stage routing via
        the top-level ``domain_engine`` is no longer used by default — that
        engine remains trained for callers of :meth:`calc_domain`/
        :meth:`calc_domains` but does not gate intent selection.

        Args:
            query (str): The input query.
            domain (Optional[str]): If given, restrict matching to this domain.

        Returns:
            MatchData: The best matching intent across all domains (or
            within the given domain).
        """
        if self.must_train:
            self.train()
        if domain:
            if domain in self.domains:
                return self.domains[domain].calc_intent(query)
            return MatchData(name=None, sent=query, matches=None, conf=0.0)
        matches = self.calc_intents(query)
        if matches:
            return matches[0]
        return MatchData(name=None, sent=query, matches=None, conf=0.0)

    def calc_intents(self, query: str, domain: Optional[str] = None,
                     top_k_domains: Optional[int] = None) -> List[MatchData]:
        """
        Calculate matching intents for a query across domains.

        Default behaviour is to score the query against every sub-domain
        container in parallel and flatten the results (adapt-style
        parallel argmax). ``top_k_domains`` is kept as an opt-in
        optimisation hint: when set, the cheap domain-fingerprint engine
        is consulted first and only the union of the top-K domains'
        intents is scored.

        Args:
            query (str): The input query.
            domain (Optional[str]): The specific domain to search in.
                When given, ``top_k_domains`` is ignored.
            top_k_domains (Optional[int]): Optional optimisation. When
                set to a positive integer, restrict scoring to the top-K
                domains as ranked by the top-level domain engine.
                Defaults to None (score every domain).

        Returns:
            List[MatchData]: MatchData objects sorted by confidence (desc).
        """
        if self.must_train:
            self.train()
        if domain:
            if domain in self.domains:
                return self.domains[domain].calc_intents(query)
            return []

        if top_k_domains and top_k_domains > 0:
            ranked = self.calc_domains(query)[:top_k_domains]
            domain_names = [d.name for d in ranked if d.name in self.domains]
        else:
            domain_names = list(self.domains.keys())

        matches: List[MatchData] = []
        for name in domain_names:
            matches += self.domains[name].calc_intents(query)
        return sorted(matches, reverse=True, key=lambda k: k.conf)

    def train(self):
        for domain, samples in dict(self.training_data).items():  # copy for thread safety
            LOG.debug(f"Training domain: {domain}")
            self.domain_engine.add_intent(domain, samples)
        self.domain_engine.train()
        for domain in dict(self.domains): # copy for thread safety
            LOG.debug(f"Training domain sub-intents: {domain}")
            self.domains[domain].train()
        self.must_train = False
