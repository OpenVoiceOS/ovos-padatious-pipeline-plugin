from collections import defaultdict
from typing import Dict, List, Optional

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.match_data import MatchData


class DomainIntentEngine:
    """
    A domain-aware intent recognition engine that organizes intents and entities
    into specific domains, providing flexible and hierarchical intent matching.
    """

    def __init__(self):
        """
        Initialize the DomainIntentEngine.

        Attributes:
            domain_engine (IntentContainer): A top-level intent container for cross-domain calculations.
            domains (Dict[str, IntentContainer]): A mapping of domain names to their respective intent containers.
            training_data (Dict[str, List[str]]): A mapping of domain names to their associated training samples.
        """
        self.domain_engine = IntentContainer()
        self.domains: Dict[str, IntentContainer] = defaultdict(IntentContainer)
        self.training_data: Dict[str, List[str]] = defaultdict(list)
        self._must_train = True

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

    def register_domain_intent(self, domain_name: str, intent_name: str, intent_samples: List[str]):
        """
        Register an intent within a specific domain.

        Args:
            domain_name (str): The name of the domain.
            intent_name (str): The name of the intent to register.
            intent_samples (List[str]): A list of sample sentences for the intent.
        """
        self.domains[domain_name].add_intent(intent_name, intent_samples)
        self.training_data[domain_name] += intent_samples
        self._must_train = True

    def remove_domain_intent(self, domain_name: str, intent_name: str):
        """
        Remove a specific intent from a domain.

        Args:
            domain_name (str): The name of the domain.
            intent_name (str): The name of the intent to remove.
        """
        self.domains[domain_name].remove_intent(intent_name)

    def register_domain_entity(self, domain_name: str, entity_name: str, entity_samples: List[str]):
        """
        Register an entity within a specific domain.

        Args:
            domain_name (str): The name of the domain.
            entity_name (str): The name of the entity to register.
            entity_samples (List[str]): A list of sample phrases for the entity.
        """
        self.domains[domain_name].add_entity(entity_name, entity_samples)

    def remove_domain_entity(self, domain_name: str, entity_name: str):
        """
        Remove a specific entity from a domain.

        Args:
            domain_name (str): The name of the domain.
            entity_name (str): The name of the entity to remove.
        """
        self.domains[domain_name].remove_entity(entity_name)

    def calc_domains(self, query: str) -> List[MatchData]:
        """
        Calculate the matching domains for a query.

        Args:
            query (str): The input query.

        Returns:
            List[MatchData]: A list of MatchData objects representing matching domains.
        """
        if self._must_train:
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
        if self._must_train:
            self.train()
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str, domain: Optional[str] = None) -> MatchData:
        """
        Calculate the best matching intent for a query within a specific domain.

        Args:
            query (str): The input query.
            domain (Optional[str]): The domain to limit the search to. Defaults to None.

        Returns:
            MatchData: The best matching intent.
        """
        if self._must_train:
            self.train()
        domain: str = domain or self.domain_engine.calc_intent(query).name
        if domain in self.domains:
            return self.domains[domain].calc_intent(query)
        return MatchData(name=None, sent=query, matches=None, conf=0.0)

    def calc_intents(self, query: str, domain: Optional[str] = None, top_k_domains: int = 2) -> List[MatchData]:
        """
        Calculate matching intents for a query across domains or within a specific domain.

        Args:
            query (str): The input query.
            domain (Optional[str]): The specific domain to search in. If None, searches across top-k domains.
            top_k_domains (int): The number of top domains to consider. Defaults to 2.

        Returns:
            List[MatchData]: A list of MatchData objects representing matching intents, sorted by confidence.
        """
        if self._must_train:
            self.train()
        if domain:
            return self.domains[domain].calc_intents(query)
        matches = []
        domains = self.calc_domains(query)[:top_k_domains]
        for domain in domains:
            if domain.name in self.domains:
                matches += self.domains[domain.name].calc_intents(query)
        return sorted(matches, reverse=True, key=lambda k: k.conf)

    def train(self):
        for domain, samples in self.training_data.items():
            self.domain_engine.add_intent(domain, samples)
        self.domain_engine.train()
        for domain in self.domains:
            self.domains[domain].train()
        self._must_train = False