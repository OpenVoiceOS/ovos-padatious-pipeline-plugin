"""Domain-aware intent container.

Each domain owns an :class:`IntentContainer`. At query time every
container scores the utterance independently; the highest-confidence
intent across the union wins. There is no top-level router — parallel
scoring is cheap and each container's confidence is comparable.
"""
from typing import Dict, List, Optional

from ovos_utils.log import LOG

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.match_data import MatchData


class DomainIntentContainer:
    """Per-domain :class:`IntentContainer` with parallel argmax over domains."""

    def __init__(self, cache_dir: Optional[str] = None,
                 disable_padaos: bool = False):
        self.cache_dir = cache_dir
        self.disable_padaos = disable_padaos
        self.domains: Dict[str, IntentContainer] = {}
        self.must_train = True

    # ── domain management ──────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        self.domains.pop(domain_name, None)

    # ── intent management ──────────────────────────────────────────────

    def add_domain_intent(self, domain_name: str, intent_name: str,
                          intent_samples: List[str],
                          blacklisted_words: Optional[List[str]] = None) -> None:
        self._get_container(domain_name).add_intent(
            intent_name, intent_samples,
            blacklisted_words=blacklisted_words,
        )
        self.must_train = True

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)

    # ── entity management ──────────────────────────────────────────────

    def add_domain_entity(self, domain_name: str, entity_name: str,
                          entity_samples: List[str]) -> None:
        self._get_container(domain_name).add_entity(entity_name, entity_samples)

    def remove_domain_entity(self, domain_name: str, entity_name: str) -> None:
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name)

    # ── matching ───────────────────────────────────────────────────────

    def calc_intent(self, query: str,
                     domain: Optional[str] = None) -> MatchData:
        """Return the best intent match across all domains.

        Each domain's :meth:`IntentContainer.calc_intent` returns its
        top-1 match; the global argmax over those candidates wins.
        ``domain`` restricts scoring to that domain.
        """
        if self.must_train:
            self.train()
        if domain is not None:
            container = self.domains.get(domain)
            return (container.calc_intent(query) if container
                    else MatchData(name=None, sent=query, matches=None, conf=0.0))
        best = MatchData(name=None, sent=query, matches=None, conf=0.0)
        for container in self.domains.values():
            match = container.calc_intent(query)
            if match.conf > best.conf:
                best = match
        return best

    def calc_intents(self, query: str,
                      domain: Optional[str] = None) -> List[MatchData]:
        """Per-domain best matches, sorted by confidence."""
        if self.must_train:
            self.train()
        if domain is not None:
            container = self.domains.get(domain)
            return container.calc_intents(query) if container else []
        candidates = [c.calc_intent(query) for c in self.domains.values()]
        return sorted(candidates, key=lambda m: m.conf, reverse=True)

    def train(self) -> None:
        for name, container in self.domains.items():
            LOG.debug(f"Training domain: {name}")
            container.train()
        self.must_train = False

    # ── internal ───────────────────────────────────────────────────────

    def _get_container(self, domain_name: str) -> IntentContainer:
        if domain_name not in self.domains:
            container = IntentContainer(
                cache_dir=self.cache_dir,
                disable_padaos=self.disable_padaos,
            )
            container.instantiate_from_disk()
            self.domains[domain_name] = container
        return self.domains[domain_name]
