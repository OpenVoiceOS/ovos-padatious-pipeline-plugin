"""Hierarchical intent container — two-stage domain routing.

Intents are grouped into domains. A top-level classifier first selects a
single domain for the query; only that domain's :class:`IntentContainer`
is then scored to resolve the intent. The top-level classifier is itself
an :class:`IntentContainer` trained with each domain name as a label over
the union of that domain's intent samples.
"""
from collections import defaultdict
from typing import Dict, List, Optional

from ovos_utils.log import LOG

from ovos_padatious.intent_container import IntentContainer
from ovos_padatious.match_data import MatchData


class HierarchicalIntentContainer:
    """Two-stage intent engine: domain classification followed by intent matching.

    Intents are grouped into *domains*. At query time the engine first selects
    the most likely domain, then runs that domain's intent container to find the
    best intent within it.

    The top-level domain classifier is trained automatically: every sample
    passed to :meth:`register_domain_intent` is also fed to :attr:`domain_engine`
    under its domain name, so the container works standalone with no manual
    classifier setup.

    Domains can also be selected explicitly, bypassing the top-level classifier.

    Args:
        cache_dir: Directory for caching neural network models, forwarded to
            every :class:`IntentContainer` created internally. The top-level
            classifier uses a ``_domains`` sub-directory.
        disable_padaos: Disable the padaos regex layer on every child container.
        domain_threshold: Minimum confidence the top-level classifier must
            reach for a query to be routed at all. When the best domain scores
            below this, :meth:`calc_intent` returns a no-match instead of
            resolving an intent — this is the off-topic rejection gate.
            ``0.0`` (default) disables the gate; every query is routed to its
            best domain.
    """

    def __init__(self, cache_dir: Optional[str] = None,
                 disable_padaos: bool = False,
                 domain_threshold: float = 0.0) -> None:
        self.cache_dir = cache_dir
        self.disable_padaos = disable_padaos
        self.domain_threshold = domain_threshold
        #: Top-level classifier that maps free-text queries to a domain name.
        self.domain_engine: IntentContainer = IntentContainer(
            cache_dir=f"{cache_dir}/_domains" if cache_dir else None,
            disable_padaos=disable_padaos,
        )
        #: Per-domain intent containers, keyed by domain name.
        self.domains: Dict[str, IntentContainer] = {}
        #: Raw training samples accumulated per domain.
        self.training_data: Dict[str, List[str]] = defaultdict(list)
        #: Domains whose classifier entry is stale and must be rebuilt.
        self._dirty_domains: set = set()
        self.must_train = True

    def instantiate_from_disk(self) -> None:
        """Compatibility hook.

        Domain membership is not known until intents are registered, so
        cached models are loaded lazily per domain by :meth:`_get_container`.
        """

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

    def _sync_domain_classifier(self) -> None:
        """Rebuild stale classifier entries.

        Registration only marks a domain dirty; the top-level classifier is
        rebuilt here, lazily, the first time a query needs it. This keeps bulk
        registration linear instead of re-expanding the whole corpus per call.
        """
        for domain_name in self._dirty_domains:
            if domain_name in self.domain_engine.intent_names:
                self.domain_engine.remove_intent(domain_name)
            samples = self.training_data.get(domain_name)
            if samples:
                self.domain_engine.add_intent(domain_name, samples)
        self._dirty_domains.clear()

    # ── domain management ──────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents, entities, and training data."""
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        self._dirty_domains.discard(domain_name)
        if domain_name in self.domain_engine.intent_names:
            self.domain_engine.remove_intent(domain_name)
        self.must_train = True

    # ── intent management ──────────────────────────────────────────────

    def add_domain_intent(self, domain_name: str, intent_name: str,
                          intent_samples: List[str],
                          blacklisted_words: Optional[List[str]] = None) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`IntentContainer` on first use. The
        top-level domain classifier is marked stale and rebuilt lazily on
        the next query or :meth:`train` call.
        """
        self._get_container(domain_name).add_intent(
            intent_name, intent_samples,
            blacklisted_words=blacklisted_words,
        )
        self.training_data[domain_name] += intent_samples
        self._dirty_domains.add(domain_name)
        self.must_train = True

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove a specific intent from a domain."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)
            self.must_train = True

    # ── entity management ──────────────────────────────────────────────

    def add_domain_entity(self, domain_name: str, entity_name: str,
                          entity_samples: List[str]) -> None:
        """Register an entity inside a domain."""
        self._get_container(domain_name).add_entity(entity_name, entity_samples)
        self.must_train = True

    def remove_domain_entity(self, domain_name: str, entity_name: str) -> None:
        """Remove a specific entity from a domain."""
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name)
            self.must_train = True

    # ── matching ───────────────────────────────────────────────────────

    def calc_domain(self, query: str) -> MatchData:
        """Classify *query* into the best-matching domain.

        Returns a :class:`MatchData` whose ``name`` is the predicted domain.
        """
        if self.must_train:
            self.train()
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str,
                    domain: Optional[str] = None) -> MatchData:
        """Return the best-matching intent for *query*, optionally within *domain*.

        If *domain* is ``None``, the domain is inferred by :meth:`calc_domain`.
        When the inferred domain scores below :attr:`domain_threshold`, or the
        inferred/supplied domain has no registered intents, a no-match result
        is returned. Passing *domain* explicitly bypasses the classifier and
        the threshold gate.
        """
        if self.must_train:
            self.train()
        no_match = MatchData(name=None, sent=query, matches=None, conf=0.0)

        resolved_domain: Optional[str] = domain
        if resolved_domain is None:
            dom_result = self.domain_engine.calc_intent(query)
            if dom_result.conf < self.domain_threshold:
                return no_match
            resolved_domain = dom_result.name

        if resolved_domain in self.domains:
            return self.domains[resolved_domain].calc_intent(query)
        return no_match

    def calc_intents(self, query: str,
                     domain: Optional[str] = None) -> List[MatchData]:
        """Return matches from the routed (or supplied) domain only.

        Two-stage routing picks a single domain; the returned list contains
        that domain's candidate matches. An empty list signals no match.
        """
        if self.must_train:
            self.train()
        resolved_domain: Optional[str] = domain
        if resolved_domain is None:
            dom_result = self.domain_engine.calc_intent(query)
            if dom_result.conf < self.domain_threshold:
                return []
            resolved_domain = dom_result.name

        if resolved_domain in self.domains:
            return self.domains[resolved_domain].calc_intents(query)
        return []

    def train(self) -> None:
        """Train the top-level classifier and every domain container."""
        self._sync_domain_classifier()
        LOG.debug("Training hierarchical domain classifier")
        self.domain_engine.train()
        for name, container in self.domains.items():
            LOG.debug(f"Training domain: {name}")
            container.train()
        self.must_train = False
