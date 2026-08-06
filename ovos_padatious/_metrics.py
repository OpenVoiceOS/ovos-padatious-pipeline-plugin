"""Fixed-cardinality process-local Padatious counters."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import Any


class CounterMetric:
    """A minimal thread-safe cumulative counter."""

    def __init__(self, name: str) -> None:
        if not name.endswith("_total"):
            raise ValueError("counter metric names must end with '_total'")
        self.name = name
        self._value = 0
        self._lock = Lock()

    def increment(self) -> None:
        """Increment the counter once."""
        with self._lock:
            self._value += 1

    def snapshot(self) -> Mapping[str, Any]:
        """Return an immutable, JSON-friendly cumulative snapshot."""
        with self._lock:
            return {
                "name": self.name,
                "type": "counter",
                "value": self._value,
            }


CACHE_HIT = CounterMetric("ovos_padatious_cache_hit_total")
CACHE_MISS = CounterMetric("ovos_padatious_cache_miss_total")
EXACT_MATCH = CounterMetric("ovos_padatious_exact_match_total")
NEURAL_MATCH = CounterMetric("ovos_padatious_neural_match_total")


def performance_metrics() -> Mapping[str, Mapping[str, Any]]:
    """Return the process-local Padatious counters."""
    return {
        counter.name: counter.snapshot()
        for counter in (CACHE_HIT, CACHE_MISS, EXACT_MATCH, NEURAL_MATCH)
    }
