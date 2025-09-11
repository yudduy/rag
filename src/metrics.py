"""Prometheus metrics helpers for RAG system.

Provides counters and histograms for per-query observations.
"""

from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# Core metrics
queries_total = Counter(
    "rag_queries_total",
    "Total number of RAG queries processed",
)

cache_hits_total = Counter(
    "rag_cache_hits_total",
    "Total number of cache hits",
)

cache_misses_total = Counter(
    "rag_cache_misses_total",
    "Total number of cache misses",
)

query_latency_seconds = Histogram(
    "rag_query_latency_seconds",
    "RAG query latency in seconds",
    buckets=(
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
    ),
)

last_query_latency_ms = Gauge(
    "rag_last_query_latency_ms",
    "Latency of the last processed query in milliseconds",
)


def observe_query(latency_seconds: float, from_cache: bool) -> None:
    """Record metrics for a single query.

    Args:
        latency_seconds: query latency in seconds
        from_cache: whether the response came from cache
    """
    queries_total.inc()
    if from_cache:
        cache_hits_total.inc()
    else:
        cache_misses_total.inc()

    query_latency_seconds.observe(latency_seconds)
    last_query_latency_ms.set(latency_seconds * 1000.0)


def prometheus_metrics() -> tuple[bytes, str]:
    """Return metrics payload and content type for HTTP handlers."""
    return generate_latest(), CONTENT_TYPE_LATEST


