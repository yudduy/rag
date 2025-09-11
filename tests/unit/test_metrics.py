def test_observe_query_updates_counters():
    from src.metrics import queries_total, cache_hits_total, cache_misses_total, observe_query

    # Snapshot current counts
    q0 = queries_total._value.get()
    h0 = cache_hits_total._value.get()
    m0 = cache_misses_total._value.get()

    observe_query(0.123, from_cache=True)
    assert queries_total._value.get() == q0 + 1
    assert cache_hits_total._value.get() == h0 + 1
    assert cache_misses_total._value.get() == m0

    observe_query(0.456, from_cache=False)
    assert queries_total._value.get() == q0 + 2
    assert cache_hits_total._value.get() == h0 + 1
    assert cache_misses_total._value.get() == m0 + 1

