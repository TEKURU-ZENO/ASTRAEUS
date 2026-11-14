"""
End-to-End Tests for ASTRAEUS Cross-Store Persistence and Referential Consistency.
"""

import pytest
import tempfile
from astraeus_knowledge.store_orchestrator import AstraeusStoreOrchestrator


@pytest.fixture
def orchestrator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = AstraeusStoreOrchestrator(
            relational_url="sqlite:///:memory:",
            neo4j_uri=None,
            qdrant_location=":memory:",
            object_store_dir=tmp_dir,
        )
        yield orch


def test_cross_store_ingestion_and_referential_consistency(orchestrator):
    """
    Executes end-to-end ingestion across all 4 persistence layers and verifies
    exact referential parity and query performance.
    """
    # 1. Ingest full 10k cohort
    exec_metrics = orchestrator.execute_full_cohort_persistence()
    assert "relational" in exec_metrics
    assert "graph" in exec_metrics
    assert "vector" in exec_metrics
    assert "object_store" in exec_metrics

    # 2. Verify Cross-Store Consistency
    consistency = orchestrator.verify_cross_store_consistency()
    assert consistency["all_passed"] is True, f"Consistency failure: {consistency['checks']}"
    assert consistency["counts"]["relational"]["persistent_sky_loci"] == 9928
    assert consistency["counts"]["relational"]["physical_transient_episodes"] == 9968
    assert consistency["counts"]["relational"]["catalog_reports"] == 4479
    assert consistency["counts"]["relational"]["host_counterparts"] == 1061
    assert consistency["counts"]["relational"]["candidates_10k"] == 10000
    assert consistency["counts"]["relational"]["raw_snapshots"] == 33

    # 3. Latency Benchmarks
    latencies = orchestrator.benchmark_query_latencies(num_trials=25)
    assert "relational_pk_lookup" in latencies
    assert "graph_locus_hierarchy" in latencies
    assert "vector_knn_search" in latencies
    assert "object_store_retrieval_and_hash" in latencies

    # Check p50 latencies are within reasonable milliseconds
    assert latencies["relational_pk_lookup"]["p50_ms"] < 20.0
    assert latencies["graph_locus_hierarchy"]["p50_ms"] < 20.0
    assert latencies["vector_knn_search"]["p50_ms"] < 50.0
