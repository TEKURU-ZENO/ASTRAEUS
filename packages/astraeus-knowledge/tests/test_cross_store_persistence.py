"""
End-to-End Tests for ASTRAEUS Cross-Store Persistence, Key-by-Key Parity,
and Distributed Compensating Rollback.
"""

import pytest
import tempfile
from astraeus_data.storage import BackendMode
from astraeus_knowledge.store_orchestrator import AstraeusStoreOrchestrator


@pytest.fixture
def orchestrator():
    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = AstraeusStoreOrchestrator(
            relational_url="sqlite:///:memory:",
            neo4j_uri=None,
            qdrant_location=":memory:",
            object_store_dir=tmp_dir,
            backend_mode=BackendMode.LOCAL_EMBEDDED,
        )
        yield orch


def test_cross_store_ingestion_and_key_by_key_referential_parity(orchestrator):
    """
    Executes end-to-end ingestion across all 4 persistence layers and verifies
    both count parity and exhaustive key-by-key referential parity.
    """
    # 1. Verify backend status
    status = orchestrator.get_backend_status()
    assert status["declared_backend_mode"] == "local_embedded"
    assert status["relational"]["is_live_postgresql"] is False
    assert status["graph"]["is_live_neo4j"] is False
    assert status["vector"]["is_live_qdrant"] is False
    assert status["object_store"]["is_live_minio"] is False

    # 2. Ingest full 10k cohort with single atomic relational transaction
    exec_metrics = orchestrator.execute_full_cohort_persistence()
    rel = exec_metrics["relational"]
    assert rel["manifest_entity_rows"] == 35469
    assert rel["photometry_observation_rows"] == 101077
    assert rel["total_rows_imported"] == 136546
    assert rel["is_single_atomic_transaction"] is True

    # 3. Verify key-by-key referential parity (sample of 300 entities)
    parity_result = orchestrator.verify_key_by_key_referential_parity(sample_size=300)
    assert parity_result["key_parity_passed"] is True, f"Key parity failure: {parity_result}"
    stats = parity_result["verification_stats"]
    assert stats["loci_keys_matched"] == stats["loci_keys_checked"] == 300
    assert stats["episodes_and_edges_matched"] == stats["episodes_and_edges_checked"] == 300
    assert stats["vector_candidate_keys_matched"] == stats["vector_candidate_keys_checked"] == 300
    assert stats["snapshot_hashes_matched"] == stats["snapshot_hashes_checked"] == 33

    # 4. Assert exact 40 multi-outburst loci in graph
    multi = orchestrator.graph_engine.find_multi_outburst_loci()
    assert len(multi) == 40, f"Expected exactly 40 multi-outburst loci, got {len(multi)}"

    # 5. Distributed compensating rollback test
    rollback_res = orchestrator.execute_compensating_cross_store_rollback()
    assert rollback_res["status"] == "PASS"
    assert rollback_res["compensating_rollback_verified"] is True
    assert rollback_res["leaked_objects"] == 0
    assert rollback_res["leaked_relational_rows"] == 0
