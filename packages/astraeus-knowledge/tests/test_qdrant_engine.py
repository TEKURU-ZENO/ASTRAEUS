"""
Tests for ASTRAEUS Qdrant Vector Engine and Similarity Retrieval.
"""

import pytest
import numpy as np
import pandas as pd
from astraeus_knowledge.vector.qdrant_engine import AstraeusVectorEngine


@pytest.fixture
def vector_engine():
    return AstraeusVectorEngine(location=":memory:", vector_dim=24)


def test_qdrant_vector_ingestion_and_search(vector_engine):
    """
    Ingests 10k candidates into Qdrant and verifies nearest-neighbor search,
    evidence-tier filtering, payload retrieval, and deletion.
    """
    candidates_df = pd.read_parquet("data/processed/real_survey_10k_v1/candidates_10k.parquet")
    
    # Ingest cohort
    metrics = vector_engine.ingest_candidates_dataframe(candidates_df, batch_size=500)
    assert metrics["total_points_ingested"] == 10000
    assert vector_engine.count_vectors() == 10000

    # Search similar
    query_vector = np.random.randn(24).tolist()
    results = vector_engine.search_similar(query_vector, limit=5)
    assert len(results) == 5
    assert "candidate_id" in results[0]["payload"]
    assert "evidence_tier" in results[0]["payload"]

    # Search with evidence tier filter
    filtered_results = vector_engine.search_similar(query_vector, limit=5, filter_tier="L1")
    assert len(filtered_results) > 0
    for r in filtered_results:
        assert r["payload"]["evidence_tier"] == "L1"

    # Retrieve point by ID
    pt = vector_engine.get_candidate(1)
    assert pt is not None
    assert pt["point_id"] == 1

    # Delete point
    del_ok = vector_engine.delete_candidate(1)
    assert del_ok is True
    assert vector_engine.count_vectors() == 9999
    assert vector_engine.get_candidate(1) is None
