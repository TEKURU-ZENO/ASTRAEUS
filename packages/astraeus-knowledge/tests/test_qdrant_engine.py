"""
Tests for ASTRAEUS Qdrant Vector Engine, Physical Embeddings, and Similarity.
"""

import pytest
import numpy as np
import pandas as pd
from astraeus_core.exceptions import LiveBackendUnavailableError
from astraeus_data.storage import BackendMode
from astraeus_knowledge.vector.qdrant_engine import AstraeusVectorEngine


@pytest.fixture
def vector_engine():
    return AstraeusVectorEngine(location=":memory:", vector_dim=24, backend_mode=BackendMode.LOCAL_EMBEDDED)


def test_strict_live_mode_prohibits_silent_fallback():
    """Verifies that requesting STRICT_LIVE with missing or unreachable Qdrant URL raises LiveBackendUnavailableError."""
    with pytest.raises(LiveBackendUnavailableError, match="Qdrant remote URL must be explicitly supplied"):
        AstraeusVectorEngine(url=None, backend_mode=BackendMode.STRICT_LIVE)

    with pytest.raises(LiveBackendUnavailableError, match="Failed to connect to live Qdrant server"):
        AstraeusVectorEngine(url="http://127.0.0.1:54321", backend_mode=BackendMode.STRICT_LIVE)


def test_astronomical_features_24d_physics_representation(vector_engine):
    """Verifies derivation of 24-dimensional physical vector embeddings from astronomical metadata."""
    sample_sn = {
        "candidate_id": "SN_TEST_001",
        "ra_deg": 180.0,
        "dec_deg": 25.0,
        "coarse_class": "SN_Ia",
        "evidence_tier": "L1",
        "n_alerce_det": 45,
        "n_alerce_ul": 12,
        "n_dr24_clean": 60,
        "first_mjd": 59000.0,
        "last_mjd": 59080.0,
        "has_msip_alert": True,
        "has_dr24_lc": True,
        "has_independent_tns": True,
        "host_counterpart_id": "HOST_001",
        "host_redshift": 0.05,
    }
    vec = vector_engine.extract_astronomical_features_24d(sample_sn)
    assert len(vec) == 24
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-5)

    # Check physical clustering: similar SN should have high cosine similarity
    sample_sn2 = dict(sample_sn, candidate_id="SN_TEST_002", ra_deg=181.0, dec_deg=25.5)
    vec2 = vector_engine.extract_astronomical_features_24d(sample_sn2)
    cosine_sim_sn = float(np.dot(vec, vec2))
    assert cosine_sim_sn > 0.85, f"Expected high SN-SN similarity, got {cosine_sim_sn}"

    # Dissimilar variable star in different sky region
    sample_var = {
        "candidate_id": "VAR_TEST_001",
        "ra_deg": 10.0,
        "dec_deg": -40.0,
        "coarse_class": "Variable_Star",
        "evidence_tier": "L4",
        "n_alerce_det": 2,
        "n_alerce_ul": 200,
        "n_dr24_clean": 0,
        "first_mjd": 58000.0,
        "last_mjd": 61000.0,
    }
    vec_var = vector_engine.extract_astronomical_features_24d(sample_var)
    cosine_sim_cross = float(np.dot(vec, vec_var))
    assert cosine_sim_cross < 0.50, f"Expected low cross-class similarity, got {cosine_sim_cross}"


def test_qdrant_vector_ingestion_and_search(vector_engine):
    """
    Ingests 10k candidates into Qdrant with physical embeddings,
    verifying nearest-neighbor search, filtering, retrieval, and deletion.
    """
    candidates_df = pd.read_parquet("data/processed/real_survey_10k_v1/candidates_10k.parquet")

    metrics = vector_engine.ingest_candidates_dataframe(candidates_df, batch_size=500)
    assert metrics["total_points_ingested"] == 10000
    assert vector_engine.count_vectors() == 10000

    # Query with physical vector from first candidate
    first_row = candidates_df.iloc[0]
    query_vector = vector_engine.extract_astronomical_features_24d(first_row)

    results = vector_engine.search_similar(query_vector, limit=5)
    assert len(results) == 5
    assert results[0]["payload"]["candidate_id"] == first_row["candidate_id"]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)

    # Filtered search
    filtered_results = vector_engine.search_similar(query_vector, limit=5, filter_tier="L1")
    assert len(filtered_results) > 0
    for r in filtered_results:
        assert r["payload"]["evidence_tier"] == "L1"

    # Delete point
    del_ok = vector_engine.delete_candidate(1)
    assert del_ok is True
    assert vector_engine.count_vectors() == 9999
    assert vector_engine.get_candidate(1) is None
