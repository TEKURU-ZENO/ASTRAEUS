"""
Tests for ASTRAEUS Graph Engine, 4-Entity Relationships, and Traversal.
"""

import pytest
import pandas as pd
from astraeus_core.exceptions import LiveBackendUnavailableError
from astraeus_data.storage import BackendMode
from astraeus_knowledge.graph.engine import AstraeusGraphEngine


@pytest.fixture
def graph_engine():
    return AstraeusGraphEngine(backend_mode=BackendMode.LOCAL_EMBEDDED)


def test_strict_live_mode_prohibits_silent_fallback():
    """Verifies that requesting STRICT_LIVE with missing or unreachable Neo4j URI raises LiveBackendUnavailableError."""
    with pytest.raises(LiveBackendUnavailableError, match="Neo4j URI must be explicitly supplied"):
        AstraeusGraphEngine(neo4j_uri=None, backend_mode=BackendMode.STRICT_LIVE)

    with pytest.raises(LiveBackendUnavailableError, match="Failed to connect to live Neo4j server"):
        AstraeusGraphEngine(neo4j_uri="bolt://127.0.0.1:54321", backend_mode=BackendMode.STRICT_LIVE)


def test_graph_ingestion_and_exact_multi_outburst_traversal(graph_engine):
    """
    Ingests real cohort tables and verifies graph node/edge integrity
    and asserts the exact count of 40 multi-outburst persistent loci.
    """
    loci_df = pd.read_parquet("data/processed/real_survey_10k_v1/sky_loci.parquet")
    episodes_df = pd.read_parquet("data/processed/real_survey_10k_v1/physical_episodes.parquet")
    reports_df = pd.read_parquet("data/processed/real_survey_10k_v1/catalog_reports.parquet")
    counterparts_df = pd.read_parquet("data/processed/real_survey_10k_v1/host_counterparts.parquet")

    metrics = graph_engine.ingest_cohort_dataframes(
        loci_df=loci_df,
        episodes_df=episodes_df,
        reports_df=reports_df,
        counterparts_df=counterparts_df,
    )

    counts = graph_engine.get_summary_counts()
    nodes = counts["nodes"]
    edges = counts["edges"]

    assert nodes["PersistentSkyLocus"] == 9928
    assert nodes["PhysicalTransientEpisode"] == 9968
    assert nodes["CatalogReportRecord"] == 4479
    assert nodes["CatalogCounterpartRecord"] == 1061

    assert edges["HOSTS"] == 9968
    assert edges["SUPPORTED_BY"] == 4479
    assert edges["ASSOCIATED_WITH"] == 1061

    # Traversal test: multi-outburst loci (EXACT ASSERTION: 40)
    multi = graph_engine.find_multi_outburst_loci()
    assert len(multi) == 40, f"Expected exactly 40 multi-outburst loci, got {len(multi)}"

    for m in multi:
        assert m["episode_count"] > 1
        assert len(m["episode_ids"]) == m["episode_count"]

    sample_locus_id = multi[0]["sky_locus_id"]
    tree = graph_engine.get_locus_hierarchy(sample_locus_id)
    assert tree["locus"]["locus_id"] == sample_locus_id
    assert len(tree["episodes"]) == multi[0]["episode_count"]
