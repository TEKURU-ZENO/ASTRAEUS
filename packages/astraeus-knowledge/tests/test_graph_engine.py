"""
Tests for ASTRAEUS Graph Engine, 4-Entity Relationships, and Traversal.
"""

import pytest
import pandas as pd
from astraeus_knowledge.graph.engine import AstraeusGraphEngine


@pytest.fixture
def graph_engine():
    return AstraeusGraphEngine()


def test_graph_ingestion_and_multi_outburst_traversal(graph_engine):
    """
    Ingests real cohort tables and verifies graph node/edge integrity
    and multi-outburst locus traversal.
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

    # Traversal test: multi-outburst loci
    multi = graph_engine.find_multi_outburst_loci()
    assert len(multi) > 0
    # Every returned locus must have episode_count > 1
    for m in multi:
        assert m["episode_count"] > 1
        assert len(m["episode_ids"]) == m["episode_count"]

    # Traversal test: hierarchy inspection
    sample_locus_id = multi[0]["sky_locus_id"]
    tree = graph_engine.get_locus_hierarchy(sample_locus_id)
    assert tree["locus"]["locus_id"] == sample_locus_id
    assert len(tree["episodes"]) == multi[0]["episode_count"]
