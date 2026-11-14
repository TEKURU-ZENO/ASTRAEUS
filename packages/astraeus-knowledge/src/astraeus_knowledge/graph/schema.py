"""
ASTRAEUS Graph Schema & Cypher Templates
Defines the 4-Entity Identity Graph Model and Traversal Queries.
"""

from typing import Dict, Any, List

# Cypher DDL & Constraints
CYPHER_CONSTRAINTS = [
    "CREATE CONSTRAINT c_locus_id IF NOT EXISTS FOR (l:PersistentSkyLocus) REQUIRE l.locus_id IS UNIQUE;",
    "CREATE CONSTRAINT c_episode_id IF NOT EXISTS FOR (e:PhysicalTransientEpisode) REQUIRE e.episode_id IS UNIQUE;",
    "CREATE CONSTRAINT c_report_id IF NOT EXISTS FOR (r:CatalogReportRecord) REQUIRE r.report_id IS UNIQUE;",
    "CREATE CONSTRAINT c_counterpart_id IF NOT EXISTS FOR (c:CatalogCounterpartRecord) REQUIRE c.counterpart_id IS UNIQUE;",
]

CYPHER_INDEXES = [
    "CREATE INDEX idx_locus_coords IF NOT EXISTS FOR (l:PersistentSkyLocus) ON (l.ra, l.dec);",
    "CREATE INDEX idx_episode_tier IF NOT EXISTS FOR (e:PhysicalTransientEpisode) ON (e.evidence_tier);",
    "CREATE INDEX idx_episode_class IF NOT EXISTS FOR (e:PhysicalTransientEpisode) ON (e.coarse_class);",
    "CREATE INDEX idx_counterpart_type IF NOT EXISTS FOR (c:CatalogCounterpartRecord) ON (c.counterpart_type);",
]

# Query Templates
CYPHER_FIND_MULTI_OUTBURST_LOCI = """
MATCH (l:PersistentSkyLocus)-[:HOSTS]->(e:PhysicalTransientEpisode)
WITH l, count(e) AS episode_count, collect(e) AS episodes
WHERE episode_count > 1
RETURN l.locus_id AS locus_id, l.ra AS ra, l.dec AS dec, episode_count, [ep IN episodes | ep.episode_id] AS episode_ids
ORDER BY episode_count DESC;
"""

CYPHER_GET_LOCUS_HIERARCHY = """
MATCH (l:PersistentSkyLocus {locus_id: $locus_id})
OPTIONAL MATCH (l)-[:HOSTS]->(e:PhysicalTransientEpisode)
OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(r:CatalogReportRecord)
OPTIONAL MATCH (l)-[:ASSOCIATED_WITH]->(c:CatalogCounterpartRecord)
RETURN l, collect(DISTINCT e) AS episodes, collect(DISTINCT r) AS reports, collect(DISTINCT c) AS counterparts;
"""
