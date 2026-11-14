"""
ASTRAEUS Graph Engine (Neo4j & Embedded Property Graph)
Implements 4-entity astronomical identity relationships and multi-outburst graph traversal.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import networkx as nx

from astraeus_knowledge.graph.schema import (
    CYPHER_CONSTRAINTS,
    CYPHER_INDEXES,
    CYPHER_FIND_MULTI_OUTBURST_LOCI,
    CYPHER_GET_LOCUS_HIERARCHY,
)


class AstraeusGraphEngine:
    """
    Property graph engine managing 4-entity astronomical hierarchies.
    Supports live Neo4j database or embedded high-performance NetworkX graph.
    """
    def __init__(self, neo4j_uri: Optional[str] = None, auth: Optional[Tuple[str, str]] = None):
        self.neo4j_uri = neo4j_uri
        self._driver = None
        self.mode = "embedded"
        self._graph = nx.MultiDiGraph()

        if neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(neo4j_uri, auth=auth or ("neo4j", "password"))
                self._driver.verify_connectivity()
                self.mode = "neo4j_live"
                self._init_neo4j_schema()
            except Exception:
                self._driver = None
                self.mode = "embedded"

    def _init_neo4j_schema(self) -> None:
        """Executes Cypher constraints and indexes on live Neo4j."""
        if not self._driver:
            return
        with self._driver.session() as session:
            for stmt in CYPHER_CONSTRAINTS + CYPHER_INDEXES:
                try:
                    session.run(stmt)
                except Exception:
                    pass

    def ingest_cohort_dataframes(
        self,
        loci_df: pd.DataFrame,
        episodes_df: pd.DataFrame,
        reports_df: pd.DataFrame,
        counterparts_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Populates the property graph with all 4 entities and their relationships.
        """
        start_time = time.perf_counter()

        # 1. Add Persistent Sky Loci nodes
        for _, row in loci_df.iterrows():
            loc_id = row["sky_locus_id"]
            self._graph.add_node(
                loc_id,
                label="PersistentSkyLocus",
                locus_id=loc_id,
                ra=float(row["ra_deg"]),
                dec=float(row["dec_deg"]),
                healpix_nside32=int(row["healpix_nside32"]),
                assigned_base_split=row["assigned_base_split"],
                episode_count=int(row["episode_count"]),
            )

        # 2. Add Physical Transient Episode nodes & HOSTS edges
        for _, row in episodes_df.iterrows():
            ep_id = row["physical_event_id"]
            loc_id = row["sky_locus_id"]
            self._graph.add_node(
                ep_id,
                label="PhysicalTransientEpisode",
                episode_id=ep_id,
                locus_id=loc_id,
                ra=float(row["ra_deg"]),
                dec=float(row["dec_deg"]),
                t_start_mjd=float(row["t_start_mjd"]),
                t_end_mjd=float(row["t_end_mjd"]),
                coarse_class=row["coarse_class"],
                fine_class=row["fine_class"],
                evidence_tier=row["evidence_tier"],
                assigned_split=row["assigned_split"],
                is_quarantined=bool(row["is_quarantined"]),
            )
            # Edge: Locus HOSTS Episode
            if loc_id in self._graph:
                self._graph.add_edge(loc_id, ep_id, rel_type="HOSTS")

        # 3. Add Catalog Report Record nodes & SUPPORTED_BY edges
        for _, row in reports_df.iterrows():
            rep_id = row["report_id"]
            ep_id = row["linked_episode_id"]
            loc_id = row["sky_locus_id"]
            self._graph.add_node(
                rep_id,
                label="CatalogReportRecord",
                report_id=rep_id,
                source_system=row["source_system"],
                source_record_id=str(row["source_record_id"]),
                raw_class=str(row.get("raw_class", "")),
                evidence_tier=row["evidence_tier"],
                discovery_mjd=float(row["discovery_mjd_utc"]) if pd.notna(row.get("discovery_mjd_utc")) else None,
            )
            # Edge: Episode SUPPORTED_BY Report
            if ep_id in self._graph:
                self._graph.add_edge(ep_id, rep_id, rel_type="SUPPORTED_BY")

        # 4. Add Catalog Counterpart Record nodes & ASSOCIATED_WITH edges
        for _, row in counterparts_df.iterrows():
            cp_id = row["counterpart_id"]
            loc_id = row["sky_locus_id"]
            self._graph.add_node(
                cp_id,
                label="CatalogCounterpartRecord",
                counterpart_id=cp_id,
                source_catalog=row["source_catalog"],
                counterpart_type=row["counterpart_type"],
                angular_offset_arcsec=float(row["angular_offset_arcsec"]),
                host_redshift=float(row["host_redshift"]) if pd.notna(row.get("host_redshift")) else None,
                is_nuclear=bool(row.get("is_nuclear_locus", False)),
            )
            # Edge: Locus ASSOCIATED_WITH Counterpart
            if loc_id in self._graph:
                self._graph.add_edge(loc_id, cp_id, rel_type="ASSOCIATED_WITH")

        duration = time.perf_counter() - start_time
        total_nodes = self._graph.number_of_nodes()
        total_edges = self._graph.number_of_edges()

        return {
            "mode": self.mode,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "duration_seconds": duration,
            "throughput_entities_per_sec": (total_nodes + total_edges) / max(0.001, duration),
        }

    def find_multi_outburst_loci(self) -> List[Dict[str, Any]]:
        """
        Executes graph traversal to identify persistent sky loci hosting
        multiple physical transient episodes (e.g. recurrent novae, AGN flares).
        """
        multi_outburst = []
        for node, data in self._graph.nodes(data=True):
            if data.get("label") == "PersistentSkyLocus":
                # Find outgoing HOSTS edges
                hosted_episodes = [
                    target for _, target, edge_data in self._graph.out_edges(node, data=True)
                    if edge_data.get("rel_type") == "HOSTS"
                ]
                if len(hosted_episodes) > 1:
                    multi_outburst.append({
                        "sky_locus_id": node,
                        "ra_deg": data.get("ra"),
                        "dec_deg": data.get("dec"),
                        "episode_count": len(hosted_episodes),
                        "episode_ids": hosted_episodes,
                    })

        multi_outburst.sort(key=lambda x: x["episode_count"], reverse=True)
        return multi_outburst

    def get_locus_hierarchy(self, locus_id: str) -> Dict[str, Any]:
        """
        Traverses and retrieves the complete entity tree for a given sky locus.
        """
        if locus_id not in self._graph:
            return {}

        locus_data = dict(self._graph.nodes[locus_id])
        hosted_episodes = []
        counterparts = []

        for _, target, edge_data in self._graph.out_edges(locus_id, data=True):
            rel = edge_data.get("rel_type")
            target_data = dict(self._graph.nodes[target])
            if rel == "HOSTS":
                # Also retrieve reports supporting this episode
                reports = [
                    dict(self._graph.nodes[rep])
                    for _, rep, rep_edge in self._graph.out_edges(target, data=True)
                    if rep_edge.get("rel_type") == "SUPPORTED_BY"
                ]
                target_data["reports"] = reports
                hosted_episodes.append(target_data)
            elif rel == "ASSOCIATED_WITH":
                counterparts.append(target_data)

        return {
            "locus": locus_data,
            "episodes": hosted_episodes,
            "counterparts": counterparts,
        }

    def get_summary_counts(self) -> Dict[str, int]:
        """Returns node and edge category counts."""
        node_counts: Dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            lbl = data.get("label", "Unknown")
            node_counts[lbl] = node_counts.get(lbl, 0) + 1

        edge_counts: Dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            rel = data.get("rel_type", "Unknown")
            edge_counts[rel] = edge_counts.get(rel, 0) + 1

        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
        }
