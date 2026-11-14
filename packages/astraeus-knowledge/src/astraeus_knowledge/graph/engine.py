"""
ASTRAEUS Graph Engine (Neo4j & Embedded Property Graph)
Implements 4-entity astronomical identity relationships and multi-outburst graph traversal.
Supports strict live Neo4j database execution or high-performance embedded NetworkX graph.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import networkx as nx

from astraeus_core.exceptions import LiveBackendUnavailableError
from astraeus_data.storage import BackendMode
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
    Prohibits silent backend fallback in STRICT_LIVE mode.
    """
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None,
        backend_mode: BackendMode = BackendMode.LOCAL_EMBEDDED,
    ):
        self.neo4j_uri = neo4j_uri
        self._driver = None
        self.backend_mode = backend_mode
        self._graph = nx.MultiDiGraph()

        if backend_mode == BackendMode.STRICT_LIVE:
            if not neo4j_uri:
                raise LiveBackendUnavailableError(
                    "Neo4j URI must be explicitly supplied in STRICT_LIVE mode. "
                    "Silent fallback is prohibited."
                )
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(neo4j_uri, auth=auth or ("neo4j", "password"))
                self._driver.verify_connectivity()
                self.mode = "neo4j_live"
                self._init_neo4j_schema()
            except Exception as exc:
                raise LiveBackendUnavailableError(
                    f"STRICT_LIVE mode: Failed to connect to live Neo4j server at '{neo4j_uri}': {exc}. "
                    "Silent fallback is strictly prohibited."
                ) from exc
        else:
            self.mode = "embedded"
            self._driver = None

    @property
    def is_live_backend(self) -> bool:
        """Returns True if connected to an actual remote/live Neo4j instance."""
        return self.mode == "neo4j_live"

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
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Populates the property graph with all 4 entities and their relationships.
        Executes native Cypher batch UNWIND queries in live mode, or NetworkX in embedded mode.
        """
        start_time = time.perf_counter()

        if self.mode == "neo4j_live" and self._driver:
            return self._ingest_neo4j_live(
                loci_df, episodes_df, reports_df, counterparts_df, batch_size, start_time
            )

        # Embedded NetworkX Path
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
            if loc_id in self._graph:
                self._graph.add_edge(loc_id, cp_id, rel_type="ASSOCIATED_WITH")

        duration = time.perf_counter() - start_time
        total_nodes = self._graph.number_of_nodes()
        total_edges = self._graph.number_of_edges()

        return {
            "mode": self.mode,
            "is_live_backend": self.is_live_backend,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "duration_seconds": duration,
            "throughput_entities_per_sec": (total_nodes + total_edges) / max(0.001, duration),
        }

    def _ingest_neo4j_live(
        self,
        loci_df: pd.DataFrame,
        episodes_df: pd.DataFrame,
        reports_df: pd.DataFrame,
        counterparts_df: pd.DataFrame,
        batch_size: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """Executes native Cypher batch queries on live Neo4j instance."""
        with self._driver.session() as session:
            # 1. Ingest Loci nodes
            loci_records = loci_df[["sky_locus_id", "ra_deg", "dec_deg", "healpix_nside32", "assigned_base_split", "episode_count"]].to_dict(orient="records")
            for i in range(0, len(loci_records), batch_size):
                batch = loci_records[i:i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (l:PersistentSkyLocus {locus_id: row.sky_locus_id})
                    SET l.ra = row.ra_deg, l.dec = row.dec_deg,
                        l.healpix_nside32 = row.healpix_nside32,
                        l.assigned_base_split = row.assigned_base_split,
                        l.episode_count = row.episode_count
                    """,
                    batch=batch,
                )

            # 2. Ingest Episode nodes & HOSTS edges
            episodes_records = episodes_df[["physical_event_id", "sky_locus_id", "ra_deg", "dec_deg", "t_start_mjd", "t_end_mjd", "coarse_class", "fine_class", "evidence_tier", "assigned_split", "is_quarantined"]].to_dict(orient="records")
            for i in range(0, len(episodes_records), batch_size):
                batch = episodes_records[i:i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (e:PhysicalTransientEpisode {episode_id: row.physical_event_id})
                    SET e.ra = row.ra_deg, e.dec = row.dec_deg,
                        e.t_start_mjd = row.t_start_mjd, e.t_end_mjd = row.t_end_mjd,
                        e.coarse_class = row.coarse_class, e.fine_class = row.fine_class,
                        e.evidence_tier = row.evidence_tier, e.assigned_split = row.assigned_split,
                        e.is_quarantined = row.is_quarantined
                    WITH e, row
                    MATCH (l:PersistentSkyLocus {locus_id: row.sky_locus_id})
                    MERGE (l)-[:HOSTS]->(e)
                    """,
                    batch=batch,
                )

            # 3. Ingest Report nodes & SUPPORTED_BY edges
            reports_records = reports_df[["report_id", "linked_episode_id", "source_system", "source_record_id", "raw_class", "evidence_tier", "discovery_mjd_utc"]].to_dict(orient="records")
            for i in range(0, len(reports_records), batch_size):
                batch = reports_records[i:i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (r:CatalogReportRecord {report_id: row.report_id})
                    SET r.source_system = row.source_system, r.source_record_id = row.source_record_id,
                        r.raw_class = row.raw_class, r.evidence_tier = row.evidence_tier,
                        r.discovery_mjd = row.discovery_mjd_utc
                    WITH r, row
                    MATCH (e:PhysicalTransientEpisode {episode_id: row.linked_episode_id})
                    MERGE (e)-[:SUPPORTED_BY]->(r)
                    """,
                    batch=batch,
                )

            # 4. Ingest Counterpart nodes & ASSOCIATED_WITH edges
            counterparts_records = counterparts_df[["counterpart_id", "sky_locus_id", "source_catalog", "counterpart_type", "angular_offset_arcsec", "host_redshift"]].to_dict(orient="records")
            for i in range(0, len(counterparts_records), batch_size):
                batch = counterparts_records[i:i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (c:CatalogCounterpartRecord {counterpart_id: row.counterpart_id})
                    SET c.source_catalog = row.source_catalog, c.counterpart_type = row.counterpart_type,
                        c.angular_offset_arcsec = row.angular_offset_arcsec, c.host_redshift = row.host_redshift
                    WITH c, row
                    MATCH (l:PersistentSkyLocus {locus_id: row.sky_locus_id})
                    MERGE (l)-[:ASSOCIATED_WITH]->(c)
                    """,
                    batch=batch,
                )

        duration = time.perf_counter() - start_time
        summary = self.get_summary_counts()
        total_nodes = summary["total_nodes"]
        total_edges = summary["total_edges"]

        return {
            "mode": self.mode,
            "is_live_backend": self.is_live_backend,
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
        if self.mode == "neo4j_live" and self._driver:
            with self._driver.session() as session:
                result = session.run(CYPHER_FIND_MULTI_OUTBURST_LOCI)
                multi_outburst = []
                for record in result:
                    multi_outburst.append({
                        "sky_locus_id": record["locus_id"],
                        "ra_deg": record["ra"],
                        "dec_deg": record["dec"],
                        "episode_count": record["episode_count"],
                        "episode_ids": record["episode_ids"],
                    })
                return multi_outburst

        # Embedded NetworkX Path
        multi_outburst = []
        for node, data in self._graph.nodes(data=True):
            if data.get("label") == "PersistentSkyLocus":
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
        if self.mode == "neo4j_live" and self._driver:
            with self._driver.session() as session:
                res = session.run(CYPHER_GET_LOCUS_HIERARCHY, locus_id=locus_id).single()
                if not res:
                    return {}
                return {
                    "locus": dict(res["l"]) if res["l"] else {},
                    "episodes": [dict(ep) for ep in res["episodes"]],
                    "reports": [dict(r) for r in res["reports"]],
                    "counterparts": [dict(c) for c in res["counterparts"]],
                }

        # Embedded NetworkX Path
        if locus_id not in self._graph:
            return {}

        locus_data = dict(self._graph.nodes[locus_id])
        hosted_episodes = []
        counterparts = []

        for _, target, edge_data in self._graph.out_edges(locus_id, data=True):
            rel = edge_data.get("rel_type")
            target_data = dict(self._graph.nodes[target])
            if rel == "HOSTS":
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

    def get_summary_counts(self) -> Dict[str, Any]:
        """Returns node and edge category counts."""
        if self.mode == "neo4j_live" and self._driver:
            with self._driver.session() as session:
                nodes_res = session.run("MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt")
                node_counts = {r["lbl"]: r["cnt"] for r in nodes_res}
                edges_res = session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt")
                edge_counts = {r["t"]: r["cnt"] for r in edges_res}
                total_n = sum(node_counts.values())
                total_e = sum(edge_counts.values())
                return {
                    "nodes": node_counts,
                    "edges": edge_counts,
                    "total_nodes": total_n,
                    "total_edges": total_e,
                }

        # Embedded NetworkX Path
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
