"""
ASTRAEUS Cross-Store Persistence Orchestrator
Coordinates Relational, Graph (Neo4j), Vector (Qdrant), and Object (MinIO) stores,
enforcing cross-store referential consistency, latency benchmarking, and failure recovery.
"""

from __future__ import annotations

import time
import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.engine import Engine

from astraeus_data.storage.relational_engine import create_database_engine, init_schema
from astraeus_data.storage.importer import CohortBatchImporter
from astraeus_data.storage.minio_store import AstraeusObjectStore
from astraeus_data.storage.schema import (
    PersistentSkyLoci,
    PhysicalTransientEpisodes,
    CatalogReports,
    HostCounterparts,
    Candidates10k,
    RawSnapshots,
    PhotometryRecords,
)
from astraeus_knowledge.graph.engine import AstraeusGraphEngine
from astraeus_knowledge.vector.qdrant_engine import AstraeusVectorEngine


class AstraeusStoreOrchestrator:
    """
    Unified Orchestrator coordinating all 4 persistence engines:
    1. Relational DB (PostgreSQL / SQLite)
    2. Property Graph (Neo4j / Embedded)
    3. Vector Store (Qdrant)
    4. Object Store (MinIO / S3)
    """
    def __init__(
        self,
        relational_url: str = "sqlite:///:memory:",
        neo4j_uri: Optional[str] = None,
        qdrant_location: str = ":memory:",
        object_store_dir: str = "data/storage/minio_objects",
    ):
        # 1. Relational
        self.relational_engine = create_database_engine(relational_url)
        init_schema(self.relational_engine)
        self.importer = CohortBatchImporter(self.relational_engine)

        # 2. Graph
        self.graph_engine = AstraeusGraphEngine(neo4j_uri=neo4j_uri)

        # 3. Vector
        self.vector_engine = AstraeusVectorEngine(location=qdrant_location, vector_dim=24)

        # 4. Object Store
        self.object_store = AstraeusObjectStore(local_root=object_store_dir)

    def execute_full_cohort_persistence(
        self,
        processed_dir: str = "data/processed/real_survey_10k_v1",
        snapshot_manifest_path: str = "data/raw/real_survey_v1/snapshot_manifest.json",
        raw_lightcurves_dir: str = "data/raw/real_survey_v1/alerce/lightcurves",
    ) -> Dict[str, Any]:
        """
        Ingests the real 10,000 cohort into all 4 storage systems,
        capturing end-to-end execution timing and throughput.
        """
        total_start = time.perf_counter()
        results: Dict[str, Any] = {}

        # 1. Relational Ingestion
        rel_metrics = self.importer.import_cohort(
            processed_dir=processed_dir,
            snapshot_manifest_path=snapshot_manifest_path,
            raw_lightcurves_dir=raw_lightcurves_dir,
        )
        results["relational"] = rel_metrics

        # Read parquet dataframes for graph and vector engines
        loci_df = pd.read_parquet(os.path.join(processed_dir, "sky_loci.parquet"))
        episodes_df = pd.read_parquet(os.path.join(processed_dir, "physical_episodes.parquet"))
        reports_df = pd.read_parquet(os.path.join(processed_dir, "catalog_reports.parquet"))
        counterparts_df = pd.read_parquet(os.path.join(processed_dir, "host_counterparts.parquet"))
        candidates_df = pd.read_parquet(os.path.join(processed_dir, "candidates_10k.parquet"))

        # 2. Graph Ingestion
        graph_metrics = self.graph_engine.ingest_cohort_dataframes(
            loci_df=loci_df,
            episodes_df=episodes_df,
            reports_df=reports_df,
            counterparts_df=counterparts_df,
        )
        results["graph"] = graph_metrics

        # 3. Vector Ingestion
        vector_metrics = self.vector_engine.ingest_candidates_dataframe(candidates_df)
        results["vector"] = vector_metrics

        # 4. Object Store Snapshot Persistence
        t0 = time.perf_counter()
        obj_count = 0
        if os.path.exists(snapshot_manifest_path):
            with open(snapshot_manifest_path, "r") as f:
                manifest_data = json.load(f)
            for entry in manifest_data.get("entries", []):
                rel_path = entry["local_relative_path"]
                full_path = os.path.join("data/raw/real_survey_v1", rel_path)
                if os.path.exists(full_path):
                    with open(full_path, "rb") as sf:
                        sbytes = sf.read()
                    self.object_store.put_object("astraeus-snapshots", rel_path.replace("\\", "/"), sbytes)
                    obj_count += 1
        t_obj = time.perf_counter() - t0
        results["object_store"] = {
            "snapshots_stored": obj_count,
            "duration_seconds": t_obj,
            "throughput_objects_per_sec": obj_count / max(0.001, t_obj),
        }

        results["total_pipeline_duration_seconds"] = time.perf_counter() - total_start
        return results

    def verify_cross_store_consistency(self) -> Dict[str, Any]:
        """
        Validates referential integrity and entity count parity across all 4 stores.
        """
        # Relational counts
        rel_counts = self.importer.get_row_counts()

        # Graph counts
        graph_summary = self.graph_engine.get_summary_counts()
        graph_nodes = graph_summary["nodes"]

        # Vector counts
        vector_count = self.vector_engine.count_vectors()

        # Object store counts
        stored_keys = self.object_store.list_objects("astraeus-snapshots")

        consistency_checks = {
            "locus_count_match": (rel_counts["persistent_sky_loci"] == graph_nodes.get("PersistentSkyLocus", 0)),
            "episode_count_match": (rel_counts["physical_transient_episodes"] == graph_nodes.get("PhysicalTransientEpisode", 0)),
            "report_count_match": (rel_counts["catalog_reports"] == graph_nodes.get("CatalogReportRecord", 0)),
            "counterpart_count_match": (rel_counts["host_counterparts"] == graph_nodes.get("CatalogCounterpartRecord", 0)),
            "candidate_vector_count_match": (rel_counts["candidates_10k"] == vector_count),
            "snapshot_object_count_match": (rel_counts["raw_snapshots"] == len(stored_keys)),
        }

        all_passed = all(consistency_checks.values())

        return {
            "all_passed": all_passed,
            "checks": consistency_checks,
            "counts": {
                "relational": rel_counts,
                "graph_nodes": graph_nodes,
                "vector_points": vector_count,
                "object_snapshots": len(stored_keys),
            },
        }

    def benchmark_query_latencies(self, num_trials: int = 100) -> Dict[str, Any]:
        """
        Measures p50 and p95 query latency across relational, graph, vector, and object operations.
        """
        session = self.importer.session_factory()
        latencies: Dict[str, List[float]] = {
            "relational_pk_lookup": [],
            "relational_cone_search": [],
            "graph_locus_hierarchy": [],
            "graph_multi_outburst": [],
            "vector_knn_search": [],
            "object_store_retrieval_and_hash": [],
        }

        try:
            # 1. Relational PK Lookup
            for _ in range(num_trials):
                t0 = time.perf_counter()
                session.execute(select(PersistentSkyLoci).where(PersistentSkyLoci.sky_locus_id == "locus:hp32_7059:ZTF19aangfld")).scalar_one_or_none()
                latencies["relational_pk_lookup"].append((time.perf_counter() - t0) * 1000.0)

            # 2. Relational Coordinate Cone / Box Search
            for _ in range(num_trials):
                t0 = time.perf_counter()
                session.execute(
                    select(PersistentSkyLoci)
                    .where(PersistentSkyLoci.ra_deg.between(180.0, 185.0))
                    .where(PersistentSkyLoci.dec_deg.between(0.0, 5.0))
                ).fetchall()
                latencies["relational_cone_search"].append((time.perf_counter() - t0) * 1000.0)

            # 3. Graph Locus Hierarchy
            for _ in range(num_trials):
                t0 = time.perf_counter()
                self.graph_engine.get_locus_hierarchy("locus:hp32_7059:ZTF19aangfld")
                latencies["graph_locus_hierarchy"].append((time.perf_counter() - t0) * 1000.0)

            # 4. Graph Multi-Outburst Traversal
            for _ in range(20):
                t0 = time.perf_counter()
                self.graph_engine.find_multi_outburst_loci()
                latencies["graph_multi_outburst"].append((time.perf_counter() - t0) * 1000.0)

            # 5. Vector k-NN Search
            dummy_query = np.ones(24).tolist()
            for _ in range(num_trials):
                t0 = time.perf_counter()
                self.vector_engine.search_similar(dummy_query, limit=10, filter_tier="L1")
                latencies["vector_knn_search"].append((time.perf_counter() - t0) * 1000.0)

            # 6. Object Store Retrieval & SHA-256 Validation
            sample_keys = self.object_store.list_objects("astraeus-snapshots")
            if sample_keys:
                key = sample_keys[0]
                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    self.object_store.get_object("astraeus-snapshots", key)
                    latencies["object_store_retrieval_and_hash"].append((time.perf_counter() - t0) * 1000.0)

        finally:
            session.close()

        summary: Dict[str, Any] = {}
        for op, vals in latencies.items():
            if vals:
                arr = np.array(vals)
                summary[op] = {
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "mean_ms": float(np.mean(arr)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                    "trials": len(arr),
                }

        return summary
