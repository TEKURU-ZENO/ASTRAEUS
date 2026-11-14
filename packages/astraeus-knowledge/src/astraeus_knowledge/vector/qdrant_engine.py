"""
ASTRAEUS Vector Engine (Qdrant Integration)
Manages multi-dimensional astronomical embedding representations, similarity search,
metadata filtering, and vector lifecycle operations.
"""

from __future__ import annotations

import time
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


class AstraeusVectorEngine:
    """
    Qdrant vector engine for astronomical transient embeddings.
    Supports in-memory high-speed execution or live distributed Qdrant instances.
    """
    def __init__(
        self,
        location: str = ":memory:",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "astraeus_candidates_v1",
        vector_dim: int = 24,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(location)

        self._init_collection()

    def _init_collection(self) -> None:
        """Initializes or resets the target collection."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def ingest_candidates_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """
        Embeds and ingests candidate records into Qdrant in batches.
        """
        start_time = time.perf_counter()
        total_points = len(df)
        points: List[PointStruct] = []

        for idx, row in df.iterrows():
            cand_id = str(row["candidate_id"])
            # Generate deterministic 24-dim pseudo-embedding based on astrometric and temporal features
            # or hash-seeded gaussian features to ensure reproducibility
            seed = int(hashlib.sha256(cand_id.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vector = rng.randn(self.vector_dim).astype(float)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            payload = {
                "candidate_id": cand_id,
                "physical_event_id": str(row["physical_event_id"]),
                "sky_locus_id": str(row["sky_locus_id"]),
                "coarse_class": str(row["coarse_class"]),
                "fine_class": str(row["fine_class"]),
                "evidence_tier": str(row["evidence_tier"]),
                "assigned_split": str(row["assigned_split"]),
                "is_quarantined": bool(row.get("is_quarantined", False)),
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
            }

            points.append(PointStruct(id=int(idx + 1), vector=vector.tolist(), payload=payload))

        # Batch upsert
        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=chunk)

        duration = time.perf_counter() - start_time
        return {
            "collection_name": self.collection_name,
            "total_points_ingested": total_points,
            "vector_dim": self.vector_dim,
            "duration_seconds": duration,
            "throughput_points_per_sec": total_points / max(0.001, duration),
        }

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_tier: Optional[str] = None,
        filter_class: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Nearest neighbor search with optional metadata filtering.
        """
        query_filter = None
        must_conditions = []
        if filter_tier:
            must_conditions.append(FieldCondition(key="evidence_tier", match=MatchValue(value=filter_tier)))
        if filter_class:
            must_conditions.append(FieldCondition(key="coarse_class", match=MatchValue(value=filter_class)))

        if must_conditions:
            query_filter = Filter(must=must_conditions)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        results = []
        for p in res.points:
            results.append({
                "point_id": p.id,
                "score": float(p.score) if p.score is not None else 0.0,
                "payload": p.payload,
            })
        return results

    def get_candidate(self, point_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a point and its payload by ID."""
        pts = self.client.retrieve(collection_name=self.collection_name, ids=[point_id])
        if pts:
            return {"point_id": pts[0].id, "payload": pts[0].payload}
        return None

    def delete_candidate(self, point_id: int) -> bool:
        """Deletes a point by ID and verifies removal."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
        return self.get_candidate(point_id) is None

    def count_vectors(self) -> int:
        """Returns total vector points in collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count
