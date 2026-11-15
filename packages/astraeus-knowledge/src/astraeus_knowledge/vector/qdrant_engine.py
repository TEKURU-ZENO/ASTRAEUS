"""
ASTRAEUS Vector Engine (Qdrant Integration)
Manages multi-dimensional astronomical embedding representations, similarity search,
metadata filtering, and vector lifecycle operations.
"""

from __future__ import annotations

import time
import math
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

from astraeus_core.exceptions import LiveBackendUnavailableError
from astraeus_data.storage import BackendMode


class AstraeusVectorEngine:
    """
    Qdrant vector engine for astronomical transient embeddings.
    Supports strict remote Qdrant cluster execution or embedded in-memory execution.
    Prohibits silent backend fallback in STRICT_LIVE mode.
    """
    def __init__(
        self,
        location: str = ":memory:",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "astraeus_candidates_v1",
        vector_dim: int = 24,
        backend_mode: BackendMode = BackendMode.LOCAL_EMBEDDED,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.backend_mode = backend_mode
        self.url = url

        if backend_mode == BackendMode.STRICT_LIVE:
            if not url:
                raise LiveBackendUnavailableError(
                    "Qdrant remote URL must be explicitly supplied in STRICT_LIVE mode. "
                    "Silent fallback is prohibited."
                )
            try:
                self.client = QdrantClient(url=url, api_key=api_key, timeout=5.0)
                # Verify live connectivity
                self.client.get_collections()
                self.mode = "qdrant_live"
            except Exception as exc:
                raise LiveBackendUnavailableError(
                    f"STRICT_LIVE mode: Failed to connect to live Qdrant server at '{url}': {exc}. "
                    "Silent fallback is strictly prohibited."
                ) from exc
        else:
            self.mode = "in_memory"
            self.client = QdrantClient(location)

        self._init_collection()

    @property
    def is_live_backend(self) -> bool:
        """Returns True if connected to an actual remote Qdrant service."""
        return self.mode == "qdrant_live"

    def _init_collection(self) -> None:
        """Initializes or resets the target collection."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    @staticmethod
    def extract_astronomical_features_24d(row: pd.Series | Dict[str, Any]) -> List[float]:
        """
        Derives a normalized 24-dimensional physical vector embedding from candidate attributes
        reflecting celestial geometry, observational cadence, baseline duration, and astrophysical classification.
        """
        ra_rad = math.radians(float(row.get("ra_deg", 0.0)))
        dec_rad = math.radians(float(row.get("dec_deg", 0.0)))

        # 1-3: Cartesian celestial unit vector
        cx = math.cos(dec_rad) * math.cos(ra_rad)
        cy = math.cos(dec_rad) * math.sin(ra_rad)
        cz = math.sin(dec_rad)

        # 4-5: Galactic coordinate projection proxy
        # Standard IAU J2000 Galactic North Pole: ra=192.85948 deg, dec=27.12825 deg
        glon_proxy = math.cos(dec_rad) * math.cos(ra_rad - math.radians(192.86))
        glat_proxy = math.sin(dec_rad) * 0.456 + math.cos(dec_rad) * 0.890 * math.cos(ra_rad)

        # 6-8: Observational cadence
        n_det = float(row.get("n_alerce_det", 0))
        n_ul = float(row.get("n_alerce_ul", 0))
        n_dr24 = float(row.get("n_dr24_clean", 0))
        log_ndet = math.log10(max(0.0, n_det) + 1.0) / 4.0
        log_nul = math.log10(max(0.0, n_ul) + 1.0) / 4.0
        log_ndr24 = math.log10(max(0.0, n_dr24) + 1.0) / 5.0

        # 9-10: Baseline duration
        first_mjd = float(row.get("first_mjd", 58000.0)) if pd.notna(row.get("first_mjd")) else 58000.0
        last_mjd = float(row.get("last_mjd", 58000.0)) if pd.notna(row.get("last_mjd")) else 58000.0
        dt_days = max(0.0, last_mjd - first_mjd)
        norm_dt = min(dt_days, 3650.0) / 3650.0
        log_dt = math.log10(dt_days + 0.1) / 4.0

        # 10-14: Survey coverage & accessibility flags (purely observational infrastructure)
        has_msip = 1.0 if row.get("has_msip_alert") else 0.0
        has_dr24 = 1.0 if row.get("has_dr24_lc") else 0.0
        has_tns = 1.0 if row.get("has_independent_tns") else 0.0
        forced_flag = 1.0 if row.get("has_forced_photometry") else 0.0
        has_alerce = 1.0 if row.get("has_alerce_access") else 0.0

        # 15-17: Photometric dynamics (cadence & ratios)
        det_ratio = n_det / max(1.0, (n_det + n_ul))
        clean_ratio = n_dr24 / max(1.0, (n_det + n_dr24))
        temporal_span = dt_days / max(1.0, n_det) if n_det > 0 else 0.0
        norm_span = min(temporal_span, 100.0) / 100.0

        # 18-19: Host counterpart association & proximity proxy (observable crossmatch)
        has_cp = 1.0 if pd.notna(row.get("host_counterpart_id")) else 0.0
        cp_proxy = 0.5 if has_cp > 0 else -0.5

        # 20-21: Spatial HEALPix harmonic projection (order 32)
        hp = float(row.get("healpix_nside32", 0))
        hp_sin = math.sin(2.0 * math.pi * hp / 12288.0)
        hp_cos = math.cos(2.0 * math.pi * hp / 12288.0)

        # 22-23: Solar avoidance observing season annual phase & total photometric depth
        annual_phase = ((first_mjd - 58000.0) % 365.25) / 365.25
        log_total_obs = math.log10(n_det + n_ul + n_dr24 + 1.0) / 5.0

        # Construct strictly target-leakage-free 24-dim physical observable vector
        raw_vec = np.array([
            cx, cy, cz, glon_proxy, glat_proxy,
            log_ndet, log_nul, log_ndr24, norm_dt, log_dt,
            has_msip, has_dr24, has_tns, forced_flag, has_alerce,
            det_ratio, clean_ratio, norm_span, has_cp, cp_proxy,
            hp_sin, hp_cos, annual_phase, log_total_obs
        ], dtype=float)

        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            raw_vec = raw_vec / norm

        return raw_vec.tolist()

    def ingest_candidates_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """
        Derives physical 24-dim embeddings from candidate metadata and ingests into Qdrant.
        """
        start_time = time.perf_counter()
        total_points = len(df)
        points: List[PointStruct] = []

        for idx, row in df.iterrows():
            cand_id = str(row["candidate_id"])
            vector = self.extract_astronomical_features_24d(row)

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

            points.append(PointStruct(id=int(idx + 1), vector=vector, payload=payload))

        # Batch upsert
        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=chunk)

        duration = time.perf_counter() - start_time
        return {
            "collection_name": self.collection_name,
            "mode": self.mode,
            "is_live_backend": self.is_live_backend,
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
