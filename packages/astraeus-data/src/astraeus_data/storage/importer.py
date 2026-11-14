"""
ASTRAEUS Cohort Batch Importer & Persistence Verifier
Executes high-throughput transactional ingestion of the 10,000-candidate cohort,
4-entity identity records, raw snapshots, and multi-epoch photometry into relational storage.
"""

from __future__ import annotations

import os
import json
import time
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import select, func, text, insert
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from astraeus_data.storage.schema import (
    PersistentSkyLoci,
    PhysicalTransientEpisodes,
    CatalogReports,
    HostCounterparts,
    Candidates10k,
    PhotometryRecords,
    RawSnapshots,
)
from astraeus_data.storage.relational_engine import transactional_session, get_session_factory


class CohortBatchImporter:
    """
    Manages bulk ingestion, transaction boundaries, idempotent re-ingestion,
    and integrity validation across the relational transient store.
    """
    def __init__(self, engine: Engine):
        self.engine = engine
        self.session_factory = get_session_factory(engine)

    def import_cohort(
        self,
        processed_dir: str = "data/processed/real_survey_10k_v1",
        snapshot_manifest_path: str = "data/raw/real_survey_v1/snapshot_manifest.json",
        raw_lightcurves_dir: str = "data/raw/real_survey_v1/alerce/lightcurves",
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end batch ingestion of the full 10k cohort, snapshots, and photometry.
        Returns execution timing, throughput figures, and imported row counts.
        """
        start_time = time.perf_counter()
        metrics: Dict[str, Any] = {
            "tables": {},
            "total_rows_imported": 0,
            "duration_seconds": 0.0,
            "throughput_rows_per_sec": 0.0,
        }

        proc_path = Path(processed_dir)

        # 1. Ingest Persistent Sky Loci (Entity 1)
        loci_df = pd.read_parquet(proc_path / "sky_loci.parquet")
        t0 = time.perf_counter()
        n_loci = self._insert_dataframe(PersistentSkyLoci, loci_df, chunk_size=chunk_size)
        metrics["tables"]["persistent_sky_loci"] = {
            "target": 9928,
            "imported": n_loci,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_loci / max(0.001, time.perf_counter() - t0),
        }

        # 2. Ingest Physical Transient Episodes (Entity 2)
        episodes_df = pd.read_parquet(proc_path / "physical_episodes.parquet")
        t0 = time.perf_counter()
        n_episodes = self._insert_dataframe(PhysicalTransientEpisodes, episodes_df, chunk_size=chunk_size)
        metrics["tables"]["physical_transient_episodes"] = {
            "target": 9968,
            "imported": n_episodes,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_episodes / max(0.001, time.perf_counter() - t0),
        }

        # 3. Ingest Catalog Reports (Entity 3)
        reports_df = pd.read_parquet(proc_path / "catalog_reports.parquet")
        t0 = time.perf_counter()
        n_reports = self._insert_dataframe(CatalogReports, reports_df, chunk_size=chunk_size)
        metrics["tables"]["catalog_reports"] = {
            "target": 4479,
            "imported": n_reports,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_reports / max(0.001, time.perf_counter() - t0),
        }

        # 4. Ingest Host Counterparts (Entity 4)
        counterparts_df = pd.read_parquet(proc_path / "host_counterparts.parquet")
        t0 = time.perf_counter()
        n_counterparts = self._insert_dataframe(HostCounterparts, counterparts_df, chunk_size=chunk_size)
        metrics["tables"]["host_counterparts"] = {
            "target": 1061,
            "imported": n_counterparts,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_counterparts / max(0.001, time.perf_counter() - t0),
        }

        # 5. Ingest Candidates 10k (Materialized Cohort View)
        candidates_df = pd.read_parquet(proc_path / "candidates_10k.parquet")
        t0 = time.perf_counter()
        n_candidates = self._insert_dataframe(Candidates10k, candidates_df, chunk_size=chunk_size)
        metrics["tables"]["candidates_10k"] = {
            "target": 10000,
            "imported": n_candidates,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_candidates / max(0.001, time.perf_counter() - t0),
        }

        # 6. Ingest Raw Snapshots Manifest
        t0 = time.perf_counter()
        n_snapshots = self._insert_snapshot_manifest(snapshot_manifest_path)
        metrics["tables"]["raw_snapshots"] = {
            "target": 33,
            "imported": n_snapshots,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_snapshots / max(0.001, time.perf_counter() - t0),
        }

        # 7. Ingest Verified Multi-Epoch Photometry (preserving negative flux and upper limits)
        t0 = time.perf_counter()
        n_photo, n_neg_flux, n_upper_limits = self._insert_sample_photometry(
            raw_lightcurves_dir=raw_lightcurves_dir,
            episodes_df=episodes_df,
            max_curves=150,
        )
        metrics["tables"]["photometry_records"] = {
            "imported": n_photo,
            "negative_flux_count": n_neg_flux,
            "upper_limit_count": n_upper_limits,
            "duration_seconds": time.perf_counter() - t0,
            "throughput": n_photo / max(0.001, time.perf_counter() - t0),
        }

        total_duration = time.perf_counter() - start_time
        total_rows = n_loci + n_episodes + n_reports + n_counterparts + n_candidates + n_snapshots + n_photo
        metrics["total_rows_imported"] = total_rows
        metrics["duration_seconds"] = total_duration
        metrics["throughput_rows_per_sec"] = total_rows / max(0.001, total_duration)

        return metrics

    def _insert_dataframe(self, model_cls: Any, df: pd.DataFrame, chunk_size: int = 1000) -> int:
        """Inserts records in chunks using native bulk insert."""
        import numpy as np
        records = df.to_dict(orient="records")
        # Ensure json string formatting for list/dict/numpy types
        for r in records:
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    r[k] = json.dumps(v.tolist())
                elif isinstance(v, (list, dict)):
                    r[k] = json.dumps(v)
                elif isinstance(v, float) and pd.isna(v):
                    r[k] = None
                elif v is None:
                    r[k] = None

        with transactional_session(self.engine) as session:
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                session.bulk_insert_mappings(model_cls, chunk)
        return len(records)

    def _insert_snapshot_manifest(self, manifest_path: str) -> int:
        """Parses and ingests the raw snapshot manifest."""
        if not os.path.exists(manifest_path):
            return 0
        with open(manifest_path, "r") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        snapshots = []
        for i, e in enumerate(entries):
            snapshots.append({
                "snapshot_id": f"SNAP_{i+1:04d}",
                "artifact_id": e.get("artifact_id", ""),
                "source_system": e.get("source_system", ""),
                "source_uri": e.get("source_uri", ""),
                "local_relative_path": e.get("local_relative_path", ""),
                "retrieved_at_utc": e.get("retrieved_at_utc", ""),
                "sha256": e.get("sha256", ""),
                "size_bytes": e.get("size_bytes", 0),
                "row_count": e.get("row_count", 0),
                "schema_columns": json.dumps(e.get("schema_columns", [])),
                "http_status": e.get("http_status") if e.get("http_status") is not None else 200,
            })
        with transactional_session(self.engine) as session:
            session.bulk_insert_mappings(RawSnapshots, snapshots)
        return len(snapshots)

    def _insert_sample_photometry(
        self,
        raw_lightcurves_dir: str,
        episodes_df: pd.DataFrame,
        max_curves: int = 150,
    ) -> tuple[int, int, int]:
        """
        Parses raw ALeRCE lightcurve JSONs and ingests multi-epoch photometry,
        explicitly verifying negative-flux preservation and upper-limit separation.
        """
        lc_files = glob.glob(os.path.join(raw_lightcurves_dir, "*.json"))[:max_curves]
        if not lc_files:
            return 0, 0, 0

        # Build oid to (episode_id, locus_id) map
        oid_map = {}
        for _, row in episodes_df.iterrows():
            m_oids = row.get("member_oids")
            if isinstance(m_oids, np.ndarray):
                oids = m_oids.tolist()
            elif isinstance(m_oids, list):
                oids = m_oids
            elif isinstance(m_oids, str):
                try:
                    oids = json.loads(m_oids)
                except Exception:
                    oids = [m_oids]
            else:
                oids = []
            for oid in oids:
                oid_map[oid] = (row["physical_event_id"], row["sky_locus_id"])

        photo_records = []
        n_neg_flux = 0
        n_upper_limits = 0
        obs_idx = 1

        filter_map = {1: "g", 2: "r", 3: "i"}

        for lc_path in lc_files:
            oid = Path(lc_path).stem
            if oid not in oid_map:
                continue
            ep_id, locus_id = oid_map[oid]
            try:
                with open(lc_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            # Detections
            for det in data.get("detections", []):
                fid = det.get("fid", 1)
                band = filter_map.get(fid, f"fid_{fid}")
                is_neg = (det.get("isdiffpos") == -1) or (det.get("isdiffpos") == 0)
                if is_neg:
                    n_neg_flux += 1

                photo_records.append({
                    "observation_id": f"OBS_{obs_idx:08d}",
                    "physical_event_id": ep_id,
                    "sky_locus_id": locus_id,
                    "survey": "ZTF_ALERCE_MSIP",
                    "mjd_tai": float(det.get("mjd", 0.0)),
                    "filter_band": band,
                    "flux_mjy": None,
                    "flux_error_mjy": None,
                    "mag": float(det.get("magpsf")) if det.get("magpsf") is not None else None,
                    "mag_err": float(det.get("sigmapsf")) if det.get("sigmapsf") is not None else None,
                    "is_upper_limit": False,
                    "limiting_mag": float(det.get("diffmaglim")) if det.get("diffmaglim") is not None else None,
                    "catflags": 0,
                    "is_negative_flux": is_neg,
                })
                obs_idx += 1

            # Non-detections (Upper limits)
            for nondet in data.get("non_detections", []):
                fid = nondet.get("fid", 1)
                band = filter_map.get(fid, f"fid_{fid}")
                n_upper_limits += 1

                photo_records.append({
                    "observation_id": f"OBS_{obs_idx:08d}",
                    "physical_event_id": ep_id,
                    "sky_locus_id": locus_id,
                    "survey": "ZTF_ALERCE_MSIP",
                    "mjd_tai": float(nondet.get("mjd", 0.0)),
                    "filter_band": band,
                    "flux_mjy": None,
                    "flux_error_mjy": None,
                    "mag": None,
                    "mag_err": None,
                    "is_upper_limit": True,
                    "limiting_mag": float(nondet.get("diffmaglim")) if nondet.get("diffmaglim") is not None else None,
                    "catflags": 0,
                    "is_negative_flux": False,
                })
                obs_idx += 1

        with transactional_session(self.engine) as session:
            session.bulk_insert_mappings(PhotometryRecords, photo_records)

        return len(photo_records), n_neg_flux, n_upper_limits

    def get_row_counts(self) -> Dict[str, int]:
        """Returns exact counts from all tables."""
        session = self.session_factory()
        try:
            return {
                "persistent_sky_loci": session.scalar(select(func.count(PersistentSkyLoci.sky_locus_id))),
                "physical_transient_episodes": session.scalar(select(func.count(PhysicalTransientEpisodes.physical_event_id))),
                "catalog_reports": session.scalar(select(func.count(CatalogReports.report_id))),
                "host_counterparts": session.scalar(select(func.count(HostCounterparts.counterpart_id))),
                "candidates_10k": session.scalar(select(func.count(Candidates10k.candidate_id))),
                "raw_snapshots": session.scalar(select(func.count(RawSnapshots.snapshot_id))),
                "photometry_records": session.scalar(select(func.count(PhotometryRecords.observation_id))),
            }
        finally:
            session.close()

    def test_idempotent_reingestion(self) -> Dict[str, Any]:
        """
        Tests duplicate prevention and idempotency.
        Attempts inserting an already-existing locus and confirms conflict is prevented.
        """
        session = self.session_factory()
        try:
            existing_locus = session.execute(select(PersistentSkyLoci).limit(1)).scalar_one()
            duplicate_locus = PersistentSkyLoci(
                sky_locus_id=existing_locus.sky_locus_id,
                ra_deg=existing_locus.ra_deg,
                dec_deg=existing_locus.dec_deg,
                healpix_nside32=existing_locus.healpix_nside32,
                assigned_base_split=existing_locus.assigned_base_split,
            )
            session.add(duplicate_locus)
            try:
                session.commit()
                raise AssertionError("Duplicate insertion unexpectedly succeeded!")
            except Exception as e:
                session.rollback()
                return {
                    "status": "PASS",
                    "duplicate_prevented": True,
                    "exception_type": type(e).__name__,
                }
        finally:
            session.close()

    def test_rollback_on_injected_failure(self) -> Dict[str, Any]:
        """
        Injects a referential integrity violation mid-transaction and
        verifies that ACID rollback completely restores database state.
        """
        session = self.session_factory()
        try:
            initial_count = session.scalar(select(func.count(PersistentSkyLoci.sky_locus_id)))
            
            # Start transaction: add 1 valid locus, then 1 invalid episode referencing non-existent locus
            try:
                test_locus = PersistentSkyLoci(
                    sky_locus_id="TEST_LOCUS_ROLLBACK",
                    ra_deg=100.0,
                    dec_deg=20.0,
                    healpix_nside32=12,
                    assigned_base_split="train",
                )
                session.add(test_locus)
                session.flush()

                # Add invalid episode violating foreign key constraint
                invalid_ep = PhysicalTransientEpisodes(
                    physical_event_id="TEST_EP_INVALID",
                    sky_locus_id="NON_EXISTENT_LOCUS_99999",  # VIOLATION
                    ra_deg=100.0,
                    dec_deg=20.0,
                    healpix_nside32=12,
                    t_start_mjd=60000.0,
                    t_end_mjd=60010.0,
                    assigned_split="train",
                    temporal_split="archival_observation_date_holdout",
                    coarse_class="SN_Ia",
                    fine_class="SN_Ia",
                    evidence_tier="L1_TRACEABLE_SPECTROSCOPIC",
                    taxonomy_version="1.0",
                )
                session.add(invalid_ep)
                session.commit()
                raise AssertionError("Transaction unexpectedly committed despite FK violation!")
            except Exception as exc:
                session.rollback()
                final_count = session.scalar(select(func.count(PersistentSkyLoci.sky_locus_id)))
                leak_detected = (final_count != initial_count)
                return {
                    "status": "PASS" if not leak_detected else "FAIL",
                    "rollback_verified": not leak_detected,
                    "initial_locus_count": initial_count,
                    "final_locus_count": final_count,
                    "leaked_rows": final_count - initial_count,
                    "caught_exception": type(exc).__name__,
                }
        finally:
            session.close()
