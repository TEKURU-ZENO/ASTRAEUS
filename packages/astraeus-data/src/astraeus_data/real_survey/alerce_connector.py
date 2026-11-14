"""
ASTRAEUS ALeRCE ZTF Alert Broker Connector.

Retrieves real ZTF alert object metadata, detections (difference-image PSF
photometry with signed `isdiffpos`), and non-detection 5-sigma upper limits
(`diffmaglim`) from `https://api.alerce.online/ztf/v1`, caches raw JSON
snapshots locally with SHA-256 provenance, and converts epochs to canonical
ASTRAEUS `ObservationRecord` + `TemporalSequence`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from astropy.time import Time
import httpx

from astraeus_data.corpus.sequence import (
    NoiseScaleSource,
    ObservationRecord,
    ObservationType,
    QualitySeverity,
    SequenceViewType,
    TemporalSequence,
)
from astraeus_data.corpus.time_conversion import compute_bmjd_tdb
from astraeus_data.real_survey.manifest import (
    RetrievalManifestEntry,
    compute_file_sha256,
    current_git_commit,
)
from astraeus_data.real_survey.timestamps import (
    AvailabilityStatus,
    build_observation_timestamps,
)

ALERCE_V1_BASE = "https://api.alerce.online/ztf/v1"
ZTF_FILTER_MAP: Dict[int, str] = {1: "g", 2: "r", 3: "i"}


def mjd_utc_to_iso_utc(mjd_utc: float) -> str:
    """Converts a scalar UTC MJD to an ISO-8601 UTC timestamp string."""
    dt = Time(float(mjd_utc), format="mjd", scale="utc").to_datetime(timezone=timezone.utc)
    return dt.isoformat()


def alerce_lightcurve_to_records(
    oid: str,
    ra_deg: float,
    dec_deg: float,
    detections: List[Dict[str, Any]],
    non_detections: Optional[List[Dict[str, Any]]] = None,
    ingested_at_utc: Optional[str] = None,
    min_rb_score: Optional[float] = None,
    known_pub_times_by_candid: Optional[Dict[str, str]] = None,
    t_retrieval_utc: Optional[str] = None,
) -> List[ObservationRecord]:
    """
    Converts raw ALeRCE ZTF detections and non-detections into canonical
    `ObservationRecord` entries in nanomaggies (Z0 = 22.500) and BMJD(TDB).

    P0.3 timestamp invariant: `mjd` in ALeRCE REST v1 is `t_observation`, NOT broker
    publication time `t_publication`. Unless `known_pub_times_by_candid` explicitly
    supplies an independently observed operational publication timestamp, `t_pub_utc`
    is preserved as `None` with `has_operational_pub_time=False`
    (`AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN`) rather than fabricated.
    """
    records: List[ObservationRecord] = []
    seen_obs_ids: set[str] = set()

    for idx, det in enumerate(detections or []):
        mjd_utc = det.get("mjd")
        magpsf = det.get("magpsf")
        sigmapsf = det.get("sigmapsf")
        fid = det.get("fid")
        if mjd_utc is None or magpsf is None or sigmapsf is None or fid not in ZTF_FILTER_MAP:
            continue
        if float(sigmapsf) <= 0.0:
            continue

        passband = ZTF_FILTER_MAP[int(fid)]
        candid = str(det.get("candid") or f"det_{idx}")
        obs_id = f"{oid}_det_{candid}"
        if obs_id in seen_obs_ids:
            continue
        seen_obs_ids.add(obs_id)

        flux_mag, sigma_nmgy = ObservationRecord.ab_magnitude_to_nanomaggies(
            float(magpsf), float(sigmapsf)
        )
        # Preserve negative subtraction flux (`isdiffpos == -1` or 'f')
        raw_isdiffpos = det.get("isdiffpos", 1)
        is_neg = str(raw_isdiffpos).strip().lower() in ("-1", "f", "false", "0")
        signed_flux_nmgy = -flux_mag if is_neg else flux_mag

        rb_val = det.get("rb")
        drb_val = det.get("drb")
        severity = QualitySeverity.INFORMATIONAL
        qflag = 0
        if min_rb_score is not None and rb_val is not None and float(rb_val) < min_rb_score:
            severity = QualitySeverity.DEGRADED
            qflag = 1
        if drb_val is not None and float(drb_val) < 0.1:
            severity = QualitySeverity.DEGRADED
            qflag = qflag | 2

        det_ra = float(det.get("ra") if det.get("ra") is not None else ra_deg)
        det_dec = float(det.get("dec") if det.get("dec") is not None else dec_deg)
        known_pub_det = (known_pub_times_by_candid or {}).get(candid)
        ts = build_observation_timestamps(
            mjd_utc=float(mjd_utc),
            ra_deg=det_ra,
            dec_deg=det_dec,
            t_pub_utc=known_pub_det,
            t_ingest_utc=ingested_at_utc,
            t_retrieval_utc=t_retrieval_utc,
            observatory="ZTF",
        )
        t_bmjd = float(ts.t_obs_bmjd_tdb)
        prov_hash = hashlib.sha256(
            f"{oid}|{obs_id}|{t_bmjd:.7f}|{passband}|{signed_flux_nmgy:.6e}".encode()
        ).hexdigest()

        records.append(
            ObservationRecord(
                object_id=oid,
                observation_id=obs_id,
                t_obs_bmjd=t_bmjd,
                t_pub_utc=ts.t_pub_utc,
                t_ingest_utc=ts.t_ingest_utc,
                has_operational_pub_time=ts.is_operationally_causal_eligible,
                passband=passband,
                survey_id="ZTF_ALERT_DIFF",
                obs_type=ObservationType.CATALOG_DETECTION,
                measured_flux_nmgy=signed_flux_nmgy,
                measurement_sigma_nmgy=max(sigma_nmgy, 1e-6),
                effective_measurement_sigma_nmgy=max(sigma_nmgy, 1e-6),
                quality_flag=qflag,
                quality_severity=severity,
                provenance_hash=prov_hash,
            )
        )

    for idx, nd in enumerate(non_detections or []):
        mjd_utc = nd.get("mjd")
        diffmaglim = nd.get("diffmaglim")
        fid = nd.get("fid")
        if mjd_utc is None or diffmaglim is None or fid not in ZTF_FILTER_MAP:
            continue
        if float(diffmaglim) <= 0.0:
            continue

        passband = ZTF_FILTER_MAP[int(fid)]
        obs_id = f"{oid}_ul_{float(mjd_utc):.6f}_{passband}_{idx}"
        if obs_id in seen_obs_ids:
            continue
        seen_obs_ids.add(obs_id)

        ul_nmgy, _ = ObservationRecord.ab_magnitude_to_nanomaggies(float(diffmaglim), 0.1)
        eff_sigma = max(ul_nmgy / 5.0, 1e-6)
        ul_lookup_key = f"ul_{int(fid)}_{float(mjd_utc):.6f}"
        known_pub_ul = (known_pub_times_by_candid or {}).get(obs_id) or (
            known_pub_times_by_candid or {}
        ).get(ul_lookup_key)
        ts_ul = build_observation_timestamps(
            mjd_utc=float(mjd_utc),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            t_pub_utc=known_pub_ul,
            t_ingest_utc=ingested_at_utc,
            t_retrieval_utc=t_retrieval_utc,
            observatory="ZTF",
        )
        t_bmjd = float(ts_ul.t_obs_bmjd_tdb)
        prov_hash = hashlib.sha256(
            f"{oid}|{obs_id}|{t_bmjd:.7f}|{passband}|UL|{ul_nmgy:.6e}".encode()
        ).hexdigest()

        records.append(
            ObservationRecord(
                object_id=oid,
                observation_id=obs_id,
                t_obs_bmjd=t_bmjd,
                t_pub_utc=ts_ul.t_pub_utc,
                t_ingest_utc=ts_ul.t_ingest_utc,
                has_operational_pub_time=ts_ul.is_operationally_causal_eligible,
                passband=passband,
                survey_id="ZTF_ALERT_DIFF",
                obs_type=ObservationType.CATALOG_UPPER_LIMIT,
                upper_limit_flux_nmgy=ul_nmgy,
                upper_limit_sigma_threshold=5.0,
                noise_scale_source=NoiseScaleSource.INFERRED_FROM_THRESHOLD,
                effective_measurement_sigma_nmgy=eff_sigma,
                quality_flag=0,
                quality_severity=QualitySeverity.INFORMATIONAL,
                provenance_hash=prov_hash,
            )
        )

    records.sort(key=lambda r: r.t_obs_bmjd)
    return records


def alerce_lightcurve_to_sequence(
    oid: str,
    ra_deg: float,
    dec_deg: float,
    payload: Dict[str, Any],
    view_type: SequenceViewType = SequenceViewType.FULL_HISTORY,
    t_cutoff_utc: Optional[str] = None,
    ingested_at_utc: Optional[str] = None,
    known_pub_times_by_candid: Optional[Dict[str, str]] = None,
    t_retrieval_utc: Optional[str] = None,
) -> TemporalSequence:
    """Builds a `TemporalSequence` from a cached or live `/objects/{oid}/lightcurve` payload."""
    records = alerce_lightcurve_to_records(
        oid=oid,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        detections=payload.get("detections") or [],
        non_detections=payload.get("non_detections") or [],
        ingested_at_utc=ingested_at_utc,
        known_pub_times_by_candid=known_pub_times_by_candid,
        t_retrieval_utc=t_retrieval_utc,
    )
    return TemporalSequence.from_records(
        object_id=oid,
        records=records,
        view_type=view_type,
        t_cutoff_utc=t_cutoff_utc,
    )


class AlerceZtfClient:
    """Thin caching client for the public ALeRCE ZTF v1 REST API."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        base_url: str = ALERCE_V1_BASE,
        timeout_s: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=self.timeout_s, follow_redirects=True)

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json(), resp.status_code

    def fetch_object_meta(
        self,
        oid: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetches or loads cached `/objects/{oid}` metadata JSON."""
        meta_dir = self.cache_dir / "objects"
        meta_dir.mkdir(parents=True, exist_ok=True)
        out_path = meta_dir / f"{oid}.json"
        if out_path.exists() and not force_refresh:
            return json.loads(out_path.read_text(encoding="utf-8"))
        payload, _ = self._get_json(f"/objects/{oid}")
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fetch_objects_page(
        self,
        snapshot_name: str,
        params: Dict[str, Any],
        force_refresh: bool = False,
        manifest_root: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], RetrievalManifestEntry]:
        """Queries `/objects/` and persists an immutable raw JSON snapshot."""
        out_path = self.cache_dir / f"{snapshot_name}.json"
        if out_path.exists() and not force_refresh:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            status = 200
            retrieved_at = datetime.fromtimestamp(
                out_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        else:
            payload, status = self._get_json("/objects/", params=params)
            out_path.write_text(json.dumps(payload), encoding="utf-8")
            retrieved_at = datetime.now(timezone.utc).isoformat()

        items = payload.get("items") or []
        rel_path = (
            str(out_path.relative_to(manifest_root)) if manifest_root else out_path.name
        )
        entry = RetrievalManifestEntry(
            artifact_id=f"alerce_objects:{snapshot_name}",
            source_system="ALERCE_ZTF_V1",
            source_uri=f"{self.base_url}/objects/",
            local_relative_path=rel_path,
            retrieved_at_utc=retrieved_at,
            sha256=compute_file_sha256(out_path),
            size_bytes=out_path.stat().st_size,
            row_count=len(items),
            schema_columns=sorted(items[0].keys()) if items else [],
            http_status=status,
            query_params=params,
            git_commit=current_git_commit(),
        )
        return payload, entry

    def fetch_lightcurve(
        self,
        oid: str,
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, Any], Path, bool]:
        """Fetches or loads cached `/objects/{oid}/lightcurve` JSON."""
        lc_dir = self.cache_dir / "lightcurves"
        lc_dir.mkdir(parents=True, exist_ok=True)
        out_path = lc_dir / f"{oid}.json"
        if out_path.exists() and not force_refresh:
            return json.loads(out_path.read_text(encoding="utf-8")), out_path, True

        payload, _ = self._get_json(f"/objects/{oid}/lightcurve")
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, out_path, False

    def cone_search(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 3.0,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Performs a live cone search against `/objects/` for TNS/ZTF reconciliation."""
        payload, _ = self._get_json(
            "/objects/",
            params={
                "ra": float(ra_deg),
                "dec": float(dec_deg),
                "radius": float(radius_arcsec),
                "page_size": int(page_size),
            },
        )
        return payload.get("items") or []
