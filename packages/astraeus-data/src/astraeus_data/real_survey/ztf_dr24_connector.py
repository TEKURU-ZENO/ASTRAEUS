"""
ASTRAEUS ZTF DR24 Enhanced HATS Parquet Connector.

Resolves adaptive HEALPix partition directories (`Norder`, `Dir`, `Npix/part0.snappy.parquet`)
from Caltech/IPAC-IRSA's public ZTF DR24 S3 HATS collection, downloads & verifies
snapshot partitions with SHA-256 provenance, and converts ZTF DR24 lightcurve arrays
(`hmjd`, `mag`, `magerr`, `catflags`) into canonical `ObservationRecord` sequences.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy_healpix import HEALPix
import httpx
import pandas as pd
import pyarrow.parquet as pq

from astraeus_data.corpus.sequence import (
    ObservationRecord,
    ObservationType,
    QualitySeverity,
)
from astraeus_data.real_survey.manifest import (
    RetrievalManifestEntry,
    compute_file_sha256,
    current_git_commit,
)

ZTF_DR24_LC_HATS_BASE = (
    "https://ipac-irsa-ztf.s3.us-east-1.amazonaws.com/ztf/enhanced/dr24/lc/hats/ztf_dr24_lc-hats"
)
ZTF_DR24_OBJECTS_HATS_BASE = (
    "https://ipac-irsa-ztf.s3.us-east-1.amazonaws.com/ztf/enhanced/dr24/objects/hats/ztf_dr24_objects-hats"
)
ZTF_DR24_FILTER_MAP: Dict[int, str] = {1: "g", 2: "r", 3: "i"}
CATFLAG_CLOUD_OR_MOON_BIT = 32768  # Bit 15 in ZTF DR24 release notes


def skycoord_to_healpix9(ra_deg: float, dec_deg: float) -> int:
    """Computes nested HEALPix order-9 pixel index matching HATS `_healpix_9`."""
    hp9 = HEALPix(nside=2**9, order="nested", frame="icrs")
    coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    return int(hp9.skycoord_to_healpix(coord))


def resolve_partition_from_healpix9(
    healpix9_idx: int,
    partition_table: pd.DataFrame,
) -> Tuple[int, int, int]:
    """
    Finds the HATS `(Norder, Dir, Npix)` tile covering `healpix9_idx` using nested order arithmetic.
    Supports partition DataFrames with columns `Norder`, `Dir`, `Npix`.
    """
    for order in sorted(partition_table["Norder"].unique()):
        shift = 2 * (9 - int(order))
        candidate_npix = int(healpix9_idx) >> shift if shift >= 0 else int(healpix9_idx) << (-shift)
        matched = partition_table[
            (partition_table["Norder"] == order) & (partition_table["Npix"] == candidate_npix)
        ]
        if not matched.empty:
            row = matched.iloc[0]
            return int(row["Norder"]), int(row["Dir"]), int(row["Npix"])
    raise KeyError(f"No HATS partition found covering _healpix_9={healpix9_idx}")


def hats_partition_relative_url(norder: int, dir_idx: int, npix: int) -> str:
    """Returns canonical ZTF DR24 HATS Parquet path within the dataset prefix."""
    return f"dataset/Norder={int(norder)}/Dir={int(dir_idx)}/Npix={int(npix)}/part0.snappy.parquet"


def classify_ztf_dr24_catflag(catflags: int) -> QualitySeverity:
    """
    Maps ZTF DR24 bitwise `catflags` to ASTRAEUS QualitySeverity.
    Bit 15 (32768) marks epochs impacted by clouds and/or the moon (FATAL per IRSA schema).
    Any other non-zero bitwise flag is tagged DEGRADED.
    """
    flag = int(catflags)
    if (flag & CATFLAG_CLOUD_OR_MOON_BIT) != 0:
        return QualitySeverity.FATAL
    if flag != 0:
        return QualitySeverity.DEGRADED
    return QualitySeverity.INFORMATIONAL


def dr24_row_to_observation_records(
    row: Dict[str, Any],
    exclude_fatal: bool = True,
) -> List[ObservationRecord]:
    """
    Converts a single ZTF DR24 HATS lightcurve row (`objectid`, `filterid`, `objra`, `objdec`,
    and `lightcurve` struct/arrays `hmjd`, `mag`, `magerr`, `catflags`) into canonical
    `ObservationRecord` entries.

    Note: ZTF DR24 is an archival static release lacking operational alert timestamps,
    so `has_operational_pub_time=False` and `t_pub_utc=None` to prevent leakage into
    operational causal-cutoff evaluations.
    """
    oid_str = f"DR24_{int(row['objectid'])}"
    fid = int(row["filterid"])
    passband = ZTF_DR24_FILTER_MAP.get(fid)
    if passband is None:
        return []

    lc = row.get("lightcurve") or row
    hmjd_list = lc["hmjd"] if isinstance(lc, dict) else row["lightcurve.hmjd"]
    mag_list = lc["mag"] if isinstance(lc, dict) else row["lightcurve.mag"]
    magerr_list = lc["magerr"] if isinstance(lc, dict) else row["lightcurve.magerr"]
    catflags_list = lc["catflags"] if isinstance(lc, dict) else row["lightcurve.catflags"]

    records: List[ObservationRecord] = []
    for idx, (hmjd, mag, magerr, cflag) in enumerate(
        zip(hmjd_list, mag_list, magerr_list, catflags_list)
    ):
        if mag is None or magerr is None or float(magerr) <= 0.0:
            continue
        severity = classify_ztf_dr24_catflag(int(cflag))
        if exclude_fatal and severity == QualitySeverity.FATAL:
            continue

        flux_nmgy, sigma_nmgy = ObservationRecord.ab_magnitude_to_nanomaggies(
            float(mag), float(magerr)
        )
        obs_id = f"{oid_str}_{passband}_{idx}_{float(hmjd):.5f}"
        prov_hash = hashlib.sha256(
            f"{oid_str}|{obs_id}|{float(hmjd):.5f}|{flux_nmgy:.6e}|{int(cflag)}".encode()
        ).hexdigest()

        records.append(
            ObservationRecord(
                object_id=oid_str,
                observation_id=obs_id,
                t_obs_bmjd=float(hmjd),
                t_pub_utc=None,
                t_ingest_utc=None,
                has_operational_pub_time=False,
                passband=passband,
                survey_id="ZTF_DR24_REF_PSF",
                obs_type=ObservationType.CATALOG_DETECTION,
                measured_flux_nmgy=flux_nmgy,
                measurement_sigma_nmgy=max(sigma_nmgy, 1e-6),
                effective_measurement_sigma_nmgy=max(sigma_nmgy, 1e-6),
                quality_flag=int(cflag),
                quality_severity=severity,
                provenance_hash=prov_hash,
            )
        )
    records.sort(key=lambda r: r.t_obs_bmjd)
    return records


class ZtfDr24HatsClient:
    """Caching HTTPS S3 client for ZTF DR24 Enhanced Lightcurves HATS Parquet shards."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        base_url: str = ZTF_DR24_LC_HATS_BASE,
        timeout_s: float = 60.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def fetch_partition_info(self, force_refresh: bool = False) -> pd.DataFrame:
        """Downloads or reads cached `partition_info.csv` for the DR24 HATS collection."""
        out = self.cache_dir / "partition_info.csv"
        if not out.exists() or force_refresh:
            with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                r = client.get(f"{self.base_url}/partition_info.csv")
                r.raise_for_status()
                out.write_bytes(r.content)
        return pd.read_csv(out)

    def fetch_partition_parquet(
        self,
        norder: int,
        dir_idx: int,
        npix: int,
        force_refresh: bool = False,
        manifest_root: Optional[Path] = None,
    ) -> Tuple[Path, RetrievalManifestEntry]:
        """Streams a single HEALPix HATS Parquet shard to disk and computes SHA-256."""
        rel_url = hats_partition_relative_url(norder, dir_idx, npix)
        local_path = self.cache_dir / f"Norder={norder}_Dir={dir_idx}_Npix={npix}.parquet"
        http_status: Optional[int] = None
        etag: Optional[str] = None

        if not local_path.exists() or force_refresh:
            tmp = local_path.with_suffix(".parquet.tmp")
            with httpx.stream(
                "GET",
                f"{self.base_url}/{rel_url}",
                timeout=self.timeout_s,
                follow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                http_status = resp.status_code
                etag = resp.headers.get("etag")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=131072):
                        fh.write(chunk)
            tmp.replace(local_path)

        pf = pq.ParquetFile(local_path)
        rel_path = (
            str(local_path.relative_to(manifest_root)) if manifest_root else local_path.name
        )
        entry = RetrievalManifestEntry(
            artifact_id=f"ztf_dr24_lc:Norder={norder}:Npix={npix}",
            source_system="ZTF_DR24_HATS",
            source_uri=f"{self.base_url}/{rel_url}",
            local_relative_path=rel_path,
            retrieved_at_utc=datetime.fromtimestamp(
                local_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            sha256=compute_file_sha256(local_path),
            size_bytes=local_path.stat().st_size,
            row_count=pf.metadata.num_rows,
            schema_columns=pf.schema_arrow.names,
            http_status=http_status,
            etag=etag,
            query_params={"Norder": norder, "Dir": dir_idx, "Npix": npix},
            git_commit=current_git_commit(),
        )
        return local_path, entry
