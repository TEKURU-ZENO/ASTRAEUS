"""
ASTRAEUS Transient Name Server (TNS) Public Catalog Connector & Label Provenance Engine.

Ingests official TNS `tns_public_objects.csv.zip` snapshots, preserves raw snapshot
cryptographic provenance, separates spectroscopic classifications from unclassified `AT`
objects and non-SN contaminants, extracts ZTF/survey aliases, and computes HEALPix +
discovery MJD metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

from astropy.time import Time
import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from astraeus_core.coordinates import CelestialCoordinates
from astraeus_data.real_survey.manifest import (
    RetrievalManifestEntry,
    compute_file_sha256,
    current_git_commit,
)

TNS_PUBLIC_OBJECTS_URL = (
    "https://www.wis-tns.org/sites/default/files/tns_public_objects/tns_public_objects.csv.zip"
)
DEFAULT_TNS_MARKER = 'tns_marker{"tns_id":0,"type":"guest","name":"ASTRAEUS_Research"}'
ZTF_ID_REGEX = re.compile(r"\bZTF\d{2}[a-z]{5,}\b")
REAL_SURVEY_TAXONOMY_VERSION = "real_survey_taxonomy_v1.0"


class EvidenceTier(str, Enum):
    """P0.1/P0.4 strict hierarchical evidence provenance tiers (`L1`–`L5`)."""

    L1_TRACEABLE_SPECTROSCOPIC = "L1"
    L2_REPORTED_CLASSIFICATION_UNVERIFIED = "L2"
    L3_EXTERNAL_CATALOG = "L3"
    L4_BROKER_OR_PHOTOMETRIC = "L4"
    L5_UNCLASSIFIED = "L5"


# Backward-compatible alias preserved for callers importing LabelProvenanceTier
LabelProvenanceTier = EvidenceTier


class CoarseTransientClass(str, Enum):
    SN_IA = "SN_Ia"
    SN_CC = "SN_CC"
    SLSN = "SLSN"
    TDE = "TDE"
    CV_NOVA = "CV_Nova"
    AGN_QSO = "AGN_QSO"
    VARSTAR = "Varstar"
    NON_SN_TRANSIENT = "NON_SN_TRANSIENT"
    OTHER_EXTRAGALACTIC = "OTHER_EXTRAGALACTIC"
    DETECTION_ARTIFACT = "DETECTION_ARTIFACT"
    UNCLASSIFIED = "UNCLASSIFIED"
    UNKNOWN_OTHER = "UNKNOWN_OTHER"


# Canonical deterministic mapping from raw TNS `type` strings to coarse family.
# Unmapped strings retain their raw TNS type and fall into UNKNOWN_OTHER unless empty/AT.
TNS_COARSE_CLASS_MAP: Dict[str, CoarseTransientClass] = {
    # Thermonuclear SNe
    "SN Ia": CoarseTransientClass.SN_IA,
    "SN Ia-91T-like": CoarseTransientClass.SN_IA,
    "SN Ia-91bg-like": CoarseTransientClass.SN_IA,
    "SN Ia-CSM": CoarseTransientClass.SN_IA,
    "SN Ia-pec": CoarseTransientClass.SN_IA,
    "SN Ia-SC": CoarseTransientClass.SN_IA,
    "SN Ia-Ca-rich": CoarseTransientClass.SN_IA,
    "SN Iax[02cx-like]": CoarseTransientClass.SN_IA,
    # Core-collapse SNe
    "SN II": CoarseTransientClass.SN_CC,
    "SN IIP": CoarseTransientClass.SN_CC,
    "SN IIL": CoarseTransientClass.SN_CC,
    "SN IIn": CoarseTransientClass.SN_CC,
    "SN IIb": CoarseTransientClass.SN_CC,
    "SN II-pec": CoarseTransientClass.SN_CC,
    "SN Ib": CoarseTransientClass.SN_CC,
    "SN Ic": CoarseTransientClass.SN_CC,
    "SN Ic-BL": CoarseTransientClass.SN_CC,
    "SN Ib/c": CoarseTransientClass.SN_CC,
    "SN Ibn": CoarseTransientClass.SN_CC,
    "SN Icn": CoarseTransientClass.SN_CC,
    "SN Ib-pec": CoarseTransientClass.SN_CC,
    "SN Ic-pec": CoarseTransientClass.SN_CC,
    "SN Ib-Ca-rich": CoarseTransientClass.SN_CC,
    # Superluminous SNe
    "SLSN-I": CoarseTransientClass.SLSN,
    "SLSN-II": CoarseTransientClass.SLSN,
    "SLSN-R": CoarseTransientClass.SLSN,
    # Tidal Disruption Events
    "TDE": CoarseTransientClass.TDE,
    "TDE-H": CoarseTransientClass.TDE,
    "TDE-He": CoarseTransientClass.TDE,
    "TDE-H-He": CoarseTransientClass.TDE,
    "TDE-featureless": CoarseTransientClass.TDE,
    # Real Non-SN Contaminants & Galactic/Nuclear Variables
    "CV": CoarseTransientClass.CV_NOVA,
    "Nova": CoarseTransientClass.CV_NOVA,
    "AGN": CoarseTransientClass.AGN_QSO,
    "QSO": CoarseTransientClass.AGN_QSO,
    "Blazar": CoarseTransientClass.AGN_QSO,
    "Varstar": CoarseTransientClass.VARSTAR,
    "M dwarf": CoarseTransientClass.VARSTAR,
    "LBV": CoarseTransientClass.VARSTAR,
    "Std-star": CoarseTransientClass.VARSTAR,
    "FRB": CoarseTransientClass.NON_SN_TRANSIENT,
    # Other Extragalactic Transients
    "FBOT": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    "LFBOT": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    "KN": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    "Afterglow": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    "ILRT": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    "LRN": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    # Generic SN without subtype
    "SN": CoarseTransientClass.OTHER_EXTRAGALACTIC,
    "SN I": CoarseTransientClass.OTHER_EXTRAGALACTIC,
}


def map_tns_type_with_provenance(
    raw_type: Optional[str],
    name_prefix: Optional[str] = None,
    class_bibcodes: Optional[str] = None,
    has_redshift: Optional[bool] = None,
) -> Tuple[CoarseTransientClass, str, EvidenceTier, Optional[bool]]:
    """
    Maps a TNS `type` string + classification reference provenance into
    `(coarse_class, normalized_fine_type, evidence_tier, is_supernova)`.

    P0.1 critical invariants:
    - Blank / 'AT' / 'Other' NEVER receive a positive or negative SN label
      (`is_supernova=None`, `EvidenceTier.L5_UNCLASSIFIED`), EVEN if host `redshift`
      or an unlinked `Class_ADS_bibcodes` is present.
    - Host redshift alone NEVER promotes an object into `L1` or `L2`.
    - `L1_TRACEABLE_SPECTROSCOPIC` requires BOTH a valid classified `type` AND a traceable
      classification reference (`Class_ADS_bibcodes` non-empty in TNS public dump).
    - Classified TNS entries lacking traceable `Class_ADS_bibcodes` receive
      `L2_REPORTED_CLASSIFICATION_UNVERIFIED`.
    """
    cleaned = "" if raw_type is None or pd.isna(raw_type) else str(raw_type).strip()
    if not cleaned or cleaned.upper() in ("AT", "NONE", "NAN", "NULL", "?"):
        return (
            CoarseTransientClass.UNCLASSIFIED,
            "UNCLASSIFIED",
            EvidenceTier.L5_UNCLASSIFIED,
            None,
        )
    if cleaned == "Other":
        return (
            CoarseTransientClass.UNKNOWN_OTHER,
            "Other",
            EvidenceTier.L5_UNCLASSIFIED,
            None,
        )

    coarse = TNS_COARSE_CLASS_MAP.get(cleaned)
    if coarse is None:
        if cleaned.startswith("SN Ia") or cleaned.startswith("SN Iax"):
            coarse = CoarseTransientClass.SN_IA
        elif cleaned.startswith("SN I"):
            coarse = CoarseTransientClass.SN_CC
        elif cleaned.startswith("SLSN"):
            coarse = CoarseTransientClass.SLSN
        elif cleaned.startswith("TDE"):
            coarse = CoarseTransientClass.TDE
        else:
            coarse = CoarseTransientClass.UNKNOWN_OTHER

    if coarse in (CoarseTransientClass.SN_IA, CoarseTransientClass.SN_CC, CoarseTransientClass.SLSN):
        is_sn: Optional[bool] = True
    elif coarse in (
        CoarseTransientClass.TDE,
        CoarseTransientClass.CV_NOVA,
        CoarseTransientClass.AGN_QSO,
        CoarseTransientClass.VARSTAR,
        CoarseTransientClass.NON_SN_TRANSIENT,
    ):
        is_sn = False
    elif cleaned in ("SN", "SN I"):
        is_sn = True
    elif coarse == CoarseTransientClass.OTHER_EXTRAGALACTIC:
        is_sn = False
    else:
        is_sn = None

    bib_clean = "" if class_bibcodes is None or pd.isna(class_bibcodes) else str(class_bibcodes).strip()
    has_traceable_bib = bool(bib_clean) and bib_clean.lower() not in ("nan", "none", "null")
    tier = (
        EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC
        if has_traceable_bib
        else EvidenceTier.L2_REPORTED_CLASSIFICATION_UNVERIFIED
    )
    return coarse, cleaned, tier, is_sn


def extract_ztf_ids(internal_names: Optional[str]) -> List[str]:
    """Extracts unique ZTF alert object IDs in deterministic order from TNS `internal_names`."""
    if internal_names is None or pd.isna(internal_names):
        return []
    seen: set[str] = set()
    ordered: List[str] = []
    for match in ZTF_ID_REGEX.findall(str(internal_names)):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def parse_tns_discovery_mjd(discoverydate: Optional[str]) -> Optional[float]:
    """Parses TNS `discoverydate` ('YYYY-MM-DD HH:MM:SS(.fff)') into UTC MJD."""
    if discoverydate is None or pd.isna(discoverydate):
        return None
    text = str(discoverydate).strip()
    if not text:
        return None
    try:
        iso = text.replace(" ", "T")
        return float(Time(iso, format="isot", scale="utc").mjd)
    except Exception:
        return None


class TnsObjectRecord(BaseModel):
    """Normalized, immutable TNS catalog record with L1–L5 label provenance & ZTF aliases."""

    model_config = ConfigDict(frozen=True)

    objid: int
    tns_name: str
    name_prefix: str
    ra_deg: float
    dec_deg: float
    healpix_nside32: int
    discovery_date_utc: Optional[str] = None
    discovery_mjd_utc: Optional[float] = None
    discovery_mag: Optional[float] = None
    discovery_filter: Optional[str] = None
    redshift: Optional[float] = None
    raw_type: Optional[str] = None
    coarse_class: CoarseTransientClass
    fine_class: str
    evidence_tier: EvidenceTier
    label_provenance: EvidenceTier
    taxonomy_version: str = REAL_SURVEY_TAXONOMY_VERSION
    is_supernova: Optional[bool] = None
    has_class_bibcode: bool = False
    has_host_redshift_only: bool = False
    reporting_group: Optional[str] = None
    source_group: Optional[str] = None
    internal_names_raw: Optional[str] = None
    ztf_ids: List[str] = Field(default_factory=list)
    class_bibcodes: Optional[str] = None
    time_received_utc: Optional[str] = None
    creation_date_utc: Optional[str] = None
    last_modified_utc: Optional[str] = None
    tns_snapshot_utc: Optional[str] = None


def download_tns_public_snapshot(
    output_zip_path: Union[str, Path],
    url: str = TNS_PUBLIC_OBJECTS_URL,
    user_agent: str = DEFAULT_TNS_MARKER,
    force_refresh: bool = False,
    manifest_root: Optional[Path] = None,
) -> RetrievalManifestEntry:
    """
    Downloads the official `tns_public_objects.csv.zip` archive into `output_zip_path`
    using the documented `tns_marker` User-Agent header and returns its manifest entry.
    """
    dest = Path(output_zip_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    http_status: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None

    if not dest.exists() or force_refresh:
        tmp_path = dest.with_suffix(".zip.tmp")
        with httpx.stream(
            "GET",
            url,
            headers={"User-Agent": user_agent},
            timeout=120.0,
            follow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            http_status = resp.status_code
            etag = resp.headers.get("etag")
            last_modified = resp.headers.get("last-modified")
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
        tmp_path.replace(dest)

    retrieved_at = datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc).isoformat()
    rel_path = str(dest.relative_to(manifest_root)) if manifest_root else dest.name

    with zipfile.ZipFile(dest, "r") as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fh:
            first_line = fh.readline().decode("utf-8", errors="replace").strip()
            has_banner = "objid" not in first_line.lower()
        with zf.open(csv_name) as fh:
            header_df = pd.read_csv(fh, skiprows=1 if has_banner else 0, nrows=0)
            columns = list(header_df.columns)
        with zf.open(csv_name) as fh:
            row_count = sum(1 for _ in fh) - (2 if has_banner else 1)

    return RetrievalManifestEntry(
        artifact_id="tns_public_objects_csv_zip",
        source_system="TNS_PUBLIC_OBJECTS",
        source_uri=url,
        local_relative_path=rel_path,
        retrieved_at_utc=retrieved_at,
        sha256=compute_file_sha256(dest),
        size_bytes=dest.stat().st_size,
        row_count=max(row_count, 0),
        schema_columns=columns,
        http_status=http_status,
        etag=etag,
        last_modified=last_modified,
        query_params={"header_mode": "tns_marker_guest"},
        git_commit=current_git_commit(),
    )


def load_tns_dataframe(
    snapshot_path: Union[str, Path],
    ztf_only: bool = False,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Loads a raw TNS public catalog `.csv.zip` or `.csv` file into a standardized DataFrame,
    returning `(df, banner_timestamp_utc)`.
    """
    p = Path(snapshot_path)
    banner_utc: Optional[str] = None

    def _read_stream(fh: Any) -> pd.DataFrame:
        nonlocal banner_utc
        first_line = fh.readline().decode("utf-8", errors="replace").strip()
        has_banner = "objid" not in first_line.lower()
        if has_banner and first_line:
            banner_utc = first_line
        fh.seek(0)
        return pd.read_csv(fh, skiprows=1 if has_banner else 0, low_memory=False)

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as fh:
                df = _read_stream(fh)
    else:
        with open(p, "rb") as fh:
            df = _read_stream(fh)

    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["declination"] = pd.to_numeric(df["declination"], errors="coerce")
    df = df.dropna(subset=["ra", "declination"]).copy()
    df["ra"] = df["ra"] % 360.0

    mapped = [
        map_tns_type_with_provenance(
            t,
            pfx,
            class_bibcodes=bib,
            has_redshift=pd.notna(zval),
        )
        for t, pfx, bib, zval in zip(
            df["type"],
            df.get("name_prefix", ""),
            df.get("Class_ADS_bibcodes", ""),
            df.get("redshift", pd.Series([None] * len(df))),
        )
    ]
    df["coarse_class"] = [m[0].value for m in mapped]
    df["fine_class"] = [m[1] for m in mapped]
    df["evidence_tier"] = [m[2].value for m in mapped]
    df["label_provenance"] = df["evidence_tier"]
    df["taxonomy_version"] = REAL_SURVEY_TAXONOMY_VERSION
    df["is_supernova"] = [m[3] for m in mapped]
    df["has_class_bibcode"] = [
        bool(str(b).strip()) and str(b).strip().lower() not in ("nan", "none", "null")
        if b is not None and not pd.isna(b)
        else False
        for b in df.get("Class_ADS_bibcodes", pd.Series([""] * len(df)))
    ]
    df["has_host_redshift_only"] = [
        (m[2] == EvidenceTier.L5_UNCLASSIFIED) and bool(zval is not None and not pd.isna(zval))
        for m, zval in zip(mapped, df.get("redshift", pd.Series([None] * len(df))))
    ]
    df["ztf_ids"] = [extract_ztf_ids(s) for s in df.get("internal_names", "")]
    df["has_ztf_id"] = df["ztf_ids"].map(bool)
    df["tns_name"] = (
        df["name_prefix"].fillna("").astype(str).str.strip()
        + df["name"].fillna("").astype(str).str.strip()
    )

    if ztf_only:
        df = df[df["has_ztf_id"]].copy()

    return df.reset_index(drop=True), banner_utc


def row_to_tns_record(
    row: Dict[str, Any],
    tns_snapshot_utc: Optional[str] = None,
) -> TnsObjectRecord:
    """Converts a DataFrame row dict into a validated `TnsObjectRecord`."""
    ra = float(row["ra"]) % 360.0
    dec = float(row["declination"])
    pix32 = CelestialCoordinates(ra_deg=ra, dec_deg=dec).compute_healpix_index(nside=32)
    class_bib_raw = (
        None
        if row.get("Class_ADS_bibcodes") is None or pd.isna(row.get("Class_ADS_bibcodes"))
        else str(row.get("Class_ADS_bibcodes")).strip()
    )
    if class_bib_raw == "":
        class_bib_raw = None
    redshift = (
        None
        if row.get("redshift") is None or pd.isna(row.get("redshift"))
        else float(row["redshift"])
    )
    coarse, fine, tier, is_sn = map_tns_type_with_provenance(
        row.get("type"),
        row.get("name_prefix"),
        class_bibcodes=class_bib_raw,
        has_redshift=redshift is not None,
    )
    disc_str = (
        None
        if row.get("discoverydate") is None or pd.isna(row.get("discoverydate"))
        else str(row["discoverydate"]).strip()
    )
    disc_mjd = parse_tns_discovery_mjd(disc_str)
    disc_mag = (
        None
        if row.get("discoverymag") is None or pd.isna(row.get("discoverymag"))
        else float(row["discoverymag"])
    )
    ztf_list = row.get("ztf_ids")
    if ztf_list is None:
        ztf_list = extract_ztf_ids(row.get("internal_names"))

    def _clean_opt(val: Any) -> Optional[str]:
        if val is None or pd.isna(val):
            return None
        s = str(val).strip()
        return s or None

    return TnsObjectRecord(
        objid=int(row["objid"]),
        tns_name=str(row.get("tns_name") or f"{row.get('name_prefix', '')}{row.get('name', '')}").strip(),
        name_prefix=str(row.get("name_prefix") or "").strip(),
        ra_deg=ra,
        dec_deg=dec,
        healpix_nside32=pix32,
        discovery_date_utc=disc_str,
        discovery_mjd_utc=disc_mjd,
        discovery_mag=disc_mag,
        discovery_filter=_clean_opt(row.get("filter")),
        redshift=redshift,
        raw_type=_clean_opt(row.get("type")),
        coarse_class=coarse,
        fine_class=fine,
        evidence_tier=tier,
        label_provenance=tier,
        taxonomy_version=REAL_SURVEY_TAXONOMY_VERSION,
        is_supernova=is_sn,
        has_class_bibcode=class_bib_raw is not None,
        has_host_redshift_only=(tier == EvidenceTier.L5_UNCLASSIFIED and redshift is not None),
        reporting_group=_clean_opt(row.get("reporting_group")),
        source_group=_clean_opt(row.get("source_group")),
        internal_names_raw=_clean_opt(row.get("internal_names")),
        ztf_ids=list(ztf_list),
        class_bibcodes=class_bib_raw,
        time_received_utc=_clean_opt(row.get("time_received")),
        creation_date_utc=_clean_opt(row.get("creationdate")),
        last_modified_utc=_clean_opt(row.get("lastmodified")),
        tns_snapshot_utc=tns_snapshot_utc,
    )
