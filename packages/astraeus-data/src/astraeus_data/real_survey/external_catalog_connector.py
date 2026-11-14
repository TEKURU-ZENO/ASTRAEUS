"""
ASTRAEUS Independent External Contaminant Catalog Connector (P0.4).

Acquires non-broker, independently sourced `EvidenceTier.L3_EXTERNAL_CATALOG` labels
for persistent astrophysical contaminants (AGN / QSOs and Galactic variable stars)
from ESA Gaia DR3 (`gaiadr3.qso_candidates`, `gaiadr3.vari_rrlyrae`,
`gaiadr3.vari_long_period_variable`, `gaiadr3.vari_eclipsing_binary`), preserving raw
CSV snapshots with SHA-256 manifests.

Critical P0.4 invariant: broker classifier outputs (such as ALeRCE `lc_classifier`)
MUST NEVER be admitted as `L3_EXTERNAL_CATALOG` ground truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from astraeus_core.coordinates import CelestialCoordinates
from astraeus_data.real_survey.manifest import (
    RetrievalManifestEntry,
    compute_file_sha256,
    current_git_commit,
)
from astraeus_data.real_survey.tns_connector import (
    CoarseTransientClass,
    EvidenceTier,
    REAL_SURVEY_TAXONOMY_VERSION,
)

GAIA_DR3_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_DR3_TAP_MIRROR_URLS: Tuple[str, ...] = (
    GAIA_DR3_TAP_SYNC_URL,
    "https://gaia.ari.uni-heidelberg.de/tap/sync",
)


class ExternalCatalogRecord(BaseModel):
    """Independent external astrophysical catalog label (`L3_EXTERNAL_CATALOG`)."""

    model_config = ConfigDict(frozen=True)

    external_id: str
    source_catalog: str
    ra_deg: float
    dec_deg: float
    healpix_nside32: int
    coarse_class: CoarseTransientClass
    fine_class: str
    evidence_tier: EvidenceTier = EvidenceTier.L3_EXTERNAL_CATALOG
    taxonomy_version: str = REAL_SURVEY_TAXONOMY_VERSION
    is_supernova: Optional[bool] = False
    is_broker_derived: bool = False
    phot_g_mean_mag: Optional[float] = None
    catalog_redshift: Optional[float] = None
    catalog_score: Optional[float] = None

    @model_validator(mode="after")
    def _enforce_non_broker_l3(self) -> "ExternalCatalogRecord":
        if self.is_broker_derived and self.evidence_tier == EvidenceTier.L3_EXTERNAL_CATALOG:
            raise ValueError(
                "P0.4 violation: broker-derived predictions cannot be promoted to L3_EXTERNAL_CATALOG."
            )
        if "alerce" in self.source_catalog.lower() or "fink" in self.source_catalog.lower():
            if self.evidence_tier == EvidenceTier.L3_EXTERNAL_CATALOG:
                raise ValueError(
                    f"P0.4 violation: alert broker `{self.source_catalog}` is L4, not L3_EXTERNAL_CATALOG."
                )
        return self


def gaia_dataframe_to_external_records(
    df: pd.DataFrame,
    source_catalog: str,
    coarse_class: CoarseTransientClass,
    fine_class: str,
) -> List[ExternalCatalogRecord]:
    """Converts a Gaia DR3 query CSV DataFrame into validated `ExternalCatalogRecord` entries."""
    records: List[ExternalCatalogRecord] = []
    if df.empty:
        return records

    for row in df.to_dict(orient="records"):
        ra_raw = row.get("ra")
        dec_raw = row.get("dec")
        sid = row.get("source_id")
        if ra_raw is None or dec_raw is None or sid is None or pd.isna(ra_raw) or pd.isna(dec_raw):
            continue
        ra = float(ra_raw) % 360.0
        dec = float(dec_raw)
        pix32 = CelestialCoordinates(ra_deg=ra, dec_deg=dec).compute_healpix_index(nside=32)

        gmag = row.get("phot_g_mean_mag")
        z_qsoc = row.get("redshift_qsoc")
        score = row.get("global_ranking") if row.get("global_ranking") is not None else row.get("best_class_score")
        fine_val = str(fine_class)
        if row.get("best_classification") is not None and not pd.isna(row.get("best_classification")):
            fine_val = f"{fine_class}:{str(row['best_classification']).strip()}"

        records.append(
            ExternalCatalogRecord(
                external_id=f"gaia_dr3:{int(sid)}",
                source_catalog=source_catalog,
                ra_deg=ra,
                dec_deg=dec,
                healpix_nside32=pix32,
                coarse_class=coarse_class,
                fine_class=fine_val,
                evidence_tier=EvidenceTier.L3_EXTERNAL_CATALOG,
                taxonomy_version=REAL_SURVEY_TAXONOMY_VERSION,
                is_supernova=False,
                is_broker_derived=False,
                phot_g_mean_mag=float(gmag) if gmag is not None and not pd.isna(gmag) else None,
                catalog_redshift=float(z_qsoc) if z_qsoc is not None and not pd.isna(z_qsoc) else None,
                catalog_score=float(score) if score is not None and not pd.isna(score) else None,
            )
        )
    return records


class GaiaDr3TapClient:
    """Caching HTTPS client for ESA Gaia DR3 ADQL TAP queries."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        tap_url: str = GAIA_DR3_TAP_SYNC_URL,
        fallback_urls: Tuple[str, ...] = GAIA_DR3_TAP_MIRROR_URLS,
        timeout_s: float = 90.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tap_url = tap_url
        ordered = [tap_url] + [u for u in fallback_urls if u != tap_url]
        self.endpoint_urls: Tuple[str, ...] = tuple(ordered)
        self.timeout_s = timeout_s

    @staticmethod
    def _is_valid_csv_payload(raw_bytes: bytes) -> bool:
        """Rejects empty, HTML downtime, or XML VOTable error payloads."""
        head = raw_bytes[:512].decode("utf-8", errors="replace").lstrip()
        if not head or head.startswith("<"):
            return False
        first_line = head.splitlines()[0].strip() if head.splitlines() else ""
        return "," in first_line or first_line.isidentifier()

    def run_adql_csv(
        self,
        snapshot_name: str,
        adql_query: str,
        force_refresh: bool = False,
        manifest_root: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, RetrievalManifestEntry]:
        """Executes (or loads cached) ADQL TAP query as CSV with SHA-256 provenance."""
        out_path = self.cache_dir / f"{snapshot_name}.csv"
        http_status: Optional[int] = None
        used_uri: str = self.tap_url
        cache_valid = out_path.exists() and self._is_valid_csv_payload(out_path.read_bytes())
        if not cache_valid or force_refresh:
            errors: List[str] = []
            fetched = False
            with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                for endpoint in self.endpoint_urls:
                    try:
                        resp = client.get(
                            endpoint,
                            params={
                                "REQUEST": "doQuery",
                                "LANG": "ADQL",
                                "FORMAT": "csv",
                                "QUERY": adql_query,
                            },
                        )
                        resp.raise_for_status()
                        if not self._is_valid_csv_payload(resp.content):
                            snippet = resp.text[:160].replace("\n", " ")
                            errors.append(f"{endpoint}: non-CSV response ({snippet})")
                            continue
                        http_status = resp.status_code
                        used_uri = endpoint
                        tmp = out_path.with_suffix(".csv.tmp")
                        tmp.write_bytes(resp.content)
                        tmp.replace(out_path)
                        fetched = True
                        break
                    except Exception as exc:
                        errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
            if not fetched:
                raise RuntimeError(
                    f"All Gaia DR3 TAP endpoints failed for `{snapshot_name}`: "
                    + "; ".join(errors)
                )

        df = pd.read_csv(out_path, low_memory=False)
        rel_path = str(out_path.relative_to(manifest_root)) if manifest_root else out_path.name
        entry = RetrievalManifestEntry(
            artifact_id=f"gaia_dr3_tap:{snapshot_name}",
            source_system="ESA_GAIA_DR3_TAP",
            source_uri=used_uri,
            local_relative_path=rel_path,
            retrieved_at_utc=datetime.fromtimestamp(
                out_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            sha256=compute_file_sha256(out_path),
            size_bytes=out_path.stat().st_size,
            row_count=len(df),
            schema_columns=list(df.columns),
            http_status=http_status,
            query_params={"query": adql_query},
            git_commit=current_git_commit(),
        )
        return df, entry
