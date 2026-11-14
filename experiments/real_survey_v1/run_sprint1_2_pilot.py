"""
ASTRAEUS Real-Survey Validation v1 — Sprint 1 & Sprint 2 Live Pilot Runner.

Executes reproducible real-sky data snapshot sealing, live ZTF DR24 HATS & ALeRCE
ingestion, TNS spectroscopic/contaminant/unclassified reconciliation, and Gate A
integrity auditing without introducing any new model/agent layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow.parquet as pq

from astraeus_data.corpus.sequence import ObservationType, QualitySeverity
from astraeus_data.real_survey import (
    AlerceZtfClient,
    CoarseTransientClass,
    SnapshotManifest,
    ZtfDr24HatsClient,
    ZtfObjectSummary,
    alerce_lightcurve_to_records,
    download_tns_public_snapshot,
    dr24_row_to_observation_records,
    load_tns_dataframe,
    reconcile_ztf_and_tns,
    row_to_tns_record,
)
from astraeus_data.real_survey.tns_connector import parse_tns_discovery_mjd

RAW_ROOT = Path("data/raw/real_survey_v1")
OUT_SUMMARY = Path("experiments/real_survey_v1/sprint1_2_pilot_summary.json")


def run_pilot(
    per_stratum_limit: int = 10,
    unmatched_page_size: int = 40,
    dr24_tile: tuple[int, int, int] = (4, 0, 1028),
) -> Dict[str, Any]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_ROOT / "snapshot_manifest.json"
    manifest = SnapshotManifest.load(manifest_path)

    # 1. TNS Official Public Snapshot
    tns_zip = RAW_ROOT / "tns" / "tns_public_objects.csv.zip"
    tns_entry = download_tns_public_snapshot(
        output_zip_path=tns_zip, force_refresh=False, manifest_root=RAW_ROOT
    )
    manifest.add_or_update(tns_entry)

    df_all, banner_utc = load_tns_dataframe(tns_zip, ztf_only=False)
    df_ztf = df_all[df_all["has_ztf_id"]].copy()
    df_ztf["disc_mjd"] = df_ztf["discoverydate"].map(parse_tns_discovery_mjd)

    # Empirical catalog coverage statistics
    tns_stats = {
        "snapshot_banner_utc": banner_utc,
        "total_tns_objects": int(len(df_all)),
        "tns_with_ztf_id": int(len(df_ztf)),
        "ztf_coarse_class_counts": {
            str(k): int(v) for k, v in df_ztf["coarse_class"].value_counts().items()
        },
        "ztf_post_2024_counts": {
            str(k): int(v)
            for k, v in df_ztf.loc[df_ztf["disc_mjd"] >= 60310.0, "coarse_class"]
            .value_counts()
            .items()
        },
        "ztf_with_redshift": int(df_ztf["redshift"].notna().sum()),
    }

    # 2. ZTF DR24 HATS Parquet Live Shard Verification
    dr24_client = ZtfDr24HatsClient(RAW_ROOT / "ztf_dr24")
    part_df = dr24_client.fetch_partition_info()
    norder, dir_idx, npix = dr24_tile
    dr24_parquet, dr24_entry = dr24_client.fetch_partition_parquet(
        norder=norder, dir_idx=dir_idx, npix=npix, manifest_root=RAW_ROOT
    )
    manifest.add_or_update(dr24_entry)

    tbl = pq.read_table(
        dr24_parquet,
        columns=["objectid", "filterid", "objra", "objdec", "nepochs", "lightcurve"],
    ).slice(0, 100)
    dr24_rows = tbl.to_pylist()
    dr24_total_epochs = 0
    dr24_retained_epochs = 0
    for r in dr24_rows:
        lc = r["lightcurve"]
        dr24_total_epochs += len(lc["hmjd"])
        recs = dr24_row_to_observation_records(r, exclude_fatal=True)
        dr24_retained_epochs += len(recs)

    dr24_stats = {
        "partition_count_total": int(len(part_df)),
        "sampled_tile": f"Norder={norder},Dir={dir_idx},Npix={npix}",
        "tile_row_count": int(dr24_entry.row_count or 0),
        "tile_size_bytes": int(dr24_entry.size_bytes),
        "sample_100_objects_raw_epochs": dr24_total_epochs,
        "sample_100_objects_clean_epochs": dr24_retained_epochs,
        "excluded_cloud_moon_or_invalid_epochs": dr24_total_epochs - dr24_retained_epochs,
    }

    # 3. Stratified Real-Sky ALeRCE + TNS Pilot Cohort
    strata_classes = [
        CoarseTransientClass.SN_IA.value,
        CoarseTransientClass.SN_CC.value,
        CoarseTransientClass.SLSN.value,
        CoarseTransientClass.TDE.value,
        CoarseTransientClass.CV_NOVA.value,
        CoarseTransientClass.AGN_QSO.value,
        CoarseTransientClass.VARSTAR.value,
        CoarseTransientClass.UNCLASSIFIED.value,
    ]
    selected_oids: List[str] = []
    seen_oids: set[str] = set()

    for c_name in strata_classes:
        sub = df_ztf[df_ztf["coarse_class"] == c_name]
        pre = sub[sub["disc_mjd"].fillna(0.0) < 60310.0].head(max(1, per_stratum_limit // 2))
        post = sub[sub["disc_mjd"].fillna(0.0) >= 60310.0].head(max(1, per_stratum_limit // 2))
        picked = pd.concat([pre, post])
        for z_list in picked["ztf_ids"]:
            if z_list and z_list[0] not in seen_oids:
                seen_oids.add(z_list[0])
                selected_oids.append(z_list[0])

    alerce = AlerceZtfClient(RAW_ROOT / "alerce", timeout_s=45.0)
    ztf_summaries: List[ZtfObjectSummary] = []
    # Query ALeRCE indexed classifier streams to mix in real AGN & variable-star stream alerts
    # (uses count="false" to avoid expensive full-table row count scans on ALeRCE)
    for stream_name, q_params in [
        (
            "stream_agn_sample",
            {
                "classifier": "lc_classifier",
                "class": "AGN",
                "probability": 0.6,
                "count": "false",
                "page_size": max(5, unmatched_page_size // 2),
            },
        ),
        (
            "stream_vs_sample",
            {
                "classifier": "lc_classifier",
                "class": "VS",
                "probability": 0.6,
                "count": "false",
                "page_size": max(5, unmatched_page_size // 2),
            },
        ),
    ]:
        stream_page, alerce_page_entry = alerce.fetch_objects_page(
            snapshot_name=stream_name,
            params=q_params,
            manifest_root=RAW_ROOT,
        )
        manifest.add_or_update(alerce_page_entry)
        for item in stream_page.get("items") or []:
            oid = str(item["oid"])
            if oid not in seen_oids and item.get("meanra") is not None:
                seen_oids.add(oid)
                ztf_summaries.append(
                    ZtfObjectSummary(
                        oid=oid,
                        ra_deg=float(item["meanra"]),
                        dec_deg=float(item["meandec"]),
                        firstmjd=item.get("firstmjd"),
                        lastmjd=item.get("lastmjd"),
                        ndet=item.get("ndet"),
                    )
                )

    # Fetch live ALeRCE object metadata + full lightcurves concurrently (keep-alive + 6 workers)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lc_metrics = {
        "fetched_lightcurves": 0,
        "failed_fetches": 0,
        "total_detections": 0,
        "negative_isdiffpos_detections": 0,
        "total_upper_limits": 0,
        "empty_lightcurves": 0,
    }

    def _fetch_one(oid: str) -> Dict[str, Any]:
        client = AlerceZtfClient(RAW_ROOT / "alerce", timeout_s=45.0)
        meta = client.fetch_object_meta(oid)
        lc_payload, _, _ = client.fetch_lightcurve(oid)
        return {"oid": oid, "meta": meta, "lc": lc_payload}

    print(f"Fetching {len(selected_oids)} stratified ALeRCE lightcurves...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch_one, oid): oid for oid in selected_oids}
        for fut in as_completed(futs):
            oid = futs[fut]
            try:
                res_one = fut.result()
                meta = res_one["meta"]
                lc_payload = res_one["lc"]
                ra = float(meta["meanra"])
                dec = float(meta["meandec"])
                ztf_summaries.append(
                    ZtfObjectSummary(
                        oid=oid,
                        ra_deg=ra,
                        dec_deg=dec,
                        firstmjd=meta.get("firstmjd"),
                        lastmjd=meta.get("lastmjd"),
                        ndet=meta.get("ndet"),
                    )
                )
                recs = alerce_lightcurve_to_records(
                    oid=oid,
                    ra_deg=ra,
                    dec_deg=dec,
                    detections=lc_payload.get("detections") or [],
                    non_detections=lc_payload.get("non_detections") or [],
                )
                lc_metrics["fetched_lightcurves"] += 1
                if not recs:
                    lc_metrics["empty_lightcurves"] += 1
                for r in recs:
                    if r.obs_type == ObservationType.CATALOG_UPPER_LIMIT:
                        lc_metrics["total_upper_limits"] += 1
                    else:
                        lc_metrics["total_detections"] += 1
                        if (r.measured_flux_nmgy or 0.0) < 0.0:
                            lc_metrics["negative_isdiffpos_detections"] += 1
            except Exception:
                lc_metrics["failed_fetches"] += 1
    print(f"Fetched {lc_metrics['fetched_lightcurves']} lightcurves (failed={lc_metrics['failed_fetches']})", flush=True)

    # 4. Run Sprint 2 Physical Event Reconciliation & Gate A Audit against ALL ZTF-TNS + nearby TNS
    # Pre-filter TNS candidates near pilot objects (within 1 deg box) or matched IDs for fast audit
    pilot_tns_rows = df_all[
        df_all["ztf_ids"].map(lambda zs: any(z in seen_oids for z in zs))
    ]
    # Fast coordinate candidate filter across all 212,832 TNS entries
    import numpy as np
    from astropy.coordinates import SkyCoord, search_around_sky
    import astropy.units as u

    c_ztf = SkyCoord(
        ra=np.array([z.ra_deg for z in ztf_summaries]) * u.deg,
        dec=np.array([z.dec_deg for z in ztf_summaries]) * u.deg,
    )
    c_tns = SkyCoord(
        ra=df_all["ra"].to_numpy(dtype=float) * u.deg,
        dec=df_all["declination"].to_numpy(dtype=float) * u.deg,
    )
    _, idx_tns, _, _ = search_around_sky(c_ztf, c_tns, 30.0 * u.arcsec)
    cand_indices = set(idx_tns.tolist()) | set(pilot_tns_rows.index.tolist())
    tns_records = [
        row_to_tns_record(df_all.loc[i].to_dict(), tns_snapshot_utc=banner_utc)
        for i in sorted(cand_indices)
    ]

    reconciled_events, gate_a = reconcile_ztf_and_tns(
        ztf_objects=ztf_summaries,
        tns_records=tns_records,
        temporal_cutoff_mjd=60310.0,
    )

    manifest.save(manifest_path)
    manifest_verification = manifest.verify_all(RAW_ROOT)

    temporal_counts: Dict[str, int] = {}
    for ev in reconciled_events:
        temporal_counts[ev.temporal_split] = temporal_counts.get(ev.temporal_split, 0) + 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_verification": manifest_verification,
        "tns_catalog_stats": tns_stats,
        "ztf_dr24_stats": dr24_stats,
        "alerce_pilot_lightcurve_stats": lc_metrics,
        "temporal_split_counts": temporal_counts,
        "gate_a_report": gate_a.model_dump(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_pilot()
    print(json.dumps(res, indent=2))
