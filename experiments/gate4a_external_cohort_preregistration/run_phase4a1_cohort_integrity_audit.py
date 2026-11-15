"""
ASTRAEUS Gate 4 Phase 4A.1: Cohort Integrity Reconciliation Audit.

GOVERNANCE CONTRACT:
- Resolves all 5 blocking findings and Phase 4A.1 review directives.
- Asserts cryptographic input hashes for all dependencies.
- Computes exact set-theoretic and sequential attrition arithmetic (|A|, |B|, |A ∩ B|, |A ∪ B|, |B \\ A|).
- Applies frozen taxonomy mapping before counting class support.
- Fully reconciles TDE counts: all raw (75) vs taxonomy-admitted (70), Train/Val overlaps (69 vs 64), spatial buffer (4), and retained (2).
- Replaces coarse power claims with disciplined cohort-level support labels:
  - LARGE_SAMPLE_SUPPORT (N >= 30)
  - DESCRIPTIVE_SUPPORT (5 <= N < 30)
  - NOT_EVALUABLE (N < 5)
- Emits complete 10,988-row accounting ledgers (Parquet and CSV) and dedicated admitted candidates CSV.
- Zero feature extraction, model training, or inference performed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import yaml

# Expected Input Hashes
EXPECTED_HASHES = {
    "bts_raw_snapshot_csv": "51d85f50772fd1412fcc2646f48b4d331b2e07f7023454f89cddd7a8e7e283ed",
    "candidates_10k_parquet": "6713faee3c2ab47b7a14baa321c540f2b18ddb9b51d6ddcf88dba7008ad214ae",
    "data_splits_healpix_json": "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b",
    "gate4a_taxonomy_mapping_yaml": "265b538f2ad040e15a4b9b0069676c7dbbaec095c9a8b88bbcbe96ad7e8364b1",
    "gate4a_raw_label_reconciliation_csv": "c5c8b06e0e3b065437224513efef33212ae4fdd4404f65c9dca6b7bdc6cef0f1",
}

# Input file paths
BTS_SNAPSHOT_PATH = Path("data/external_validity_v1/raw/bts/bts_raw_snapshot.csv")
CANDIDATES_10K_PATH = Path("data/processed/real_survey_10k_v1/candidates_10k.parquet")
HEALPIX_SPLITS_PATH = Path("configs/data_splits_healpix.json")
TAXONOMY_YAML_PATH = Path("configs/gate4/gate4a_taxonomy_mapping.yaml")
RECONCILIATION_CSV_PATH = Path("configs/gate4/gate4a_raw_label_reconciliation.csv")

OUT_DIR = Path("experiments/gate4a_external_cohort_preregistration")


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 80)
    print("ASTRAEUS GATE 4 PHASE 4A.1: COHORT INTEGRITY RECONCILIATION AUDIT")
    print(f"Execution Timestamp (UTC): {now_utc}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Raw Caltech BTS Snapshot & Assert Hash
    print(f"\n[1/6] Ingesting raw BTS catalog: {BTS_SNAPSHOT_PATH}")
    assert BTS_SNAPSHOT_PATH.exists(), f"BTS snapshot missing at {BTS_SNAPSHOT_PATH}"
    bts_sha = compute_sha256(BTS_SNAPSHOT_PATH)
    assert bts_sha == EXPECTED_HASHES["bts_raw_snapshot_csv"], (
        f"BTS SHA mismatch: {bts_sha} vs {EXPECTED_HASHES['bts_raw_snapshot_csv']}"
    )
    df_bts = pd.read_csv(BTS_SNAPSHOT_PATH)
    n_raw = len(df_bts)
    print(f"  -> Raw BTS records: {n_raw}")
    print(f"  -> Raw BTS SHA-256: {bts_sha} (MATCHED)")

    # 2. Ingest Accessible Train and Validation Candidates & Assert Hash
    print(f"\n[2/6] Ingesting accessible Train/Val candidates: {CANDIDATES_10K_PATH}")
    assert CANDIDATES_10K_PATH.exists(), f"Candidates parquet missing at {CANDIDATES_10K_PATH}"
    cand_sha = compute_sha256(CANDIDATES_10K_PATH)
    assert cand_sha == EXPECTED_HASHES["candidates_10k_parquet"], (
        f"Candidates Parquet SHA mismatch: {cand_sha} vs {EXPECTED_HASHES['candidates_10k_parquet']}"
    )
    df_tv = pd.read_parquet(
        CANDIDATES_10K_PATH,
        filters=[("assigned_split", "in", ["train", "val"])],
        columns=["candidate_id", "ra_deg", "dec_deg", "coarse_class", "assigned_split"],
    )
    n_tv = len(df_tv)
    print(f"  -> Ingested Train/Val candidates: {n_tv}")
    print(f"  -> candidates_10k.parquet SHA-256: {cand_sha} (MATCHED)")

    # 3. Ingest HEALPix Partition Configuration & Assert Hash
    print(f"\n[3/6] Ingesting HEALPix partition configuration: {HEALPIX_SPLITS_PATH}")
    assert HEALPIX_SPLITS_PATH.exists(), f"HEALPix split config missing at {HEALPIX_SPLITS_PATH}"
    splits_sha = compute_sha256(HEALPIX_SPLITS_PATH)
    assert splits_sha == EXPECTED_HASHES["data_splits_healpix_json"], (
        f"HEALPix splits SHA mismatch: {splits_sha} vs {EXPECTED_HASHES['data_splits_healpix_json']}"
    )
    with open(HEALPIX_SPLITS_PATH) as f:
        hp_cfg = json.load(f)
    test_pixel_set = set(hp_cfg["splits"]["test"])
    n_test_pixels = len(test_pixel_set)
    print(f"  -> Test pixels in config: {n_test_pixels} (out of {hp_cfg['total_pixels']})")
    print(f"  -> data_splits_healpix.json SHA-256: {splits_sha} (MATCHED)")

    # 4. Ingest Taxonomy YAML and Reconciliation CSV & Assert Hashes
    print(f"\n[4/6] Ingesting taxonomy reconciliation mapping: {RECONCILIATION_CSV_PATH}")
    tax_yaml_sha = compute_sha256(TAXONOMY_YAML_PATH)
    assert tax_yaml_sha == EXPECTED_HASHES["gate4a_taxonomy_mapping_yaml"], (
        f"Taxonomy YAML SHA mismatch: {tax_yaml_sha} vs {EXPECTED_HASHES['gate4a_taxonomy_mapping_yaml']}"
    )
    recon_csv_sha = compute_sha256(RECONCILIATION_CSV_PATH)
    assert recon_csv_sha == EXPECTED_HASHES["gate4a_raw_label_reconciliation_csv"], (
        f"Reconciliation CSV SHA mismatch: {recon_csv_sha} vs {EXPECTED_HASHES['gate4a_raw_label_reconciliation_csv']}"
    )
    df_recon = pd.read_csv(RECONCILIATION_CSV_PATH)
    recon_map = df_recon.set_index("raw_bts_label").to_dict(orient="index")
    print(f"  -> Unique raw labels reconciled: {len(recon_map)}")
    print(f"  -> Taxonomy YAML SHA-256: {tax_yaml_sha} (MATCHED)")
    print(f"  -> Reconciliation CSV SHA-256: {recon_csv_sha} (MATCHED)")

    # --------------------------------------------------------------------------
    # 5. Deterministic Screening & Exact Set-Theoretic Arithmetic
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[5/6] EXECUTING DETERMINISTIC DEDUPLICATION & ATTRITION ARITHMETIC")
    print("=" * 80)

    # Coordinates
    c_bts = SkyCoord(df_bts["RA"], df_bts["Dec"], unit=(u.hourangle, u.deg))
    c_tv = SkyCoord(ra=df_tv["ra_deg"].values * u.deg, dec=df_tv["dec_deg"].values * u.deg)

    # Set A: Train/Val Overlap (Exact ID match OR positional separation <= 1.5 arcsec)
    cand_ids_tv = set(df_tv["candidate_id"])
    exact_id_mask = df_bts["ZTFID"].isin(cand_ids_tv)
    idx_tv, d2d_tv, _ = c_bts.match_to_catalog_sky(c_tv)
    pos_mask_tv = (d2d_tv.arcsec <= 1.5)
    set_A_mask = (exact_id_mask | pos_mask_tv)
    set_A_indices = set(df_bts[set_A_mask].index)
    size_A = len(set_A_indices)

    # Set B: Spatial Test Footprint Overlap (Intersects test pixel within 60 arcsec cone buffer)
    hp = HEALPix(nside=32, order="nested", frame="icrs")
    set_B_indices = set()
    for idx_b, c in enumerate(c_bts):
        cone_pixels = hp.cone_search_skycoord(c, radius=60.0 * u.arcsec)
        if any(int(p) in test_pixel_set for p in cone_pixels):
            set_B_indices.add(idx_b)
    size_B = len(set_B_indices)

    # Set-Theoretic Intersection and Union
    set_intersection = set_A_indices.intersection(set_B_indices)
    size_intersection = len(set_intersection)

    set_union = set_A_indices.union(set_B_indices)
    size_union = len(set_union)

    # Sequential Incremental Exclusions
    # Stage 1: Exclude Set A (Train/Val overlap)
    stage1_exclusions = size_A
    # Stage 2: Exclude new candidates in Set B not in Set A (Set B \ Set A)
    set_B_minus_A = set_B_indices - set_A_indices
    stage2_incremental_exclusions = len(set_B_minus_A)

    # Retained candidates after sequential screening
    screened_indices = set(range(n_raw)) - set_union
    n_screened = len(screened_indices)

    print(f"Raw Input BTS Records (N_raw)                   : {n_raw}")
    print(f"Set A: Train/Val Overlap Records (|A|)          : {size_A}")
    print(f"Set B: Spatial Test Buffer Records (|B|)        : {size_B}")
    print(f"Intersection: In BOTH Set A and Set B (|A ∩ B|) : {size_intersection}")
    print(f"Joint Union: Total Unique Exclusions (|A ∪ B|)  : {size_union}")
    print(f"Incremental Stage 2 Exclusions (|B \\ A|)       : {stage2_incremental_exclusions}")
    print(f"Sequential Verification: {size_A} + {stage2_incremental_exclusions} = {size_A + stage2_incremental_exclusions} (Matches Union: {size_A + stage2_incremental_exclusions == size_union})")
    print(f"Retained Screened Candidates (N_screened)       : {n_screened}")
    assert n_raw - size_union == n_screened, "Arithmetic mismatch in screened candidates!"
    assert size_intersection == 5, f"Expected intersection size 5, got {size_intersection}"
    assert size_union == 3372, f"Expected union size 3372, got {size_union}"
    assert n_screened == 7616, f"Expected screened size 7616, got {n_screened}"

    # --------------------------------------------------------------------------
    # 6. Taxonomy Mapping, TDE Deep-Dive & Accounting Ledgers
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[6/6] APPLYING FROZEN TAXONOMY & AUDITING COHORT TOTALS")
    print("=" * 80)

    # Add classification and exclusion columns to event dataframe
    ledger_records = []
    for idx_b in range(n_raw):
        row = df_bts.iloc[idx_b]
        raw_type = str(row["type"])
        recon_info = recon_map.get(raw_type, {
            "normalized_subtype": raw_type,
            "astraeus_class": "UNMAPPED",
            "admitted_status": "Unmapped",
            "reconciliation_rationale": "Missing from reconciliation table",
        })

        is_in_A = idx_b in set_A_indices
        is_in_B = idx_b in set_B_indices
        is_screened_admitted = idx_b in screened_indices

        # Primary exclusion reason code
        if is_in_A and is_in_B:
            excl_reason = "EXCL_DUAL_TRAINVAL_AND_SPATIAL_BUFFER"
        elif is_in_A:
            excl_reason = "EXCL_TRAIN_VAL_OVERLAP"
        elif is_in_B:
            excl_reason = "EXCL_SPATIAL_GATE3_FOOTPRINT_BUFFER"
        else:
            excl_reason = "NONE_SCREENED_ADMITTED"

        # Taxonomy status
        admitted_status = recon_info["admitted_status"]
        mapped_class = recon_info["astraeus_class"]
        norm_subtype = recon_info["normalized_subtype"]

        # Screened and Taxonomy-Approved Flag
        is_tax_screened = is_screened_admitted and (admitted_status == "Approved")

        ledger_records.append({
            "bts_row_id": idx_b,
            "ztf_id": row["ZTFID"],
            "iau_id": row["IAUID"],
            "ra_raw": row["RA"],
            "dec_raw": row["Dec"],
            "ra_deg": c_bts[idx_b].ra.deg,
            "dec_deg": c_bts[idx_b].dec.deg,
            "peakt": row["peakt"],
            "peakfilt": row["peakfilt"],
            "peakmag": row["peakmag"],
            "raw_type": raw_type,
            "normalized_subtype": norm_subtype,
            "astraeus_class": mapped_class,
            "taxonomy_admitted_status": admitted_status,
            "is_in_train_val": is_in_A,
            "is_in_spatial_buffer": is_in_B,
            "exclusion_reason": excl_reason,
            "is_screened_candidate": is_screened_admitted,
            "is_taxonomy_screened_candidate": is_tax_screened,
            "spectral_evidence_verified": False,  # Pending Phase 4A.2 FITS audit
            "feature_evaluable": False,            # Pending Phase 4A.2 light-curve match
            "redshift": row["redshift"],
            "b_gal": row["b"],
            "A_V": row["A_V"],
        })

    df_ledger = pd.DataFrame(ledger_records)

    # 1. Total Screened Candidates (N = 7,616)
    df_screened = df_ledger[df_ledger["is_screened_candidate"]].copy()
    assert len(df_screened) == n_screened == 7616

    # 2. Taxonomy-Admitted Candidates (Excluding 2 quarantined records)
    df_tax_admitted = df_screened[df_screened["is_taxonomy_screened_candidate"]].copy()
    n_tax_admitted = len(df_tax_admitted)
    n_screened_quarantined = len(df_screened) - n_tax_admitted
    assert n_tax_admitted == 7614, f"Expected 7614 taxonomy admitted, got {n_tax_admitted}"
    assert n_screened_quarantined == 2, f"Expected 2 quarantined in screened, got {n_screened_quarantined}"

    print(f"\nFour Distinct Cohort Accounting Totals:")
    print(f"  Total 1: Geometrically & Identity-Screened Candidates : {n_screened}")
    print(f"     - Quarantined in Screened Set (raw '-' and 'Other'): {n_screened_quarantined}")
    print(f"  Total 2: Taxonomy-Admitted Candidates                 : {n_tax_admitted}")
    print(f"  Total 3: Spectroscopically Supported Candidates (S1/S2): PENDING_PHASE_4A2_FITS_AUDIT")
    print(f"  Total 4: Feature-Evaluable Candidates (ZTF LCs)       : PENDING_PHASE_4A2_LIGHTCURVE_MATCH")

    # 3. Class Support Breakdown for Taxonomy-Admitted Candidates
    class_counts = df_tax_admitted["astraeus_class"].value_counts().to_dict()
    print(f"\nClass Support Breakdown for Taxonomy-Admitted Candidates (N = {n_tax_admitted}):")
    class_support_designations = {}
    for cname in ["SN_Ia", "SN_CC", "SLSN", "TDE", "OTHER_EXTRAGALACTIC", "AGN_QSO", "CV_Nova", "Varstar"]:
        cnt = class_counts.get(cname, 0)
        if cnt >= 30:
            p_status = "LARGE_SAMPLE_SUPPORT"
        elif 5 <= cnt < 30:
            p_status = "DESCRIPTIVE_SUPPORT"
        else:
            p_status = "NOT_EVALUABLE"
        class_support_designations[cname] = {
            "count": cnt,
            "support_designation": p_status,
        }
        print(f"  - {cname:20s}: {cnt:5d} candidates -> {p_status}")

    assert class_counts["SN_Ia"] == 5916
    assert class_counts["SN_CC"] == 1693
    assert class_counts["SLSN"] == 1
    assert class_counts["TDE"] == 2
    assert class_counts["OTHER_EXTRAGALACTIC"] == 1
    assert class_counts["AGN_QSO"] == 1

    # --------------------------------------------------------------------------
    # 7. Rigorous Reconciled TDE Audit
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RECONCILED TDE POPULATION AUDIT")
    print("=" * 80)
    df_tde_all = df_ledger[df_ledger["raw_type"].str.contains("TDE", na=False)].copy()
    n_raw_tde_strings = len(df_tde_all)
    n_raw_tde_string_tv = int(df_tde_all["is_in_train_val"].sum())

    # Taxonomy-Admitted TDE records (TDE: 66 + TDE-He: 4 = 70)
    df_tde_admitted_raw = df_tde_all[df_tde_all["taxonomy_admitted_status"] == "Approved"]
    n_admitted_tde_raw = len(df_tde_admitted_raw)
    n_admitted_tde_tv = int(df_tde_admitted_raw["is_in_train_val"].sum())
    n_admitted_tde_spatial = int(df_tde_admitted_raw["is_in_spatial_buffer"].sum())

    # Disjoint admitted TDEs
    df_tde_retained = df_tde_admitted_raw[df_tde_admitted_raw["is_screened_candidate"]]
    n_admitted_tde_retained = len(df_tde_retained)

    assert n_raw_tde_strings == 75, f"Expected 75 raw TDE strings, got {n_raw_tde_strings}"
    assert n_admitted_tde_raw == 70, f"Expected 70 admitted raw TDEs, got {n_admitted_tde_raw}"
    assert n_raw_tde_string_tv == 69, f"Expected 69 raw TDE Train/Val overlaps, got {n_raw_tde_string_tv}"
    assert n_admitted_tde_tv == 64, f"Expected 64 admitted TDE Train/Val overlaps, got {n_admitted_tde_tv}"
    assert n_admitted_tde_spatial == 4, f"Expected 4 admitted TDE spatial exclusions, got {n_admitted_tde_spatial}"
    assert n_admitted_tde_retained == 2, f"Expected 2 admitted TDE retained, got {n_admitted_tde_retained}"
    assert n_admitted_tde_raw - n_admitted_tde_tv - n_admitted_tde_spatial == n_admitted_tde_retained

    pct_raw_overlap = (n_raw_tde_string_tv / n_raw_tde_strings) * 100
    pct_admitted_overlap = (n_admitted_tde_tv / n_admitted_tde_raw) * 100

    print(f"All Raw TDE-String Records in BTS           : {n_raw_tde_strings}")
    print(f"Taxonomy-Admitted Raw TDE Records (TDE+He)  : {n_admitted_tde_raw}")
    print(f"Unmapped Raw TDE Records (H-He, featureless): {n_raw_tde_strings - n_admitted_tde_raw}")
    print(f"All Raw TDE-String Train/Val Overlaps       : {n_raw_tde_string_tv} ({pct_raw_overlap:.1f}%)")
    print(f"Taxonomy-Admitted TDE Train/Val Overlaps    : {n_admitted_tde_tv} ({pct_admitted_overlap:.1f}%)")
    print(f"Taxonomy-Admitted TDE Spatial Buffer Excl   : {n_admitted_tde_spatial}")
    print(f"Taxonomy-Admitted TDE Candidates Retained   : {n_admitted_tde_retained}")
    print(f"Proof: {n_admitted_tde_raw} (Admitted Raw) - {n_admitted_tde_tv} (Train/Val) - {n_admitted_tde_spatial} (Spatial) = {n_admitted_tde_retained} (Retained)")

    for _, r in df_tde_retained.iterrows():
        print(f"  -> Disjoint Event: ZTFID={r['ztf_id']}, IAUID={r['iau_id']}, Raw={r['raw_type']}, Subtype={r['normalized_subtype']}, z={r['redshift']}, Mag={r['peakmag']}")

    # --------------------------------------------------------------------------
    # 8. Save Machine-Readable Accounting Artifacts
    # --------------------------------------------------------------------------
    parquet_path = OUT_DIR / "gate4a1_event_accounting_ledger.parquet"
    csv_path = OUT_DIR / "gate4a1_event_accounting_ledger.csv"
    admitted_csv_path = OUT_DIR / "gate4a1_screened_admitted_candidates.csv"

    # Full 10,988 records saved to both Parquet and full CSV
    df_ledger.to_parquet(parquet_path, index=False)
    df_ledger.to_csv(csv_path, index=False)
    # The 7,614 taxonomy-admitted subset saved to dedicated CSV
    df_tax_admitted.to_csv(admitted_csv_path, index=False)

    # Save summary metrics to JSON
    metrics_payload = {
        "milestone": "ASTRAEUS Gate 4 Phase 4A.1 Cohort Integrity Reconciliation",
        "audit_timestamp_utc": now_utc,
        "input_artifacts": {
            "bts_raw_snapshot_csv": {
                "path": str(BTS_SNAPSHOT_PATH),
                "sha256": bts_sha,
                "record_count": n_raw,
                "hash_verified": True,
            },
            "candidates_10k_parquet": {
                "path": str(CANDIDATES_10K_PATH),
                "sha256": cand_sha,
                "train_val_records_ingested": n_tv,
                "hash_verified": True,
            },
            "healpix_splits_json": {
                "path": str(HEALPIX_SPLITS_PATH),
                "sha256": splits_sha,
                "test_pixels_count": n_test_pixels,
                "hash_verified": True,
            },
            "taxonomy_mapping_yaml": {
                "path": str(TAXONOMY_YAML_PATH),
                "sha256": tax_yaml_sha,
                "hash_verified": True,
            },
            "raw_label_reconciliation_csv": {
                "path": str(RECONCILIATION_CSV_PATH),
                "sha256": recon_csv_sha,
                "hash_verified": True,
            },
        },
        "exact_attrition_arithmetic": {
            "raw_bts_records": n_raw,
            "set_A_train_val_overlap": size_A,
            "set_B_spatial_test_buffer": size_B,
            "set_intersection_A_and_B": size_intersection,
            "set_union_total_excluded": size_union,
            "stage1_train_val_exclusions": size_A,
            "stage2_incremental_spatial_exclusions": stage2_incremental_exclusions,
            "retained_screened_candidates": n_screened,
            "formula_proof": f"{size_A} (Train/Val) + {stage2_incremental_exclusions} (New Spatial Buffer) = {size_union} (Total Excluded)",
            "retention_proof": f"{n_raw} (Raw) - {size_union} (Excluded) = {n_screened} (Screened Retained)",
        },
        "four_distinct_cohort_totals": {
            "total_1_geometrically_and_identity_screened_candidates": n_screened,
            "screened_quarantined_or_unclassified_candidates": n_screened_quarantined,
            "total_2_taxonomy_admitted_candidates": n_tax_admitted,
            "total_3_spectroscopically_supported_events_s1_s2": "PENDING_PHASE_4A2_FITS_AUDIT",
            "total_4_feature_evaluable_events": "PENDING_PHASE_4A2_LIGHTCURVE_MATCH",
        },
        "class_support_breakdown": class_support_designations,
        "reconciled_tde_metrics": {
            "all_raw_tde_string_records": n_raw_tde_strings,
            "taxonomy_admitted_raw_tde_records": n_admitted_tde_raw,
            "unmapped_raw_tde_records": n_raw_tde_strings - n_admitted_tde_raw,
            "all_raw_tde_string_train_val_overlaps": n_raw_tde_string_tv,
            "all_raw_tde_string_historical_overlap_pct": round(pct_raw_overlap, 2),
            "taxonomy_admitted_tde_train_val_overlaps": n_admitted_tde_tv,
            "taxonomy_admitted_tde_historical_overlap_pct": round(pct_admitted_overlap, 2),
            "taxonomy_admitted_tde_spatial_buffer_exclusions": n_admitted_tde_spatial,
            "taxonomy_admitted_tde_retained": n_admitted_tde_retained,
            "retained_tde_candidates": df_tde_retained[["ztf_id", "iau_id", "raw_type", "normalized_subtype", "redshift", "peakmag"]].to_dict(orient="records"),
        },
    }

    metrics_json_path = OUT_DIR / "gate4a1_audit_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[Artifact Output] Full accounting ledger (10,988 rows) saved to Parquet: {parquet_path}")
    print(f"[Artifact Output] Full accounting ledger (10,988 rows) saved to CSV    : {csv_path}")
    print(f"[Artifact Output] Screened admitted subset (7,614 rows) saved to CSV   : {admitted_csv_path}")
    print(f"[Artifact Output] Audit metrics JSON saved to: {metrics_json_path}")
    print(f"[Artifact Output] Metrics JSON SHA-256: {compute_sha256(metrics_json_path)}")
    print("=" * 80)
    print("PHASE 4A.1 RECONCILIATION AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
