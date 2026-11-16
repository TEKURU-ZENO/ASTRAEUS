"""
ASTRAEUS Phase 5: ATLAS Forced Photometry Request Manifest Generator.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ATLAS-MANIFEST-001

Implements:
1. Stratified 10-candidate operational validation sample selection spanning:
   - 5 taxonomic classes (SN_Ia, SN_CC, TDE, SLSN, OTHER_EXTRAGALACTIC)
   - Declination spread (-23.2 deg to +70.98 deg)
   - Apparent peak magnitude (16.69 to 18.91 mag)
   - Galactic latitude (-32.9 deg to +66.8 deg)
2. Preserves 140 candidates in the held-back expansion cohort.
3. Computes deterministic 24-character idempotency keys.
4. Initializes the durable application-level request ledger in Parquet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
from astropy_healpix import HEALPix
import astropy.units as u

from astraeus_data.brokers import (
    AtlasRequestState,
    AtlasRequestRecord,
    AtlasForcedPhotometryClient,
)
from astraeus_data.registry import StableEntityRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
P5_DIR = Path(__file__).resolve().parent

# Exact ZTF IDs selected for the 10-candidate operational validation sample
TEN_OBJECT_PILOT_ZTF_IDS = [
    "ZTF22aaucozn",  # SN_Ia (bright, high northern circumpolar dec=+70.98 deg, m=16.69)
    "ZTF19accdehb",  # SN_Ia (negative dec=-14.72 deg, m=18.12)
    "ZTF21abwxhje",  # SN_Ia (faint southern dec=-16.48 deg, m=18.91)
    "ZTF21acbmxgn",  # SN_Ia (high northern dec=+55.67 deg, m=17.22)
    "ZTF23aaahtnj",  # SN_CC (SN Ib, negative dec=-23.16 deg, m=17.28)
    "ZTF23abdshvl",  # SN_CC (SN Ic, northern dec=+48.91 deg, m=17.53)
    "ZTF18aabtxvd",  # TDE (AT2018zr / PS18kh, benchmark, dec=+33.87 deg, m=16.92)
    "ZTF18aabdajx",  # TDE (AT2022dbl, clean host galaxy, dec=+29.48 deg, m=18.42)
    "ZTF24abdhylt",  # SLSN (SN2024uea, superluminous, dec=+27.92 deg, m=18.33)
    "ZTF19abrdxbh",  # OTHER_EXTRAGALACTIC (SN2019ofm, rare, dec=+19.57 deg, m=17.84)
]


def generate_manifests():
    print("[ASTRAEUS Phase 5] Generating ATLAS Forced Photometry Request Manifests...")

    # 1. Load 150-candidate cohort
    cohort_path = P5_DIR / "phase5_feasibility_pilot_cohort.parquet"
    assert cohort_path.exists(), f"Missing cohort file: {cohort_path}"
    df_cohort = pd.read_parquet(cohort_path)
    assert len(df_cohort) == 150

    # 2. Load durable entity registry to obtain permanent ASTRAEUS-EVT IDs
    reg_path = P5_DIR / "phase5_entity_registry.parquet"
    assert reg_path.exists(), f"Missing entity registry: {reg_path}"
    registry = StableEntityRegistry.load_from_parquet(reg_path)

    # 3. Assert Gate 3 Quarantine Disjointness
    hp_path = REPO_ROOT / "configs" / "data_splits_healpix.json"
    with open(hp_path, "r", encoding="utf-8") as f:
        hp_config = json.load(f)
    test_pixels = set(int(p) for p in hp_config["splits"]["test"])
    assert len(test_pixels) == 1808

    hp = HEALPix(nside=hp_config.get("healpix_nside", 32), order="nested", frame="icrs")
    cand_pixels = hp.lonlat_to_healpix(df_cohort["ra_deg"].values * u.deg, df_cohort["dec_deg"].values * u.deg)
    overlap = set(cand_pixels).intersection(test_pixels)
    assert len(overlap) == 0, f"Violation: {len(overlap)} candidates touch Gate 3 test pixels!"
    print("  ✓ Zero-Touch Gate 3 Quarantine Verified (0 overlap with 1,808 test pixels).")

    # 4. Build Request Records
    records_150 = []
    pilot_10_records = []

    for idx, row in df_cohort.iterrows():
        ztf_id = str(row["ztf_id"])
        iau_name = str(row["iau_id"])
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        peakt = float(row["peakt"])
        # In this dataset, peakt is offset from MJD 58000.0 (JD 2458000)
        disc_mjd = 58000.0 + peakt

        # Entity lookup
        ent_id = registry._ztf_index.get(ztf_id)
        assert ent_id is not None, f"Entity not found in registry for {ztf_id}"

        # Standard MJD request window: [disc_mjd - 60.0, disc_mjd + 180.0]
        # ATLAS operations began around MJD 57300 (late 2015)
        mjd_min = max(57300.0, round(disc_mjd - 60.0, 2))
        mjd_max = round(disc_mjd + 180.0, 2)

        is_in_10obj_pilot = ztf_id in TEN_OBJECT_PILOT_ZTF_IDS

        req = AtlasForcedPhotometryClient.create_request(
            astraeus_id=ent_id,
            target_name=iau_name,
            ra_deg=ra,
            dec_deg=dec,
            mjd_min=mjd_min,
            mjd_max=mjd_max,
            use_reduced=False,  # Difference-image forced photometry
        )

        rec_dict = req.model_dump()
        rec_dict["ztf_id"] = ztf_id
        rec_dict["astraeus_class"] = str(row["astraeus_class"])
        rec_dict["raw_type"] = str(row.get("raw_type", "Unknown"))
        rec_dict["peakmag"] = float(row["peakmag"])
        rec_dict["b_gal"] = float(row["b_gal"])
        rec_dict["A_V"] = float(row["A_V"])
        rec_dict["cohort_stage"] = "STAGE1_10OBJ_PILOT" if is_in_10obj_pilot else "STAGE2_HELD_BACK_EXPANSION"

        records_150.append(rec_dict)
        if is_in_10obj_pilot:
            pilot_10_records.append(rec_dict)

    assert len(pilot_10_records) == 10, f"Expected 10 pilot records, found {len(pilot_10_records)}"
    assert len(records_150) == 150

    # Verify 10-object class diversity
    pilot_classes = set(r["astraeus_class"] for r in pilot_10_records)
    expected_classes = {"SN_Ia", "SN_CC", "TDE", "SLSN", "OTHER_EXTRAGALACTIC"}
    assert pilot_classes == expected_classes, f"Pilot missing classes: {expected_classes - pilot_classes}"
    print(f"  ✓ 10-Object Pilot Sample satisfies all 5 taxonomy classes: {sorted(pilot_classes)}")

    # 5. Output Manifests
    pilot_manifest_path = P5_DIR / "phase5_atlas_10obj_pilot_manifest.json"
    full_manifest_path = P5_DIR / "phase5_atlas_150obj_request_manifest.json"
    ledger_parquet_path = P5_DIR / "phase5_atlas_request_ledger.parquet"

    pilot_manifest = {
        "manifest_identifier": "ASTRAEUS-P5-ATLAS-MANIFEST-PILOT10",
        "author": "Dev Mehta",
        "lead_investigator": "Dev Mehta",
        "date": "2026-11-16",
        "protocol_version": "ASTRAEUS-P5-ATLAS-V1",
        "stage": "STAGE1_OPERATIONAL_VALIDATION_PILOT",
        "candidates_count": 10,
        "taxonomic_coverage": {
            "SN_Ia": 4,
            "SN_CC": 2,
            "TDE": 2,
            "SLSN": 1,
            "OTHER_EXTRAGALACTIC": 1,
        },
        "declination_range_deg": [-23.16, 70.98],
        "peak_magnitude_range": [16.69, 18.91],
        "governance_status": "GATE3_QUARANTINE_RESPECTED_ZERO_TOUCH",
        "idempotency_policy": "DETERMINISTIC_APPLICATION_LEVEL_KEY",
        "requests": pilot_10_records,
    }

    full_manifest = {
        "manifest_identifier": "ASTRAEUS-P5-ATLAS-MANIFEST-FULL150",
        "author": "Dev Mehta",
        "lead_investigator": "Dev Mehta",
        "date": "2026-11-16",
        "protocol_version": "ASTRAEUS-P5-ATLAS-V1",
        "stage": "STAGE1_10OBJ_PILOT_PLUS_STAGE2_140OBJ_HELD_BACK",
        "total_candidates_count": 150,
        "stage1_active_count": 10,
        "stage2_held_back_count": 140,
        "governance_status": "GATE3_QUARANTINE_RESPECTED_ZERO_TOUCH",
        "idempotency_policy": "DETERMINISTIC_APPLICATION_LEVEL_KEY",
        "requests": records_150,
    }

    with open(pilot_manifest_path, "w", encoding="utf-8") as f:
        json.dump(pilot_manifest, f, indent=2)

    with open(full_manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2)

    # Durable Request Ledger
    df_ledger = pd.DataFrame(records_150)
    df_ledger.to_parquet(ledger_parquet_path, index=False)

    print(f"  ✓ Saved Manifests & Ledger:")
    print(f"    - {pilot_manifest_path.name} (10 active validation requests)")
    print(f"    - {full_manifest_path.name} (150 total requests)")
    print(f"    - {ledger_parquet_path.name} (150 durable ledger rows, state=PENDING_SUBMISSION)")
    print("[ASTRAEUS Phase 5] Manifest generation complete.")


if __name__ == "__main__":
    generate_manifests()
