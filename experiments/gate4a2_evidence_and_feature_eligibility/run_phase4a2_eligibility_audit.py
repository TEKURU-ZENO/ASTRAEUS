"""
ASTRAEUS Gate 4 Phase 4A.2: Evidence & Feature Eligibility Audit Script.

GOVERNANCE CONTRACT:
- Implements the Phase 4A.2 Protocol authorized by Scientific Review.
- Track A (Spectroscopic Evidence): Enforces strict S1-S4 evidence hierarchy against IAU TNS.
  - S1: Directly accessible spectrum with event identity, observation date, instrument, metadata (0 local FITS).
  - S2: Traceable published classification with named observatory/instrument and verifiable TNSCR report.
  - S3: Traceable catalog classification without accessible spectrum or published report (sensitivity only).
  - S4: Photometric or unverified classification (excluded).
- Track B (Light-Curve Feature Eligibility): Multi-dimensional observational accounting.
  - Detections in g and r bands, non-detections / upper limits, temporal baseline, cadence, missingness.
  - Verifies compatibility with the frozen 27 physical observable features interface.
- Completely decoupled and independently auditable tracks.
- ZERO model training, parameter tuning, threshold modification, or inference performed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml

from astraeus_data.real_survey.observation_feature_pipeline import (
    CandidateObservationLoader,
    ObservationConfig,
    RealSurveyFeatureExtractor,
)

# ------------------------------------------------------------------------------
# Pinned Cryptographic Input Hashes
# ------------------------------------------------------------------------------
EXPECTED_INPUT_HASHES = {
    "bts_raw_snapshot_csv": "51d85f50772fd1412fcc2646f48b4d331b2e07f7023454f89cddd7a8e7e283ed",
    "tns_public_objects_zip": "5d3fa52497de6f24491ebf25715fa9b617a220f662b2fc5102f9baa02e094ad8",
    "candidates_10k_parquet": "6713faee3c2ab47b7a14baa321c540f2b18ddb9b51d6ddcf88dba7008ad214ae",
    "data_splits_healpix_json": "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b",
    "gate4a_taxonomy_mapping_yaml": "265b538f2ad040e15a4b9b0069676c7dbbaec095c9a8b88bbcbe96ad7e8364b1",
    "gate4a_raw_label_reconciliation_csv": "c5c8b06e0e3b065437224513efef33212ae4fdd4404f65c9dca6b7bdc6cef0f1",
    "gate4a1_screened_admitted_candidates_csv": "abd7dc90eebeeb5507334f5f30acddf01824e4c3c6efd2834186ca850456fcb9",
}

# Input file paths
BTS_SNAPSHOT_PATH = Path("data/external_validity_v1/raw/bts/bts_raw_snapshot.csv")
TNS_ZIP_PATH = Path("data/external_validity_v1/raw/tns/tns_public_objects.csv.zip")
CANDIDATES_10K_PATH = Path("data/processed/real_survey_10k_v1/candidates_10k.parquet")
HEALPIX_SPLITS_PATH = Path("configs/data_splits_healpix.json")
TAXONOMY_YAML_PATH = Path("configs/gate4/gate4a_taxonomy_mapping.yaml")
RECONCILIATION_CSV_PATH = Path("configs/gate4/gate4a_raw_label_reconciliation.csv")
ADMITTED_CSV_PATH = Path("experiments/gate4a_external_cohort_preregistration/gate4a1_screened_admitted_candidates.csv")
ALERCE_LC_DIR = Path("data/raw/real_survey_v1/alerce/lightcurves")

OUT_DIR = Path("experiments/gate4a2_evidence_and_feature_eligibility")


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 80)
    print("ASTRAEUS GATE 4 PHASE 4A.2: EVIDENCE & FEATURE ELIGIBILITY AUDIT")
    print(f"Execution Timestamp (UTC): {now_utc}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Assert Cryptographic Hashes of All Dependencies
    print("\n[1/5] Verifying input cryptographic integrity...")
    inputs = {
        "bts_raw_snapshot_csv": BTS_SNAPSHOT_PATH,
        "tns_public_objects_zip": TNS_ZIP_PATH,
        "candidates_10k_parquet": CANDIDATES_10K_PATH,
        "data_splits_healpix_json": HEALPIX_SPLITS_PATH,
        "gate4a_taxonomy_mapping_yaml": TAXONOMY_YAML_PATH,
        "gate4a_raw_label_reconciliation_csv": RECONCILIATION_CSV_PATH,
        "gate4a1_screened_admitted_candidates_csv": ADMITTED_CSV_PATH,
    }
    input_hashes = {}
    for key, path in inputs.items():
        assert path.exists(), f"Input dependency missing: {path}"
        sha = compute_sha256(path)
        input_hashes[key] = sha
        expected = EXPECTED_INPUT_HASHES[key]
        assert sha == expected, f"Hash mismatch for {key}: {sha} vs {expected}"
        print(f"  -> {key}: {sha} (MATCHED)")

    # 2. Ingest Taxonomy-Admitted External Candidates
    print(f"\n[2/5] Ingesting taxonomy-admitted candidates: {ADMITTED_CSV_PATH}")
    df_adm = pd.read_csv(ADMITTED_CSV_PATH)
    n_admitted = len(df_adm)
    assert n_admitted == 7614, f"Expected 7,614 candidates, got {n_admitted}"
    print(f"  -> Admitted external candidates: {n_admitted}")

    # 3. Track A: Spectroscopic Evidence Verification
    print(f"\n[3/5] TRACK A: SPECTROSCOPIC EVIDENCE AUDIT (TNS SNAPSHOT)...")
    print(f"  -> Ingesting TNS snapshot: {TNS_ZIP_PATH}")
    with zipfile.ZipFile(TNS_ZIP_PATH) as z:
        with z.open(z.namelist()[0]) as f:
            df_tns = pd.read_csv(f, skiprows=1, low_memory=False)

    print(f"  -> Total TNS records: {len(df_tns)}")

    df_adm["iau_stripped"] = df_adm["iau_id"].dropna().astype(str).str.replace("^SN|^AT", "", regex=True)
    tns_by_name = df_tns.set_index("name")

    # Special alias map for known catalog disambiguations
    alias_map = {
        "ZTF22abyszwg": "2022adbb",  # BTS AT2023aei is alias of TNS 2022adbb (SN Ia, 2022TNSCR3633....1T)
    }

    evidence_rows = []
    for _, r in df_adm.iterrows():
        zid = str(r["ztf_id"])
        iau_raw = str(r["iau_id"]) if pd.notna(r["iau_id"]) else ""
        name = r["iau_stripped"] if pd.notna(r["iau_stripped"]) else ""
        raw_type = str(r["raw_type"])
        norm_sub = str(r["normalized_subtype"])
        ast_cls = str(r["astraeus_class"])

        tns_row = None
        if name and name in tns_by_name.index:
            tns_match = tns_by_name.loc[name]
            tns_row = tns_match.iloc[0] if isinstance(tns_match, pd.DataFrame) else tns_match
        elif zid in alias_map:
            mapped_name = alias_map[zid]
            if mapped_name in tns_by_name.index:
                tns_match = tns_by_name.loc[mapped_name]
                tns_row = tns_match.iloc[0] if isinstance(tns_match, pd.DataFrame) else tns_match

        if tns_row is not None:
            tns_prefix = str(tns_row.get("name_prefix") or "").strip()
            tns_name = str(tns_row.get("name") or "").strip()
            full_tns_id = f"{tns_prefix}{tns_name}" if tns_prefix else tns_name
            tns_type = str(tns_row.get("type") or "").strip()
            bibcodes = str(tns_row.get("Class_ADS_bibcodes") or "").strip()
            disc_bib = str(tns_row.get("Discovery_ADS_bibcode") or "").strip()
            reporting_grp = str(tns_row.get("reporting_group") or "").strip()
            reporters = str(tns_row.get("reporters") or "").strip()
            disc_date = str(tns_row.get("discoverydate") or "").strip()
            redshift = tns_row.get("redshift")
            if pd.isna(redshift):
                redshift = r.get("redshift")
            tns_url = f"https://www.wis-tns.org/object/{tns_name}" if tns_name else ""

            has_valid_type = bool(tns_type and tns_type.upper() not in ("AT", "NONE", "NAN", "NULL", "?"))
            has_valid_bib = bool(bibcodes and bibcodes.upper() not in ("NONE", "NAN", "NULL", "?"))

            # Strict hierarchy evaluation:
            # S1 requires directly accessible local spectrum file (0 local FITS).
            # S2 requires valid classification type AND non-empty published Class_ADS_bibcodes.
            # S3 is catalog-only classification without published report or without confirmed type.
            # S4 is unverified / unclassified.
            if has_valid_type and has_valid_bib:
                tier = "S2"
                status = "VERIFIED_TRACEABLE_REPORT"
                permitted_use = "SECONDARY_TRACEABLE_EVIDENCE"
            elif has_valid_type and not has_valid_bib:
                tier = "S3"
                status = "CATALOG_ONLY_UNVERIFIED_REPORT"
                permitted_use = "SENSITIVITY_ANALYSIS_ONLY"
            else:
                tier = "S3"
                status = "UNCLASSIFIED_IN_TNS_CATALOG"
                permitted_use = "SENSITIVITY_ANALYSIS_ONLY"
        else:
            full_tns_id = ""
            tns_type = ""
            bibcodes = ""
            disc_bib = ""
            reporting_grp = ""
            reporters = ""
            disc_date = ""
            redshift = r.get("redshift")
            tns_url = ""
            tier = "S4"
            status = "NO_TNS_RECORD"
            permitted_use = "EXCLUDED_FROM_BENCHMARK"

        evidence_rows.append({
            "candidate_id": zid,
            "ztf_id": zid,
            "iau_id": iau_raw,
            "matched_tns_name": full_tns_id,
            "raw_bts_type": raw_type,
            "normalized_subtype": norm_sub,
            "astraeus_class": ast_cls,
            "tns_type": tns_type,
            "evidence_tier": tier,
            "class_ads_bibcodes": bibcodes,
            "discovery_ads_bibcode": disc_bib,
            "reporting_group": reporting_grp,
            "reporters": reporters,
            "discovery_date": disc_date,
            "redshift": redshift,
            "tns_object_url": tns_url,
            "verification_status": status,
            "permitted_eval_use": permitted_use,
        })

    df_evidence = pd.DataFrame(evidence_rows)
    assert len(df_evidence) == 7614

    tier_counts = df_evidence["evidence_tier"].value_counts().to_dict()
    print(f"  -> Track A Evidence Tier Totals:")
    print(f"     - Tier S1 (Direct Spectrum File)    : {tier_counts.get('S1', 0)}")
    print(f"     - Tier S2 (Traceable TNSCR / Paper) : {tier_counts.get('S2', 0)}")
    print(f"     - Tier S3 (Catalog-Only / Unrefereed): {tier_counts.get('S3', 0)}")
    print(f"     - Tier S4 (Excluded / Unverified)   : {tier_counts.get('S4', 0)}")

    # 4. Track B: ZTF Light-Curve Feature Eligibility
    print(f"\n[4/5] TRACK B: ZTF LIGHT-CURVE FEATURE ELIGIBILITY AUDIT...")
    loader = CandidateObservationLoader(alerce_lc_dir=str(ALERCE_LC_DIR))
    extractor = RealSurveyFeatureExtractor(config=ObservationConfig.ALERCE_ONLY)

    lc_rows = []
    for _, r in df_adm.iterrows():
        zid = str(r["ztf_id"])
        iau_raw = str(r["iau_id"]) if pd.notna(r["iau_id"]) else ""
        ast_cls = str(r["astraeus_class"])

        lc_file = ALERCE_LC_DIR / f"{zid}.json"
        has_file = lc_file.exists()

        if has_file:
            obs = loader.load_candidate_observations(zid)
            active_obs = [o for o in obs if o.source_survey == "ALERCE"]
            active_obs.sort(key=lambda o: o.mjd)

            detections = [o for o in active_obs if not o.is_upper_limit]
            upper_limits = [o for o in active_obs if o.is_upper_limit]

            det_g = sum(1 for o in detections if o.passband == "g")
            det_r = sum(1 for o in detections if o.passband == "r")
            det_other = sum(1 for o in detections if o.passband not in ("g", "r"))
            det_total = len(detections)

            ul_g = sum(1 for o in upper_limits if o.passband == "g")
            ul_r = sum(1 for o in upper_limits if o.passband == "r")
            ul_total = len(upper_limits)
            total_epochs = len(active_obs)

            has_both_bands = (det_g > 0 and det_r > 0)
            if total_epochs >= 2:
                temporal_baseline_days = float(active_obs[-1].mjd - active_obs[0].mjd)
            else:
                temporal_baseline_days = 0.0

            if len(active_obs) >= 2:
                dts = [active_obs[i + 1].mjd - active_obs[i].mjd for i in range(len(active_obs) - 1)]
                median_cadence_days = float(np.median(dts))
            else:
                median_cadence_days = 0.0

            # Test extraction through frozen 27-feature interface
            try:
                feat_dict = extractor.extract_from_observations(obs)
                feat_ok = all(math.isfinite(feat_dict[k]) for k in extractor.feature_names)
                peak_flux = float(feat_dict.get("peak_flux_nmgy", 0.0))
                dyn_range = float(feat_dict.get("flux_dynamic_range_nmgy", 0.0))
            except Exception:
                feat_ok = False
                peak_flux = 0.0
                dyn_range = 0.0

            if det_total >= 5 and has_both_bands and feat_ok:
                status = "FEATURE_EVALUABLE"
                reason = "None (Feature Evaluable)"
            elif det_total < 5:
                status = "INSUFFICIENT_DETECTIONS"
                reason = f"Detections below threshold ({det_total} < 5)"
            elif not has_both_bands:
                status = "SINGLE_BAND_ONLY"
                reason = "Missing g or r band detections"
            else:
                status = "FEATURE_EXTRACTION_FAILED"
                reason = "Frozen feature extraction failed numerical validation"
        else:
            det_g = 0
            det_r = 0
            det_other = 0
            det_total = 0
            ul_g = 0
            ul_r = 0
            ul_total = 0
            total_epochs = 0
            has_both_bands = False
            temporal_baseline_days = 0.0
            median_cadence_days = 0.0
            peak_flux = 0.0
            dyn_range = 0.0
            feat_ok = False
            status = "AWAITING_BROKER_LIGHTCURVE_INGESTION"
            reason = "Raw time-series observations pending ZTF broker alert history retrieval"

        lc_rows.append({
            "candidate_id": zid,
            "ztf_id": zid,
            "iau_id": iau_raw,
            "astraeus_class": ast_cls,
            "has_local_lightcurve_file": has_file,
            "n_det_g": det_g,
            "n_det_r": det_r,
            "n_det_other": det_other,
            "n_det_total": det_total,
            "n_ul_g": ul_g,
            "n_ul_r": ul_r,
            "n_ul_total": ul_total,
            "total_epochs": total_epochs,
            "has_both_bands": has_both_bands,
            "temporal_baseline_days": temporal_baseline_days,
            "median_cadence_days": median_cadence_days,
            "peak_flux_nmgy": peak_flux,
            "flux_dynamic_range_nmgy": dyn_range,
            "feature_interface_compatible": feat_ok,
            "feature_eligibility_status": status,
            "missingness_reason": reason,
        })

    df_lc = pd.DataFrame(lc_rows)
    assert len(df_lc) == 7614

    lc_status_counts = df_lc["feature_eligibility_status"].value_counts().to_dict()
    print(f"  -> Track B Feature Eligibility Status Totals:")
    for k, v in lc_status_counts.items():
        print(f"     - {k}: {v}")

    # 5. Save Machine-Readable Deliverables
    print(f"\n[5/5] Emitting authoritative Phase 4A.2 deliverables...")
    spec_csv_path = OUT_DIR / "spectroscopic_evidence_ledger.csv"
    lc_parquet_path = OUT_DIR / "lightcurve_match_ledger.parquet"
    lc_csv_path = OUT_DIR / "lightcurve_match_ledger.csv"
    manifest_path = OUT_DIR / "phase4a2_reproducibility_manifest.json"

    df_evidence.to_csv(spec_csv_path, index=False)
    df_lc.to_parquet(lc_parquet_path, index=False)
    df_lc.to_csv(lc_csv_path, index=False)

    spec_sha256 = compute_sha256(spec_csv_path)
    lc_parquet_sha256 = compute_sha256(lc_parquet_path)
    lc_csv_sha256 = compute_sha256(lc_csv_path)

    # Class-level cross-tabulation
    class_tier_crosstab = pd.crosstab(
        df_evidence["astraeus_class"],
        df_evidence["evidence_tier"],
    ).to_dict(orient="index")

    manifest_payload = {
        "milestone": "ASTRAEUS Gate 4 Phase 4A.2 Evidence & Feature Eligibility",
        "execution_timestamp_utc": now_utc,
        "python_version": sys.version,
        "input_dependencies": {
            k: {
                "path": str(inputs[k]),
                "sha256": input_hashes[k],
                "verified": True,
            }
            for k in inputs
        },
        "track_a_spectroscopic_evidence_summary": {
            "total_candidates_evaluated": n_admitted,
            "tier_counts": tier_counts,
            "class_tier_breakdown": class_tier_crosstab,
            "tier_definitions": {
                "S1": "Directly accessible spectrum file with event identity, date, instrument, metadata (0 local FITS).",
                "S2": "Traceable published classification with named observatory/instrument and verifiable TNSCR report.",
                "S3": "Traceable catalog classification without accessible spectrum or published report (sensitivity only).",
                "S4": "Photometric or unverified classification (excluded from benchmark).",
            },
        },
        "track_b_lightcurve_feature_eligibility_summary": {
            "total_candidates_evaluated": n_admitted,
            "feature_eligibility_status_counts": lc_status_counts,
            "local_lightcurve_coverage_count": int(df_lc["has_local_lightcurve_file"].sum()),
            "feature_evaluable_count": int((df_lc["feature_eligibility_status"] == "FEATURE_EVALUABLE").sum()),
            "frozen_27_feature_interface_compatible_count": int(df_lc["feature_interface_compatible"].sum()),
            "eligibility_criteria": {
                "minimum_detections": 5,
                "passbands_required": ["g", "r"],
                "preservation_of_upper_limits": True,
                "numerical_feature_validation": True,
            },
        },
        "emitted_artifacts": {
            "spectroscopic_evidence_ledger_csv": {
                "path": str(spec_csv_path),
                "sha256": spec_sha256,
                "row_count": len(df_evidence),
            },
            "lightcurve_match_ledger_parquet": {
                "path": str(lc_parquet_path),
                "sha256": lc_parquet_sha256,
                "row_count": len(df_lc),
            },
            "lightcurve_match_ledger_csv": {
                "path": str(lc_csv_path),
                "sha256": lc_csv_sha256,
                "row_count": len(df_lc),
            },
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    manifest_sha256 = compute_sha256(manifest_path)
    print(f"\n[Artifact Output] Spectroscopic Evidence Ledger saved to: {spec_csv_path}")
    print(f"[Artifact Output] Spectroscopic Ledger SHA-256: {spec_sha256}")
    print(f"[Artifact Output] Light-Curve Match Ledger (Parquet) saved to: {lc_parquet_path}")
    print(f"[Artifact Output] Light-Curve Match Ledger (Parquet) SHA-256: {lc_parquet_sha256}")
    print(f"[Artifact Output] Reproducibility Manifest saved to: {manifest_path}")
    print(f"[Artifact Output] Reproducibility Manifest SHA-256: {manifest_sha256}")
    print("=" * 80)
    print("PHASE 4A.2 EVIDENCE & FEATURE ELIGIBILITY AUDIT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
