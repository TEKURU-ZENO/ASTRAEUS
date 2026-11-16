"""
ASTRAEUS Gate 4: Final Evidence Manifest Generator.

Author: Dev Mehta
Date: November 16, 2026
Milestone: Gate 4 Final Scientific Closure

Aggregates all file hashes, dataset checksums, model fingerprints,
split configurations, and workstream dispositions across Gate 4 (4A, 4B, 4C, 4D, 4E).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest():
    repo_root = Path(__file__).resolve().parents[1]

    manifest = {
        "manifest_title": "ASTRAEUS Gate 4 Final Evidence & Governance Manifest",
        "milestone": "Gate 4 — External Validity & Scientific Robustness: Final Closure",
        "author": "Dev Mehta",
        "timestamp_utc": "2026-11-16T13:30:00Z",
        "git_head_base_commit": "0d888720e88eb1ffa728f608941ef9f37e3c0d0e",
        "governance_locks": {
            "gate3_test_cohort_quarantined": True,
            "gate3_quarantined_candidate_count": 586,
            "gate3_quarantined_test_pixels": 1808,
            "gate3_unblinding_leaks_detected": 0,
            "champion_model_frozen": True,
            "champion_model_name": "CalibratedBalancedSubsampleRF",
            "champion_model_path": "experiments/sprint4h_final_release_remediation/sprint4h_champion_model.joblib",
            "champion_model_sha256": "466c2752808be01e4f41bd8fdb9c0804a57b03b2c02c35e5172ccebda411fce5",
        },
        "datasets": {
            "bts_raw_snapshot": {
                "path": "data/external_validity_v1/raw/bts/bts_raw_snapshot.csv",
                "sha256": compute_sha256(repo_root / "data/external_validity_v1/raw/bts/bts_raw_snapshot.csv"),
                "records": 10988,
            },
            "candidates_10k": {
                "path": "data/processed/real_survey_10k_v1/candidates_10k.parquet",
                "sha256": compute_sha256(repo_root / "data/processed/real_survey_10k_v1/candidates_10k.parquet"),
                "records": 10000,
            },
            "healpix_splits": {
                "path": "configs/data_splits_healpix.json",
                "sha256": compute_sha256(repo_root / "configs/data_splits_healpix.json"),
                "test_pixels": 1808,
            },
        },
        "workstreams": {
            "Phase_4A": {
                "title": "Cohort Integrity & Geometric Attrition Reconciliation",
                "disposition": "CLOSED",
                "key_metrics": {
                    "raw_screened": 10988,
                    "total_excluded": 3372,
                    "retained_screened": 7616,
                    "taxonomy_admitted": 7614,
                },
                "primary_artifact": "experiments/gate4a_external_cohort_preregistration/GATE_4A_COHORT_INDEPENDENCE_CERTIFICATION.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4a_external_cohort_preregistration/GATE_4A_COHORT_INDEPENDENCE_CERTIFICATION.md"),
            },
            "Phase_4B1": {
                "title": "Controlled 35-Candidate Broker Acquisition Pilot",
                "disposition": "CLOSED",
                "key_metrics": {
                    "requested": 35,
                    "retrieved_http_200": 35,
                    "retries": 0,
                    "effective_rate_req_per_sec": 0.523,
                    "rate_ceiling_violations": 0,
                    "reproducibility_delta_on_5_refs": 0.0,
                },
                "primary_artifact": "experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B1_PILOT_ACQUISITION_AUDIT_REPORT.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B1_PILOT_ACQUISITION_AUDIT_REPORT.md"),
            },
            "Phase_4B2": {
                "title": "Full External Cohort Acquisition Audit & Accounting",
                "disposition": "CLOSED",
                "key_metrics": {
                    "total_accounted": 7614,
                    "pilot_ingested": 35,
                    "local_baseline_ingested": 8,
                    "staged_catalog_accounted": 7571,
                    "rare_candidates_accounted": 4,
                },
                "primary_artifact": "experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B2_FULL_COHORT_ACQUISITION_AUDIT.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B2_FULL_COHORT_ACQUISITION_AUDIT.md"),
            },
            "Phase_4C": {
                "title": "Rare-Class Representation, Failure Diagnosis & Open-Set Analysis",
                "disposition": "CLOSED",
                "key_metrics": {
                    "evaluable_classes": ["SN_Ia", "SN_CC"],
                    "descriptive_only_classes": ["SLSN", "CV_Nova", "AGN_QSO", "Varstar"],
                    "insufficient_external_support_classes": ["TDE", "OTHER_EXTRAGALACTIC"],
                    "other_extragalactic_recall": 0.0,
                    "tde_recall": 0.375,
                    "sn_ia_f1": 0.7817,
                    "sn_cc_f1": 0.5455,
                },
                "primary_artifact": "experiments/gate4c_rare_class_and_failure_diagnosis/GATE_4C_SCIENTIFIC_CLOSURE.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4c_rare_class_and_failure_diagnosis/GATE_4C_SCIENTIFIC_CLOSURE.md"),
            },
            "Phase_4D": {
                "title": "Early-Epoch Physics & SN_CC / SN_Ia Disambiguation",
                "disposition": "CLOSED",
                "key_metrics": {
                    "horizon_n5_macro_f1": 0.3855,
                    "horizon_n10_macro_f1": 0.3855,
                    "horizon_n20_macro_f1": 0.5163,
                    "horizon_full_macro_f1": 0.6754,
                    "earliest_defensible_horizon": "Horizon N=20 (Pre-peak discrimination unsupported at N<=10)",
                },
                "primary_artifact": "experiments/gate4d_early_epoch_physics/GATE_4D_SCIENTIFIC_REPORT.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4d_early_epoch_physics/GATE_4D_SCIENTIFIC_REPORT.md"),
            },
            "Phase_4E": {
                "title": "Shift-Aware Uncertainty, Conformal Calibration & Cross-Survey Robustness",
                "disposition": "CLOSED",
                "key_metrics": {
                    "calibration_population_size": 562,
                    "conformal_rank_k": 507,
                    "target_coverage": 0.90,
                    "nominal_calibration_coverage": 0.9004,
                    "unshifted_validation_coverage": 0.7139,
                    "sparse_observation_coverage": 0.5827,
                    "cross_survey_transfer_accuracy_delta": -0.044,
                    "operating_envelope_established": True,
                },
                "primary_artifact": "experiments/gate4e_shift_aware_uncertainty/GATE_4E_SCIENTIFIC_CLOSURE.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4e_shift_aware_uncertainty/GATE_4E_SCIENTIFIC_CLOSURE.md"),
            },
        },
        "final_gate_disposition": "GATE 4 OFFICIALLY CLOSED",
    }

    manifest_path = repo_root / "experiments" / "gate4_final_evidence_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Gate 4 Final Evidence Manifest Generated -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_manifest()
