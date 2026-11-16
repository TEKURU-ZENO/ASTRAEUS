"""
ASTRAEUS Gate 4: Final Evidence Manifest Generator.

Author: Dev Mehta
Date: November 16, 2026
Milestone: Gate 4 Final Scientific Closure

Aggregates all file hashes, dataset checksums, model fingerprints,
split configurations, deliverable audit entries, and workstream dispositions
across Gate 4 (4A, 4B, 4C, 4D, 4E, Synthesis).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


DELIVERABLES_CATALOG: List[Tuple[str, str, str]] = [
    # Work Package 4A: Cohort Independence, Screening & Eligibility
    ("configs/gate4/gate4a_taxonomy_mapping.yaml", "Phase_4A", "Taxonomy mapping YAML definition for 8-class real survey schema"),
    ("configs/gate4/gate4a_raw_label_reconciliation.csv", "Phase_4A", "Raw BTS 33-label reconciliation table (Approved, Quarantined, Unmapped)"),
    ("experiments/gate4a_external_cohort_preregistration/gate4a1_audit_metrics.json", "Phase_4A", "Phase 4A.1 screening attrition metrics and set arithmetic counts"),
    ("experiments/gate4a_external_cohort_preregistration/gate4a1_event_accounting_ledger.csv", "Phase_4A", "10,988-row complete event accounting ledger (CSV)"),
    ("experiments/gate4a_external_cohort_preregistration/gate4a1_event_accounting_ledger.parquet", "Phase_4A", "10,988-row complete event accounting ledger (Parquet)"),
    ("experiments/gate4a_external_cohort_preregistration/gate4a1_screened_admitted_candidates.csv", "Phase_4A", "7,614 taxonomy-admitted external candidate subset"),
    ("experiments/gate4a_external_cohort_preregistration/GATE_4A_CLASS_SUPPORT_REPORT.md", "Phase_4A", "Class support and statistical power report across external classes"),
    ("experiments/gate4a_external_cohort_preregistration/GATE_4A_COHORT_INDEPENDENCE_CERTIFICATION.md", "Phase_4A", "Primary cohort independence and sequential attrition certification"),
    ("experiments/gate4a_external_cohort_preregistration/GATE_4A_HEALPIX_FOOTPRINT_VERIFICATION.md", "Phase_4A", "HEALPix spatial exclusion footprint verification report"),
    ("experiments/gate4a_external_cohort_preregistration/run_phase4a1_cohort_integrity_audit.py", "Phase_4A", "Executable audit script for Phase 4A.1 cohort screening"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/spectroscopic_evidence_ledger.csv", "Phase_4A", "Track A spectroscopic classification evidence ledger (7,614 records)"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.parquet", "Phase_4A", "Track B photometric evaluability ledger (Parquet, 7,614 records)"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.csv", "Phase_4A", "Track B photometric evaluability ledger (CSV, 7,614 records)"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/feature_eligibility_report.md", "Phase_4A", "Detailed feature eligibility analysis report across Track A and B"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/GATE_4A2_ELIGIBILITY_CERTIFICATION.md", "Phase_4A", "Phase 4A.2 formal eligibility certification sign-off"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/phase4a2_reproducibility_manifest.json", "Phase_4A", "Cryptographic reproducibility manifest for Phase 4A.2"),
    ("experiments/gate4a2_evidence_and_feature_eligibility/run_phase4a2_eligibility_audit.py", "Phase_4A", "Executable audit script for Phase 4A.2 eligibility verification"),

    # Work Package 4B: Broker Acquisition & Full Cohort Accounting
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B_BROKER_ACQUISITION_AND_COHORT_EXPANSION_PROTOCOL.md", "Phase_4B", "Protocol specification for controlled broker acquisition (v1.3)"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/pilot_cohort_specification.json", "Phase_4B", "Pre-registered 35-candidate pilot cohort specification"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/phase4b1_photometric_audit.csv", "Phase_4B", "Controlled 35-candidate photometric acquisition audit ledger"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B1_PILOT_ACQUISITION_AUDIT_REPORT.md", "Phase_4B", "Controlled pilot acquisition audit and reproducibility report"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/phase4b2_full_cohort_accounting_ledger.csv", "Phase_4B", "Full external cohort accounting ledger (CSV, 7,614 records)"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/phase4b2_full_cohort_accounting_ledger.parquet", "Phase_4B", "Full external cohort accounting ledger (Parquet, 7,614 records)"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/phase4b2_full_cohort_integrity_manifest.json", "Phase_4B", "Integrity manifest for full external cohort accounting"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/GATE_4B2_FULL_COHORT_ACQUISITION_AUDIT.md", "Phase_4B", "Full external cohort accounting and provenance reconciliation audit report"),
    ("experiments/gate4b_broker_acquisition_and_cohort_expansion/run_phase4b2_full_cohort_audit.py", "Phase_4B", "Executable audit script for Phase 4B.2 full cohort accounting"),

    # Work Package 4C: Rare-Class Representation & Failure Diagnosis
    ("experiments/gate4c_rare_class_and_failure_diagnosis/phase4c_rare_class_failure_ledger.csv", "Phase_4C", "Diagnosed event-level rare-class failure ledger (156 events)"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/phase4c_rare_class_metrics.json", "Phase_4C", "Per-class performance metrics with Wilson 95% confidence intervals"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/phase4c_subtype_hierarchy_audit.json", "Phase_4C", "Subtype hierarchy audit and boundary ambiguity analysis"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/phase4c_rare_class_ablation_results.json", "Phase_4C", "Feature ablation impact results across rare transient classes"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/GATE_4C_RARE_CLASS_EVIDENCE_AND_COHORT_REPORT.md", "Phase_4C", "Rare-class representation evidence and cohort distribution report"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/phase4c_open_set_and_abstention_report.md", "Phase_4C", "Open-set risk analysis and rare-class abstention paradox report"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/GATE_4C_SCIENTIFIC_CLOSURE.md", "Phase_4C", "Work Package 4C scientific closure report and sign-off"),
    ("experiments/gate4c_rare_class_and_failure_diagnosis/run_phase4c_rare_class_analysis.py", "Phase_4C", "Executable engine for Phase 4C rare-class and failure analysis"),

    # Work Package 4D: Early-Epoch Physics & SN_CC/SN_Ia Disambiguation
    ("experiments/gate4d_early_epoch_physics/phase4d_horizon_benchmark.json", "Phase_4D", "Quantitative early-epoch observation horizon benchmark metrics"),
    ("experiments/gate4d_early_epoch_physics/phase4d_sncc_snia_error_analysis.csv", "Phase_4D", "Detailed event error analysis for early horizon N=10"),
    ("experiments/gate4d_early_epoch_physics/phase4d_physical_feature_dictionary.json", "Phase_4D", "Dictionary of 27 physical features for early-epoch disambiguation"),
    ("experiments/gate4d_early_epoch_physics/phase4d_physical_ablation_results.json", "Phase_4D", "Feature ablation results comparing physical models vs baselines"),
    ("experiments/gate4d_early_epoch_physics/phase4d_early_epoch_cohort_manifest.parquet", "Phase_4D", "Early-epoch supernova cohort manifest (2,083 candidates)"),
    ("experiments/gate4d_early_epoch_physics/GATE_4D_EARLY_EPOCH_PHYSICS_PROTOCOL.md", "Phase_4D", "Protocol for early-epoch physical feature extraction and horizons"),
    ("experiments/gate4d_early_epoch_physics/GATE_4D_SCIENTIFIC_REPORT.md", "Phase_4D", "Work Package 4D scientific synthesis report on early physics horizons"),
    ("experiments/gate4d_early_epoch_physics/run_phase4d_early_epoch_physics.py", "Phase_4D", "Executable benchmark script for Phase 4D early-epoch horizons"),

    # Work Package 4E: Shift-Aware Uncertainty & Domain Transfer
    ("experiments/gate4e_shift_aware_uncertainty/phase4e_conformal_calibration_report.json", "Phase_4E", "Split-conformal calibration specification (k=507) and coverage metrics"),
    ("experiments/gate4e_shift_aware_uncertainty/phase4e_shift_stratified_metrics.parquet", "Phase_4E", "Shift-stratified empirical coverage metrics across cadence and epochs"),
    ("experiments/gate4e_shift_aware_uncertainty/phase4e_shift_factor_manifest.json", "Phase_4E", "Manifest of astronomical domain shift factors and risks"),
    ("experiments/gate4e_shift_aware_uncertainty/phase4e_cross_survey_transfer_results.json", "Phase_4E", "Processing-domain transfer results: ZTF Alert vs DR24 coadds"),
    ("experiments/gate4e_shift_aware_uncertainty/phase4e_negative_transfer_abstention_report.md", "Phase_4E", "Negative transfer analysis and validated operating envelope rules"),
    ("experiments/gate4e_shift_aware_uncertainty/GATE_4E_SHIFT_AND_UNCERTAINTY_PROTOCOL.md", "Phase_4E", "Protocol for shift-aware uncertainty and conformal calibration"),
    ("experiments/gate4e_shift_aware_uncertainty/GATE_4E_SCIENTIFIC_CLOSURE.md", "Phase_4E", "Work Package 4E scientific closure report and sign-off"),
    ("experiments/gate4e_shift_aware_uncertainty/run_phase4e_uncertainty_and_shift.py", "Phase_4E", "Executable engine for Phase 4E uncertainty, conformal, and shift audit"),

    # Final Synthesis & Governance
    ("experiments/GATE_4_FINAL_GOVERNANCE_AUDIT.md", "Synthesis", "Final governance, quarantine, and population separation audit report"),
    ("experiments/GATE_4_FINAL_SCIENTIFIC_CLOSURE_REPORT.md", "Synthesis", "Final scientific closure and external validity synthesis report"),
    ("tests/test_gate4_final_closure.py", "Synthesis", "Gate 4 final closure regression verification test suite")
]


def generate_manifest():
    repo_root = Path(__file__).resolve().parents[1]

    # Process all deliverables
    deliverables_records = {}
    for rel_path, wp, desc in DELIVERABLES_CATALOG:
        full_path = repo_root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Deliverable missing from repository: {rel_path}")
        deliverables_records[rel_path] = {
            "work_package": wp,
            "description": desc,
            "sha256": compute_sha256(full_path),
            "size_bytes": os.path.getsize(full_path),
        }

    manifest = {
        "manifest_title": "ASTRAEUS Gate 4 Final Evidence & Governance Manifest",
        "milestone": "Gate 4 — External Validity & Scientific Robustness: Final Closure",
        "author": "Dev Mehta",
        "timestamp_utc": "2026-11-16T14:15:00Z",
        "git_head_base_commit": "113f26d965b65ebd8f0de2ec76b4ac17afcd0619",
        "governance_locks": {
            "gate3_test_cohort_quarantined": True,
            "gate3_quarantined_candidate_count": 586,
            "gate3_quarantined_test_pixels": 1808,
            "gate3_quarantine_verification_method": "Non-disclosing HEALPix geometric disjointness proof & custodian release manifest seal",
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
                "title": "Full External Cohort Accounting & Provenance Reconciliation",
                "disposition": "CLOSED",
                "key_metrics": {
                    "total_accounted": 7614,
                    "pilot_ingested_lightcurves": 35,
                    "local_baseline_lightcurves": 8,
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
                    "horizon_n20_sncc_precision": 0.8500,
                    "horizon_n20_sncc_recall": 0.1417,
                    "horizon_full_macro_f1": 0.6754,
                    "earliest_defensible_horizon": "Horizon N=20 (Early high-precision 85% triage signal; balanced discrimination requires full post-peak light curve)",
                },
                "primary_artifact": "experiments/gate4d_early_epoch_physics/GATE_4D_SCIENTIFIC_REPORT.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4d_early_epoch_physics/GATE_4D_SCIENTIFIC_REPORT.md"),
            },
            "Phase_4E": {
                "title": "Shift-Aware Uncertainty, Conformal Calibration & Domain Transfer",
                "disposition": "CLOSED",
                "key_metrics": {
                    "calibration_population_size": 562,
                    "conformal_rank_k": 507,
                    "target_coverage": 0.90,
                    "nominal_calibration_coverage": 0.9004,
                    "unshifted_validation_coverage": 0.7139,
                    "sparse_observation_coverage": 0.5827,
                    "in_envelope_coverage_mitigated": "76-78% (78.2% for N>=20, 76.7% for cadence<=3d)",
                    "guaranteed_90_percent_coverage_under_shift": False,
                    "cross_pipeline_transfer_accuracy_delta": -0.044,
                    "cross_survey_replication_certified": False,
                    "operating_envelope_established": True,
                },
                "primary_artifact": "experiments/gate4e_shift_aware_uncertainty/GATE_4E_SCIENTIFIC_CLOSURE.md",
                "primary_artifact_sha256": compute_sha256(repo_root / "experiments/gate4e_shift_aware_uncertainty/GATE_4E_SCIENTIFIC_CLOSURE.md"),
            },
        },
        "deliverables": deliverables_records,
        "final_gate_disposition": "GATE 4 OFFICIALLY CLOSED",
    }

    manifest_path = repo_root / "experiments" / "gate4_final_evidence_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Gate 4 Final Evidence Manifest Generated ({len(deliverables_records)} deliverables hashed) -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_manifest()
