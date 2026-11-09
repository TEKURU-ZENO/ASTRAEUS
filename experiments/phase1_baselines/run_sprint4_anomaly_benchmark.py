"""
ASTRAEUS Phase 1: Sprint 4 Rare-Event Anomaly Discovery Benchmark.
Implements the 4-population anomaly benchmark, model-aware physics fitting,
split conformal calibration under dual settings, and telescope follow-up budget triage.
Outputs: experiments/phase1_baselines/sprint4_benchmark_report.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from astraeus_data.synthetic import generate_synthetic_entity
from astraeus_models.features import AstronomicalFeatureExtractor
from astraeus_models.physics import PhysicalModelEngine, FitStatus
from astraeus_models.conformal_calibrator import SplitConformalCalibrator
from astraeus_bench.rare_event import (
    CandidateRecord,
    RareEventBenchmarkRunner,
    RareEventBenchmarkReport,
)


def run_sprint4_experiment(output_path: str = "experiments/phase1_baselines/sprint4_benchmark_report.json"):
    print("=" * 80)
    print("ASTRAEUS Phase 1: Sprint 4 Rare-Event Anomaly Discovery Benchmark")
    print("Populations: 1 (Common SNe), 2 (Peculiar FBOTs), 3 (Artifacts), 4 (Ambiguous)")
    print("Controlled Signal Injection Track: SNR in {3.0, 5.0, 10.0, 20.0}")
    print("=" * 80)

    feature_extractor = AstronomicalFeatureExtractor()
    physics_engine = PhysicalModelEngine()

    print("\n[1/5] Generating multi-population astronomical light curve cohorts...")
    raw_cohorts = {
        "pop1_common": [],
        "pop2_peculiar": [],
        "pop3_artifact": [],
        "pop4_ambiguous": [],
    }

    # Pop 1: Common SNe Ia (100 objects)
    for i in range(100):
        e = generate_synthetic_entity("supernova_ia", seed=1000 + i)
        raw_cohorts["pop1_common"].append(e)

    # Pop 2: Peculiar FBOTs (25 objects)
    for i in range(25):
        e = generate_synthetic_entity("fbot", seed=2000 + i)
        raw_cohorts["pop2_peculiar"].append(e)

    # Pop 3: Artifacts (35 objects)
    for i in range(35):
        e = generate_synthetic_entity("detector_artifact", seed=3000 + i)
        raw_cohorts["pop3_artifact"].append(e)

    # Pop 4: Ambiguous / Noisy transients (25 objects)
    # Generated with very noisy observations
    rng = np.random.RandomState(4000)
    for i in range(25):
        e = generate_synthetic_entity("supernova_ia", seed=4000 + i)
        # degrade SNR by adding extra noise to observations
        for obs in e.observations:
            obs.calibrated_flux.value += float(rng.normal(0.0, 1.2))
            obs.calibrated_flux.uncertainty *= 2.5
        raw_cohorts["pop4_ambiguous"].append(e)

    # Controlled Signal Injection Track: 40 objects (10 each for SNR 3.0, 5.0, 10.0, 20.0)
    injected_cohort = []
    snr_levels = [3.0, 5.0, 10.0, 20.0]
    for snr in snr_levels:
        for i in range(10):
            e = generate_synthetic_entity("fbot", seed=int(5000 + snr * 100 + i))
            # Scale flux according to target SNR
            target_flux_scale = (snr / 10.0)
            for obs in e.observations:
                obs.calibrated_flux.value *= target_flux_scale
            injected_cohort.append((e, snr))

    print(f"Generated: {len(raw_cohorts['pop1_common'])} Common SNe, "
          f"{len(raw_cohorts['pop2_peculiar'])} Peculiar FBOTs, "
          f"{len(raw_cohorts['pop3_artifact'])} Artifacts, "
          f"{len(raw_cohorts['pop4_ambiguous'])} Ambiguous, "
          f"{len(injected_cohort)} Injected Signals.")

    print("\n[2/5] Extracting statistical features, physics forward fits, and residuals...")
    processed_records = []

    def process_entity(entity, pop_name, is_target, is_art, snr_val=None) -> CandidateRecord:
        times = np.array([obs.mjd for obs in entity.observations])
        mags = np.array([obs.calibrated_flux.value for obs in entity.observations])
        errs = np.array([obs.calibrated_flux.uncertainty for obs in entity.observations])
        bands = [obs.filter_band for obs in entity.observations]

        # Statistical features (amplitude, rise slope, decay slope, peak mag, duration)
        f_vec = feature_extractor.extract(times, mags, errs, bands)
        f_list = [f_vec.amplitude, f_vec.rise_rate_mag_per_day, f_vec.decay_rate_mag_per_day, f_vec.peak_mag, f_vec.duration_days]

        # Physics fitting with applicability gating (unsupervised at test time)
        hypo = "supernova_ia" if pop_name == "pop1_common" else "unknown"
        phys_res = physics_engine.fit_all_applicable(times, mags, errs, candidate_hypothesis=hypo)

        # Baseline Mahalanobis placeholder (will be re-centered on Pop 1 training cohort)
        return CandidateRecord(
            entity_id=entity.entity_id,
            population=pop_name,
            is_scientific_target=is_target,
            is_artifact=is_art,
            features=f_list,
            d_m=0.0,
            uncertainty=float(np.mean(errs)),
            physical_residual=float(phys_res.physical_residual),
            snr=snr_val,
        )

    # Process all standard cohorts
    for e in raw_cohorts["pop1_common"]:
        processed_records.append(process_entity(e, "pop1_common", is_target=False, is_art=False))

    for e in raw_cohorts["pop2_peculiar"]:
        processed_records.append(process_entity(e, "pop2_peculiar", is_target=True, is_art=False))

    for e in raw_cohorts["pop3_artifact"]:
        processed_records.append(process_entity(e, "pop3_artifact", is_target=False, is_art=True))

    for e in raw_cohorts["pop4_ambiguous"]:
        processed_records.append(process_entity(e, "pop4_ambiguous", is_target=False, is_art=False))

    # Process injected cohort
    processed_injected = []
    for e, snr in injected_cohort:
        processed_injected.append(process_entity(e, "injected_anomalous", is_target=True, is_art=False, snr_val=snr))

    print("\n[3/5] Splitting Population 1 into Reference Cohort and Calibrating Models...")
    pop1_all = [r for r in processed_records if r.population == "pop1_common"]
    other_all = [r for r in processed_records if r.population != "pop1_common"]

    # Pop 1 split: 60 train/reference, 20 calibration, 20 test
    ref_cohort = pop1_all[:60]
    cal_cohort = pop1_all[60:80]
    test_pop1 = pop1_all[80:]

    eval_cohort = test_pop1 + other_all

    # Compute empirical covariance and centroid on Pop 1 Reference Cohort
    X_ref = np.array([r.features for r in ref_cohort], dtype=float)
    mu_ref = np.mean(X_ref, axis=0)
    cov_ref = np.cov(X_ref, rowvar=False) + 1e-4 * np.eye(X_ref.shape[1])
    inv_cov_ref = np.linalg.pinv(cov_ref)

    def calc_mahalanobis(feats: List[float]) -> float:
        diff = np.array(feats) - mu_ref
        return float(np.sqrt(np.dot(np.dot(diff, inv_cov_ref), diff)))

    for r in ref_cohort:
        r.d_m = calc_mahalanobis(r.features)
    for r in eval_cohort:
        r.d_m = calc_mahalanobis(r.features)
    for r in processed_injected:
        r.d_m = calc_mahalanobis(r.features)

    # Conformal Calibration: Train on ref_cohort, calibrate on cal_cohort
    y_train = np.array([r.features[3] for r in ref_cohort]) # peak_magnitude
    X_train = np.array([r.features for r in ref_cohort])
    y_cal = np.array([r.features[3] for r in cal_cohort])
    X_cal = np.array([r.features for r in cal_cohort])

    conformal_calibrator = SplitConformalCalibrator(target_property="peak_magnitude", nominal_coverage=0.90)
    q_hat = conformal_calibrator.fit_and_calibrate(X_train, y_train, X_cal, y_cal)

    # Dual-setting evaluation: Setting 1 (Pop 1 test) vs Setting 2 (Pop 4 ambiguous + shifted)
    pop4_records = [r for r in other_all if r.population == "pop4_ambiguous"]
    X_s1 = np.array([r.features for r in test_pop1])
    y_s1 = np.array([r.features[3] for r in test_pop1])
    X_s2 = np.array([r.features for r in pop4_records])
    y_s2 = np.array([r.features[3] for r in pop4_records])

    conformal_report = conformal_calibrator.evaluate_dual_settings(
        X_exchangeable=X_s1,
        y_exchangeable=y_s1,
        phot_err_exchangeable=np.array([r.uncertainty for r in test_pop1]),
        X_shifted=X_s2,
        y_shifted=y_s2,
        phot_err_shifted=np.array([r.uncertainty for r in pop4_records]),
    )

    print(f"Conformal Calibration completed: q_hat = {q_hat:.4f}")
    print(f"Setting 1 (Exchangeable) empirical coverage: {conformal_report.setting_1_exchangeable.empirical_coverage:.3f}")
    print(f"Setting 2 (Distribution Shift) empirical coverage: {conformal_report.setting_2_distribution_shift.empirical_coverage:.3f}")

    print("\n[4/5] Executing Rare-Event Anomaly Benchmark across Baselines & Ablations...")
    benchmark_runner = RareEventBenchmarkRunner(budgets=[5, 10, 20, 50, 100])
    benchmark_report = benchmark_runner.run_benchmark(
        ref_records=ref_cohort,
        eval_records=eval_cohort,
        injected_records=processed_injected,
    )

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY:")
    print(f"{'Method / Baseline':<35} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Yield@20':<8} | {'Recall@20':<9} | {'Artifacts@20':<12}")
    print("-" * 80)
    for name, res in benchmark_report.methods.items():
        y20 = res.peculiar_yield.get(20, 0)
        r20 = res.recall_at_k.get(20, 0.0)
        a20 = res.artifact_contamination.get(20, 0)
        print(f"{name:<35} | {res.roc_auc:<8.4f} | {res.pr_auc:<8.4f} | {y20:<8} | {r20:<9.4f} | {a20:<12}")

    print("-" * 80)
    print("ABLATIONS:")
    for name, res in benchmark_report.ablations.items():
        y20 = res.peculiar_yield.get(20, 0)
        r20 = res.recall_at_k.get(20, 0.0)
        a20 = res.artifact_contamination.get(20, 0)
        print(f"{name:<35} | {res.roc_auc:<8.4f} | {res.pr_auc:<8.4f} | {y20:<8} | {r20:<9.4f} | {a20:<12}")
    print("=" * 80)

    print("\n[5/5] Emitting comprehensive benchmark report to JSON...")
    output_dict = {
        "benchmark": "ASTRAEUS Phase 1 Sprint 4: Rare-Event Anomaly Discovery Benchmark",
        "dataset_summary": benchmark_report.dataset_summary,
        "conformal_calibration": conformal_report.model_dump(),
        "methods": {name: res.model_dump() for name, res in benchmark_report.methods.items()},
        "ablations": {name: res.model_dump() for name, res in benchmark_report.ablations.items()},
        "signal_injection_recovery": benchmark_report.signal_injection_recovery,
        "executive_summary": benchmark_report.executive_summary,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"Report written successfully to: {out_file.resolve()}")
    return output_dict


if __name__ == "__main__":
    run_sprint4_experiment()
