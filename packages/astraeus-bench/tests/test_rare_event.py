"""
Unit tests for ASTRAEUS Rare-Event Anomaly Discovery Benchmark Suite.
Tests multi-population ranking, frozen reference CDFs, baselines, and budget triage.
"""

import numpy as np
import pytest

from astraeus_bench.rare_event import (
    CandidateRecord,
    MethodRankingResult,
    RareEventBenchmarkReport,
    RareEventBenchmarkRunner,
)


def create_synthetic_benchmark_records():
    rng = np.random.RandomState(42)
    records = []

    # Population 1: Common transients (e.g. SNe Ia) - 100 objects
    # Centered around (0, 0), low d_m, low uncertainty, low physical residual
    for i in range(100):
        records.append(
            CandidateRecord(
                entity_id=f"pop1_common_{i:03d}",
                population="pop1_common",
                is_scientific_target=False,
                is_artifact=False,
                features=(rng.normal(0.0, 0.5, size=5)).tolist(),
                d_m=float(rng.exponential(1.0)),
                uncertainty=float(rng.uniform(0.1, 0.3)),
                physical_residual=float(rng.exponential(0.8)),
            )
        )

    # Population 2: Peculiar transients (e.g. FBOTs) - 15 objects
    # High d_m, moderate uncertainty, high physical residual
    for i in range(15):
        records.append(
            CandidateRecord(
                entity_id=f"pop2_peculiar_{i:03d}",
                population="pop2_peculiar",
                is_scientific_target=True,
                is_artifact=False,
                features=(rng.normal(3.5, 0.8, size=5)).tolist(),
                d_m=float(rng.uniform(4.0, 8.0)),
                uncertainty=float(rng.uniform(0.3, 0.6)),
                physical_residual=float(rng.uniform(3.0, 7.0)),
            )
        )

    # Population 3: Artifacts - 25 objects
    # High d_m, high uncertainty, failed physical fit
    for i in range(25):
        records.append(
            CandidateRecord(
                entity_id=f"pop3_artifact_{i:03d}",
                population="pop3_artifact",
                is_scientific_target=False,
                is_artifact=True,
                features=(rng.normal(6.0, 1.5, size=5)).tolist(),
                d_m=float(rng.uniform(5.0, 10.0)),
                uncertainty=float(rng.uniform(0.8, 1.5)),
                physical_residual=float(rng.uniform(5.0, 12.0)),
            )
        )

    # Population 4: Ambiguous - 20 objects
    for i in range(20):
        records.append(
            CandidateRecord(
                entity_id=f"pop4_ambiguous_{i:03d}",
                population="pop4_ambiguous",
                is_scientific_target=False,
                is_artifact=False,
                features=(rng.normal(1.5, 1.0, size=5)).tolist(),
                d_m=float(rng.uniform(1.5, 3.0)),
                uncertainty=float(rng.uniform(0.4, 0.8)),
                physical_residual=float(rng.uniform(1.0, 2.5)),
            )
        )

    # Injected Anomalies - 10 objects (at SNR = 10.0)
    injected = []
    for i in range(10):
        injected.append(
            CandidateRecord(
                entity_id=f"injected_{i:03d}",
                population="injected_anomalous",
                is_scientific_target=True,
                is_artifact=False,
                features=(rng.normal(4.0, 0.5, size=5)).tolist(),
                d_m=float(rng.uniform(4.5, 7.5)),
                uncertainty=float(rng.uniform(0.2, 0.4)),
                physical_residual=float(rng.uniform(3.5, 6.0)),
                snr=10.0,
            )
        )

    return records, injected


def test_rare_event_benchmark_runner():
    records, injected = create_synthetic_benchmark_records()

    # Split: pop1 into reference (50) and eval (remaining 50)
    pop1_cands = [r for r in records if r.population == "pop1_common"]
    other_cands = [r for r in records if r.population != "pop1_common"]

    ref_cohort = pop1_cands[:50]
    eval_cohort = pop1_cands[50:] + other_cands

    runner = RareEventBenchmarkRunner(budgets=[5, 10, 20, 50])
    report = runner.run_benchmark(ref_cohort, eval_cohort, injected_records=injected)

    assert isinstance(report, RareEventBenchmarkReport)
    assert "Isolation_Forest" in report.methods
    assert "Local_Outlier_Factor" in report.methods
    assert "Centroid_Euclidean_Distance" in report.methods
    assert "ASTRAEUS_Calibrated_Composite" in report.methods

    # Ablations check
    assert "Ablation_No_Physics" in report.ablations
    assert "Ablation_No_Uncertainty" in report.ablations
    assert "Ablation_No_Latent" in report.ablations

    # Verify ASTRAEUS Calibrated Composite suppresses artifacts and achieves high ROC-AUC
    ast_res = report.methods["ASTRAEUS_Calibrated_Composite"]
    assert ast_res.roc_auc > 0.70
    assert 10 in ast_res.peculiar_yield
    assert ast_res.peculiar_yield[10] > 0

    # Injection analysis
    assert "SNR_10.0" in report.signal_injection_recovery
    assert report.signal_injection_recovery["SNR_10.0"]["astraeus_recall_top50"] > 0.0
