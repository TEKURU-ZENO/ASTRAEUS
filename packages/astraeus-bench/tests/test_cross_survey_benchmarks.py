"""
Unit tests for ASTRAEUS Cross-Survey Benchmarks.
Tests Physical Parameter Consistency (Exp A), Rank Stability (Exp B), and Negative Transfer Abstention (Exp C).
"""

import numpy as np
import pytest

from astraeus_bench.cross_survey import (
    PhysicalParameterConsistencyRecord,
    RankStabilityRecord,
    NegativeTransferRecord,
    CrossSurveyBenchmarkRunner,
)
from astraeus_models.physics.bazin import bazin_function


def test_cross_survey_physical_consistency():
    runner = CrossSurveyBenchmarkRunner()

    # Generate matched light curve for a supernova observed by both ZTF and Pan-STARRS
    t_ztf = np.linspace(60000.0, 60040.0, 20)
    f_ztf = bazin_function(t_ztf, t0=60015.0, a=50.0, b=2.0, tau_rise=3.5, tau_fall=14.0)
    e_ztf = np.full_like(f_ztf, 0.4)

    # Target (PS1) observation has different cadences and noise
    t_ps1 = np.linspace(60001.0, 60039.0, 16)
    f_ps1 = bazin_function(t_ps1, t0=60015.0, a=50.0, b=2.0, tau_rise=3.5, tau_fall=14.0) + np.random.RandomState(42).normal(0.0, 0.3, size=len(t_ps1))
    e_ps1 = np.full_like(f_ps1, 0.5)

    matched_data = [
        {
            "id": "matched_sn_001",
            "class": "supernova_ia",
            "ztf": (t_ztf, f_ztf, e_ztf),
            "target": (t_ps1, f_ps1, e_ps1),
            "target_name": "Pan-STARRS1",
        }
    ]

    records = runner.evaluate_physical_consistency(matched_data)
    assert len(records) == 1
    rec = records[0]
    assert rec.object_id == "matched_sn_001"
    assert rec.preferred_model_ztf in {"bazin", "arnett"}
    assert rec.preferred_model_target in {"bazin", "arnett"}
    assert rec.mean_relative_parameter_error < 0.25
    assert rec.is_physically_consistent is True


def test_cross_survey_rank_stability():
    runner = CrossSurveyBenchmarkRunner()
    rng = np.random.RandomState(42)

    # 30 matched objects with correlated anomaly scores across surveys
    ztf_scores = rng.uniform(0.0, 1.0, size=30)
    # Target survey has slight noise shift
    target_scores = ztf_scores + rng.normal(0.0, 0.1, size=30)

    rec = runner.evaluate_rank_stability("DECam", ztf_scores, target_scores)
    assert isinstance(rec, RankStabilityRecord)
    assert rec.target_survey_name == "DECam"
    assert rec.spearman_rho > 0.80
    assert rec.top_10_jaccard > 0.50
    assert rec.stability_verdict in {"STABLE", "MODERATE_SHIFT"}


def test_negative_transfer_and_calibrated_abstention():
    runner = CrossSurveyBenchmarkRunner()

    # Ground truth: 20 samples, 2 classes (0 and 1)
    y_true = np.array([0] * 10 + [1] * 10)
    # In-domain F1 is perfect (1.0)
    in_domain_f1 = 1.0

    # On target survey, severe covariate shift corrupted the last 5 predictions
    preds_target = np.array([0] * 10 + [1] * 5 + [0] * 5)
    # Shift distances: first 15 have low shift <= 2.0, last 5 have extreme shift = 6.0
    shift_dists = np.array([1.2] * 15 + [6.0] * 5)

    rec = runner.evaluate_negative_transfer_and_abstention(
        target_name="Pan-STARRS1",
        in_domain_f1=in_domain_f1,
        model_predictions=preds_target,
        true_labels=y_true,
        shift_distances=shift_dists,
        abstention_threshold=3.0,
    )

    assert isinstance(rec, NegativeTransferRecord)
    assert rec.zero_shot_f1 < 1.0
    assert rec.f1_degradation > 0.0
    assert rec.abstention_rate == pytest.approx(0.25) # 5 / 20 = 25% abstained
    assert rec.post_abstention_f1 > rec.zero_shot_f1 # Abstaining from corrupted high-shift samples recovered F1
    assert rec.negative_transfer_mitigated is True
