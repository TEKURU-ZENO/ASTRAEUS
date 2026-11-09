"""
Unit tests for ASTRAEUS Split Conformal Calibration Engine.
Verifies finite-sample quantile validity, uncertainty decomposition, and dual-setting evaluation.
"""

import numpy as np
import pytest

from astraeus_models.conformal_calibrator import (
    SplitConformalCalibrator,
    ConformalEvaluationSettingResult,
    ConformalCalibrationReport,
)


def test_split_conformal_calibration_guarantee():
    rng = np.random.RandomState(42)
    # Synthetic feature data and linear regression target with additive noise
    n_train = 400
    n_cal = 300
    n_test = 300
    d = 4

    X_train = rng.normal(0.0, 1.0, size=(n_train, d))
    y_train = 2.0 * X_train[:, 0] - 1.5 * X_train[:, 1] + rng.normal(0.0, 0.5, size=n_train)

    X_cal = rng.normal(0.0, 1.0, size=(n_cal, d))
    y_cal = 2.0 * X_cal[:, 0] - 1.5 * X_cal[:, 1] + rng.normal(0.0, 0.5, size=n_cal)

    X_test = rng.normal(0.0, 1.0, size=(n_test, d))
    y_test = 2.0 * X_test[:, 0] - 1.5 * X_test[:, 1] + rng.normal(0.0, 0.5, size=n_test)

    calibrator = SplitConformalCalibrator(target_property="peak_magnitude", nominal_coverage=0.90)
    q_hat = calibrator.fit_and_calibrate(X_train, y_train, X_cal, y_cal)

    assert q_hat > 0.0
    assert calibrator.is_calibrated is True

    intervals = calibrator.predict_intervals(X_test)
    assert len(intervals) == n_test

    # Check finite-sample empirical coverage on exchangeable test set
    covered = sum(1 for interval, y_true in zip(intervals, y_test) if interval.contains(y_true))
    empirical_coverage = covered / n_test

    # For 90% nominal coverage with finite sample size N=300, empirical coverage is within 2 standard deviations
    assert empirical_coverage >= 0.85
    assert intervals[0].uncertainty_decomposition is not None
    assert intervals[0].uncertainty_decomposition.calibration_margin > 0.0


def test_dual_setting_evaluation():
    rng = np.random.RandomState(42)
    d = 4
    n_train = 200
    n_cal = 150
    n_test = 150

    X_train = rng.normal(0.0, 1.0, size=(n_train, d))
    y_train = 1.0 * X_train[:, 0] + rng.normal(0.0, 0.3, size=n_train)

    X_cal = rng.normal(0.0, 1.0, size=(n_cal, d))
    y_cal = 1.0 * X_cal[:, 0] + rng.normal(0.0, 0.3, size=n_cal)

    # Setting 1: Exchangeable
    X_s1 = rng.normal(0.0, 1.0, size=(n_test, d))
    y_s1 = 1.0 * X_s1[:, 0] + rng.normal(0.0, 0.3, size=n_test)

    # Setting 2: Severe Covariate & Variance Shift
    X_s2 = rng.normal(3.0, 2.0, size=(n_test, d)) # Shifted feature distribution
    y_s2 = 1.0 * X_s2[:, 0] + rng.normal(0.0, 1.5, size=n_test) # Higher variance noise

    calibrator = SplitConformalCalibrator(target_property="next_epoch_flux", nominal_coverage=0.90)
    calibrator.fit_and_calibrate(X_train, y_train, X_cal, y_cal)

    report = calibrator.evaluate_dual_settings(
        X_exchangeable=X_s1,
        y_exchangeable=y_s1,
        phot_err_exchangeable=np.full(n_test, 0.3),
        X_shifted=X_s2,
        y_shifted=y_s2,
        phot_err_shifted=np.full(n_test, 1.5)
    )

    assert isinstance(report, ConformalCalibrationReport)
    assert report.setting_1_exchangeable.empirical_coverage >= 0.85
    assert report.setting_1_exchangeable.exchangeability_satisfied is True

    # Shifted setting should reflect higher distribution shift score
    assert report.setting_2_distribution_shift.mean_distribution_shift_score > report.setting_1_exchangeable.mean_distribution_shift_score
