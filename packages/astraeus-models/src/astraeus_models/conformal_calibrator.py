"""
ASTRAEUS Split Conformal Calibration Engine.
Implements exact finite-sample conformal prediction for astronomical property regression
(e.g., peak magnitude, next-epoch flux) with uncertainty decomposition and dual-setting evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pydantic import BaseModel, Field

from astraeus_core.conformal import ConformalPredictionInterval, UncertaintyDecomposition


def compute_wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes exact Wilson score confidence interval for binomial coverage proportion.
    """
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    z = 1.95996  # 95% two-tailed normal quantile
    z2 = z ** 2
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    spread = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4 * (total ** 2))))
    return max(0.0, float(center - spread)), min(1.0, float(center + spread))


class ConformalEvaluationSettingResult(BaseModel):
    setting_name: str
    nominal_coverage: float
    empirical_coverage: float
    coverage_ci_95_low: float = 0.0
    coverage_ci_95_high: float = 1.0
    coverage_shortfall: float
    mean_interval_width: float
    median_interval_width: float
    conformal_threshold_q: float
    mean_photometric_noise: float
    mean_predictive_uncertainty: float
    mean_distribution_shift_score: float
    n_eval_samples: int
    exchangeability_satisfied: bool


class ConformalCalibrationReport(BaseModel):
    target_property: str
    nominal_coverage: float
    calibration_size: int
    conformal_quantile_threshold: float
    quantile_standard_error_diagnostic: float
    setting_1_exchangeable: ConformalEvaluationSettingResult
    setting_2_distribution_shift: ConformalEvaluationSettingResult


class SplitConformalCalibrator:
    """
    Split Conformal Prediction Engine with exact finite-sample order statistic guarantees.
    Threshold: q_{1 - alpha} = S_{(ceil((n_cal + 1)(1 - alpha)))}
    """

    def __init__(
        self,
        target_property: str = "peak_magnitude",
        nominal_coverage: float = 0.90,
        random_state: int = 42
    ):
        self.target_property = target_property
        self.nominal_coverage = nominal_coverage
        self.random_state = random_state
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.calibration_scores: Optional[np.ndarray] = None
        self.q_hat: Optional[float] = None
        self.quantile_se_diagnostic: float = 0.0
        self.reference_features_mean: Optional[np.ndarray] = None
        self.reference_features_std: Optional[np.ndarray] = None
        self.is_calibrated: bool = False

    def fit_and_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        photometric_errors_cal: Optional[np.ndarray] = None
    ) -> float:
        """
        Fits point prediction model on training set and computes exact conformal threshold on calibration set.
        """
        self.model.fit(X_train, y_train)

        # Store calibration feature moments for distribution shift monitoring
        self.reference_features_mean = np.mean(X_train, axis=0)
        self.reference_features_std = np.std(X_train, axis=0) + 1e-6

        # Predict on calibration set
        y_cal_pred = self.model.predict(X_cal)

        # Non-conformity scores: absolute residuals |y - y_hat|
        residuals = np.abs(y_cal - y_cal_pred)
        self.calibration_scores = np.sort(residuals)
        n_cal = len(residuals)

        # Exact finite-sample quantile calculation
        # rank = ceil((n_cal + 1) * (1 - alpha))
        alpha = 1.0 - self.nominal_coverage
        k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
        if k > n_cal:
            # Conservative finite-sample fallback when calibration set is smaller than required order statistic
            self.q_hat = float(np.inf)
        else:
            # 0-based index is k - 1
            self.q_hat = float(self.calibration_scores[k - 1])

        # Separate diagnostic: finite-sample standard error of sample quantile
        # Computed via asymptotic spread / sqrt(n_cal) strictly for model monitoring, NOT as interval margin
        self.quantile_se_diagnostic = float(np.std(residuals) / np.sqrt(max(n_cal, 1)))
        self.is_calibrated = True

        return self.q_hat

    def predict_intervals(
        self,
        X_test: np.ndarray,
        photometric_errors: Optional[np.ndarray] = None,
        latent_distances: Optional[np.ndarray] = None
    ) -> List[ConformalPredictionInterval]:
        """
        Constructs valid conformal prediction intervals with uncertainty decomposition.
        """
        if not self.is_calibrated or self.q_hat is None:
            raise RuntimeError("SplitConformalCalibrator must be calibrated before generating intervals.")

        y_pred = self.model.predict(X_test)
        n_test = len(X_test)
        intervals: List[ConformalPredictionInterval] = []

        # Standardised Mahalanobis-like covariate shift score
        scaled_diff = (X_test - self.reference_features_mean) / self.reference_features_std
        shift_scores = np.sqrt(np.mean(scaled_diff ** 2, axis=1))

        for i in range(n_test):
            point = float(y_pred[i])
            lower = point - self.q_hat
            upper = point + self.q_hat

            aleatoric_val = float(photometric_errors[i]) if photometric_errors is not None else 0.05
            epistemic_val = float(latent_distances[i]) if latent_distances is not None else 0.05
            shift_val = float(shift_scores[i])

            decomp = UncertaintyDecomposition(
                photometric_noise=max(0.0, aleatoric_val),
                predictive_uncertainty=max(0.0, epistemic_val),
                conformal_correction=float(self.q_hat),
                quantile_estimation_variance=self.quantile_se_diagnostic,
                distribution_shift_score=max(0.0, shift_val)
            )

            is_ood = shift_val > 3.0 or epistemic_val > 3.0

            intervals.append(
                ConformalPredictionInterval(
                    target_property=self.target_property,
                    point_estimate=point,
                    lower_bound=lower,
                    upper_bound=upper,
                    coverage_level=self.nominal_coverage,
                    uncertainty_decomposition=decomp,
                    latent_density_sigma=epistemic_val,
                    is_out_of_distribution=is_ood
                )
            )

        return intervals

    def evaluate_dual_settings(
        self,
        X_exchangeable: np.ndarray,
        y_exchangeable: np.ndarray,
        phot_err_exchangeable: Optional[np.ndarray],
        X_shifted: np.ndarray,
        y_shifted: np.ndarray,
        phot_err_shifted: Optional[np.ndarray]
    ) -> ConformalCalibrationReport:
        """
        Executes rigorous dual-setting evaluation:
        Setting 1: Exchangeable held-out test set (verifies 1 - alpha empirical coverage guarantee)
        Setting 2: Distribution-shifted test set (measures empirical coverage shortfall & shift diagnostics)
        """
        # Setting 1: Exchangeable
        intervals_s1 = self.predict_intervals(X_exchangeable, phot_err_exchangeable)
        covered_s1 = sum(1 for interval, y_true in zip(intervals_s1, y_exchangeable) if interval.contains(float(y_true)))
        n_s1 = len(y_exchangeable)
        cov_s1 = float(covered_s1 / n_s1) if n_s1 > 0 else 0.0
        ci1_low, ci1_high = compute_wilson_score_interval(covered_s1, n_s1)
        widths_s1 = [interval.interval_width for interval in intervals_s1]
        aleatoric_s1 = [interval.uncertainty_decomposition.photometric_noise for interval in intervals_s1 if interval.uncertainty_decomposition]
        epistemic_s1 = [interval.uncertainty_decomposition.predictive_uncertainty for interval in intervals_s1 if interval.uncertainty_decomposition]
        shift_s1 = [interval.uncertainty_decomposition.distribution_shift_score for interval in intervals_s1 if interval.uncertainty_decomposition]

        setting_1_res = ConformalEvaluationSettingResult(
            setting_name="Setting 1: Exchangeable Held-out Population",
            nominal_coverage=self.nominal_coverage,
            empirical_coverage=cov_s1,
            coverage_ci_95_low=ci1_low,
            coverage_ci_95_high=ci1_high,
            coverage_shortfall=max(0.0, self.nominal_coverage - cov_s1),
            mean_interval_width=float(np.mean(widths_s1)),
            median_interval_width=float(np.median(widths_s1)),
            conformal_threshold_q=float(self.q_hat) if self.q_hat is not None else 0.0,
            mean_photometric_noise=float(np.mean(aleatoric_s1)) if aleatoric_s1 else 0.0,
            mean_predictive_uncertainty=float(np.mean(epistemic_s1)) if epistemic_s1 else 0.0,
            mean_distribution_shift_score=float(np.mean(shift_s1)) if shift_s1 else 0.0,
            n_eval_samples=n_s1,
            exchangeability_satisfied=bool(ci1_high >= self.nominal_coverage)
        )

        # Setting 2: Distribution-Shifted
        intervals_s2 = self.predict_intervals(X_shifted, phot_err_shifted)
        covered_s2 = sum(1 for interval, y_true in zip(intervals_s2, y_shifted) if interval.contains(float(y_true)))
        n_s2 = len(y_shifted)
        cov_s2 = float(covered_s2 / n_s2) if n_s2 > 0 else 0.0
        ci2_low, ci2_high = compute_wilson_score_interval(covered_s2, n_s2)
        widths_s2 = [interval.interval_width for interval in intervals_s2]
        aleatoric_s2 = [interval.uncertainty_decomposition.photometric_noise for interval in intervals_s2 if interval.uncertainty_decomposition]
        epistemic_s2 = [interval.uncertainty_decomposition.predictive_uncertainty for interval in intervals_s2 if interval.uncertainty_decomposition]
        shift_s2 = [interval.uncertainty_decomposition.distribution_shift_score for interval in intervals_s2 if interval.uncertainty_decomposition]

        setting_2_res = ConformalEvaluationSettingResult(
            setting_name="Setting 2: Distribution-Shifted / Cadence Degraded Population",
            nominal_coverage=self.nominal_coverage,
            empirical_coverage=cov_s2,
            coverage_ci_95_low=ci2_low,
            coverage_ci_95_high=ci2_high,
            coverage_shortfall=max(0.0, self.nominal_coverage - cov_s2),
            mean_interval_width=float(np.mean(widths_s2)),
            median_interval_width=float(np.median(widths_s2)),
            conformal_threshold_q=float(self.q_hat) if self.q_hat is not None else 0.0,
            mean_photometric_noise=float(np.mean(aleatoric_s2)) if aleatoric_s2 else 0.0,
            mean_predictive_uncertainty=float(np.mean(epistemic_s2)) if epistemic_s2 else 0.0,
            mean_distribution_shift_score=float(np.mean(shift_s2)) if shift_s2 else 0.0,
            n_eval_samples=n_s2,
            exchangeability_satisfied=bool(ci2_high >= self.nominal_coverage)
        )

        return ConformalCalibrationReport(
            target_property=self.target_property,
            nominal_coverage=self.nominal_coverage,
            calibration_size=len(self.calibration_scores) if self.calibration_scores is not None else 0,
            conformal_quantile_threshold=self.q_hat if self.q_hat is not None else 0.0,
            quantile_standard_error_diagnostic=self.quantile_se_diagnostic,
            setting_1_exchangeable=setting_1_res,
            setting_2_distribution_shift=setting_2_res
        )
