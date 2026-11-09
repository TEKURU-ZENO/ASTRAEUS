"""
ASTRAEUS Benchmark Metrics Module.
Implements Expected Calibration Error (ECE), Continuous Ranked Probability Score (CRPS),
Brier score, confusion matrices, and anomaly budget discovery metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from pydantic import BaseModel, Field


class CalibrationMetrics(BaseModel):
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]


class ClassificationMetrics(BaseModel):
    macro_f1: float
    weighted_f1: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: List[List[int]]
    classes: List[str]
    calibration: Optional[CalibrationMetrics] = None


class AnomalyBenchmarkMetrics(BaseModel):
    recall_at_budget: float
    false_discovery_rate: float
    precision_at_budget: float
    budget_n: int
    true_anomalies_found: int
    total_true_anomalies: int


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    by partitioning predictions into confidence bins B_m.
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == true_labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    mce = 0.0
    n = len(true_labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)

            diff = abs(bin_acc - bin_conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # Multi-class Brier score: 1/N sum_i sum_k (p_ik - y_ik)^2
    one_hot = np.zeros_like(probabilities)
    for idx, label in enumerate(true_labels):
        one_hot[idx, label] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    return CalibrationMetrics(
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        brier_score=brier,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


def compute_crps_gaussian(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray
) -> float:
    """
    Computes the Continuous Ranked Probability Score (CRPS) under Gaussian predictive assumption.
    CRPS(y, mu, sigma) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.
    """
    sigma = np.clip(sigma_pred, 1e-6, None)
    z = (y_true - mu_pred) / sigma
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)

    crps_vals = sigma * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_vals))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    probabilities: Optional[np.ndarray] = None
) -> ClassificationMetrics:
    """
    Computes balanced macro-F1, per-class metrics, confusion matrix, and optional calibration.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec_vals = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec_vals = recall_score(y_true, y_pred, average=None, zero_division=0)

    f1_dict = {cls_name: float(val) for cls_name, val in zip(classes, per_class_f1_vals)}
    prec_dict = {cls_name: float(val) for cls_name, val in zip(classes, per_class_prec_vals)}
    rec_dict = {cls_name: float(val) for cls_name, val in zip(classes, per_class_rec_vals)}

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

    calibration = None
    if probabilities is not None:
        calibration = compute_expected_calibration_error(probabilities, y_true)

    return ClassificationMetrics(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1=f1_dict,
        per_class_precision=prec_dict,
        per_class_recall=rec_dict,
        confusion_matrix=cm,
        classes=classes,
        calibration=calibration,
    )


def compute_anomaly_budget_metrics(
    anomaly_scores: np.ndarray,
    is_true_anomaly: np.ndarray,
    budget_n: int = 100
) -> AnomalyBenchmarkMetrics:
    """
    Evaluates anomaly discovery performance under a fixed telescope follow-up budget N.
    """
    budget = min(budget_n, len(anomaly_scores))
    # Rank descending by anomaly score
    ranked_indices = np.argsort(anomaly_scores)[::-1]
    top_indices = ranked_indices[:budget]

    found_anomalies = int(np.sum(is_true_anomaly[top_indices]))
    total_anomalies = int(np.sum(is_true_anomaly))

    recall = float(found_anomalies / max(total_anomalies, 1))
    precision = float(found_anomalies / max(budget, 1))
    fdr = float(1.0 - precision)

    return AnomalyBenchmarkMetrics(
        recall_at_budget=recall,
        false_discovery_rate=fdr,
        precision_at_budget=precision,
        budget_n=budget,
        true_anomalies_found=found_anomalies,
        total_true_anomalies=total_anomalies,
    )
