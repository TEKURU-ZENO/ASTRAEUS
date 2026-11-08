"""
ASTRAEUS Baseline Classification Module.
Implements task-separated Random Forest classifiers distinguishing periodicity,
transient evolution, and artifact filtering.
"""

from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from pydantic import BaseModel, Field

from astraeus_models.features import LightCurveFeatures


class TaskDomain(str, Enum):
    PERIODIC_VARIABLE = "periodic_variable"  # RR Lyrae, Cepheids, Eclipsing Binaries
    EXPLOSIVE_TRANSIENT = "explosive_transient" # SN Ia, SN II, FBOT
    ARTIFACT_FILTER = "artifact_filter"       # Astrophysical Source vs Detector Artifact


class ClassificationBenchmarkResult(BaseModel):
    task_domain: TaskDomain
    num_train_samples: int
    num_test_samples: int
    macro_f1: float
    per_class_f1: Dict[str, float]
    feature_importances: Dict[str, float]


class TaskSeparatedClassifier:
    """
    Evaluates machine learning baselines across segregated scientific tasks.
    """

    def __init__(self, task_domain: TaskDomain, n_estimators: int = 100, random_state: int = 42):
        self.task_domain = task_domain
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_names: List[str] = [
            "num_epochs", "duration_days", "peak_mag", "faintest_mag",
            "amplitude", "rise_rate_mag_per_day", "decay_rate_mag_per_day",
            "flux_skewness", "flux_kurtosis", "stetson_j", "clean_fraction", "mean_error"
        ]

    def _features_to_matrix(self, feature_list: List[LightCurveFeatures]) -> np.ndarray:
        return np.array([
            [
                f.num_epochs, f.duration_days, f.peak_mag, f.faintest_mag,
                f.amplitude, f.rise_rate_mag_per_day, f.decay_rate_mag_per_day,
                f.flux_skewness, f.flux_kurtosis, f.stetson_j, f.clean_fraction, f.mean_error
            ]
            for f in feature_list
        ])

    def train_and_evaluate(
        self,
        train_features: List[LightCurveFeatures],
        train_labels: List[str],
        test_features: List[LightCurveFeatures],
        test_labels: List[str]
    ) -> ClassificationBenchmarkResult:
        """
        Trains the Random Forest model and reports macro-F1 and per-class metrics.
        """
        X_train = self._features_to_matrix(train_features)
        y_train = np.array(train_labels)

        X_test = self._features_to_matrix(test_features)
        y_test = np.array(test_labels)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        macro_f1 = float(f1_score(y_test, predictions, average="macro"))
        unique_classes = sorted(list(set(y_test)))
        per_class_f1_scores = f1_score(y_test, predictions, average=None, labels=unique_classes)
        per_class_dict = {cls_name: float(score) for cls_name, score in zip(unique_classes, per_class_f1_scores)}

        importances = {
            name: float(imp)
            for name, imp in zip(self.feature_names, self.model.feature_importances_)
        }

        return ClassificationBenchmarkResult(
            task_domain=self.task_domain,
            num_train_samples=len(X_train),
            num_test_samples=len(X_test),
            macro_f1=macro_f1,
            per_class_f1=per_class_dict,
            feature_importances=importances,
        )
