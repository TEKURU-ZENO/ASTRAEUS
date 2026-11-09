"""
ASTRAEUS AstroTemporal Benchmark Runner.
Standardized benchmark comparing Feature-Engineered Baselines (Lomb-Scargle + Random Forest)
against Deep Representation Learning (ASTROMER) on task-separated light curve classification.
"""

from typing import List, Dict, Any, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

from astraeus_models.features import AstronomicalFeatureExtractor, LightCurveFeatures
from astraeus_models.classifier import TaskDomain, TaskSeparatedClassifier
from astraeus_models.astromer import AstromerEncoder
from astraeus_bench.metrics import compute_classification_metrics, ClassificationMetrics


class AstroTemporalComparisonReport(BaseModel):
    task_name: str
    num_classes: int
    classes: List[str]
    num_train: int
    num_test: int
    random_forest_metrics: ClassificationMetrics
    astromer_metrics: ClassificationMetrics
    rf_latency_ms_per_sample: float
    astromer_latency_ms_per_sample: float
    astromer_f1_delta: float = Field(
        description="Macro-F1 difference: astromer_f1 - rf_f1. Quantifies representation benefit."
    )


class AstroTemporalBenchmark:
    """
    Executes head-to-head evaluation between Random Forest and ASTROMER.
    """

    def __init__(self, d_model: int = 64, num_layers: int = 2, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature_extractor = AstronomicalFeatureExtractor()

    def _pad_sequences(
        self,
        light_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]],
        max_len: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts list of variable-length light curves to padded PyTorch batch tensors."""
        b = len(light_curves)
        t_batch = torch.zeros(b, max_len, dtype=torch.float32)
        m_batch = torch.zeros(b, max_len, dtype=torch.float32)
        e_batch = torch.zeros(b, max_len, dtype=torch.float32)
        band_batch = torch.zeros(b, max_len, dtype=torch.long)
        mask_batch = torch.ones(b, max_len, dtype=torch.bool)  # True = padding

        band_map = {"ZTF_g": 1, "ZTF_r": 2, "ZTF_i": 3}

        for i, (times, mags, errs, bands) in enumerate(light_curves):
            length = min(len(times), max_len)
            if length > 0:
                t_batch[i, :length] = torch.from_numpy(times[:length] - times[0]).float()
                m_batch[i, :length] = torch.from_numpy(mags[:length]).float()
                e_batch[i, :length] = torch.from_numpy(errs[:length]).float()
                band_batch[i, :length] = torch.tensor([band_map.get(b, 1) for b in bands[:length]], dtype=torch.long)
                mask_batch[i, :length] = False

        return (
            t_batch.to(self.device),
            m_batch.to(self.device),
            e_batch.to(self.device),
            band_batch.to(self.device),
            mask_batch.to(self.device),
        )

    def run_benchmark(
        self,
        task_name: str,
        classes: List[str],
        train_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]],
        train_labels: List[int],
        test_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]],
        test_labels: List[int],
        astromer_epochs: int = 15,
    ) -> AstroTemporalComparisonReport:
        """
        Runs full training, inference, and metric extraction across both models.
        """
        num_classes = len(classes)

        # -------------------------------------------------------------
        # 1. Evaluate Feature-Engineered Baseline (Random Forest)
        # -------------------------------------------------------------
        train_features = [
            self.feature_extractor.extract(t, m, e, bands)
            for t, m, e, bands in train_data
        ]
        test_features = [
            self.feature_extractor.extract(t, m, e, bands)
            for t, m, e, bands in test_data
        ]

        rf_clf = TaskSeparatedClassifier(task_domain=TaskDomain.EXPLOSIVE_TRANSIENT)
        X_train = rf_clf._features_to_matrix(train_features)
        y_train = np.array(train_labels)
        X_test = rf_clf._features_to_matrix(test_features)
        y_test = np.array(test_labels)

        rf_clf.model.fit(X_train, y_train)

        start_rf = time.perf_counter()
        rf_probs = rf_clf.model.predict_proba(X_test)
        rf_preds = np.argmax(rf_probs, axis=1)
        rf_time = time.perf_counter() - start_rf
        rf_latency_ms = (rf_time / max(len(test_data), 1)) * 1000.0

        rf_metrics = compute_classification_metrics(y_test, rf_preds, classes, rf_probs)

        # -------------------------------------------------------------
        # 2. Evaluate Deep Representation Learning (ASTROMER)
        # -------------------------------------------------------------
        t_tr, m_tr, e_tr, b_tr, mask_tr = self._pad_sequences(train_data)
        t_te, m_te, e_te, b_te, mask_te = self._pad_sequences(test_data)
        y_tr_tensor = torch.tensor(train_labels, dtype=torch.long, device=self.device)

        model = AstromerEncoder(
            d_model=self.d_model,
            nhead=4,
            num_layers=self.num_layers,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
        ).to(self.device)
        model.build_classifier_head(num_classes)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(astromer_epochs):
            optimizer.zero_grad()
            logits = model.forward_classification(t_tr, m_tr, e_tr, b_tr, mask_tr)
            loss = criterion(logits, y_tr_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        start_astromer = time.perf_counter()
        with torch.no_grad():
            test_logits = model.forward_classification(t_te, m_te, e_te, b_te, mask_te)
            astromer_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
            astromer_preds = np.argmax(astromer_probs, axis=1)
        astromer_time = time.perf_counter() - start_astromer
        astromer_latency_ms = (astromer_time / max(len(test_data), 1)) * 1000.0

        astromer_metrics = compute_classification_metrics(y_test, astromer_preds, classes, astromer_probs)

        f1_delta = float(astromer_metrics.macro_f1 - rf_metrics.macro_f1)

        return AstroTemporalComparisonReport(
            task_name=task_name,
            num_classes=num_classes,
            classes=classes,
            num_train=len(train_data),
            num_test=len(test_data),
            random_forest_metrics=rf_metrics,
            astromer_metrics=astromer_metrics,
            rf_latency_ms_per_sample=rf_latency_ms,
            astromer_latency_ms_per_sample=astromer_latency_ms,
            astromer_f1_delta=f1_delta,
        )
