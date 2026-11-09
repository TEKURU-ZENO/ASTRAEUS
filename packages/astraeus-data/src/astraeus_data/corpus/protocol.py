"""
ASTRAEUS Frozen Evaluation Protocol Module.
Implements the mandatory research gate: freezing benchmark evaluation targets, metrics,
and held-out test splits before foundation model training to prevent methodological data leakage.
"""

from typing import List, Dict, Any, Literal
from datetime import datetime, timezone
import hashlib
import json
from pydantic import BaseModel, Field, ConfigDict

from astraeus_data.corpus.record import ReferenceObjectRecord


class EvaluationTask(BaseModel):
    """Specification of an immutable evaluation task."""
    model_config = ConfigDict(frozen=True)
    task_id: str
    name: str
    description: str
    primary_metric: str
    secondary_metrics: List[str]
    target_split: Literal["val", "test"]
    baseline_benchmarks: List[str]


class FrozenEvaluationProtocol(BaseModel):
    """
    Immutable evaluation contract frozen before foundation model pretraining.
    Mathematically prevents using test-set data for model tuning or architecture selection.
    """
    model_config = ConfigDict(frozen=True)
    protocol_version: str = "1.0.0"
    frozen_epoch_utc: str
    protocol_sha256: str
    tasks: List[EvaluationTask]
    frozen_val_ids: List[str]
    frozen_test_ids: List[str]
    metric_definitions: Dict[str, str]

    @classmethod
    def freeze(
        cls,
        records: List[ReferenceObjectRecord],
        version: str = "1.0.0",
    ) -> "FrozenEvaluationProtocol":
        """Freezes evaluation tasks and held-out test/val splits from a reference corpus."""
        val_ids = sorted([r.canonical_id for r in records if r.spatial_partition.split == "val"])
        test_ids = sorted([r.canonical_id for r in records if r.spatial_partition.split == "test"])

        tasks = [
            EvaluationTask(
                task_id="task_1_temporal_reconstruction",
                name="Irregular Masked Temporal Autoencoding",
                description="Reconstruction of masked photometric epochs on irregular cadences.",
                primary_metric="CRPS_Gaussian",
                secondary_metrics=["MAE_flux", "NLL_uncertainty"],
                target_split="test",
                baseline_benchmarks=["Interpolation_Linear", "ASTROMER_Base", "GaussianProcess"],
            ),
            EvaluationTask(
                task_id="task_2_cross_survey_transfer",
                name="Zero-Shot Cross-Survey Property Estimation",
                description="Evaluation of feature representations transferred from ZTF to PS1 and DECam.",
                primary_metric="Macro_F1",
                secondary_metrics=["Expected_Calibration_Error", "Abstention_Recall"],
                target_split="test",
                baseline_benchmarks=["RandomForest_Engineered", "XGBoost_Passbands"],
            ),
            EvaluationTask(
                task_id="task_3_rare_event_triage",
                name="Rare Transient Discovery & Ranking",
                description="Ranking anomalous rapid transients (FBOTs, SLSNe) within fixed follow-up budgets.",
                primary_metric="Precision_At_Budget_K10",
                secondary_metrics=["Top_K_Recall_K20", "Artifact_FDR"],
                target_split="test",
                baseline_benchmarks=["IsolationForest_Uncalibrated", "Mahalanobis_Distance"],
            ),
        ]

        metric_defs = {
            "CRPS_Gaussian": "Continuous Ranked Probability Score under Gaussian predictive distribution.",
            "Macro_F1": "Unweighted mean of per-class F1 scores across standard transient classes.",
            "Expected_Calibration_Error": "10-bin confidence calibration error against ground-truth labels.",
            "Precision_At_Budget_K10": "Fraction of genuine high-priority transients among top 10 ranked candidates.",
        }

        payload = {
            "version": version,
            "tasks": [t.model_dump() for t in tasks],
            "val_ids": val_ids,
            "test_ids": test_ids,
            "metric_definitions": metric_defs,
        }
        canon_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        proto_hash = hashlib.sha256(canon_json.encode("utf-8")).hexdigest()

        return cls(
            protocol_version=version,
            frozen_epoch_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=proto_hash,
            tasks=tasks,
            frozen_val_ids=val_ids,
            frozen_test_ids=test_ids,
            metric_definitions=metric_defs,
        )

    def verify_no_contamination(self, train_ids: List[str]) -> bool:
        """Verifies that no training IDs contaminate the frozen validation or test sets."""
        train_set = set(train_ids)
        if train_set.intersection(set(self.frozen_val_ids)):
            return False
        if train_set.intersection(set(self.frozen_test_ids)):
            return False
        return True
