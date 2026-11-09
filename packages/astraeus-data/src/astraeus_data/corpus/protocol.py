"""
ASTRAEUS Frozen Evaluation Protocol Module.
Implements the mandatory research gate: freezing benchmark evaluation targets, metrics,
and held-out test splits before foundation model training to prevent methodological data leakage.
"""

from typing import List, Dict, Any, Literal, Tuple, Optional
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
    evaluation_mode: str
    description: str
    primary_metric: str
    secondary_metrics: List[str]
    target_split: Literal["val", "test"]
    baseline_benchmarks: List[str]
    transfer_directions: List[str] = Field(default_factory=list)
    information_separation_guarantee: str


class FrozenEvaluationProtocol(BaseModel):
    """
    Immutable evaluation contract frozen before foundation model pretraining.
    Mathematically prevents using test-set data for model tuning or architecture selection.
    Codifies cross-corpus contamination invariant: Dataset B (pretraining) cannot contain
    any physical objects, counterparts, or injected/transformed derivatives in Dataset A held-out test split.
    """
    model_config = ConfigDict(frozen=True)
    protocol_version: str = "1.0.3"
    frozen_epoch_utc: str
    protocol_sha256: str
    tasks: List[EvaluationTask]
    frozen_val_ids: List[str]
    frozen_test_ids: List[str]
    frozen_quarantine_ids: List[str]
    metric_definitions: Dict[str, str]
    dataset_b_exclusion_contract: str = (
        "Dataset B (temporal foundation pretraining corpus) MUST NOT contain any physical object, "
        "IAU counterpart, cross-survey observation, host galaxy, alias, or injected/transformed derivative matching "
        "Dataset A's held-out test split (N=45 objects). All Dataset B samples require certified DAG provenance tracking "
        "via ProvenanceGraphAuditor; samples lacking lineage are quarantined. Spatial segregation enforces "
        "non-geodesic spherical cell-boundary distance with normal-projected astrometric covariance "
        "(d_edge <= 5.0'' OR d_edge <= 3*sigma_perp). Validation objects (N=47) may be used for supervised tuning "
        "and model selection, but never for self-supervised foundation pretraining."
    )

    @classmethod
    def freeze(
        cls,
        records: List[ReferenceObjectRecord],
        version: str = "1.0.3",
    ) -> "FrozenEvaluationProtocol":
        """Freezes evaluation tasks and held-out test/val/quarantine splits from a reference corpus."""
        val_ids = sorted([r.canonical_id for r in records if r.spatial_partition.split == "val"])
        test_ids = sorted([r.canonical_id for r in records if r.spatial_partition.split == "test"])
        quarantine_ids = sorted([r.canonical_id for r in records if r.spatial_partition.split == "quarantine"])

        tasks = [
            EvaluationTask(
                task_id="task_1_temporal_reconstruction",
                name="Irregular Masked Span Reconstruction & Forecasting",
                evaluation_mode="HELD_OUT_OBSERVATIONS_FORECASTING",
                description=(
                    "Evaluates probabilistic reconstruction of contiguous masked temporal spans "
                    "(simulating daytime and weather gaps) and future epoch forecasting ($t > t_{\\mathrm{cutoff}}$)."
                ),
                primary_metric="CRPS_Gaussian",
                secondary_metrics=["MAE_flux", "NLL_uncertainty", "Coverage_90_Conformal"],
                target_split="test",
                baseline_benchmarks=["Interpolation_Linear", "ASTROMER_Base", "GaussianProcess_Matern"],
                information_separation_guarantee=(
                    "Input context and evaluated target observations are strictly disjoint; "
                    "target points are masked before encoder forward pass."
                ),
            ),
            EvaluationTask(
                task_id="task_2_cross_survey_transfer",
                name="Zero-Shot Cross-Survey Property Estimation",
                evaluation_mode="UNSEEN_PHYSICAL_OBJECTS_AND_UNSEEN_SURVEYS",
                description=(
                    "Evaluates bidirectional zero-shot transfer of learned representations across "
                    "ZTF, Pan-STARRS1, and DECam without fine-tuning or target-domain label exposure."
                ),
                primary_metric="Macro_F1",
                secondary_metrics=["Expected_Calibration_Error", "Abstention_Recall", "Covariate_Shift_Distance"],
                target_split="test",
                baseline_benchmarks=["RandomForest_Engineered", "XGBoost_Passbands"],
                transfer_directions=[
                    "ZTF -> Pan-STARRS1",
                    "ZTF -> DECam",
                    "Pan-STARRS1 -> ZTF",
                    "Pan-STARRS1 -> DECam",
                    "DECam -> ZTF",
                    "DECam -> Pan-STARRS1",
                ],
                information_separation_guarantee=(
                    "Evaluates transfer to entirely unseen physical objects in disjoint HEALPix pixels; "
                    "no counterparts of target evaluation objects exist in the training corpus."
                ),
            ),
            EvaluationTask(
                task_id="task_3_rare_event_triage",
                name="Rare Transient Discovery & Ranking",
                evaluation_mode="FIXED_RESOURCE_BUDGET_TRIAGE",
                description=(
                    "Evaluates discovery prioritization of anomalous rapid transients (FBOTs, SLSNe, TDEs) "
                    "against normal SNe Ia, variable stars, and subtraction dipole artifacts."
                ),
                primary_metric="Precision_At_Budget_K10",
                secondary_metrics=[
                    "Recall_At_Budget_K10",
                    "Precision_At_Budget_K5",
                    "Precision_At_Budget_K20",
                    "PR_AUC",
                    "Artifact_Contamination_Rate",
                    "False_Positives_Per_Thousand",
                ],
                target_split="test",
                baseline_benchmarks=["IsolationForest_Standardized", "Mahalanobis_Distance_Empirical"],
                information_separation_guarantee=(
                    "Baselines receive strictly identical input features (standardized optical_pseudoflux "
                    "and native uncertainties) to isolate model architecture contribution."
                ),
            ),
        ]

        metric_defs = {
            "CRPS_Gaussian": "Continuous Ranked Probability Score under Gaussian predictive distribution.",
            "Macro_F1": "Unweighted arithmetic mean of per-class F1 scores across standard transient classes.",
            "Expected_Calibration_Error": "10-bin confidence calibration error against ground-truth labels.",
            "Precision_At_Budget_K10": "Fraction of genuine high-priority transients among top 10 ranked candidates.",
            "Recall_At_Budget_K10": "Fraction of total test-set rare events successfully retrieved in top 10 queue.",
            "Artifact_Contamination_Rate": "Fraction of top-ranked follow-up queue occupied by subtraction artifacts.",
        }

        exclusion_contract = (
            "Dataset B (temporal foundation pretraining corpus) MUST NOT contain any physical object, "
            "IAU counterpart, cross-survey observation, host galaxy, alias, or injected/transformed derivative matching "
            "Dataset A's held-out test split (N=45 objects). All Dataset B samples require certified DAG provenance tracking "
            "via ProvenanceGraphAuditor; samples lacking lineage are quarantined. Spatial segregation enforces "
            "non-geodesic spherical cell-boundary distance with normal-projected astrometric covariance "
            "(d_edge <= 5.0'' OR d_edge <= 3*sigma_perp). Validation objects (N=47) may be used for supervised tuning "
            "and model selection, but never for self-supervised foundation pretraining."
        )

        payload = {
            "version": version,
            "tasks": [t.model_dump() for t in tasks],
            "val_ids": val_ids,
            "test_ids": test_ids,
            "quarantine_ids": quarantine_ids,
            "metric_definitions": metric_defs,
            "dataset_b_exclusion_contract": exclusion_contract,
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
            frozen_quarantine_ids=quarantine_ids,
            metric_definitions=metric_defs,
            dataset_b_exclusion_contract=exclusion_contract,
        )

    def verify_no_contamination(self, train_ids: List[str]) -> bool:
        """Verifies that no training IDs contaminate the frozen validation, test, or quarantine sets."""
        train_set = set(train_ids)
        if train_set.intersection(set(self.frozen_val_ids)):
            return False
        if train_set.intersection(set(self.frozen_test_ids)):
            return False
        if train_set.intersection(set(self.frozen_quarantine_ids)):
            return False
        return True

    def verify_dataset_b_exclusion(
        self,
        dataset_b_ids: List[str],
        dataset_b_aliases: Optional[List[str]] = None,
        dataset_b_derivative_tags: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that no candidate object ID, alias, or injected derivative from Dataset B
        (pretraining corpus) overlaps with Dataset A held-out test objects.
        Returns (is_clean, violations_list).
        """
        test_set = set(self.frozen_test_ids)
        violations: List[str] = list(test_set.intersection(set(dataset_b_ids)))

        if dataset_b_aliases:
            alias_violations = list(test_set.intersection(set(dataset_b_aliases)))
            violations.extend(alias_violations)

        if dataset_b_derivative_tags:
            for tag in dataset_b_derivative_tags:
                for test_id in test_set:
                    if test_id in tag:
                        violations.append(f"DERIVATIVE_TAG:{tag}->{test_id}")

        for b_id in dataset_b_ids:
            for test_id in test_set:
                if test_id in b_id and b_id != test_id:
                    violations.append(f"DERIVATIVE_ID:{b_id}->{test_id}")

        clean_violations = sorted(list(set(violations)))
        return len(clean_violations) == 0, clean_violations
