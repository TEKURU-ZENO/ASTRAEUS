"""
ASTRAEUS Cross-Survey Generalization, Physical Consistency & Transfer Audit Module.
Implements the 3 Senior-Mandated Cross-Survey Experiments:
- Experiment A: Cross-Survey Physical Parameter Consistency (comparing theta_ZTF vs theta_PS1 / theta_DECam)
- Experiment B: Cross-Survey Anomaly-Rank Stability (Spearman rho and Top-K overlap)
- Experiment C: Negative Transfer & Calibrated Abstention Criteria
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from astraeus_models.physics.engine import PhysicalModelEngine, PhysicsFitResult
from astraeus_models.features import AstronomicalFeatureExtractor, LightCurveFeatures


class PhysicalParameterConsistencyRecord(BaseModel):
    object_id: str
    transient_class: str
    preferred_model_ztf: Optional[str]
    preferred_model_target: Optional[str]
    ztf_parameters: Dict[str, float]
    target_parameters: Dict[str, float]
    mean_relative_parameter_error: float
    is_physically_consistent: bool # True if relative error on key timescales <= 25%


class RankStabilityRecord(BaseModel):
    target_survey_name: str
    spearman_rho: float
    spearman_pvalue: float
    top_5_jaccard: float
    top_10_jaccard: float
    top_20_jaccard: float
    mean_rank_displacement: float
    stability_verdict: str


class NegativeTransferRecord(BaseModel):
    target_survey_name: str
    in_domain_f1: float
    zero_shot_f1: float
    f1_degradation: float
    abstention_threshold: float
    abstention_rate: float
    post_abstention_f1: float
    negative_transfer_mitigated: bool


class CrossSurveyBenchmarkReport(BaseModel):
    num_matched_sources: int
    experiment_a_parameter_consistency: List[PhysicalParameterConsistencyRecord]
    mean_parameter_consistency_rate: float
    experiment_b_rank_stability: Dict[str, RankStabilityRecord]
    experiment_c_negative_transfer: Dict[str, NegativeTransferRecord]
    executive_summary: str


class CrossSurveyBenchmarkRunner:
    """
    Executes cross-survey evaluation comparing ZTF to Pan-STARRS1 and DECam.
    """

    def __init__(self, physics_engine: Optional[PhysicalModelEngine] = None):
        self.physics_engine = physics_engine or PhysicalModelEngine()
        self.feature_extractor = AstronomicalFeatureExtractor()

    def evaluate_physical_consistency(
        self,
        matched_lightcurves: List[Dict[str, Any]] # Each item: {'id', 'class', 'ztf': (t, m, e), 'target': (t, m, e), 'target_name'}
    ) -> List[PhysicalParameterConsistencyRecord]:
        """
        Experiment A: Compares inferred physical parameters between ZTF and target survey observations.
        """
        results = []
        for item in matched_lightcurves:
            obj_id = item["id"]
            cls_name = item["class"]
            ztf_t, ztf_f, ztf_e = item["ztf"]
            tgt_t, tgt_f, tgt_e = item["target"]

            # Fit physics on ZTF observations
            fit_ztf = self.physics_engine.fit_all_applicable(ztf_t, ztf_f, ztf_e, candidate_hypothesis=cls_name)
            # Fit physics on target observations
            fit_tgt = self.physics_engine.fit_all_applicable(tgt_t, tgt_f, tgt_e, candidate_hypothesis=cls_name)

            p_ztf = fit_ztf.physical_parameters
            p_tgt = fit_tgt.physical_parameters

            # Compute relative discrepancy over common parameters
            common_keys = set(p_ztf.keys()).intersection(set(p_tgt.keys()))
            rel_errs = []
            for k in common_keys:
                v1 = p_ztf[k]
                v2 = p_tgt[k]
                denom = max(abs(v1), abs(v2), 1e-3)
                rel_errs.append(abs(v1 - v2) / denom)

            mean_rel_err = float(np.mean(rel_errs)) if rel_errs else 1.0
            is_consistent = (mean_rel_err <= 0.25) and (fit_ztf.preferred_model == fit_tgt.preferred_model)

            results.append(
                PhysicalParameterConsistencyRecord(
                    object_id=obj_id,
                    transient_class=cls_name,
                    preferred_model_ztf=fit_ztf.preferred_model,
                    preferred_model_target=fit_tgt.preferred_model,
                    ztf_parameters=p_ztf,
                    target_parameters=p_tgt,
                    mean_relative_parameter_error=mean_rel_err,
                    is_physically_consistent=is_consistent,
                )
            )

        return results

    def evaluate_rank_stability(
        self,
        target_name: str,
        ztf_scores: np.ndarray,
        target_scores: np.ndarray
    ) -> RankStabilityRecord:
        """
        Experiment B: Computes Spearman rank correlation and Top-K Jaccard overlap between surveys.
        """
        n = len(ztf_scores)
        rho, pval = stats.spearmanr(ztf_scores, target_scores)

        # Ranks
        rank_ztf = np.argsort(np.argsort(-ztf_scores))
        rank_tgt = np.argsort(np.argsort(-target_scores))
        mean_displacement = float(np.mean(np.abs(rank_ztf - rank_tgt)))

        def jaccard_k(k: int) -> float:
            top_ztf = set(np.argsort(-ztf_scores)[:k])
            top_tgt = set(np.argsort(-target_scores)[:k])
            intersection = top_ztf.intersection(top_tgt)
            union = top_ztf.union(top_tgt)
            return float(len(intersection) / max(len(union), 1))

        j5 = jaccard_k(min(5, n))
        j10 = jaccard_k(min(10, n))
        j20 = jaccard_k(min(20, n))

        verdict = "STABLE" if rho >= 0.70 and j20 >= 0.60 else "MODERATE_SHIFT" if rho >= 0.40 else "UNSTABLE"

        return RankStabilityRecord(
            target_survey_name=target_name,
            spearman_rho=float(rho) if not np.isnan(rho) else 0.0,
            spearman_pvalue=float(pval) if not np.isnan(pval) else 1.0,
            top_5_jaccard=j5,
            top_10_jaccard=j10,
            top_20_jaccard=j20,
            mean_rank_displacement=mean_displacement,
            stability_verdict=verdict,
        )

    def evaluate_negative_transfer_and_abstention(
        self,
        target_name: str,
        in_domain_f1: float,
        model_predictions: np.ndarray,
        true_labels: np.ndarray,
        shift_distances: np.ndarray,
        abstention_threshold: float = 3.0
    ) -> NegativeTransferRecord:
        """
        Experiment C: Detects negative transfer degradation and applies calibrated abstention.
        If shift_distance > abstention_threshold, model abstains rather than emitting corrupted prediction.
        """
        from sklearn.metrics import f1_score

        # Baseline zero-shot F1 over all target objects
        zero_shot_f1 = float(f1_score(true_labels, model_predictions, average="macro", zero_division=0))
        degradation = float(in_domain_f1 - zero_shot_f1)

        # Apply abstention filter
        retain_mask = shift_distances <= abstention_threshold
        abstention_rate = float(1.0 - (np.sum(retain_mask) / max(len(retain_mask), 1)))

        if np.sum(retain_mask) > 0:
            post_f1 = float(f1_score(true_labels[retain_mask], model_predictions[retain_mask], average="macro", zero_division=0))
        else:
            post_f1 = zero_shot_f1

        mitigated = bool(post_f1 >= (in_domain_f1 - 0.10) or post_f1 > zero_shot_f1)

        return NegativeTransferRecord(
            target_survey_name=target_name,
            in_domain_f1=in_domain_f1,
            zero_shot_f1=zero_shot_f1,
            f1_degradation=degradation,
            abstention_threshold=abstention_threshold,
            abstention_rate=abstention_rate,
            post_abstention_f1=post_f1,
            negative_transfer_mitigated=mitigated,
        )
