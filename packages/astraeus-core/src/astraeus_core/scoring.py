"""
ASTRAEUS Anomaly Scoring & Evidence Profile Module.
Implements frozen empirical reference CDF normalization, component decoupling,
operational triage ranking, and diagnostic evidence profiles.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from pydantic import BaseModel, Field


class EmpiricalReferenceCDF(BaseModel):
    """
    Frozen empirical cumulative distribution function (CDF) fitted strictly on reference samples.
    Prevents test-distribution leakage into score normalization.
    """
    metric_name: str = Field(..., description="Name of the underlying metric: 'mahalanobis_distance', 'uncertainty', 'physical_residual'.")
    reference_population: str = Field(..., description="Identifier of reference dataset (e.g. 'ZTF_BTS_Common_v1.0').")
    version: str = Field(default="1.0.0", description="Immutable version tag.")
    reference_sample_size: int = Field(..., ge=1)
    sorted_reference_values: List[float] = Field(..., description="Sorted reference observations for percentile mapping.")

    def transform(self, value: float) -> float:
        """
        Maps a raw score to its empirical percentile: F_hat(s) = (1/N) * sum(s_i <= s).
        Returns value in [0.0, 1.0].
        """
        arr = np.array(self.sorted_reference_values)
        idx = np.searchsorted(arr, value, side="right")
        return float(idx / self.reference_sample_size)

    @classmethod
    def fit_from_samples(
        cls,
        metric_name: str,
        samples: List[float],
        population_name: str,
        version: str = "1.0.0"
    ) -> "EmpiricalReferenceCDF":
        """Fits and freezes an empirical CDF from a reference cohort."""
        clean = [float(x) for x in samples if not np.isnan(x) and not np.isinf(x)]
        if not clean:
            clean = [0.0]
        sorted_vals = sorted(clean)
        return cls(
            metric_name=metric_name,
            reference_population=population_name,
            version=version,
            reference_sample_size=len(sorted_vals),
            sorted_reference_values=sorted_vals,
        )


class AnomalyEvidenceProfile(BaseModel):
    """
    Transparent diagnostic profile preserving individual constituent components,
    reference population metadata, and physical model applicability.
    """
    entity_id: str
    raw_scores: Dict[str, float] = Field(
        ...,
        description="Raw metrics: {'d_M': float, 'u': float, 'r_phys': float}."
    )
    normalized_percentiles: Dict[str, float] = Field(
        ...,
        description="Empirical percentiles in [0, 1] relative to frozen reference CDFs."
    )
    reference_population_version: str
    physical_model_applied: str = "none_applicable"
    fit_status: str = "MODEL_NOT_APPLICABLE"
    primary_driver: str = Field(
        default="latent_distance",
        description="Component driving the anomalous designation: 'latent_distance', 'epistemic_uncertainty', or 'physical_residual'."
    )
    component_correlations: Optional[Dict[str, float]] = Field(
        default=None,
        description="Pairwise correlation metrics to detect collinear evidence inflation."
    )


class AnomalyRankScore(BaseModel):
    """
    Operational triage ranking score for telescope and agent follow-up prioritization.
    Explicitly decoupled from posterior discovery probability.
    """
    entity_id: str
    operational_rank_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Triage ranking in [0, 1] computed over normalized components."
    )
    evidence_profile: AnomalyEvidenceProfile
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3

    def is_candidate_for_followup(self, threshold: float = 0.80) -> bool:
        """Flags candidate for immediate robotic follow-up or agentic scrutiny."""
        return self.operational_rank_score >= threshold


class CompositeAnomalyScore(BaseModel):
    """
    Calibrated composite anomaly evaluation combining normalized terms:
    s(x) = alpha * d_M_norm + beta * u_norm + gamma * r_phys_norm
    Ensures backward compatibility with Phase 1 Sprint 1-3 schemas while enforcing normalization.
    """
    entity_id: str = Field(..., description="Astronomical candidate ID.")
    mahalanobis_distance: float = Field(
        ...,
        ge=0.0,
        description="d_M: Minimum Mahalanobis distance to known population centroids in latent space."
    )
    epistemic_uncertainty: float = Field(
        ...,
        ge=0.0,
        description="u: Conformal quantile width scaled by local latent density."
    )
    physical_residual: float = Field(
        default=0.0,
        ge=0.0,
        description="r_phys: Weighted residual discrepancy against applicable physical forward models."
    )
    mahalanobis_normalized: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Percentile-normalized d_M against calibration sample."
    )
    uncertainty_normalized: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Percentile-normalized u against calibration sample."
    )
    residual_normalized: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Percentile-normalized r_phys against calibration sample."
    )
    physical_model_applied: str = Field(
        default="none_applicable",
        description="Specific forward physical model tested (e.g. 'bazin_transient', 'arnett_sn_ia', 'none_applicable')."
    )
    physical_model_valid: bool = Field(
        default=True,
        description="Whether a valid physical model was available for this candidate class."
    )
    alpha: float = Field(default=0.4, ge=0.0, description="Weight on latent manifold distance.")
    beta: float = Field(default=0.3, ge=0.0, description="Weight on conformal epistemic uncertainty.")
    gamma: float = Field(default=0.3, ge=0.0, description="Weight on physical model forward residuals.")
    composite_score: float = Field(
        default=0.0,
        description="Total integrated anomaly score: s(x)."
    )
    anomaly_category: str = Field(
        default="behavioral",
        description="Classification: 'behavioral', 'contextual', 'spectroscopic', or 'instrumental_artifact'."
    )
    details: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Computes the weighted composite score automatically using normalized or raw values."""
        d_val = self.mahalanobis_normalized if self.mahalanobis_normalized is not None else self.mahalanobis_distance
        u_val = self.uncertainty_normalized if self.uncertainty_normalized is not None else self.epistemic_uncertainty
        r_val = self.residual_normalized if self.residual_normalized is not None else self.physical_residual

        if not self.physical_model_valid or self.physical_model_applied == "none_applicable":
            total_weight = self.alpha + self.beta
            a_weight = self.alpha / total_weight if total_weight > 0 else 0.5
            b_weight = self.beta / total_weight if total_weight > 0 else 0.5
            computed = a_weight * d_val + b_weight * u_val
        else:
            computed = self.alpha * d_val + self.beta * u_val + self.gamma * r_val

        self.composite_score = float(computed)

    def is_candidate_for_followup(self, threshold: float = 2.5) -> bool:
        """Determines if candidate warrants autonomous multi-agent review and telescope follow-up."""
        return self.composite_score >= threshold
