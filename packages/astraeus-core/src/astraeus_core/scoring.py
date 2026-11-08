"""
ASTRAEUS Anomaly Scoring Module.
Implements composite anomaly scoring combining latent Mahalanobis distance, conformal uncertainty, and physical residuals.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CompositeAnomalyScore(BaseModel):
    """
    Calibrated composite anomaly evaluation combining normalized terms:
    s(x) = alpha * d_M_norm + beta * u_norm + gamma * r_phys_norm
    Ensures commensurate numerical scales across disparate statistical and physical domains.
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
    # Normalized components mapped onto calibrated [0, 1] scales
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
        # Use normalized terms if available, otherwise fallback with clear logging
        d_val = self.mahalanobis_normalized if self.mahalanobis_normalized is not None else self.mahalanobis_distance
        u_val = self.uncertainty_normalized if self.uncertainty_normalized is not None else self.epistemic_uncertainty
        r_val = self.residual_normalized if self.residual_normalized is not None else self.physical_residual

        # Re-weight dynamically if no physical model is applicable
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
