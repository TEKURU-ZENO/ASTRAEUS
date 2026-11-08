"""
ASTRAEUS Anomaly Scoring Module.
Implements composite anomaly scoring combining latent Mahalanobis distance, conformal uncertainty, and physical residuals.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CompositeAnomalyScore(BaseModel):
    """
    Composite anomaly evaluation:
    s(x) = alpha * d_M + beta * u + gamma * r_phys
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
        ...,
        ge=0.0,
        description="r_phys: Weighted residual discrepancy against standard physical forward models."
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
        """Computes the weighted composite score automatically."""
        computed = (
            self.alpha * self.mahalanobis_distance
            + self.beta * self.epistemic_uncertainty
            + self.gamma * self.physical_residual
        )
        self.composite_score = float(computed)

    def is_candidate_for_followup(self, threshold: float = 2.5) -> bool:
        """Determines if candidate warrants autonomous multi-agent review and telescope follow-up."""
        return self.composite_score >= threshold
