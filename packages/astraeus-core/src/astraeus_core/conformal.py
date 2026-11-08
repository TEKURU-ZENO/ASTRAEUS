"""
ASTRAEUS Conformal Uncertainty Module.
Implements Conformalized Quantile Regression (CQR) interval representations and coverage checks.
"""

from typing import Optional
from pydantic import BaseModel, Field


class UncertaintyDecomposition(BaseModel):
    """
    Decomposed uncertainty components preventing the conflation of distinct physical and statistical sources.
    """
    aleatoric: float = Field(
        ...,
        ge=0.0,
        description="Measurement error and intrinsic astrophysical scatter (e.g. photometric noise)."
    )
    epistemic: float = Field(
        ...,
        ge=0.0,
        description="Model parameter/architecture uncertainty and lack of local training examples."
    )
    calibration_margin: float = Field(
        ...,
        ge=0.0,
        description="Finite-sample uncertainty of the empirical non-conformity threshold."
    )
    distribution_shift: float = Field(
        default=0.0,
        ge=0.0,
        description="Covariate shift penalty between calibration survey and target observation."
    )


class ConformalPredictionInterval(BaseModel):
    r"""
    Finite-sample distribution-free prediction interval with latent density scaling.
    Guarantees: P(y \in [lower_bound, upper_bound]) >= 1 - alpha under exchangeability.
    """
    target_property: str = Field(..., description="Target quantity (e.g. 'photometric_redshift', 'peak_magnitude').")
    point_estimate: float = Field(..., description="Central conditional expectation or median prediction.")
    lower_bound: float = Field(..., description="Conditional quantile lower bound: q_{alpha/2}.")
    upper_bound: float = Field(..., description="Conditional quantile upper bound: q_{1-alpha/2}.")
    coverage_level: float = Field(default=0.90, ge=0.50, lt=1.0, description="Nominal confidence level 1 - alpha.")
    uncertainty_decomposition: Optional[UncertaintyDecomposition] = Field(
        default=None,
        description="Decomposed uncertainty constituents (aleatoric, epistemic, calibration, shift)."
    )
    latent_density_sigma: float = Field(
        default=0.0,
        ge=0.0,
        description="Local k-NN distance in foundation embedding space; proxy for epistemic sparsity."
    )
    is_out_of_distribution: bool = Field(
        default=False,
        description="Flagged True if latent_density_sigma exceeds the calibration threshold."
    )

    @property
    def interval_width(self) -> float:
        """Returns the width of the prediction interval."""
        return max(0.0, self.upper_bound - self.lower_bound)

    def contains(self, true_value: float) -> bool:
        """Checks if the empirical ground truth is covered by the conformal interval."""
        return self.lower_bound <= true_value <= self.upper_bound
