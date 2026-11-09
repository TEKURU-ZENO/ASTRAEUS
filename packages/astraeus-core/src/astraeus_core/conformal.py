"""
ASTRAEUS Conformal Uncertainty Module.
Implements Conformalized Prediction Interval representations, strict separation of
conformal nonconformity thresholds from quantile estimation variances, and distribution shift diagnostics.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator


class UncertaintyDecomposition(BaseModel):
    """
    Decomposed uncertainty constituents strictly separating physical measurement noise,
    model epistemic variance, the conformal nonconformity correction, finite-sample quantile variance,
    and covariate distribution shift diagnostics.
    """
    photometric_noise: float = Field(
        default=0.0,
        ge=0.0,
        description="Aleatoric measurement noise and intrinsic observational scatter (e.g., sigma_phot)."
    )
    predictive_uncertainty: float = Field(
        default=0.0,
        ge=0.0,
        description="Epistemic model parameter/architecture uncertainty or latent k-NN distance."
    )
    conformal_correction: float = Field(
        default=0.0,
        ge=0.0,
        description="Split-conformal empirical nonconformity quantile q_{1-alpha} applied directly to construct intervals."
    )
    quantile_estimation_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Asymptotic sampling variance Var(q_hat) of the finite-sample empirical quantile estimate; diagnostic only."
    )
    distribution_shift_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Covariate shift diagnostic between reference calibration distribution and target observation."
    )

    @model_validator(mode="before")
    @classmethod
    def map_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "aleatoric" in data and "photometric_noise" not in data:
                data["photometric_noise"] = data["aleatoric"]
            if "epistemic" in data and "predictive_uncertainty" not in data:
                data["predictive_uncertainty"] = data["epistemic"]
            if "calibration_margin" in data and "conformal_correction" not in data:
                data["conformal_correction"] = data["calibration_margin"]
            if "distribution_shift" in data and "distribution_shift_score" not in data:
                data["distribution_shift_score"] = data["distribution_shift"]
        return data

    # Backward-compatible property aliases for existing pipeline integration
    @property
    def aleatoric(self) -> float:
        return self.photometric_noise

    @property
    def epistemic(self) -> float:
        return self.predictive_uncertainty

    @property
    def calibration_margin(self) -> float:
        return self.quantile_estimation_variance

    @property
    def distribution_shift(self) -> float:
        return self.distribution_shift_score


class ConformalPredictionInterval(BaseModel):
    r"""
    Finite-sample distribution-free prediction interval with latent density scaling.
    Under exchangeability, guarantees marginal coverage: P(Y_{n+1} \in [lower_bound, upper_bound]) >= 1 - alpha.
    """
    target_property: str = Field(..., description="Target physical quantity (e.g. 'peak_magnitude', 'next_epoch_flux').")
    point_estimate: float = Field(..., description="Central conditional expectation or median prediction.")
    lower_bound: float = Field(..., description="Lower prediction bound: point_estimate - q_{1-alpha}.")
    upper_bound: float = Field(..., description="Upper prediction bound: point_estimate + q_{1-alpha}.")
    coverage_level: float = Field(default=0.90, ge=0.50, lt=1.0, description="Nominal confidence level 1 - alpha.")
    uncertainty_decomposition: Optional[UncertaintyDecomposition] = Field(
        default=None,
        description="Explicitly decomposed physical and statistical uncertainty components."
    )
    latent_density_sigma: float = Field(
        default=0.0,
        ge=0.0,
        description="Local k-NN distance in foundation embedding space; proxy for epistemic sparsity."
    )
    is_out_of_distribution: bool = Field(
        default=False,
        description="Flagged True if latent_density_sigma or distribution_shift_score exceeds the calibration threshold."
    )

    @property
    def interval_width(self) -> float:
        """Returns the width of the prediction interval: 2 * q_{1-alpha}."""
        return max(0.0, self.upper_bound - self.lower_bound)

    def contains(self, true_value: float) -> bool:
        """Checks if the empirical ground truth is covered by the conformal interval."""
        return self.lower_bound <= true_value <= self.upper_bound
