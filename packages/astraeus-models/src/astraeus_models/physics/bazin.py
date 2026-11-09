"""
ASTRAEUS Bazin Morphology Parameterization Fitter.
Fits the empirical Bazin function:
f(t) = A * [exp(-(t - t0) / tau_fall) / (1 + exp(-(t - t0) / tau_rise))] + B
with parameter uncertainties, covariance, and reduced chi-square.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.optimize import curve_fit
from pydantic import BaseModel, Field

from astraeus_models.physics.fit_status import FitStatus


class BazinFitResult(BaseModel):
    status: FitStatus
    best_fit_params: Dict[str, float] = Field(default_factory=dict)
    param_uncertainties: Dict[str, float] = Field(default_factory=dict)
    reduced_chi2: float = 0.0
    bic: float = 0.0
    residuals: List[float] = Field(default_factory=list)
    applicability_justification: str = ""


def bazin_function(
    t: np.ndarray,
    t0: float,
    a: float,
    b: float,
    tau_rise: float,
    tau_fall: float
) -> np.ndarray:
    """Standard 5-parameter Bazin profile."""
    rise_term = np.clip(-(t - t0) / max(tau_rise, 1e-4), -50.0, 50.0)
    fall_term = np.clip(-(t - t0) / max(tau_fall, 1e-4), -50.0, 50.0)
    return a * (np.exp(fall_term) / (1.0 + np.exp(rise_term))) + b


def fit_bazin(
    times_mjd: np.ndarray,
    fluxes: np.ndarray,
    flux_errors: np.ndarray,
    is_applicable: bool = True,
    justification: str = ""
) -> BazinFitResult:
    """
    Fits the Bazin function to photometric flux data using non-linear least squares.
    """
    if not is_applicable:
        return BazinFitResult(
            status=FitStatus.MODEL_NOT_APPLICABLE,
            applicability_justification=justification,
        )

    n = len(times_mjd)
    if n < 5:
        return BazinFitResult(
            status=FitStatus.INSUFFICIENT_OBSERVATIONS,
            applicability_justification=f"Epochs {n} < 5",
        )

    # Initial parameter heuristics
    t_peak_idx = int(np.argmax(fluxes))
    t0_init = float(times_mjd[t_peak_idx])
    a_init = float(np.max(fluxes) - np.min(fluxes))
    b_init = float(np.min(fluxes))
    tau_rise_init = 4.0
    tau_fall_init = 20.0

    p0 = [t0_init, a_init, b_init, tau_rise_init, tau_fall_init]
    bounds = (
        [times_mjd[0] - 30.0, 0.0, -10.0, 0.1, 1.0],
        [times_mjd[-1] + 30.0, a_init * 10.0 + 100.0, 100.0, 50.0, 200.0],
    )

    try:
        popt, pcov = curve_fit(
            bazin_function,
            times_mjd,
            fluxes,
            p0=p0,
            sigma=np.clip(flux_errors, 1e-4, None),
            absolute_sigma=True,
            bounds=bounds,
            maxfev=2000,
        )

        perr = np.sqrt(np.diag(pcov))
        param_names = ["t0", "a", "b", "tau_rise", "tau_fall"]
        params_dict = {k: float(v) for k, v in zip(param_names, popt)}
        err_dict = {k: float(v) for k, v in zip(param_names, perr)}

        # Evaluate model fit and chi-square
        f_pred = bazin_function(times_mjd, *popt)
        residuals = (fluxes - f_pred) / np.clip(flux_errors, 1e-4, None)
        chi2 = float(np.sum(residuals ** 2))
        dof = max(n - 5, 1)
        chi2_red = float(chi2 / dof)
        bic = float(chi2 + 5 * np.log(n))

        # Check for unphysical or poor fit
        if np.any(np.isnan(perr)) or np.any(np.isinf(perr)):
            status = FitStatus.NON_IDENTIFIABLE_PARAMETERS
        elif chi2_red > 10.0:
            status = FitStatus.FIT_POOR_QUALITY
        else:
            status = FitStatus.FIT_SUCCESS

        return BazinFitResult(
            status=status,
            best_fit_params=params_dict,
            param_uncertainties=err_dict,
            reduced_chi2=chi2_red,
            bic=bic,
            residuals=residuals.tolist(),
            applicability_justification="Bazin transient morphology fit converged.",
        )

    except Exception as e:
        return BazinFitResult(
            status=FitStatus.OPTIMIZATION_FAILED,
            applicability_justification=f"Optimizer failed: {str(e)}",
        )
