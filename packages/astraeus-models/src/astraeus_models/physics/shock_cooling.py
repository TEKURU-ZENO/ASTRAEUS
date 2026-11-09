"""
ASTRAEUS Shock-Cooling Emission Forward Fitter.
Models early-time post-breakout shock cooling of extended circumstellar or stellar envelopes
(Rabinak & Waxman 2011; Piro & Nakar 2013; Waxman & Katz 2017).
Valid strictly for early epochs: (t - t0) <= 5.0 days post-breakout.
Estimates: R_* (progenitor radius in R_sun), v_s (shock velocity in 10^4 km/s), t0 (breakout epoch).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.optimize import curve_fit
from pydantic import BaseModel, Field

from astraeus_models.physics.fit_status import FitStatus

# Physical constants for shock breakout cooling
RSUN_CM = 6.957e10    # Solar radius in cm
C_LIGHT = 2.998e10    # Speed of light in cm/s


class ShockCoolingFitResult(BaseModel):
    status: FitStatus
    progenitor_radius_rsun: Optional[float] = None
    shock_velocity_10e4_kms: Optional[float] = None
    explosion_mjd: Optional[float] = None
    best_fit_params: Dict[str, float] = Field(default_factory=dict)
    param_uncertainties: Dict[str, float] = Field(default_factory=dict)
    reduced_chi2: float = 0.0
    bic: float = 0.0
    residuals: List[float] = Field(default_factory=list)
    applicability_justification: str = ""


def shock_cooling_luminosity(
    dt_days: np.ndarray,
    r_rsun: float,
    v_shock: float
) -> np.ndarray:
    """
    Computes shock cooling bolometric luminosity in units of 10^42 erg/s
    following Rabinak & Waxman (2011) / Piro (2015):
    L(t) ~ 1.0 * (R_* / 100 R_sun) * (v_s / 10^4 km/s)^2 * (t / 1 d)^(-4/3).
    """
    dt_pos = np.clip(dt_days, 0.05, 10.0)
    # L in 10^42 erg/s:
    l_42 = (r_rsun / 100.0) * (v_shock ** 2) * (dt_pos ** (-4.0 / 3.0))
    return l_42


def shock_cooling_photometric_flux(
    t: np.ndarray,
    t0: float,
    r_rsun: float,
    v_shock: float,
    flux_scale: float
) -> np.ndarray:
    """
    Photometric flux forward model for early-time envelope cooling.
    """
    dt = t - t0
    mask_pre = dt <= 0.0
    mask_post_valid = dt > 5.0  # Beyond validity envelope

    dt_pos = np.clip(dt, 0.05, 5.0)
    l_cool = shock_cooling_luminosity(dt_pos, r_rsun, v_shock)
    flux = flux_scale * l_cool
    flux[mask_pre] = 0.0
    # Rapid suppression beyond 5 days as core diffusion or recombination dominates
    if np.any(mask_post_valid):
        decay_factor = np.exp(-0.5 * (dt[mask_post_valid] - 5.0))
        flux[mask_post_valid] = flux_scale * shock_cooling_luminosity(np.full_like(dt[mask_post_valid], 5.0), r_rsun, v_shock) * decay_factor

    return flux


def fit_shock_cooling(
    times_mjd: np.ndarray,
    fluxes: np.ndarray,
    flux_errors: np.ndarray,
    is_applicable: bool = True,
    justification: str = ""
) -> ShockCoolingFitResult:
    """
    Fits the early-time shock cooling forward model to photometric data.
    """
    if not is_applicable:
        return ShockCoolingFitResult(
            status=FitStatus.MODEL_NOT_APPLICABLE,
            applicability_justification=justification,
        )

    n = len(times_mjd)
    if n < 3:
        return ShockCoolingFitResult(
            status=FitStatus.INSUFFICIENT_OBSERVATIONS,
            applicability_justification=f"Epochs {n} < 3",
        )

    # Initial parameter heuristics
    t_min = float(times_mjd[0])
    t0_init = t_min - 0.5  # Breakout occurred shortly before first detection
    r_init = 50.0          # 50 Solar radii
    v_init = 1.0           # 1.0 * 10^4 km/s = 10,000 km/s
    scale_init = float(np.max(fluxes)) / 2.0

    p0 = [t0_init, r_init, v_init, scale_init]
    bounds = (
        [t_min - 5.0, 0.1, 0.2, 1e-3],
        [t_min + 0.5, 1500.0, 5.0, 1e5],
    )

    try:
        popt, pcov = curve_fit(
            shock_cooling_photometric_flux,
            times_mjd,
            fluxes,
            p0=p0,
            sigma=np.clip(flux_errors, 1e-4, None),
            absolute_sigma=True,
            bounds=bounds,
            maxfev=2500,
        )

        perr = np.sqrt(np.diag(pcov))
        param_names = ["t0", "r_rsun", "v_shock", "flux_scale"]
        params_dict = {k: float(v) for k, v in zip(param_names, popt)}
        err_dict = {k: float(v) for k, v in zip(param_names, perr)}

        # Evaluate model fit and chi-square
        f_pred = shock_cooling_photometric_flux(times_mjd, *popt)
        residuals = (fluxes - f_pred) / np.clip(flux_errors, 1e-4, None)
        chi2 = float(np.sum(residuals ** 2))
        dof = max(n - 4, 1)
        chi2_red = float(chi2 / dof)
        bic = float(chi2 + 4 * np.log(n))

        r_fit = params_dict["r_rsun"]
        v_fit = params_dict["v_shock"]

        if np.any(np.isnan(perr)) or np.any(np.isinf(perr)):
            status = FitStatus.NON_IDENTIFIABLE_PARAMETERS
        elif r_fit < 0.1 or r_fit > 1500.0 or v_fit < 0.2:
            status = FitStatus.FIT_POOR_QUALITY
        elif chi2_red > 10.0:
            status = FitStatus.FIT_POOR_QUALITY
        else:
            status = FitStatus.FIT_SUCCESS

        return ShockCoolingFitResult(
            status=status,
            progenitor_radius_rsun=r_fit,
            shock_velocity_10e4_kms=v_fit,
            explosion_mjd=params_dict["t0"],
            best_fit_params=params_dict,
            param_uncertainties=err_dict,
            reduced_chi2=chi2_red,
            bic=bic,
            residuals=residuals.tolist(),
            applicability_justification="Shock cooling model converged on early-time photometric data.",
        )

    except Exception as e:
        return ShockCoolingFitResult(
            status=FitStatus.OPTIMIZATION_FAILED,
            applicability_justification=f"Optimizer failed: {str(e)}",
        )
