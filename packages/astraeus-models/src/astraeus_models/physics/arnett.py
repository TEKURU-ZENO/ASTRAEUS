"""
ASTRAEUS Arnett Radioactive Nickel Diffusion Fitter.
Models radioactive heating and photon diffusion for Type Ia and core-collapse supernovae:
56Ni -> 56Co -> 56Fe decay chain.
Estimates: M_Ni (Nickel mass in M_sun), tau_d (diffusion timescale in days), t0 (explosion epoch).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from pydantic import BaseModel, Field

from astraeus_models.physics.fit_status import FitStatus

# Physical constants for radioactive decay chain
TAU_NI = 8.77    # Nickel-56 half-life in days / ln(2) -> mean lifetime in days
TAU_CO = 111.3   # Cobalt-56 mean lifetime in days
EPS_NI = 3.90e10 # erg / s / g
EPS_CO = 6.78e9  # erg / s / g
MSUN_G = 1.989e33 # Grams per solar mass


class ArnettFitResult(BaseModel):
    status: FitStatus
    nickel_mass_msun: Optional[float] = None
    diffusion_time_days: Optional[float] = None
    explosion_mjd: Optional[float] = None
    best_fit_params: Dict[str, float] = Field(default_factory=dict)
    param_uncertainties: Dict[str, float] = Field(default_factory=dict)
    reduced_chi2: float = 0.0
    bic: float = 0.0
    residuals: List[float] = Field(default_factory=list)
    applicability_justification: str = ""


def arnett_bolometric_luminosity(t_days: np.ndarray, m_ni_msun: float, tau_d: float) -> np.ndarray:
    """
    Computes bolometric luminosity L(t) in units of 10^43 erg/s using Arnett (1982) formulation.
    """
    m_ni_g = m_ni_msun * MSUN_G
    t_clamped = np.clip(t_days, 1e-3, 200.0)
    x = t_clamped / max(tau_d, 1.0)

    # Arnett diffusion integral kernel: e^{-x^2} * int_0^x 2*z * e^{z^2} * eps(z*tau_d) dz
    # Approximated analytically for fast vectorised execution
    # For z <= x:
    decay_energy = (
        EPS_NI * np.exp(-t_clamped / TAU_NI)
        + EPS_CO * (np.exp(-t_clamped / TAU_CO) - np.exp(-t_clamped / TAU_NI))
    )
    # Trapping and diffusion factor: (1 - exp(-(tau_d / t)^2 / 2)) * decay_energy
    # Normalized so peak occurs around tau_d
    diff_factor = (1.0 - np.exp(-0.5 * (x ** 2))) / (1.0 + 0.1 * (x ** 2))
    lum_cgs = m_ni_g * decay_energy * diff_factor
    return lum_cgs / 1e43  # Scale to 10^43 erg/s


def arnett_photometric_flux(
    t: np.ndarray,
    t0: float,
    m_ni: float,
    tau_d: float,
    flux_scale: float
) -> np.ndarray:
    """Photometric flux profile forward model."""
    dt = t - t0
    mask_pre = dt <= 0.0
    dt_pos = np.clip(dt, 1e-4, 300.0)

    l_bolo = arnett_bolometric_luminosity(dt_pos, m_ni, tau_d)
    flux = flux_scale * l_bolo
    flux[mask_pre] = 0.0
    return flux


def fit_arnett(
    times_mjd: np.ndarray,
    fluxes: np.ndarray,
    flux_errors: np.ndarray,
    is_applicable: bool = True,
    justification: str = ""
) -> ArnettFitResult:
    """
    Fits the Arnett radioactive diffusion model to photometric light curve observations.
    """
    if not is_applicable:
        return ArnettFitResult(
            status=FitStatus.MODEL_NOT_APPLICABLE,
            applicability_justification=justification,
        )

    n = len(times_mjd)
    if n < 5:
        return ArnettFitResult(
            status=FitStatus.INSUFFICIENT_OBSERVATIONS,
            applicability_justification=f"Epochs {n} < 5",
        )

    t_peak_idx = int(np.argmax(fluxes))
    t_peak = float(times_mjd[t_peak_idx])
    t0_init = t_peak - 18.0  # Typical SNe Ia rise time ~18 days
    m_ni_init = 0.5          # Typical SNe Ia M_Ni ~ 0.5 M_sun
    tau_d_init = 15.0        # Diffusion time ~15 days
    scale_init = float(np.max(fluxes)) / 1.5

    p0 = [t0_init, m_ni_init, tau_d_init, scale_init]
    bounds = (
        [times_mjd[0] - 40.0, 0.01, 3.0, 1e-3],
        [t_peak, 3.0, 60.0, 1e5],
    )

    try:
        popt, pcov = curve_fit(
            arnett_photometric_flux,
            times_mjd,
            fluxes,
            p0=p0,
            sigma=np.clip(flux_errors, 1e-4, None),
            absolute_sigma=True,
            bounds=bounds,
            maxfev=3000,
        )

        perr = np.sqrt(np.diag(pcov))
        param_names = ["t0", "m_ni", "tau_d", "flux_scale"]
        params_dict = {k: float(v) for k, v in zip(param_names, popt)}
        err_dict = {k: float(v) for k, v in zip(param_names, perr)}

        # Evaluate model fit and chi-square
        f_pred = arnett_photometric_flux(times_mjd, *popt)
        residuals = (fluxes - f_pred) / np.clip(flux_errors, 1e-4, None)
        chi2 = float(np.sum(residuals ** 2))
        dof = max(n - 4, 1)
        chi2_red = float(chi2 / dof)
        bic = float(chi2 + 4 * np.log(n))

        # Check physical parameter bounds
        m_ni_fit = params_dict["m_ni"]
        if np.any(np.isnan(perr)) or np.any(np.isinf(perr)):
            status = FitStatus.NON_IDENTIFIABLE_PARAMETERS
        elif m_ni_fit < 0.01 or m_ni_fit > 2.5:
            status = FitStatus.FIT_POOR_QUALITY  # Unphysical Nickel mass
        elif chi2_red > 10.0:
            status = FitStatus.FIT_POOR_QUALITY
        else:
            status = FitStatus.FIT_SUCCESS

        return ArnettFitResult(
            status=status,
            nickel_mass_msun=m_ni_fit,
            diffusion_time_days=params_dict["tau_d"],
            explosion_mjd=params_dict["t0"],
            best_fit_params=params_dict,
            param_uncertainties=err_dict,
            reduced_chi2=chi2_red,
            bic=bic,
            residuals=residuals.tolist(),
            applicability_justification="Arnett radioactive diffusion model converged.",
        )

    except Exception as e:
        return ArnettFitResult(
            status=FitStatus.OPTIMIZATION_FAILED,
            applicability_justification=f"Optimizer failed: {str(e)}",
        )
