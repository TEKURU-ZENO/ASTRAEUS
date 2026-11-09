"""
ASTRAEUS Photometric Harmonization Module.
Provides cross-survey photometric normalization across ZTF, Pan-STARRS1, and DECam.
Enforces AB system conversions, heteroscedastic uncertainty propagation,
upper-limit preservation, Galactic extinction corrections, and passband mapping.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


# Central effective wavelengths in Angstroms (lambda_eff) and FWHM bandwidths
SURVEY_FILTER_METADATA = {
    # ZTF (Bellm et al. 2019, Dekany et al. 2020)
    "ZTF_g": {"lambda_eff_angstrom": 4770.0, "fwhm_angstrom": 1400.0, "extinction_coeff": 3.686},
    "ZTF_r": {"lambda_eff_angstrom": 6390.0, "fwhm_angstrom": 1500.0, "extinction_coeff": 2.449},
    "ZTF_i": {"lambda_eff_angstrom": 7880.0, "fwhm_angstrom": 1300.0, "extinction_coeff": 1.859},
    # Pan-STARRS1 (Tonry et al. 2012)
    "g_P1": {"lambda_eff_angstrom": 4866.0, "fwhm_angstrom": 1166.0, "extinction_coeff": 3.613},
    "r_P1": {"lambda_eff_angstrom": 6215.0, "fwhm_angstrom": 1315.0, "extinction_coeff": 2.585},
    "i_P1": {"lambda_eff_angstrom": 7545.0, "fwhm_angstrom": 1290.0, "extinction_coeff": 1.908},
    "z_P1": {"lambda_eff_angstrom": 8679.0, "fwhm_angstrom": 1040.0, "extinction_coeff": 1.501},
    "y_P1": {"lambda_eff_angstrom": 9633.0, "fwhm_angstrom": 638.0, "extinction_coeff": 1.252},
    # DECam (Flaugher et al. 2015, Abbott et al. 2018)
    "u_DECam": {"lambda_eff_angstrom": 3560.0, "fwhm_angstrom": 600.0, "extinction_coeff": 4.144},
    "g_DECam": {"lambda_eff_angstrom": 4720.0, "fwhm_angstrom": 1520.0, "extinction_coeff": 3.214},
    "r_DECam": {"lambda_eff_angstrom": 6415.0, "fwhm_angstrom": 1480.0, "extinction_coeff": 2.165},
    "i_DECam": {"lambda_eff_angstrom": 7835.0, "fwhm_angstrom": 1470.0, "extinction_coeff": 1.592},
    "z_DECam": {"lambda_eff_angstrom": 9260.0, "fwhm_angstrom": 1520.0, "extinction_coeff": 1.211},
    "Y_DECam": {"lambda_eff_angstrom": 10095.0, "fwhm_angstrom": 980.0, "extinction_coeff": 1.064},
}

# Standard AB System constants: m_AB = -2.5 * log10(F_uJy) + 23.90
AB_MAG_ZERO_UJY = 23.90
LN10_OVER_2P5 = np.log(10.0) / 2.5  # ~ 0.9210340371976184


class HarmonizedPhotometricEpoch(BaseModel):
    """
    Standardized photometric epoch across optical/NIR surveys.
    Preserves original survey filter, standardizes flux to microJansky,
    and guarantees non-destructive retention of upper limits.
    """
    mjd: float = Field(..., description="Observation mid-point in MJD.")
    survey_origin: str = Field(..., description="'ZTF', 'Pan-STARRS1', or 'DECam'.")
    native_filter: str
    effective_wavelength_angstrom: float
    filter_bandwidth_angstrom: float
    is_upper_limit: bool = Field(..., description="True if non-detection (< 5 sigma); False if secure detection.")
    calibrated_flux_ujy: float = Field(..., description="Standardized flux density in microJanskys.")
    flux_err_ujy: float = Field(..., gt=0.0, description="1-sigma uncertainty on flux density.")
    apparent_mag_ab: Optional[float] = Field(default=None, description="Apparent AB magnitude; None for upper limits.")
    mag_err_ab: Optional[float] = Field(default=None, ge=0.0, description="1-sigma magnitude uncertainty.")
    five_sigma_limiting_mag: float = Field(..., description="5-sigma limiting magnitude of the exposure.")
    dereddened_flux_ujy: Optional[float] = Field(default=None, description="Extinction-corrected flux density.")


class PhotometricHarmonizer:
    """
    Translates heterogeneous survey alert photometry into unified AB flux standards.
    """

    @staticmethod
    def mag_to_flux_ujy(mag_ab: float, mag_err: float) -> Tuple[float, float]:
        """
        Converts apparent AB magnitude to flux density in microJanskys with first-order uncertainty propagation.
        F = 10^(-(m - 23.90) / 2.5)
        sigma_F = (ln(10) / 2.5) * F * sigma_m
        """
        flux = float(10.0 ** (-(mag_ab - AB_MAG_ZERO_UJY) / 2.5))
        flux_err = float(LN10_OVER_2P5 * flux * mag_err)
        return flux, max(flux_err, 1e-4)

    @staticmethod
    def flux_ujy_to_mag(flux_ujy: float, flux_err_ujy: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Converts positive microJansky flux to apparent AB magnitude.
        Returns (None, None) if flux <= 0.
        """
        if flux_ujy <= 0.0:
            return None, None
        mag = float(-2.5 * np.log10(flux_ujy) + AB_MAG_ZERO_UJY)
        mag_err = float((2.5 / np.log(10.0)) * (flux_err_ujy / flux_ujy))
        return mag, mag_err

    @classmethod
    def apply_galactic_extinction(
        cls,
        flux_ujy: float,
        native_filter: str,
        ebv: float
    ) -> float:
        """
        Corrects flux for foreground Milky Way dust extinction:
        A_lambda = R_lambda * E(B - V)
        F_dered = F_obs * 10^(A_lambda / 2.5)
        """
        meta = SURVEY_FILTER_METADATA.get(native_filter)
        if meta is None or ebv <= 0.0:
            return flux_ujy
        a_lambda = meta["extinction_coeff"] * ebv
        return float(flux_ujy * (10.0 ** (a_lambda / 2.5)))

    @classmethod
    def find_nearest_passband(cls, source_filter: str, target_survey: str) -> Tuple[str, float]:
        """
        Finds the closest passband in target_survey by minimizing |lambda_eff_1 - lambda_eff_2|.
        Returns (nearest_filter_name, delta_angstrom).
        """
        src_meta = SURVEY_FILTER_METADATA.get(source_filter)
        if not src_meta:
            raise KeyError(f"Filter '{source_filter}' not recognized in SURVEY_FILTER_METADATA.")

        src_wl = src_meta["lambda_eff_angstrom"]
        candidates = {
            f_name: meta for f_name, meta in SURVEY_FILTER_METADATA.items()
            if f_name.endswith(target_survey) or (target_survey == "ZTF" and f_name.startswith("ZTF"))
        }

        if not candidates:
            raise ValueError(f"No filters found for target survey '{target_survey}'.")

        best_f = min(candidates.keys(), key=lambda k: abs(candidates[k]["lambda_eff_angstrom"] - src_wl))
        delta_wl = abs(candidates[best_f]["lambda_eff_angstrom"] - src_wl)
        return best_f, float(delta_wl)
