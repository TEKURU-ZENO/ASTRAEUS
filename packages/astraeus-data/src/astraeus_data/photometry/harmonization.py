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
    # ATLAS (Tonry et al. 2018, Smith et al. 2020)
    "ATLAS_c": {"lambda_eff_angstrom": 5330.0, "fwhm_angstrom": 2280.0, "extinction_coeff": 3.120},
    "ATLAS_o": {"lambda_eff_angstrom": 6790.0, "fwhm_angstrom": 2570.0, "extinction_coeff": 2.220},
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
    # Rubin LSST (Ivezić et al. 2019)
    "u_LSST": {"lambda_eff_angstrom": 3671.0, "fwhm_angstrom": 574.0, "extinction_coeff": 4.810},
    "g_LSST": {"lambda_eff_angstrom": 4827.0, "fwhm_angstrom": 1488.0, "extinction_coeff": 3.640},
    "r_LSST": {"lambda_eff_angstrom": 6223.0, "fwhm_angstrom": 1378.0, "extinction_coeff": 2.700},
    "i_LSST": {"lambda_eff_angstrom": 7546.0, "fwhm_angstrom": 1260.0, "extinction_coeff": 2.060},
    "z_LSST": {"lambda_eff_angstrom": 8691.0, "fwhm_angstrom": 1070.0, "extinction_coeff": 1.580},
    "y_LSST": {"lambda_eff_angstrom": 9712.0, "fwhm_angstrom": 970.0, "extinction_coeff": 1.310},
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
    survey_origin: str = Field(..., description="'ZTF', 'Pan-STARRS1', 'DECam', 'ATLAS', or 'Rubin'.")
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
            if f_name.endswith(target_survey)
            or (target_survey == "ZTF" and f_name.startswith("ZTF"))
            or (target_survey == "ATLAS" and f_name.startswith("ATLAS"))
            or (target_survey in ("LSST", "Rubin") and f_name.endswith("LSST"))
        }

        if not candidates:
            raise ValueError(f"No filters found for target survey '{target_survey}'.")

        best_f = min(candidates.keys(), key=lambda k: abs(candidates[k]["lambda_eff_angstrom"] - src_wl))
        delta_wl = abs(candidates[best_f]["lambda_eff_angstrom"] - src_wl)
        return best_f, float(delta_wl)

    @classmethod
    def ztf_to_atlas_flux(
        cls,
        flux_g_ujy: float,
        flux_r_ujy: float,
        flux_g_err_ujy: float,
        flux_r_err_ujy: float,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Transforms synchronous ZTF (g, r) fluxes to ATLAS (c, o) fluxes using
        empirical stellar and supernova color transformation polynomials
        (Tonry et al. 2018, Smith et al. 2020) with first-order error propagation.

        Polynomials (in AB magnitudes):
          (c - g) = -0.407 * (g - r) + 0.016
          (o - r) =  0.222 * (g - r) - 0.035
        """
        if flux_g_ujy > 0.0 and flux_r_ujy > 0.0:
            g_mag, g_err = cls.flux_ujy_to_mag(flux_g_ujy, flux_g_err_ujy)
            r_mag, r_err = cls.flux_ujy_to_mag(flux_r_ujy, flux_r_err_ujy)
            if g_mag is not None and r_mag is not None and g_err is not None and r_err is not None:
                g_r = g_mag - r_mag
                g_r_err = np.sqrt(g_err**2 + r_err**2)

                c_mag = g_mag - 0.407 * g_r + 0.016
                c_err = np.sqrt(g_err**2 + (0.407 * g_r_err)**2 + 0.015**2)

                o_mag = r_mag + 0.222 * g_r - 0.035
                o_err = np.sqrt(r_err**2 + (0.222 * g_r_err)**2 + 0.015**2)

                flux_c, flux_c_err = cls.mag_to_flux_ujy(c_mag, c_err)
                flux_o, flux_o_err = cls.mag_to_flux_ujy(o_mag, o_err)
                return {
                    "ATLAS_c": (flux_c, flux_c_err),
                    "ATLAS_o": (flux_o, flux_o_err),
                }

        # Fallback linear effective wavelength interpolation for non-positive / difference flux
        w_c_g = (6390.0 - 5330.0) / (6390.0 - 4770.0)  # ~0.654
        w_c_r = 1.0 - w_c_g
        flux_c = w_c_g * flux_g_ujy + w_c_r * flux_r_ujy
        err_c = np.sqrt((w_c_g * flux_g_err_ujy)**2 + (w_c_r * flux_r_err_ujy)**2 + (0.05 * abs(flux_c))**2)

        flux_o = 0.25 * flux_g_ujy + 0.75 * flux_r_ujy
        err_o = np.sqrt((0.25 * flux_g_err_ujy)**2 + (0.75 * flux_r_err_ujy)**2 + (0.05 * abs(flux_o))**2)

        return {
            "ATLAS_c": (float(flux_c), float(err_c)),
            "ATLAS_o": (float(flux_o), float(err_o)),
        }

    @classmethod
    def atlas_to_ztf_flux(
        cls,
        flux_c_ujy: float,
        flux_o_ujy: float,
        flux_c_err_ujy: float,
        flux_o_err_ujy: float,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Transforms synchronous ATLAS (c, o) fluxes to ZTF (g, r) fluxes by inverting
        the Tonry et al. (2018) color equations with uncertainty propagation.
        """
        if flux_c_ujy > 0.0 and flux_o_ujy > 0.0:
            c_mag, c_err = cls.flux_ujy_to_mag(flux_c_ujy, flux_c_err_ujy)
            o_mag, o_err = cls.flux_o_ujy_to_mag(flux_o_ujy, flux_o_err_ujy) if hasattr(cls, "flux_o_ujy_to_mag") else cls.flux_ujy_to_mag(flux_o_ujy, flux_o_err_ujy)
            if c_mag is not None and o_mag is not None and c_err is not None and o_err is not None:
                # c - o = (g - r) * (1 - 0.407 - 0.222) + 0.016 + 0.035 = 0.371 * (g - r) + 0.051
                c_o = c_mag - o_mag
                g_r = (c_o - 0.051) / 0.371
                g_r_err = np.sqrt(c_err**2 + o_err**2) / 0.371

                g_mag = c_mag + 0.407 * g_r - 0.016
                g_err = np.sqrt(c_err**2 + (0.407 * g_r_err)**2 + 0.020**2)

                r_mag = o_mag - 0.222 * g_r + 0.035
                r_err = np.sqrt(o_err**2 + (0.222 * g_r_err)**2 + 0.020**2)

                flux_g, flux_g_err = cls.mag_to_flux_ujy(g_mag, g_err)
                flux_r, flux_r_err = cls.mag_to_flux_ujy(r_mag, r_err)
                return {
                    "ZTF_g": (flux_g, flux_g_err),
                    "ZTF_r": (flux_r, flux_r_err),
                }

        # Fallback linear approximation
        flux_g = 1.25 * flux_c_ujy - 0.25 * flux_o_ujy
        err_g = np.sqrt((1.25 * flux_c_err_ujy)**2 + (0.25 * flux_o_err_ujy)**2 + (0.05 * abs(flux_g))**2)

        flux_r = 0.15 * flux_c_ujy + 0.85 * flux_o_ujy
        err_r = np.sqrt((0.15 * flux_c_err_ujy)**2 + (0.85 * flux_o_err_ujy)**2 + (0.05 * abs(flux_r))**2)

        return {
            "ZTF_g": (float(flux_g), float(err_g)),
            "ZTF_r": (float(flux_r), float(err_r)),
        }
