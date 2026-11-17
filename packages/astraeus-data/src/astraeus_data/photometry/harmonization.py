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
        if flux_g_ujy <= 0.0 or flux_r_ujy <= 0.0:
            raise ValueError(
                "Non-positive difference flux cannot be transformed across filters via empirical color polynomials. "
                "Per ASTRAEUS-P5-HARMON-002, negative difference flux must be retained in its native passband."
            )

        # Enforce SNR >= 3.0 threshold
        snr_g = flux_g_ujy / flux_g_err_ujy
        snr_r = flux_r_ujy / flux_r_err_ujy
        if snr_g < 3.0 or snr_r < 3.0:
            raise ValueError(
                f"Insufficient SNR for empirical color transformation (SNR_g={snr_g:.2f}, SNR_r={snr_r:.2f}). "
                "Minimum SNR >= 3.0 required per ASTRAEUS-P5-HARMON-002."
            )

        g_mag, g_err = cls.flux_ujy_to_mag(flux_g_ujy, flux_g_err_ujy)
        r_mag, r_err = cls.flux_ujy_to_mag(flux_r_ujy, flux_r_err_ujy)
        if g_mag is None or r_mag is None or g_err is None or r_err is None:
            raise ValueError("Failed to convert positive flux to AB magnitude.")

        g_r = g_mag - r_mag
        if not (-0.4 <= g_r <= 2.0):
            raise ValueError(
                f"Color (g - r) = {g_r:.3f} is outside empirical polynomial validity domain [-0.4, 2.0] "
                "per ASTRAEUS-P5-HARMON-002."
            )

        # Exact transformation polynomials:
        # c = 0.593 * g + 0.407 * r + 0.016
        # o = 0.222 * g + 0.778 * r - 0.035
        c_mag = 0.593 * g_mag + 0.407 * r_mag + 0.016
        o_mag = 0.222 * g_mag + 0.778 * r_mag - 0.035

        # Exact first-order error propagation on independent input variables:
        # Var(c) = 0.593^2 * Var(g) + 0.407^2 * Var(r) + Var(model)
        # Var(o) = 0.222^2 * Var(g) + 0.778^2 * Var(r) + Var(model)
        c_var = (0.593 ** 2) * (g_err ** 2) + (0.407 ** 2) * (r_err ** 2) + (0.015 ** 2)
        o_var = (0.222 ** 2) * (g_err ** 2) + (0.778 ** 2) * (r_err ** 2) + (0.015 ** 2)
        c_err = np.sqrt(c_var)
        o_err = np.sqrt(o_var)

        flux_c, flux_c_err = cls.mag_to_flux_ujy(c_mag, c_err)
        flux_o, flux_o_err = cls.mag_to_flux_ujy(o_mag, o_err)
        return {
            "ATLAS_c": (flux_c, flux_c_err),
            "ATLAS_o": (flux_o, flux_o_err),
        }

    @classmethod
    def compute_atlas_cross_filter_covariance(
        cls,
        g_err_mag: float,
        r_err_mag: float,
    ) -> float:
        """
        Computes analytical magnitude covariance between transformed ATLAS_c and ATLAS_o
        arising from shared (g, r) measurements (partial uncertainty model with independent inputs):
          Cov(c, o) = (0.593 * 0.222) * Var(g) + (0.407 * 0.778) * Var(r)
        """
        return float(0.593 * 0.222 * (g_err_mag ** 2) + 0.407 * 0.778 * (r_err_mag ** 2))

    @classmethod
    def compute_atlas_full_covariance(
        cls,
        g_err_mag: float,
        r_err_mag: float,
        cov_gr_mag: float = 0.0,
        sigma_cal_mag: float = 0.010,
        sigma_model_c: float = 0.015,
        sigma_model_o: float = 0.015,
        cov_model_co: float = 0.0001,
        dt_mismatch_days: float = 0.0,
        mag_rate_per_day: float = 0.0,
        *,
        sigma_cal_g_mag: Optional[float] = None,
        sigma_cal_r_mag: Optional[float] = None,
        rho_cal: float = 1.0,
        mag_rate_c_per_day: Optional[float] = None,
        mag_rate_o_per_day: Optional[float] = None,
        rho_temporal: float = 0.0,
    ) -> np.ndarray:
        """
        Computes the complete 2x2 covariance matrix in magnitude space between transformed ATLAS_c and ATLAS_o:
          Sigma_target = J * Sigma_source * J^T + J * Sigma_cal * J^T + Sigma_model + Sigma_temporal

        Where:
        - J = [[0.593, 0.407], [0.222, 0.778]]
        - Sigma_source = [[Var(g), Cov(g,r)], [Cov(g,r), Var(r)]]
        - Sigma_cal is the input passband calibration covariance matrix:
            [[sigma_cal_g^2, rho_cal * sigma_cal_g * sigma_cal_r],
             [rho_cal * sigma_cal_g * sigma_cal_r, sigma_cal_r^2]]
          Defaults to rho_cal=1.0 and equal uncertainties sigma_cal_mag for fully correlated zero-point shifts.
        - Sigma_model captures transformation model residual variance and cross-filter covariance.
        - Sigma_temporal accounts for light-curve evolution across intra-epoch time mismatch dt:
            [[(rate_c * dt)^2, rho_temp * (rate_c * dt) * (rate_o * dt)],
             [rho_temp * (rate_c * dt) * (rate_o * dt), (rate_o * dt)^2]]
          Allows separate band rates and cross-band temporal correlation.
        """
        # Fail-closed input validation
        for name, val in [
            ("g_err_mag", g_err_mag),
            ("r_err_mag", r_err_mag),
            ("cov_gr_mag", cov_gr_mag),
            ("sigma_cal_mag", sigma_cal_mag),
            ("sigma_model_c", sigma_model_c),
            ("sigma_model_o", sigma_model_o),
            ("cov_model_co", cov_model_co),
            ("dt_mismatch_days", dt_mismatch_days),
            ("mag_rate_per_day", mag_rate_per_day),
        ]:
            if not np.isfinite(val):
                raise ValueError(f"Input '{name}' must be finite, got {val}")

        if g_err_mag < 0.0 or r_err_mag < 0.0:
            raise ValueError(f"Measurement errors must be non-negative: g_err={g_err_mag}, r_err={r_err_mag}")
        if sigma_cal_mag < 0.0:
            raise ValueError(f"Calibration uncertainty sigma_cal_mag={sigma_cal_mag} must be non-negative.")
        if sigma_cal_g_mag is not None and (not np.isfinite(sigma_cal_g_mag) or sigma_cal_g_mag < 0.0):
            raise ValueError(f"sigma_cal_g_mag must be non-negative and finite, got {sigma_cal_g_mag}")
        if sigma_cal_r_mag is not None and (not np.isfinite(sigma_cal_r_mag) or sigma_cal_r_mag < 0.0):
            raise ValueError(f"sigma_cal_r_mag must be non-negative and finite, got {sigma_cal_r_mag}")
        if not (-1.0 <= rho_cal <= 1.0):
            raise ValueError(f"Calibration correlation rho_cal={rho_cal} out of valid range [-1.0, 1.0]")
        if not (-1.0 <= rho_temporal <= 1.0):
            raise ValueError(f"Temporal correlation rho_temporal={rho_temporal} out of valid range [-1.0, 1.0]")
        if dt_mismatch_days < 0.0:
            raise ValueError(f"dt_mismatch_days={dt_mismatch_days} must be non-negative.")
        if mag_rate_per_day < 0.0:
            raise ValueError(f"mag_rate_per_day={mag_rate_per_day} must be non-negative.")

        # Positive semi-definiteness checks on input covariance blocks
        max_source_cov = (g_err_mag * r_err_mag) + 1e-12
        if abs(cov_gr_mag) > max_source_cov:
            raise ValueError(
                f"Source measurement covariance is not positive semi-definite: "
                f"|cov_gr_mag|={abs(cov_gr_mag):.6f} exceeds g_err * r_err = {g_err_mag * r_err_mag:.6f}"
            )

        max_model_cov = (sigma_model_c * sigma_model_o) + 1e-12
        if abs(cov_model_co) > max_model_cov:
            raise ValueError(
                f"Model residual covariance is not positive semi-definite: "
                f"|cov_model_co|={abs(cov_model_co):.6f} exceeds sigma_c * sigma_o = {sigma_model_c * sigma_model_o:.6f}"
            )

        J = np.array([
            [0.593, 0.407],
            [0.222, 0.778]
        ], dtype=np.float64)

        # 1. Source input measurement covariance
        sigma_source = np.array([
            [g_err_mag ** 2, cov_gr_mag],
            [cov_gr_mag, r_err_mag ** 2]
        ], dtype=np.float64)

        # Propagate source measurement error
        sigma_prop = J @ sigma_source @ J.T

        # 2. Shared/correlated calibration covariance across input passbands
        cal_g = sigma_cal_g_mag if sigma_cal_g_mag is not None else sigma_cal_mag
        cal_r = sigma_cal_r_mag if sigma_cal_r_mag is not None else sigma_cal_mag
        cov_cal = rho_cal * cal_g * cal_r
        sigma_cal = np.array([
            [cal_g ** 2, cov_cal],
            [cov_cal, cal_r ** 2]
        ], dtype=np.float64)
        sigma_cal_target = J @ sigma_cal @ J.T

        # 3. Model residual covariance
        sigma_model = np.array([
            [sigma_model_c ** 2, cov_model_co],
            [cov_model_co, sigma_model_o ** 2]
        ], dtype=np.float64)

        # 4. Temporal mismatch variance and cross-band evolution covariance
        rate_c = mag_rate_c_per_day if mag_rate_c_per_day is not None else mag_rate_per_day
        rate_o = mag_rate_o_per_day if mag_rate_o_per_day is not None else mag_rate_per_day
        var_temporal_c = (rate_c * dt_mismatch_days) ** 2
        var_temporal_o = (rate_o * dt_mismatch_days) ** 2
        cov_temporal_co = rho_temporal * (rate_c * dt_mismatch_days) * (rate_o * dt_mismatch_days)
        sigma_temporal = np.array([
            [var_temporal_c, cov_temporal_co],
            [cov_temporal_co, var_temporal_o]
        ], dtype=np.float64)

        # Full combined magnitude covariance
        sigma_total = sigma_prop + sigma_cal_target + sigma_model + sigma_temporal
        return sigma_total

    @classmethod
    def propagate_mag_covariance_to_flux(
        cls,
        cov_mag: np.ndarray,
        flux_c_ujy: float,
        flux_o_ujy: float,
    ) -> np.ndarray:
        """
        Transforms the 2x2 magnitude covariance matrix into physical flux space (uJy^2)
        via first-order Taylor expansion (Jacobian transformation):
          F = F0 * 10^(-0.4 * m)
          dF/dm = -(ln(10) / 2.5) * F
          Sigma_flux = D_flux * Sigma_mag * D_flux^T
        """
        factor = LN10_OVER_2P5
        D_flux = np.diag([
            factor * flux_c_ujy,
            factor * flux_o_ujy,
        ])
        return D_flux @ cov_mag @ D_flux.T

    @classmethod
    def transformation_condition_number(cls) -> float:
        """
        Returns the 2-norm condition number of the ZTF->ATLAS transformation matrix J.
        kappa(J) = sigma_max / sigma_min ~ 2.85 (well-conditioned inversion).
        """
        J = np.array([
            [0.593, 0.407],
            [0.222, 0.778]
        ], dtype=np.float64)
        s = np.linalg.svd(J, compute_uv=False)
        return float(s[0] / s[-1])

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
        the Tonry et al. (2018) color matrix with exact covariance propagation.
        """
        if flux_c_ujy <= 0.0 or flux_o_ujy <= 0.0:
            raise ValueError(
                "Non-positive difference flux cannot be transformed across filters via empirical color polynomials. "
                "Per ASTRAEUS-P5-HARMON-002, negative difference flux must be retained in its native passband."
            )

        snr_c = flux_c_ujy / flux_c_err_ujy
        snr_o = flux_o_ujy / flux_o_err_ujy
        if snr_c < 3.0 or snr_o < 3.0:
            raise ValueError(
                f"Insufficient SNR for empirical color transformation (SNR_c={snr_c:.2f}, SNR_o={snr_o:.2f}). "
                "Minimum SNR >= 3.0 required per ASTRAEUS-P5-HARMON-002."
            )

        c_mag, c_err = cls.flux_ujy_to_mag(flux_c_ujy, flux_c_err_ujy)
        o_mag, o_err = cls.flux_ujy_to_mag(flux_o_ujy, flux_o_err_ujy)
        if c_mag is None or o_mag is None or c_err is None or o_err is None:
            raise ValueError("Failed to convert positive flux to AB magnitude.")

        # Inverted linear system:
        # [g, r]^T = A^-1 * ([c, o]^T - [0.016, -0.035]^T)
        # A^-1 = [[ 2.097035, -1.097035],
        #         [-0.598383,  1.598383]]
        g_mag = 2.097035 * c_mag - 1.097035 * o_mag - 0.07195
        r_mag = -0.598383 * c_mag + 1.598383 * o_mag + 0.06551

        # Error propagation from independent c and o:
        g_var = (2.097035 ** 2) * (c_err ** 2) + (1.097035 ** 2) * (o_err ** 2) + (0.020 ** 2)
        r_var = (0.598383 ** 2) * (c_err ** 2) + (1.598383 ** 2) * (o_err ** 2) + (0.020 ** 2)
        g_err = np.sqrt(g_var)
        r_err = np.sqrt(r_var)

        flux_g, flux_g_err = cls.mag_to_flux_ujy(g_mag, g_err)
        flux_r, flux_r_err = cls.mag_to_flux_ujy(r_mag, r_err)
        return {
            "ZTF_g": (flux_g, flux_g_err),
            "ZTF_r": (flux_r, flux_r_err),
        }

    @classmethod
    def ztf_to_atlas_flux_with_covariance(
        cls,
        flux_g_ujy: float,
        flux_r_ujy: float,
        flux_g_err_ujy: float,
        flux_r_err_ujy: float,
        cov_gr_flux: float = 0.0,
        sigma_cal_mag: float = 0.010,
        sigma_model_c: float = 0.015,
        sigma_model_o: float = 0.015,
        cov_model_co: float = 0.0001,
        dt_mismatch_days: float = 0.0,
        mag_rate_per_day: float = 0.0,
        *,
        sigma_cal_g_mag: Optional[float] = None,
        sigma_cal_r_mag: Optional[float] = None,
        rho_cal: float = 1.0,
        mag_rate_c_per_day: Optional[float] = None,
        mag_rate_o_per_day: Optional[float] = None,
        rho_temporal: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transforms synchronous ZTF (g, r) fluxes into ATLAS (c, o) fluxes with full
        end-to-end 2x2 physical flux covariance matrix propagation.
        """
        if flux_g_ujy <= 0.0 or flux_r_ujy <= 0.0:
            raise ValueError(
                "Non-positive difference flux cannot be transformed across filters via empirical color polynomials. "
                "Per ASTRAEUS-P5-HARMON-002, negative difference flux must be retained in its native passband."
            )

        snr_g = flux_g_ujy / flux_g_err_ujy
        snr_r = flux_r_ujy / flux_r_err_ujy
        if snr_g < 3.0 or snr_r < 3.0:
            raise ValueError(
                f"Insufficient SNR for empirical color transformation (SNR_g={snr_g:.2f}, SNR_r={snr_r:.2f}). "
                "Minimum SNR >= 3.0 required per ASTRAEUS-P5-HARMON-002."
            )

        g_mag, g_err = cls.flux_ujy_to_mag(flux_g_ujy, flux_g_err_ujy)
        r_mag, r_err = cls.flux_ujy_to_mag(flux_r_ujy, flux_r_err_ujy)
        if g_mag is None or r_mag is None or g_err is None or r_err is None:
            raise ValueError("Failed to convert positive flux to AB magnitude.")

        factor = LN10_OVER_2P5
        cov_gr_mag = cov_gr_flux / ((factor * flux_g_ujy) * (factor * flux_r_ujy)) if cov_gr_flux != 0.0 else 0.0

        # Tonry et al. (2018) forward polynomials:
        # c = g - 0.407 * (g - r) + 0.016 = 0.593 * g + 0.407 * r + 0.016
        # o = r + 0.222 * (g - r) - 0.035 = 0.222 * g + 0.778 * r - 0.035
        c_mag = 0.593 * g_mag + 0.407 * r_mag + 0.016
        o_mag = 0.222 * g_mag + 0.778 * r_mag - 0.035

        cov_mag = cls.compute_atlas_full_covariance(
            g_err_mag=g_err,
            r_err_mag=r_err,
            cov_gr_mag=cov_gr_mag,
            sigma_cal_mag=sigma_cal_mag,
            sigma_model_c=sigma_model_c,
            sigma_model_o=sigma_model_o,
            cov_model_co=cov_model_co,
            dt_mismatch_days=dt_mismatch_days,
            mag_rate_per_day=mag_rate_per_day,
            sigma_cal_g_mag=sigma_cal_g_mag,
            sigma_cal_r_mag=sigma_cal_r_mag,
            rho_cal=rho_cal,
            mag_rate_c_per_day=mag_rate_c_per_day,
            mag_rate_o_per_day=mag_rate_o_per_day,
            rho_temporal=rho_temporal,
        )

        c_err = float(np.sqrt(cov_mag[0, 0]))
        o_err = float(np.sqrt(cov_mag[1, 1]))

        flux_c, flux_c_err = cls.mag_to_flux_ujy(c_mag, c_err)
        flux_o, flux_o_err = cls.mag_to_flux_ujy(o_mag, o_err)

        cov_flux = cls.propagate_mag_covariance_to_flux(cov_mag, flux_c, flux_o)

        return {
            "fluxes": {
                "ATLAS_c": (flux_c, flux_c_err),
                "ATLAS_o": (flux_o, flux_o_err),
            },
            "magnitudes": {
                "ATLAS_c": (c_mag, c_err),
                "ATLAS_o": (o_mag, o_err),
            },
            "cov_mag": cov_mag,
            "cov_flux": cov_flux,
        }
