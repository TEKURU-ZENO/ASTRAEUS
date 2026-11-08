"""
ASTRAEUS Astronomical Feature Extraction Module.
Extracts physically motivated statistical, kinematic, and morphological features
from light curves across optical passbands.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field


class LightCurveFeatures(BaseModel):
    """
    Extracted feature vector for classical machine learning baselines (Random Forest).
    """
    num_epochs: int
    duration_days: float
    peak_mag: float
    faintest_mag: float
    amplitude: float
    rise_rate_mag_per_day: float
    decay_rate_mag_per_day: float
    flux_skewness: float
    flux_kurtosis: float
    stetson_j: float
    clean_fraction: float
    mean_error: float
    median_color_gr: Optional[float] = None


class AstronomicalFeatureExtractor:
    """
    Computes statistical and kinematic features from multi-epoch light curves.
    """

    def extract(
        self,
        times_mjd: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray,
        filter_bands: Optional[List[str]] = None,
        quality_flags: Optional[np.ndarray] = None
    ) -> LightCurveFeatures:
        """
        Extracts scalar feature vector from photometric arrays.
        """
        n = len(times_mjd)
        if n < 3:
            return LightCurveFeatures(
                num_epochs=n,
                duration_days=0.0,
                peak_mag=float(np.min(magnitudes)) if n > 0 else 0.0,
                faintest_mag=float(np.max(magnitudes)) if n > 0 else 0.0,
                amplitude=0.0,
                rise_rate_mag_per_day=0.0,
                decay_rate_mag_per_day=0.0,
                flux_skewness=0.0,
                flux_kurtosis=0.0,
                stetson_j=0.0,
                clean_fraction=1.0,
                mean_error=float(np.mean(errors)) if n > 0 else 0.0,
            )

        duration = float(times_mjd[-1] - times_mjd[0])
        peak_idx = int(np.argmin(magnitudes))  # Brightest = lowest magnitude
        peak_mag = float(magnitudes[peak_idx])
        faintest_mag = float(np.max(magnitudes))
        amplitude = float(faintest_mag - peak_mag)

        # Rise and decay rates (mag per day)
        t_peak = times_mjd[peak_idx]
        t_start = times_mjd[0]
        t_end = times_mjd[-1]

        rise_rate = (faintest_mag - peak_mag) / max(t_peak - t_start, 0.5) if peak_idx > 0 else 0.0
        decay_rate = (faintest_mag - peak_mag) / max(t_end - t_peak, 0.5) if peak_idx < (n - 1) else 0.0

        # Higher-order statistical moments
        skew = float(stats.skew(magnitudes)) if n >= 3 else 0.0
        kurt = float(stats.kurtosis(magnitudes)) if n >= 4 else 0.0

        # Stetson J variability index
        median_mag = np.median(magnitudes)
        normalized_residuals = (magnitudes - median_mag) / np.clip(errors, 1e-4, 10.0)
        stetson_j = float(np.sum(np.sign(normalized_residuals[:-1] * normalized_residuals[1:]) *
                                 np.sqrt(np.abs(normalized_residuals[:-1] * normalized_residuals[1:]))) / (n - 1))

        # Clean fraction
        clean_fraction = float(np.mean(quality_flags == 0)) if quality_flags is not None and len(quality_flags) > 0 else 1.0

        # Median color (g - r) if bands available
        median_color = None
        if filter_bands and len(filter_bands) == n:
            g_mags = [m for m, b in zip(magnitudes, filter_bands) if "g" in b]
            r_mags = [m for m, b in zip(magnitudes, filter_bands) if "r" in b]
            if g_mags and r_mags:
                median_color = float(np.median(g_mags) - np.median(r_mags))

        return LightCurveFeatures(
            num_epochs=n,
            duration_days=duration,
            peak_mag=peak_mag,
            faintest_mag=faintest_mag,
            amplitude=amplitude,
            rise_rate_mag_per_day=rise_rate,
            decay_rate_mag_per_day=decay_rate,
            flux_skewness=skew,
            flux_kurtosis=kurt,
            stetson_j=stetson_j,
            clean_fraction=clean_fraction,
            mean_error=float(np.mean(errors)),
            median_color_gr=median_color,
        )
