"""
ASTRAEUS Periodicity Extraction Module.
Computes Lomb-Scargle periodograms, peak periods, false alarm probabilities, and phase folding.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from astropy.timeseries import LombScargle
from pydantic import BaseModel, Field


class PeriodicityResult(BaseModel):
    """
    Extracted periodic properties from light-curve analysis.
    """
    best_period_days: float = Field(..., ge=0.0)
    best_frequency_1_per_day: float = Field(..., ge=0.0)
    peak_power: float = Field(..., ge=0.0, le=1.0)
    false_alarm_probability: float = Field(..., ge=0.0, le=1.0)
    is_periodic: bool = Field(default=False)
    folded_phases: Optional[list[float]] = None


class LombScargleExtractor:
    """
    Computes Lomb-Scargle periodograms on unevenly sampled photometric time series.
    """

    def __init__(
        self,
        min_period_days: float = 0.05,
        max_period_days: float = 100.0,
        samples_per_peak: int = 5,
        fap_threshold: float = 0.01
    ):
        self.min_period = min_period_days
        self.max_period = max_period_days
        self.samples_per_peak = samples_per_peak
        self.fap_threshold = fap_threshold

    def extract(
        self,
        times_mjd: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray
    ) -> PeriodicityResult:
        """
        Runs Lomb-Scargle periodicity analysis on clean observation epochs.
        """
        if len(times_mjd) < 5:
            return PeriodicityResult(
                best_period_days=0.0,
                best_frequency_1_per_day=0.0,
                peak_power=0.0,
                false_alarm_probability=1.0,
                is_periodic=False,
            )

        ls = LombScargle(times_mjd, magnitudes, errors)
        min_freq = 1.0 / self.max_period
        max_freq = 1.0 / self.min_period

        frequency, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=self.samples_per_peak
        )

        best_idx = int(np.argmax(power))
        best_freq = float(frequency[best_idx])
        best_power = float(power[best_idx])
        best_period = float(1.0 / best_freq) if best_freq > 0 else 0.0

        # Calculate False Alarm Probability (FAP)
        try:
            fap = float(ls.false_alarm_probability(best_power))
            fap = np.clip(fap, 0.0, 1.0)
        except Exception:
            fap = 1.0

        is_periodic = (fap < self.fap_threshold) and (best_power > 0.30)

        # Compute folded phases in [0, 1)
        t0 = times_mjd[0]
        folded = ((times_mjd - t0) / best_period) % 1.0 if best_period > 0 else []

        return PeriodicityResult(
            best_period_days=best_period,
            best_frequency_1_per_day=best_freq,
            peak_power=best_power,
            false_alarm_probability=fap,
            is_periodic=is_periodic,
            folded_phases=folded.tolist() if len(folded) > 0 else None,
        )
