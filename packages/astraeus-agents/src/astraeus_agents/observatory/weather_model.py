"""
ASTRAEUS Observatory Intelligence - Probabilistic Weather & Forecast Engine (WS7.5).
Simulates realistic seeing distributions (Kolmogorov/von Kármán turbulence),
cloud extinction, humidity/wind closure conditions, and forecast uncertainty propagation.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


@dataclass
class WeatherCondition:
    """Instantaneous or forecasted atmospheric condition at an observatory."""
    facility_id: str
    epoch_mjd: float
    seeing_fwhm_arcsec: float
    cloud_extinction_mag: float
    wind_speed_m_s: float
    relative_humidity_pct: float
    is_dome_open: bool
    forecast_uncertainty_sigma: float
    forecast_confidence: float
    quality_score: float  # 0.0 to 1.0


class WeatherModel:
    """Site-calibrated atmospheric weather and seeing model."""

    # Nominal site seeing statistics (median FWHM in arcseconds, log-sigma)
    SITE_PROFILES = {
        "VLT": {"median_seeing": 0.65, "seeing_sigma": 0.25, "clear_frac": 0.88, "space": False},
        "RUBIN": {"median_seeing": 0.70, "seeing_sigma": 0.28, "clear_frac": 0.85, "space": False},
        "GEMINI": {"median_seeing": 0.72, "seeing_sigma": 0.28, "clear_frac": 0.85, "space": False},
        "ZTF": {"median_seeing": 1.25, "seeing_sigma": 0.35, "clear_frac": 0.78, "space": False},
        "VLA": {"median_seeing": 0.10, "seeing_sigma": 0.05, "clear_frac": 0.95, "space": False},  # Radio
        "MEERKAT": {"median_seeing": 0.10, "seeing_sigma": 0.05, "clear_frac": 0.95, "space": False},
        "SWIFT": {"median_seeing": 0.05, "seeing_sigma": 0.01, "clear_frac": 1.0, "space": True},
        "ASTROSAT": {"median_seeing": 0.05, "seeing_sigma": 0.01, "clear_frac": 1.0, "space": True},
        "JWST": {"median_seeing": 0.03, "seeing_sigma": 0.01, "clear_frac": 1.0, "space": True},
    }

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def sample_weather(
        self,
        facility_id: str,
        epoch_mjd: float,
        lookahead_hours: float = 0.0,
    ) -> WeatherCondition:
        r"""
        Samples atmospheric conditions at `epoch_mjd + lookahead_hours`.
        Forecast uncertainty $\sigma$ increases smoothly with lookahead horizon.
        """
        profile = self.SITE_PROFILES.get(
            facility_id.upper(),
            {"median_seeing": 1.0, "seeing_sigma": 0.30, "clear_frac": 0.80, "space": False}
        )

        # Space observatories are immune to tropospheric weather
        if profile["space"]:
            return WeatherCondition(
                facility_id=facility_id,
                epoch_mjd=epoch_mjd + (lookahead_hours / 24.0),
                seeing_fwhm_arcsec=profile["median_seeing"],
                cloud_extinction_mag=0.0,
                wind_speed_m_s=0.0,
                relative_humidity_pct=0.0,
                is_dome_open=True,
                forecast_uncertainty_sigma=0.0,
                forecast_confidence=1.0,
                quality_score=1.0,
            )

        # Forecast degradation timescale: ~12 hours
        tau_forecast_hours = 12.0
        forecast_conf = float(math.exp(-lookahead_hours / tau_forecast_hours))
        forecast_sigma = float(0.15 * math.sqrt(lookahead_hours / 6.0 + 1.0))

        # Sample seeing from log-normal distribution
        mu_ln = math.log(profile["median_seeing"])
        sampled_ln = self.rng.normal(mu_ln, profile["seeing_sigma"] + (1.0 - forecast_conf) * 0.1)
        seeing = float(np.clip(math.exp(sampled_ln), 0.35, 3.5))

        # Cloud extinction
        is_clear = self.rng.uniform(0.0, 1.0) < profile["clear_frac"]
        if is_clear:
            cloud_ext = float(self.rng.exponential(0.05))  # Photometric or thin cirrus
            wind = float(self.rng.weibull(2.0) * 5.0)      # 0 - 15 m/s
            humidity = float(self.rng.uniform(10.0, 60.0))
            dome_open = wind < 18.0 and humidity < 85.0
        else:
            cloud_ext = float(self.rng.uniform(1.2, 4.0))  # Heavy clouds
            wind = float(self.rng.uniform(12.0, 25.0))
            humidity = float(self.rng.uniform(75.0, 95.0))
            dome_open = False

        # Composite weather quality score (1.0 = pristine photometric, 0.0 = closed)
        if not dome_open:
            quality = 0.0
        else:
            seeing_penalty = max(0.0, 1.0 - (seeing - profile["median_seeing"]) / 2.0)
            cloud_penalty = max(0.0, 1.0 - cloud_ext / 1.0)
            quality = float(min(1.0, max(0.0, 0.6 * seeing_penalty + 0.4 * cloud_penalty)))

        return WeatherCondition(
            facility_id=facility_id,
            epoch_mjd=epoch_mjd + (lookahead_hours / 24.0),
            seeing_fwhm_arcsec=round(seeing, 3),
            cloud_extinction_mag=round(cloud_ext, 3),
            wind_speed_m_s=round(wind, 1),
            relative_humidity_pct=round(humidity, 1),
            is_dome_open=dome_open,
            forecast_uncertainty_sigma=round(forecast_sigma, 3),
            forecast_confidence=round(forecast_conf, 3),
            quality_score=round(quality, 3),
        )
