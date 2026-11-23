"""
ASTRAEUS Experimental Intelligence: Information-Theoretic EIG Objective (Track C).
Calculates Expected Information Gain (EIG) and Bayesian D-optimality utility for
transient follow-up scheduling, atmospheric extinction, and Fisher uncertainty reduction.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class TransientTarget:
    """Astrophysical transient target representation."""
    target_id: str
    ra_deg: float
    dec_deg: float
    base_magnitude: float            # Apparent magnitude at trigger
    decay_timescale_hours: float     # Exponential flux decay timescale tau
    time_since_trigger_hours: float  # Elapsed time since detection
    priority_weight: float = 1.0     # Scientific priority multiplier
    is_too: bool = False             # Target of Opportunity flag
    prior_variance: float = 1.0      # Current parameter variance sigma_theta^2
    times_observed: int = 0          # Number of visits performed


class ExpectedInformationGain:
    """
    Computes information-theoretic utility (Shannon Mutual Information and Fisher EIG)
    for scheduled telescope pointings.
    """

    def __init__(
        self,
        zero_point_mag: float = 25.0,
        sky_brightness_mag: float = 21.0,
        atmospheric_extinction_coeff: float = 0.15,
        readout_noise_electrons: float = 5.0,
    ):
        self.zp = zero_point_mag
        self.sky_mag = sky_brightness_mag
        self.ext_coeff = atmospheric_extinction_coeff
        self.ron = readout_noise_electrons

    def calculate_apparent_magnitude(self, target: TransientTarget) -> float:
        """Computes current decayed apparent magnitude."""
        # F(t) = F_0 * exp(-t / tau) ==> m(t) = m_0 + 2.5 * log10(e) * (t / tau)
        delta_m = 1.0857 * (target.time_since_trigger_hours / max(target.decay_timescale_hours, 0.1))
        return target.base_magnitude + delta_m

    def compute_snr(
        self,
        apparent_mag: float,
        airmass: float,
        exposure_time_sec: float,
    ) -> float:
        """
        Computes photometric Signal-to-Noise Ratio (SNR) taking into account
        atmospheric extinction at given airmass: m_eff = m + kappa * (X - 1).
        """
        if airmass <= 0 or airmass > 3.0:
            return 0.0

        eff_mag = apparent_mag + self.ext_coeff * (airmass - 1.0)
        # Source flux in electrons/sec: F = 10^(0.4 * (zp - eff_mag))
        source_flux = 10.0 ** (0.4 * (self.zp - eff_mag))
        # Sky background flux in aperture
        sky_flux = 10.0 ** (0.4 * (self.zp - self.sky_mag))

        n_signal = source_flux * exposure_time_sec
        n_sky = sky_flux * exposure_time_sec
        n_noise = math.sqrt(n_signal + n_sky + self.ron ** 2)

        return float(n_signal / max(n_noise, 1e-6))

    def evaluate_target_eig(
        self,
        target: TransientTarget,
        airmass: float,
        exposure_time_sec: float = 300.0,
    ) -> float:
        """
        Computes Expected Information Gain (EIG):
        EIG = 0.5 * log(1 + sigma_prior^2 / sigma_meas^2) * priority
        where measurement variance sigma_meas^2 = 1 / SNR^2.
        """
        if airmass > 2.2:
            return 0.0  # Above operational horizon limit

        cur_mag = self.calculate_apparent_magnitude(target)
        snr = self.compute_snr(cur_mag, airmass, exposure_time_sec)
        if snr < 1.0:
            return 0.0

        # Measurement precision (Fisher information of flux measurement)
        # Var(theta_obs) ~ 1 / snr^2
        sigma_meas_sq = 1.0 / (snr ** 2)

        # Prior variance on astrophysical model parameters
        prior_var = target.prior_variance

        # Mutual information: Delta H = 0.5 * ln(1 + Var_prior / Var_meas)
        mutual_info = 0.5 * math.log(1.0 + prior_var / sigma_meas_sq)

        # Scale by target priority and ToO multiplier
        too_bonus = 2.5 if target.is_too else 1.0
        eig = mutual_info * target.priority_weight * too_bonus

        return float(eig)

    def posterior_update(self, target: TransientTarget, snr: float) -> None:
        """Updates target prior variance following Bayesian Gaussian conjugate update."""
        if snr > 1.0:
            sigma_meas_sq = 1.0 / (snr ** 2)
            # Posterior precision = prior precision + measurement precision
            new_precision = (1.0 / target.prior_variance) + (1.0 / sigma_meas_sq)
            target.prior_variance = float(1.0 / new_precision)
            target.times_observed += 1
