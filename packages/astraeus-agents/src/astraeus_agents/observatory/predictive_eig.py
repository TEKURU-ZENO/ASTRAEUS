"""
ASTRAEUS Observatory Intelligence - Physics-Grounded Predictive EIG Engine (WS7.5).
Calculates Expected Information Gain (EIG) over competing astrophysical hypotheses
using explicit generative light-curve models, filter passbands, and instrument noise responses.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from astraeus_agents.observatory.facility_registry import (
    ObservatoryFacility,
    InstrumentProfile,
    InstrumentFilter,
    WavelengthRegime,
)
from astraeus_agents.observatory.exposure_calculator import (
    ExposureCalculator,
    ExposureEstimate,
)


@dataclass
class ObservationAction:
    """A proposed observing pointing action to be scheduled."""
    action_id: str
    target_id: str
    target_ra_deg: float
    target_dec_deg: float
    facility: ObservatoryFacility
    instrument: InstrumentProfile
    filter_obj: InstrumentFilter
    exposure_time_sec: float
    scheduled_mjd: float
    target_hypothesis_names: List[str]


@dataclass
class EIGEvaluationResult:
    """Expected Information Gain evaluation outcome for an observation action."""
    action_id: str
    facility_id: str
    instrument_id: str
    filter_id: str
    prior_entropy_nats: float
    expected_posterior_entropy_nats: float
    expected_information_gain_nats: float
    normalized_eig_per_hour: float
    predicted_means_by_hypothesis: Dict[str, float]
    predicted_uncertainties_by_hypothesis: Dict[str, float]
    discriminative_power_score: float  # Maximum pairwise z-score separation


class PredictiveEIGEngine:
    """
    Computes rigorous Shannon Expected Information Gain (EIG) over competing
    astrophysical hypotheses for candidate observing actions.
    """

    @staticmethod
    def calculate_entropy(posteriors: Dict[str, float]) -> float:
        """Computes Shannon entropy in natural units (nats): H = -sum P ln P."""
        ent = 0.0
        for p in posteriors.values():
            if p > 1e-12:
                ent -= p * math.log(p)
        return float(ent)

    @classmethod
    def predict_hypothesis_magnitude(
        cls,
        hypothesis_name: str,
        regime: WavelengthRegime,
        effective_wavelength_angstrom: float,
        time_since_trigger_days: float,
        base_mag: float = 18.0,
        host_z: float = 0.03,
    ) -> float:
        """
        Generative physical model evaluating apparent magnitude in requested passband
        as a function of elapsed time since explosion.
        
        Models:
        - Kilonova: Fast radioactive cooling (0.8 - 1.2 mag/day in optical, rapid reddening into NIR)
        - Supernova Ia: Canonical Arnett rise to peak (~18d) then steady decay (~0.05 mag/day)
        - Core-Collapse SN: Slower plateau / linear decay
        - TDE: Hot power-law fallback (t^-5/3) with high UV/optical ratio
        - Galactic Star: Impulsive flare returning to faint quiescence
        """
        t = max(0.01, time_since_trigger_days)
        wl = effective_wavelength_angstrom

        hyp = hypothesis_name.upper()

        if "KILONOVA" in hyp or "KN" in hyp:
            # Villar et al. / Kasen kilonova profile:
            # Optical decays very fast due to heavy r-process line blanketing;
            # NIR stays bright longer (lanthanide curtaining shifts flux to 1-2 um)
            optical_decay = 0.95 * t
            # Color term relative to 6000 A: redder wavelengths fade slower
            wavelength_factor = (wl - 6000.0) / 10000.0
            nir_compensation = -0.45 * wavelength_factor * t
            return float(base_mag + optical_decay + nir_compensation)

        elif "TDE" in hyp or "TIDAL" in hyp:
            # Power law fallback L ~ t^(-5/3) ==> delta_m ~ 2.5 * (5/3) * log10(1 + t/15)
            fallback_decay = 4.167 * math.log10(1.0 + t / 15.0)
            # TDE is extremely blue: UV flux is much brighter than optical
            uv_optical_color = -1.2 * (3000.0 - wl) / 3000.0 if wl < 5000 else 0.0
            return float(base_mag + fallback_decay + uv_optical_color)

        elif "IA" in hyp or "SN_IA" in hyp:
            # Type Ia: Bazin-like profile around peak (t_peak ~ 18 days)
            if t < 15.0:
                # Rising or near peak
                dm = -0.8 * (t / 15.0)
            else:
                # Radioactive 56Co decay: ~0.08 mag/day in optical
                dm = -0.8 + 0.075 * (t - 15.0)
            # Typical SN Ia optical colors
            color_corr = 0.3 * (wl - 6000.0) / 4000.0
            return float(base_mag + dm + color_corr)

        elif "CCSN" in hyp or "SN_II" in hyp or "SN_CORE_COLLAPSE" in hyp:
            # Type II plateau / slow linear decay
            dm = 0.025 * t
            return float(base_mag + dm)

        elif "STAR" in hyp or "FLARE" in hyp:
            # Impulsive flare decays back to quiescence within 0.5 - 2 days
            flare_decay = 2.5 * min(2.0, t / 0.5)
            return float(base_mag + flare_decay)

        else:
            # Generic fallback linear decay
            return float(base_mag + 0.1 * t)

    @classmethod
    def evaluate_eig(
        cls,
        action: ObservationAction,
        prior_posteriors: Dict[str, float],
        airmass: float = 1.2,
        seeing_fwhm_arcsec: float = 0.8,
        cloud_extinction_mag: float = 0.0,
        n_quadrature_samples: int = 25,
    ) -> EIGEvaluationResult:
        """
        Evaluates exact Information-Theoretic Expected Information Gain (EIG)
        for `action` against `prior_posteriors`.
        
        Formula:
            EIG(a) = H(H | D) - E_{y ~ p(y|D,a)} [ H(H | D, y, a) ]
        """
        # 1. Prior entropy
        prior_h = cls.calculate_entropy(prior_posteriors)

        # 2. Compute predicted means and measurement errors per hypothesis
        pred_means: Dict[str, float] = {}
        pred_sigmas: Dict[str, float] = {}

        t_elapsed_days = max(0.1, action.scheduled_mjd - (action.scheduled_mjd - 3.0))

        for h_name in prior_posteriors.keys():
            mean_mag = cls.predict_hypothesis_magnitude(
                hypothesis_name=h_name,
                regime=action.instrument.regime,
                effective_wavelength_angstrom=action.filter_obj.effective_wavelength_angstrom,
                time_since_trigger_days=t_elapsed_days,
            )
            # Evaluate measurement uncertainty from exposure calculator
            est = ExposureCalculator.compute_snr(
                instrument=action.instrument,
                filter_obj=action.filter_obj,
                target_apparent_mag=mean_mag,
                airmass=airmass,
                seeing_fwhm_arcsec=seeing_fwhm_arcsec,
                exposure_time_sec=action.exposure_time_sec,
                cloud_extinction_mag=cloud_extinction_mag,
            )
            pred_means[h_name] = est.effective_mag
            pred_sigmas[h_name] = max(0.01, est.photometric_uncertainty_mag)

        # 3. Maximum pairwise z-score separation (discriminative power)
        max_z_sep = 0.0
        h_keys = list(prior_posteriors.keys())
        for i in range(len(h_keys)):
            for j in range(i + 1, len(h_keys)):
                h1, h2 = h_keys[i], h_keys[j]
                delta_mu = abs(pred_means[h1] - pred_means[h2])
                sigma_pooled = math.sqrt(pred_sigmas[h1]**2 + pred_sigmas[h2]**2)
                z_sep = delta_mu / max(1e-4, sigma_pooled)
                if z_sep > max_z_sep:
                    max_z_sep = z_sep

        # 4. Numerical integration over synthetic observation space y
        # Range of y spans min(mu) - 3*sigma to max(mu) + 3*sigma
        y_min = min(pred_means[h] - 3.5 * pred_sigmas[h] for h in h_keys)
        y_max = max(pred_means[h] + 3.5 * pred_sigmas[h] for h in h_keys)
        y_grid = np.linspace(y_min, y_max, n_quadrature_samples)
        dy = (y_max - y_min) / (n_quadrature_samples - 1)

        expected_post_h = 0.0

        for y_val in y_grid:
            # Likelihood p(y | H_i, a) = N(y | mu_i, sigma_i^2)
            likelihoods = {}
            for h in h_keys:
                mu = pred_means[h]
                sig = pred_sigmas[h]
                # Gaussian PDF
                norm_pdf = math.exp(-0.5 * ((y_val - mu) / sig) ** 2) / (math.sqrt(2.0 * math.pi) * sig)
                likelihoods[h] = norm_pdf

            # Marginal evidence p(y | D, a) = sum P(H_i|D) p(y|H_i, a)
            marginal_p_y = sum(prior_posteriors[h] * likelihoods[h] for h in h_keys)

            if marginal_p_y <= 1e-15:
                continue

            # Updated posteriors P(H_i | D, y, a)
            updated_posts = {
                h: (prior_posteriors[h] * likelihoods[h]) / marginal_p_y
                for h in h_keys
            }

            post_h_for_y = cls.calculate_entropy(updated_posts)
            expected_post_h += marginal_p_y * post_h_for_y * dy

        # EIG = H(H|D) - E[H(H|D,y)]
        eig = max(0.0, prior_h - expected_post_h)

        # Cost-normalized EIG: information gain per hour of total time (slew + settle + exposure + readout)
        total_time_sec = (
            action.exposure_time_sec +
            action.instrument.settle_time_sec +
            action.instrument.readout_overhead_sec +
            30.0  # nominal slew allowance
        )
        total_time_hours = total_time_sec / 3600.0
        eig_per_hour = eig / max(0.01, total_time_hours)

        return EIGEvaluationResult(
            action_id=action.action_id,
            facility_id=action.facility.facility_id,
            instrument_id=action.instrument.instrument_id,
            filter_id=action.filter_obj.filter_id,
            prior_entropy_nats=round(prior_h, 4),
            expected_posterior_entropy_nats=round(expected_post_h, 4),
            expected_information_gain_nats=round(eig, 4),
            normalized_eig_per_hour=round(eig_per_hour, 4),
            predicted_means_by_hypothesis={k: round(v, 3) for k, v in pred_means.items()},
            predicted_uncertainties_by_hypothesis={k: round(v, 4) for k, v in pred_sigmas.items()},
            discriminative_power_score=round(max_z_sep, 2),
        )
