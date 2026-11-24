"""
ASTRAEUS Computational Experiment Engine (WS8.5).
Provides forward simulation, inverse parameter recovery, counterfactual testing,
and Bayesian posterior predictive checks with strict data provenance boundaries.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import math
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


class DataProvenanceTag(str, Enum):
    """Rigorous scientific data provenance demarcation."""
    SYNTHETIC_SIMULATOR = "SYNTHETIC_SIMULATOR"
    SIMULATED_TESTBED = "SIMULATED_TESTBED"
    HISTORICAL_REAL_ARCHIVAL = "HISTORICAL_REAL_ARCHIVAL"
    BLIND_HELD_OUT_EVALUATION = "BLIND_HELD_OUT_EVALUATION"


class SimulatedObservationPoint(BaseModel):
    """Individual data point generated or ingested with explicit provenance."""
    epoch_days: float
    filter_id: str
    effective_wavelength_angstrom: float
    simulated_mag: float
    uncertainty_mag: float
    provenance: DataProvenanceTag
    true_parameter_ground_truth: Optional[Dict[str, float]] = None


class ParameterRecoveryResult(BaseModel):
    """Output of inverse parameter estimation fitting observational data."""
    parameter_name: str
    recovered_mean: float
    recovered_std: float
    credible_interval_95: Tuple[float, float]
    true_value: Optional[float] = None
    absolute_error: Optional[float] = None
    z_score_recovery_error: Optional[float] = None
    is_contained_in_95_ci: bool = True


class PosteriorPredictiveCheckResult(BaseModel):
    """Formal Bayesian Posterior Predictive Check (PPC) validation."""
    model_name: str
    num_replications: int
    observed_discrepancy_stat: float
    replicated_discrepancy_stats: List[float]
    bayesian_p_value: float = Field(..., ge=0.0, le=1.0)
    ppc_passed: bool
    diagnostic_comment: str


class CounterfactualExperimentResult(BaseModel):
    """Comparative evaluation between baseline hypothesis and counterfactual parameter shift."""
    experiment_id: str
    baseline_parameters: Dict[str, float]
    counterfactual_parameters: Dict[str, float]
    delta_parameters: Dict[str, float]
    baseline_predictions: List[SimulatedObservationPoint]
    counterfactual_predictions: List[SimulatedObservationPoint]
    mean_absolute_difference_mag: float
    critical_divergence_epoch_days: float


class ComputationalExperimentEngine:
    """
    Executes forward simulations, parameter inversions, and counterfactuals,
    while enforcing Gate 3 zero-touch quarantine and strict data provenance.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.RandomState(random_seed)

    def verify_quarantine_safety(self, ra_deg: float, dec_deg: float) -> None:
        """Enforces Gate 3 Zero-Touch Quarantine: exactly 1,808 pixels remain sealed."""
        QuarantineFirewall._initialize()
        pix = QuarantineFirewall.compute_healpix_nside32(ra_deg, dec_deg)
        if QuarantineFirewall.is_quarantined_pixel(pix):
            raise QuarantineViolationError(
                f"Quarantine Breach Prevented: target at (ra={ra_deg:.4f}, dec={dec_deg:.4f}, "
                f"pixel={pix}) falls inside Gate 3 sealed test partition!"
            )

    def forward_simulate_kilonova(
        self,
        ejecta_mass_msun: float,
        expansion_velocity_c: float,
        lanthanide_mass_fraction: float,
        epochs_days: List[float],
        filter_wavelengths: Dict[str, float],
        provenance: DataProvenanceTag = DataProvenanceTag.SYNTHETIC_SIMULATOR,
        inject_noise: bool = False,
        photometric_noise_mag: float = 0.05,
    ) -> List[SimulatedObservationPoint]:
        """
        Physics-based forward simulator for two-component r-process kilonovae.
        Luminosity based on radioactive heating: q(t) = 4e10 * (t/1d)^-1.3 erg/s/g.
        Diffusion time: t_diff ~ sqrt(2 * kappa * M_ej / (4 * pi * v * c)).
        """
        sim_points: List[SimulatedObservationPoint] = []
        M_ej_grams = ejecta_mass_msun * 1.98847e33
        v_cm_s = expansion_velocity_c * 2.99792458e10
        c_cm_s = 2.99792458e10

        # Opacity: lanthanide-poor vs lanthanide-rich
        # kappa ranges from 0.5 cm^2/g (blue) to 10.0 cm^2/g (red)
        kappa = 0.5 + 9.5 * min(1.0, max(0.0, lanthanide_mass_fraction))
        t_diff_sec = math.sqrt((2.0 * kappa * M_ej_grams) / (4.0 * math.pi * v_cm_s * c_cm_s))
        t_diff_days = t_diff_sec / 86400.0

        for t in epochs_days:
            t_safe = max(0.05, t)
            # Radioactive specific heating rate (erg / s / g)
            q_dot = 4.0e10 * (t_safe ** -1.30)
            # Bolometric luminosity with diffusion leak factor
            leak_factor = 1.0 - math.exp(-((t_diff_days / t_safe) ** 2))
            L_bol = M_ej_grams * q_dot * leak_factor

            # Approximate effective temperature from Stefan-Boltzmann: R ~ v * t
            R_phot = max(1e13, v_cm_s * t_safe * 86400.0)
            T_eff = max(1500.0, ((L_bol) / (4.0 * math.pi * (R_phot**2) * 5.670374e-5)) ** 0.25)

            # Absolute bolometric magnitude
            M_bol = 4.74 - 2.5 * math.log10(max(1e30, L_bol) / 3.828e33)

            for flt_name, wl_ang in filter_wavelengths.items():
                # Color correction relative to bolometric using Planck approx
                wl_cm = wl_ang * 1e-8
                h_planck = 6.626e-27
                k_b = 1.38e-16
                x = (h_planck * c_cm_s) / (wl_cm * k_b * T_eff)
                planck_factor = (x**4) / (math.exp(min(50.0, x)) - 1.0)
                bolometric_corr = -2.5 * math.log10(max(1e-6, planck_factor))

                # Distance modulus for typical nearby event (d = 40 Mpc, mu = 33.0)
                mu_dist = 33.0
                app_mag = M_bol + bolometric_corr + mu_dist

                if inject_noise:
                    noise = self.rng.normal(0.0, photometric_noise_mag)
                    app_mag += noise

                sim_points.append(SimulatedObservationPoint(
                    epoch_days=t,
                    filter_id=flt_name,
                    effective_wavelength_angstrom=wl_ang,
                    simulated_mag=round(float(app_mag), 3),
                    uncertainty_mag=photometric_noise_mag,
                    provenance=provenance,
                    true_parameter_ground_truth={
                        "ejecta_mass_msun": ejecta_mass_msun,
                        "expansion_velocity_c": expansion_velocity_c,
                        "lanthanide_mass_fraction": lanthanide_mass_fraction,
                    },
                ))

        return sim_points

    def perform_inverse_parameter_recovery(
        self,
        observations: List[SimulatedObservationPoint],
        grid_resolution: int = 15,
    ) -> Dict[str, ParameterRecoveryResult]:
        """
        Executes inverse grid search / maximum likelihood parameter estimation,
        computing posterior mean, standard deviation, and 95% confidence intervals.
        """
        if not observations:
            return {}

        # Parameter search bounds
        m_grid = np.linspace(0.01, 0.08, grid_resolution)
        v_grid = np.linspace(0.08, 0.30, grid_resolution)
        x_grid = np.linspace(1e-4, 1e-1, grid_resolution)

        filter_wls = {p.filter_id: p.effective_wavelength_angstrom for p in observations}
        epochs = sorted(list(set(p.epoch_days for p in observations)))

        best_chi2 = float("inf")
        best_params = (0.04, 0.15, 0.01)

        # Weighted likelihood evaluation
        chi2_values: List[Tuple[float, float, float, float]] = []

        for m_val in m_grid:
            for v_val in v_grid:
                # Approximate forward call
                sim = self.forward_simulate_kilonova(
                    ejecta_mass_msun=float(m_val),
                    expansion_velocity_c=float(v_val),
                    lanthanide_mass_fraction=0.01,
                    epochs_days=epochs,
                    filter_wavelengths=filter_wls,
                    provenance=DataProvenanceTag.SIMULATED_TESTBED,
                    inject_noise=False,
                )
                pred_map = {(p.epoch_days, p.filter_id): p.simulated_mag for p in sim}

                chi2 = 0.0
                for obs in observations:
                    key = (obs.epoch_days, obs.filter_id)
                    if key in pred_map:
                        res = obs.simulated_mag - pred_map[key]
                        chi2 += (res / max(0.01, obs.uncertainty_mag)) ** 2

                chi2_values.append((float(m_val), float(v_val), 0.01, chi2))
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (float(m_val), float(v_val), 0.01)

        # Convert chi2 to posterior weights
        min_c2 = min(c[3] for c in chi2_values)
        weights = [math.exp(-0.5 * (c[3] - min_c2)) for c in chi2_values]
        sum_w = sum(weights)
        if sum_w <= 0:
            sum_w = 1.0

        # Marginal moments
        m_arr = np.array([c[0] for c in chi2_values])
        v_arr = np.array([c[1] for c in chi2_values])

        m_mean = float(np.average(m_arr, weights=weights))
        m_raw_std = float(np.sqrt(np.average((m_arr - m_mean)**2, weights=weights)))
        v_mean = float(np.average(v_arr, weights=weights))
        v_raw_std = float(np.sqrt(np.average((v_arr - v_mean)**2, weights=weights)))

        # Grid discretization floor
        m_step = float(m_grid[1] - m_grid[0])
        v_step = float(v_grid[1] - v_grid[0])
        m_std = max(m_step * 0.75, m_raw_std)
        v_std = max(v_step * 0.75, v_raw_std)

        true_gt = observations[0].true_parameter_ground_truth or {}
        true_m = true_gt.get("ejecta_mass_msun")
        true_v = true_gt.get("expansion_velocity_c")

        res_m = ParameterRecoveryResult(
            parameter_name="ejecta_mass_msun",
            recovered_mean=round(m_mean, 4),
            recovered_std=round(m_std, 4),
            credible_interval_95=(round(m_mean - 1.96 * m_std, 4), round(m_mean + 1.96 * m_std, 4)),
            true_value=true_m,
            absolute_error=round(abs(m_mean - true_m), 4) if true_m is not None else None,
            z_score_recovery_error=round(abs(m_mean - true_m) / max(1e-4, m_std), 2) if true_m is not None else None,
            is_contained_in_95_ci=(m_mean - 1.96 * m_std <= true_m <= m_mean + 1.96 * m_std) if true_m is not None else True,
        )

        res_v = ParameterRecoveryResult(
            parameter_name="expansion_velocity_c",
            recovered_mean=round(v_mean, 4),
            recovered_std=round(v_std, 4),
            credible_interval_95=(round(v_mean - 1.96 * v_std, 4), round(v_mean + 1.96 * v_std, 4)),
            true_value=true_v,
            absolute_error=round(abs(v_mean - true_v), 4) if true_v is not None else None,
            z_score_recovery_error=round(abs(v_mean - true_v) / max(1e-4, v_std), 2) if true_v is not None else None,
            is_contained_in_95_ci=(v_mean - 1.96 * v_std <= true_v <= v_mean + 1.96 * v_std) if true_v is not None else True,
        )

        return {"ejecta_mass_msun": res_m, "expansion_velocity_c": res_v}

    def generate_counterfactual(
        self,
        experiment_id: str,
        baseline_params: Dict[str, float],
        counterfactual_params: Dict[str, float],
        epochs_days: List[float],
        filter_wavelengths: Dict[str, float],
    ) -> CounterfactualExperimentResult:
        """
        Evaluates exact physical consequences of a counterfactual intervention.
        """
        base_sim = self.forward_simulate_kilonova(
            ejecta_mass_msun=baseline_params.get("ejecta_mass_msun", 0.04),
            expansion_velocity_c=baseline_params.get("expansion_velocity_c", 0.15),
            lanthanide_mass_fraction=baseline_params.get("lanthanide_mass_fraction", 0.01),
            epochs_days=epochs_days,
            filter_wavelengths=filter_wavelengths,
            provenance=DataProvenanceTag.SYNTHETIC_SIMULATOR,
        )

        cf_sim = self.forward_simulate_kilonova(
            ejecta_mass_msun=counterfactual_params.get("ejecta_mass_msun", 0.04),
            expansion_velocity_c=counterfactual_params.get("expansion_velocity_c", 0.15),
            lanthanide_mass_fraction=counterfactual_params.get("lanthanide_mass_fraction", 0.01),
            epochs_days=epochs_days,
            filter_wavelengths=filter_wavelengths,
            provenance=DataProvenanceTag.SYNTHETIC_SIMULATOR,
        )

        diffs = [abs(b.simulated_mag - c.simulated_mag) for b, c in zip(base_sim, cf_sim)]
        mean_diff = float(np.mean(diffs))

        # Identify epoch of maximum divergence
        max_idx = int(np.argmax(diffs))
        divergence_epoch = base_sim[max_idx].epoch_days

        delta_params = {
            k: counterfactual_params.get(k, 0.0) - baseline_params.get(k, 0.0)
            for k in set(baseline_params.keys()) | set(counterfactual_params.keys())
        }

        return CounterfactualExperimentResult(
            experiment_id=experiment_id,
            baseline_parameters=baseline_params,
            counterfactual_parameters=counterfactual_params,
            delta_parameters=delta_params,
            baseline_predictions=base_sim,
            counterfactual_predictions=cf_sim,
            mean_absolute_difference_mag=round(mean_diff, 3),
            critical_divergence_epoch_days=divergence_epoch,
        )

    def evaluate_posterior_predictive_check(
        self,
        model_name: str,
        observed_data: List[SimulatedObservationPoint],
        predicted_means: List[float],
        predicted_sigmas: List[float],
        n_replications: int = 200,
    ) -> PosteriorPredictiveCheckResult:
        """
        Evaluates Bayesian Posterior Predictive Check (PPC):
        Computes discrepancy measure T(y) = sum (y_i - mu_i)^2 / sigma_i^2.
        Bayesian p-value = fraction of replicated data sets where T(y_rep) >= T(y_obs).
        """
        y_obs = np.array([p.simulated_mag for p in observed_data])
        sigmas = np.array(predicted_sigmas)
        mus = np.array(predicted_means)

        # Observed discrepancy statistic
        t_obs = float(np.sum(((y_obs - mus) / np.maximum(1e-4, sigmas)) ** 2))

        # Sample replicated data sets from posterior predictive distribution
        t_reps: List[float] = []
        for _ in range(n_replications):
            y_rep = self.rng.normal(mus, sigmas)
            t_rep = float(np.sum(((y_rep - mus) / np.maximum(1e-4, sigmas)) ** 2))
            t_reps.append(t_rep)

        # Bayesian p-value
        p_val = float(np.mean(np.array(t_reps) >= t_obs))

        # Model is rejected if p_val < 0.05 (severe underfitting) or p_val > 0.95 (severe overfitting)
        passed = 0.05 <= p_val <= 0.95

        comment = (
            f"Posterior predictive check passed (p_B={p_val:.3f} within [0.05, 0.95]). Model is well-calibrated."
            if passed
            else f"PPC Failed (p_B={p_val:.3f}): model exhibits systematic lack-of-fit or over-confidence."
        )

        return PosteriorPredictiveCheckResult(
            model_name=model_name,
            num_replications=n_replications,
            observed_discrepancy_stat=round(t_obs, 2),
            replicated_discrepancy_stats=[round(x, 2) for x in t_reps[:10]],
            bayesian_p_value=round(p_val, 4),
            ppc_passed=passed,
            diagnostic_comment=comment,
        )
