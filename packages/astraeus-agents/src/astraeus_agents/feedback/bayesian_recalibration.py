"""
ASTRAEUS Sequential Scientific Feedback - Bayesian Posterior Recalibration Engine (WS7.6-C).
Implements exact Bayesian updating over competing astrophysical hypotheses:
- Rigorous Gaussian likelihoods for point detections incorporating extinction, redshift, and instrument noise.
- Censored survival probability likelihoods for non-detections / upper limits (Phi((m_lim - mu) / sigma)).
- Correlated measurement handling via non-diagonal covariance matrices for shared calibration references.
- Analytically verifiable reference test cases.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    MeasurementType,
)
from astraeus_agents.observatory.facility_registry import WavelengthRegime
from astraeus_agents.observatory.predictive_eig import PredictiveEIGEngine


@dataclass
class RecalibrationUpdateResult:
    """Result of a Bayesian posterior recalibration step."""
    observation_id: str
    target_id: str
    measurement_type: str
    prior_posteriors: Dict[str, float]
    updated_posteriors: Dict[str, float]
    log_likelihoods: Dict[str, float]
    marginal_log_evidence: float
    prior_entropy_nats: float
    posterior_entropy_nats: float
    entropy_delta_nats: float  # Negative means uncertainty reduced!
    dominant_hypothesis: str
    dominant_probability: float
    is_upper_limit: bool = False
    audit_notes: str = ""


class BayesianRecalibrationEngine:
    """
    Core statistical engine executing rigorous Bayesian evidence assimilation.
    Supports optical, NIR, UV, X-ray, and radio regimes across arbitrary hypotheses.
    """

    SYSTEMATIC_CALIBRATION_SIGMA_MAG = 0.03  # Baseline systematic calibration uncertainty
    MODEL_PHYSICS_SIGMA_MAG = 0.05           # Intrinsic theoretical model variance

    @classmethod
    def calculate_entropy(cls, posteriors: Dict[str, float]) -> float:
        """Shannon entropy: H = -sum P ln P in natural units (nats)."""
        ent = 0.0
        for p in posteriors.values():
            if p > 1e-15:
                ent -= p * math.log(p)
        return float(ent)

    @classmethod
    def evaluate_detection_log_likelihood(
        cls,
        measured_mag: float,
        measured_err: float,
        predicted_mag: float,
        extra_sys_sigma: float = 0.0,
    ) -> float:
        """
        Computes Gaussian log-likelihood for a point detection:
            ln L = -0.5 * [ (y - mu)^2 / sigma_tot^2 + ln(2*pi*sigma_tot^2) ]
        """
        sigma_tot = math.sqrt(
            measured_err**2 +
            cls.SYSTEMATIC_CALIBRATION_SIGMA_MAG**2 +
            cls.MODEL_PHYSICS_SIGMA_MAG**2 +
            extra_sys_sigma**2
        )
        resid = measured_mag - predicted_mag
        ln_L = -0.5 * ((resid / sigma_tot) ** 2 + math.log(2.0 * math.pi * (sigma_tot ** 2)))
        return float(ln_L)

    @classmethod
    def evaluate_censored_upper_limit_log_likelihood(
        cls,
        limiting_mag_5sigma: float,
        predicted_mag: float,
        filter_regime: str = "OPTICAL",
        sigma_limit_mag: float = 0.20,
    ) -> float:
        """
        Computes exact censored survival probability log-likelihood for a non-detection.
        
        Physics & Mathematical Formulation:
        In astronomical magnitudes, fainter sources correspond to larger numbers (m_obs > m_lim).
        Under hypothesis H, m_obs ~ N(mu_pred, sigma_tot^2).
        
        The survival probability that a source is fainter than m_lim is:
            P(m_obs > m_lim | H) = 1 - Phi((m_lim - mu_pred) / sigma_tot)
                                 = Phi((mu_pred - m_lim) / sigma_tot)
        
        Numerically stable log probability:
            ln L = norm.logcdf((mu_pred - m_lim) / sigma_tot)
                 = norm.logsf((m_lim - mu_pred) / sigma_tot)
        
        Behavior:
        - Faint source (mu >> m_lim): z >> 0 => ln L -> 0.0 (consistent with non-detection).
        - Bright source (mu << m_lim): z << 0 => ln L << 0 (penalized, as it should have been detected).
        - At threshold (mu == m_lim): z = 0 => ln L = ln(0.5) = -0.6931 nats.
        """
        sigma_tot = math.sqrt(sigma_limit_mag**2 + cls.MODEL_PHYSICS_SIGMA_MAG**2)
        z = (predicted_mag - limiting_mag_5sigma) / max(1e-6, sigma_tot)
        # Use norm.logcdf directly - fully stable across extreme negative tails without underflow
        return float(norm.logcdf(z))

    @classmethod
    def evaluate_censored_flux_upper_limit_log_likelihood(
        cls,
        limiting_flux_njy: float,
        predicted_flux_njy: float,
        flux_uncertainty_njy: float,
    ) -> float:
        """
        Computes exact censored survival probability log-likelihood for a non-detection in linear flux space.
        In flux units, fainter sources correspond to smaller flux values (F_obs < F_lim).
        Under hypothesis H, F_obs ~ N(F_pred, sigma_F^2).
        
        The probability that a source flux is below detection limit F_lim is:
            P(F_obs < F_lim | H) = Phi((F_lim - F_pred) / sigma_F)
            ln L = norm.logcdf((F_lim - F_pred) / sigma_F)
        """
        sigma_tot = max(1e-6, flux_uncertainty_njy)
        z_flux = (limiting_flux_njy - predicted_flux_njy) / sigma_tot
        return float(norm.logcdf(z_flux))

    @classmethod
    def evaluate_correlated_group_log_likelihood(
        cls,
        residuals: np.ndarray,
        cov_matrix: np.ndarray,
        regularization_epsilon: float = 1e-7,
    ) -> float:
        """
        Evaluates exact N-dimensional multivariate Gaussian log-likelihood for an arbitrary
        group of correlated observations sharing calibration references or systematic errors.
        
        Formula:
            ln L = -0.5 * [ r^T Sigma^-1 r + ln|det(Sigma)| + N * ln(2*pi) ]
            
        Numerical stability:
        - Regularizes near-singular matrices: Sigma_reg = Sigma + eps * I
        - Employs Cholesky decomposition: Sigma_reg = L L^T
        - Solves L v = r -> r^T Sigma^-1 r = v^T v
        - Evaluates log determinant: ln|det| = 2 * sum(ln(diag(L)))
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        n = len(residuals)
        
        # Ensure symmetric
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        cov_reg = cov_matrix + regularization_epsilon * np.eye(n)
        
        try:
            L = np.linalg.cholesky(cov_reg)
            v = np.linalg.solve(L, residuals)
            quad_form = float(np.sum(v ** 2))
            log_det = float(2.0 * np.sum(np.log(np.diag(L))))
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_reg)
            quad_form = float(residuals.T @ inv_cov @ residuals)
            sign, log_det = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                log_det = float(n * math.log(regularization_epsilon))

        ln_L = -0.5 * (quad_form + log_det + n * math.log(2.0 * math.pi))
        return float(ln_L)

    @classmethod
    def calculate_exact_two_model_bayes_factor(
        cls,
        mu1: float,
        mu2: float,
        y: float,
        sigma_tot: float,
    ) -> Dict[str, float]:
        """
        Computes analytical conjugate Gaussian Bayes factor and intermediate terms:
            ln BF_12 = [(y - mu2)^2 - (y - mu1)^2] / [2 * sigma_tot^2]
        """
        res1 = y - mu1
        res2 = y - mu2
        sq_res1 = res1 ** 2
        sq_res2 = res2 ** 2
        numerator = sq_res2 - sq_res1
        denominator = 2.0 * (sigma_tot ** 2)
        ln_bf = numerator / denominator
        bf = math.exp(ln_bf) if ln_bf < 700.0 else float("inf")
        if ln_bf > 50.0:
            post_h1 = 1.0 - math.exp(-ln_bf)
            post_h2 = math.exp(-ln_bf)
        elif ln_bf < -50.0:
            post_h1 = math.exp(ln_bf)
            post_h2 = 1.0 - math.exp(ln_bf)
        else:
            post_h1 = 1.0 / (1.0 + math.exp(-ln_bf))
            post_h2 = 1.0 - post_h1

        return {
            "mu1": mu1,
            "mu2": mu2,
            "y": y,
            "sigma_tot": sigma_tot,
            "res1": res1,
            "res2": res2,
            "sq_res1": sq_res1,
            "sq_res2": sq_res2,
            "numerator": numerator,
            "denominator": denominator,
            "ln_bayes_factor": ln_bf,
            "bayes_factor": bf,
            "posterior_h1": post_h1,
            "posterior_h2": post_h2,
        }

    @classmethod
    def evaluate_correlated_pair_log_likelihood(
        cls,
        m1: float,
        err1: float,
        pred1: float,
        m2: float,
        err2: float,
        pred2: float,
        shared_cal_sigma: float = 0.05,
    ) -> float:
        """
        Evaluates bivariate Gaussian log-likelihood for two observations sharing a calibration reference.
        Covariance matrix:
            Sigma = [[s1^2 + s_cal^2, s_cal^2], [s_cal^2, s2^2 + s_cal^2]]
        """
        v1 = err1**2 + cls.MODEL_PHYSICS_SIGMA_MAG**2 + shared_cal_sigma**2
        v2 = err2**2 + cls.MODEL_PHYSICS_SIGMA_MAG**2 + shared_cal_sigma**2
        cov12 = shared_cal_sigma**2

        cov_matrix = np.array([[v1, cov12], [cov12, v2]])
        resids = np.array([m1 - pred1, m2 - pred2])

        det = np.linalg.det(cov_matrix)
        inv = np.linalg.inv(cov_matrix)

        quad = float(resids.T @ inv @ resids)
        ln_L = -0.5 * (quad + math.log(det) + 2.0 * math.log(2.0 * math.pi))
        return float(ln_L)

    @classmethod
    def recalibrate_posteriors(
        cls,
        current_posteriors: Dict[str, float],
        observation: CalibratedObservation,
        effective_wavelength_angstrom: float = 6000.0,
        regime: WavelengthRegime = WavelengthRegime.OPTICAL,
        time_since_trigger_days: float = 2.0,
        base_mag: float = 18.0,
        extra_sys_sigma: float = 0.0,
    ) -> RecalibrationUpdateResult:
        """
        Updates hypothesis posteriors using Bayes' rule with exact likelihood evaluation:
            P(H_i | D, y) = P(y | H_i, D) * P(H_i | D) / sum P(y | H_j) P(H_j)
            
        Supports:
        - Point detections (Gaussian likelihood)
        - Censored non-detection upper limits (Survival probability)
        - Unmodeled transient hypothesis (Epistemic guardrail against false discovery)
        """
        prior_h = cls.calculate_entropy(current_posteriors)
        h_keys = list(current_posteriors.keys())

        # 1. Evaluate predicted magnitude per hypothesis
        pred_mags: Dict[str, float] = {}
        for h in h_keys:
            if h.upper() in ("UNMODELED_TRANSIENT", "OUT_OF_MODEL", "UNMODELED"):
                pred_mags[h] = 20.0  # nominal center
            else:
                pred_mags[h] = PredictiveEIGEngine.predict_hypothesis_magnitude(
                    hypothesis_name=h,
                    regime=regime,
                    effective_wavelength_angstrom=effective_wavelength_angstrom,
                    time_since_trigger_days=time_since_trigger_days,
                    base_mag=base_mag,
                )

        # 2. Compute log likelihoods ln L(y | H_i)
        ln_likelihoods: Dict[str, float] = {}
        is_censored = observation.is_upper_limit()

        for h in h_keys:
            if h.upper() in ("UNMODELED_TRANSIENT", "OUT_OF_MODEL", "UNMODELED"):
                # Broad, uninformative likelihood absorbing unmodeled anomalies
                if is_censored:
                    lim_mag = observation.limiting_mag_5sigma or 21.0
                    # Over broad [15.0, 25.0] prior range:
                    p_surv = max(0.01, min(0.99, (25.0 - lim_mag) / 10.0))
                    ln_likelihoods[h] = float(math.log(p_surv))
                else:
                    # Broad uniform density over 10 mag domain: ln(1/10) = -2.3026
                    ln_likelihoods[h] = float(math.log(1.0 / 10.0))
                continue

            pred_m = pred_mags[h]
            if is_censored:
                # Censored upper limit likelihood
                lim_mag = observation.limiting_mag_5sigma or 21.0
                ln_l = cls.evaluate_censored_upper_limit_log_likelihood(
                    limiting_mag_5sigma=lim_mag,
                    predicted_mag=pred_m,
                    filter_regime=regime.value,
                )
            else:
                # Point detection likelihood
                meas_m = observation.apparent_mag if observation.apparent_mag is not None else 19.0
                meas_err = observation.apparent_mag_err if observation.apparent_mag_err is not None else 0.05
                ln_l = cls.evaluate_detection_log_likelihood(
                    measured_mag=meas_m,
                    measured_err=meas_err,
                    predicted_mag=pred_m,
                    extra_sys_sigma=extra_sys_sigma,
                )
            ln_likelihoods[h] = ln_l

        # 3. Bayes update with log-sum-exp numerical stabilization
        max_ln_l = max(ln_likelihoods.values())
        unnorm_posts = {}
        for h in h_keys:
            prior_p = current_posteriors[h]
            # Stable log posterior numerator
            if prior_p > 1e-15:
                unnorm_posts[h] = prior_p * math.exp(ln_likelihoods[h] - max_ln_l)
            else:
                unnorm_posts[h] = 0.0

        total_evidence = sum(unnorm_posts.values())
        if total_evidence <= 1e-30:
            # Fallback to prior if completely degenerate
            updated_posts = dict(current_posteriors)
            marginal_log_ev = -999.0
        else:
            updated_posts = {h: float(unnorm_posts[h] / total_evidence) for h in h_keys}
            marginal_log_ev = max_ln_l + math.log(total_evidence)

        post_h = cls.calculate_entropy(updated_posts)
        delta_h = post_h - prior_h

        # Dominant hypothesis
        dom_h = max(updated_posts.items(), key=lambda x: x[1])

        return RecalibrationUpdateResult(
            observation_id=observation.observation_id,
            target_id=observation.target_id,
            measurement_type=observation.measurement_type.value,
            prior_posteriors={k: round(v, 4) for k, v in current_posteriors.items()},
            updated_posteriors={k: round(v, 4) for k, v in updated_posts.items()},
            log_likelihoods={k: round(v, 3) for k, v in ln_likelihoods.items()},
            marginal_log_evidence=round(marginal_log_ev, 3),
            prior_entropy_nats=round(prior_h, 4),
            posterior_entropy_nats=round(post_h, 4),
            entropy_delta_nats=round(delta_h, 4),
            dominant_hypothesis=dom_h[0],
            dominant_probability=round(dom_h[1], 4),
            is_upper_limit=is_censored,
            audit_notes=f"Recalibration completed via {'Censored Survival' if is_censored else 'Gaussian'} likelihood.",
        )
