"""
ASTRAEUS Sequential Scientific Feedback - Posterior Predictive Checking & Model Criticism (WS7.6-E).
Calculates normalized residuals, reduced chi-square goodness-of-fit, posterior predictive p-values,
detects model misspecification, and assigns 4-tier scientific evaluation statuses.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import chi2
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from astraeus_agents.feedback.observation_models import CalibratedObservation
from astraeus_agents.observatory.predictive_eig import PredictiveEIGEngine
from astraeus_agents.observatory.facility_registry import WavelengthRegime
from astraeus_agents.reground.models import HypothesisEvaluationStatus


@dataclass
class HypothesisCriticismResult:
    """Statistical model evaluation and diagnostics for a specific hypothesis."""
    hypothesis_name: str
    current_posterior: float
    degrees_of_freedom: int
    reduced_chi2: float
    goodness_of_fit_p_value: float          # Classical upper-tail chi2 survival p-value (chi2.sf)
    bayesian_ppc_p_value: float             # Genuine Monte Carlo replicated posterior predictive p-value
    observed_discrepancy_stat: float        # Observed T(y_obs) = chi2
    replicated_discrepancy_mean: float      # Mean T(y_rep) across posterior replicates
    replicated_discrepancy_std: float       # Std of T(y_rep)
    mean_normalized_residual: float
    max_normalized_residual: float
    is_statistically_consistent: bool
    evaluation_status: HypothesisEvaluationStatus
    criticism_summary: str


@dataclass
class PosteriorPredictiveAssessment:
    """Comprehensive multi-model posterior predictive criticism report."""
    target_id: str
    num_observations_evaluated: int
    hypotheses_critiques: Dict[str, HypothesisCriticismResult] = field(default_factory=dict)
    all_models_misspecified: bool = False
    preferred_hypothesis: Optional[str] = None
    unresolved_alternatives_present: bool = False
    model_discrepancy_detected: bool = False
    audit_verdict: str = ""


class PosteriorPredictiveChecker:
    """
    Executes posterior predictive checks (PPC) and model criticism
    across accumulated observations using both classical goodness-of-fit
    and genuine Monte Carlo replicated datasets.
    """

    MODEL_VARIANCE_MAG = 0.05

    @classmethod
    def simulate_posterior_predictive_replicates(
        cls,
        predicted_magnitudes: List[float],
        total_uncertainties: List[float],
        num_replicates: int = 500,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Draws replicated datasets y_rep^(s) ~ N(mu_pred, sigma_tot^2) under the candidate model.
        Returns simulated discrepancy statistics T(y_rep^(s)) = sum (y_rep - mu)^2 / sigma_tot^2.
        """
        rng = np.random.default_rng(random_seed)
        mu = np.array(predicted_magnitudes)
        sig = np.array(total_uncertainties)
        n_obs = len(mu)

        # Draw (num_replicates, n_obs)
        y_rep = rng.normal(loc=mu, scale=sig, size=(num_replicates, n_obs))
        # Standardized squared residuals for each replicate
        residuals_rep = (y_rep - mu) / sig
        discrepancies_rep = np.sum(residuals_rep ** 2, axis=1)
        return discrepancies_rep

    @classmethod
    def evaluate_model_fit(
        cls,
        target_id: str,
        hypothesis_posteriors: Dict[str, float],
        accumulated_observations: List[CalibratedObservation],
        trigger_mjd: float = 60000.0,
        fitted_parameter_count: int = 0,
        base_mag: float = 18.0,
    ) -> PosteriorPredictiveAssessment:
        """
        Computes classical goodness-of-fit and genuine Monte Carlo Bayesian PPC
        for all candidate hypotheses against active point detections.
        """
        detections = [o for o in accumulated_observations if o.is_detection()]
        n_obs = len(detections)

        if n_obs == 0:
            return PosteriorPredictiveAssessment(
                target_id=target_id,
                num_observations_evaluated=0,
                audit_verdict="Insufficient point detections to perform posterior predictive check.",
            )

        critiques: Dict[str, HypothesisCriticismResult] = {}
        all_misspecified = True
        dof = max(1, n_obs - fitted_parameter_count)

        for h_name, post_prob in hypothesis_posteriors.items():
            residuals = []
            preds = []
            tot_errs = []
            
            for obs in detections:
                t_days = max(0.01, obs.epoch_mjd - trigger_mjd)
                meas_m = obs.apparent_mag or 19.0
                meas_err = obs.apparent_mag_err or 0.05
                tot_err = math.sqrt(meas_err**2 + cls.MODEL_VARIANCE_MAG**2)

                # Determine regime and effective wavelength from filter_id
                regime = WavelengthRegime.OPTICAL
                wl = 6000.0
                fid = getattr(obs, "filter_id", "") or ""
                if fid:
                    fb = fid.lower()
                    if any(uv_key in fb for uv_key in ("uvw", "uvm", "fuv", "nuv", "uv")):
                        regime = WavelengthRegime.UV
                        wl = 2600.0
                    elif any(nir_key in fb for nir_key in ("nir", "spec", "j", "h", "k", "z", "y")):
                        regime = WavelengthRegime.NIR
                        wl = 12500.0
                    elif "xray" in fb or "xrt" in fb:
                        regime = WavelengthRegime.XRAY
                        wl = 5.0
                    elif "radio" in fb:
                        regime = WavelengthRegime.RADIO
                        wl = 1.0e8

                if h_name.upper() in ("UNMODELED_TRANSIENT", "OUT_OF_MODEL", "UNMODELED"):
                    pred_m = 20.0  # unmodeled flat centroid
                    tot_err = 3.0   # wide uncertainty
                else:
                    pred_m = PredictiveEIGEngine.predict_hypothesis_magnitude(
                        hypothesis_name=h_name,
                        regime=regime,
                        effective_wavelength_angstrom=wl,
                        time_since_trigger_days=t_days,
                        base_mag=base_mag,
                    )

                r = (meas_m - pred_m) / max(1e-4, tot_err)
                residuals.append(r)
                preds.append(pred_m)
                tot_errs.append(tot_err)

            res_arr = np.array(residuals)
            chi2_stat = float(np.sum(res_arr ** 2))
            red_chi2 = float(chi2_stat / dof)
            # Classical upper-tail survival function: p = P(chi2 >= chi2_obs)
            p_classical = float(chi2.sf(chi2_stat, dof))

            # Monte Carlo Bayesian PPC simulation
            discrepancies_rep = cls.simulate_posterior_predictive_replicates(
                predicted_magnitudes=preds,
                total_uncertainties=tot_errs,
                num_replicates=500,
            )
            # Bayesian PPC p-value: fraction of replicates with discrepancy >= observed
            p_ppc = float(np.mean(discrepancies_rep >= chi2_stat))
            rep_mean = float(np.mean(discrepancies_rep))
            rep_std = float(np.std(discrepancies_rep))

            mean_r = float(np.mean(np.abs(res_arr)))
            max_r = float(np.max(np.abs(res_arr)))

            # Consistency criteria: p_classical >= 0.01, p_ppc >= 0.01, red_chi2 < 3.5
            is_consistent = (p_classical >= 0.01) and (p_ppc >= 0.01) and (red_chi2 < 3.5)
            if is_consistent:
                all_misspecified = False

            # Assign 4-tier taxonomy
            if not is_consistent:
                status = HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS
                summary = f"Rejected under assumptions: red_chi2={red_chi2:.2f}, p_sf={p_classical:.4f}, p_PPC={p_ppc:.4f} (max res {max_r:.1f} sigma)."
            elif post_prob >= 0.60:
                status = HypothesisEvaluationStatus.STRONGER_EVIDENCE
                summary = f"Supported by observations: red_chi2={red_chi2:.2f}, p_sf={p_classical:.3f}, p_PPC={p_ppc:.3f}, posterior={post_prob:.3f}."
            elif post_prob >= 0.20:
                status = HypothesisEvaluationStatus.UNRESOLVED_ALTERNATIVE
                summary = f"Plausible alternative model: red_chi2={red_chi2:.2f}, p_sf={p_classical:.3f}, p_PPC={p_ppc:.3f}, posterior={post_prob:.3f}."
            else:
                status = HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS
                summary = f"Likelihood deficit: posterior={post_prob:.3f} despite acceptable residual fit."

            critiques[h_name] = HypothesisCriticismResult(
                hypothesis_name=h_name,
                current_posterior=round(post_prob, 4),
                degrees_of_freedom=dof,
                reduced_chi2=round(red_chi2, 3),
                goodness_of_fit_p_value=round(p_classical, 4),
                bayesian_ppc_p_value=round(p_ppc, 4),
                observed_discrepancy_stat=round(chi2_stat, 3),
                replicated_discrepancy_mean=round(rep_mean, 3),
                replicated_discrepancy_std=round(rep_std, 3),
                mean_normalized_residual=round(mean_r, 3),
                max_normalized_residual=round(max_r, 3),
                is_statistically_consistent=is_consistent,
                evaluation_status=status,
                criticism_summary=summary,
            )

        # Identify preferred model or unresolved state
        if all_misspecified:
            verdict = "MODEL MISSPECIFICATION DETECTED: All candidate hypotheses show significant residual divergence (p < 0.01)."
            pref_hyp = None
            unresolved = True
        else:
            consistent_models = [h for h, c in critiques.items() if c.is_statistically_consistent]
            pref_hyp = max(consistent_models, key=lambda h: hypothesis_posteriors[h])
            unresolved = any(h != pref_hyp and hypothesis_posteriors[h] >= 0.10 for h in consistent_models)
            verdict = f"Predictive check passed. Preferred hypothesis: {pref_hyp}."

        return PosteriorPredictiveAssessment(
            target_id=target_id,
            num_observations_evaluated=n_obs,
            hypotheses_critiques=critiques,
            all_models_misspecified=all_misspecified,
            preferred_hypothesis=pref_hyp,
            unresolved_alternatives_present=unresolved,
            model_discrepancy_detected=all_misspecified,
            audit_verdict=verdict,
        )
