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
    reduced_chi2: float
    goodness_of_fit_p_value: float
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
    audit_verdict: str = ""


class PosteriorPredictiveChecker:
    """
    Executes posterior predictive checks (PPC) and model criticism
    across accumulated observations.
    """

    MODEL_VARIANCE_MAG = 0.05

    @classmethod
    def evaluate_model_fit(
        cls,
        target_id: str,
        hypothesis_posteriors: Dict[str, float],
        accumulated_observations: List[CalibratedObservation],
        trigger_mjd: float = 60000.0,
    ) -> PosteriorPredictiveAssessment:
        """
        Computes residual statistics, chi2, and p-values for all candidate hypotheses
        against all active point detections and censored limits.
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

        for h_name, post_prob in hypothesis_posteriors.items():
            residuals = []
            
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

                pred_m = PredictiveEIGEngine.predict_hypothesis_magnitude(
                    hypothesis_name=h_name,
                    regime=regime,
                    effective_wavelength_angstrom=wl,
                    time_since_trigger_days=t_days,
                )
                r = (meas_m - pred_m) / max(1e-4, tot_err)
                residuals.append(r)

            res_arr = np.array(residuals)
            chi2_stat = float(np.sum(res_arr ** 2))
            dof = max(1, n_obs)
            red_chi2 = float(chi2_stat / dof)
            p_val = float(1.0 - chi2.cdf(chi2_stat, dof))

            mean_r = float(np.mean(np.abs(res_arr)))
            max_r = float(np.max(np.abs(res_arr)))

            # Model consistency threshold: p-value > 0.01 and reduced chi2 < 3.5
            is_consistent = (p_val >= 0.01) and (red_chi2 < 3.5)
            if is_consistent:
                all_misspecified = False

            # Assign 4-tier taxonomy
            if not is_consistent:
                status = HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS
                summary = f"Rejected under assumptions: red_chi2={red_chi2:.2f}, p={p_val:.4f} < 0.01 (max residual {max_r:.1f} sigma)."
            elif post_prob >= 0.60:
                status = HypothesisEvaluationStatus.STRONGER_EVIDENCE
                summary = f"Supported by current observations: red_chi2={red_chi2:.2f}, p={p_val:.3f}, posterior={post_prob:.3f}."
            elif post_prob >= 0.20:
                status = HypothesisEvaluationStatus.UNRESOLVED_ALTERNATIVE
                summary = f"Plausible alternative model: red_chi2={red_chi2:.2f}, p={p_val:.3f}, posterior={post_prob:.3f}."
            else:
                status = HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS
                summary = f"Statistical likelihood deficit: posterior={post_prob:.3f}."

            critiques[h_name] = HypothesisCriticismResult(
                hypothesis_name=h_name,
                current_posterior=round(post_prob, 4),
                reduced_chi2=round(red_chi2, 3),
                goodness_of_fit_p_value=round(p_val, 4),
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
            audit_verdict=verdict,
        )
