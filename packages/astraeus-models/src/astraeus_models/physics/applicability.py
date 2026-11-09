"""
ASTRAEUS Model Applicability Matrix.
Enforces physical validity envelopes prior to running numerical forward-fitters.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class ModelApplicabilityMatrix:
    """
    Validates whether a candidate light curve satisfies the physical prerequisites
    for Bazin, Arnett, or Shock-Cooling forward models.
    """

    @staticmethod
    def evaluate_applicability(
        candidate_hypothesis: str,
        times_mjd: np.ndarray,
        magnitudes: np.ndarray
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Returns a dictionary mapping model_name -> (is_applicable: bool, justification: str).
        """
        n_epochs = len(times_mjd)
        duration = float(times_mjd[-1] - times_mjd[0]) if n_epochs > 1 else 0.0
        results: Dict[str, Tuple[bool, str]] = {}

        # 1. Bazin Applicability
        if candidate_hypothesis in {"detector_artifact", "cosmic_ray"}:
            results["bazin"] = (False, "Candidate classified as instrumental artifact; analytical transient profile invalid.")
        elif n_epochs < 5:
            results["bazin"] = (False, f"Insufficient epochs ({n_epochs} < 5) to constrain Bazin morphology parameters.")
        else:
            results["bazin"] = (True, "Candidate is an evolving transient with sufficient sampling.")

        # 2. Arnett Radioactive Diffusion Applicability
        if candidate_hypothesis in {"detector_artifact", "rr_lyrae", "periodic_variable"}:
            results["arnett"] = (False, "Candidate is non-explosive or artifact; radioactive diffusion model invalid.")
        elif n_epochs < 6:
            results["arnett"] = (False, f"Insufficient epochs ({n_epochs} < 6) to infer Nickel mass and diffusion timescale.")
        elif duration < 8.0:
            results["arnett"] = (False, f"Baseline duration ({duration:.1f} d) too short to constrain radioactive decay tail.")
        else:
            results["arnett"] = (True, "Candidate is an explosive transient with multi-week light curve coverage.")

        # 3. Early-Time Shock-Cooling Applicability
        if candidate_hypothesis in {"detector_artifact", "rr_lyrae"}:
            results["shock_cooling"] = (False, "Shock cooling model inapplicable to non-explosive/artifact candidates.")
        elif n_epochs < 3:
            results["shock_cooling"] = (False, "Insufficient early epochs.")
        else:
            # Requires rapid early coverage: earliest epoch must be near initial rise
            t_min = times_mjd[0]
            t_peak = times_mjd[np.argmin(magnitudes)]
            rise_time = t_peak - t_min
            if rise_time > 6.0:
                results["shock_cooling"] = (False, f"Rise time ({rise_time:.1f} d) exceeds early-time shock breakout envelope (<= 5 d).")
            else:
                results["shock_cooling"] = (True, "Candidate exhibits fast early-time rise consistent with shock cooling.")

        return results
