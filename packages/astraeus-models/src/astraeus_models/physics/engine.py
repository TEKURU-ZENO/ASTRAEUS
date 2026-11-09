"""
ASTRAEUS Physical Model Engine.
Orchestrates class-conditional and phase-conditional physical forward fitting,
evaluates parameter identifiability, performs BIC model comparison,
and returns structured evidence without conflating fit failures with anomalies.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel, Field

from astraeus_models.physics.fit_status import FitStatus
from astraeus_models.physics.applicability import ModelApplicabilityMatrix
from astraeus_models.physics.bazin import fit_bazin, BazinFitResult
from astraeus_models.physics.arnett import fit_arnett, ArnettFitResult
from astraeus_models.physics.shock_cooling import fit_shock_cooling, ShockCoolingFitResult


class PhysicsFitResult(BaseModel):
    """
    Unified result of physical forward modeling across candidate models.
    """
    preferred_model: Optional[str] = None
    best_bic: Optional[float] = None
    physical_residual: float = 0.0
    bazin_result: BazinFitResult
    arnett_result: ArnettFitResult
    shock_cooling_result: ShockCoolingFitResult
    model_comparison: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    summary_status: str = "COMPLETED"


class PhysicalModelEngine:
    """
    Coordinates physics model applicability evaluation, fitting, and model selection.
    """

    def __init__(self):
        self.applicability_matrix = ModelApplicabilityMatrix()

    def fit_all_applicable(
        self,
        times_mjd: np.ndarray,
        fluxes: np.ndarray,
        flux_errors: np.ndarray,
        candidate_hypothesis: str = "unknown"
    ) -> PhysicsFitResult:
        """
        Runs applicability checks and forward fits for all candidate physical models.
        """
        # 1. Applicability Matrix Check
        # Convert flux to pseudo-magnitude for applicability heuristics if needed
        # (mag ~ -2.5 log10(flux) + 25)
        pos_flux = np.clip(fluxes, 1e-4, None)
        pseudo_mag = -2.5 * np.log10(pos_flux) + 25.0

        app_results = self.applicability_matrix.evaluate_applicability(
            candidate_hypothesis=candidate_hypothesis,
            times_mjd=times_mjd,
            magnitudes=pseudo_mag
        )

        # 2. Fit Bazin
        bazin_app, bazin_reason = app_results.get("bazin", (True, ""))
        bazin_res = fit_bazin(
            times_mjd=times_mjd,
            fluxes=fluxes,
            flux_errors=flux_errors,
            is_applicable=bazin_app,
            justification=bazin_reason
        )

        # 3. Fit Arnett
        arnett_app, arnett_reason = app_results.get("arnett", (True, ""))
        arnett_res = fit_arnett(
            times_mjd=times_mjd,
            fluxes=fluxes,
            flux_errors=flux_errors,
            is_applicable=arnett_app,
            justification=arnett_reason
        )

        # 4. Fit Shock Cooling
        shock_app, shock_reason = app_results.get("shock_cooling", (True, ""))
        shock_res = fit_shock_cooling(
            times_mjd=times_mjd,
            fluxes=fluxes,
            flux_errors=flux_errors,
            is_applicable=shock_app,
            justification=shock_reason
        )

        # 5. Model Comparison (BIC & Residual Ranking)
        comparison: Dict[str, Dict[str, Any]] = {}
        successful_models: List[Tuple[str, float, float]] = [] # (name, bic, chi2_red)

        for name, res in [("bazin", bazin_res), ("arnett", arnett_res), ("shock_cooling", shock_res)]:
            comparison[name] = {
                "status": res.status.value,
                "bic": res.bic if res.status == FitStatus.FIT_SUCCESS else None,
                "reduced_chi2": res.reduced_chi2 if res.status == FitStatus.FIT_SUCCESS else None,
                "justification": res.applicability_justification,
            }
            if res.status == FitStatus.FIT_SUCCESS:
                successful_models.append((name, res.bic, res.reduced_chi2))

        preferred_model: Optional[str] = None
        best_bic: Optional[float] = None
        physical_residual: float = 0.0

        if successful_models:
            # Sort by lowest BIC
            successful_models.sort(key=lambda x: x[1])
            preferred_model = successful_models[0][0]
            best_bic = successful_models[0][1]
            physical_residual = successful_models[0][2] # reduced chi2 of best physical fit

            # Add delta BIC to comparison
            for name in comparison:
                if comparison[name]["bic"] is not None and best_bic is not None:
                    comparison[name]["delta_bic"] = float(comparison[name]["bic"] - best_bic)
        else:
            # If no model converged or was applicable, compute empirical scatter as baseline residual
            # (standard deviation of flux normalised by median uncertainty)
            med_err = float(np.median(np.clip(flux_errors, 1e-4, None)))
            physical_residual = float(np.std(fluxes) / med_err) if med_err > 0 else 0.0

        return PhysicsFitResult(
            preferred_model=preferred_model,
            best_bic=best_bic,
            physical_residual=physical_residual,
            bazin_result=bazin_res,
            arnett_result=arnett_res,
            shock_cooling_result=shock_res,
            model_comparison=comparison,
            summary_status="CONVERGED" if preferred_model is not None else "NO_MODEL_CONVERGED"
        )
