"""
ASTRAEUS Physical Model Engine.
Orchestrates class-conditional and phase-conditional physical forward fitting,
evaluates parameter identifiability, performs BIC model comparison,
and returns three decoupled scientific outputs:
1. Model Fit Quality (reduced chi2, BIC): Does the model describe the data?
2. Physical Parameter Vector (theta_x): Inferred physical parameters.
3. Parameter-Space Deviation (D_phys): Statistical novelty relative to known population.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel, Field

from astraeus_models.physics.fit_status import FitStatus
from astraeus_models.physics.applicability import ModelApplicabilityMatrix
from astraeus_models.physics.bazin import fit_bazin, BazinFitResult
from astraeus_models.physics.arnett import fit_arnett, ArnettFitResult
from astraeus_models.physics.shock_cooling import fit_shock_cooling, ShockCoolingFitResult


# Reference parameter priors for ordinary SNe Ia population (mean, std)
# Based on Bazin et al. 2009 / Kessler et al. 2009 empirical distributions
REFERENCE_SN_PARAMETERS = {
    "tau_rise": (3.5, 0.8),    # days (Bazin e-folding rise time)
    "tau_fall": (22.0, 5.5),   # days (Bazin exponential decay time)
    "m_ni": (0.50, 0.15),      # solar masses (Arnett radioactive Nickel mass)
    "tau_d": (16.0, 3.5),      # days (Arnett effective diffusion time)
}


class PhysicsFitResult(BaseModel):
    """
    Decoupled physical forward modeling outputs preventing fit failures
    from being conflated with astrophysical novelty.
    """
    preferred_model: Optional[str] = None
    best_bic: Optional[float] = None
    model_fit_quality: Optional[float] = None  # reduced chi2 of the best-fitting model
    physical_parameters: Dict[str, float] = Field(default_factory=dict)
    parameter_space_deviation: float = 0.0     # D_phys: distance in physical parameter manifold
    physical_residual: float = 0.0             # Preserved for backward compatibility
    bazin_result: BazinFitResult
    arnett_result: ArnettFitResult
    shock_cooling_result: ShockCoolingFitResult
    model_comparison: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    summary_status: str = "COMPLETED"


class PhysicalModelEngine:
    """
    Coordinates physics model applicability evaluation, fitting, model comparison,
    and parameter-space deviation estimation.
    """

    def __init__(self):
        self.applicability_matrix = ModelApplicabilityMatrix()

    def compute_parameter_deviation(self, params: Dict[str, float]) -> float:
        """
        Computes standardized parameter deviation D_phys against normal supernova population:
        D_phys = sqrt(sum_i ((theta_i - mu_i) / sigma_i)^2)
        """
        deviations = []
        for p_name, val in params.items():
            if p_name in REFERENCE_SN_PARAMETERS:
                mu, sigma = REFERENCE_SN_PARAMETERS[p_name]
                deviations.append(((val - mu) / sigma) ** 2)

        if not deviations:
            return 0.0
        return float(np.sqrt(np.mean(deviations)))

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
        successful_models: List[Tuple[str, float, float, Dict[str, float]]] = [] # (name, bic, chi2_red, params)

        for name, res in [("bazin", bazin_res), ("arnett", arnett_res), ("shock_cooling", shock_res)]:
            comparison[name] = {
                "status": res.status.value,
                "bic": res.bic if res.status == FitStatus.FIT_SUCCESS else None,
                "reduced_chi2": res.reduced_chi2 if res.status == FitStatus.FIT_SUCCESS else None,
                "justification": res.applicability_justification,
            }
            if res.status == FitStatus.FIT_SUCCESS:
                successful_models.append((name, res.bic, res.reduced_chi2, res.best_fit_params))

        preferred_model: Optional[str] = None
        best_bic: Optional[float] = None
        model_fit_quality: Optional[float] = None
        physical_parameters: Dict[str, float] = {}
        parameter_deviation: float = 0.0
        physical_residual: float = 0.0

        if successful_models:
            # Sort by lowest BIC
            successful_models.sort(key=lambda x: x[1])
            preferred_model = successful_models[0][0]
            best_bic = successful_models[0][1]
            model_fit_quality = successful_models[0][2] # reduced chi2 of best physical fit
            physical_parameters = successful_models[0][3]
            parameter_deviation = self.compute_parameter_deviation(physical_parameters)
            physical_residual = model_fit_quality

            # Add delta BIC to comparison
            for name in comparison:
                if comparison[name]["bic"] is not None and best_bic is not None:
                    comparison[name]["delta_bic"] = float(comparison[name]["bic"] - best_bic)
        else:
            # If no physical model converged, compute observational scatter
            med_err = float(np.median(np.clip(flux_errors, 1e-4, None)))
            physical_residual = float(np.std(fluxes) / med_err) if med_err > 0 else 0.0
            parameter_deviation = 0.0

        return PhysicsFitResult(
            preferred_model=preferred_model,
            best_bic=best_bic,
            model_fit_quality=model_fit_quality,
            physical_parameters=physical_parameters,
            parameter_space_deviation=parameter_deviation,
            physical_residual=physical_residual,
            bazin_result=bazin_res,
            arnett_result=arnett_res,
            shock_cooling_result=shock_res,
            model_comparison=comparison,
            summary_status="CONVERGED" if preferred_model is not None else "NO_MODEL_CONVERGED"
        )
