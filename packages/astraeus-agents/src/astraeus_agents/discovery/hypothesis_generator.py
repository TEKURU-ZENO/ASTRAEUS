"""
ASTRAEUS Hypothesis Generation Engine & Evolutionary Tournament (WS8.3).
Provides neuro-symbolic and evolutionary hypothesis synthesis, structured falsification criteria,
and iterative hypothesis refinement.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field

from astraeus_data.discovery.anomaly_engine import AnomalyDiagnostic, AnomalySource


class HypothesisPrediction(BaseModel):
    """Quantitative, testable prediction derived from a scientific hypothesis."""
    observable_name: str = Field(..., description="Target observable, e.g. optical_mag_r, x_ray_flux_cgs")
    predicted_value: float = Field(..., description="Nominal predicted central value")
    uncertainty_range: Tuple[float, float] = Field(..., description="(lower_bound, upper_bound) at 95% credible level")
    epoch_days_from_trigger: float = Field(..., description="Time of predicted observable relative to trigger")
    measurement_condition: str = Field(..., description="Required instrument filter, energy band, or cadence")
    is_falsified: bool = Field(default=False)


class StructuredHypothesis(BaseModel):
    """
    Standardized, executable scientific hypothesis schema.
    Explicitly binds phenomenon, physical mechanism, mathematical formulation,
    parameter priors, quantitative predictions, and falsification conditions.
    """
    hypothesis_id: str = Field(..., description="Unique deterministic hypothesis identifier")
    phenomenon: str = Field(..., description="Empirical target phenomenon being explained")
    explanation: str = Field(..., description="Prose description of physical mechanism")
    assumptions: List[str] = Field(default_factory=list, description="Explicit theoretical and physical assumptions")
    mathematical_formulation: str = Field(..., description="Governing symbolic/analytical equations")
    parameters: Dict[str, Tuple[float, float, float, str]] = Field(
        default_factory=dict,
        description="Free parameters: {name: (nominal, min_val, max_val, unit)}"
    )
    predictions: List[HypothesisPrediction] = Field(default_factory=list, description="Testable predictions")
    alternatives: List[str] = Field(default_factory=list, description="Competing hypothesis IDs or names")
    falsification_conditions: List[str] = Field(default_factory=list, description="Definitive falsification criteria")
    discriminating_experiment: str = Field(..., description="Decisive discriminating observation or experiment")
    lineage_parent_ids: List[str] = Field(default_factory=list, description="Parent hypothesis IDs if derived/mutated")
    tournament_score: float = Field(default=0.0, description="Fitness score from evolutionary tournament")
    falsification_status: str = Field(default="ACTIVE", description="ACTIVE, FALSIFIED, or CONFIRMED")
    sha256_hash: str = Field(default="")
    created_at_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_hash(self) -> str:
        preds = [
            {
                "observable_name": p.observable_name,
                "predicted_value": round(float(p.predicted_value), 6),
                "uncertainty_range": [round(float(p.uncertainty_range[0]), 6), round(float(p.uncertainty_range[1]), 6)],
                "epoch_days_from_trigger": round(float(p.epoch_days_from_trigger), 4),
                "measurement_condition": p.measurement_condition.strip(),
            }
            for p in self.predictions
        ]
        preds = sorted(preds, key=lambda x: (x["observable_name"], x["epoch_days_from_trigger"]))

        params = {}
        for k, v in sorted(self.parameters.items()):
            params[k] = [round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6), str(v[3])]

        payload = {
            "hypothesis_id": self.hypothesis_id,
            "phenomenon": self.phenomenon.strip(),
            "explanation": self.explanation.strip(),
            "assumptions": sorted(self.assumptions),
            "mathematical_formulation": self.mathematical_formulation.strip(),
            "parameters": params,
            "predictions": preds,
            "alternatives": sorted(self.alternatives),
            "falsification_conditions": sorted(self.falsification_conditions),
            "discriminating_experiment": self.discriminating_experiment.strip(),
            "lineage_parent_ids": sorted(self.lineage_parent_ids),
        }
        raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        if not self.sha256_hash:
            self.sha256_hash = self.compute_hash()


class HypothesisGenerationEngine:
    """
    Synthesizes structured hypotheses from anomaly diagnostics and observational evidence.
    Executes evolutionary tournament selection, parsimony scoring, and refinement cycles.
    """

    def generate_candidate_hypotheses(
        self,
        anomaly_diag: AnomalyDiagnostic,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[StructuredHypothesis]:
        """
        Synthesizes a set of competing hypotheses tailored to the detected physical anomaly.
        """
        hypotheses: List[StructuredHypothesis] = []
        tid = anomaly_diag.target_id
        src = anomaly_diag.source_classification

        # Hypothesis 1: Relativistic Shock Interaction with Dense Circumstellar Medium (CSM)
        h1 = StructuredHypothesis(
            hypothesis_id=f"hyp_{tid}_csm_interaction_v1",
            phenomenon=f"Rapid early excess and optical/UV emission in transient {tid}",
            explanation=(
                "Fast ejecta from core-collapse or engine-driven transient collides with dense circumstellar shell "
                "deposited by pre-supernova mass loss, converting kinetic energy to early luminous thermal radiation."
            ),
            assumptions=[
                "Spherically symmetric wind profile: rho(r) = D_star * r^(-2)",
                "Electron scattering opacity kappa = 0.34 cm^2/g dominates pre-shock medium",
                "Radiative shock breakout condition: tau_shock ~ c / v_shock",
            ],
            mathematical_formulation="L_shock(t) = 0.5 * (4 * pi * rho_0 * v_shock^5 * t^2) / (1 + t / t_diff)",
            parameters={
                "ejecta_velocity_c": (0.10, 0.03, 0.30, "c"),
                "csm_mass_msun": (0.05, 0.001, 1.0, "M_sun"),
                "csm_radius_cm": (1e14, 1e13, 1e16, "cm"),
                "shock_efficiency": (0.35, 0.05, 0.80, "dimensionless"),
            },
            predictions=[
                HypothesisPrediction(
                    observable_name="optical_mag_g",
                    predicted_value=18.5,
                    uncertainty_range=(17.8, 19.2),
                    epoch_days_from_trigger=1.0,
                    measurement_condition="Optical g-band photometry at t=1d",
                ),
                HypothesisPrediction(
                    observable_name="x_ray_flux_cgs",
                    predicted_value=5e-13,
                    uncertainty_range=(1e-13, 2e-12),
                    epoch_days_from_trigger=3.0,
                    measurement_condition="Swift-XRT 0.3-10 keV flux at t=3d",
                ),
                HypothesisPrediction(
                    observable_name="ha_emission_fwhm_km_s",
                    predicted_value=2500.0,
                    uncertainty_range=(1500.0, 4000.0),
                    epoch_days_from_trigger=5.0,
                    measurement_condition="Optical spectroscopy at H-alpha",
                ),
            ],
            alternatives=[f"hyp_{tid}_magnetar_injection_v1", f"hyp_{tid}_lanthanide_kilonova_v1"],
            falsification_conditions=[
                "Absence of narrow or intermediate-width Balmer/He emission lines in optical spectra (FWHM < 4000 km/s)",
                "Radio synchrotron flux upper limit < 10 uJy at 5 GHz at t=10d (rules out shock into dense medium)",
                "Optical g-r color turning red (g-r > +0.8 mag) within first 48 hours",
            ],
            discriminating_experiment="Early optical spectroscopy to detect narrow emission lines + VLA radio follow-up at 5 GHz",
        )
        hypotheses.append(h1)

        # Hypothesis 2: Millisecond Magnetar Central Engine Injection
        h2 = StructuredHypothesis(
            hypothesis_id=f"hyp_{tid}_magnetar_injection_v1",
            phenomenon=f"Persistent high luminosity and slow decline or plateau in transient {tid}",
            explanation=(
                "A rapidly spinning, highly magnetized neutron star (P_0 ~ 1-5 ms, B ~ 10^14-10^15 G) formed in "
                "the merger or collapse deposits magnetic dipole spindown energy into expanding ejecta."
            ),
            assumptions=[
                "Magnetic dipole radiation spindown: L_sd(t) = E_rot / t_sd * (1 + t / t_sd)^(-2)",
                "Thermalization efficiency eta ~ 1.0 at early times",
                "Ejecta remains optically thick to Poynting flux bubble",
            ],
            mathematical_formulation="L_sd(t) = L_0 * (1.0 + t / t_sd)^(-2.0)",
            parameters={
                "initial_period_ms": (2.0, 0.7, 10.0, "ms"),
                "magnetic_field_gauss": (2e14, 1e14, 5e15, "Gauss"),
                "ejecta_mass_msun": (0.08, 0.01, 0.50, "M_sun"),
            },
            predictions=[
                HypothesisPrediction(
                    observable_name="optical_mag_g",
                    predicted_value=19.2,
                    uncertainty_range=(18.5, 19.8),
                    epoch_days_from_trigger=1.0,
                    measurement_condition="Optical g-band at t=1d",
                ),
                HypothesisPrediction(
                    observable_name="x_ray_flux_cgs",
                    predicted_value=1.5e-11,
                    uncertainty_range=(5e-12, 4e-11),
                    epoch_days_from_trigger=3.0,
                    measurement_condition="Chandra or Swift 0.3-10 keV flux",
                ),
            ],
            alternatives=[f"hyp_{tid}_csm_interaction_v1"],
            falsification_conditions=[
                "Total radiated energy exceeds maximum available rotational kinetic energy E_rot,max = 0.5 * I * Omega^2 ~ 3e52 erg",
                "Rapid catastrophic fading by > 3 magnitudes within 24 hours (inconsistent with (1+t/t_sd)^-2 decline)",
                "Non-detection in deep X-rays (< 1e-14 erg/s/cm^2) at t=10d",
            ],
            discriminating_experiment="High-cadence X-ray monitoring with Swift/Chandra to track dipole spindown slope",
        )
        hypotheses.append(h2)

        # Hypothesis 3: Binary Neutron Star Merger with Multi-Component Kilonova
        if (
            "KILONOVA" in str(src)
            or anomaly_diag.multimodal_inconsistency_score > 2.0
            or anomaly_diag.temporal_residual_score > 3.0
        ):
            h3 = StructuredHypothesis(
                hypothesis_id=f"hyp_{tid}_lanthanide_kilonova_v1",
                phenomenon=f"Fast-evolving red/NIR transient with rapid optical fading in {tid}",
                explanation=(
                    "Decompression of cold neutron-rich matter from BNS/NS-BH merger drives rapid neutron-capture (r-process) "
                    "nucleosynthesis. Lanthanide-free polar wind powers early blue emission (t < 1d); lanthanide-rich dynamical "
                    "ejecta powers red/NIR plateau (t ~ 3-10d) due to high opacity."
                ),
                assumptions=[
                    "Radioactive heating rate: q_dot(t) = q_0 * (t / 1_day)^(-1.3)",
                    "Two-component geometry: blue component (Y_e > 0.25) + red component (Y_e < 0.20)",
                    "Opacity kappa_blue ~ 0.5 cm^2/g, kappa_red ~ 10.0 cm^2/g",
                ],
                mathematical_formulation="L_bol(t) = M_ej * q_0 * (t / t_0)^(-1.3) * (1.0 - exp(-t_diff^2 / t^2))",
                parameters={
                    "blue_ejecta_mass_msun": (0.015, 0.002, 0.04, "M_sun"),
                    "red_ejecta_mass_msun": (0.035, 0.005, 0.08, "M_sun"),
                    "v_ejecta_c": (0.18, 0.08, 0.35, "c"),
                },
                predictions=[
                    HypothesisPrediction(
                        observable_name="optical_mag_g",
                        predicted_value=20.5,
                        uncertainty_range=(19.8, 21.3),
                        epoch_days_from_trigger=1.0,
                        measurement_condition="Optical g-band at t=1d",
                    ),
                    HypothesisPrediction(
                        observable_name="nir_mag_k",
                        predicted_value=17.8,
                        uncertainty_range=(17.2, 18.4),
                        epoch_days_from_trigger=3.0,
                        measurement_condition="NIR Ks-band at t=3d",
                    ),
                    HypothesisPrediction(
                        observable_name="color_g_minus_k",
                        predicted_value=3.2,
                        uncertainty_range=(2.5, 4.0),
                        epoch_days_from_trigger=3.0,
                        measurement_condition="g - Ks color at t=3d",
                    ),
                ],
                alternatives=[f"hyp_{tid}_csm_interaction_v1", f"hyp_{tid}_magnetar_injection_v1"],
                falsification_conditions=[
                    "Absence of rapid reddening: g - Ks color remaining < 1.0 mag at t=3 days",
                    "Continued optical plateau at g < 20.0 mag beyond t=5 days",
                    "Spectroscopic identification of unshifted narrow H-alpha or Fe lines",
                ],
                discriminating_experiment="Simultaneous Optical (g, r) and Near-Infrared (J, Ks) photometry within 72 hours",
            )
            hypotheses.append(h3)

        return hypotheses

    def evaluate_tournament(
        self,
        hypotheses: List[StructuredHypothesis],
        empirical_observations: List[Dict[str, Any]],
    ) -> List[StructuredHypothesis]:
        """
        Executes evolutionary tournament scoring over candidate hypotheses:
        Fitness = log_likelihood - parsimony_penalty + falsifiability_bonus - physical_penalty.
        """
        scored_hypotheses = []

        for h in hypotheses:
            # 1. Parsimony penalty: penalty per free parameter
            n_params = len(h.parameters)
            parsimony_penalty = 0.5 * n_params

            # 2. Falsifiability score: reward tight, quantitative predictions (relative precision)
            falsifiability_bonus = 0.0
            for pred in h.predictions:
                span = pred.uncertainty_range[1] - pred.uncertainty_range[0]
                val = abs(pred.predicted_value)
                if span > 0.0 and val > 0.0:
                    rel_span = span / val
                    falsifiability_bonus += max(0.5, min(5.0, 0.5 / max(0.05, rel_span)))

            # 3. Empirical Likelihood vs Observations
            log_likelihood = 0.0
            is_falsified = False
            falsification_reasons = []

            for obs in empirical_observations:
                obs_name = obs.get("observable_name")
                obs_val = obs.get("value")
                obs_err = obs.get("uncertainty", 0.1)

                matching_pred = next((p for p in h.predictions if p.observable_name == obs_name), None)
                if matching_pred and obs_val is not None:
                    pred_val = matching_pred.predicted_value
                    pred_low, pred_high = matching_pred.uncertainty_range
                    pred_sigma = max(1e-4, (pred_high - pred_low) / 3.92)
                    comb_sigma = math.sqrt(obs_err**2 + pred_sigma**2)

                    residual = abs(obs_val - pred_val)
                    chi = residual / comb_sigma
                    log_likelihood -= 0.5 * (chi**2)

                    # Check hard falsification threshold (> 5 sigma mismatch)
                    if chi > 5.0:
                        is_falsified = True
                        matching_pred.is_falsified = True
                        falsification_reasons.append(
                            f"Decisive falsification on {obs_name}: measured={obs_val:.3f}+/-{obs_err:.3f}, "
                            f"predicted={pred_val:.3f}+/-{pred_sigma:.3f} (residual={chi:.1f} sigma)"
                        )

            # 4. Total Tournament Fitness Score
            if is_falsified:
                h.falsification_status = "FALSIFIED"
                fitness = -999.0 + log_likelihood
            else:
                fitness = log_likelihood - parsimony_penalty + falsifiability_bonus
                h.falsification_status = "ACTIVE"

            h.tournament_score = round(fitness, 3)
            scored_hypotheses.append(h)

        # Sort descending by tournament score
        scored_hypotheses.sort(key=lambda x: x.tournament_score, reverse=True)
        return scored_hypotheses

    def refine_falsified_hypothesis(
        self,
        hypothesis: StructuredHypothesis,
        falsification_evidence: str,
        generation_index: int = 2,
    ) -> StructuredHypothesis:
        """
        Creates an evolved mutant hypothesis addressing specific falsification evidence,
        recording parent lineage.
        """
        mutant = copy.deepcopy(hypothesis)
        mutant.hypothesis_id = f"{hypothesis.hypothesis_id}_gen{generation_index}"
        mutant.lineage_parent_ids = [hypothesis.hypothesis_id]
        mutant.falsification_status = "ACTIVE"
        mutant.tournament_score = 0.0

        # Adjust parameters / assumptions to absorb empirical tension
        mutant.assumptions.append(f"Revised under falsification feedback: {falsification_evidence}")
        if "csm_mass_msun" in mutant.parameters:
            nom, pmin, pmax, u = mutant.parameters["csm_mass_msun"]
            mutant.parameters["csm_mass_msun"] = (nom * 0.2, pmin * 0.1, pmax * 0.5, u)

        # Recompute hash
        mutant.sha256_hash = mutant.compute_hash()
        return mutant
