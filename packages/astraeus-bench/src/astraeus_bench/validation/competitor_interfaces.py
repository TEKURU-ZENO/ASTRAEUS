"""
ASTRAEUS 2.0 Competitor Interfaces and Baseline Contracts (WS-B).
Establishes standardized benchmark inputs, execution contracts, and comparative evaluation interfaces
against external comparator systems across all five scientific comparison groups.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
import numpy as np


class ComparatorSystem(str, Enum):
    # Alert Processing & Transient Classification
    FINK_BROKER = "Fink Broker (Möller et al. 2021)"
    ALERCE_BROKER = "ALeRCE Broker (Sánchez-Sáez et al. 2021)"
    ANTARES_BROKER = "ANTARES (Matheson et al. 2021)"
    LASAIR_BROKER = "Lasair (Smith et al. 2019)"
    ELASTICC_BENCHMARK = "ELAsTiCC Participant Ensemble"

    # Multimodal Scientific AI
    ASTROCLIP = "AstroCLIP (Parker et al. 2024)"
    ASTRO_VL_BENCH = "AstroVLBench 2026 (4,100+ Expert Examples)"

    # Physical Simulation & Radiative Transfer
    KASEN_BARNES_2019 = "Kasen & Barnes (2019) Numerical Radiation-Hydro"
    ARNETT_KASEN_ANALYTIC = "Arnett (1982) / Kasen Analytic Rules"
    BOXFIT_NUMERICAL = "BoxFit Numerical Afterglow Code (van Eerten et al.)"

    # Bayesian & Statistical Inference
    PYMC_STANDARD_MCMC = "PyMC Standard No-U-Turn Sampler (NUTS)"
    EMCEE_ENSEMBLE = "emcee Ensemble MCMC Sampler (Foreman-Mackey et al.)"

    # Autonomous Research Systems
    CHEMCROW_AGENT = "ChemCrow Scientific Agent (Bran et al. 2023)"
    AI_SCIENTIST = "The AI Scientist (Sakana AI 2024)"
    HUMAN_ASTRONOMER_EXPERT = "Human-Led Expert Research Workflow (Baseline)"


class LightCurveData(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_id: str
    ra: float
    dec: float
    mjd: List[float]
    flux: List[float]
    flux_err: List[float]
    passband: List[str]


class ClassificationPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_id: str
    predicted_class: str
    class_probabilities: Dict[str, float]
    epistemic_uncertainty: float = 0.0
    inference_time_ms: float = 0.0


class MultimodalQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_id: str
    image_pixels: Optional[List[List[float]]] = None
    spectrum_wavelengths: Optional[List[float]] = None
    spectrum_flux: Optional[List[float]] = None
    scientific_question: str
    ground_truth_answer: Optional[str] = None


class MultimodalResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_id: str
    predicted_answer: str
    confidence_score: float
    grounded_evidence_citations: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class SimulationTrajectory(BaseModel):
    model_config = ConfigDict(frozen=True)
    times_days: List[float]
    luminosities_erg_s: List[float]
    effective_temperatures_k: Optional[List[float]] = None
    energy_conservation_error: float = 0.0
    compute_time_seconds: float = 0.0


# Abstract Base Classes for Comparators

class AstronomicalBrokerBaseline(ABC):
    """Interface for community alert brokers (Fink, ALeRCE, ANTARES, Lasair)."""

    @abstractmethod
    def get_system_name(self) -> ComparatorSystem:
        pass

    @abstractmethod
    def classify_transient(self, alert_data: LightCurveData) -> ClassificationPrediction:
        pass


class MultimodalScientificAIBaseline(ABC):
    """Interface for astronomy vision-language models (AstroCLIP, AstroVLBench)."""

    @abstractmethod
    def get_system_name(self) -> ComparatorSystem:
        pass

    @abstractmethod
    def evaluate_astronomical_query(self, query: MultimodalQuery) -> MultimodalResponse:
        pass


class PhysicalSimulationBaseline(ABC):
    """Interface for published radiative-transfer and hydro solvers."""

    @abstractmethod
    def get_system_name(self) -> ComparatorSystem:
        pass

    @abstractmethod
    def simulate_light_curve(self, ejecta_mass_msun: float, velocity_c: float, times_days: List[float]) -> SimulationTrajectory:
        pass


class StatisticalInferenceBaseline(ABC):
    """Interface for standard MCMC and Bayesian samplers."""

    @abstractmethod
    def get_system_name(self) -> ComparatorSystem:
        pass

    @abstractmethod
    def sample_posterior(self, observed_lc: LightCurveData, n_samples: int = 1000) -> Dict[str, Any]:
        pass


class AutonomousAgentBaseline(ABC):
    """Interface for agentic scientific research baselines."""

    @abstractmethod
    def get_system_name(self) -> ComparatorSystem:
        pass

    @abstractmethod
    def execute_investigation(self, trigger_prompt: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        pass


# Reference Mock / Baseline Implementations for Unit Testing & Calibration

class ReferenceFinkBrokerBaseline(AstronomicalBrokerBaseline):
    """Calibrated reference implementation of Fink alert classifier."""

    def get_system_name(self) -> ComparatorSystem:
        return ComparatorSystem.FINK_BROKER

    def classify_transient(self, alert_data: LightCurveData) -> ClassificationPrediction:
        # Standard Fink Random Forest / SuperNNova heuristic baseline
        probs = {"KILONOVA": 0.15, "SN_IA": 0.55, "SN_II": 0.20, "AGN": 0.10}
        return ClassificationPrediction(
            target_id=alert_data.target_id,
            predicted_class="SN_IA",
            class_probabilities=probs,
            epistemic_uncertainty=0.25,
            inference_time_ms=12.5
        )


class ReferenceAstroClipBaseline(MultimodalScientificAIBaseline):
    """Calibrated reference implementation of AstroCLIP embedding evaluation."""

    def get_system_name(self) -> ComparatorSystem:
        return ComparatorSystem.ASTROCLIP

    def evaluate_astronomical_query(self, query: MultimodalQuery) -> MultimodalResponse:
        return MultimodalResponse(
            query_id=query.query_id,
            predicted_answer="Binary Neutron Star Merger Ejecta",
            confidence_score=0.88,
            grounded_evidence_citations=["Parker et al. 2024"],
            latency_ms=45.0
        )


class ReferenceArnettKasenBaseline(PhysicalSimulationBaseline):
    """Analytic Arnett-Kasen rule solver for comparison with ASTRAEUS RK4."""

    def get_system_name(self) -> ComparatorSystem:
        return ComparatorSystem.ARNETT_KASEN_ANALYTIC

    def simulate_light_curve(self, ejecta_mass_msun: float, velocity_c: float, times_days: List[float]) -> SimulationTrajectory:
        kappa = 1.0
        beta = 13.8
        c = 3.0e10
        t_peak_days = np.sqrt((2.0 * kappa * ejecta_mass_msun * 1.989e33) / (beta * c * velocity_c * c)) / 86400.0
        l_peak = 1.0e41 * (ejecta_mass_msun / 0.05)

        lums = []
        for t in times_days:
            if t <= t_peak_days:
                lum = l_peak * (t / t_peak_days) ** 2.0
            else:
                lum = l_peak * (t / t_peak_days) ** (-1.3)
            lums.append(float(lum))

        return SimulationTrajectory(
            times_days=times_days,
            luminosities_erg_s=lums,
            effective_temperatures_k=[5000.0] * len(times_days),
            energy_conservation_error=0.005,
            compute_time_seconds=0.001
        )
