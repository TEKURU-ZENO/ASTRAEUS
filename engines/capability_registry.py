"""
ASTRAEUS Scientific Capability Registry & Epistemic Guardrails (Phase 7 Foundation).
Declares and enforces the formal epistemic and operational status of every computational engine.

Epistemic Mandate:
Prevents agentic systems from retroactively promoting experimental prototypes,
uncalibrated numerical models, or synthetic benchmarks into validated scientific instruments.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class EngineCapabilityTier(str, Enum):
    """
    Formal operational classification of ASTRAEUS computational components.
    Enforces strict epistemic constraints on autonomous agent reasoning and claims.
    """
    VALIDATED_INSTRUMENT = "VALIDATED_INSTRUMENT"
    # Certified for independent scientific inference, empirical discovery, and publication.

    EXPERIMENTAL_PROTOTYPE = "EXPERIMENTAL_PROTOTYPE"
    # Working prototype under research investigation. Predictions carry mandatory epistemic
    # qualifiers; output CANNOT be used to assert confirmed astrophysical discoveries.

    NUMERICAL_BENCHMARK_ONLY = "NUMERICAL_BENCHMARK_ONLY"
    # Verified numerical/mathematical operator (e.g. Poisson solver, symplectic integrator).
    # Correct algorithmic implementation, but uncalibrated to real multi-survey observations.

    UNAVAILABLE_FOR_SCIENTIFIC_USE = "UNAVAILABLE_FOR_SCIENTIFIC_USE"
    # Quarantined, falsified, or superseded component. Ingestion into science loop prohibited.


class EpistemicPromotionViolation(Exception):
    """Raised when an autonomous agent or workflow attempts to cite an unvalidated model as ground truth."""
    pass


@dataclass(frozen=True)
class EngineDeclaration:
    """Formal declaration of an engine's scientific status and operational boundaries."""
    engine_id: str
    name: str
    tier: EngineCapabilityTier
    version: str
    empirical_limitations: List[str]
    authorized_use_cases: List[str]
    prohibited_use_cases: List[str]
    provenance_hash: Optional[str] = None
    notes: Optional[str] = None

    def assert_usage_authorized(self, intended_use: str, claiming_validated_discovery: bool = False) -> None:
        """Enforces operational boundaries and blocks unearned scientific promotion."""
        if claiming_validated_discovery and self.tier != EngineCapabilityTier.VALIDATED_INSTRUMENT:
            raise EpistemicPromotionViolation(
                f"EPISTEMIC VIOLATION: Engine '{self.engine_id}' has status '{self.tier.value}'. "
                f"It cannot be cited as a validated scientific instrument for confirmed discoveries. "
                f"Empirical limitations: {'; '.join(self.empirical_limitations)}"
            )

        if self.tier == EngineCapabilityTier.UNAVAILABLE_FOR_SCIENTIFIC_USE:
            raise EpistemicPromotionViolation(
                f"OPERATIONAL PROHIBITION: Engine '{self.engine_id}' is marked UNAVAILABLE_FOR_SCIENTIFIC_USE."
            )


class ScientificCapabilityRegistry:
    """Central registry tracking and auditing capability tiers across Project ASTRAEUS."""

    _REGISTRY: Dict[str, EngineDeclaration] = {}

    @classmethod
    def register_engine(cls, declaration: EngineDeclaration) -> None:
        cls._REGISTRY[declaration.engine_id] = declaration

    @classmethod
    def get_engine(cls, engine_id: str) -> EngineDeclaration:
        if engine_id not in cls._REGISTRY:
            raise KeyError(f"Engine '{engine_id}' is not registered in the Scientific Capability Registry.")
        return cls._REGISTRY[engine_id]

    @classmethod
    def list_all(cls) -> Dict[str, EngineDeclaration]:
        return dict(cls._REGISTRY)

    @classmethod
    def initialize_phase7_defaults(cls) -> None:
        """Initializes authoritative Phase 6.1 -> Phase 7 capability declarations."""
        cls._REGISTRY.clear()

        # 1. Hydrodynamics FNO3D/PINO Surrogate
        cls.register_engine(
            EngineDeclaration(
                engine_id="engines.hydrodynamics.pino3d",
                name="PINO3D Neural Hydrodynamic Surrogate",
                tier=EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE,
                version="1.0.0-phase6.1",
                empirical_limitations=[
                    "IllustrisTNG held-out relative L2 error is 0.9999 (target < 0.03 not met)",
                    "Speedup of 619.7x reflects single-pass infilling vs 2,000-step Euler solver (different computational workloads)",
                    "Trained on 6 small CAMELS boxes on local GPU; uncalibrated for cosmological baryon fractions",
                ],
                authorized_use_cases=[
                    "Hypothesis exploration and qualitative spatial pattern generation",
                    "Surrogate pipeline architecture development and integration testing",
                    "Fast candidate visualization under explicit experimental disclaimer",
                ],
                prohibited_use_cases=[
                    "Autonomous assertion of physical gas properties or temperatures",
                    "Scientific cosmological parameter constraints",
                    "Claiming validated 600x speedup over equivalent hydrodynamics simulations",
                ],
                provenance_hash="6e5e808...pino_hydro_surrogate.pt",
                notes="Phase 6.1 negative result: retained as experimental prototype only.",
            )
        )

        # 2. Cosmological Parameter Inference Suite
        cls.register_engine(
            EngineDeclaration(
                engine_id="engines.cosmology.differentiable",
                name="Independent Cosmological Parameter Recovery Suite",
                tier=EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE,
                version="1.0.0-phase6.1",
                empirical_limitations=[
                    "Ensemble recovery tested on only N=5 mocks; statistical coverage has large binomial uncertainty",
                    "Cramér-Rao lower bound is a theoretical variance bound, not an empirically calibrated confidence interval",
                    "Inference performed on simplified matter power spectrum, not full survey window function or galaxy bias",
                ],
                authorized_use_cases=[
                    "Cross-model transfer sensitivity analysis (Model A != Model B)",
                    "Fisher identifiability verification and gradient trajectory research",
                    "Initial parameter proposal generation for multi-agent deliberation",
                ],
                prohibited_use_cases=[
                    "Claiming calibrated cosmological parameter constraints from observational surveys",
                    "Claiming proven statistical coverage across general cosmological priors",
                ],
                provenance_hash="phase6.1-independent-cosmology",
                notes="Promising cross-model prototype; requires larger ensemble and real survey calibrations.",
            )
        )

        # 3. Constrained PPO Telescope Scheduler
        cls.register_engine(
            EngineDeclaration(
                engine_id="engines.experimental_design.ppo_scheduler",
                name="Constrained PPO Observing Scheduler",
                tier=EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE,
                version="1.0.0-phase6.1",
                empirical_limitations=[
                    "Empirically outperformed by analytical Greedy EIG oracle (Oracle EIG 122.3 vs PPO 55.1 - 60.8)",
                    "Previous claim of +24% neural superiority retracted as unsupported by 50-night paired benchmark",
                    "Learned policy requires action-masking heuristic to prevent target starvation",
                ],
                authorized_use_cases=[
                    "Research challenger policy to benchmark against analytical scheduling algorithms",
                    "Complex multi-constraint optimization research (slew + weather + airmass)",
                ],
                prohibited_use_cases=[
                    "Promotion as primary or preferred telescope scheduling controller",
                    "Autonomous operational telescope queue management over Greedy EIG without human oversight",
                ],
                provenance_hash="phase6.1-ppo-scheduler",
                notes="Analytical Greedy EIG remains the primary baseline scheduler for Phase 7.",
            )
        )

        # 4. Analytical Greedy EIG Scheduler
        cls.register_engine(
            EngineDeclaration(
                engine_id="engines.experimental_design.greedy_eig",
                name="Analytical Expected Information Gain (EIG) Scheduler",
                tier=EngineCapabilityTier.VALIDATED_INSTRUMENT,
                version="1.0.0-canonical",
                empirical_limitations=[
                    "Myopic 1-step horizon; does not optimize multi-step slew path graph geodesics",
                    "Assumes Gaussian conjugate Bayesian update on transient variance",
                ],
                authorized_use_cases=[
                    "Primary reference scheduler for target follow-up prioritization",
                    "Ground-truth benchmark for experimental design and information gain evaluation",
                ],
                prohibited_use_cases=[
                    "Scheduling across extreme weather degradation where dome closures dominate",
                ],
                notes="Validated oracle baseline demonstrating top performance in Phase 6.1 paired audit.",
            )
        )

        # 5. Classical Symplectic KDK PM Simulator
        cls.register_engine(
            EngineDeclaration(
                engine_id="engines.cosmology.pm_solver",
                name="KDK Symplectic Particle-Mesh Gravity Solver",
                tier=EngineCapabilityTier.NUMERICAL_BENCHMARK_ONLY,
                version="1.0.0-ws1-audited",
                empirical_limitations=[
                    "Fourier mesh resolution limited by memory; mesh Nyquist frequency cutoff",
                    "Does not include baryonic feedback, hydrodynamics, or star formation",
                ],
                authorized_use_cases=[
                    "Numerical trajectory integration under specified gravitational potentials",
                    "Symplectic energy conservation benchmarks and Zel'dovich pancake tests",
                ],
                prohibited_use_cases=[
                    "Direct physical galaxy morphology prediction",
                ],
                notes="Audited in WS1/WS2.1; verified numerical foundation.",
            )
        )


# Auto-initialize Phase 7 defaults on import
ScientificCapabilityRegistry.initialize_phase7_defaults()
