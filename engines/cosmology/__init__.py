"""
ASTRAEUS Cosmology Engine (Phase 6).
Provides analytical background cosmology, particle-mesh gravity, symplectic integrators,
and validated numerical reference solvers.
"""

from engines.cosmology.constants import (
    CosmologicalParameters,
    SPEED_OF_LIGHT_KM_S,
    GRAVITATIONAL_CONSTANT_KPC,
    HUBBLE_DISTANCE_MPC,
)
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.mesh import ParticleMeshSolver3D
from engines.cosmology.integrator import SymplecticKDKLeapfrog, SimulationState
from engines.cosmology.analytic_tests import ZeldovichPancakeSolution
from engines.cosmology.jax_engine import (
    JaxParticleMeshSolver3D,
    JaxSimulationState,
    JaxSymplecticKDKLeapfrog,
)

__all__ = [
    "CosmologicalParameters",
    "SPEED_OF_LIGHT_KM_S",
    "GRAVITATIONAL_CONSTANT_KPC",
    "HUBBLE_DISTANCE_MPC",
    "BackgroundCosmology",
    "ParticleMeshSolver3D",
    "SymplecticKDKLeapfrog",
    "SimulationState",
    "ZeldovichPancakeSolution",
    "JaxParticleMeshSolver3D",
    "JaxSimulationState",
    "JaxSymplecticKDKLeapfrog",
]
