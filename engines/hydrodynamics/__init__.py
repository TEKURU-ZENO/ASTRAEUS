"""
ASTRAEUS Hydrodynamics Engine (Phase 6 - Track B).
Physics-Informed Neural Operator (PINO), 3D Fourier Neural Operator (FNO3D),
CAMELS/IllustrisTNG data contracts, conservation loss, and surrogate benchmarking.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from engines.hydrodynamics.contracts import (
    HydroChannel,
    SimulationSuite,
    HydroFieldMetadata,
    HydroFieldCube,
    generate_synthetic_hydro_cube,
)
from engines.hydrodynamics.pino_model import (
    SpectralConv3d,
    FourierBlock3D,
    PINO3DModel,
)
from engines.hydrodynamics.physics_loss import (
    PhysicsConservationLoss,
    periodic_central_diff,
    periodic_divergence_3d,
    periodic_poisson_solver_3d,
)
from engines.hydrodynamics.surrogate_benchmark import (
    ReferenceHydroSolver3D,
    HydroSurrogateBenchmark,
)

__all__ = [
    "HydroChannel",
    "SimulationSuite",
    "HydroFieldMetadata",
    "HydroFieldCube",
    "generate_synthetic_hydro_cube",
    "SpectralConv3d",
    "FourierBlock3D",
    "PINO3DModel",
    "PhysicsConservationLoss",
    "periodic_central_diff",
    "periodic_divergence_3d",
    "periodic_poisson_solver_3d",
    "ReferenceHydroSolver3D",
    "HydroSurrogateBenchmark",
]
