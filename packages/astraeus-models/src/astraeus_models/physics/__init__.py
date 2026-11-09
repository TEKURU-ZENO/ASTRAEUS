"""
ASTRAEUS Physics Package.
Contains physically-grounded forward modeling, parameterization, and model selection tools.
"""

from astraeus_models.physics.fit_status import FitStatus
from astraeus_models.physics.applicability import ModelApplicabilityMatrix
from astraeus_models.physics.bazin import bazin_function, fit_bazin, BazinFitResult
from astraeus_models.physics.arnett import (
    arnett_bolometric_luminosity,
    arnett_photometric_flux,
    fit_arnett,
    ArnettFitResult,
)
from astraeus_models.physics.shock_cooling import (
    shock_cooling_luminosity,
    shock_cooling_photometric_flux,
    fit_shock_cooling,
    ShockCoolingFitResult,
)
from astraeus_models.physics.engine import PhysicalModelEngine, PhysicsFitResult

__all__ = [
    "FitStatus",
    "ModelApplicabilityMatrix",
    "bazin_function",
    "fit_bazin",
    "BazinFitResult",
    "arnett_bolometric_luminosity",
    "arnett_photometric_flux",
    "fit_arnett",
    "ArnettFitResult",
    "shock_cooling_luminosity",
    "shock_cooling_photometric_flux",
    "fit_shock_cooling",
    "ShockCoolingFitResult",
    "PhysicalModelEngine",
    "PhysicsFitResult",
]
