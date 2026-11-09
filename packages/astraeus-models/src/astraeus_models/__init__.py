"""
ASTRAEUS Models Package.
Provides periodicity extraction, astronomical feature engineering,
task-separated classification baselines, and hardware VRAM profiling.
"""

from astraeus_models.periodicity import (
    LombScargleExtractor,
    PeriodicityResult,
)
from astraeus_models.features import (
    AstronomicalFeatureExtractor,
    LightCurveFeatures,
)
from astraeus_models.classifier import (
    TaskDomain,
    TaskSeparatedClassifier,
    ClassificationBenchmarkResult,
)
from astraeus_models.hardware_benchmark import (
    HardwareBenchmarkRunner,
    VramBenchmarkResult,
)
from astraeus_models.astromer import (
    AstromerEncoder,
    ContinuousTimePositionalEncoding,
)
from astraeus_models.conformal_calibrator import (
    SplitConformalCalibrator,
    ConformalEvaluationSettingResult,
    ConformalCalibrationReport,
)
from astraeus_models.physics import (
    FitStatus,
    ModelApplicabilityMatrix,
    bazin_function,
    fit_bazin,
    BazinFitResult,
    arnett_bolometric_luminosity,
    arnett_photometric_flux,
    fit_arnett,
    ArnettFitResult,
    shock_cooling_luminosity,
    shock_cooling_photometric_flux,
    fit_shock_cooling,
    ShockCoolingFitResult,
    PhysicalModelEngine,
    PhysicsFitResult,
)

__version__ = "0.1.0"

__all__ = [
    "LombScargleExtractor",
    "PeriodicityResult",
    "AstronomicalFeatureExtractor",
    "LightCurveFeatures",
    "TaskDomain",
    "TaskSeparatedClassifier",
    "ClassificationBenchmarkResult",
    "HardwareBenchmarkRunner",
    "VramBenchmarkResult",
    "AstromerEncoder",
    "ContinuousTimePositionalEncoding",
    "SplitConformalCalibrator",
    "ConformalEvaluationSettingResult",
    "ConformalCalibrationReport",
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
