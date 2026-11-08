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
]
