"""
ASTRAEUS Benchmark Evaluation Suite.
Standardized benchmark suites for AstroTemporal, AstroAnomaly, and scientific verification.
"""

from astraeus_bench.metrics import (
    compute_expected_calibration_error,
    compute_crps_gaussian,
    compute_classification_metrics,
    compute_anomaly_budget_metrics,
    CalibrationMetrics,
    ClassificationMetrics,
    AnomalyBenchmarkMetrics,
)
from astraeus_bench.astro_temporal import (
    AstroTemporalBenchmark,
    AstroTemporalComparisonReport,
)
from astraeus_bench.rare_event import (
    CandidateRecord,
    MethodRankingResult,
    RareEventBenchmarkReport,
    RareEventBenchmarkRunner,
)

__version__ = "0.1.0"

__all__ = [
    "compute_expected_calibration_error",
    "compute_crps_gaussian",
    "compute_classification_metrics",
    "compute_anomaly_budget_metrics",
    "CalibrationMetrics",
    "ClassificationMetrics",
    "AnomalyBenchmarkMetrics",
    "AstroTemporalBenchmark",
    "AstroTemporalComparisonReport",
    "CandidateRecord",
    "MethodRankingResult",
    "RareEventBenchmarkReport",
    "RareEventBenchmarkRunner",
]
