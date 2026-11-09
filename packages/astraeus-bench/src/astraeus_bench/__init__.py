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
from astraeus_bench.cross_survey import (
    PhysicalParameterConsistencyRecord,
    RankStabilityRecord,
    NegativeTransferRecord,
    CrossSurveyBenchmarkReport,
    CrossSurveyBenchmarkRunner,
)
from astraeus_bench.statistical_testing import (
    StatisticalContrast,
    Top10PermutationDiagnostic,
    compute_paired_wilcoxon,
    compute_exact_mcnemar,
    apply_benjamini_hochberg,
    compute_top10_permutation_test,
)
from astraeus_bench.stage2_temporal_benchmark import (
    ModelTaskEvaluationSummary,
    Stage2BenchmarkReport,
    Stage2TemporalBenchmarkOrchestrator,
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
    "PhysicalParameterConsistencyRecord",
    "RankStabilityRecord",
    "NegativeTransferRecord",
    "CrossSurveyBenchmarkReport",
    "CrossSurveyBenchmarkRunner",
    "StatisticalContrast",
    "Top10PermutationDiagnostic",
    "compute_paired_wilcoxon",
    "compute_exact_mcnemar",
    "apply_benjamini_hochberg",
    "compute_top10_permutation_test",
    "ModelTaskEvaluationSummary",
    "Stage2BenchmarkReport",
    "Stage2TemporalBenchmarkOrchestrator",
]
