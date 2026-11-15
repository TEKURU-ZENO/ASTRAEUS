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
from astraeus_bench.real_survey_benchmark_protocol import (
    REAL_SURVEY_CLASSES_8,
    TRACK_COUNTS_COHORT_10K,
    TRACK1_SPLIT_COUNTS,
    TRACK1_SPATIAL_TEST_SUPPORT,
    WilsonScoreInterval,
    ClassEvaluationWithCI,
    compute_wilson_score_interval,
    verify_cohort_partition_invariants,
    GOVERNANCE_STATEMENT,
)

ACTIVE_SCIENTIFIC_BENCHMARK_REGISTRY = {
    "governance_lock": True,
    "status": "FROZEN_PENDING_AUTHORIZATION",
    "synthetic_benchmarks_allowed": False,
    "governance_statement": GOVERNANCE_STATEMENT,
    "cohort_path": "data/raw/real_survey_v1/cohort_manifest_10k.json",
    "taxonomy": list(REAL_SURVEY_CLASSES_8),
    "track1_labelled_support": 4135,
    "track1_train_support": 2808,
    "track1_val_support": 741,
    "track1_spatial_test_support": 586,
    "track2_unlabelled_support": 4365,
    "track3_artifact_support": 1500,
    "quarantined_support": 79,
    "spatial_test_sealed": True,
    "sprint_3_baselines_status": "FROZEN",
}

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
    "REAL_SURVEY_CLASSES_8",
    "TRACK_COUNTS_COHORT_10K",
    "TRACK1_SPLIT_COUNTS",
    "TRACK1_SPATIAL_TEST_SUPPORT",
    "WilsonScoreInterval",
    "ClassEvaluationWithCI",
    "compute_wilson_score_interval",
    "verify_cohort_partition_invariants",
    "GOVERNANCE_STATEMENT",
    "ACTIVE_SCIENTIFIC_BENCHMARK_REGISTRY",
]
