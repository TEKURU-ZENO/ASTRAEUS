"""
ASTRAEUS Real-Survey Evaluation Protocol & Three-Track Benchmark Specification.
Sprint 2D / Stage 2 Closure Specification.

Defines the scientific contracts for evaluating ASTRAEUS on real survey data:
1. Track 1: Real Labelled 8-Class Benchmark (N = 4,135)
   - Train (2,808), Validation (741), Spatially Held-Out Test (586)
   - Taxonomy: SN_Ia, SN_CC, SLSN, TDE, CV_Nova, AGN_QSO, Varstar, OTHER_EXTRAGALACTIC
   - Wilson score 95% confidence intervals on all classes (especially rare TDE N=11, Other N=15)
2. Track 2: Unclassified L5 Open-Set & Candidate Ranking Track (N = 4,365)
   - Open-set anomaly detection, abstention coverage, and risk-coverage evaluation
   - Strictly prohibited from being cast into supervised negative labels
3. Track 3: Detection Artifact Observation-Quality Track (N = 1,500)
   - Separate subtraction artifact / dipole quality discrimination task
4. Quarantined Candidates (N = 79)
   - Strictly excluded from headline benchmark scores; reported in quarantine audits

Model Governance:
Sprint 3 baseline model training remains strictly FROZEN until real observation
feature extraction pipelines and data missingness handlers are certified.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Authoritative 8-Class Real-Survey Benchmark Taxonomy
REAL_SURVEY_CLASSES_8: List[str] = [
    "SN_Ia",
    "SN_CC",
    "SLSN",
    "TDE",
    "CV_Nova",
    "AGN_QSO",
    "Varstar",
    "OTHER_EXTRAGALACTIC",
]

# Mandatory Scientific Governance Lock
GOVERNANCE_STATEMENT: str = (
    "Synthetic benchmark results are excluded from all scientific performance claims, "
    "model-selection decisions, and generalization assessments. Synthetic fixtures may be retained "
    "solely for software verification and controlled unit testing. All scientific model performance "
    "must be reported against the designated real-survey evaluation protocol, with the spatially held-out "
    "test partition kept isolated until its authorized evaluation."
)

# Track Cohort Invariant Sizes
TRACK_COUNTS_COHORT_10K = {
    "track1_labelled_astrophysical": 4135,
    "track2_unclassified_l5": 4365,
    "track3_detection_artifacts": 1500,
    "total_cohort": 10000,
    "quarantined_candidates": 79,
}

# Track 1 Exact Split Allocation
TRACK1_SPLIT_COUNTS = {
    "train": 2808,
    "val": 741,
    "test": 586,
    "total": 4135,
}

# Track 1 Spatially Held-Out Test Set Class Support
TRACK1_SPATIAL_TEST_SUPPORT = {
    "SN_Ia": 182,
    "SN_CC": 126,
    "AGN_QSO": 105,
    "Varstar": 68,
    "SLSN": 41,
    "CV_Nova": 38,
    "OTHER_EXTRAGALACTIC": 15,
    "TDE": 11,
    "total": 586,
}


class WilsonScoreInterval(BaseModel):
    """Wilson score 95% confidence interval for binomial proportions."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    successes: int
    trials: int


class ClassEvaluationWithCI(BaseModel):
    """Detailed class evaluation with support and 95% confidence intervals."""
    class_name: str
    support: int
    precision: float
    precision_ci_lower: float
    precision_ci_upper: float
    recall: float
    recall_ci_lower: float
    recall_ci_upper: float
    f1_score: float
    is_rare_class: bool = Field(description="True if support < 25 in test set")


def compute_wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Computes Wilson score interval for binomial proportions.
    Particularly critical for rare astronomical classes (e.g. TDE N=11, Other N=15)
    where normal approximation intervals fail.
    """
    if trials <= 0:
        return 0.0, 0.0, 0.0

    p = successes / trials
    # Two-sided standard normal quantile for 95% confidence
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else 1.96

    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / trials) + (z2 / (4.0 * trials * trials)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(p), float(lower), float(upper)


def verify_cohort_partition_invariants(candidates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Strictly verifies that the cohort DataFrame conforms to the frozen
    3-track evaluation architecture without label manufacture or leakage.
    """
    assert len(candidates_df) == 10000, f"Expected 10,000 candidates, got {len(candidates_df)}"

    # 1. Track 1: Labelled astrophysical classes
    labelled_mask = candidates_df["coarse_class"].isin(REAL_SURVEY_CLASSES_8)
    n_labelled = int(labelled_mask.sum())
    assert n_labelled == TRACK_COUNTS_COHORT_10K["track1_labelled_astrophysical"], (
        f"Track 1 count mismatch: {n_labelled} != 4135"
    )

    # 2. Track 2: Unclassified L5
    unclass_mask = candidates_df["coarse_class"] == "UNCLASSIFIED"
    n_unclass = int(unclass_mask.sum())
    assert n_unclass == TRACK_COUNTS_COHORT_10K["track2_unclassified_l5"], (
        f"Track 2 count mismatch: {n_unclass} != 4365"
    )

    # 3. Track 3: Detection artifacts
    artifact_mask = candidates_df["coarse_class"] == "DETECTION_ARTIFACT"
    n_artifacts = int(artifact_mask.sum())
    assert n_artifacts == TRACK_COUNTS_COHORT_10K["track3_detection_artifacts"], (
        f"Track 3 count mismatch: {n_artifacts} != 1500"
    )

    # 4. Quarantined count
    n_quarantined = int(candidates_df["is_quarantined"].sum())
    assert n_quarantined == TRACK_COUNTS_COHORT_10K["quarantined_candidates"], (
        f"Quarantine count mismatch: {n_quarantined} != 79"
    )

    # 5. Track 1 Spatial Test Support
    test_labelled = candidates_df[labelled_mask & (candidates_df["assigned_split"] == "test")]
    assert len(test_labelled) == TRACK1_SPLIT_COUNTS["test"], (
        f"Track 1 test split count mismatch: {len(test_labelled)} != 586"
    )

    test_counts = test_labelled["coarse_class"].value_counts().to_dict()
    for cls_name, expected_count in TRACK1_SPATIAL_TEST_SUPPORT.items():
        if cls_name == "total":
            continue
        actual_count = test_counts.get(cls_name, 0)
        assert actual_count == expected_count, (
            f"Test support mismatch for {cls_name}: {actual_count} != {expected_count}"
        )

    # 6. Archival Holdout AVAILABILITY_TIME_UNKNOWN Invariant
    archival_holdout = candidates_df[
        candidates_df["temporal_split"] == "archival_observation_date_holdout"
    ]
    unknown_avail = (archival_holdout["availability_status"] == "AVAILABILITY_TIME_UNKNOWN").all()
    assert unknown_avail, "Archival holdout contains non-UNKNOWN availability status!"

    return {
        "status": "PASS",
        "cohort_size": 10000,
        "track1_labelled_count": n_labelled,
        "track2_unclassified_l5_count": n_unclass,
        "track3_detection_artifacts_count": n_artifacts,
        "quarantined_count": n_quarantined,
        "spatial_test_support": test_counts,
        "archival_holdout_size": len(archival_holdout),
        "archival_unknown_availability_verified": True,
        "sprint3_training_governance": "STRICTLY_FROZEN",
    }
