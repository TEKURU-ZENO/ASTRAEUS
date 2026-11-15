"""
Tests for ASTRAEUS Real-Survey Evaluation Protocol, Three-Track Invariants,
and Target Leakage Prohibition.
"""

import pytest
import numpy as np
import pandas as pd
from astraeus_bench.real_survey_benchmark_protocol import (
    REAL_SURVEY_CLASSES_8,
    TRACK_COUNTS_COHORT_10K,
    TRACK1_SPLIT_COUNTS,
    TRACK1_SPATIAL_TEST_SUPPORT,
    compute_wilson_score_interval,
    verify_cohort_partition_invariants,
)
from astraeus_models.baselines.common_classifier import CommonFeatureClassifier
from astraeus_knowledge.vector.qdrant_engine import AstraeusVectorEngine
from astraeus_data.storage import BackendMode


def test_wilson_score_interval_rare_classes():
    """
    Verifies Wilson score 95% confidence intervals on small sample sizes
    characteristic of rare test classes (e.g. TDE N=11, Other Extragalactic N=15).
    """
    # Test case: 9 correct out of 11 TDEs
    p, lo, hi = compute_wilson_score_interval(successes=9, trials=11, confidence=0.95)
    assert p == pytest.approx(9 / 11, abs=1e-4)
    assert 0.0 < lo < p
    assert p < hi < 1.0
    assert (hi - lo) > 0.25  # Wide confidence interval reflects small sample size uncertainty

    # Edge cases: 0 / 11 and 11 / 11
    p_zero, lo_zero, hi_zero = compute_wilson_score_interval(successes=0, trials=11)
    assert lo_zero == 0.0
    assert hi_zero > 0.0

    p_full, lo_full, hi_full = compute_wilson_score_interval(successes=11, trials=11)
    assert hi_full == 1.0
    assert lo_full < 1.0


def test_real_survey_cohort_invariants():
    """
    Verifies that the sealed 10k cohort Parquet conforms strictly to the
    frozen 3-track evaluation architecture without label manufacture.
    """
    candidates_path = "data/processed/real_survey_10k_v1/candidates_10k.parquet"
    candidates_df = pd.read_parquet(candidates_path)

    result = verify_cohort_partition_invariants(candidates_df)
    assert result["status"] == "PASS"
    assert result["track1_labelled_count"] == 4135
    assert result["track2_unclassified_l5_count"] == 4365
    assert result["track3_detection_artifacts_count"] == 1500
    assert result["quarantined_count"] == 79
    assert result["archival_unknown_availability_verified"] is True
    assert result["sprint3_training_governance"] == "STRICTLY_FROZEN"


def test_target_leakage_prohibition_in_vector_engine():
    """
    Mathematically verifies that modifying target fields (coarse_class, fine_class,
    evidence_tier, is_quarantined) results in zero difference in extracted vector embeddings.
    """
    engine = AstraeusVectorEngine(location=":memory:", vector_dim=24, backend_mode=BackendMode.LOCAL_EMBEDDED)

    row_sn = {
        "candidate_id": "TEST_CAND_001",
        "ra_deg": 185.0,
        "dec_deg": 15.0,
        "coarse_class": "SN_Ia",
        "fine_class": "SN Ia-norm",
        "evidence_tier": "L1",
        "is_quarantined": False,
        "n_alerce_det": 25,
        "n_alerce_ul": 10,
        "n_dr24_clean": 40,
        "first_mjd": 59100.0,
        "last_mjd": 59180.0,
        "has_msip_alert": True,
        "has_dr24_lc": True,
        "has_independent_tns": True,
        "host_counterpart_id": "HOST_CP_001",
        "healpix_nside32": 3500,
    }

    # Alter target/label fields completely
    row_altered = dict(
        row_sn,
        coarse_class="TDE",
        fine_class="TDE-H",
        evidence_tier="L5",
        is_quarantined=True,
    )

    vec1 = engine.extract_astronomical_features_24d(row_sn)
    vec2 = engine.extract_astronomical_features_24d(row_altered)

    diff = np.sum(np.abs(np.array(vec1) - np.array(vec2)))
    assert diff == pytest.approx(0.0, abs=1e-12), f"Target leakage detected! Vector difference = {diff}"


def test_classifier_real_survey_taxonomy_initialization():
    """
    Verifies that CommonFeatureClassifier supports initialization with
    the real-survey 8-class taxonomy.
    """
    clf = CommonFeatureClassifier(classes=REAL_SURVEY_CLASSES_8)
    assert clf.num_classes == 8
    assert clf.classes == REAL_SURVEY_CLASSES_8
    assert "SN_CC" in clf.classes
    assert "AGN_QSO" in clf.classes
    assert "Varstar" in clf.classes
    assert "OTHER_EXTRAGALACTIC" in clf.classes
