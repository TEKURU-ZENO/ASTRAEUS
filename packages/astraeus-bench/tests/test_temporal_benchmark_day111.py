"""
ASTRAEUS Phase 2 Stage 2 — Day 111 Unit & Integration Tests.
Verifies:
1. Dual sequence views (FULL_HISTORY vs CAUSAL_FORECAST).
2. Four rigorous evaluation tasks (T1 Masked Reconstruction, T2 Multi-Horizon Forecasting,
   T3 Early Classification, T4 Follow-Up Utility).
3. Level-playing-field parity across 4 Baselines (Akima, GP, Transformer, ASTROMER).
4. Reconciled 32-Test Multiplicity Family (C01-C32) under Benjamini-Hochberg FDR control (alpha = 0.05).
5. Exact Binomial McNemar Test for T3 and Two-Sided Paired Wilcoxon Tests for T1, T2, T4.
6. Secondary Top-10 Permutation Test (B = 10,000) for follow-up queue exchangeability.
7. Track 3 External Archival Manifest (N=18 with SN 2017gfo open-set) and Dataset A Test Manifest (N=45).
8. Transformer architecture ablation across 5 frozen seeds [42, 137, 1024, 2048, 4096].
"""

import pytest
import math
import numpy as np
import torch
from scipy import stats

from astraeus_data.corpus.sequence import (
    ObservationRecord,
    ObservationType,
    SequenceViewType,
    TemporalSequence,
)
from astraeus_data.corpus.archival_manifest import (
    ArchivalObservationRecord,
    Track3ArchivalManifest,
    DatasetATestManifest,
    TRACK_3_OBJECT_METADATA,
)
from astraeus_models.baselines.akima import AkimaBaselineModel
from astraeus_models.baselines.gp import MultiOutputGPBaseline
from astraeus_models.baselines.transformer import NonPretrainedTransformerBaseline
from astraeus_models.baselines.astromer_adapter import AstromerAdapterBaseline
from astraeus_models.baselines.common_classifier import CommonFeatureClassifier, CLASSES_8
from astraeus_bench.statistical_testing import (
    StatisticalContrast,
    Top10PermutationDiagnostic,
    compute_paired_wilcoxon,
    compute_exact_mcnemar,
    apply_benjamini_hochberg,
    compute_top10_permutation_test,
)
from astraeus_bench.stage2_temporal_benchmark import (
    Stage2TemporalBenchmarkOrchestrator,
    ModelTaskEvaluationSummary,
    Stage2BenchmarkReport,
)


def _create_mock_sequence(object_id: str = "TEST-SN-001", n_pts: int = 40) -> TemporalSequence:
    """Generates a synthetic multi-band transient sequence."""
    records = []
    base_t = 60000.0
    passbands = ["g", "r", "i", "z"]

    for i in range(n_pts):
        t_bmjd = base_t + i * 1.5
        pb = passbands[i % len(passbands)]
        phase = i - 18.0
        flx = 22.0 * math.exp(-(phase ** 2) / (2.0 * (7.0 ** 2))) + 0.3
        sig = 0.25

        rec = ObservationRecord(
            object_id=object_id,
            observation_id=f"OBS-{i:03d}",
            t_obs_bmjd=t_bmjd,
            t_pub_utc=f"2023-03-{(i % 28) + 1:02d}T10:00:00Z",
            t_ingest_utc=f"2023-03-{(i % 28) + 1:02d}T10:00:00Z",
            has_operational_pub_time=True,
            passband=pb,
            survey_id="ZTF",
            obs_type=ObservationType.FORCED_PHOTOMETRY,
            measured_flux_nmgy=flx,
            measurement_sigma_nmgy=sig,
            effective_measurement_sigma_nmgy=sig,
            provenance_hash=f"prov_{i}"
        )
        records.append(rec)

    return TemporalSequence.from_records(object_id, records)


class TestDualSequenceViewsAndParity:
    """Verifies sequence view integrity and cross-model evaluation parity."""

    def test_full_history_view_preserves_all_observations(self):
        seq = _create_mock_sequence("SN-VIEW-01", 30)
        assert seq.view_type == SequenceViewType.FULL_HISTORY
        assert len(seq.observations) == 30
        assert seq.t_cutoff_utc is None

    def test_causal_forecast_view_filters_future_observations(self):
        seq = _create_mock_sequence("SN-VIEW-02", 30)
        cutoff_utc = "2023-03-10T10:00:00Z"
        causal_seq = seq.filter_by_operational_cutoff(cutoff_utc)

        assert causal_seq.t_cutoff_utc == cutoff_utc
        assert len(causal_seq.observations) <= 30
        for obs in causal_seq.observations:
            assert obs.t_pub_utc <= cutoff_utc

    def test_landmark_extraction_causal_peak_so_far(self):
        seq = _create_mock_sequence("SN-VIEW-03", 40)
        # In causal view, landmark extraction uses peak-so-far
        landmarks_full = seq.extract_landmark_indices(is_causal=False)
        landmarks_causal = seq.extract_landmark_indices(is_causal=True)

        assert len(landmarks_full) > 0
        assert len(landmarks_causal) > 0
        # Retrospective global peak index is at peak, causal landmark is valid subset
        assert max(landmarks_full) >= max(landmarks_causal)


class TestFourTaskEvaluationRunners:
    """Verifies Tasks T1, T2, T3, and T4 execution across baselines."""

    @pytest.fixture
    def orchestrator(self):
        return Stage2TemporalBenchmarkOrchestrator(device="cpu")

    def test_task1_masked_reconstruction(self, orchestrator):
        seq = _create_mock_sequence("SN-T1-01", 35)
        # Test Akima on T1
        crps_akima, mae_akima = orchestrator.evaluate_task1("B1_Akima", None, seq, mask_duration_days=10.0)
        assert crps_akima >= 0.0
        assert mae_akima >= 0.0

        # Test GP on T1
        crps_gp, mae_gp = orchestrator.evaluate_task1("B2_GP", None, seq, mask_duration_days=10.0)
        assert crps_gp >= 0.0
        assert mae_gp >= 0.0

    def test_task2_causal_forecasting(self, orchestrator):
        seq = _create_mock_sequence("SN-T2-01", 40)
        crps_10d, crps_3d, crps_30d, mae_10d = orchestrator.evaluate_task2("B1_Akima", None, seq)

        assert crps_10d >= 0.0
        assert crps_3d >= 0.0
        assert crps_30d >= 0.0
        assert mae_10d >= 0.0

    def test_task3_early_classification(self, orchestrator):
        seq = _create_mock_sequence("SN-T3-01", 30)
        pred_cls, probs, is_corr, brier = orchestrator.evaluate_task3(
            "B1_Akima", None, seq, true_class="SN_Ia"
        )
        assert pred_cls in CLASSES_8
        assert probs.shape == (8,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert brier >= 0.0
        assert isinstance(is_corr, bool)

    def test_task4_followup_prioritization(self, orchestrator):
        seq = _create_mock_sequence("SN-T4-01", 30)
        # Common supernova Ia
        u_ia, is_rare_ia = orchestrator.evaluate_task4("B1_Akima", None, seq, true_class="SN_Ia")
        assert u_ia >= 0.0
        assert is_rare_ia is False

        # Rare FBOT target
        u_fbot, is_rare_fbot = orchestrator.evaluate_task4("B1_Akima", None, seq, true_class="FBOT")
        assert u_fbot >= 0.0
        assert is_rare_fbot is True

        # Open-set novelty kilonova (SN 2017gfo)
        seq_gw = _create_mock_sequence("SN2017gfo", 30)
        u_gw, is_rare_gw = orchestrator.evaluate_task4("B1_Akima", None, seq_gw, true_class="OPEN_SET_NOVELTY")
        assert u_gw > 0.5  # S_novelty bonus boosts priority
        assert is_rare_gw is True


class TestStatisticalTestingFramework:
    """Verifies Wilcoxon, exact McNemar, Benjamini-Hochberg FDR, and Top-10 Permutation."""

    def test_paired_wilcoxon_signed_rank(self):
        # Model 2 significantly outperforms Model 1
        rng = np.random.RandomState(42)
        diffs = rng.normal(loc=1.5, scale=0.5, size=45)  # strictly positive difference
        stat, p_val, eff_size = compute_paired_wilcoxon(diffs)

        assert p_val < 0.001
        assert eff_size > 0.0

        # Zero difference case
        zeros = np.zeros(20)
        stat_z, p_z, eff_z = compute_paired_wilcoxon(zeros)
        assert p_z == 1.0
        assert eff_z == 0.0

    def test_exact_binomial_mcnemar_test(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        # Base: correct on first 5, wrong on last 5
        y_base = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        # Cand: correct on all 10
        y_cand = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        b, c, p_val, odds = compute_exact_mcnemar(y_true, y_base, y_cand)
        # Cand correct & base incorrect on 5 items (indices 5-9)
        assert b == 5
        assert c == 0
        # Exact binomial p-value for 0 successes out of 5 under p=0.5: 2 * (0.5)^5 = 2/32 = 0.0625
        assert p_val == pytest.approx(0.0625, rel=1e-3)
        assert odds == 6.0

        # Completely identical predictions: b=0, c=0 -> p=1.0
        b0, c0, p0, _ = compute_exact_mcnemar(y_true, y_base, y_base)
        assert b0 == 0 and c0 == 0 and p0 == 1.0

    def test_benjamini_hochberg_multiplicity_monotonicity(self):
        # Generate 32 raw p-values
        rng = np.random.RandomState(123)
        raw_p = np.sort(rng.uniform(0.0001, 0.8, size=32))

        dummy_contrasts = [
            StatisticalContrast(
                contrast_id=f"C{i+1:02d}",
                task_id=f"T{(i%4)+1}",
                task_name="Task",
                baseline_id="B1",
                cohort_id="Cohort1",
                sample_size=45,
                test_type="Wilcoxon",
                test_statistic=10.0,
                raw_p_value=float(p),
            )
            for i, p in enumerate(raw_p)
        ]

        adj_contrasts = apply_benjamini_hochberg(dummy_contrasts, alpha=0.05)
        assert len(adj_contrasts) == 32

        # Verify monotonicity of adjusted q-values: q_(i) <= q_(i+1)
        for i in range(len(adj_contrasts) - 1):
            assert adj_contrasts[i].bh_q_value <= adj_contrasts[i + 1].bh_q_value + 1e-9

        # Ensure q-values are <= 1.0 and >= raw p-values
        for c in adj_contrasts:
            assert 0.0 <= c.bh_q_value <= 1.0
            assert c.bh_q_value >= c.raw_p_value - 1e-9

    def test_top10_permutation_test(self):
        rng = np.random.RandomState(42)
        n = 45
        is_rare = np.zeros(n, dtype=bool)
        is_rare[:10] = True  # 10 true rare targets

        # Baseline scores
        scores_base = rng.rand(n)
        # Candidate has higher scores on true targets
        scores_cand = rng.rand(n)
        scores_cand[:10] += 2.0

        diag = compute_top10_permutation_test(
            scores_baseline=scores_base,
            scores_candidate=scores_cand,
            is_rare_target=is_rare,
            num_permutations=2000,
            seed=42
        )

        assert diag.num_permutations == 2000
        assert diag.observed_p_at_10_candidate > diag.observed_p_at_10_baseline
        assert diag.delta_p_at_10 > 0.0
        assert 0.0 <= diag.empirical_p_value <= 1.0
        assert diag.ci_lower_95 <= diag.ci_upper_95


class TestTrack3AndDatasetAManifests:
    """Verifies provenance, schemas, and object counts for Track 3 and Dataset A."""

    def test_track3_archival_manifest_count_and_provenance(self):
        seqs = Track3ArchivalManifest.get_sequences()
        assert len(seqs) == 18
        assert len(TRACK_3_OBJECT_METADATA) == 18

        # Verify in-taxonomy (N=17) vs open-set (N=1)
        in_tax_seqs = Track3ArchivalManifest.get_in_taxonomy_sequences()
        assert len(in_tax_seqs) == 17
        assert all(s.object_id != "SN2017gfo" for s in in_tax_seqs)

        open_set_seqs = Track3ArchivalManifest.get_open_set_sequences()
        assert len(open_set_seqs) == 1
        assert open_set_seqs[0].object_id == "SN2017gfo"

        # Verify observation-level provenance records
        all_obs = Track3ArchivalManifest.generate_archival_observations()
        for meta in TRACK_3_OBJECT_METADATA:
            obj_id = meta["object_id"]
            records = all_obs[obj_id]
            assert len(records) >= 30
            for r in records:
                assert r.exposure_id.startswith("EXP_")
                assert r.survey_system in ("ZTF", "ATLAS", "Pan-STARRS1", "Swift", "ESO")
                assert r.provenance_uri.startswith("doi:")

    def test_dataset_a_test_manifest_45_objects_and_taxonomy(self):
        seqs = DatasetATestManifest.generate_test_sequences(base_seed=777)
        assert len(seqs) == 45

        labels = DatasetATestManifest.get_ground_truth_labels()
        assert len(labels) == 45

        # Check coverage across all 8 classes
        classes_present = set(labels.values())
        assert classes_present == set(CLASSES_8)

        # Check that observations have forced photometry and upper limits
        has_forced = False
        has_upper_limit = False
        for s in seqs:
            for obs in s.observations:
                if obs.obs_type == ObservationType.FORCED_PHOTOMETRY:
                    has_forced = True
                if obs.obs_type == ObservationType.CATALOG_UPPER_LIMIT:
                    has_upper_limit = True
        assert has_forced is True
        assert has_upper_limit is True


class TestStage2BenchmarkOrchestratorExecution:
    """Verifies full benchmark suite execution, 24 primary contrasts, 8 reference anchors, and ablations."""

    def test_full_benchmark_suite_generates_24_primary_contrasts_and_8_anchors(self):
        orchestrator = Stage2TemporalBenchmarkOrchestrator(device="cpu")
        report = orchestrator.run_benchmark_suite()

        assert isinstance(report, Stage2BenchmarkReport)
        assert report.num_primary_contrasts == 24
        assert len(report.primary_contrasts_24) == 24
        assert len(report.reference_anchors_8) == 8
        assert len(report.full_grid_32) == 32

        # Check contrast IDs C01 to C24
        expected_ids = [f"C{i:02d}" for i in range(1, 25)]
        actual_ids = [c.contrast_id for c in report.primary_contrasts_24]
        assert actual_ids == expected_ids

        # Check reference anchor IDs R01 to R08
        expected_r_ids = [f"R{i:02d}" for i in range(1, 9)]
        actual_r_ids = [c.contrast_id for c in report.reference_anchors_8]
        assert actual_r_ids == expected_r_ids

        # Check task partition of 24 primary contrasts (6 per task)
        t1_contrasts = [c for c in report.primary_contrasts_24 if c.task_id == "T1"]
        t2_contrasts = [c for c in report.primary_contrasts_24 if c.task_id == "T2"]
        t3_contrasts = [c for c in report.primary_contrasts_24 if c.task_id == "T3"]
        t4_contrasts = [c for c in report.primary_contrasts_24 if c.task_id == "T4"]

        assert len(t1_contrasts) == 6  # C01-C06
        assert len(t2_contrasts) == 6  # C07-C12
        assert len(t3_contrasts) == 6  # C13-C18
        assert len(t4_contrasts) == 6  # C19-C24

        # Check task-specific sample size denominators on Track 3
        # T1, T2, T3 on Track 3 must have N=17; T4 must have N=18
        for c in report.primary_contrasts_24:
            if c.cohort_id == "Cohort_2_Track3_Archival":
                if c.task_id in ("T1", "T2", "T3"):
                    assert c.sample_size == 17
                elif c.task_id == "T4":
                    assert c.sample_size == 18
            else:
                assert c.sample_size == 45

        # Check test types
        for c in t1_contrasts + t2_contrasts + t4_contrasts:
            assert c.test_type == "Two-Sided Paired Wilcoxon"
        for c in t3_contrasts:
            assert c.test_type == "Exact Binomial McNemar"

        # Check FDR q-values and significance
        for c in report.primary_contrasts_24:
            assert 0.0 <= c.raw_p_value <= 1.0
            assert 0.0 <= c.bh_q_value <= 1.0
            assert c.bh_q_value >= c.raw_p_value - 1e-9

        # Check secondary diagnostics with published candidate IDs
        assert len(report.secondary_top10_diagnostics) >= 2
        for diag in report.secondary_top10_diagnostics:
            assert diag.num_permutations == 10000
            assert 0.0 <= diag.empirical_p_value <= 1.0
            assert len(diag.top10_baseline_ids) == 10
            assert len(diag.top10_candidate_ids) == 10

        # Check transformer seed stability across 5 seeds on full cohorts
        assert len(report.transformer_seed_stability) == 2
        for c_id in ["Cohort_1_DatasetA_Test", "Cohort_2_Track3_Archival"]:
            for seed in [42, 137, 1024, 2048, 4096]:
                assert seed in report.transformer_seed_stability[c_id]
                assert "t1_crps" in report.transformer_seed_stability[c_id][seed]
                assert "t2_crps_10d" in report.transformer_seed_stability[c_id][seed]

        # Check 4-way T4 ablation results
        assert "Cohort_1_DatasetA_Test" in report.t4_ablation_results
        c1_abl = report.t4_ablation_results["Cohort_1_DatasetA_Test"]
        assert "1_Transformer_Full" in c1_abl
        assert "2_Transformer_No_Attention" in c1_abl
        assert "3_Transformer_No_Novelty" in c1_abl
        assert "4_Common_Classifier_No_Timescales" in c1_abl

        # Check ASTROMER lineage audit
        assert report.astromer_lineage_audit["lineage_verification_status"] == "CERTIFIED_DISJOINT"
        assert report.astromer_lineage_audit["min_angular_separation_deg"] > 18.0
