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

        b, c, p_val, cohens_g, odds = compute_exact_mcnemar(y_true, y_base, y_cand)
        # Cand correct & base incorrect on 5 items (indices 5-9)
        assert b == 5
        assert c == 0
        # Exact binomial p-value for 0 successes out of 5 under p=0.5: 2 * (0.5)^5 = 2/32 = 0.0625
        assert p_val == pytest.approx(0.0625, rel=1e-3)
        assert cohens_g == 0.5
        assert odds is None  # When c=0, odds ratio is undefined/None

        # Completely identical predictions: b=0, c=0 -> p=1.0, Cohen's g=None, odds=None (not estimable)
        b0, c0, p0, g0, odds0 = compute_exact_mcnemar(y_true, y_base, y_base)
        assert b0 == 0 and c0 == 0 and p0 == 1.0 and g0 is None and odds0 is None

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
        assert diag.subtraction_order == "candidate_minus_baseline"
        assert diag.ci_method == "paired_object_bootstrap_2000"
        assert diag.observed_p_at_10_candidate > diag.observed_p_at_10_baseline
        assert diag.delta_p_at_10 > 0.0
        assert 0.0 <= diag.empirical_p_value <= 1.0
        assert diag.ci_lower_95 <= diag.delta_p_at_10 <= diag.ci_upper_95
        assert diag.null_ci_lower_95 <= 0.0 <= diag.null_ci_upper_95


class TestTrack3AndDatasetAManifests:
    """Verifies provenance, schemas, and object counts for Track 3 and Dataset A."""

    def test_track3_archival_manifest_count_and_provenance(self):
        seqs = Track3ArchivalManifest.get_sequences()
        assert len(seqs) == 18
        assert len(TRACK_3_OBJECT_METADATA) == 18

        # Verify uniqueness of canonical IDs (deduplication check)
        canonical_ids = [obj["canonical_id"] for obj in TRACK_3_OBJECT_METADATA]
        assert len(set(canonical_ids)) == 18, "Duplicate canonical ID found in Track 3 metadata!"

        # Verify AT2018cow verified coordinates and regression check
        cow = next(obj for obj in TRACK_3_OBJECT_METADATA if obj["canonical_id"] == "AT2018cow")
        assert abs(cow["ra_deg"] - 244.000917) < 1e-4, f"AT2018cow RA mismatch: {cow['ra_deg']}"
        assert abs(cow["dec_deg"] - 22.268031) < 1e-4, f"AT2018cow Dec mismatch: {cow['dec_deg']}"
        assert abs(cow["ra_deg"] - 244.5971) > 0.1, "Regression: erroneous coordinate 244.5971 re-entered AT2018cow"
        assert cow["host_galaxy"] == "CGCG 137-068"
        assert "SN2018cow" in cow["aliases"]

        # Verify AT2020xnd deduplication and verified ZTF20acigmel alias (The Camel)
        camel = next(obj for obj in TRACK_3_OBJECT_METADATA if obj["canonical_id"] == "AT2020xnd")
        assert "ZTF20acigmel" in camel["aliases"]
        assert "The Camel" in camel["aliases"]
        assert not any(obj["canonical_id"] == "ZTF20acvhrvo" for obj in TRACK_3_OBJECT_METADATA), "ZTF20acvhrvo must not be a canonical object!"
        assert "ZTF20acvhrvo" not in camel["aliases"]

        # Verify PTF12dam (SLSN) and AT2019dsg (TDE) replacements
        assert "PTF12dam" in canonical_ids
        assert "AT2019dsg" in canonical_ids
        assert "AT2021lwx" not in canonical_ids
        assert "Gaia14aaa" not in canonical_ids

        # Verify full 18-object astrometric provenance & uncertainty propagation audit
        prov_audit = Track3ArchivalManifest.audit_astrometric_provenance()
        assert prov_audit["passed"] is True, f"Astrometric provenance failures: {prov_audit['failures']}"
        assert prov_audit["total_records"] == 18
        assert prov_audit["check_scope"] == "AUTOMATED_MATHEMATICAL_AND_SCHEMA_CONSISTENCY"
        assert len(prov_audit["records"]) == 18
        for obj_diag in prov_audit["records"]:
            assert obj_diag["sep_err_arcsec"] >= 0.0
            assert obj_diag["projected_offset_err_kpc"] > 0.0
            assert obj_diag["coord_frame"] == "ICRS"
            assert obj_diag["transient_coord_source"]
            assert obj_diag["host_coord_source"]
            assert obj_diag["distance_source"]

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
                assert r.survey_system in (
                    "ZTF", "PTF", "ATLAS", "Pan-STARRS1", "Swift", "ESO", "ROTSE-III", "ASAS-SN", "Swope"
                )
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

        # Check secondary diagnostics with published candidate IDs & corrected bootstrap CIs across both cohorts
        assert len(report.secondary_top10_diagnostics) == 4
        diag_keys = {(d.cohort_id, d.baseline_id, d.candidate_id) for d in report.secondary_top10_diagnostics}
        assert ("Cohort_2_Track3_Archival", "B1_Akima", "B3_Transformer") in diag_keys
        for diag in report.secondary_top10_diagnostics:
            assert diag.num_permutations == 10000
            assert diag.subtraction_order == "candidate_minus_baseline"
            assert 0.0 <= diag.empirical_p_value <= 1.0
            assert diag.ci_lower_95 <= diag.delta_p_at_10 <= diag.ci_upper_95
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

        # Check paired bootstrap difference CIs
        assert "Cohort_1_DatasetA_Test" in report.t4_paired_difference_ci
        assert "Cohort_2_Track3_Archival" in report.t4_paired_difference_ci
        assert len(report.t4_paired_difference_ci["Cohort_2_Track3_Archival"]) == 2

        # Check ASTROMER lineage audit
        assert report.astromer_lineage_audit["lineage_verification_status"] == "OVERLAP_AUDIT_INCOMPLETE"
        assert report.astromer_lineage_audit["min_angular_separation_canonical_deg"] > 18.0
        assert "sn1987a_environmental_audit" in report.astromer_lineage_audit
        assert report.astromer_lineage_audit["sn1987a_environmental_audit"]["in_canonical_track3_manifest"] is False

        # Check Day 111.2.3 Evidence Closure invariants: T3 present-class Macro-F1, T4 class/open-set ranking,
        # retired OvR Platt default, post-hoc OOD abstention, and scoped p50/p95 operational triage latency
        t3_b1 = report.cohort_summaries["Cohort_2_Track3_Archival"]["B1_Akima"]
        assert len(t3_b1.t3_present_classes) == 6
        assert t3_b1.t3_present_class_macro_f1 >= t3_b1.t3_macro_f1
        assert t3_b1.t4_ranking_metrics["open_set_separation"]["contains_open_set_sn2017gfo"] is True
        assert t3_b1.t4_ranking_metrics["in_taxonomy_only"]["n_queue"] == 17
        assert t3_b1.t4_ranking_metrics["full_queue"]["n_queue"] == 18
        for req_metric in ("recall_at_5", "recall_at_10", "average_precision", "ndcg_at_10"):
            assert req_metric in t3_b1.t4_ranking_metrics["full_queue"]
            assert f"{req_metric}_ci_95" in t3_b1.t4_ranking_metrics["full_queue"]

        for c_id in ["Cohort_1_DatasetA_Test", "Cohort_2_Track3_Archival"]:
            for cfg_name, p10_val in report.t4_ablation_results[c_id].items():
                q_info = report.t4_ablation_queues[c_id][cfg_name]
                assert abs(q_info["recalculated_precision_at_10"] - p10_val) < 1e-9
                assert len(q_info["top10_candidate_ids"]) == 10

        assert "DatasetA_Val_N24" in report.calibration_diagnostics
        assert "Cohort_2_Track3_Archival_N17" in report.calibration_diagnostics
        cal_val = report.calibration_diagnostics["DatasetA_Val_N24"]
        assert cal_val["default_calibration_mode"] == "temperature_scaled"
        assert cal_val["ovr_platt_retired_from_default"] is True
        assert "ovr_platt_16param_legacy" in cal_val
        assert "per_class_reliability" in cal_val["global_temp_1param"]

        assert "ID_Test_vs_Track3_InTax_N17" in report.ood_abstention_diagnostics
        assert "auroc_feature_l2" in report.ood_abstention_diagnostics["ID_Test_vs_Track3_InTax_N17"]
        assert "ood_abstention_rate_at_val95" in report.ood_abstention_diagnostics["ID_Test_vs_Track3_InTax_N17"]

        rt = report.runtime_environment_and_latency
        assert "cpu_model" in rt
        assert "scoped_operational_triage_latency_p50_p95" in rt
        p50_b1 = rt["scoped_operational_triage_latency_p50_p95"]["B1_Akima"]["Cohort_1_DatasetA_Test"]
        assert p50_b1["repetitions"] == 50
        assert p50_b1["warmup"] == 5
        assert p50_b1["batch_size"] == 1
        assert p50_b1["p95_ms_per_seq"] >= p50_b1["p50_ms_per_seq"]
        assert "OVERLAP_AUDIT_INCOMPLETE" in report.executive_summary
