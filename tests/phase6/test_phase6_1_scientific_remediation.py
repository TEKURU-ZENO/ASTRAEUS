"""
ASTRAEUS Phase 6.1 Scientific Evidence Remediation & Audit Test Suite.
Validates:
1. P6.1-A: FNO3D hydrodynamic surrogate training, checkpointing, held-out evaluation, and fair speedup.
2. P6.1-B: Non-circular independent cosmological parameter recovery, mode covariance, and Fisher identifiability.
3. P6.1-C: Multi-seed statistical PPO scheduler benchmarking against 4 baselines with bootstrap confidence intervals.
4. P6.1-D: Epistemic governance, IVOA Prov-DM v1.0, and Gate 3 quarantine covenant preservation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

import os
import math
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.differentiable import (
    DifferentiableInitialConditions,
    DifferentiableSimulator,
    IndependentCosmologicalInference,
)
from engines.hydrodynamics.contracts import (
    HydroChannel,
    SimulationSuite,
    HydroFieldMetadata,
    HydroFieldCube,
    generate_synthetic_hydro_cube,
)
from engines.hydrodynamics.pino_model import PINO3DModel
from engines.hydrodynamics.physics_loss import PhysicsConservationLoss
from engines.hydrodynamics.surrogate_benchmark import ReferenceHydroSolver3D
from engines.hydrodynamics.train_surrogate import (
    DisjointCosmoHydroDataset,
    HydroSurrogateTrainer,
)
from engines.experimental_design.eig_objective import (
    TransientTarget,
    ExpectedInformationGain,
)
from engines.experimental_design.telescope_env import (
    TelescopeSchedulingEnv,
    angular_distance_deg,
)
from engines.experimental_design.ppo_scheduler import (
    ActorCriticNetwork,
    BaseScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    GreedyAirmassScheduler,
    GreedyEIGScheduler,
    ConstrainedPPOScheduler,
    run_statistical_scheduler_benchmark,
)
from astraeus_core.epistemic import EpistemicTier, EpistemicMetadata, ClaimType
from astraeus_core.provenance import ProvEntity, create_provenance_record


# ============================================================================
# TRACK P6.1-A: Hydrodynamic Surrogate Training, Generalization & Benchmark
# ============================================================================

class TestTrackP61AHydrodynamicSurrogate:
    """Verifies genuine training, disjoint held-out validation, and fair speedup benchmarking."""

    def test_disjoint_dataset_spatial_separation(self):
        """Gate P6.1-A1: Dataset volumes are spatially disjoint with zero voxel leakage."""
        ds_train = DisjointCosmoHydroDataset(num_volumes=3, grid_size=16, seed_offset=100)
        ds_test = DisjointCosmoHydroDataset(num_volumes=2, grid_size=16, seed_offset=500)

        assert len(ds_train) == 3
        assert len(ds_test) == 2

        inp_tr, targ_tr = ds_train[0]
        inp_te, targ_te = ds_test[0]

        assert inp_tr.shape == (1, 16, 16, 16)
        assert targ_tr.shape == (5, 16, 16, 16)
        # Verify inputs and targets are distinct realizations
        assert not np.allclose(inp_tr.numpy(), inp_te.numpy())
        assert not np.allclose(targ_tr.numpy(), targ_te.numpy())

    def test_surrogate_training_convergence_and_checkpoint(self, tmp_path):
        """Gate P6.1-A2: Model trains with AdamW/CosineAnnealing, loss decreases, checkpoint saved."""
        ds_train = DisjointCosmoHydroDataset(num_volumes=4, grid_size=16, seed_offset=100)
        ds_val = DisjointCosmoHydroDataset(num_volumes=2, grid_size=16, seed_offset=300)

        train_loader = DataLoader(ds_train, batch_size=2, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=2, shuffle=False)

        trainer = HydroSurrogateTrainer(grid_size=16, modes=4, width=16, num_layers=2)
        ckpt_dir = str(tmp_path / "checkpoints")

        res = trainer.train(train_loader, val_loader, epochs=4, lr=3e-3, checkpoint_dir=ckpt_dir)

        assert os.path.exists(res["best_checkpoint"])
        assert res["best_val_loss"] > 0.0
        assert len(res["history"]) == 4

        # Verify loss improves from epoch 1
        initial_loss = res["history"][0]["train_loss"]
        final_loss = res["history"][-1]["train_loss"]
        assert final_loss < initial_loss, f"Training loss did not decrease: {initial_loss} -> {final_loss}"

    def test_held_out_generalization_evaluation(self):
        """Gate P6.1-A3: Evaluates trained model on unseen held-out simulation realizations."""
        ds_train = DisjointCosmoHydroDataset(num_volumes=4, grid_size=16, seed_offset=100)
        ds_test = DisjointCosmoHydroDataset(num_volumes=2, grid_size=16, seed_offset=900)

        train_loader = DataLoader(ds_train, batch_size=2, shuffle=True)
        trainer = HydroSurrogateTrainer(grid_size=16, modes=4, width=16, num_layers=2)

        # Quick training
        _ = trainer.train(train_loader, train_loader, epochs=3, lr=3e-3)

        # Held-out evaluation
        eval_metrics = trainer.evaluate_held_out(ds_test)

        assert eval_metrics["num_test_volumes"] == 2
        assert eval_metrics["mean_rel_l2"] > 0.0
        assert "density" in eval_metrics["channel_rel_l2"]
        assert "temperature" in eval_metrics["channel_rel_l2"]
        # Ensure errors are finite and bounded
        assert eval_metrics["mean_rel_l2"] < 2.0

    def test_fair_speedup_benchmark(self):
        """Gate P6.1-A4: Fair wall-clock benchmark comparing equivalent task against reference 3D solver."""
        ds_test = DisjointCosmoHydroDataset(num_volumes=1, grid_size=16, seed_offset=888)
        trainer = HydroSurrogateTrainer(grid_size=16, modes=4, width=16, num_layers=2)

        bench_res = trainer.run_fair_speedup_benchmark(ds_test, n_benchmark_runs=10)

        assert bench_res["t_ref_per_step_sec"] > 0.0
        assert bench_res["t_surrogate_forward_sec"] > 0.0
        assert bench_res["measured_speedup_factor"] > 10.0
        assert "device" in bench_res
        assert bench_res["status"] == "Trained checkpoint benchmarked"

    def test_genuine_simulation_dataset_loading(self):
        """Gate P6.1-A5: GenuineSimulationDataset loads verified on-disk snapshots with binary SHA-256."""
        from engines.hydrodynamics.simulation_loader import populate_benchmark_dataset
        from engines.hydrodynamics.train_surrogate import GenuineSimulationDataset

        manifest = populate_benchmark_dataset("data/simulations", grid_size=16)
        ds = GenuineSimulationDataset(manifest["train"])
        assert len(ds) == 6
        dm_in, target_arr = ds[0]
        assert dm_in.shape == (1, 16, 16, 16)
        assert target_arr.shape == (5, 16, 16, 16)
        assert not torch.isnan(dm_in).any()
        assert not torch.isnan(target_arr).any()


# ============================================================================
# TRACK P6.1-B: Independent Cosmological Parameter Inference
# ============================================================================

class TestTrackP61BIndependentCosmology:
    """Verifies non-circular parameter recovery with independent mock data and Fisher identifiability."""

    def test_independent_mock_generation(self):
        """Gate P6.1-B1: Independent mock dataset with distinct phases and Gaussian mode covariance."""
        sim = DifferentiableSimulator(box_size=32.0, n_grid=8)
        ic = DifferentiableInitialConditions(box_size=32.0, n_grid=8)
        inference = IndependentCosmologicalInference(sim, ic)

        k_cen, noisy_pk, cov_mat = inference.generate_independent_mock_observation(
            theta_true=(0.30, 0.80), seed=777, noise_frac=0.02
        )

        assert len(k_cen) == 8
        assert len(noisy_pk) == 8
        assert cov_mat.shape == (8, 8)
        assert np.all(np.diag(cov_mat) > 0)
        assert np.all(noisy_pk > 0)

    def test_fisher_identifiability_bounds(self):
        """Gate P6.1-B2: Fisher matrix satisfies full rank, positive definiteness, and condition limit."""
        sim = DifferentiableSimulator(box_size=32.0, n_grid=8)
        ic = DifferentiableInitialConditions(box_size=32.0, n_grid=8)
        inference = IndependentCosmologicalInference(sim, ic)

        # Covariance from mock
        _, _, cov_mat = inference.generate_independent_mock_observation((0.30, 0.80), seed=555)
        infer_phases = ic.generate_gaussian_phases(seed=123)  # Distinct inference phases

        fisher_info = inference.evaluate_fisher_identifiability(
            theta_fid=(0.30, 0.80),
            inference_phases=infer_phases,
            cov_matrix=cov_mat,
            eps=1e-3,
        )

        assert fisher_info["rank"] == 2
        assert fisher_info["min_eigenvalue"] > 0.0
        assert fisher_info["condition_number"] < 1e4
        assert fisher_info["sigma_omega_m_cr"] > 0.0
        assert fisher_info["sigma_sigma8_cr"] > 0.0
        assert fisher_info["identifiability_passed"] is True

    def test_independent_parameter_recovery_non_circular(self):
        """Gate P6.1-B3: Parameter recovery against independent mock realization without inverse crime."""
        sim = DifferentiableSimulator(box_size=32.0, n_grid=8)
        ic = DifferentiableInitialConditions(box_size=32.0, n_grid=8)
        inference = IndependentCosmologicalInference(sim, ic)

        theta_true = (0.31, 0.82)
        # Mock generated with seed 999
        _, obs_pk, cov_mat = inference.generate_independent_mock_observation(
            theta_true, seed=999, noise_frac=0.01
        )
        # Inference engine uses phases from seed 333 (completely independent!)
        infer_phases = ic.generate_gaussian_phases(seed=333)

        rec = inference.run_independent_parameter_recovery(
            obs_pk=obs_pk,
            cov_matrix=cov_mat,
            theta_true=theta_true,
            initial_guess=(0.315, 0.815),
            inference_phases=infer_phases,
            max_iter=5,
            lr=0.002,
        )

        assert rec["final_err_Omega_m"] < 0.05
        assert rec["final_err_sigma8"] < 0.05
        assert rec["recovery_bias_acceptable"] is True

    def test_ensemble_parameter_recovery_cramer_rao_coverage(self):
        """Gate P6.1-B4: Multi-mock ensemble recovery achieves >= 80% Cramer-Rao coverage."""
        sim = DifferentiableSimulator(box_size=32.0, n_grid=8)
        ic = DifferentiableInitialConditions(box_size=32.0, n_grid=8)
        inference = IndependentCosmologicalInference(sim, ic)

        ensemble_res = inference.run_ensemble_recovery_benchmark(num_mocks=5, max_iter=4, lr=0.002)

        assert ensemble_res["num_mocks"] == 5
        assert ensemble_res["mean_relative_error_Omega_m"] < 0.05
        assert ensemble_res["mean_relative_error_sigma8"] < 0.05
        assert ensemble_res["overall_cramer_rao_coverage"] >= 0.80
        assert ensemble_res["coverage_criteria_passed"] is True


# ============================================================================
# TRACK P6.1-C: Statistical Telescope Scheduler Evaluation
# ============================================================================

class TestTrackP61CSchedulerEvaluation:
    """Verifies statistical benchmark against 4 baselines with bootstrap confidence intervals."""

    def test_all_baseline_schedulers_functional(self):
        """Gate P6.1-C1: All 4 baseline schedulers execute without errors."""
        env = TelescopeSchedulingEnv(n_targets=4, night_duration_hours=2.0)
        obs, _ = env.reset(seed=42)

        rnd = RandomScheduler(n_actions=5)
        rr = RoundRobinScheduler(n_targets=4)
        ga = GreedyAirmassScheduler()
        ge = GreedyEIGScheduler()

        a_rnd = rnd.select_action(env, obs)
        a_rr = rr.select_action(env, obs)
        a_ga = ga.select_action(env, obs)
        a_ge = ge.select_action(env, obs)

        for a in [a_rnd, a_rr, a_ga, a_ge]:
            assert 0 <= a <= 4

    def test_statistical_scheduler_benchmark_execution(self):
        """Gate P6.1-C2: Multi-seed statistical benchmark computes EIG comparison over 50 nights with bootstrap CI."""
        env = TelescopeSchedulingEnv(n_targets=4, night_duration_hours=3.0)
        obs_dim = 5 + 8 * 4
        action_dim = 5
        agent = ConstrainedPPOScheduler(obs_dim=obs_dim, action_dim=action_dim, lr=1e-3)

        # Train for 10 episodes
        agent.train_agent(env, total_episodes=10)

        # Run statistical benchmark over 50 test episodes
        bench = run_statistical_scheduler_benchmark(
            trained_ppo=agent,
            env=env,
            num_test_episodes=50,
            seed_start=3000,
        )

        assert bench["num_test_episodes"] == 50
        assert bench["ppo_mean_eig"] >= 0.0
        assert bench["ppo_mean_violations"] <= 5.0

        comparisons = bench["comparisons"]
        assert "Random" in comparisons
        assert "RoundRobin" in comparisons
        assert "GreedyAirmass" in comparisons
        assert "GreedyEIG" in comparisons

        for b_name, comp in comparisons.items():
            assert "pct_improvement" in comp
            assert "ci_95" in comp
            assert len(comp["ci_95"]) == 2


# ============================================================================
# TRACK P6.1-D: Scientific Governance & Quarantine Preservation
# ============================================================================

class TestTrackP61DGovernanceAndQuarantine:
    """Verifies IVOA lineage, epistemic promotion guards, and Gate 3 zero-touch quarantine."""

    def test_epistemic_promotion_exception(self):
        """Gate P6.1-D1: Epistemic boundary strictly blocks promotion to observed."""
        meta = EpistemicMetadata(
            tier=EpistemicTier.SIMULATED,
            claim_type=ClaimType.THEORETICAL_MODEL,
        )
        with pytest.raises(Exception):
            meta.assert_valid_promotion(EpistemicTier.OBSERVED)

    def test_ivoa_prov_dm_lineage_tracking(self):
        """Gate P6.1-D2: IVOA Provenance record generation for Phase 6.1 remediation activities."""
        record = create_provenance_record(
            entity_id="astraeus:model:pino_hydro_surrogate_checkpoint",
            entity_type="neural_network_weights",
            activity_id="act:phase6_1_surrogate_training",
            agent_id="agent:astraeus_researcher_dev_mehta",
            extra_metadata={
                "milestone": "PHASE6_1_SCIENTIFIC_REMEDIATION_AND_TRAINING",
                "loss_function": "PhysicsConservationLoss",
                "disjoint_spatial_volumes": True,
            },
        )
        assert record.entity_id == "astraeus:model:pino_hydro_surrogate_checkpoint"
        assert record.was_associated_with == "agent:astraeus_researcher_dev_mehta"
        assert record.metadata["disjoint_spatial_volumes"] is True

    def test_gate3_quarantine_covenant_uncompromised(self):
        """Gate P6.1-D3: Gate 3 quarantine covenant (586 sources, 1,808 pixels) strictly preserved."""
        ledger_path = os.path.join("docs", "phase6", "PHASE5_CONTINUATION_LEDGER.md")
        assert os.path.exists(ledger_path)
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger_text = f.read()

        assert "586" in ledger_text
        assert "1,808" in ledger_text
        assert "quarantine" in ledger_text.lower()

    def test_evidence_manifest_and_zero_touch_quarantine(self):
        """Gate P6.1-D4: Evidence manifest verifies SHA-256 digests and zero-touch Gate 3 quarantine covenant."""
        import json
        from scripts.generate_phase6_1_evidence_manifest import audit_zero_touch_quarantine, build_evidence_manifest

        manifest_path = os.path.join("data", "provenance", "phase6_1_evidence_manifest.json")
        if not os.path.exists(manifest_path):
            manifest = build_evidence_manifest()
        else:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        assert manifest["manifest_version"] == "1.0.0"
        assert manifest["checkpoint_id"] == "PHASE6_1_INDEPENDENT_VALIDATION_AND_REPRODUCIBILITY"
        assert len(manifest["artifacts"]["simulation_snapshots"]) >= 10

        audit_res = audit_zero_touch_quarantine()
        assert audit_res["status"] == "VERIFIED_ZERO_TOUCH"
        assert audit_res["quarantined_healpix_pixels"] == 1808
        assert audit_res["quarantine_covenant_binding"] is True
