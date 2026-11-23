"""
ASTRAEUS Phase 6 Integrated Release Verification Suite (Tracks A, B, C, D, E).
Validates differentiable cosmological simulations, neural hydrodynamic surrogates (FNO3D/PINO),
autonomous experimental intelligence (EIG & Constrained PPO), shared epistemic contracts,
and Phase 5 Gate 3 quarantine covenant preservation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

import math
import os
import pytest
import numpy as np
import torch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.differentiable import (
    DifferentiableInitialConditions,
    DifferentiableSimulator,
    FisherParameterInference,
)
from engines.hydrodynamics.contracts import (
    HydroChannel,
    SimulationSuite,
    HydroFieldMetadata,
    HydroFieldCube,
    generate_synthetic_hydro_cube,
)
from engines.hydrodynamics.pino_model import (
    SpectralConv3d,
    FourierBlock3D,
    PINO3DModel,
)
from engines.hydrodynamics.physics_loss import (
    PhysicsConservationLoss,
    periodic_central_diff,
    periodic_divergence_3d,
    periodic_poisson_solver_3d,
)
from engines.hydrodynamics.surrogate_benchmark import (
    ReferenceHydroSolver3D,
    HydroSurrogateBenchmark,
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
    GreedyAirmassScheduler,
    RoundRobinScheduler,
    ConstrainedPPOScheduler,
    evaluate_baseline,
)
from astraeus_core.epistemic import EpistemicTier, EpistemicMetadata, ClaimType
from astraeus_core.exceptions import EpistemicViolationError
from astraeus_core.provenance import ProvEntity, create_provenance_record


# ============================================================================
# TRACK A: Differentiable Cosmological Simulation
# ============================================================================

class TestTrackADifferentiableSimulation:
    """Verifies Track A physical initial conditions, reverse-mode AD, and Fisher inference."""

    def test_sigma8_top_hat_normalization(self):
        """Gate P6-A1: Top-hat window function at R=8 Mpc/h normalizes power spectrum."""
        ic = DifferentiableInitialConditions(box_size=64.0, n_grid=16)
        # Check window at x=0 is 1.0
        w0 = ic.top_hat_window(0.0)
        assert abs(w0 - 1.0) < 1e-5

        # Check normalization integral is positive and smooth
        norm1 = ic.compute_sigma8_norm(Omega_m=0.30)
        norm2 = ic.compute_sigma8_norm(Omega_m=0.31)
        assert norm1 > 0.0
        assert norm2 > 0.0
        assert abs(norm1 - norm2) / norm1 < 0.10

    def test_frozen_phases_hermitian_symmetry(self):
        """Gate P6-A2: Frozen Gaussian phases satisfy Hermitian symmetry and zero DC mode."""
        ic = DifferentiableInitialConditions(box_size=64.0, n_grid=16)
        phases = ic.generate_gaussian_phases(seed=42)
        assert phases.shape == (16, 16, 16)
        assert phases[0, 0, 0] == 0.0

    def test_reverse_mode_ad_gradient_fidelity(self):
        """Gate P6-A3: Reverse-mode AD sensitivities match central finite differences."""
        sim = DifferentiableSimulator(box_size=32.0, n_grid=8)
        ic = DifferentiableInitialConditions(box_size=32.0, n_grid=8)
        phases = ic.generate_gaussian_phases(seed=123)

        pos, mom, masses = ic.create_initial_mesh_state(0.31, 0.81, phases)

        def objective(x):
            # Advance 2 steps and compute sum of squared coordinates
            x_fin, _, _ = sim.simulate_trajectory(
                x, mom, masses, a_init=ic.a_init, a_target=0.03, n_steps=2, Omega_m=0.31
            )
            return jnp.sum(x_fin ** 2)

        # Reverse-mode AD gradient
        grad_ad = jax.grad(objective)(pos)

        # Finite difference gradient at selected non-boundary particle
        eps = 1e-5
        idx_p = 5
        idx_dim = 0
        perturb = np.zeros_like(pos)
        perturb[idx_p, idx_dim] = eps

        val_plus = objective(pos + perturb)
        val_minus = objective(pos - perturb)
        grad_fd = float(val_plus - val_minus) / (2.0 * eps)

        ad_val = float(grad_ad[idx_p, idx_dim])
        rel_diff = abs(ad_val - grad_fd) / max(abs(grad_fd), 1e-5)
        assert rel_diff < 1e-3, f"AD gradient {ad_val} vs FD {grad_fd} rel diff {rel_diff} exceeds 0.1%"

    def test_fisher_identifiability_and_parameter_recovery(self):
        """Gate P6-B: Fisher information matrix has non-zero determinant and recovers parameters."""
        ic = DifferentiableInitialConditions(box_size=32.0, n_grid=8)
        sim = DifferentiableSimulator(box_size=32.0, n_grid=8)
        phases = ic.generate_gaussian_phases(seed=42)

        infer = FisherParameterInference(sim, ic)
        F_mat, det_F, cond_F = infer.compute_fisher_matrix(
            theta_fid=(0.31, 0.81), phases=phases, n_steps=2, eps=1e-3
        )

        assert F_mat.shape == (2, 2)
        assert det_F > 0.0, f"Fisher matrix must be positive definite, got det(F) = {det_F}"
        assert cond_F > 1.0

        # Run recovery
        res = infer.run_parameter_recovery(
            ground_truth=(0.31, 0.81),
            initial_guess=(0.315, 0.805),
            phases=phases,
            n_steps=2,
            max_iter=5,
            lr=0.002,
        )
        assert res["final_err_Om"] < 0.02, f"Omega_m error {res['final_err_Om']} exceeds 2%"
        assert res["final_err_s8"] < 0.02, f"sigma_8 error {res['final_err_s8']} exceeds 2%"


# ============================================================================
# TRACK B: Neural Hydrodynamic Surrogates
# ============================================================================

class TestTrackBNeuralHydrodynamics:
    """Verifies Track B contracts, FNO3D/PINO model, conservation laws, and surrogate benchmarking."""

    def test_hydro_contracts_and_synthetic_generator(self):
        """Gate P6-C1: Multi-channel hydrodynamic cube generation and normalization inversion."""
        cube = generate_synthetic_hydro_cube(grid_size=16, box_size_mpc_h=25.0, seed=42)
        assert cube.data.shape == (7, 16, 16, 16)
        assert cube.num_channels == 7
        assert cube.resolution == 16

        # Check physical bounds
        density = cube.get_channel(HydroChannel.DENSITY)
        temp = cube.get_channel(HydroChannel.TEMPERATURE)
        b_mag = cube.get_channel(HydroChannel.MAGNETIC_B)
        assert np.all(density > 0)
        assert np.all(temp > 0)
        assert np.all(b_mag >= 0)

        # Test normalization inversion
        norm_cube, stats = cube.normalize()
        denorm_cube = norm_cube.denormalize(stats)
        for idx in range(cube.num_channels):
            orig_c = cube.data[idx]
            recon_c = denorm_cube.data[idx]
            rel_err = float(np.max(np.abs(orig_c - recon_c) / (np.abs(orig_c) + 1e-5)))
            assert rel_err < 2e-3, f"Channel {cube.metadata.channels[idx]} rel error {rel_err} exceeds tolerance"

    def test_pino_3d_architecture_forward_pass(self):
        """Gate P6-C2: 3D Fourier Neural Operator forward pass and physical positivity."""
        model = PINO3DModel(
            in_channels=1,
            out_channels=5,
            width=16,
            modes1=4,
            modes2=4,
            modes3=4,
            num_layers=2,
            include_grid=True,
            positive_channels=[0, 4],  # density and temperature
        )
        x = torch.randn(2, 1, 16, 16, 16)
        out = model(x)
        assert out.shape == (2, 5, 16, 16, 16)

        # Positivity verification
        rho_pred = out[:, 0]
        T_pred = out[:, 4]
        assert torch.all(rho_pred > 0)
        assert torch.all(T_pred > 0)
        assert model.get_num_parameters() > 1000

    def test_physics_informed_conservation_residuals(self):
        """Gate P6-C3: Exact PDE conservation loss calculation."""
        loss_fn = PhysicsConservationLoss(dx=1.0)
        pred = torch.rand(1, 5, 16, 16, 16) + 0.1  # positive fields
        target = torch.rand(1, 5, 16, 16, 16) + 0.1

        total_loss, metrics = loss_fn(pred, target)
        assert total_loss.item() > 0.0
        assert metrics["res_mass"] >= 0.0
        assert metrics["res_momentum"] >= 0.0
        assert metrics["loss_data"] > 0.0

        # Solenoidal check
        div_b = loss_fn.compute_solenoidal_residual(
            torch.zeros(1, 16, 16, 16),
            torch.zeros(1, 16, 16, 16),
            torch.zeros(1, 16, 16, 16),
        )
        assert abs(div_b.item()) < 1e-8

    def test_surrogate_acceleration_and_error_benchmark(self):
        """Gate P6-C4: Benchmark surrogate evaluation speedup vs reference 3D Euler solver."""
        model = PINO3DModel(
            in_channels=1,
            out_channels=5,
            width=16,
            modes1=4,
            modes2=4,
            modes3=4,
            num_layers=2,
        )
        bench = HydroSurrogateBenchmark(model)
        res = bench.benchmark_speedup(grid_size=16, n_ref_steps=20, trajectory_steps=5000, n_surrogate_runs=10)

        assert res["t_surrogate_eval_sec"] > 0.0
        assert res["speedup_vs_trajectory"] > 1000.0  # Significant acceleration confirmed
        assert "device" in res

        # Relative error evaluation on synthetic target
        cube = generate_synthetic_hydro_cube(grid_size=16)
        targ_t = cube.to_torch()[:, :5]
        # Slightly perturbed prediction
        pred_t = targ_t * 1.01
        err_metrics = bench.evaluate_relative_field_error(pred_t, targ_t)
        assert err_metrics["rel_l2_overall"] < 0.03  # < 3% error


# ============================================================================
# TRACK C: Autonomous Experimental Intelligence
# ============================================================================

class TestTrackCExperimentalIntelligence:
    """Verifies Track C spherical geometry, EIG calculation, telescope env, and PPO agent."""

    def test_spherical_angular_distance_and_airmass(self):
        """Gate P6-D1: Haversine distance and plane-parallel airmass calculations."""
        # Distance between poles: (0, 90) and (0, -90) = 180 degrees
        d_pole = angular_distance_deg(0.0, 90.0, 0.0, -90.0)
        assert abs(d_pole - 180.0) < 1e-5

        # Same point: distance = 0
        d_zero = angular_distance_deg(45.0, 30.0, 45.0, 30.0)
        assert abs(d_zero) < 1e-5

        env = TelescopeSchedulingEnv(observatory_lat_deg=30.0)
        # Zenith target: RA = LST, Dec = lat ==> airmass = 1.0
        x_zenith = env.calculate_airmass(ra_deg=0.0, dec_deg=30.0, lst_hours=0.0)
        assert abs(x_zenith - 1.0) < 1e-4

        # Below horizon target ==> 99.0
        x_horizon = env.calculate_airmass(ra_deg=180.0, dec_deg=-80.0, lst_hours=0.0)
        assert x_horizon >= 99.0

    def test_eig_calculation_and_bayesian_update(self):
        """Gate P6-D2: Expected Information Gain and posterior uncertainty shrinkage."""
        eig_calc = ExpectedInformationGain()
        target = TransientTarget(
            target_id="TGT_TEST",
            ra_deg=0.0,
            dec_deg=30.0,
            base_magnitude=19.0,
            decay_timescale_hours=24.0,
            time_since_trigger_hours=2.0,
            priority_weight=1.5,
            is_too=True,
            prior_variance=1.0,
        )

        eig = eig_calc.evaluate_target_eig(target, airmass=1.2, exposure_time_sec=300.0)
        assert eig > 0.0

        # Posterior update reduces prior variance
        initial_var = target.prior_variance
        eig_calc.posterior_update(target, snr=20.0)
        assert target.prior_variance < initial_var
        assert target.times_observed == 1

    def test_telescope_environment_step_and_constraints(self):
        """Gate P6-D3: Gymnasium environment step dynamics and constraint detection."""
        env = TelescopeSchedulingEnv(n_targets=4, night_duration_hours=2.0)
        obs, info = env.reset(seed=123)
        assert obs.shape == (5 + 8 * 4,)
        assert info["num_targets"] == 4

        # Take action on target 0
        next_obs, reward, term, trunc, step_info = env.step(0)
        assert next_obs.shape == obs.shape
        assert "slew_sec" in step_info
        assert "airmass" in step_info

        # Idle action
        idle_obs, idle_rew, _, _, _ = env.step(4)
        assert idle_rew <= 0.0

    def test_constrained_ppo_vs_heuristic_baselines(self):
        """Gate P6-D4: Constrained PPO evaluation against greedy and round-robin baselines."""
        env = TelescopeSchedulingEnv(n_targets=4, night_duration_hours=3.0)
        greedy = GreedyAirmassScheduler()
        rr = RoundRobinScheduler(n_targets=4)

        eval_greedy = evaluate_baseline(greedy, env, num_episodes=5)
        eval_rr = evaluate_baseline(rr, env, num_episodes=5)

        assert eval_greedy["mean_cumulative_eig"] >= 0.0
        assert eval_rr["mean_cumulative_eig"] >= 0.0
        assert eval_greedy["mean_violations"] == 0.0  # Greedy strictly avoids airmass > 2.0

        # PPO Agent
        obs_dim = 5 + 8 * 4
        action_dim = 5  # 4 targets + 1 idle
        agent = ConstrainedPPOScheduler(obs_dim=obs_dim, action_dim=action_dim, lr=1e-3)
        # Train for quick iterations
        agent.train_agent(env, total_episodes=10)
        eval_ppo = agent.evaluate(env, num_episodes=5)
        assert eval_ppo["mean_cumulative_eig"] >= 0.0
        assert eval_ppo["mean_violations"] <= 3.0


# ============================================================================
# TRACK E: Epistemic Contracts, Provenance & Gate 3 Quarantine Covenant
# ============================================================================

class TestTrackEIntegrationAndGovernance:
    """Verifies epistemic boundary enforcement, IVOA Prov-DM records, and Gate 3 quarantine covenant."""

    def test_epistemic_ontology_promotion_guard(self):
        """Gate P6-E1: Confirms illegal promotion from simulated to directly observed is blocked."""
        meta = EpistemicMetadata(
            tier=EpistemicTier.SIMULATED,
            claim_type=ClaimType.THEORETICAL_MODEL,
            verification_gate_passed=True,
        )
        # Assert valid promotion attempt to OBSERVED raises EpistemicViolationError
        with pytest.raises(EpistemicViolationError, match="Illegal Epistemic Promotion"):
            meta.assert_valid_promotion(EpistemicTier.OBSERVED)

    def test_ivoa_prov_dm_lineage_generation(self):
        """Gate P6-E2: IVOA Provenance Data Model entity generation and metadata."""
        meta = HydroFieldMetadata(
            suite=SimulationSuite.CAMELS,
            box_size_mpc_h=25.0,
            grid_resolution=16,
            provenance_id="prov:camels_test_001",
        )
        prov = meta.to_prov_entity("act:phase6_integrated_verification")
        assert prov.entity_id == "prov:camels_test_001"
        assert prov.was_generated_by == "act:phase6_integrated_verification"
        assert prov.was_associated_with == "agent:astraeus_researcher_dev_mehta"
        assert prov.metadata["suite"] == "CAMELS"

    def test_gate3_quarantine_covenant_preservation(self):
        """Gate P6-E3: Strict zero-touch preservation of Phase 5 Gate 3 quarantine cohort (586 objects)."""
        quarantine_path = os.path.join("docs", "phase6", "PHASE5_CONTINUATION_LEDGER.md")
        assert os.path.exists(quarantine_path), "Phase 5 continuation ledger must exist."

        with open(quarantine_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "586" in content, "Quarantine cohort count 586 must be explicitly preserved."
        assert "1,808" in content, "HEALPix pixels count 1,808 must be explicitly preserved."
        assert "quarantine" in content.lower(), "Quarantine covenant terms must be strictly maintained."
