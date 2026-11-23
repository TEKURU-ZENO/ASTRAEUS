# ASTRAEUS Phase 6.1: Scientific Evidence Correction & Remediation Addendum

**Document Identifier:** `ASTRAEUS-P6.1-CORRECTION-001`  
**Classification:** Scientific Governance, Source-Level Audit & Evidence Rectification  
**Version:** 1.0.0 (Official Addendum)  
**Author:** Dev Mehta (Lead Investigator & Mission Director, Project ASTRAEUS)  
**Date:** November 23, 2026  
**Parent Checkpoint:** `21d211d0195600634652e021b240fdc2fa79dd4e`  
**Target Milestone:** `PHASE6_1_SCIENTIFIC_REMEDIATION_AND_TRAINING`  
**Branch:** `phase6.1-scientific-remediation`  

---

## 1. Executive Statement of Correction

Following an exhaustive source-level scientific audit of commit `21d211d`, Project ASTRAEUS formally issues this **Scientific Evidence Correction Addendum**. 

A critical epistemic distinction must be maintained: **Phase 6 delivered functional software components, mathematical operators, and architectural prototypes, but it did NOT deliver trained scientific models, independent parameter inference, or validated empirical discoveries.**

Presenting software execution as empirical scientific closure violates the core epistemic covenants of Project ASTRAEUS. Accordingly:
1. **The engineering foundations implemented in `21d211d` are frozen and preserved.**
2. **All unearned scientific completion claims are formally WITHDRAWN.**
3. **Phase 6 scientific closure is REOPENED and gated on Phase 6.1 (`PHASE6_1_SCIENTIFIC_REMEDIATION_AND_TRAINING`).**
4. **Transition to Phase 7 is DEFERRED until Phase 6.1 achieves independent scientific verification.**

---

## 2. Source-Level Audit Findings & Claim Rectifications

### 2.1 Track A: Cosmological Parameter Inference — Inverse Crime & Numerical Derivatives
* **Audit Finding A1 (Inverse Crime in Parameter Recovery):** `FisherParameterInference.run_parameter_recovery()` generated its reference observation using the exact same simulator:
  $$P_{\mathrm{obs}}(k) = P_{\mathrm{sim}}(k \mid \theta_{\mathrm{true}})$$
  The optimizer then minimized $\|P_{\mathrm{sim}}(k \mid \theta) - P_{\mathrm{obs}}(k)\|^2$ against its own internal output. This is a classic textbook **inverse crime**: the forward model merely recognized its own synthetic signal. This does not demonstrate cosmological parameter inference or physical identifiability.
* **Audit Finding A2 (AD vs. Finite Differences):** The reverse-mode automatic differentiation (AD) test in `test_phase6_integrated_release.py` differentiated final particle coordinates with respect to initial particle positions. However, parameter sensitivities with respect to $(\Omega_m, \sigma_8)$ were computed via central finite differences. Describing parameter sensitivities as "reverse-mode AD verified" was inaccurate.
* **Audit Finding A3 (Fisher Identifiability Criteria):** The assertion that $\operatorname{det}(F) > 0$ and condition number $\kappa(F) > 1$ established cosmological identifiability was scientifically invalid. Non-zero determinant and condition number $> 1$ are trivial mathematical properties of any non-singular 2x2 matrix; they do not establish empirical parameter degeneracy breaking or observational identifiability.
* **Rectification:**
  - **RETAIN:** 3D particle-mesh dynamics, JAX float64 parity, and reverse-mode coordinate AD test.
  - **WITHDRAW:** Claims of demonstrated cosmological parameter recovery and Fisher identifiability.
  - **REOPEN:** Parameter inference under an independent forward model with realistic cosmic variance covariance.

---

### 2.2 Track B: Hydrodynamic Surrogate — Untrained Network & Benchmarking Deficiencies
* **Audit Finding B1 (Untrained Architecture Prototype):** `engines/hydrodynamics/pino_model.py` defined an FNO3D/PINO neural network architecture. However, no trained checkpoint, training dataset, optimizer history, or loss convergence curve was produced or committed. An untrained model's forward pass is not a validated hydrodynamic surrogate.
* **Audit Finding B2 (Methodologically Invalid Speedup Benchmark):** `surrogate_benchmark.py` evaluated an untrained network with random weights on a random Gaussian dummy tensor. It measured the latency of a single forward pass and compared it against an extrapolated 5,000-step trajectory of a 3D Euler solver. Comparing one forward pass of an untrained network to an extrapolated multi-thousand-step numerical integration does not establish a $1,162\times$ acceleration of an equivalent scientific workload.
* **Audit Finding B3 (Artificial Field Error Construction):** In `test_phase6_integrated_release.py`, the field error test constructed:
  $$\mathrm{pred\_t} = \mathrm{targ\_t} \times 1.01$$
  It merely measured the arithmetic difference between an array and its own $1\%$ scalar multiple. The neural network was never evaluated against a held-out hydrodynamic simulation target. Reporting $1.0\% - 1.4\%$ relative field error was completely unearned.
* **Audit Finding B4 (Scaffolding vs. Real Data):** The synthetic cube generator was a log-normal proxy, not genuine CAMELS or IllustrisTNG simulation data.
* **Audit Finding B5 (Physics Loss Omissions):**
  - The continuity and momentum residuals lacked temporal partial derivatives ($\partial_t \rho, \partial_t (\rho \mathbf{v})$) necessary for dynamic compressible hydrodynamics.
  - While magnetic solenoidality $\mathcal{R}_B = \nabla \cdot \mathbf{B}$ was computed, it was entirely omitted from `total_loss` in `PhysicsConservationLoss.forward()`.
  - Applying Softplus directly to normalized/standardized channels can enforce invalid physical bounds in normalized feature space.
* **Rectification:**
  - **RETAIN:** FNO3D architecture, data contract schemas, and numerical differential operators.
  - **WITHDRAW:** Claims of a trained PINO surrogate, field prediction accuracy, and $1,162\times$ scientific speedup.
  - **REOPEN:** Genuine dataset acquisition, training loop execution, and held-out evaluation.

---

### 2.3 Track C: Autonomous Telescope Scheduling — Unbenchmarked Prototype
* **Audit Finding C1 (Under-Trained Policy):** The integrated test ran PPO for only 10 episodes.
* **Audit Finding C2 (Missing Empirical Demonstration):** The test assertions checked only that cumulative EIG was non-negative and constraint violations were $\le 3$. The test never executed a statistical comparison against the heuristic baselines (Greedy Airmass, Round Robin) to prove the claimed $\ge 20\%$ EIG gain.
* **Rectification:**
  - **RETAIN:** `TelescopeSchedulingEnv`, EIG utility formulation, and `ConstrainedPPOScheduler` implementation.
  - **WITHDRAW:** Claims of demonstrated $\ge 20\%$ scheduling performance improvement.
  - **REOPEN:** Multi-seed training and statistical baseline evaluation across realistic observing nights.

---

### 2.4 Track D & E: Governance & Test Status Qualification
* **Audit Finding D1 (Quarantine Assertion Rigor):** The integrated quarantine test checked only whether the continuation ledger contained the string "586" and the word "quarantine". That verified documentation integrity, not automated prevention of runtime data access.
* **Audit Finding D2 (Local vs. Independent Test Qualification):** The reported 136/136 passing tests represent local execution on the Lead Investigator's workstation, establishing that tested software units behave as coded. They do not constitute an independent GitHub Actions run, nor do they validate empirical scientific discoveries.
* **Rectification:**
  - **RETAIN:** Epistemic boundary promotion exception guards (`EpistemicViolationError`) and IVOA Prov-DM v1.0 data classes.
  - **CLARIFY:** Original Phase 5 cryptographic manifests (`phase5_final_evidence_manifest.sha256`) remain the authoritative seal for the Gate 3 quarantine covenant.

---

## 3. Epistemic Reclassification Matrix

Every component of Phase 6 is hereby reclassified according to the strict five-tier scientific status ontology:

| Subsystem / Deliverable | Previous Claim | Corrected Scientific Status | Action Under Phase 6.1 |
|:---|:---|:---|:---:|
| **WS1: Cosmological Background & PM Solver** | Validated scientific foundation | **Engineering implemented & tested** | Preserved baseline |
| **WS2.1: JAX Float64 Parity** | Parity verified | **Numerical behavior tested** | Preserved baseline |
| **Track A: Trajectory Reverse-Mode AD** | End-to-end AD verified | **Numerical behavior tested** | Preserved baseline |
| **Track A: Fisher Parameter Recovery** | Validated cosmological inference | **Scientific performance claim withdrawn** | Reopened under P6.1-B |
| **Track B: FNO3D/PINO Architecture** | Complete surrogate model | **Engineering implemented** | Preserved baseline |
| **Track B: Surrogate Model Training** | Trained on CAMELS/TNG | **Model training pending** | Executed under P6.1-A |
| **Track B: Field Prediction Error (< 3%)** | Empirical < 3% error | **Scientific performance claim withdrawn** | Reopened under P6.1-A |
| **Track B: Trajectory Speedup (> 1,000x)** | $1,162\times$ accelerated solver | **Scientific performance claim withdrawn** | Reopened under P6.1-A |
| **Track C: Telescope Environment & EIG** | Validated environment | **Engineering implemented** | Preserved baseline |
| **Track C: PPO Scheduling Gain (>= 20%)** | Demonstrated 20% gain | **Independent validation pending** | Executed under P6.1-C |
| **Track D/E: Epistemic & Provenance Guards** | Verified ontology | **Numerical behavior tested** | Preserved baseline |
| **Track E: Gate 3 Quarantine Covenant** | Certified preserved | **Original Phase 5 manifests authoritative** | Invariant preserved |

---

## 4. Phase 6.1 Action Plan

Phase 6.1 (`PHASE6_1_SCIENTIFIC_REMEDIATION_AND_TRAINING`) will execute four focused remediation tracks:

1. **Track P6.1-A (Hydrodynamic Surrogate Training & Fair Benchmarking):**
   - Acquire/synthesize genuine simulation training volumes with spatial independence (train/val/test disjoint volumes to prevent spatial voxel leakage).
   - Formulate unambiguous input-to-output mapping task: dark matter / gravitational potential $\to$ multi-channel hydrodynamic gas fields.
   - Fix physics loss: incorporate magnetic solenoidality into total loss, fix Softplus vs normalization.
   - Train FNO3D model with AdamW and learning rate schedule, logging loss curves and checkpoints.
   - Evaluate on held-out test volumes for relative $L_1, L_2$ field errors, power spectrum recovery, and physical bounds.
   - Conduct a fair benchmark comparing the surrogate against reference solver on an identical input-output task.
2. **Track P6.1-B (Independent Cosmological Parameter Inference):**
   - Break the inverse crime by generating reference mock power spectra using an independent realization / independent forward model with independent Gaussian phases.
   - Evaluate parameter recovery with realistic cosmic variance covariance $\mathbf{C}(k)$ across repeated mock datasets.
   - Pre-register Fisher matrix rank and condition number acceptance criteria.
3. **Track P6.1-C (Statistical Telescope Scheduler Evaluation):**
   - Train PPO agent over multiple observing night episodes.
   - Execute statistical evaluation across multiple random seeds comparing PPO against Greedy Airmass, Round Robin, Random, and Greedy EIG baselines.
   - Calculate empirical confidence intervals for EIG improvement and constraint compliance.
4. **Track P6.1-D (Scientific Audit & Formal Closure):**
   - Execute automated Phase 6.1 audit verifying trained checkpoints, held-out accuracy, independent recovery, and statistical baseline comparisons.
   - Author authoritative Phase 6.1 Scientific Closure Report documenting both achieved targets and any honest negative results.
