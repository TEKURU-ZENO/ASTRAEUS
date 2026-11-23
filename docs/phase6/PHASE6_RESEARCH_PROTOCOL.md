# ASTRAEUS Phase 6 Master Research Protocol
## Computational Universe, Differentiable Physics & Experimental Intelligence

**Document Identifier:** `ASTRAEUS-P6-PROTO-001`  
**Classification:** Strategic Scientific Charter & Master Engineering Protocol  
**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Baseline Git Checkpoint:** `0601beb`  
**Date:** November 23, 2026  
**Status:** APPROVED & ACTIVE  

---

## 1. Executive Summary & Programme Transition

Project ASTRAEUS transitions from observational transient astronomy and cross-survey domain adaptation (Phase 5) to computational scientific experimentation (Phase 6). Phase 5 established the limits of empirical feature transfer, demonstrated that manual feature compression can lose critical temporal coherence, and codified the strict separation between synthetic algorithmic demonstrations and physical telescope observations.

Phase 6 addresses the fundamental challenge of autonomous computational science:
> **Central Research Mission:**  
> *Can ASTRAEUS build computationally efficient, differentiable, and physically constrained models of astrophysical systems, and use those models to autonomously design optimal scientific observation experiments?*

Rather than forcing heterogeneous physical domains into a monolithic architecture, Phase 6 institutes three specialized computational engines governed by shared IVOA/epistemic data contracts:
1. **Cosmology Engine:** Differentiable particle-mesh N-body dynamics (JAX, comoving Poisson solver, symplectic integration).
2. **Hydrodynamics Engine:** Physics-Informed Neural Operators (PINO, Fourier Neural Operators) trained on cosmological hydrodynamic suites (CAMELS, IllustrisTNG).
3. **Experiment Engine:** Information-theoretic autonomous experimental design (Bayesian optimization, Gymnasium-compatible simulator, constrained PPO telescope scheduler).

---

## 2. Flagship Research Questions

| ID | Domain | Flagship Scientific Research Question | Primary Falsification Target |
| :---: | :--- | :--- | :--- |
| **RQ16** | **Differentiable Cosmology** | *Can differentiable cosmological particle-mesh simulations support reliable, gradient-based parameter inference ($\Omega_m, \sigma_8$) without suffering from vanishing/exploding adjoint gradients or non-physical local minima?* | Adjoint gradients diverge by $> 10\%$ from finite-difference reference or fail to recover injected cosmological parameters within $2\sigma$. |
| **RQ17** | **Neural Operator Surrogates** | *Can Physics-Informed Neural Operators (PINO) accelerate 3D hydrodynamic simulations by $> 1,000\times$ while preserving physical conservation laws (mass, momentum) and keeping relative field errors $< 3\%$ across cosmological parameter variations?* | Surrogate field error $\ge 3\%$ relative to full hydrodynamic solver, or conservation law violation residuals $> 1\%$. |
| **RQ18** | **Autonomous Experimental Intelligence** | *Can an autonomous agent using Expected Information Gain (EIG) and constrained Reinforcement Learning (PPO) schedule multi-facility telescope follow-up to achieve $\ge 20\%$ higher scientific parameter information gain than standard greedy dispatch heuristics under realistic weather and slewing constraints?* | EIG improvement $< 20\%$ over preregistered greedy scheduler across 100 stochastic weather/visibility trials. |

---

## 3. Computational Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ASTRAEUS SCIENTIFIC ORCHESTRATOR                             │
│        IVOA Provenance · Epistemic Contracts · Experiment Configuration         │
└────────┬───────────────────────────────┬───────────────────────────────┬────────┘
         │                               │                               │
         ▼                               ▼                               ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────┐
│    COSMOLOGY ENGINE     │  │   HYDRODYNAMICS ENGINE  │  │   EXPERIMENT ENGINE   │
│  - Comoving PM Solver   │  │  - Fourier Neural Oper. │  │  - Bayesian Exp Design│
│  - Symplectic Leapfrog  │  │  - PINO Conservation    │  │  - Constrained PPO    │
│  - JAX Differentiability│  │  - CAMELS / IllustrisTNG│  │  - Multi-Facility EIG │
│  - Adjoint Sensitivity  │  │  - Sub-grid Feedback    │  │  - Slewing & Weather  │
└────────┬────────────────┘  └───────────┬─────────────┘  └───────────┬───────────┘
         │                               │                            │
         └───────────────────────────────┼────────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 INDEPENDENT SCIENTIFIC VERIFICATION LAYER                       │
│    Conservation Invariants · Layzer-Irvine Residuals · Finite-Diff Gradients    │
│            Power Spectra $P(k)$ · Hardware Benchmarks · Manifest Audits         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 6 Workstream Architecture

Phase 6 is structured into six strictly sequential, milestone-gated workstreams:

### Workstream 1: Scientific Computing Foundation (Weeks 1–2)
- **Objective:** Establish the analytical and numerical foundation before training neural surrogates.
- **Topics:** Hamiltonian mechanics, comoving coordinates, cosmological expansion $a(t)$, Cloud-in-Cell (CIC) mass assignment, 3D FFT periodic Green's function Poisson solver, Symplectic Kick-Drift-Kick (KDK) leapfrog integration, Layzer-Irvine cosmic energy equation.
- **Deliverable:** Validated comoving Particle-Mesh reference simulation engine (`engines/cosmology/`) passing exact analytical benchmarks (Zel'dovich pancake collapse, momentum conservation to machine epsilon).

### Workstream 2: Differentiable Cosmological Engine (Weeks 3–4)
- **Objective:** Enable automatic differentiation and adjoint sensitivity analysis through simulation time steps.
- **Topics:** Differentiable particle-mesh operations in JAX, reverse-mode automatic differentiation, adjoint sensitivity formulation, gradient checks against central finite differences, memory-efficient checkpointing.
- **Deliverable:** Differentiable simulation suite with exact gradient verification and reproducible parameter posterior estimation benchmark.

### Workstream 3: Physics-Informed Neural Simulation (Weeks 5–7)
- **Objective:** Train and evaluate neural operators mapping initial conditions and cosmological parameters to evolved density and velocity fields.
- **Topics:** Fourier Neural Operators (FNO), Physics-Informed Neural Operators (PINO), CAMELS Latin Hypercube sub-volumes, IllustrisTNG hydrodynamic data adapter, mass/momentum conservation penalty losses.
- **Deliverable:** Trained neural surrogate checkpoint with strict latency, memory, and relative field error disclosures.

### Workstream 4: Scientific Verification & Benchmarking (Week 8)
- **Objective:** Conduct independent validation decoupled from model training routines.
- **Topics:** Matter power spectrum $P(k)$ comparisons, halo mass function deviations, Layzer-Irvine energy conservation residuals, extreme parameter extrapolation testing, failure mode taxonomy.
- **Deliverable:** Independent benchmark dossier and formal scientific verification report.

### Workstream 5: Autonomous Experimental Intelligence (Weeks 9–10)
- **Objective:** Formulate telescope scheduling as a constrained Markov Decision Process (MDP) optimizing Expected Information Gain (EIG).
- **Topics:** Multi-facility observatory simulator (visibility windows, moon phase, atmospheric seeing, slewing kinematics, instrument overheads), Expected Information Gain metric for parameter constraint, Constrained Proximal Policy Optimization (PPO).
- **Deliverable:** Benchmark suite comparing autonomous RL policy against classical dispatch and greedy heuristics.

### Workstream 6: Scientific Reconciliation & Phase Closure (Weeks 11–12)
- **Objective:** Cryptographic sealing of all simulation artifacts, test verification, and publication release.
- **Topics:** Master evidence manifest creation, hardware disclosure certification, final manuscript drafting, formal Phase 6 sign-off.
- **Deliverable:** `PHASE6_FINAL_SCIENTIFIC_CLOSURE_REPORT.md`, `PHASE6_RESEARCH_PAPER.md`, and cryptographic manifest.

---

## 5. Pre-Registered Hypotheses & Acceptance Criteria

| Hypothesis | Description | Evaluated Estimand | Pre-Registered Target | Falsification Boundary |
| :---: | :--- | :--- | :---: | :---: |
| **H16** | **Numerical Invariant Conservation** | Comoving momentum and mass conservation across 100 simulation dynamical times | $\Delta P / P_0 \le 10^{-10}$ | Residual drift $> 10^{-8}$ |
| **H17** | **Adjoint Gradient Exactness** | Relative discrepancy between reverse-mode adjoint gradients $\nabla_{\Omega_m}\mathcal{L}$ and 5-point central finite differences | $\frac{\|\mathbf{g}_{\mathrm{adj}} - \mathbf{g}_{\mathrm{fd}}\|}{\|\mathbf{g}_{\mathrm{fd}}\|} \le 0.02$ | Discrepancy $> 5.0\%$ |
| **H18** | **Surrogate Acceleration** | Inference speedup of PINO surrogate over reference numerical solver on identical 3D grid resolution ($64^3$) | $\text{Speedup} \ge 1,000\times$ | $\text{Speedup} < 500\times$ |
| **H19** | **Physical Field Fidelity** | Normalized root-mean-square error (NRMSE) on evolved gas density and dark matter fields | $\mathrm{NRMSE} \le 0.030$ ($3.0\%$) | $\mathrm{NRMSE} > 0.050$ ($5.0\%$) |
| **H20** | **Autonomous EIG Optimization** | Percentage improvement in cumulative Expected Information Gain over greedy heuristic | $\Delta \mathrm{EIG} \ge +20.0\%$ | $\Delta \mathrm{EIG} < +10.0\%$ |
| **H21** | **Rare-Regime Robustness** | Surrogate stability and relative field error under extreme feedback parameters ($A_{\mathrm{AGN}} > 3.0\sigma$) | $\mathrm{NRMSE}_{\mathrm{extreme}} \le 0.080$ | Divergence or $\mathrm{NRMSE} > 0.150$ |

---

## 6. Scientific Governance & Phase 5 Continuation Covenant

1. **Phase 5 Checkpoint Immutability:** Commit `0601beb` remains the authoritative, immutable checkpoint for Phase 5.
2. **Phase 5 Continuation Ledger:** The empirical cross-survey observational gaps (live ATLAS/DECam broker acquisition, H1/H3/H4 empirical resolution) are formally maintained in `docs/phase6/PHASE5_CONTINUATION_LEDGER.md`. Phase 6 computational experiments shall not be misconstrued as empirical observational resolution.
3. **Quarantine Invariant:** The 586 Gate 3 test objects and 1,808 HEALPix pixels remain permanently sealed under zero-touch quarantine.
4. **Hardware Honesty:** All latency and throughput benchmarks must disclose exact hardware specifications (Local RTX 5050 Laptop GPU vs Cloud GPU vs HPC cluster).

---

**Certified and Authorized by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 23, 2026
