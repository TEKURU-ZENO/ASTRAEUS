# ASTRAEUS Scientific Research Agenda
## Primary Research Questions, Scientific Hypotheses & Verification Rubrics

**Document Identifier:** `ASTRAEUS-AGENDA-001`  
**Classification:** Scientific Programme Specification  
**Version:** 1.0.0  

---

## 1. Overarching Programme Research Question

> **Can an integrated, uncertainty-aware, physics-grounded autonomous AI ecosystem improve the efficiency, reliability, and reproducibility of scientific discovery across heterogeneous astronomical observations and computational physical models?**

---

## 2. Subsidiary Research Questions & Falsifiable Hypotheses

| ID | Research Question (RQ) | Scientific Hypothesis ($H$) | Empirical Verification Test |
| :--- | :--- | :--- | :--- |
| **RQ1** | Can a single representation model multiple heterogeneous astronomical modalities? | Shared representations preserve intrinsic astrophysical properties while achieving cross-modal zero-shot transfer. | Contrastive cross-modal retrieval Recall@K on cross-matched SDSS/Gaia/ZTF samples vs. unimodal baselines. |
| **RQ2** | Can models reliably identify previously uncatalogued astronomical populations without catastrophic false discovery? | Open-world triage combining latent distance, epistemic uncertainty, and physical forward residuals isolates true novelties. | Discovery recall at a fixed follow-up budget ($N \le 100$) evaluated against known rare transients (e.g., AT2018cow-like FBOTs). |
| **RQ3** | Does evidence-provenance Graph RAG improve factual precision over standard dense retrieval? | Linking observational entities to peer-reviewed literature through formal IVOA/W3C PROV relations eliminates citation hallucination. | Hallucination rate on expert-curated astronomical literature benchmarks evaluated via nDCG@K and Provenance Path Recall. |
| **RQ4** | Can autonomous multi-agent systems conduct end-to-end reproducible astronomical investigations? | Structured tool execution, schema-enforced contracts, and deterministic execution environments yield auditable research workflows. | Re-execution fidelity of computational scripts across independent environments (measured via hash verification and CRPS). |
| **RQ5** | Can agent systems generate physically consistent, falsifiable scientific hypotheses? | Imposing conservation-law constraints and dimensional penalties prevents mathematically ungrounded speculative reasoning. | Hypothesis falsification rate under adversarial review by domain skeptic agents; rejection of 100% of injected false priors. |
| **RQ6** | Can hybrid neural-symbolic systems discover closed-form physical laws from observational residuals? | Pareto-optimal symbolic regression constrained by dimensional homogeneity recovers physical relationships in noisy regimes. | Exact recovery rate of synthetic astrophysics laws (e.g., modified Tully-Fisher, Kepler's third law) under injected observational noise. |
| **RQ7** | Can neural operators accelerate cosmological hydrodynamic simulations without unacceptable physical error? | Physics-Informed Neural Operators (PINO) respecting mass, momentum, and energy conservation maintain generalizability across cosmologies. | Relative $L_2$ error and mass/momentum conservation residuals on CAMELS parameter variations vs. full AREPO runs. |
| **RQ8** | Can differentiable digital twins bridge the gap between simulation and real-world observations? | Adjoint-differentiated field-level inference allows unbiased parameter estimation without information loss from 2-point summaries. | Bayesian parameter recovery on synthetic and held-out cosmological mock surveys vs. standard MCMC power-spectrum summaries. |
| **RQ9** | Can Bayesian neural architectures deliver well-calibrated cosmological posterior distributions? | Hierarchical latent uncertainty modelling with conformal prediction bounds prevents overconfident parameter inference. | Empirical coverage probability $1-\alpha$ across held-out test surveys; Expected Calibration Error (ECE) minimization. |
| **RQ10** | Can domain-constrained causal discovery uncover physically causal relationships in astrophysical databases? | Encoding physical time ordering and conservation invariants restricts causal DAG search space to verifiable physical mechanisms. | Structural Hamming Distance (SHD) against established astrophysical causal graphs (e.g., stellar mass $\to$ metallicity relations). |
| **RQ11** | Can reinforcement learning improve telescope scheduling under stochastic weather and rapid alert dynamics? | Constrained PPO with adaptive Lagrangian penalties optimizes Expected Information Gain (EIG) over greedy heuristics. | Scientific yield and slew efficiency under simulated Rubin/GW alert bursts with real-world weather dropouts. |
| **RQ12** | Can active learning reduce the volume of human expert spectroscopic labels required for transient surveys? | Query-by-committee sampling prioritizing high conformal uncertainty candidates maximizes classification boundary precision. | Data efficiency curves: Macro-F1 vs. number of spectroscopic follow-up allocations compared to random and greedy baselines. |
| **RQ13** | Does structured adversarial multi-agent debate reduce scientific groupthink and errors? | Isolated agent sandboxes and explicit falsification protocols prevent early consensus on spurious or instrumental artifacts. | Falsification accuracy on synthetic artifact injection challenges (diffraction spikes, satellite streaks, cosmic rays). |
| **RQ14** | Can joint multi-messenger fusion improve rapid event localization and association? | Joint spatio-temporal graph networks combining gravitational-wave sky maps with optical alert streams accelerate counterpart recovery. | Time-to-identification and spatial search area reduction for compact binary coalescences (NS-NS / NS-BH mergers). |
| **RQ15** | Can automated Bayesian experimental design outperform static observational campaigns? | Dynamic selection of telescope pointings and filter bands based on mutual information maximization minimizes parameter uncertainty faster. | Rate of posterior entropy reduction per unit of exposure time across multi-epoch observation campaigns. |

---

## 3. The Hypothesis Scoring Rubric

Hypotheses formulated during autonomous investigations are scored objectively:

$$Q(H) = w_e E(H) + w_p P(H) + w_s S(H) + w_r R(H)$$

- **Observational Evidence Quality $E(H) \in [0, 1]$:** Traceability of supporting measurements to primary detector readouts.
- **Physical Consistency $P(H) \in [0, 1]$:** Conformance to conservation of mass, energy, momentum, and dimensional homogeneity.
- **Statistical Significance $S(H) \in [0, 1]$:** Likelihood ratio / Bayes factor against null hypothesis; conformal coverage validation.
- **Computational Reproducibility $R(H) \in [0, 1]$:** Execution repeatability across isolated containers with fixed random seeds.

*Engineering Note:* $Q(H)$ serves as an engineering prioritization metric for scheduling follow-up compute or observation, not as an epistemic substitute for empirical proof.
