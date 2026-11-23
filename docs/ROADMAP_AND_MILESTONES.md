# ASTRAEUS Roadmap & Milestone Verification Matrix
## Phased Multi-Year Development Programme & Gate Acceptance Criteria

**Document Identifier:** `ASTRAEUS-ROADMAP-001`  
**Classification:** Strategic Planning & Milestone Verification  
**Version:** 2.4.0 (Phase 6 Initialized: Computational Universe & Experimental Intelligence Foundation)  

---

## 1. 15-Month Master Programme Timeline

```
Months 1-2    Months 3-4     Months 5-6     Months 7-8     Months 9-10    Months 11-12   Months 13-14   Month 15
┌───────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ Phase 1:  │ │ Phase 2:   │ │ Phase 3:   │ │ Phase 4:   │ │ Phase 5:   │ │ Phase 6:   │ │ Phase 7:   │ │ Phase 8:   │
│ Scientific│ │ Multimodal │ │ Knowledge  │ │ Multi-Agent│ │ Cross-Survey│ │ Computat.  │ │ Full Closed│ │ Open-Source│
│ Data      │ │ Latent     │ │ Fabric &   │ │ Adversarial│ │ Generaliz- │ │ Universe & │ │ Loop       │ │ Research   │
│ Foundation│ │ Foundation │ │ Graph RAG  │ │ Reasoning  │ │ ation & Rob│ │ RL Sched    │ │ Integration│ │ Ecosystem  │
└───────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

---

## 2. Milestone Acceptance Gates

| Phase / Milestone | Focus Area | Core Deliverables | Formal Acceptance Gate |
| :--- | :--- | :--- | :--- |
| **Phase 1 (M1-2)** | **Scientific Data Foundations** | Epistemic schemas, IVOA PROV-DM, Astroquery/MAST pipelines, baseline classifiers, composite anomaly triage. | Ingestion of $\ge 5 \times 10^5$ light curves with spatial HEALPix ($N_{\mathrm{side}}=32$) zero-leakage guarantee; Day 90 discovery report. |
| **Phase 2 (M3-4)** | **Multimodal Foundations** | Cross-modal contrastive encoders, masked autoencoders, missing-modality masks, conformal quantile heads. | Cross-survey zero-shot evaluation exceeds unimodal baselines by $\ge 15\%$ on property estimation ($R^2$ / CRPS). |
| **Phase 3 (M5-6)** | **Knowledge Fabric** | Neo4j property graph, Qdrant vector index, NASA ADS literature ingestion, citation verification engine. | $\ge 90\%$ accuracy on expert-curated citation verification; 0% hallucinated bibcodes. |
| **Phase 4 (M7-8)** | **Multi-Agent Reasoning** | LangGraph orchestration engine, 24 specialized agents, isolated sandboxes, REGROUND debate protocol. | REGROUND verification rejects 100% of injected false premises and instrumental artifacts in validation tests. |
| **Phase 5 (M9-10)** | **Cross-Survey Generalization & Domain Robustness** | Photometric/temporal harmonization, ZTF-ATLAS-DECam-PS1 benchmark, zero-shot & domain adaptation experiments, shift & uncertainty validation. *(Computational Universe deferred to Phase 6)*. | Rigorous falsification of H1–H5; quantified negative transfer & selective prediction operating envelope across $\ge 2$ independent external survey domains; zero-leakage sealed audit. |
| **Phase 6 (M11-12)** | **Computational Universe & Experimental Intelligence** | JAX `pmwd` integration, PINO hydrodynamic surrogates trained on CAMELS and IllustrisTNG; Constrained PPO telescope scheduler. | Neural surrogate achieves $>10^3\times$ speedup over full hydrodynamics with $<3\%$ relative field error; RL scheduler delivers $\ge 20\%$ higher EIG. |
| **Phase 7 (M13-14)** | **Closed-Loop Integration** | End-to-end integration: alert ingestion $\to$ triage $\to$ debate $\to$ simulation $\to$ follow-up planning. | Automated end-to-end processing of simulated Rubin/GW alert bursts without manual pipeline intervention. |
| **Phase 8 (M15)** | **Community Release** | Open-source release under Apache-2.0, technical documentation, ASTRAEUS-Bench public leaderboard. | Reproducible installation runbooks, open-weight model weights, complete scientific benchmark suite. |

---

## 3. Revised Phase 1 Sprint Milestones (Days 1–90)

- **Sprint 1 (Days 1–15) [COMPLETE]:** Setup monorepo, author 6 master specifications, implement `astraeus-core` epistemic data structures and initial IVOA bindings. Zero HEALPix assignment overlap manifest generated. 20 automated unit tests passing.
- **Sprint 2 (Days 16–30) [REVISED WORKSTREAM ARCHITECTURE]:**
  1. *Workstream 1 (Data Contracts):* Frozen ZTF alert and light-curve schemas preserving raw packets, timescale standards (MJD/BJD), filters, flux vs magnitude, upper limits vs detections, and quality flags.
  2. *Workstream 2 (Ingestion & Replay Engine):* Historical backfill and streaming alert processor with idempotent replay (duplicate prevention), schema-version tracking, dead-letter quarantine, and failure recovery.
  3. *Workstream 3 (Probabilistic Gaia Cross-Matcher):* Bayesian association component with proper motion epoch propagation, local stellar density estimation ($\rho_{\mathrm{local}}$), chance coincidence calculation, and tri-state classification (`SECURE_MATCH`, `AMBIGUOUS_MATCH`, `NO_RELIABLE_MATCH`).
  4. *Workstream 4 (Dataset Construction & Multi-Tier Leakage Audit):* First real-data Parquet release with canonical object identity, multi-tier leakage audits (HEALPix boundary buffer, object aliases, repeated detections, host galaxy co-splitting), and immutable dataset manifests.
  5. *Workstream 5 (Baseline Modeling Scaffolding):* Feature-engineered baselines (periodicity extraction via Lomb-Scargle, transient features), task-separated classifiers (Random Forest), and hardware VRAM/throughput benchmarks on RTX 5050 (8 GB) before freezing transformer configs.
- **Sprint 3 (Days 31–45):** Train and benchmark baseline temporal models on real ZTF/BTS data; evaluate representation learning against feature-engineered baselines; log to MLflow.
- **Sprint 4 (Days 46–60) [COMPLETE - STATISTICAL SCORING & PHYSICS FITTING ENGINE]:**
  1. *Workstream A (Statistical Scoring Foundation):* Empirical reference CDF normalization ($\widehat{F}_j$) frozen on reference populations; separation into operational `AnomalyRankScore` and diagnostic `AnomalyEvidenceProfile`; component correlation and ablation analysis.
  2. *Workstream B (Model-Aware Physics Fitting):* Model applicability matrix; first-class `FitStatus` outcomes; analytical Bazin morphology, Arnett $^{56}\mathrm{Ni}$ diffusion, and early-time shock-cooling models with parameter covariance, BIC model selection, and heteroscedastic residuals.
  3. *Workstream C (Split Conformal Calibration):* Finite-sample nonconformity quantile calculation; dual-setting evaluation (Setting 1 Exchangeable coverage = 0.850 vs Setting 2 Distribution Shift coverage = 0.080, demonstrating empirical coverage shortfall); explicit prediction targets (`peak_magnitude`, `next_epoch_flux`).
  4. *Workstream D (Rare-Event Anomaly Benchmark):* 4-population benchmark (common transients, confirmed peculiar FBOTs/SLSNe, artifacts, unclassified) + injection-and-recovery track; comparisons against Isolation Forest, LOF, Euclidean distance, and component ablations; fixed follow-up budget simulation ($K \in \{5, 10, 20, 50, 100\}$); 57 unit tests passing with 94% test statement coverage across all 4 packages.
- **Sprint 5 (Days 61–75) [COMPLETE - CROSS-SURVEY TRANSFER & GENERALIZATION]:**
  1. *Workstream 1 (Cross-Survey Data Foundations):* Pan-STARRS1 ($g_{\mathrm{P1}}, r_{\mathrm{P1}}, i_{\mathrm{P1}}, z_{\mathrm{P1}}, y_{\mathrm{P1}}$) and DECam ($u, g, r, i, z, Y$) schemas; `PhotometricHarmonizer` with transmission curves, AB flux conversions, heteroscedastic uncertainty propagation, Galactic extinction corrections, and non-destructive upper-limit preservation.
  2. *Workstream 2 (Bayesian Matcher & Identity Exclusion):* Sutherland & Saunders likelihood ratio matching; 3.0" spatial exclusion audit formally guaranteeing zero object-identity leakage between ZTF training set and PS1/DECam evaluation sets (0 contaminated overlaps).
  3. *Physical Engine Refactoring:* Decoupled physical engine into Model Fit Quality ($\chi^2_{\mathrm{red}}, \mathrm{BIC}$), Inferred Physical Parameters ($\theta_x$), and Parameter-Space Deviation ($D_{\mathrm{phys}}$ against reference SN priors).
  4. *Senior Mandated Experiment A (Physical Parameter Consistency):* Evaluated on 25 matched cross-survey transients; measured 24.0% consistency rate, demonstrating empirical sensitivity of physical parameters to survey cadence and filter windows.
  5. *Senior Mandated Experiment B (Anomaly-Rank Stability):* Verified Spearman rank correlation $\rho_s = 0.8485$ and Top-10 Jaccard overlap = 0.82 between ZTF and Pan-STARRS1, confirming `STABLE` cross-survey ranking.
  6. *Senior Mandated Experiment C (Negative Transfer & Calibrated Abstention):* Detected 27.89% macro-F1 degradation under DECam's dilated cadence; calibrated covariate-shift abstention filter ($d_{\mathrm{shift}} \le 2.5$) successfully mitigated negative transfer and recovered post-abstention F1; 65 automated tests passing with 94% statement coverage across all 4 packages.
- **Sprint 6 (Days 76–90) [COMPLETE - AUTONOMOUS MULTI-AGENT RUNTIME & FLAGSHIP DISCOVERY DOSSIER]:**
  1. *Multi-Agent Architecture (`astraeus-agents`):* 4-agent collaborative runtime (`DataCrossMatchAgent`, `LiteratureContextAgent`, `SkepticValidationAgent`, `PhysicalModelEngine`) coordinated via typed contracts and IVOA Prov-DM provenance tracking.
  2. *Adversarial Skeptic Gatekeeper:* Systematic execution of 4-category challenges (PSF shape failure, image subtraction dipole defects, crowded field blending, and unphysical parameters such as $M_{\mathrm{Ni}} > 2.5\,M_\odot$).
  3. *Deterministic Scientific Evidence Gates:* Replaced LLM discretion with mathematical code gates enforcing terminal statuses (`CANDIDATE_REQUIRES_FOLLOWUP`, `KNOWN_OBJECT`, `KNOWN_CLASS_UNUSUAL`, `REJECTED_AS_ARTIFACT`, `INSUFFICIENT_EVIDENCE`).
  4. *Flagship Day 90 Reproducible Discovery Dossier & Governance Benchmark:* Evaluated on 4 canonical deep-dive transients (AT2018cow FBOT analogue, normal SN Ia SN2026abc, subtraction dipole defect, and sparse trigger) and a 50-candidate validation cohort; demonstrated **0.0% epistemic violation rate**, **0.0% artifact false alarm rate**, and **93.8% ToO precision**; 81 unit tests passing with **94% statement test coverage** across all 5 monorepo packages.
  5. *Day 90 Scientific Audit & Reconciliation v2.2.1 (Post-Senior Independent Audit):* Formally reconciled flagship discovery epoch to $t_0 = 58286.04 \pm 0.21\,\mathrm{MJD}$ (June 17.04, 2018 at 00:58 UTC) matching published ATLAS discovery alerts; confirmed host galaxy CGCG 137-068 separation at $\Delta \theta = 5.9''$ ($d_{\mathrm{proj}} \approx 1.73\,\mathrm{kpc}$ in Flat $\Lambda\mathrm{CDM}$) with qualified progenitor interpretation; formally designated Wilson score intervals without continuity correction ($[0.00\%, 27.75\%]$ for 0/10; $[0.00\%, 7.13\%]$ for 0/50) alongside both exact two-sided 95% Clopper-Pearson intervals ($[0.00\%, 30.85\%]$ for 0/10; $[0.00\%, 7.11\%]$ for 0/50) and exact one-sided 95% upper bounds ($\le 25.89\%$ for 0/10; $\le 5.82\%$ for 0/50); reported exact paired McNemar test for Config A vs E ($p = 0.000244, \chi^2_{\mathrm{cc}} = 11.08$) with full paired $2 \times 2$ correctness table ($a=36, b=13, c=0, d=1$); reported strictly paired McNemar test for D2 vs E on 10 artifacts ($p = 0.50$ two-sided, $p = 0.25$ one-sided; $b=2, c=0$, noting $N=10$ cohort is underpowered to establish significance at 5% level); published 5 individual seed runs ($53.73\% \pm 2.78\%$) establishing that baseline achieves 100% target recall but exhibits high false-alarm variance; reconciled $D_{\mathrm{morph\_RMS}}$ derivations (3.98 for unconstrained fit, 3.92 for literature fiducial); published 6-row $\chi^2_{\mathrm{red}}$ reconciliation table ($6.16$ vs $2.78$, clarifying $\mathbb{E}[\chi^2_{\mathrm{red}}] = 1$); published full $5 \times 5$ correlation matrix $R$; designated multi-band feature as `optical_pseudoflux`; verified queue boundary security via `FollowUpQueueManager` with both SHA-256 integrity digests and keyed HMAC-SHA256 authority signatures (`test_gate_bypass_resistance.py`); and restructured Phase 2 research stages. Phase 1 accepted; Phase 2 authorized.

---

## 4. Phase 2 Architecture & Strategic Research Tracks (Days 91–180)

**Theme:** Multimodal Latent Foundation Models for Autonomous Scientific Intelligence  
**Core Objective:** Learn scientific representations of irregular astronomical time series, spectra, and multi-band imaging that generalize across surveys without destroying physical information.

### Restructured Phase 2 Research Stages (Ordered Sequence):
- **Stage 1 (Days 91–105) — Astronomical Reference Corpus & Provenance Hygiene [COMPLETE & FORMALLY CERTIFIED - v1.0.5]:**
  Curated Astronomical Reference Corpus v1.0.5 across genuine archival ($N=3$), numerically simulated ($N=105$), and standardized synthetic ($N=175$) transient populations ($N_{\mathrm{total}}=283$). Enforced spatial HEALPix ($N_{\mathrm{side}}=32$) zero-overlap partitioning with non-geodesic curved boundary sampling (400 vertices/cell, analytical sagitta error bound $\delta \le 0.00236'' \ll 5.0''$ derived from maximum geodesic curvature $\kappa_g = 2/\sqrt{5} \approx 0.8944$; Richardson extrapolation $\delta_{100} \approx 0.00208''$), 2D tangent plane anisotropic covariance projection with Fréchet/Boole union upper bound corner handling ($P_{\mathrm{cross}} \le \min(1.0, P_1 + P_2)$), global convergence validation across 128 systematic sample pixels (51,200 segments; global worst-case delta $0.0078'' \ll 5.0''$, mean $0.0010''$), exact Clopper-Pearson binomial intervals ($[42.13\%, 99.64\%]$ for $6/7$ successes), object-level manifest diff stability (0 partition or class changes between `40e3e60` and `HEAD`), full resolution of historical preliminary narrative draft tables vs authoritative committed manifest counts, calibrated host association posterior thresholding ($P_{\mathrm{match}} \ge 0.80$ with $[0.50, 0.80, 0.95]$ sensitivity analysis), explicit disclosure that held-out test split contains zero real telescope observations, specification of Track 3 External Archival Evaluation Cohort (15–20 genuine un-trained transients), and DAG provenance auditing (`DatasetBSampleLineage` and `ProvenanceGraphAuditor`) with cycle detection, transitive ancestry, and pilot Dataset B ingestion audit ($N=500$: 460 verified clean, 25 quarantined missing lineage, 15 rejected test overlaps, 0 cycles). All 7 audit gates passed; Parquet partitions verified with SHA-256 digests; authoritative protocol hash `4d4aca4024be50ae1f7412201b0f42c45eb6a3efdb6edef7e95ce852385b680c` cryptographically sealed. Stage 1 formally certified; Stage 2 Days 106–108 input manifest freeze approved.
- **Stage 2 (Days 106–120) — Astronomical Temporal Foundation Model (Track A) [STAGE 2 INPUT FREEZE APPROVED]:**
  Continuous-time positional encoding, masked observation autoencoding on irregular cadences, and uncertainty-aware latent projections.
  *Approved 5-Stage Workstream Schedule:*
  1. *Days 106–108 (Data & Protocol Reconciliation):* Frozen, reproducible input manifest & Dataset B DAG auditor sealed.
  2. *Days 109–111 (Temporal Representation Design):* Sequence specification, continuous-time encoding & simple baselines (Linear, Gaussian Process, ASTROMER, Non-pretrained sequence model; paired Wilcoxon signed-rank and McNemar tests predefined).
  3. *Days 112–115 (Initial Controlled Training):* Small-scale pretrained checkpoint on verified spatial split.
  4. *Days 116–118 (Comprehensive 3-Track Evaluation):*
     - *Track 1 (Temporal Representation):* Masked reconstruction (CRPS, MAE), temporal forecasting ($t > t_{\mathrm{cutoff}}$), and conformal calibration.
     - *Track 2 (Physical Transfer):* Zero-shot performance on unseen physical objects and unseen transient classes, decoupling simulated vs synthetic data.
     - *Track 3 (External Archival Generalization):* Zero-shot evaluation on genuine un-trained real telescope transients (SN 1998bw, SN 2020bvc, AT2018lqh, SN 2018oh, ASASSN-14li, SN 2006gy, iPTF14hls, AT2020xnd).
  5. *Days 119–120 (Scientific Audit):* Stage 2 experiment report, baseline comparison verification, and senior independent go/no-go sign-off for full-scale pretraining.
- **Stage 3 (Days 126–140) — Cross-Survey Robustness & Transfer (Track D):**
  Zero-shot transfer benchmarks from ZTF to Pan-STARRS1 and DECam, calibrated covariate-shift abstention heads, and robustness testing under severe cadence degradation ($N < 6$).
- **Stage 4 (Days 141–160) — Multimodal Astronomical Representation (Track B):**
  Joint cross-attention transformer fusing multi-band light curves, low-resolution classification spectra, difference-imaging cutouts, and host galaxy photometric redshifts.
- **Stage 5 (Days 161–180) — Physics-Conditioned Latent Spaces & Paper Preparation (Tracks C & E):**
  Self-supervised regularizers enforcing energy conservation and diffusion scaling relations; formal benchmarking against Random Forest and XGBoost baselines; research paper preparation.

### Compute Strategy:
- **Local Prototyping:** RTX 5050 Laptop GPU (8 GB VRAM) allocated for small-batch profiling, single-curve tokenizers, unit test execution, and pipeline validation.
- **Pretraining at Scale:** Survey-scale masked autoencoding across $\ge 10^6$ light curves allocated to cloud/institutional GPU clusters (A100/H100 nodes).

---

## 5. Phase 5 Master Research Programme (M9–10)

**Theme:** Independent Cross-Survey Generalization and Domain Robustness  
**Lead Investigator:** Dev Mehta  
**Governance Invariant:** Frozen Gate 4 Champion Model (`466c2752808b...`), Quarantined Gate 3 Spatial Test Set (586 objects), Zero Leakage.  

### Core Scientific Objective:
Determine whether ASTRAEUS preserves astrophysical information, physical parameter estimates, classification performance, and uncertainty calibration when observations originate from genuinely independent survey instruments and processing pipelines without relying on survey-specific observational artifacts.

### Multi-Survey Strategy:
1. **ZTF (Source Domain):** Reference calibration system, frozen training and validation partitions.
2. **ATLAS (Primary External Domain):** Forced photometry across all-sky positions; primary external light-curve source for temporal overlap and Cadence/Filter shift analysis.
3. **DECam DDF (Secondary External Domain):** Deep Drilling Fields (e.g. COSMOS, ECDFS); independent replication for high-cadence and faint transient experiments.
4. **Pan-STARRS1 (Complementary Domain):** Medium Deep fields, 3π archival reference, host-galaxy context, and deep multi-band stack photometry.
5. **Rubin LSST (Prospective Extension):** Public alert interface adapter; full inclusion conditional on data rights and processed product access.

### Sequential Workstreams:
- **Workstream 1:** Scientific Preflight and Protocol Freeze (`PHASE5_SCIENTIFIC_PROTOCOL.md`) [COMPLETE]
- **Workstream 2:** Survey Reconnaissance and Feasibility Audit (`PHASE5_SURVEY_FEASIBILITY_AUDIT.md`) [COMPLETE]
- **Workstream 3:** Cross-Survey Identity and Provenance Resolution [COMPLETE & RECLASSIFIED]:
  - Reclassified 150-candidate pilot as Synthetic Matcher Simulation Benchmark (`SYNTHETIC_MATCHER_SIMULATION_BENCHMARK`).
  - Implemented 2D anisotropic diagonal Gaussian positional likelihood with Mahalanobis distance in local tangent plane.
  - Durable Parquet-backed entity registry (`StableEntityRegistry`) with collision guards, monotonic IDs, and multi-survey indexing.
  - Canonical Observation Contract v2.0 timestamp semantics coupling timestamps with physical time scales (`get_time_coordinate()`).
  - Proved analytical positive harmonization covariance $\operatorname{Cov}(c, o) \approx 0.132\sigma_g^2 + 0.317\sigma_r^2 > 0$ with exact $A^{-1}$ inversion.
  - All 47 tests passing across the repository.
- **Workstream 3.1:** Controlled ATLAS Forced-Photometry Broker Acquisition Protocol [PROTOCOL FROZEN & AUDIT READY — LIVE EXECUTION GATED]:
  - Authored formal acquisition charter: `PHASE5_ATLAS_FORCED_PHOTOMETRY_ACQUISITION_PROTOCOL.md`.
  - Staged acquisition plan: 10-candidate operational validation pilot (`phase5_atlas_10obj_pilot_manifest.json`) covering all 5 physical classes; 140 candidates frozen in held-back expansion cohort.
  - Application-level deterministic 24-character SHA-256 idempotency key and queue reconciliation.
  - Application-level durable request ledger initialized (`phase5_atlas_request_ledger.parquet`).
  - Raw payload vaulting architecture preserving byte-exact provider responses with SHA-256 digests.
  - Non-destructive preservation of negative difference-flux measurements.
  - Four-tier outcome disaggregation: Request status, Payload schema validity, Counterpart identity confidence, and Photometric feature evaluability.
- **Workstream 4:** Photometric and Temporal Harmonization Framework [REMEDIATION IMPLEMENTED; EMPIRICAL VALIDATION PENDING]:
  - Master scientific protocol: `PHASE5_WORKSTREAM4_HARMONIZATION_PROTOCOL.md` (3-state taxonomy, negative flux preservation).
  - Temporal harmonization specification: `PHASE5_TEMPORAL_HARMONIZATION_SPEC.md` (UTC/TAI/TT/TDB, Roemer delay across 7 major observatories, BMJD_TDB).
  - Photometric harmonization specification: `PHASE5_PHOTOMETRIC_HARMONIZATION_SPEC.md` (photon-weighted $\langle F_\nu \rangle$, total system throughput, full covariance propagation).
  - Statistical and governance plan: `PHASE5_HARMONIZATION_STATISTICAL_AND_GOVERNANCE_PLAN.md` (Bland-Altman, ODR regression, causal prefix filtering).
  - Implementation: `TemporalHarmonizer` in `packages/astraeus-data/src/astraeus_data/photometry/temporal.py` (fail-closed, sub-second precision, stale invalidation).
  - Full covariance engine in `PhotometricHarmonizer` ($\kappa(\mathbf{J}) \approx 2.85$, positive-definite flux covariance).
  - Sealed in `phase5_ws3_ws4_transition_manifest.json` and anchored in `phase5_ws3_ws4_transition_manifest.sha256`.
- **Workstream 5:** Zero-Shot and Domain-Adaptation Experiments [SYNTHETIC BENCHMARK ONLY]:
  - Protocol: `PHASE5_DOMAIN_ADAPTATION_EXPERIMENTS.md`.
  - Execution engine: `run_phase5_domain_adaptation.py`.
  - Evaluated zero-shot transfer baseline from simulated ZTF (Macro-F1 0.1999) to simulated ATLAS (0.1748) and simulated DECam DDF (0.1893).
  - Unsupervised Correlation Alignment (CORAL) restores target Macro-F1 to 0.1969 (+0.0221 gain in simulation).
  - 3-Way Leave-One-Survey-Out (LOSO) cross-evaluation yields an 84.2% retention ratio (unmet preregistered threshold >= 90%, H3 not confirmed).
  - Physical features underperformed raw representations (delta Macro-F1 = -0.0408, bootstrap mean -0.0497, p=1.00; H2 falsified in benchmark).
  - Deliverables: `phase5_domain_adaptation_results.parquet` and `phase5_domain_adaptation_summary.json`.
- **Workstream 6:** Uncertainty, Robustness, and Scientific Validation [SYNTHETIC SIMULATION HARNESS RESULTS; EMPIRICAL CLAIMS PENDING]:
  - Protocol & Report: `PHASE5_UNCERTAINTY_AND_ROBUSTNESS_REPORT.md`.
  - Execution engine: `run_phase5_uncertainty_and_validation.py`.
  - Conformal prediction calibration evaluated in simulation (95.40% coverage); exchangeability breakdown under domain shift documented.
  - Covariate shift detection AUROC evaluated in simulation harness (AUROC = 0.9530; empirical survey AUROC pending live data).
  - Selective prediction risk-coverage analysis verifies error suppression from 33.33% to 6.98% at 50% coverage, but non-monotonic dip to 5.77% at 60% coverage due to discrete sample artifact (strict monotonicity unmet).
  - Deliverables: `phase5_hypothesis_testing_results.json` and `phase5_uncertainty_metrics.parquet`.
- **Workstream 7:** Independent Audit, Reconciliation, and Manuscript Release [RECONCILED DRAFT COMPLETED; EVIDENCE GAPS AUDITED]:
  - Full scientific research manuscript authored: `PHASE5_RESEARCH_PAPER.md` (ApJ / MNRAS publication format).
  - Master scientific reconciliation report: `PHASE5_FINAL_SCIENTIFIC_CLOSURE_REPORT.md`.
  - Master cryptographic evidence manifest: `phase5_final_evidence_manifest.json` sealing 28 assets (including Python execution engines), anchored in `phase5_final_evidence_manifest.sha256`.
  - Comprehensive closure test suite: `tests/test_phase5_final_closure.py`.
  - Phase 5 Reconciled Status: Software architecture and simulation benchmarks frozen; empirical cross-survey generalization remains open pending live external broker ingestion. Phase 6 empirical foundation modeling gated on real multi-survey acquisition.

---

## 6. Phase 6 Master Research Programme (M11–12)

**Theme:** Computational Universe & Experimental Intelligence  
**Lead Investigator:** Dev Mehta  
**Scientific Mission:** Can ASTRAEUS build computationally efficient, differentiable and physically constrained models of astrophysical systems, then use those models to design better scientific experiments?  
**Hardware Strategy:** 3-Tier Compute Architecture (Local RTX 5050 8 GB VRAM prototyping vs Cloud A100 training vs Institutional HPC supercomputing). Anti-overclaiming rules strictly enforced.  
**Epistemic Governance Invariant:** Phase 5 checkpoint `0601beb` permanently frozen. Gate 3 Spatial Test Set (586 objects, 1808 HEALPix pixels) quarantined under zero-touch covenant. Open empirical obligations tracked in `docs/phase6/PHASE5_CONTINUATION_LEDGER.md`.

### Flagship Research Questions:
1. **RQ16:** Can differentiable cosmological simulations support reliable parameter inference?
2. **RQ17:** Can physics-informed neural operators accelerate simulations while preserving physical fidelity?
3. **RQ18:** Can information-theoretic autonomous agents improve observational experiment selection?

### Three Decoupled Computational Engines:
1. **Cosmology Engine (`engines/cosmology`):**
   - Analytical background expansion $E(a)$, $H(a)$, comoving distance measures $\chi(z), d_A(z), d_L(z)$, conformal time $\tau(a)$, and ODE linear growth factor $D_1(a)$ / growth rate $f(a)$.
   - 3D Particle-Mesh (PM) solver with trilinear Cloud-in-Cell (CIC) mass assignment and matching force interpolation guaranteeing exact momentum conservation ($\|\sum m_i \mathbf{g}_i\| \sim 10^{-13}$).
   - 3D discrete 7-point finite-difference Green's function FFT Poisson solver with zero-mean cosmological potential boundary condition ($G(\mathbf{0}) = 0$).
   - Symplectic Kick-Drift-Kick (KDK) leapfrog time integrator in comoving FLRW coordinates preserving phase space volume (Liouville's theorem) and evaluating Layzer-Irvine cosmic energy equation discrete residuals.
   - Exact analytical benchmarks: 1D Zel'dovich pancake plane wave collapse solution and Keplerian two-body circular orbit solver.
2. **Hydrodynamics Engine (`engines/hydrodynamics`):**
   - Physics-Informed Neural Operator (PINO) surrogate architecture combining Fourier Neural Operators (FNO) with conservative PDE residual losses (continuity, Euler, energy).
   - Data compatibility contracts for CAMELS-IllustrisTNG and CAMELS-SIMBA suites ($25^3\,(h^{-1}\mathrm{Mpc})^3$, $1000$ cosmological volumes across $(\Omega_m, \sigma_8, A_{\mathrm{SN1}}, A_{\mathrm{AGN1}}, \dots)$).
3. **Experiment Engine (`engines/experimental_design`):**
   - Bayesian experimental design with Expected Information Gain (EIG) utility objective: $U(d) = \mathbb{E}_{y|d} [ D_{\mathrm{KL}}(p(\theta|y, d) \parallel p(\theta)) ]$.
   - Gymnasium-compatible telescope scheduling environment with slew dynamics, airmass limits, weather degradation, and ToO alert preemption.
   - Constrained Proximal Policy Optimization (PPO) agent with safety Lagrangian multipliers.

### Sequential Workstreams & Progress:
- **Workstream 1: Scientific Computing Foundation [COMPLETE & AUDIT-CERTIFIED]:**
  - Foundational specs authored: `PHASE6_RESEARCH_PROTOCOL.md`, `PHASE5_CONTINUATION_LEDGER.md`, `PHASE6_HARDWARE_AND_COMPUTE_FEASIBILITY.md`, `PHASE6_NUMERICAL_FOUNDATION_SPEC.md`.
  - Modules implemented: `engines/cosmology/constants.py`, `engines/cosmology/background.py`, `engines/cosmology/mesh.py`, `engines/cosmology/integrator.py`, `engines/cosmology/analytic_tests.py`.
  - Comprehensive unit & benchmark test suite implemented: `tests/phase6/test_phase6_ws1_cosmological_foundation.py` (15 tests passing, verifying expansion functions, distance reciprocity, EdS growth $D_1(a)=a$, CIC mass conservation, Poisson dipole antisymmetry, comoving momentum conservation to $10^{-11}$, symplectic time-reversibility to $10^{-8}$, Layzer-Irvine residuals, Kepler invariants, and Zel'dovich pancake convergence with $L_1 < 2.0\%$ and $L_\infty < 5.0\%$).
  - Independent Scientific Audit checks remediated & certified: `tests/phase6/test_phase6_ws1_advanced_audit.py` (15 tests passing, verifying full Fourier discrete Laplacian operator symmetry across all modes, general N-body momentum conservation across random configurations and boundary translations, exact KDK leapfrog symplecticity $\|J^T \Omega J - \Omega\| < 10^{-9}$, PM grid quasi-symplecticity resolution study bounded by $\mathcal{O}(\Delta x^2)$, Zel'dovich fixed-particle mesh study and joint refinement study across $16^3, 32^3, 64^3$, and Layzer-Irvine 4-level refinement scaling across 5, 10, 20, 40 steps).
  - Full repository regression validation: 121/121 tests passing across all packages with zero regressions.
- **Workstream 2: Differentiable Cosmological Simulation & Parameter Recovery [WS2.1 COMPLETE]:**
  - Authorized milestone: `PHASE6_WS2_1_JAX_NUMERICAL_PARITY`.
  - Master scientific protocol authored & certified: `docs/phase6/PHASE6_WS2_RESEARCH_PROTOCOL.md` (formally qualifying CIC $C^0$ continuity and Clarke subgradient conventions at knots; spherical top-hat $\sigma_8$ window function normalization; baseline flat $\Lambda\mathrm{CDM}$ parameter freeze; $\Omega_m - \sigma_8$ Fisher information identifiability matrix rank condition; realistic volume and mode covariance; and empirical peak allocated VRAM profiling).
  - Native JAX numerical engine implemented: `engines/cosmology/jax_engine.py` (`JaxParticleMeshSolver3D`, `JaxSimulationState`, `JaxSymplecticKDKLeapfrog` supporting 64-bit float64 precision).
  - Parity test suite executed & certified: `tests/phase6/test_phase6_ws2_jax_parity.py` (8/8 gates passing, verifying machine-level forward numerical parity with NumPy reference at $< 10^{-12}$ on CIC deposit, FFT Poisson potential, central-difference acceleration, force interpolation, single-step KDK state, 15-step trajectory, physical invariants, and runtime execution).
  - Sub-workstream roadmap: WS2.1 (JAX Numerical Port [COMPLETE]), WS2.2 (Differentiable Dynamics [NEXT]), WS2.3 (Gradient Verification), WS2.4 (Cosmological Parameter Recovery), WS2.5 (Independent Scientific Audit).
- **Workstream 3: Neural Hydrodynamic Surrogate Modeling (PINO) [SCHEDULED]:**
  - Implement 3D FNO spectral convolution layers; CAMELS dataset loader; PDE residual physics loss.
- **Workstream 4: Bayesian Experimental Design & Autonomous Scheduling [SCHEDULED]:**
  - Gymnasium telescope scheduling environment; Mutual information / EIG estimator; Constrained PPO agent.
- **Workstream 5: Joint Multi-Engine Experiments & Benchmarks [SCHEDULED]:**
  - Execute pre-registered hypothesis tests H16–H21; evaluate speedup vs accuracy, EIG improvement, and posterior parameter recovery.
- **Workstream 6: Independent Audit, Paper Preparation, and Community Release [SCHEDULED]:**
  - Cryptographic evidence manifest; ApJ/MNRAS research manuscript; open-source release runbooks.
