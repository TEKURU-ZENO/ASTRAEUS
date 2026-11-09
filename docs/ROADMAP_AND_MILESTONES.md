# ASTRAEUS Roadmap & Milestone Verification Matrix
## Phased Multi-Year Development Programme & Gate Acceptance Criteria

**Document Identifier:** `ASTRAEUS-ROADMAP-001`  
**Classification:** Strategic Planning & Milestone Verification  
**Version:** 2.2.1 (Formal Scientific Sign-Off Post-Independent Audit v2.2.1)  

---

## 1. 15-Month Master Programme Timeline

```
Months 1-2    Months 3-4     Months 5-6     Months 7-8     Months 9-10    Months 11-12   Months 13-14   Month 15
┌───────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ Phase 1:  │ │ Phase 2:   │ │ Phase 3:   │ │ Phase 4:   │ │ Phase 5:   │ │ Phase 6:   │ │ Phase 7:   │ │ Phase 8:   │
│ Scientific│ │ Multimodal │ │ Knowledge  │ │ Multi-Agent│ │ Differenti-│ │ Experiment │ │ Full Closed│ │ Open-Source│
│ Data      │ │ Latent     │ │ Fabric &   │ │ Adversarial│ │ able Cosmic│ │ Intelligence│ │ Loop       │ │ Research   │
│ Foundation│ │ Foundation │ │ Graph RAG  │ │ Reasoning  │ │ Surrogates │ │ & RL Sched │ │ Integration│ │ Ecosystem  │
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
| **Phase 5 (M9-10)** | **Computational Universe** | JAX `pmwd` integration, PINO hydrodynamic surrogates trained on CAMELS and IllustrisTNG. | Neural surrogate achieves $>10^3\times$ speedup over full hydrodynamics with $<3\%$ relative field error. |
| **Phase 6 (M11-12)** | **Experimental Intelligence** | Constrained PPO telescope scheduler, Bayesian optimal experimental design (BED) follow-up planner. | RL scheduler delivers $\ge 20\%$ higher scientific Expected Information Gain over greedy dispatch under weather noise. |
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
- **Stage 1 (Days 91–105) — Astronomical Reference Corpus & Provenance Hygiene [COMPLETE & FORMALLY CERTIFIED - v1.0.4]:**
  Curated Astronomical Reference Corpus v1.0.4 across genuine archival ($N=3$), numerically simulated ($N=105$), and standardized synthetic ($N=175$) transient populations ($N_{\mathrm{total}}=283$). Enforced spatial HEALPix ($N_{\mathrm{side}}=32$) zero-overlap partitioning with non-geodesic curved boundary sampling (400 vertices/cell, analytical sagitta error bound $\delta \le 0.0021'' \ll 5.0''$), 2D tangent plane anisotropic covariance projection with conservative multi-edge corner handling, exact Clopper-Pearson binomial intervals ($[42.13\%, 99.64\%]$ for $6/7$ successes), object-level manifest diff stability (0 partition or class changes), and DAG provenance auditing (`DatasetBSampleLineage` and `ProvenanceGraphAuditor`) with cycle detection, transitive ancestry, and pilot Dataset B ingestion audit ($N=500$: 460 verified clean, 25 quarantined missing lineage, 15 rejected test overlaps, 0 cycles). All 7 audit gates passed; Parquet partitions verified with SHA-256 digests; authoritative protocol hash `7df44b73fb80148c14718104da07393699e1ad002a4116401fba79a65b221583` cryptographically sealed. Stage 1 formally certified; Stage 2 initialization approved.
- **Stage 2 (Days 106–120) — Astronomical Temporal Foundation Model (Track A) [INITIALIZATION APPROVED]:**
  Continuous-time positional encoding, masked observation autoencoding on irregular cadences, and uncertainty-aware latent projections.
  *Approved 5-Stage Workstream Schedule:*
  1. *Days 106–108 (Data & Protocol Reconciliation):* Frozen, reproducible input manifest & Dataset B DAG auditor.
  2. *Days 109–111 (Temporal Representation Design):* Sequence specification, continuous-time encoding & simple baselines (Linear, Gaussian Process, ASTROMER, Non-pretrained sequence model).
  3. *Days 112–115 (Initial Controlled Training):* Small-scale pretrained checkpoint on verified spatial split.
  4. *Days 116–118 (Comprehensive 3-Track Evaluation):*
     - *Track 1 (Temporal Representation):* Masked reconstruction (CRPS, MAE), temporal forecasting ($t > t_{\mathrm{cutoff}}$), and conformal calibration.
     - *Track 2 (Physical Transfer):* Zero-shot performance on unseen physical objects and unseen transient classes, decoupling simulated vs archival data.
     - *Track 3 (Survey Generalization):* Bidirectional zero-shot transfer across ZTF, Pan-STARRS1, and DECam under dilated cadences.
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
