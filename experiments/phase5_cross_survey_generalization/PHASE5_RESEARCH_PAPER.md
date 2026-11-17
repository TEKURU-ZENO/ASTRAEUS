# ASTRAEUS Phase 5: Framework for Cross-Survey Generalization, Photometric Harmonization, and Uncertainty Calibration in Time-Domain Astronomy

**Dev Mehta**$^{1,*}$, and the **Project ASTRAEUS Collaboration**  
$^{1}$*Lead Investigator & Mission Director, Project ASTRAEUS*  
$^{*}$*Correspondence: dev.mehta@astraeus-project.org*  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-PAPER-V1`  
**Milestone:** Phase 5 Scientific Synthesis (Reconciled)  
**Target Publication:** *The Astrophysical Journal* (ApJ) / *Monthly Notices of the Royal Astronomical Society* (MNRAS)  

---

## Abstract

Machine-learning sequence models for astronomical transient classification frequently fail when deployed on telescopes differing from their training surveys due to compounding instrumental, temporal, and photometric distribution shifts. We present the architectural framework, mathematical formulation, and simulation benchmark evaluation of **ASTRAEUS Phase 5**, an end-to-end system designed to preserve astrophysical parameter estimation, classification accuracy, and uncertainty calibration across disparate astronomical surveys.

Operating under strict governance invariants—including zero-touch quarantine of the Gate 3 test footprint (586 objects across 1,808 HEALPix pixels) and a permanently frozen Champion Model (`CalibratedBalancedSubsampleRF`, SHA-256 `466c2752808b...`)—we develop:
1. A **Temporal Harmonization Engine** standardizing heterogeneous timestamps to Barycentric Modified Julian Date on the TDB scale ($\mathrm{BMJD}_{\mathrm{TDB}}$) across 7 major observatories (Palomar, Haleakala, Mauna Loa, Sutherland, El Sauce, Cerro Tololo, Cerro Pachón) with fail-closed validation and sub-second precision.
2. A **Photometric Harmonization Framework** implementing a formal three-state transformation taxonomy, full $2 \times 2$ covariance propagation ($\mathbf{\Sigma}_{\mathrm{target}} = \mathbf{J} \mathbf{\Sigma}_{\mathrm{input}} \mathbf{J}^T + \mathbf{J} \mathbf{\Sigma}_{\mathrm{cal}} \mathbf{J}^T + \mathbf{\Sigma}_{\mathrm{model}} + \mathbf{\Sigma}_{\Delta t}$) with well-conditioned inversion ($\kappa(\mathbf{J}) \approx 2.85$), and strictly non-destructive preservation of negative difference-imaging fluxes.
3. A **Multi-Survey Domain Adaptation Simulation Benchmark** evaluating zero-shot transfer from ZTF to simulated ATLAS and DECam DDF distributions over a 150-candidate feasibility cohort ($N_{\mathrm{eval}} = 87$). We find that unsupervised Correlation Alignment (CORAL) restores target Macro-F1 from $0.1748$ to $0.1969$, while Leave-One-Survey-Out (LOSO) transfer yields an $84.2\%$ retention ratio (missing the preregistered $90\%$ threshold). Crucially, 27 physical summary features underperformed raw representations in this simulation ($\Delta \text{Macro-F1} = -0.0408$, bootstrap mean $-0.0497$, $p=1.00$), falsifying the physical transfer superiority hypothesis in this benchmark regime.
4. A **Shift-Aware Conformal Uncertainty and Abstention Harness**, analyzing conformal exchangeability breakdown under covariate shift and demonstrating that while selective abstention suppresses empirical error from $33.33\%$ to $6.98\%$ at $50\%$ coverage, discrete small-sample artifacts yield a non-monotonic step ($5.77\%$ error at $60\%$ coverage).

We distinguish between validated software architecture and empirical survey generalization, establishing the rigorous benchmark and acquisition protocols required before survey-scale foundation models can be deployed.

---

## 1. Introduction

The advent of wide-field synoptic time-domain surveys—most notably the Zwicky Transient Facility (ZTF; Bellm et al. 2019), the Asteroid Terrestrial-impact Last Alert System (ATLAS; Tonry et al. 2018), the Dark Energy Camera (DECam; Flaugher et al. 2015), and the Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST; Ivezić et al. 2019)—has transformed transient astronomy into a high-data-rate discipline. Automated machine learning (ML) architectures are indispensable for classifying supernovae (SNe), tidal disruption events (TDEs), kilonovae (KNe), and exotic transients within minutes of discovery.

However, astronomical classifiers almost universally exhibit **survey-bound brittleness**: an algorithm trained and calibrated on ZTF alerts typically suffers performance degradation when evaluated on ATLAS forced photometry or DECam alerts. This vulnerability arises from three compounding distribution shifts:
- **Passband Response Disparities:** The transmission profiles of intermediate-band optical filters (e.g. ZTF $g, r, i$) differ markedly from broad-band composite filters (e.g. ATLAS $c, o$). Non-physical interpolation across filters introduces significant artificial color artifacts.
- **Cadence and Sampling Variations:** Rolling cadence surveys observe transients with irregular, season-dependent intervals ranging from hours to weeks, breaking the uniform-sampling assumptions of classical neural sequence architectures.
- **Photometric Noise and Detector Response:** Differences in aperture size (0.5m ATLAS vs. 1.2m ZTF vs. 4.0m Blanco) generate distinct signal-to-noise ratio (SNR) distributions and depth limits.

Project **ASTRAEUS Phase 5** addresses this challenge through rigorous software engineering, physical time-scale standardization, full-covariance photometric transformation, and pre-registered falsification testing.

---

## 2. Scientific Quarantine Governance & Cohort Design

### 2.1 The Zero-Touch Scientific Quarantine Invariant
To ensure that evaluation represents genuine external generalization without data snooping, we enforce a strict spatial and identity quarantine inherited from Gate 3:
- The **586 sequestered test candidates** located within the **1,808 frozen HEALPix pixels** (`configs/data_splits_healpix.json`, SHA-256 `234bb08c...`) remain completely sealed.
- No Phase 5 routine, script, feature extraction, or model training has accessed or loaded these test observations. Historical metadata/identifier-only access conducted during Gate 4 closure is fully disclosed and creates zero authorization for downstream access.
- All Phase 5 pilot candidates were proven strictly disjoint from the 1,808 test pixels prior to cohort admission (`len(overlap) == 0`).

### 2.2 Frozen Champion Model Baseline
The production classifier (`CalibratedBalancedSubsampleRF`, serialized artifact SHA-256 `466c2752808be01e...`) is permanently immutable. It serves strictly as an immutable baseline; zero weights, trees, or hyperparameters were adjusted during zero-shot benchmarking.

### 2.3 Cohort Stratification & Preregistered Allocation
The feasibility cohort comprises $N = 150$ spectroscopically confirmed astrophysical transients spanning 5 physical classes ($96\,\mathrm{SN\_Ia}$, $50\,\mathrm{SN\_CC}$, $2\,\mathrm{TDE}$, $1\,\mathrm{SLSN}$, $1\,\mathrm{OTHER\_EXTRAGALACTIC}$). 

To protect statistical power under empirical yield regimes where $N_{\mathrm{total}} < 334$, the cohort was allocated using a preregistered deterministic SHA-256 modular hashing algorithm:
- **Sealed Independent Evaluation Holdout:** $N_{\mathrm{eval}} = 87$ ($58.0\%$), satisfying the power threshold $N_{\mathrm{eval}} \ge 85$ for object-level paired bootstrap evaluation on continuous $\Delta \text{Macro-F1}$. Secondary discrete McNemar testing remains underpowered at this sample size ($N_{\mathrm{req}} \approx 210\text{--}350$ required for $\psi \approx 2.0$).
- **Unlabeled Target Adaptation Set:** $N_{\mathrm{unlab}} = 38$ ($25.3\%$), used strictly for covariance alignment and feature re-centering.
- **Labeled Target Adaptation Set:** $N_{\mathrm{label}} = 25$ ($16.7\%$), used for few-shot probability calibration.

The sealed evaluation cohort was withheld from all adaptation fitting steps.

---

## 3. Photometric and Temporal Harmonization Framework

### 3.1 Temporal Standardization Engine
Heterogeneous archives report exposure timestamps across conflicting coordinate frames and time scales. Treating timestamps as raw numerical scalars introduces Roemer light-travel delays of up to $\pm 499\,\mathrm{seconds}$ ($\pm 0.00578\,\mathrm{days}$) across Earth's orbit.

The `TemporalHarmonizer` engine executes:
1. Physical time-scale transformations:
   $$\text{MJD}_{\mathrm{UTC}} \xrightarrow{+37\,\mathrm{s}} \text{MJD}_{\mathrm{TAI}} \xrightarrow{+32.184\,\mathrm{s}} \text{MJD}_{\mathrm{TT}} \xrightarrow{\text{Fairhead \& Bretagnon (1990)}} \text{MJD}_{\mathrm{TDB}}$$
2. Topocentric-to-barycentric light-travel delay calculation via Astropy's builtin Solar System ephemeris (`ephemeris="builtin"`, IAU/SOFA routines) accounting for topocentric geodetic coordinates across 7 major observatories.
3. Fail-closed error handling (unknown observatories and unsupported scales immediately raise `ValueError`) and stale-result invalidation (recomputing BMJD if coordinates or observatory metadata change).

### 3.2 Photometric Three-State Taxonomy
To avoid circular dependencies or synthetic artifacts, photometry is governed by a strict 3-state taxonomy:
- **State 1 (Synthetic Passband Convolution):** When a calibrated spectral energy distribution $f_\lambda(\lambda, t)$ is available, band-averaged flux densities are computed using the photon-counting detector convention:
   $$\langle F_\nu \rangle_P = \frac{1}{c} \frac{\int_0^\infty f_\lambda(\lambda, t) \, T_P(\lambda) \, \lambda \, d\lambda}{\int_0^\infty T_P(\lambda) \, \lambda^{-1} \, d\lambda}$$
   where $T_P(\lambda) = T_{\mathrm{atm}} \times T_{\mathrm{optics}} \times T_{\mathrm{filter}} \times \mathrm{QE}$ is the total system photon transmission.
- **State 2 (Empirical Color Polynomials with Full Covariance):** Synchronous positive detections ($F > 0$, $\mathrm{SNR} \ge 3.0$, $-0.4 \le g-r \le 2.0$) are transformed via empirical polynomials (Tonry et al. 2018) with the complete uncertainty model:
   $$\mathbf{\Sigma}_{\mathrm{target}} = \mathbf{J} \mathbf{\Sigma}_{\mathrm{source}} \mathbf{J}^T + \mathbf{J} \mathbf{\Sigma}_{\mathrm{cal}} \mathbf{J}^T + \mathbf{\Sigma}_{\mathrm{model}} + \mathbf{\Sigma}_{\Delta t}$$
   where $\mathbf{J} = \begin{pmatrix} 0.593 & 0.407 \\ 0.222 & 0.778 \end{pmatrix}$ has condition number $\kappa(\mathbf{J}) \approx 2.85$.
- **State 3 (Native Passband Retention):** Non-positive difference fluxes ($F \le 0$), low-SNR points, or non-detections remain strictly in their native passbands with symmetric Gaussian errors in $\mu\mathrm{Jy}$. Downstream architectures consume passband-specific positional tokens without cross-filter imputation.

---

## 4. Multi-Survey Domain Adaptation Simulation Benchmarks

Pending live broker network acquisition of ATLAS forced photometry and DECam imaging, Workstream 5 evaluated domain adaptation techniques within a **controlled synthetic domain-shift benchmark**. Target survey representations were generated by perturbing the 150-candidate physical representations under characteristic filter throughput shifts, rolling cadence distributions, and elevated photometric scatter.

### 4.1 Benchmark Results:
| Adaptation Framework | Target Domain | Evaluated Partition | Macro-F1 | Overall Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Source Baseline (ZTF)** | Simulated ZTF | $N_{\mathrm{eval}} = 87$ | **$0.1999$** | $67.82\%$ |
| **Zero-Shot Transfer (ATLAS)** | Simulated ATLAS | $N_{\mathrm{eval}} = 87$ | $0.1748$ | $66.67\%$ |
| **Zero-Shot Transfer (DECam)** | Simulated DECam DDF | $N_{\mathrm{eval}} = 87$ | $0.1893$ | $67.82\%$ |
| **Target Feature Standardization** | Simulated ATLAS | $N_{\mathrm{eval}} = 87$ | $0.1893$ | $66.67\%$ |
| **Correlation Alignment (CORAL)** | Simulated ATLAS | $N_{\mathrm{eval}} = 87$ | **$0.1969$** | $67.82\%$ |
| **Supervised Few-Shot Fine-Tuning** | Simulated ATLAS | $N_{\mathrm{eval}} = 87$ | $0.1583$ | $66.67\%$ |

### 4.2 Benchmark Analysis & Negative Findings:
1. **CORAL Covariance Alignment:** Unsupervised second-order covariance matching ($\mathcal{L}_{\mathrm{CORAL}} = \frac{1}{4d^2}\|\mathbf{C}_S - \mathbf{C}_T\|_F^2$) recovered target Macro-F1 from $0.1748$ to $0.1969$ ($+0.0221$ gain over zero-shot), demonstrating the engineering value of feature covariance alignment.
2. **Falsification of Hypothesis H2 (Physical vs. Raw Transfer):**  
   Contrary to initial expectations, 27 summary physical features underperformed raw unaligned representations:
   - Physical Feature Macro-F1: $0.1748$
   - Raw Representation Macro-F1: $0.2156$
   - Point Estimate Difference: $\Delta = -0.0408$
   - Paired Object Bootstrap ($B=10,000$): Mean $\Delta = -0.0497$ ($95\%$ BCa CI: $[-0.1171, 0.0000]$, $p = 1.00$).
   - McNemar Test: 0 cases where physical was correct and raw wrong; 3 cases where raw was correct and physical wrong ($p=0.25$).
   *Astrophysical Interpretation:* Aggregating multi-band time-series into 27 summary scalars discards temporal phase coherence and fine-grained intra-night cadence information that the non-linear tree ensemble leveraged in raw representations. Under synthetic noise, summary parameter estimators degraded faster than dense flux vectors. **Hypothesis H2 is falsified in this benchmark.**
3. **Hypothesis H3 (LOSO Retention):** Adapting from ZTF to ATLAS and testing on simulated DECam yielded a retention ratio of $84.2\%$ ($0.1594 / 0.1893$), failing the preregistered threshold of $\ge 90\%$. **Hypothesis H3 is not confirmed.**

---

## 5. Shift-Aware Uncertainty & Conformal Calibration in Simulation

### 5.1 Conformal Prediction Calibration & Exchangeability Breakdown
Using non-conformity scores $s_i = 1 - P(Y_i \mid \mathbf{x}_i)$ calibrated on $N_{\mathrm{calib}} = 562$ source instances at target coverage $1 - \alpha = 90\%$ ($k = 507$, $\hat{q} = 0.812$):
- **Empirical Coverage in Simulation:** **$95.40\%$** ($83/87$ objects covered; Wilson $95\%$ CI: $[88.77\%, 98.20\%]$).
- **Efficiency:** Mean prediction set size was $1.54$ classes.
- **Theoretical Limitation:** Split-conformal coverage guarantees assume exchangeability ($P_{\mathrm{source}} = P_{\mathrm{target}}$). Under domain shift, exchangeability is violated. While empirical coverage reached $95.4\%$ in this simulation due to conservative thresholding, this does not constitute a mathematical guarantee for empirical survey data without covariate-shift importance weighting.

### 5.2 Covariate Shift Detection (Hypothesis H4)
- In the simulation harness, the misclassification detection AUROC evaluated via an uncertainty-correlated perturbation score ($s_{\mathrm{shift}} = u(\mathbf{x}) + \epsilon$) was **$0.9530$** (falsification threshold $\le 0.60$).
- *Reconciliation Note:* A preliminary exploratory draft reported $0.724$; the final simulation run metric is $0.9530$. Because the shift score was generated using synthetic perturbations within the simulation harness rather than true physical latent Mahalanobis distances on actual ATLAS/DECam light curves, true empirical cross-survey shift detection AUROC remains unmeasured pending live survey data.

### 5.3 Calibrated Selective Abstention (Hypothesis H5)
- Full coverage error ($c = 1.00$): $33.33\%$.
- $50\%$ coverage error ($c = 0.50$): **$6.98\%$**.
- *Monotonicity Analysis:* While overall risk decreased substantially from $33.33\%$ to $6.98\%$, the curve dipped to $5.77\%$ at $c=0.60$ because the 9 additional accepted objects were all correctly classified. Because the curve is not strictly monotonic across all deciles, **Hypothesis H5 is not confirmed under its strict preregistered criterion.**

---

## 6. Pre-Registered Hypothesis Testing Outcomes (Reconciled)

| Hypothesis | Description | Evaluated Estimand / Statistical Test | Empirical Outcome | Pre-Registered Falsification Boundary | Scientific Disposition |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **H1** | Astrophysical Parameter Consistency | ODR regression slope & Bland-Altman mean bias $\bar{d}/\sigma$ | $\bar{d}/\sigma = 0.082$ (Prototype / Simulation) | Residual discrepancy $> 25\%$ | **EMPIRICAL EVALUATION PENDING** |
| **H2** | Physical Transfer Superiority | Object-Level Paired Bootstrap ($B=10,000$) on $\Delta \text{Macro-F1}$ | $\Delta = -0.0408$ (point estimate)<br>$\Delta_{\mathrm{boot}} = -0.0497$ [$95\%$ BCa: $-0.117, 0.000$] | $\Delta \le 0.0$ or $p \ge 0.05$ | **FALSIFIED IN BENCHMARK** |
| **H3** | Bounded Domain Adaptation | 3-Way LOSO Retention Ratio ($\text{ZTF} \to \text{ATLAS} \to \text{DECam}$) | $84.2\%$ | Retention $< 90\%$ | **UNMET CRITERION** |
| **H4** | Covariate Shift Detectability | Uncertainty-Correlated Perturbation Score | $\mathrm{AUROC} = 0.9530$ (simulation harness) | $\mathrm{AUROC} \le 0.60$ | **SIMULATION BENCHMARK** |
| **H5** | Calibrated Selective Abstention | Monotonic Risk Reduction across Deciles | $R(0.50)=0.0698 < R(1.00)=0.3333$, but dips at $c=0.60$ ($0.0577$) | Non-monotonic curve | **STRICT MONOTONICITY UNMET** |

---

## 7. Discussion & Astronomical Implications

1. **The Gap Between Simulation and Empirical Ingestion:** Building automated domain adaptation architectures requires separating algorithm prototyping from observational validation. While CORAL covariance alignment and conformal sets show promise in simulation, empirical verification requires live forced photometry with native telescope point spread functions, background subtraction artifacts, and true survey footprints.
2. **Summary Features vs. End-to-End Representations:** The falsification of Hypothesis H2 demonstrates that manual feature engineering into 27 summary scalars can lose critical discriminative signals present in raw photometric series. Future foundation model architectures must consume calibrated fluxes natively with passband and temporal positional encodings.
3. **Preservation of Negative Fluxes is Mandatory:** Difference-imaging residuals frequently fluctuate below zero due to sky Poisson noise. Clipping negative difference fluxes to zero or taking logarithms severely biases rise-time fitting and early shock breakout detection. Retaining negative fluxes natively in State 3 preserves statistical unbiasedness.

---

## 8. Data and Software Availability

All software packages, specifications, and data artifacts are open-source and reproducible:
- **Repository:** `https://github.com/TEKURU-ZENO/ASTRAEUS`
- **Frozen Champion Checkpoint:** `sprint4h_champion_model.joblib` (SHA-256 `466c2752808be01e4f41bd8fdb9c0804a57b03b2c02c35e5172ccebda411fce5`)
- **Master Final Manifest:** `phase5_final_evidence_manifest.json` (SHA-256 sealed in `phase5_final_evidence_manifest.sha256`)

---

## 9. Conclusions

Project **ASTRAEUS Phase 5** successfully delivers a production-grade software architecture for temporal and photometric harmonization across astronomical surveys. By formalizing fail-closed barycentric time transformations ($\mathrm{BMJD}_{\mathrm{TDB}}$), rigorous photometric covariance propagation, and a transparent simulation benchmark, Phase 5 establishes the necessary foundation for cross-survey transient analysis. However, because target-domain light curves were evaluated under simulated shift rather than empirical broker retrieval, Phase 5 is preserved as a software, protocol, and simulation milestone. Empirical cross-survey validation remains the central gate before advancing to large-scale foundation models.

---

**References:**
- Bellm, E. C., et al. 2019, PASP, 131, 018002 (ZTF Overview)
- Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
- Connor, R. J. 1987, Biometrics, 43, 207 (McNemar Power)
- Fitzpatrick, E. L. 1999, PASP, 111, 63 (Dust Extinction)
- Fukugita, M., et al. 1996, AJ, 111, 1748 (AB Photometry)
- Ivezić, Ž., et al. 2019, ApJ, 873, 111 (Rubin LSST Overview)
- Oke, J. B., & Gunn, J. E. 1983, ApJ, 266, 713 (AB Standard)
- Schlafly, E. F., & Finkbeiner, D. P. 2011, ApJ, 737, 103 (Dust Maps)
- Smith, K. W., et al. 2020, PASP, 132, 085002 (ATLAS Photometry)
- Sun, B., Feng, J., & Saenko, K. 2016, AAAI, 30, 2058 (CORAL)
- Tibshirani, R. J., et al. 2019, NeurIPS, 32 (Conformal Prediction Under Covariate Shift)
- Tonry, J. L., et al. 2018, PASP, 130, 064505 (ATLAS System)
