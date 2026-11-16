# ASTRAEUS Phase 5: Domain Adaptation & Cross-Survey Transfer Protocol

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-ADAPT-SPEC-001`  
**Milestone:** Phase 5 — Workstream 5 (Deliverable D5.1)  
**Governing Status:** Cross-Survey Experimental Protocol  

---

## 1. Executive Summary & Experimental Objectives

Machine learning models trained on time-domain observations from a single survey (the *source domain*, ZTF) invariably suffer performance degradation when applied to independent surveys (the *target domains*, ATLAS, DECam DDF, Pan-STARRS1). This degradation stems from three compounding distribution shifts:
1. **Instrumental & Passband Shift:** Variations in filter transmission profiles ($g, r, i$ vs. $c, o$).
2. **Cadence & Sampling Shift:** Differences in survey revisit time (ZTF 2-day cadence vs. ATLAS rolling daily vs. DECam deep drilling).
3. **Photometric Noise Floor & Sensitivity:** Disparities in aperture depth, sky background, and signal-to-noise distributions.

**Workstream 5** executes four controlled experimental investigations:
- **Experiment 5A — Zero-Shot Transfer Baseline:** Evaluating the frozen Gate 4 Champion Model (`CalibratedBalancedSubsampleRF`, hash `466c2752808b...`) directly on external surveys without target-domain parameter tuning.
- **Experiment 5B — Physical Feature Transfer Superiority (Hypothesis H2):** Demonstrating that 27 physical, filter-standardized features transfer with statistically superior Macro-F1 compared to raw uncalibrated representations.
- **Experiment 5C — Controlled Domain Adaptation Benchmarks (Hypothesis H3):** Evaluating unsupervised and few-shot adaptation methods (Feature Standardization, CORAL, Adversarial DANN, Supervised Fine-Tuning).
- **Experiment 5D — Leave-One-Survey-Out (LOSO) Cross-Evaluation:** Verifying bounded adaptation and certifying the absence of negative transfer on third held-out survey regimes.

---

## 2. Evaluated Adaptation Methodologies

### 2.1 Baseline 0: Zero-Shot Transfer
The frozen Champion Model is evaluated directly on target survey observations $\mathcal{D}_T$ without updating weights, calibration thresholds, or feature representations:
$$\hat{Y}_{\mathrm{zero\_shot}} = f_{\mathrm{champion}}(\mathbf{x}_T)$$

### 2.2 Method 1: Target Feature Standardization (Z-Score Re-Centering)
Standardizes target-domain features using source-domain reference statistics to compensate for first-order covariate shifts:
$$\tilde{\mathbf{x}}_T = \left( \frac{\mathbf{x}_T - \hat{\mu}_T}{\hat{\sigma}_T} \right) \odot \hat{\sigma}_S + \hat{\mu}_S$$
Where $\hat{\mu}_T, \hat{\sigma}_T$ are computed strictly on the **Unlabeled Target Adaptation Split** ($N_{\mathrm{unlabeled}} = 50\%$).

### 2.3 Method 2: Correlation Alignment (CORAL)
Matches second-order feature statistics (covariances) between source and target domains without modifying classifier weights (Sun, Feng & Saenko 2016):
$$\mathcal{L}_{\mathrm{CORAL}} = \frac{1}{4 d^2} \| \mathbf{C}_S - \mathbf{C}_T \|_F^2$$
Where $\mathbf{C}_S, \mathbf{C}_T \in \mathbb{R}^{d \times d}$ are the sample covariance matrices. The linear whitening and re-coloring transformation is:
$$\mathbf{A} = \mathbf{C}_S^{1/2} (\mathbf{C}_T + \lambda \mathbf{I})^{-1/2}$$
$$\mathbf{x}_{\mathrm{CORAL}} = \mathbf{A} (\mathbf{x}_T - \hat{\mu}_T) + \hat{\mu}_S$$

### 2.4 Method 3: Supervised Few-Shot Fine-Tuning
Calibrates output probabilities using the **Labeled Target Adaptation Split** ($N_{\mathrm{labeled}} = 20\%$) using temperature scaling and isotonic regression, leaving base representation trees intact.

---

## 3. Leave-One-Survey-Out (LOSO) Protocol & Negative Transfer Guard

To test **Hypothesis H3** (Bounded Domain Adaptation), ASTRAEUS implements a 3-way cyclic transfer matrix across ZTF, ATLAS, and DECam DDF:
- Model trained on Survey $A$, adapted to Survey $B$, and evaluated on held-out Survey $C$.
- **Negative Transfer Invariant:** Domain adaptation to Survey $B$ is certified *valid* if and only if:
  $$\text{Macro-F1}_{A \to B \to C} \ge 0.90 \times \text{Macro-F1}_{A \to C}$$
  A performance drop $\ge 10\%$ on the independent held-out survey falsifies H3.

---

## 4. Cohort Partitioning & Governance Invariant

All adaptations adhere to the preregistered deterministic allocation algorithm:
- 50% Unlabeled Target Domain Adaptation (fitting $\hat{\mu}_T, \mathbf{C}_T$).
- 20% Labeled Target Domain Adaptation (calibrating few-shot heads).
- 30% Sealed Independent Evaluation Split ($N_{\mathrm{eval}} \ge 85$).

**Evaluation Invariant:** The sealed evaluation holdout is never seen during covariance estimation, feature re-centering, or fine-tuning.

---

**Approved by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026
