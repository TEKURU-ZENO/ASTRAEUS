# ASTRAEUS Phase 5: Uncertainty Calibration, Shift Detection & Hypothesis Testing Report

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-UNCERT-REPORT-001`  
**Milestone:** Phase 5 — Workstream 6 (Deliverable D6.1)  
**Governing Status:** Statistical Analysis, Simulation Benchmark & Falsification Audit (Reconciled)  

---

## 1. Executive Summary & Evaluation Context

A core scientific objective of Project ASTRAEUS Phase 5 is establishing whether transient classification models can produce **reliable uncertainty estimates** and **detect covariate shift** when confronted with cross-survey distribution changes.

> [!IMPORTANT]
> **Scientific Evidence Category Qualification:**
> Workstream 6 was executed using a controlled **Synthetic Domain-Shift Simulation Harness** constructed over the 150-candidate feasibility cohort ($N_{\mathrm{eval}} = 87$). As established during independent audit, these results represent algorithm and harness verification under simulated shift distributions, not validated empirical external-survey evaluations. Real multi-survey empirical evaluations remain pending live broker acquisition of ATLAS and DECam forced photometry.

Workstream 6 evaluates:
1. **Conformal Prediction Calibration under Domain Shift:** Evaluating marginal prediction set coverage ($1 - \alpha = 90\%$) using finite-sample conformal calibration ($k = 507$) and analyzing exchangeability breakdown under covariate shift.
2. **Covariate Shift Detection (Hypothesis H4):** Evaluating Mahalanobis distance in physical latent space to predict classification errors prior to downstream inference.
3. **Calibrated Selective Abstention (Hypothesis H5):** Constructing Risk-Coverage curves $R(c)$ to evaluate error reduction on accepted candidates under an uncertainty budget.
4. **Rigorous Hypothesis Testing:** Executing object-level paired bootstrap ($B = 10,000$ resamples) for $\Delta \text{Macro-F1}$ (H2) and paired McNemar tests across discordant classifications.

---

## 2. Conformal Prediction Calibration & Exchangeability Analysis

### 2.1 Calibration Methodology
Using non-conformity scores derived from the Champion Model output probabilities $s_i = 1 - P(Y_i \mid \mathbf{x}_i)$:
- Target error rate: $\alpha = 0.10$ ($90\%$ coverage).
- Quantile rank on calibration set ($N_{\mathrm{calib}} = 562$): $k = \lceil (N_{\mathrm{calib}} + 1)(1 - \alpha) \rceil = 507$.
- Calibrated non-conformity threshold: $\hat{q} = s_{(507)} = 0.812$.
- Prediction set for candidate $\mathbf{x}$:
  $$C(\mathbf{x}) = \{c \in \mathcal{Y} : P(c \mid \mathbf{x}) \ge 1 - \hat{q}\}$$

### 2.2 Simulation Harness Coverage Results
- **Empirical Coverage on Target Domain:** **$95.40\%$** ($83/87$ objects covered; Wilson $95\%$ CI: $[88.77\%, 98.20\%]$).
- **Prediction Set Efficiency:** Mean set size of $1.54$ classes (out of 8).

### 2.3 Critical Theoretical Qualification: Exchangeability Violation under Shift
Standard split-conformal marginal coverage guarantees rely fundamentally on the assumption that calibration and test examples are independent and identically distributed (exchangeability: $P_{\mathrm{source}} = P_{\mathrm{target}}$). 

Under cross-survey domain shift, this assumption is strictly violated ($\mathcal{P}_S(X, Y) \neq \mathcal{P}_T(X, Y)$). The observed $95.40\%$ coverage in this simulation harness is an empirical artifact of the conservative quantile $\hat{q} = 0.812$ and the specific simulated target probability distribution. It **cannot** be cited as a mathematical guarantee of 90% coverage on genuine external surveys. Rigorous distribution-free coverage under domain shift requires weighted split conformal prediction (Tibshirani et al. 2019) using estimated likelihood ratios $w(x) = dP_T(x) / dP_S(x)$, which will be integrated once empirical survey data is ingested.

---

## 3. Covariate Shift Detection (Hypothesis H4)

### 3.1 Mahalanobis Metric Formulation
To detect out-of-distribution observations without ground-truth labels, ASTRAEUS computes the Mahalanobis distance from source reference class centroids:
$$D_M(\mathbf{x}) = \min_{c} \sqrt{(\mathbf{x} - \hat{\mu}_c)^T \mathbf{\Sigma}_{\mathrm{pooled}}^{-1} (\mathbf{x} - \hat{\mu}_c)}$$

### 3.2 Misclassification Predictability & Audit Reconciliation
- In the primary simulation harness run, the misclassification detection AUROC was **$0.9530$** on the full evaluation set ($N_{\mathrm{eval}} = 87$).
- An exploratory early subset calculation yielded an AUROC of **$0.724$**.
- **Audit Reconciliation:** The discrepancy between $0.9530$ (in `phase5_hypothesis_testing_results.json`) and $0.724$ is hereby reconciled: $0.9530$ represents the final simulation run metric on the full 87-object partition, while $0.724$ was an un-reconciled preliminary draft figure.
- **Independence & Simulation Qualification:** In the committed simulation code, shift scores were generated using predictive uncertainty plus random perturbations rather than computed on genuine physical latent features from actual ATLAS/DECam light curves. True empirical cross-survey shift detection AUROC remains unmeasured pending live survey data.

---

## 4. Selective Prediction & Calibrated Abstention (Hypothesis H5)

### 4.1 Risk-Coverage Trade-Off
Given uncertainty score $u(\mathbf{x}) = 1 - \max_c P(c \mid \mathbf{x})$, candidates are accepted when $u(\mathbf{x}) \le \tau$:
$$\text{Coverage}(c) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[u(\mathbf{x}_i) \le \tau], \quad \text{Risk}(c) = \frac{\sum_{i=1}^N \ell(\hat{Y}_i, Y_i) \, \mathbb{I}[u(\mathbf{x}_i) \le \tau]}{\sum_{i=1}^N \mathbb{I}[u(\mathbf{x}_i) \le \tau]}$$

### 4.2 Empirical Risk Curve & Monotonicity Audit
The recorded risk-coverage values across the 10 deciles are:

| Coverage | Accepted Objects ($N_{\mathrm{acc}}$) | Error Count | Empirical Risk $R(c)$ | Monotonic Step? |
| :---: | :---: | :---: | :---: | :---: |
| **$10\%$** | 8 | 0 | $0.0000$ ($0.0\%$) | Baseline |
| **$20\%$** | 17 | 0 | $0.0000$ ($0.0\%$) | Flat |
| **$30\%$** | 26 | 0 | $0.0000$ ($0.0\%$) | Flat |
| **$40\%$** | 34 | 0 | $0.0000$ ($0.0\%$) | Flat |
| **$50\%$** | 43 | 3 | **$0.0698$ ($6.98\%$)** | Rise |
| **$60\%$** | 52 | 3 | **$0.0577$ ($5.77\%$)** | **VIOLATION (Dip)** |
| **$70\%$** | 60 | 4 | **$0.0667$ ($6.67\%$)** | Rise |
| **$80\%$** | 69 | 11 | $0.1594$ ($15.94\%$) | Rise |
| **$90\%$** | 78 | 20 | $0.2564$ ($25.64\%$) | Rise |
| **$100\%$** | 87 | 29 | **$0.3333$ ($33.33\%$)** | Rise |

### 4.3 Monotonicity Finding
- Overall, abstaining on the 50% most uncertain candidates substantially suppresses error from $33.33\%$ to $6.98\%$.
- However, the curve is **NOT strictly monotonic**: between $c=0.50$ and $c=0.60$, the 9 additionally admitted candidates happen to be all correctly classified, causing the cumulative error rate to dip from $3/43 = 6.98\%$ to $3/52 = 5.77\%$.
- **Typo Correction:** The prior text citation of $18.4\%$ was an un-reconciled draft placeholder; the actual recorded value in the machine-readable dataset is $6.98\%$.
- Under the strict preregistered criterion requiring monotonicity across all coverage deciles, **Hypothesis H5 is NOT satisfied**.

---

## 5. Formal Pre-Registered Hypothesis Testing Summary (Reconciled)

| Hypothesis | Investigated Property | Estimand / Test Metric | Empirical Outcome | Falsification Boundary | Scientific Disposition |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **H1** | Astrophysical Parameter Consistency | ODR residual slope & Bland-Altman LoA | $\bar{d} / \sigma = 0.082$ (Prototype / Simulation) | Residual bias $> 25\%$ | **EMPIRICAL EVALUATION PENDING** |
| **H2** | Physical Transfer Superiority | Object-Level Paired Bootstrap $\Delta \text{Macro-F1}$ | $\Delta = -0.0408$ (point estimate)<br>$\Delta_{\mathrm{boot}} = -0.0497$ [$95\%$ BCa: $-0.117, 0.000$] | $\Delta \le 0.0$ or $p \ge 0.05$ | **FALSIFIED IN BENCHMARK** |
| **H3** | Bounded Domain Adaptation | 3-Way LOSO Retention Ratio ($\text{ZTF} \to \text{ATLAS} \to \text{DECam}$) | $84.2\%$ | Retention $< 90\%$ | **UNMET CRITERION** |
| **H4** | Covariate Shift Detectability | Uncertainty-Correlated Perturbation Score | $\mathrm{AUROC} = 0.9530$ (simulation harness) | $\text{AUROC} \le 0.60$ | **SIMULATION BENCHMARK** |
| **H5** | Calibrated Selective Abstention | Monotonic Risk Reduction across Coverage Deciles | $R(0.50)=0.0698 < R(1.00)=0.3333$, but dips at $c=0.60$ ($0.0577$) | Non-monotonic curve | **STRICT MONOTONICITY UNMET** |

---

**Certified by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026
