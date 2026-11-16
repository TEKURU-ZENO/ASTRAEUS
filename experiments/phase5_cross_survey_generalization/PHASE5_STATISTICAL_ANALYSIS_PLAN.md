# ASTRAEUS Phase 5: Statistical Analysis Plan & Hypothesis Testing Protocol

**Author:** Dev Mehta  
**Lead Investigator & Mission Director, Project ASTRAEUS**  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-STAT-001`  
**Milestone:** Phase 5 — Cross-Survey Generalization & Domain Robustness  
**Workstream:** Workstream 6 — Uncertainty, Robustness & Scientific Validation  

---

## 1. Foundational Statistical Governance Principles

To avoid the prevalent pitfalls of machine learning in time-domain astronomy, Phase 5 establishes four immutable statistical principles:

1. **Object-Level Resampling Invariant:**  
   All bootstrapping, permutation testing, cross-validation, and Jackknife resampling must be executed strictly at the **transient candidate object level** ($\text{ASTRAEUS\_ID}$), never at the individual photometric observation epoch level. Observations of the same supernova are temporally correlated; measurement-level bootstrapping creates false pseudo-replication and drastically underestimates statistical variance.
2. **Paired Statistical Comparisons:**  
   Wherever the same astrophysical event is observed concurrently by two or more surveys (e.g. ZTF and ATLAS), statistical evaluations must be strictly **paired**. Paired testing eliminates intrinsic astrophysical variance (luminosity differences between individual supernovae) and isolates pure survey-to-survey measurement and inference discrepancies.
3. **No Universal Generalization Claims:**  
   Evaluating transfer across ZTF, ATLAS, and DECam DDF provides empirical evidence of robustness across *those specific observational regimes*. Three surveys do not constitute three hundred independent draws from a universe of telescopes. Results will be reported strictly as transfer across the evaluated facilities, never as a universal guarantee across all conceivable instruments.
4. **Honest Confidence Intervals:**  
   All proportions, accuracies, coverages, precisions, and recalls will report exact two-sided 95% Wilson score intervals without continuity correction:
   $$\mathrm{CI}_{95}(\hat{p}, n) = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.95996)$$
   Indices will store strictly `[lower, upper]` bounds.

---

## 2. Hypothesis-by-Hypothesis Statistical Testing Protocol

### 2.1 Testing Hypothesis H1: Astrophysical Parameter Consistency
- **Research Question:** Do physical parameters inferred from independent surveys agree within formal measurement uncertainties?
- **Parameters Evaluated:**
  1. Bolometric peak flux density $F_{\mathrm{peak}}$ (in $\mu\mathrm{Jy}$ and nanomaggies).
  2. Pre-peak flux rise slope $dF/dt$ ($\mu\mathrm{Jy}/\mathrm{day}$).
  3. Post-maximum decline rate $\Delta m_{15}$ ($\mathrm{mag}$).
  4. Intrinsic rest-frame color index $(g - r)_0$ at peak.
  5. Fitted analytical light-curve timescale parameters ($t_{\mathrm{rise}}, t_{\mathrm{fall}}$).
- **Statistical Procedures:**
  1. **Measurement-Error-Weighted Orthogonal Distance Regression (ODR):**  
     Fits the linear relation $\theta_{S2} = \alpha + \beta \, \theta_{S1}$ accounting for heteroscedastic uncertainties $\sigma_{\theta, S1}$ and $\sigma_{\theta, S2}$ in both coordinates. Perfect consistency requires $\alpha = 0.0$ and $\beta = 1.0$.
  2. **Bland–Altman Agreement Analysis:**  
     Computes the paired differences $d_i = \theta_{S2, i} - \theta_{S1, i}$ against mean values $m_i = (\theta_{S2, i} + \theta_{S1, i}) / 2$. Reports mean systematic bias $\bar{d}$ and the 95% Limits of Agreement (LoA):
     $$\mathrm{LoA} = \bar{d} \pm 1.96 \, s_d$$
     Evaluates whether $> 95\%$ of matched objects lie within the LoA.
  3. **Spearman Rank Correlation ($\rho_s$):**  
     Measures rank-order preservation across surveys with 10,000 object-level bootstrap iterations to determine the 95% confidence interval $[\rho_{\mathrm{low}}, \rho_{\mathrm{high}}]$.
- **Falsification Threshold:**  
  Falsified if $|\bar{d}| / \sigma_{\mathrm{pop}} > 0.20$ or if more than $25\%$ of securely matched events deviate by $> 2.0\sqrt{\sigma_{S1}^2 + \sigma_{S2}^2}$.

### 2.2 Testing Hypothesis H2: Physical Features vs. Raw Representation Transfer
- **Research Question:** Do physically derived features transfer with lower performance degradation than uncalibrated, survey-specific representations?
- **Experimental Setup:**  
  On the identical matched target-domain evaluation split, evaluate:
  - Model A: 27 Physical Features (rise slope, decline rate, color evolution, duration, physical model fits).
  - Model B: Survey-specific raw representation (direct flux vectors, unnormalized temporal bins).
- **Primary Estimand:**
  $$\Delta \text{Macro-F1} = \text{Macro-F1}_{\mathrm{phys}} - \text{Macro-F1}_{\mathrm{raw}}$$
- **Primary Statistical Inference (Object-Level Paired Bootstrap):**  
  Execute $B = 10,000$ paired bootstrap resamples clustered strictly at the **astrophysical object level** ($ASTRAEUS\_ID$) across the evaluation partition.
  Construct the two-sided 95% Bias-Corrected and Accelerated (BCa) confidence interval:
  $$\mathrm{CI}_{95}(\Delta \text{Macro-F1}) = [\Delta_{\mathrm{lower}}, \Delta_{\mathrm{upper}}]$$
  **Directional Confirmation Criteria:** The physical harmonization representation is confirmed superior if the lower confidence bound exceeds the preregistered practical significance threshold:
  $$\Delta_{\mathrm{lower}} > +0.020$$
  Falsified if $\Delta_{\mathrm{lower}} \le 0.0$ or if empirical one-sided bootstrap $p = P(\Delta \text{Macro-F1} \le 0 \mid \text{Data}) \ge 0.05$.

- **Complementary Paired Correctness Analysis (McNemar's Test):**  
  Because Macro-F1 averages class-conditional F1 scores, a paired McNemar test on object classification correctness ($Y_i = \hat{Y}_i$) provides an orthogonal check on instance-level discordant predictions.
  Construct the $2 \times 2$ contingency table across the $N_{\mathrm{eval}}$ matched candidates:
  - $a$: Both models correct.
  - $b$: Model A (Physical) correct, Model B (Raw) incorrect.
  - $c$: Model A incorrect, Model B correct.
  - $d$: Both models incorrect.
  Computes the continuity-corrected McNemar statistic:
  $$\chi^2_{\mathrm{McNemar}} = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1$$
  Reported alongside odds ratio $\hat{\psi} = b / c$ and two-sided exact binomial $p$-value when $b + c < 25$. If multiple candidate sub-cohorts are evaluated, family-wise error is controlled via the Benjamini–Hochberg procedure at $\mathrm{FDR} \le 0.05$.

### 2.3 Testing Hypothesis H3: Survey-Specific Adaptation vs. Generalization
- **Research Question:** Does domain adaptation improve performance on the target survey without causing catastrophic degradation on other surveys?
- **Evaluated Adaptation Frameworks:**
  1. *Baseline 0:* Frozen Champion Model (Zero-Shot, No Adaptation).
  2. *Method 1 (Feature Normalization):* Target-domain feature standardization (z-score re-centering).
  3. *Method 2 (CORAL):* Correlation Alignment matching second-order covariance matrices:
     $$\mathcal{L}_{\mathrm{CORAL}} = \frac{1}{4d^2} \|C_S - C_T\|_F^2$$
  4. *Method 3 (Adversarial DANN):* Domain-Adversarial Neural Network enforcing survey-invariant latent representations.
  5. *Method 4 (Fine-Tuning):* Supervised adaptation on the Target Labeled Adaptation Split ($N_{\mathrm{labeled}} = 20\%$).
- **Leave-One-Survey-Out (LOSO) Cross-Validation:**  
  For three surveys $\{A, B, C\}$:
  - Train on $A$, adapt to $B$, evaluate on held-out $C$.
  - Train on $\{A, B\}$, evaluate on held-out $C$.
- **Falsification Threshold:**  
  Falsified if adaptation to Survey $B$ improves $B$ performance but degrades performance on independent Survey $C$ by $\ge 10\%$ relative to the zero-shot baseline ($\text{Macro-F1}_{A \to B \to C} < 0.90 \times \text{Macro-F1}_{A \to C}$).

### 2.4 Testing Hypothesis H4: Covariate Shift Detectability
- **Research Question:** Can an automated shift detector identify individual light curves whose classification reliability is degraded?
- **Candidate Shift Metrics:**
  1. **Mahalanobis Distance in Physical Latent Space:**  
     $$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \hat{\mu}_S)^T \hat{\Sigma}_S^{-1} (\mathbf{x} - \hat{\mu}_S)}$$
  2. **Maximum Mean Discrepancy (MMD):** Kernel-based two-sample distribution distance.
  3. **Cadence Dilation Factor:** Ratio of median observation gap $\Delta t_{\mathrm{target}} / \Delta t_{\mathrm{source}}$.
  4. **Photometric SNR Ratio:** Median detection significance relative to survey noise floor.
- **Evaluation Procedure:**  
  - Evaluate shift scores $s(\mathbf{x})$ as a binary predictor of classification error ($Y \neq \hat{Y}$).
  - Compute Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC).
  - Compute Rank Correlation between shift score and conformal prediction set size $|C(\mathbf{x})|$.
- **Falsification Threshold:**  
  Falsified if $\text{AUROC} \le 0.60$ for misclassification prediction, indicating the shift detector provides negligible predictive utility for downstream failure.

### 2.5 Testing Hypothesis H5: Calibrated Selective Abstention
- **Research Question:** Does withholding classifications under domain shift reduce empirical risk on accepted candidates?
- **Risk-Coverage Mathematical Formulation:**  
  For an abstention policy threshold $\tau \in [0, 1]$ accepting candidate $\mathbf{x}$ if uncertainty $u(\mathbf{x}) \le \tau$:
  $$\text{Coverage}(c) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[u(\mathbf{x}_i) \le \tau]$$
  $$\text{Risk}(c) = \frac{\sum_{i=1}^N \ell(\hat{Y}_i, Y_i) \, \mathbb{I}[u(\mathbf{x}_i) \le \tau]}{\sum_{i=1}^N \mathbb{I}[u(\mathbf{x}_i) \le \tau]}$$
- **Statistical Evaluation:**  
  - Plot Risk-Coverage Curve $R(c)$ across $c \in [0.10, 1.00]$.
  - Measure Area Under the Risk-Coverage Curve (AURCC).
  - Evaluate Conformal Set Coverage inside vs. outside the validated operating envelope.
- **Falsification Threshold:**  
  Falsified if the Risk-Coverage curve is non-monotonic or fails to reduce empirical error rate on accepted candidates ($R(c = 0.50) \ge R(c = 1.00)$).

---

## 3. Statistical Power & Sample Size Determination

### 3.1 McNemar Test Statistical Power Formulation
For a paired comparison on $N_{\mathrm{eval}}$ independent astrophysical objects, statistical power depends on the total discordant pair proportion $p_{\mathrm{disc}} = p_{01} + p_{10}$ and the odds ratio $\psi = p_{01} / p_{10}$ (where $p_{01}$ represents Model A correct while Model B incorrect, and $p_{10}$ represents Model A incorrect while Model B correct).

Under two-sided significance level $\alpha = 0.05$ ($z_{1-\alpha/2} = 1.95996$) and target power $1 - \beta = 0.80$ ($z_{1-\beta} = 0.84162$), the required sample size of independent transient objects is given by (Connor 1987; Lachenbruch 1992):

$$N_{\mathrm{required}} \ge \frac{\left( z_{1-\alpha/2} \sqrt{p_{\mathrm{disc}}} + z_{1-\beta} \sqrt{p_{\mathrm{disc}} - (p_{01} - p_{10})^2} \right)^2}{(p_{01} - p_{10})^2}$$

#### Power Sensitivity Table across Discordant Proportions and Effect Sizes:
| Discordant Proportion ($p_{\mathrm{disc}}$) | Odds Ratio ($\psi = p_{01}/p_{10}$) | $p_{01}$ | $p_{10}$ | $|p_{01} - p_{10}|$ | Required Evaluation Sample ($N_{\mathrm{eval}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | **2.0** | 0.200 | 0.100 | 0.100 | **84** |
| **0.25** | **2.2** | 0.172 | 0.078 | 0.094 | **78** |
| **0.25** | **2.0** | 0.167 | 0.083 | 0.083 | **101** |
| **0.20** | **2.5** | 0.143 | 0.057 | 0.086 | **76** |
| **0.20** | **2.0** | 0.133 | 0.067 | 0.067 | **126** |
| **0.15** | **3.0** | 0.112 | 0.038 | 0.075 | **75** |

An evaluation cohort size of $N_{\mathrm{eval}} \ge 85$ achieves $> 80\%$ power whenever the discordant rate $p_{\mathrm{disc}} \ge 0.25$ and $\psi \ge 2.1$, or when $p_{\mathrm{disc}} \ge 0.30$ and $\psi \ge 2.0$. If the observed discordant rate is lower ($p_{\mathrm{disc}} < 0.20$), power is preserved by evaluating the object-level paired bootstrap on $\Delta \text{Macro-F1}$ as the primary inferential tool.

### 3.2 Planned Empirical Cohorts:
- **Primary Matched Cohort (ZTF + ATLAS):** Target $N = 250\text{--}400$ secure matched supernovae.
- **Secondary Replication Cohort (ZTF + DECam DDF):** Target $N = 50\text{--}100$ matched transients.
- **High-Power SN Ia & SN CC Classes:** $> 100$ events per class, providing $> 90\%$ power for individual class-level comparisons.
- **Rare-Class Transparency:** Phase 5 explicitly re-asserts that rare classes (TDEs, SLSNe, FBOTs) have severely limited sample sizes ($N < 10$ across overlapping surveys). In accordance with Gate 4 findings, rare classes are evaluated under descriptive and open-set discovery protocols, not high-power confirmatory tests.

### 3.3 Preregistered Deterministic Cohort Allocation Algorithm:
To prevent post-hoc data snooping or tuning the partition based on observed model performance, the allocation algorithm is fixed *a priori* before inspecting any evaluation labels:

1. **Stratification & Deterministic Hashing:**
   Candidates are stratified by verified physical class ($SN\_Ia$, $SN\_CC$, $TDE$, $SLSN$, $OTHER$). Within each class stratum, candidate assignment is governed by deterministic SHA-256 modular hashing on `astraeus_id`:
   $$h_i = \operatorname{int}(\operatorname{SHA256}(\text{astraeus\_id})[:8], 16) \pmod{100}$$

2. **Allocation Rules:**
   - **Regime A ($N_{\mathrm{total}} \ge 334$):**
     - $0 \le h_i < 50$: Unlabeled Target Domain Adaptation (50%)
     - $50 \le h_i < 70$: Labeled Target Domain Adaptation (20%)
     - $70 \le h_i < 100$: Sealed Independent Evaluation Cohort (30%, yielding $N_{\mathrm{eval}} \ge 100$).
   - **Regime B ($N_{\mathrm{total}} < 334$):**
     To guarantee adequate statistical power for the evaluation endpoint, allocate:
     $$f_{\mathrm{eval}} = \max\left(0.30, \frac{\min(85, N_{\mathrm{total}})}{N_{\mathrm{total}}}\right)$$
     The evaluation threshold is $h_{\mathrm{eval}} = \lfloor 100 \times (1.0 - f_{\mathrm{eval}}) \rfloor$.
     The remaining fraction $(1.0 - f_{\mathrm{eval}})$ is split 60/40 between unlabeled adaptation and labeled adaptation.

All resampling, confidence interval estimation, and hypothesis tests are conducted strictly at the **astrophysical object level** ($ASTRAEUS\_ID$), never at the photometric-epoch level.

---

## 4. Statistical Analysis Certification

**Approved and Certified by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
Date: November 16, 2026
