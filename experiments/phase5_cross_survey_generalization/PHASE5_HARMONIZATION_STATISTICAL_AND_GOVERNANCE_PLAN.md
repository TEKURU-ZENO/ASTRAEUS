# ASTRAEUS Phase 5: Harmonization Statistical Evaluation & Governance Plan

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-HARMON-STAT-001`  
**Milestone:** Phase 5 — Workstream 4 (Deliverable D4.3)  
**Governing Status:** Formal Statistical Evaluation & Leakage Prevention Protocol  

---

## 1. Executive Summary & Objectives

Before harmonized time series are ingested into downstream machine learning pipelines (Workstream 5 and 6), the photometric and temporal transformations must undergo rigorous statistical validation. 

**The Core Scientific Question:**  
*Does photometric and temporal harmonization preserve intrinsic transient physics, or does it introduce systematic biases, spurious flux correlations, or artificial variance into the light curves?*

This plan preregisters:
1. Exact statistical procedures to evaluate transformed fluxes against synchronous multi-survey observations.
2. Formal mathematical criteria for Bland–Altman agreement, Orthogonal Distance Regression (ODR), and information preservation.
3. Strict governance invariants ensuring zero leakage from future epochs, sealed test sets, or unblinded evaluation splits.

---

## 2. Statistical Validation Procedures

```mermaid
flowchart TD
    Synch[Identify Synchronous Cross-Survey Observation Pairs: Delta_t <= 0.1 days] --> Check1[Procedure 1: Bland-Altman Limits of Agreement Analysis]
    Synch --> Check2[Procedure 2: Measurement-Error-Weighted Orthogonal Distance Regression]
    Synch --> Check3[Procedure 3: Information Preservation & Residual Gaussianity]

    Check1 --> Pass1{Mean Bias |d_bar| / sigma <= 0.20 & LoA Coverage >= 95%?}
    Check2 --> Pass2{ODR Slope beta in 1.00 +/- 0.05 & Intercept alpha in 0 +/- 2.0 uJy?}
    Check3 --> Pass3{Normalized Residuals N_sigma in [-3, +3] & No Tail Inflation?}

    Pass1 -->|All Passed| Cert[Harmonization Certified for Downstream Machine Learning]
    Pass2 -->|All Passed| Cert
    Pass3 -->|All Passed| Cert

    Pass1 -->|Failed| Rej[Transformation Falsified: Fall Back to State 3 Native Passband]
    Pass2 -->|Failed| Rej
    Pass3 -->|Failed| Rej

    style Cert fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Rej fill:#ffebee,stroke:#c62828,stroke-width:2px
```

### Procedure 1: Bland–Altman Agreement Analysis
- **Definition:** For $N_{\mathrm{sync}}$ synchronous epoch pairs ($F_{\mathrm{trans}}, F_{\mathrm{obs}}$), calculate:
  $$d_i = F_{\mathrm{trans}, i} - F_{\mathrm{obs}, i}, \quad m_i = \frac{F_{\mathrm{trans}, i} + F_{\mathrm{obs}, i}}{2}$$
- **Metrics:**
  - Mean systematic difference: $\bar{d} = \frac{1}{N} \sum d_i$
  - Sample standard deviation of differences: $s_d = \sqrt{\frac{1}{N-1} \sum (d_i - \bar{d})^2}$
  - 95% Limits of Agreement (LoA): $[\bar{d} - 1.96 s_d, \bar{d} + 1.96 s_d]$
- **Pass Threshold:** $|\bar{d}| / \sigma_{\mathrm{flux}} \le 0.20$ and $\ge 95\%$ of synchronous points lie within the LoA.

### Procedure 2: Measurement-Error-Weighted Orthogonal Distance Regression (ODR)
- **Model:** $F_{\mathrm{obs}} = \alpha + \beta \, F_{\mathrm{trans}}$ accounting for heteroscedastic uncertainties $\sigma_{\mathrm{obs}}$ and $\sigma_{\mathrm{trans}}$ on both axes.
- **Pass Threshold:**
  - Slope: $\beta \in [0.95, 1.05]$
  - Intercept: $|\alpha| \le 2.0\,\mu\mathrm{Jy}$ (consistent with zero within reference calibration accuracy).

### Procedure 3: Normalized Residual Analysis
- **Pull Metric:** $p_i = \frac{F_{\mathrm{trans}, i} - F_{\mathrm{obs}, i}}{\sqrt{\sigma_{\mathrm{trans}, i}^2 + \sigma_{\mathrm{obs}, i}^2}}$
- **Pass Threshold:** Pull distribution must have mean $\mu_p \in [-0.15, +0.15]$ and standard deviation $\sigma_p \in [0.85, 1.15]$, confirming well-calibrated error propagation.

---

## 3. Governance Invariants & Leakage Prevention Rules

1. **Gate 3 Test Set Quarantine (Zero Touch):**  
   Harmonization polynomials, empirical color transforms, and SED templates must never be fit, tuned, or calibrated using candidates located within the 1,808 quarantined Gate 3 HEALPix test pixels.
2. **Strictly Causal Prefix Filtering (No Future Leakage):**  
   When calculating spline fits, Gaussian Process interpolations, or rolling color estimates, the algorithm must evaluate light curves using strictly causal prefixes:
   $$t_{\mathrm{context}} \le t_{\mathrm{eval}}$$
   Observations at $t > t_{\mathrm{eval}}$ are strictly forbidden from informing the measurement or color at $t_{\mathrm{eval}}$.
3. **Fail-Closed Negative Flux Invariant:**  
   Under no circumstances may a negative difference flux be clamped, floored, converted to magnitude, or linearly interpolated across wavelengths. Non-positive fluxes strictly bypass color transforms and remain in State 3 (`NATIVE_PASSBAND_UNTRANSFORMED`).

---

**Approved by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026
