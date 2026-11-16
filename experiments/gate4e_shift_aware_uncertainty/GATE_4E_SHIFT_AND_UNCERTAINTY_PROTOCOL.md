# ASTRAEUS Gate 4 — Work Package 4E: Shift-Aware Uncertainty & Calibration Protocol

**Author:** Dev Mehta  
**Date:** 2026-11-16  
**Status:** COMPLETE — SUBMITTED FOR FORMAL CLOSURE REVIEW  
**Work Package:** Gate 4E — Shift-Aware Uncertainty & Conformal Calibration  
**Deliverable ID:** E1 (`GATE_4E_SHIFT_AND_UNCERTAINTY_PROTOCOL.md`)  

---

## 1. Mathematical Formulation & Calibration Policy

Conformal prediction provides distribution-free, finite-sample guarantees on prediction set coverage, *provided that calibration and test data are exchangeable*. 

In real astronomical alert streams, observational conditions vary continuously: telescope cadence shifts, interstellar extinction reddens spectra, and survey depth fluctuates. Work Package 4E establishes a shift-aware uncertainty protocol to quantify how conformal calibration behaves when exchangeability is violated.

### Finite-Sample Quantile Specification:
- **Calibration Population ($N_{\text{calib}} = 562$):**  
  Constructed from an 80/20 stratified split of the Track 1 Training population ($N=2,808$). Strictly isolated from proper model fitting and completely disjoint from the quarantined Gate 3 test set ($N=586$).
- **Target Coverage ($1 - \alpha$):**  
  $1 - \alpha = 0.90$ ($\alpha = 0.10$).
- **Conformal Index ($k$):**  
  $$k = \lceil (N_{\text{calib}} + 1)(1 - \alpha) \rceil = \lceil 563 \times 0.90 \rceil = 507$$
  The critical non-conformity threshold $\hat{q}$ is the 507th order statistic (0-indexed rank $506$) of calibration non-conformity scores:
  $$s_i = 1 - \hat{P}(y_i \mid x_i)$$
- **Prediction Set Construction:**  
  For any test observation $x$, the prediction set $C(x)$ includes all classes whose predicted probability exceeds the critical threshold:
  $$C(x) = \left\{ c \in \mathcal{C} : \hat{P}(y = c \mid x) \ge 1 - \hat{q} \right\}$$

---

## 2. Decomposition of Domain Shift Factors

To diagnose coverage breakdowns, we evaluate performance across five empirical shift axes:
1. **Photometric Shift (Flux & Depth):**  
   Comparison between high-SNR ($m_{\text{peak}} < 18.0\text{ mag}$) and low-SNR ($m_{\text{peak}} \ge 18.0\text{ mag}$) transients.
2. **Temporal Cadence Shift:**  
   Comparison between dense sampling (median cadence $\le 3\text{ days}$) and sparse sampling (median cadence $> 3\text{ days}$).
3. **Temporal Horizon Shift:**  
   Comparison between rich light curves ($N_{\text{obs}} \ge 20$) and sparse light curves ($N_{\text{obs}} < 20$).
4. **Passband Shift:**  
   Dual-band ($g$ and $r$) vs. single-band alert streams.
5. **Cross-Survey Shift:**  
   Real-time difference alert stream vs. coadded archival deep imaging (ZTF DR24).

---

## 3. Mandatory Abstention Policy

Under distribution shift, a point prediction or over-broad prediction set ($|C(x)| > 3$) produces false scientific assurance. The protocol mandates explicit abstention under any of the following triggers:
- **Trigger T1 (Empty Set):** $|C(x)| = 0$ (Observation is an out-of-distribution anomaly).
- **Trigger T2 (Excessive Ambiguity):** $|C(x)| \ge 4$ (Model cannot isolate candidate to fewer than 4 classes).
- **Trigger T3 (Sparse Ingestion):** $N_{\text{obs}} < 5$ (Insufficient physical observations).
- **Trigger T4 (Single-Band Alert):** Observations present in only one filter band.
