# ASTRAEUS Gate 4 — Work Package 4E: Scientific Closure & Operating Envelope Certification

**Author:** Dev Mehta  
**Date:** 2026-11-16  
**Status:** COMPLETE — SUBMITTED FOR FORMAL CLOSURE REVIEW  
**Work Package:** Gate 4E — Shift-Aware Uncertainty & Cross-Survey Robustness  
**Deliverable ID:** E7 (`GATE_4E_SCIENTIFIC_CLOSURE.md`)  

---

## 1. Work Package 4E Exit Verdict

### Overall Disposition: **WORK PACKAGE 4E SCIENTIFICALLY CLOSED**

Work Package 4E has achieved all pre-registered scientific requirements:
1. **Calibration Integrity Verified:**  
   Finite-sample conformal specification ($N_{\text{calib}}=562$, $k=507$ for $90\%$ nominal coverage, $\hat{q} = 0.5696$) verified with zero leakage into the 586 quarantined test records.
2. **Empirical Shift Reality Established:**  
   Proves empirically that nominal conformal coverage degrades from $90.0\%$ (on exchangeable calibration holdout) to $71.4\%$ on unshifted validation data, and down to $58.3\%$ under sparse cadence ($N_{\text{obs}} < 20$) and $62.3\%$ under cross-pipeline transfer.
3. **Validated Operating Envelope Certified:**  
   Defines hard operational boundaries and mandates automated abstention outside that envelope. Inside the operating envelope, empirical coverage recovers to **$76\text{--}78\%$** ($78.2\%$ for $N_{\text{obs}} \ge 20$, $76.7\%$ for dense cadence $\le 3\text{d}$), mitigating loss compared to sparse sampling. However, we explicitly retract any claim of guaranteed $90\%$ coverage under real shift.
4. **Processing-Domain Boundary Documented:**  
   Evaluates cross-pipeline / cross-product processing-domain transfer between ZTF alert difference photometry and ZTF DR24 archival coadded photometry, measuring a $-4.4\%$ accuracy drop. Full external cross-survey replication (e.g. Pan-STARRS1, DECam) remains uncertified in this release.

---

## 2. Deliverables Verification Table

| Artifact ID | File Path | Description | Verification Status |
| :--- | :--- | :--- | :--- |
| **E1** | `experiments/.../GATE_4E_SHIFT_AND_UNCERTAINTY_PROTOCOL.md` | Protocol defining split-conformal policy and shift taxonomy | Verified by Dev Mehta |
| **E2** | `experiments/.../phase4e_shift_factor_manifest.json` | Formal specification of domain shift factors and risks | Generated & Verified |
| **E3** | `experiments/.../phase4e_conformal_calibration_report.json` | Quantile k=507 specification, empirical coverage, Wilson CIs | Generated & Verified |
| **E4** | `experiments/.../phase4e_shift_stratified_metrics.parquet` | Stratified empirical coverage across cadence, epochs, survey | Generated & Verified |
| **E5** | `experiments/.../phase4e_cross_survey_transfer_results.json` | Cross-pipeline rank consistency vs. classification degradation | Generated & Verified |
| **E6** | `experiments/.../phase4e_negative_transfer_abstention_report.md` | Negative transfer analysis and operating envelope rules | Verified by Dev Mehta |
| **E7** | `experiments/.../GATE_4E_SCIENTIFIC_CLOSURE.md` | Scientific closure report and approval checklist | Signed by Dev Mehta |

---

## 3. Operational Safety Conditions

1. **Retraction of 90% Empirical Coverage Guarantee Under Shift:**  
   Production documentation must state: *"Nominal 90% conformal coverage holds on exchangeable calibration data. Under real survey domain shift, empirical coverage degrades to 58–65%. The validated operating envelope mitigates loss to 76–78% empirical coverage, but 90% coverage is not achieved under real shift. Automated abstention is mandatory outside the operating envelope."*
2. **Automated Abstention Implementation:**  
   The inference server must reject unhedged classification on single-band light curves, sparse cadence ($> 3.5\text{ days}$), or out-of-envelope candidates by outputting `ABSTAIN_OUT_OF_ENVELOPE`.
3. **Continuous Shift Monitoring:**  
   Production inference batches must continuously track empirical prediction set size $\mathbb{E}[|C(x)|]$. An increase in mean set size above $2.5$ triggers an automated alert for calibration drift.
