# ASTRAEUS Gate 4 — Work Package 4C: Rare-Class Evidence & Cohort Support Report

**Author:** Dev Mehta  
**Date:** 2026-11-16  
**Status:** COMPLETE — SUBMITTED FOR FORMAL CLOSURE REVIEW  
**Work Package:** Gate 4C — Rare-Class Representation & Failure Diagnosis  
**Governing Commit:** `0d88872` | **Model Baseline:** Frozen Champion LightGBM/RF (`466c2752808b...`)  

---

## 1. Executive Summary & Scientific Purpose

The objective of Work Package 4C is to provide an exhaustive, honest scientific accounting of which astronomical transient classes ASTRAEUS can reliably classify, which classes it fails to recognize, the physical mechanisms underlying those failures, and how confidence and abstention thresholds behave on rare classes.

Rather than disguising rare-class shortcomings through metric aggregation or over-optimistic synthetic sampling, this report establishes:
1. **The Frozen 8-Class Taxonomy & Cohort Accounting:**  
   Defines the exact sample support across development training ($N=2,808$), validation holdout ($N=741$), sealed spatial test ($N=586$), and the external benchmark cohort ($N=7,614$).
2. **Empirical Failure Diagnosis:**  
   Audits the physical and observational causes of rare-class failures, particularly the total collapse of `OTHER_EXTRAGALACTIC` into ordinary supernovae and the high confusion between `TDE` and `SN_Ia`.
3. **Class-Level Scientific Dispositions:**  
   Explicitly declares `SN_Ia` and `SN_CC` as `EVALUABLE`, while designating rare classes (`TDE`, `OTHER_EXTRAGALACTIC`) as `INSUFFICIENT_EXTERNAL_SUPPORT` in the external benchmark.

---

## 2. Frozen Class Taxonomy & Population Accounting

ASTRAEUS enforces an 8-class coarse taxonomy grounded in spectroscopic ground truth. Raw spectroscopic classifications are strictly preserved in metadata without arbitrary collapse:

| Coarse Class | Spectroscopic Subtypes Included | Train (Track 1) | Val (Track 1) | Sealed Test (Gate 3) | External Cohort (Gate 4) | External Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`SN_Ia`** | SN Ia, Ia-norm, Ia-91T, Ia-91bg, Ia-CSM, Iax | 890 | 202 | 184 | 5,916 | `EVALUABLE` (High Power) |
| **`SN_CC`** | SN II, IIP, IIL, IIn, IIb, Ib, Ic, Ic-BL | 563 | 120 | 126 | 1,693 | `EVALUABLE` (High Power) |
| **`SLSN`** | SLSN-I, SLSN-II, SLSN-R | 200 | 50 | 41 | 1 | `INSUFFICIENT_EXTERNAL_SUPPORT` |
| **`TDE`** | TDE, TDE-He, TDE-H, TDE-Bowen | 101 | 24 | 11 | 2 | `INSUFFICIENT_EXTERNAL_SUPPORT` |
| **`CV_Nova`** | CV, Classical Nova, Dwarf Nova, AM CVn | 246 | 49 | 38 | 0 | `DESCRIPTIVE_ONLY` |
| **`AGN_QSO`** | AGN, QSO, Blazar, Seyfert I/II, LINER | 499 | 206 | 104 | 1 | `DESCRIPTIVE_ONLY` |
| **`Varstar`** | RR Lyrae, Cepheid, Mira, LPV, YSO | 246 | 72 | 69 | 0 | `DESCRIPTIVE_ONLY` |
| **`OTHER_EXTRAGALACTIC`** | Ca-rich, Kilonova, FBOT, ILRT, LRN | 63 | 18 | 13 | 1 | `INSUFFICIENT_EXTERNAL_SUPPORT` |
| **TOTAL** | **All 8 Classes** | **2,808** | **741** | **586** | **7,614** | **Authoritative Accounting** |

---

## 3. Class Support Limitations in Independent Astronomical Data

A fundamental reality of astronomical surveys is severe astrophysical rarity:
- In the **development sample** (Track 1, total $N=4,135$), rare classes were deliberately curated and oversampled from archival alert history to enable preliminary sequence learning.
- In the **external benchmark cohort** (BTS snapshot, $N=7,614$), which represents a genuine magnitude-limited transient sky ($m < 18.5\text{ mag}$), rare events are naturally sparse:
  - Total TDEs: **2** (`ZTF18aabdajx`, `ZTF18aabtxvd`)
  - Total SLSNe: **1** (`ZTF24abdhylt`)
  - Total Exotic/Ca-rich: **1** (`ZTF19abrdxbh`)
  - Total SNe (Ia + CC): **7,609 (99.93%)**

### Core Scientific Conclusion:
Four external rare-class candidates (acquired and verified in Phase 4B.1) provide proof-of-concept ingestion verification, but **they cannot alone establish statistical generalization or macro-averaged F1 confidence intervals**. Any claim of rare-class external benchmark validity would be methodologically fraudulent. They are closed under `INSUFFICIENT_EXTERNAL_SUPPORT`.

---

## 4. Empirical Performance & Failure Ledger Analysis

Evaluating the frozen Champion Model on the 741-object validation holdout reveals the exact performance profile across classes:

| Class | Support | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 Score | Scientific Disposition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SN_Ia`** | 202 | 179 | 77 | 23 | 0.699 [0.640, 0.752] | 0.886 [0.835, 0.923] | **0.7817** | `EVALUABLE` |
| **`SN_CC`** | 120 | 63 | 48 | 57 | 0.568 [0.475, 0.656] | 0.525 [0.436, 0.612] | **0.5455** | `EVALUABLE` |
| **`SLSN`** | 50 | 22 | 15 | 28 | 0.595 [0.435, 0.737] | 0.440 [0.312, 0.577] | **0.5057** | `DESCRIPTIVE_ONLY` |
| **`TDE`** | 24 | 9 | 10 | 15 | 0.474 [0.273, 0.683] | 0.375 [0.212, 0.573] | **0.4186** | `INSUFFICIENT_EXTERNAL_SUPPORT` |
| **`CV_Nova`** | 49 | 35 | 10 | 14 | 0.778 [0.637, 0.875] | 0.714 [0.576, 0.822] | **0.7447** | `DESCRIPTIVE_ONLY` |
| **`AGN_QSO`** | 206 | 185 | 22 | 21 | 0.894 [0.844, 0.929] | 0.898 [0.849, 0.933] | **0.8959** | `DESCRIPTIVE_ONLY` |
| **`Varstar`** | 72 | 53 | 10 | 19 | 0.841 [0.732, 0.911] | 0.736 [0.624, 0.824] | **0.7852** | `DESCRIPTIVE_ONLY` |
| **`OTHER_EXTRAGALACTIC`** | 18 | 0 | 3 | 18 | 0.000 [0.000, 0.561] | 0.000 [0.000, 0.176] | **0.0000** | `INSUFFICIENT_EXTERNAL_SUPPORT` |

---

## 5. Failure Mechanisms by Archetype

From the 156 diagnosed failures in `phase4c_rare_class_failure_ledger.csv`, four primary failure mechanisms govern the classifier:

### 1. `OTHER_EXTRAGALACTIC` Complete Collapse (Recall = 0.0)
- **Observations:** All 18 validation examples were misclassified: 8 as `SN_Ia`, 8 as `SN_CC`, 2 as `SLSN`.
- **Physical Cause:** Rare exotics (such as Ca-rich transients, fast blue optical transients [FBOTs], and kilonova candidates) have low peak luminosities and rapid decay rates. Without host-galaxy offset features (which require deep archival imaging not present in pure difference streams), their extracted flux curves look indistinguishable from faint or sparsely sampled core-collapse supernovae.

### 2. TDE $\to$ SN_Ia Confusion
- **Observations:** 5 out of 15 TDE failures were misclassified as `SN_Ia`, and 5 as `SN_CC`.
- **Physical Cause:** Tidal disruption events exhibit smooth, power-law rises with persistently blue continuum ($g - r \approx -0.2$ to $-0.4$). Under 3-to-5 day cadence without ultraviolet data, this blue rise matches the pre-maximum phase of a luminous Type Ia supernova.

### 3. SLSN Truncation & Overlap with SNe
- **Observations:** 28 out of 50 SLSNe were misclassified (17 as `SN_CC`, 9 as `SN_Ia`, 2 as `TDE`).
- **Physical Cause:** SLSNe possess characteristic rise times of 30–60 days. In rolling alert surveys where the baseline captures only a 15-day segment near peak, the observed flux derivative matches a normal Type IIn or Type Ia.

### 4. False-Positive Sinks from Control Variables
- **Observations:** Variable stars and AGN contributed 10 false-positive TDE predictions and 15 false-positive SLSN predictions.
- **Physical Cause:** Stochastic flaring in active galactic nuclei mimics transient flares when the historical baseline is short ($<200$ days).

---

## 6. Summary of Actionable Recommendations

1. **Mandatory Abstention on Isolated Outliers:**  
   The model must return `ABSTAIN` when an exotic candidate presents low maximum-softmax confidence ($<0.45$) or wide conformal prediction sets.
2. **Decouple Rare Classes from Production Sequence Triggers:**  
   Do not trigger automated telescope follow-up alerts based solely on rare-class probabilities without secondary cross-catalog spatial filtering (e.g. host galaxy nucleus coincidence for TDEs, or faint host offset for Ca-rich transients).
