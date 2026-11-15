# ASTRAEUS Research Governance & Evaluation Lock

**Document Version:** 1.0.0  
**Effective Date:** 2026-11-15  
**Governing Authority:** Principal Investigator & System Audit Team  
**Status:** ACTIVE & BINDING  

---

## 1. Formal Scientific Governance Statement

> **"Synthetic benchmark results are excluded from all scientific performance claims, model-selection decisions, and generalization assessments. Synthetic fixtures may be retained solely for software verification and controlled unit testing. All scientific model performance must be reported against the designated real-survey evaluation protocol, with the spatially held-out test partition kept isolated until its authorized evaluation."**

---

## 2. Benchmark Retirement Directive

1. **Permanent Forensic Retirement of Synthetic Metrics:**
   - The legacy $N=18$ synthetic benchmark result (**Macro-F1 = 1.00**) across `rr_lyrae`, `supernova_ia`, and `detector_artifact` is permanently retired as evidence of astrophysical generalization.
   - All synthetic datasets, injection harnesses, and baseline logs have been quarantined in `experiments/archived_synthetic_smoke_benchmarks/` under explicit marker `SYNTHETIC_SMOKE_ONLY — NOT SCIENTIFIC EVIDENCE`.
   - Synthetic fixtures may be used only for software unit tests exercising edge-case arithmetic (e.g. negative difference flux, non-detection upper limits, missing observational bands).
   - Under no circumstances shall synthetic scores be cited as baseline performance or scientific progress in reports, publications, or leaderboards.

---

## 3. Active Scientific Benchmark Protocol

All empirical claims regarding ASTRAEUS representation learning and classification capabilities must be evaluated under the **Three-Track Real-Survey Evaluation Protocol** (`packages/astraeus-bench/src/astraeus_bench/real_survey_benchmark_protocol.py`) applied to the audited 10,000-candidate real-survey cohort (`real_survey_taxonomy_v1.0`):

### Track 1: Supervised 8-Class Astrophysical Classification
- **Total Support:** $N = 4,135$ candidates with verified ground-truth labels.
- **Taxonomy (8 Classes):**
  1. `SN_Ia` (Type Ia Supernovae)
  2. `SN_CC` (Core-Collapse Supernovae: II, Ib, Ic)
  3. `SLSN` (Superluminous Supernovae)
  4. `TDE` (Tidal Disruption Events)
  5. `CV_Nova` (Cataclysmic Variables & Classical Novae)
  6. `AGN_QSO` (Active Galactic Nuclei & Quasars)
  7. `Varstar` (Periodic & Aperiodic Variable Stars)
  8. `OTHER_EXTRAGALACTIC` (Other verified extragalactic transients)
- **Partitions:**
  - **Training Partition:** $N = 2,808$ candidates.
  - **Validation Partition:** $N = 741$ candidates.
  - **Spatially Held-Out Test Partition:** $N = 586$ candidates.
- **Reporting Metrics:**
  - Macro-F1 across 8 real classes with Wilson score 95% confidence intervals.
  - Per-class precision, recall, and F1.
  - Expected Calibration Error (ECE) and Brier score.
  - Three certified observation configurations: `ALERCE_ONLY`, `ZTF_DR24_ONLY`, `COMBINED`.

### Track 2: Open-Set Candidate Ranking & Anomaly Evaluation
- **Total Support:** $N = 4,365$ unclassified candidates (L5 evidence tier: $is\_supernova = None$).
- **Objective:** Evaluate abstention calibration, novelty ranking, and unlabelled candidate scoring.
- **Constraint:** Zero synthetic labels or artificial negative conversions.

### Track 3: Detection Artifact & Observation-Quality Task
- **Total Support:** $N = 1,500$ candidates from the isolated detection artifact track.
- **Objective:** Discriminate optical/subtraction artifacts (dipoles, bad pixels, diffraction spikes) from genuine astrophysical sources.
- **Reporting:** Kept strictly separate from Track 1 astrophysical metrics.

### Quarantined Candidates
- **Total Support:** $N = 79$ candidates with ambiguous or overlapping counterparts.
- **Treatment:** Quarantined and excluded from all headline metrics; audited separately.

---

## 4. Current Milestone Freezes & Restrictions

| Milestone / Component | Operational Status | Access Rule |
| :--- | :--- | :--- |
| **Sprint 2E.1 Governance Lock** | **ACTIVE** | Full enforcement of synthetic retirement and feature certification |
| **Sprint 3 (Baseline Model Training)** | **STRICTLY FROZEN** | Zero model fitting or evaluation allowed without explicit user authorization |
| **Track 1 Spatial Test Partition ($N=586$)** | **SEALED / ISOLATED** | No training, calibration, or exploratory evaluation on this partition |
| **Dataset B Pretraining** | **ON HOLD** | Pretraining remains paused pending downstream protocol verification |
| **Model Architecture** | **FROZEN** | No architectural additions permitted during validation sprints |

---

## 5. Certification & Compliance Verification

Any modification to this governance document requires written confirmation from the Principal Investigator. Automated CI and unit tests enforce adherence to these boundaries.
