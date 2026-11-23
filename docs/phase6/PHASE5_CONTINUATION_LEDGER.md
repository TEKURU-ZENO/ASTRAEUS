# ASTRAEUS Phase 5 Continuation Ledger & Governance Covenant
## Tracking Open Observational Obligations, Hypothesis Dispositions & Quarantine Invariants

**Document Identifier:** `ASTRAEUS-P5-CONT-LEDGER-001`  
**Classification:** Scientific Governance & Epistemic Audit Registry  
**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Baseline Git Checkpoint:** `0601beb`  
**Date:** November 23, 2026  
**Status:** ACTIVE & BINDING  

---

## 1. Executive Purpose

Phase 5 (*Independent Cross-Survey Generalization and Domain Robustness*) successfully established the software harmonization architecture, canonical contracts, bayesian cross-matching engine, and simulation benchmarks. However, formal astronomical sign-off was conditioned on resolving empirical cross-survey observations from live broker streams.

This Continuation Ledger serves as an **epistemic firewall**:
> **Core Governance Invariant:**  
> Transitioning to Phase 6 (*Computational Universe & Experimental Intelligence*) does **not** erase, dilute, or retroactively resolve Phase 5's empirical observational obligations. Phase 5 unresolved hypotheses remain open, tracked, and binding on future observational milestones.

---

## 2. Phase 5 Open Hypothesis Tracking Ledger

| Hypothesis | Description | Phase 5 Closure Status | Required Empirical Evidence for Final Sign-Off | Phase 6 Status |
| :---: | :--- | :---: | :--- | :---: |
| **H1** | **Astrophysical Parameter Consistency** | `EMPIRICAL EVALUATION PENDING` | Live forced photometry retrieval from ATLAS and NOIRLab DECam DDF; validation against dual falsification boundary: $|\bar{d}|/\sigma_{\mathrm{pop}} \le 0.20$ **AND** $\le 25\%$ outlier rate accounting for cross-band calibration covariance $\mathbf{\Sigma}_{\mathrm{cal}}$. | **SEQUESTERED (Gated on Live Broker Network)** |
| **H2** | **Physical Feature Transfer Superiority** | `FALSIFIED IN BENCHMARK` | Status preserved permanently as **FALSIFIED** in the synthetic benchmark ($\Delta = -0.0408$, bootstrap mean $-0.0497$, $95\%$ BCa CI: $[-0.0993, 0.0000]$, $p=1.00$). Any future claim of physical feature superiority requires an entirely new preregistered protocol. | **PERMANENTLY PRESERVED AS FALSIFIED** |
| **H3** | **Bounded Domain Adaptation** | `CRITERION UNMET` | Revisit Leave-One-Survey-Out (LOSO) retention ratio using genuine multi-survey observations once ATLAS/DECam cohorts are ingested; must satisfy $\ge 90\%$ retention. | **SEQUESTERED (Gated on Live Cohort)** |
| **H4** | **Covariate Shift Detectability** | `NOT VALID EVIDENCE (LABEL-CONDITIONED)` | Execute true out-of-distribution detection using feature-space Mahalanobis distance on unlabelled target light curves without label-conditioning. | **SEQUESTERED (Gated on Unlabelled Broker Stream)** |
| **H5** | **Calibrated Selective Abstention** | `CRITERION UNMET (SYNTHETIC & NON-MONOTONIC)` | Evaluate empirical risk-coverage curves on live cross-survey transient classifications; test for monotonic risk reduction without small-sample discrete binning artifacts. | **SEQUESTERED (Gated on Live Classifications)** |

---

## 3. Scientific Quarantine Covenant

The zero-touch scientific quarantine instituted at Gate 3 remains in full, permanent legal and technical effect:

1. **Quarantined Artifacts:**
   - 586 test transient candidate objects in `candidates_10k.parquet` (`assigned_split == 'test'`).
   - 1,808 frozen HEALPix pixels at $N_{\mathrm{side}}=32$, NESTED, ICRS (`configs/data_splits_healpix.json`, SHA-256 `234bb08c9098...`).
2. **Historical Access Disclosure:**
   - Historical metadata-only access occurred strictly during Gate 4 closure (`candidates_10k.parquet` identifier reconciliation).
   - Exactly zero Phase 5 scientific feature extraction, model tuning, inference, or evaluation accessed these records.
3. **Phase 6 Prohibition:**
   - Under no circumstances shall Phase 6 cosmological simulations, neural operator training sets, or reinforcement learning scheduling environments ingest, query, or reference the quarantined Gate 3 records.
4. **Authoritative Evidence Standard:**
   - Independent custodian attestation and geometric footprint disjointness across NSIDE=32 HEALPix pixels constitute the only permitted verification evidence.

---

## 4. Frozen Checkpoint Anchors

| Milestone | Git Checkpoint SHA | Author | Scope |
| :--- | :--- | :--- | :--- |
| **Gate 4 Final Closure** | `0f62a3b4582f24a082ef9b83733967d169be03d0` | Dev Mehta | Champion model frozen (`466c2752808b...`), Gate 3 quarantine certified. |
| **Phase 5 WS3/WS4 Transition** | `f53b881d18a667b1d470718a684c9cd6aa8a798e` | Dev Mehta | WS3.1 frozen, WS4 initialized, 18-asset transition manifest sealed. |
| **Phase 5 Final Closure** | `0601beb` | Dev Mehta | WS4 covariance invariants, WS6 circularity removed, 83/83 tests passing, 28-asset final manifest sealed. |

---

**Certified and Maintained by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 23, 2026
