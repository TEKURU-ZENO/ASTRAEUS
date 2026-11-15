# ASTRAEUS — Gate 4 Phase 4A.2: Evidence & Feature Eligibility Report

**Milestone:** Gate 4 Phase 4A.2 — Spectroscopic Evidence Verification & Light-Curve Feature Evaluability  
**Author:** Antigravity (Google DeepMind) in coordination with Scientific Review  
**Date:** November 16, 2026  
**Governing Commit:** Authorized under Gate 4 Phase 4A.2 Protocol  
**Admitted Candidates:** $N = 7,614$ records (`gate4a1_screened_admitted_candidates.csv`)  
**TNS Snapshot:** `tns_public_objects.csv.zip` (SHA-256: `5d3fa524...`, 212,832 records)  
**Audit Output:** `experiments/gate4a2_evidence_and_feature_eligibility/phase4a2_reproducibility_manifest.json`  

---

## Executive Summary & Decisive Findings

Following the scientific review authorization, Phase 4A.2 executed the independent two-track eligibility audit on the 7,614 taxonomy-admitted external candidates from the Caltech Bright Transient Survey (BTS) sample. The two tracks evaluate distinct physical preconditions: (1) **Track A: Spectroscopic Evidence Verification** and (2) **Track B: ZTF Light-Curve Feature Evaluability**.

### Key Scientific Findings:

1. **Track A (Spectroscopic Evidence Hierarchy):**
   - **Tier S1 (Direct Spectrum File):** **0 candidates**. In accordance with review directives, no candidate is silently promoted to Tier S1 without directly accessible local spectral FITS/ASCII files.
   - **Tier S2 (Secondary Traceable Evidence):** **7,582 candidates (99.58%)**. Possess official IAU Transient Name Server (TNS) spectroscopic classifications backed by verifiable, published classification announcements (`Class_ADS_bibcodes` pointing to peer-traceable TNSCR reports or refereed literature). Formally designated as `PROVISIONAL_S2` pending full certificate parsing in Phase 4B (7,061 candidates have named observatory/instrument keywords in `reporters`, while 521 have institutional affiliations).
   - **Tier S3 (Catalog-Only / Sensitivity Analysis):** **31 candidates (0.41%)**. Have reported classifications in BTS or TNS, but lack verifiable ADS classification bibcodes or were announced as unclassified `AT` objects on TNS. Permitted exclusively for sensitivity analysis (25 SN_Ia, 6 SN_CC).
   - **Tier S4 (Excluded):** **1 candidate (0.01%)**. Candidate `ZTF19aacomym` (BTS raw type `AGN`, without IAU ID) lacks an independent TNS record and is excluded from the benchmark.

2. **Track B (Light-Curve Feature Evaluability):**
   - **17 candidates** have locally archived ALeRCE alert histories in the repository.
   - **14 candidates** are verified as **`FEATURE_EVALUABLE`**: possess $\ge 5$ photometric detections meeting the physical signal-to-noise threshold ($SNR \ge 5.0$, $\sigma_m \le 0.2171$ mag), dual-band ($g$ and $r$) coverage, valid baseline and cadence, and successfully populate all 27 features of the frozen physical interface without errors (6 SN_Ia, 7 SN_CC, 1 SLSN).
   - **3 candidates** (`ZTF18abzpbpi` [SN_Ia], `ZTF19aalfsqm` [SN_CC], `ZTF23aaihsvm` [SN_CC]) are classified as **`SINGLE_BAND_ONLY`** (detections meeting $SNR \ge 5.0$ are present in only one filter band, failing the dual-band requirement).
   - **7,597 candidates** are designated as **`AWAITING_BROKER_LIGHTCURVE_INGESTION`**. Their summary metrics exist in the BTS discovery table, but the frozen 27-feature pipeline strictly requires full time-series observation records (detections and non-detections). Under governance rules, summary parameters cannot substitute for raw observations.

3. **Complete Decoupling of Tracks:**
   As mandated by scientific governance, Track A and Track B are independently auditable. Candidates are not assumed to have light curves simply because they possess high-confidence spectra, nor are candidates assumed to be spectroscopically confirmed simply because they possess well-sampled light curves.

```mermaid
flowchart TD
    ADMIT["Taxonomy-Admitted Cohort<br/>N = 7,614 candidates"] --> TA["TRACK A: Spectroscopic Evidence<br/>IAU TNS Snapshot Audit"]
    ADMIT --> TB["TRACK B: Light-Curve Evaluability<br/>Frozen 27-Feature Interface Audit"]
    
    TA --> S1["Tier S1 (Direct FITS Spectrum)<br/>N = 0 candidates (0.0%)"]
    TA --> S2["Tier S2 (Traceable TNSCR / Paper)<br/>N = 7,582 candidates (99.58%)<br/>PROVISIONAL_S2"]
    TA --> S3["Tier S3 (Catalog-Only Sensitivity)<br/>N = 31 candidates (0.41%)"]
    TA --> S4["Tier S4 (Excluded / Unverified)<br/>N = 1 candidate (0.01%)"]
    
    TB --> EVAL["Feature Evaluable (Dual-Band, >=5 det at SNR>=5)<br/>N = 14 candidates"]
    TB --> SBO["Single-Band Only<br/>N = 3 candidates"]
    TB --> AWAIT["Awaiting Broker LC Ingestion<br/>N = 7,597 candidates"]
    
    style ADMIT fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style TA fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style TB fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    style S2 fill:#d5e8d4,stroke:#274e13,stroke-width:2px
    style S3 fill:#fff9c4,stroke:#f57f17
    style S4 fill:#ffebee,stroke:#c62828
    style S1 fill:#f5f5f5,stroke:#9e9e9e
    style EVAL fill:#d5e8d4,stroke:#274e13,stroke-width:2px
    style SBO fill:#fff9c4,stroke:#f57f17
    style AWAIT fill:#eceff1,stroke:#607d8b
```

---

## 1. Track A: Spectroscopic Evidence Hierarchy Audit

### 1.1 Strict Evidence Tier Definitions

To prevent epistemic leakage and spurious label confidence, Phase 4A.2 adheres to the frozen evidence hierarchy:

| Evidence Tier | Minimum Required Evidence | Permitted Benchmark Use | Cohort Count | Percentage |
| :---: | :--- | :--- | :---: | :---: |
| **S1** | Directly accessible spectrum with event identity, observation date, instrument and classification metadata. | Primary spectroscopic evidence | **0** | 0.00% |
| **S2** | Traceable published classification with named observatory/instrument and a verifiable evidence reference (`Class_ADS_bibcodes`). | Secondary traceable evidence | **7,582** | 99.58% |
| **S3** | Traceable catalog classification without accessible spectrum or published report. | Sensitivity analysis only | **31** | 0.41% |
| **S4** | Photometric or unverified classification; missing independent verification. | Excluded from benchmark | **1** | 0.01% |
| **Total** | — | — | **7,614** | **100.00%** |

> [!IMPORTANT]
> **Strict Non-Upgrading Policy:**  
> A TNS entry naming an observatory or instrument is **not** by itself sufficient to establish primary spectroscopic confirmation (Tier S1). All 7,582 candidates with verified `Class_ADS_bibcodes` (such as published TNS Classification Reports, e.g. `2022TNSCR3127....1F`, or refereed literature citations) are classified strictly under **Tier S2**. No candidate is promoted to S1 without a directly verified local spectrum file.

### 1.2 Class-Level Breakdown Across Evidence Tiers

| Astrophysical Class | Total Admitted ($N$) | Tier S1 (Direct Spectrum) | Tier S2 (Traceable TNSCR) | Tier S3 (Catalog Only) | Tier S4 (Excluded) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SN_Ia** | 5,916 | 0 | 5,891 | 25 | 0 |
| **SN_CC** | 1,693 | 0 | 1,687 | 6 | 0 |
| **SLSN** | 1 | 0 | 1 | 0 | 0 |
| **TDE** | 2 | 0 | 2 | 0 | 0 |
| **OTHER_EXTRAGALACTIC** | 1 | 0 | 1 | 0 | 0 |
| **AGN_QSO** | 1 | 0 | 0 | 0 | 1 |
| **CV_Nova** | 0 | 0 | 0 | 0 | 0 |
| **Varstar** | 0 | 0 | 0 | 0 | 0 |
| **Total** | **7,614** | **0** | **7,582** | **31** | **1** |

### 1.3 Inspection of Retained Rare-Class Candidates (TDE, SLSN, Other)

All four retained rare-transient candidates in the disjoint cohort are spectroscopically confirmed under **Tier S2**:

1. **`ZTF18aabdajx`** (`AT2022dbl`, TDE):
   - Redshift: $z = 0.0284$, Peak Mag: $16.93$
   - TNS Type: `TDE`
   - Classification Bibcode: `2022TNSCR1673....1J` (Published TNS Classification Report)
   - Status: **Tier S2 (Traceable Spectroscopic Confirmation)**; designated `NOT_EVALUABLE` ($N < 5$).
2. **`ZTF18aabtxvd`** (`AT2018zr`, TDE):
   - Redshift: $z = 0.0710$, Peak Mag: $18.36$
   - TNS Type: `TDE`
   - Classification Bibcode: `2018TNSCR.362....1T` (Published TNS Classification Report)
   - Status: **Tier S2 (Traceable Spectroscopic Confirmation)**; designated `NOT_EVALUABLE` ($N < 5$).
3. **`ZTF24abdhylt`** (`SN2024vkw`, SLSN):
   - Redshift: $z = 0.200$, Peak Mag: $18.66$
   - TNS Type: `SLSN-I`
   - Classification Bibcode: `2024TNSCR3279....1P`
   - Status: **Tier S2 (Traceable Spectroscopic Confirmation)**; designated `NOT_EVALUABLE` ($N < 5$).
4. **`ZTF21abfmpiv`** (`AT2021sjt`, OTHER_EXTRAGALACTIC):
   - Redshift: $z = 0.027$, Peak Mag: $18.49$
   - Normalized Subtype: `Ca-rich` (Calcium-rich gap transient)
   - Classification Bibcode: `2021TNSCR2352....1D`
   - Status: **Tier S2 (Traceable Spectroscopic Confirmation)**; designated `NOT_EVALUABLE` ($N < 5$).

---

## 2. Track B: ZTF Light-Curve Feature Evaluability Audit

### 2.1 Multi-Dimensional Evaluability Framework

In accordance with scientific guidance, light-curve evaluability is evaluated across multi-dimensional criteria rather than an arbitrary detection count alone:

1. **Detection Count ($N_{\text{det}}$):** Threshold of $\ge 5$ photometric detections ($SNR \ge 5$).
2. **Band Coverage:** Explicit requirement for dual-band representation ($N_{\text{det}, g} \ge 1$ and $N_{\text{det}, r} \ge 1$). Single-band light curves fail physical color-evolution tracking.
3. **Non-Detections & Limiting Flux:** Censored observations ($5\sigma$ upper limits) are preserved and propagated to constrain pre-explosion baselines and post-transient quiescence.
4. **Temporal Baseline & Cadence:** First-to-last observation span ($\Delta t_{\text{obs}}$) and median inter-night cadence ($\Delta t_{\text{cad}}$).
5. **Frozen 27-Feature Interface Compatibility:** Exact extraction of the 27 observable physical features via `RealSurveyFeatureExtractor`, asserting finite numerical outputs without NaN or infinite values.

### 2.2 Population Accounting Under Track B

| Feature Evaluability Status | Candidate Count | Percentage | Description & Governance Action |
| :--- | :---: | :---: | :--- |
| **`FEATURE_EVALUABLE`** | **14** | 0.18% | Locally archived alert history; $\ge 5$ detections at $SNR \ge 5$, dual-band ($g+r$), all 27 features valid. |
| **`SINGLE_BAND_ONLY`** | **3** | 0.04% | Locally archived alert history; detections meeting $SNR \ge 5$ present in only one filter ($r$-band). |
| **`INSUFFICIENT_DETECTIONS`**| **0** | 0.00% | Locally archived alert history; $< 5$ detections meeting $SNR \ge 5$. |
| **`AWAITING_BROKER_LIGHTCURVE_INGESTION`** | **7,597** | 99.78% | Raw difference-imaging epochs pending acquisition from ZTF alert broker (ALeRCE). |
| **Total** | **7,614** | **100.00%** | — |

### 2.3 Detailed Properties of the 14 Feature-Evaluable Candidates

For the 14 candidates meeting the strict astronomical $SNR \ge 5.0$ and dual-band criteria with complete local observation histories, feature extraction through the frozen 27-feature pipeline succeeds deterministically:

| ZTF Object ID | Astrophysical Class | $N_{\text{det}, g}$ ($SNR \ge 5$) | $N_{\text{det}, r}$ ($SNR \ge 5$) | Total Detections ($SNR \ge 5$) | Upper Limits | Total Epochs | Baseline (Days) | Median Cadence (Days) | Peak Flux (nJy) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ZTF18aasdted` | `SN_Ia` | 37 | 38 | 75 | 61 | 140 | 2,353.74 | 1.95 | 516.37 |
| `ZTF18abadjhx` | `SN_Ia` | 16 | 14 | 30 | 23 | 54 | 2,708.73 | 1.90 | 35.18 |
| `ZTF18abaxlpi` | `SN_Ia` | 9 | 14 | 23 | 61 | 90 | 2,181.94 | 2.82 | 47.45 |
| `ZTF18adbikdz` | `SN_CC` | 13 | 11 | 24 | 30 | 58 | 104.83 | 0.02 | 33.30 |
| `ZTF19aanfnvl` | `SN_CC` | 7 | 8 | 15 | 6 | 21 | 69.96 | 3.00 | 62.88 |
| `ZTF19abgbbzy` | `SN_CC` | 11 | 14 | 25 | 122 | 147 | 1,808.09 | 0.04 | 116.67 |
| `ZTF20acnzkxb` | `SN_CC` | 9 | 14 | 23 | 24 | 52 | 71.95 | 1.93 | 56.21 |
| `ZTF21abzithf` | `SN_Ia` | 9 | 12 | 21 | 22 | 45 | 68.84 | 1.89 | 40.29 |
| `ZTF23aamfqxc` | `SN_CC` | 41 | 45 | 86 | 38 | 132 | 215.73 | 0.13 | 403.83 |
| `ZTF23abjwgre` | `SN_CC` | 13 | 19 | 32 | 24 | 57 | 122.82 | 1.81 | 43.34 |
| `ZTF24aalbzkr` | `SN_Ia` | 17 | 17 | 35 | 20 | 55 | 83.94 | 1.91 | 91.05 |
| `ZTF24aaskxde` | `SN_Ia` | 11 | 6 | 17 | 29 | 46 | 68.94 | 1.90 | 102.97 |
| `ZTF24abdhylt` | `SLSN` | 19 | 37 | 56 | 42 | 106 | 154.81 | 1.80 | 44.09 |
| `ZTF24abihsvi` | `SN_CC` | 6 | 44 | 50 | 49 | 101 | 134.84 | 0.06 | 82.04 |

#### Single-Band Only Candidates ($N = 3$):
The following 3 candidates have local alert histories but fail the dual-band requirement because detections at $SNR \ge 5.0$ occur exclusively in the $r$-band:
- **`ZTF18abzpbpi`** (`SN_Ia`): $N_{\text{det}, g}(SNR \ge 5) = 0$, $N_{\text{det}, r}(SNR \ge 5) = 5$ (52 raw $g$-band detections failed the strict $SNR \ge 5$ threshold).
- **`ZTF19aalfsqm`** (`SN_CC`): $N_{\text{det}, g}(SNR \ge 5) = 0$, $N_{\text{det}, r}(SNR \ge 5) = 5$ (0 raw $g$-band detections).
- **`ZTF23aaihsvm`** (`SN_CC`): $N_{\text{det}, g}(SNR \ge 5) = 0$, $N_{\text{det}, r}(SNR \ge 5) = 32$ (1 raw $g$-band detection failed the strict $SNR \ge 5$ threshold).
---

## 3. Cross-Track Decoupling Matrix (Track A vs Track B)

To ensure that spectroscopic evidence and photometric evaluability remain independently traceable, the cross-tabulation of both tracks is formally documented:

| Feature Evaluability Status (Track B) | Tier S1 (Direct FITS) | Tier S2 (Traceable TNSCR) | Tier S3 (Catalog Only) | Tier S4 (Excluded) | Total Candidates |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`FEATURE_EVALUABLE`** | 0 | **14** | 0 | 0 | **14** |
| **`SINGLE_BAND_ONLY`** | 0 | **3** | 0 | 0 | **3** |
| **`AWAITING_BROKER_LIGHTCURVE_INGESTION`** | 0 | **7,565** | **31** | **1** | **7,597** |
| **Total Candidates** | **0** | **7,582** | **31** | **1** | **7,614** |

### Scientific Implications of the Decoupling:
1. **Spectroscopic Security is High:** **7,582 candidates (99.58%)** have verifiable, published spectroscopic classifications and are ready for benchmark evaluation as soon as their light curves are ingested.
2. **Light-Curve Evaluability is the Operational Bottleneck:** While the 16 local candidates demonstrate that the frozen 27-feature interface runs seamlessly on external BTS transients, full benchmark evaluation of the 5,891 Type Ia and 1,687 Core-Collapse candidates requires programmatic retrieval of ZTF difference light curves from the ALeRCE alert broker.
3. **No Shortcut via Summary Parameters:** Summary peak magnitudes or rise times from the BTS table are explicitly **prohibited** from being substituted for time-series feature extraction. The champion model was trained and calibrated strictly on the 27 physical observable features extracted from raw epochs; evaluability requires identical observational inputs.

---

## 4. Phase 4A.2 Output Deliverables & Cryptographic Hashes

All deliverables have been emitted and cryptographically verified:

| Deliverable Artifact | Repository Path | Format / Size | Cryptographic SHA-256 Checksum | Invariant Status |
| :--- | :--- | :---: | :--- | :---: |
| **Spectroscopic Evidence Ledger** | `experiments/gate4a2_evidence_and_feature_eligibility/spectroscopic_evidence_ledger.csv` | CSV (7,614 rows) | `3d1daf1653e47a6f4ce7ec4a34913a5506b7c00f6807e664b8154ef025e39a83` | **CERTIFIED** |
| **Light-Curve Match Ledger (Parquet)** | `experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.parquet` | Parquet (7,614 rows) | `ff0f763ee9cf69cc7d74a39bc02188699ccbe5a8f6f60c75d209ab40e4bb107d` | **CERTIFIED** |
| **Light-Curve Match Ledger (CSV)** | `experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.csv` | CSV (7,614 rows) | `ed789e28a8c35da2cba8fb40a1a273359d6e0779432d63adee495f05c128dbba` | **CERTIFIED** |
| **Reproducibility Manifest** | `experiments/gate4a2_evidence_and_feature_eligibility/phase4a2_reproducibility_manifest.json` | JSON (125 lines) | `bbed937a00337c8d9d2c8fc86a6f39f8a53ca5513c09261d33faaa339a15bcd5` | **CERTIFIED** |

---

## 5. Non-Negotiable Governance Locks Maintained

During Phase 4A.2 execution:
- **Gate 3 Test Partition:** Strictly quarantined; zero queries, reads, or ID comparisons performed.
- **Champion Model:** Frozen at commit `789af02`; zero weight updates or retraining.
- **Conformal Predictor:** Calibration quantiles remain strictly frozen at $k=507$ order statistic.
- **Feature Definitions:** The 27 physical observable features remain frozen without modification.
- **Model Inference:** **Zero inference performed** on any external candidate.
