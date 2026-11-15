# ASTRAEUS — Gate 4 Phase 4A: Cohort Independence & Governance Certification

**Milestone:** Gate 4 Phase 4A.1 — Cohort Independence & Quarantine Certification  
**Author:** Antigravity (Google DeepMind) in coordination with Scientific Review  
**Date:** November 15, 2026  
**Status:** **PHASE 4A.1 RECONCILIATION COMPLETED — SUBMITTED FOR SCIENTIFIC SIGN-OFF**  
**Repository Branch:** `master`  

---

## Executive Summary

Phase 4A.1 resolves all blocking findings identified in the Phase 4A Scientific Review:
1. **Arithmetic Consistency:** Explicitly distinguishes raw set sizes ($|A| = 1,712$, $|B| = 1,665$), the intersection ($|A \cap B| = 5$), the joint union ($|A \cup B| = 3,372$), and sequential incremental exclusions ($|B \setminus A| = 1,660$), proving exact consistency with the 7,616 screened candidates.
2. **Accurately Bounded Independence Claim:** Bounded strictly to what was empirically tested: *no detected overlap with accessible Train/Validation records, and spatial exclusion from the defined Gate 3 test footprint under the 60-arcsecond buffered geometry*.
3. **Four Distinct Cohort Totals:** Separates Screened Candidates ($7,616$), Taxonomy-Admitted Candidates ($7,614$), Spectroscopically Supported Candidates ($S1/S2$), and Feature-Evaluable Candidates.
4. **TDE Subtype Reconciliation:** Reconciled taxonomy-admitted raw TDEs ($N=70$) vs unmapped subtypes ($N=5$), proving that the 2 admitted disjoint TDE candidates (`ZTF18aabdajx` / `AT2022dbl` and `ZTF18aabtxvd` / `AT2018zr`) are approved `TDE` subtypes, with class status officially designated as **`NOT_EVALUABLE`** ($N < 5$).

---

## 1. Compliance Matrix: Resolution of the 5 Blocking Findings

| Item | Review Finding | Audited Resolution & Scientific Evidence | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Attrition Arithmetic Inconsistency** | Disentangled set theory vs sequential stages. $|A| = 1,712$, $|B| = 1,665$, $|A \cap B| = 5$, $|A \cup B| = 3,372$. Sequential Stage 2 incremental exclusions $= 1,660$. Exactly $10,988 - 3,372 = 7,616$. | **RESOLVED & VERIFIED** |
| **2** | **TDE Taxonomy Reconciliation** | Reconciled 70 approved raw TDE records ($66$ TDE + $4$ TDE-He) vs $5$ unmapped records. Proved exactly 2 admitted disjoint TDE events exist; class designated `NOT_EVALUABLE` ($N < 5$). | **RESOLVED & VERIFIED** |
| **3** | **Bounding Independence Claims** | Replaced overbroad "universal cohort independence" with mathematically accurate bounded claim: *zero detected overlap with accessible Train/Val records and 60'' spatial buffer exclusion*. | **RESOLVED & BOUNDED** |
| **4** | **Four Distinct Cohort Totals** | Separated Screened ($7,616$), Taxonomy-Admitted ($7,614$), Spectroscopically Supported ($S1/S2$), and Feature-Evaluable candidates. Isolated the 2 quarantined records (`-`, `Other`). | **RESOLVED & CODIFIED** |
| **5** | **Statistical Power Nuance** | Discarded coarse $N \ge 30$ rule as proof of power; mandated separate Wilson score intervals for precision/recall and stratified bootstrap for F1/Macro-F1/Balanced Accuracy. | **RESOLVED & CODIFIED** |

---

## 2. Complete Cryptographic Manifest & Audit Trail

| Governance Object | Repository Path | Cryptographic SHA-256 Checksum | Record Count / Size | Status |
| :--- | :--- | :--- | :---: | :---: |
| **BTS Raw Snapshot** | `data/external_validity_v1/raw/bts/bts_raw_snapshot.csv` | `51d85f50772fd1412fcc2646f48b4d331b2e07f7023454f89cddd7a8e7e283ed` | 10,988 records (1.34 MB) | **IMMUTABLE RAW** |
| **TNS Raw Snapshot** | `data/external_validity_v1/raw/tns/tns_public_objects.csv.zip` | `5d3fa52497de6f24491ebf25715fa9b617a220f662b2fc5102f9baa02e094ad8` | 212,833 objects (14.7 MB) | **IMMUTABLE RAW** |
| **Raw Acquisition Manifest** | `data/external_validity_v1/provenance/raw_acquisition_manifest.json` | `b0d143471f599c65823a28ae2404b357bfff90ab9fb781d7b249428d4f6961a9` | 1,475 bytes | **CERTIFIED** |
| **Taxonomy Mapping** | `configs/gate4/gate4a_taxonomy_mapping.yaml` | `265b538f2ad040e15a4b9b0069676c7dbbaec095c9a8b88bbcbe96ad7e8364b1` | Version `gate4a_taxonomy_v1.0` | **FROZEN** |
| **Label Reconciliation** | `configs/gate4/gate4a_raw_label_reconciliation.csv` | `c5c8b06e0e3b065437224513efef33212ae4fdd4404f65c9dca6b7bdc6cef0f1` | Exactly 33 raw label rows | **FROZEN** |
| **HEALPix Splits Manifest** | `configs/data_splits_healpix.json` | `234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b` | 1,808 test pixels (NSIDE=32) | **LOCKED** |
| **Candidate Parquet** | `data/processed/real_survey_10k_v1/candidates_10k.parquet` | `6713faee3c2ab47b7a14baa321c540f2b18ddb9b51d6ddcf88dba7008ad214ae` | 10,000 historical candidates | **LOCKED** |
| **Audit Script** | `experiments/gate4a_external_cohort_preregistration/run_phase4a1_cohort_integrity_audit.py` | `949d09b3a84c498e7c859250aad7a2f97dfebbbd37e5a24de1fedbe85a18ea7d` | Deterministic audit script | **EXECUTABLE** |
| **Audit Metrics Output** | `experiments/gate4a_external_cohort_preregistration/gate4a1_audit_metrics.json` | `3db98a22aa98bfb2e5f2aadd017ba6caf21cea2e263e56e11c884072c5e54ab3` | Exact machine-readable results | **CERTIFIED** |
| **Integrity Test Suite** | `tests/test_phase4a1_cohort_integrity.py` | Passing (6/6 tests) | Automated pytest verification | **VERIFIED** |

---

## 3. Four Distinct Cohort Accounting Totals

```
Total 1: Geometrically & Identity-Screened Candidates : 7,616
  - Quarantined / Unclassified in Screened Set        :     2 (Raw types '-' and 'Other')
Total 2: Taxonomy-Admitted Candidates                 : 7,614
  - Type Ia Supernovae (SN_Ia)                        : 5,916 -> LARGE_SAMPLE_SUPPORT (N >= 30)
  - Core-Collapse Supernovae (SN_CC)                  : 1,693 -> LARGE_SAMPLE_SUPPORT (N >= 30)
  - Super-Luminous Supernovae (SLSN)                  :     1 -> NOT_EVALUABLE (N < 5)
  - Tidal Disruption Events (TDE)                     :     2 -> NOT_EVALUABLE (N < 5)
  - Other Extragalactic Transients                    :     1 -> NOT_EVALUABLE (N < 5)
  - Active Galactic Nuclei (AGN_QSO)                  :     1 -> NOT_EVALUABLE (N < 5)
  - Cataclysmic Variables (CV_Nova)                   :     0 -> NOT_EVALUABLE (N = 0)
  - Variable Stars (Varstar)                          :     0 -> NOT_EVALUABLE (N = 0)
Total 3: Spectroscopically Supported Candidates       : In Progress (S1/S2 Evidence Audit)
Total 4: Feature-Evaluable Candidates                 : Pending (Light-Curve Epoch Matching)
```

---

## 4. Final Scientific Certification Verdict

Phase 4A.1 has completed the reconciliation:
1. The 7,614 taxonomy-admitted candidates are proven to be free of detected overlap with accessible Train/Val records and geographically excluded from the buffered Gate 3 spatial footprint.
2. The cohort is formally recognized as an **independent, ZTF-derived external evaluation test bed for the SN_Ia and SN_CC tasks**, rather than a universal external-validity benchmark.
3. Model development, feature extraction, and inference remain 100% frozen awaiting formal Phase 4A.1 sign-off.
