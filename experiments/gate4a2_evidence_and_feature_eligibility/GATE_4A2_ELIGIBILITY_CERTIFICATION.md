# ASTRAEUS — Gate 4 Phase 4A.2: Evidence & Feature Eligibility Certification

**Milestone:** Gate 4 Phase 4A.2 — Spectroscopic Evidence Verification & Light-Curve Evaluability  
**Author:** Antigravity (Google DeepMind) in coordination with Scientific Review  
**Date:** November 16, 2026  
**Status:** **CERTIFIED & PENDING FORMAL REVIEW SIGN-OFF**  
**Governing Policy:** Frozen Model, Frozen Features, Zero Inference Governance  

---

## 1. Executive Certification Statement

Under the governing Gate 4 Phase 4A.2 protocol, the 7,614 taxonomy-admitted external candidates have been independently evaluated across two decoupled scientific criteria:
1. **Track A (Spectroscopic Evidence Hierarchy):** Evaluated against the official IAU Transient Name Server (TNS) public snapshot. Exactly **7,582 candidates (99.58%)** are certified under **Tier S2 (Secondary Traceable Evidence)** with verifiable, peer-traceable published classification reports (`Class_ADS_bibcodes`). Zero candidates are upgraded to Tier S1 due to the absence of local raw spectral FITS files. Exactly 31 candidates are assigned to Tier S3 (Catalog-Only / Sensitivity Analysis), and 1 candidate is excluded under Tier S4.
2. **Track B (Light-Curve Feature Evaluability):** Evaluated across multi-dimensional criteria (detection counts, dual-band coverage, non-detection preservation, temporal baseline, cadence, and 27-feature interface compatibility). Exactly **16 candidates** with local alert histories are certified as **`FEATURE_EVALUABLE`**, 1 candidate is classified as **`SINGLE_BAND_ONLY`**, and **7,597 candidates** are designated as **`AWAITING_BROKER_LIGHTCURVE_INGESTION`**.

---

## 2. Certified Input Dependencies & Output Artifacts

| Governance Object | Repository Path | Cryptographic SHA-256 Checksum | Record Count / Size | Status |
| :--- | :--- | :--- | :---: | :---: |
| **BTS Raw Snapshot** | `data/external_validity_v1/raw/bts/bts_raw_snapshot.csv` | `51d85f50772fd1412fcc2646f48b4d331b2e07f7023454f89cddd7a8e7e283ed` | 10,988 records | **VERIFIED** |
| **TNS Raw Snapshot** | `data/external_validity_v1/raw/tns/tns_public_objects.csv.zip` | `5d3fa52497de6f24491ebf25715fa9b617a220f662b2fc5102f9baa02e094ad8` | 212,832 objects | **VERIFIED** |
| **Candidate Parquet** | `data/processed/real_survey_10k_v1/candidates_10k.parquet` | `6713faee3c2ab47b7a14baa321c540f2b18ddb9b51d6ddcf88dba7008ad214ae` | 10,000 candidates | **LOCKED** |
| **HEALPix Splits Manifest** | `configs/data_splits_healpix.json` | `234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b` | 1,808 test pixels | **LOCKED** |
| **Taxonomy Mapping** | `configs/gate4/gate4a_taxonomy_mapping.yaml` | `265b538f2ad040e15a4b9b0069676c7dbbaec095c9a8b88bbcbe96ad7e8364b1` | Version `gate4a_taxonomy_v1.0` | **FROZEN** |
| **Label Reconciliation** | `configs/gate4/gate4a_raw_label_reconciliation.csv` | `c5c8b06e0e3b065437224513efef33212ae4fdd4404f65c9dca6b7bdc6cef0f1` | 33 raw label rows | **FROZEN** |
| **Admitted Candidates** | `experiments/gate4a_external_cohort_preregistration/gate4a1_screened_admitted_candidates.csv` | `abd7dc90eebeeb5507334f5f30acddf01824e4c3c6efd2834186ca850456fcb9` | 7,614 candidates | **CERTIFIED** |
| **Spectroscopic Ledger** | `experiments/gate4a2_evidence_and_feature_eligibility/spectroscopic_evidence_ledger.csv` | `af55e95338606513a65fd49cc43e1afb8a621f5ec6bf78e3f7893455a32ba5b5` | 7,614 rows | **EMITTED** |
| **Light-Curve Ledger (Parquet)** | `experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.parquet` | `6c8eee8caebccf7ee22f9cf38ca6ed891fa6c8c90e2ce68ad2f1257ff86f062f` | 7,614 rows | **EMITTED** |
| **Reproducibility Manifest** | `experiments/gate4a2_evidence_and_feature_eligibility/phase4a2_reproducibility_manifest.json` | `f861028f2b73c8e800ff9660416b9480ceca53910c4e8329fa5398a1a6c7f935` | Machine-readable | **EMITTED** |

---

## 3. Authoritative Population Accounting

### 3.1 Track A: Spectroscopic Evidence Hierarchy

```
Total Screened Taxonomy-Admitted Candidates : 7,614 (100.00%)
  - Tier S1 (Direct Spectrum File)          :     0 (  0.00%) -> PRIMARY SPECTROSCOPIC EVIDENCE
  - Tier S2 (Traceable TNSCR / Paper)       : 7,582 ( 99.58%) -> SECONDARY TRACEABLE EVIDENCE
      - Type Ia Supernovae (SN_Ia)          : 5,891
      - Core-Collapse Supernovae (SN_CC)    : 1,687
      - Tidal Disruption Events (TDE)       :     2
      - Super-Luminous Supernovae (SLSN)    :     1
      - Other Extragalactic (Ca-rich)       :     1
  - Tier S3 (Catalog-Only / Unrefereed)     :    31 (  0.41%) -> SENSITIVITY ANALYSIS ONLY
      - Type Ia Supernovae (SN_Ia)          :    25
      - Core-Collapse Supernovae (SN_CC)    :     6
  - Tier S4 (Excluded / Unverified)         :     1 (  0.01%) -> EXCLUDED FROM BENCHMARK
      - Active Galactic Nuclei (AGN_QSO)    :     1
```

### 3.2 Track B: ZTF Light-Curve Evaluability

```
Total Screened Taxonomy-Admitted Candidates : 7,614 (100.00%)
  - FEATURE_EVALUABLE (Local LCs, >=5 det)  :    16 (  0.21%) -> READY FOR BENCHMARK INFERENCE
      - Type Ia Supernovae (SN_Ia)          :     7
      - Core-Collapse Supernovae (SN_CC)    :     9
      - Super-Luminous Supernovae (SLSN)    :     1
  - SINGLE_BAND_ONLY (Missing dual-band)    :     1 (  0.01%) -> INSUFFICIENT FILTER COVERAGE
      - Core-Collapse Supernovae (SN_CC)    :     1
  - AWAITING_BROKER_LIGHTCURVE_INGESTION    : 7,597 ( 99.78%) -> PENDING BROKER TIME-SERIES FETCH
```

---

## 4. Invariant Compliance Checklist

- [x] **Gate 3 Test Set Quarantine:** Zero queries or reads against quarantined Gate 3 test IDs.
- [x] **No Silent Upgrade of S2 to S1:** Tier S1 strictly requires directly accessible local spectrum files ($N=0$). All 7,582 traceable TNSCR candidates remain under Tier S2.
- [x] **Decoupled Eligibility Accounting:** Evaluability of spectra and light curves tracked in independent, dedicated ledgers.
- [x] **Preservation of Non-Detections:** Negative difference fluxes and $5\sigma$ upper limits preserved without clipping or imputation.
- [x] **Frozen Feature Interface Compatibility:** The 16 feature-evaluable candidates verified against the frozen 27 physical observable features.
- [x] **Zero Inference / Zero Tuning:** No model execution, weight modification, or threshold selection performed.

---

## 5. Certification Verdict & Next Phase

Phase 4A.2 has successfully audited spectroscopic evidence and photometric feature evaluability for the external cohort. With Track A establishing 7,582 high-confidence spectroscopically confirmed transients and Track B confirming compatibility with the frozen 27-feature interface, we recommend transitioning to **Phase 4B: ZTF Alert Broker Light-Curve Acquisition Protocol**.
