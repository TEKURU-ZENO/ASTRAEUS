# SYNTHETIC_SMOKE_ONLY — NOT SCIENTIFIC EVIDENCE

> [!WARNING]
> **RETIRED FROM SCIENTIFIC EVALUATION:**
> The benchmarks, scripts, and results contained in this directory (`experiments/archived_synthetic_smoke_benchmarks/`) are legacy synthetic unit fixtures. The historic **Macro-F1 = 1.00** achieved by Random Forest across $N=18$ synthetic test lightcurves (`rr_lyrae`, `supernova_ia`, `detector_artifact`) has been permanently retired as evidence of astrophysical generalization or operational capability.

---

## 1. Archival Purpose & Status

1. **Software Verification Only:**
   - Synthetic fixtures and test scripts in this directory are preserved strictly for low-level software verification, pipeline regression testing, and synthetic edge-case handling (e.g., negative difference flux handling, zero-imputation behavior, and missing data mechanics).
2. **Exclusion from Scientific Claims:**
   - **Zero Scientific Weight:** Synthetic benchmark scores are excluded from all scientific claims, model selection decisions, leaderboard summaries, and headline metrics for the ASTRAEUS project.
   - **No Replacement with Synthetic Metrics:** Synthetic scores will not be replaced with higher-fidelity synthetic results.
3. **Certified Real-Survey Replacement:**
   - All scientific evaluation of ASTRAEUS baseline models must be conducted exclusively against the audited **10,000-candidate real-survey cohort** (`data/raw/real_survey_v1/`) under the frozen **Three-Track Evaluation Protocol** codified in `packages/astraeus-bench/src/astraeus_bench/real_survey_benchmark_protocol.py`:
     - **Track 1:** Labelled Astrophysical Classification Benchmark ($N=4,135$ candidates across 8 real astronomical classes: 2,808 train, 741 validation, 586 spatial test).
     - **Track 2:** Open-Set Anomaly Detection & Ranking ($N=4,365$ L5 unclassified candidates).
     - **Track 3:** Detection Artifact & Subtraction Quality Discrimination ($N=1,500$ subtraction/instrumental artifacts).

---

## 2. Inventory of Archived Legacy Artifacts

| File | Legacy Role | Forensic Finding | Active Status |
| :--- | :--- | :--- | :--- |
| `benchmark_results.json` | Baseline comparison ($N=18$) | Macro-F1 = 1.00 on synthetic Gaussian fixtures | **RETIRED / ARCHIVED** |
| `run_baseline_comparison.py` | Phase 1 baseline script | Generates synthetic 3-class test set | **RETIRED / ARCHIVED** |
| `sprint4_benchmark_report.json` | Anomaly benchmark | Evaluated on synthetic anomaly injections | **RETIRED / ARCHIVED** |
| `run_sprint4_anomaly_benchmark.py` | Sprint 4 anomaly runner | Synthetic anomaly injection harness | **RETIRED / ARCHIVED** |
| `sprint5_transfer_report.json` | Cross-survey transfer report | Synthetic cross-survey perturbation test | **RETIRED / ARCHIVED** |
| `run_sprint5_cross_survey_transfer.py` | Sprint 5 transfer runner | Synthetic survey noise simulation | **RETIRED / ARCHIVED** |
| `day90_investigation_dossier.json` | Day 90 discovery dossier | Early prototype agent investigation run | **RETIRED / ARCHIVED** |
| `DAY90_FLAGSHIP_INVESTIGATION_DOSSIER.md` | Day 90 investigation report | Early prototype investigation report | **RETIRED / ARCHIVED** |
| `run_day90_discovery_dossier.py` | Day 90 dossier generator | Early prototype investigation script | **RETIRED / ARCHIVED** |

---

## 3. Mandatory Governance Directive

In accordance with the project governance lock:
- No file or script in this directory may be invoked by active training or evaluation pipelines.
- No metric from this directory may appear in external publications, dashboards, or project status summaries.
- Sprint 3 (Baseline Model Training) remains **STRICTLY FROZEN** until explicit authorization from the principal investigator.
