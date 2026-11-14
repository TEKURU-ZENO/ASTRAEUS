# ASTRAEUS Real-Survey Validation v1

## Benchmark Provenance & Status Notice

- `experiments/phase1_baselines/` (`run_baseline_comparison.py`, `benchmark_results.json`) is
  preserved **untouched** strictly as **`SYNTHETIC_SMOKE_ONLY`** (`N=78` synthetic curves across
  `N=18` test examples; retired as evidence of astrophysical generalization).
- `experiments/real_survey_v1/` constructs the audited **10,000-candidate real-survey validation
  cohort** (`real_survey_taxonomy_v1.0`) enforcing Sprint 2B P0 invariants:
  - **P0.1 (`L1–L5`):** Strict separation of `L1` traceable spectroscopic TNS (`Class_ADS_bibcodes`)
    from `L2` reported classifications lacking traceable bibcodes, `L3` independent external
    catalogs (ESA Gaia DR3 QSO/Variability), `L4` broker/photometric labels (excluded from primary
    ground truth), and `L5` unclassified `AT` (`is_supernova=None`, never negative). Host redshift
    alone never promotes an object into `L1` or `L2`.
  - **P0.2 (4-Entity Identity Graph):** Separates `PersistentSkyLocus`,
    `PhysicalTransientEpisode`, `CatalogReportRecord`, and `HostCounterpartRecord` using
    epoch-aware episode clustering with zero cross-split sky-locus or episode leakage.
  - **P0.3 (5-Timestamp & `AVAILABILITY_TIME_UNKNOWN`):** Separates `t_observation`,
    `t_publication`, `t_ingestion`, `t_classification`, and `t_retrieval`. Retrospective ALeRCE
    and DR24 records without an independently observed broker alert publication timestamp carry
    `t_pub_utc=None` and `AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN`.
  - **P0.4 (Non-Broker External Contaminant Truth):** Primary AGN and Galactic variable-star
    contaminants are sourced from ESA Gaia DR3 external catalogs (`L3`) and TNS spectroscopic
    reports (`L1`), never ALeRCE `lc_classifier` outputs. Instrumental/subtraction artifacts are
    isolated in `DETECTION_ARTIFACT_TRACK`.

## Rebuilding the 10k Cohort (Without Running Sprint 3 Baselines)

```bash
python experiments/real_survey_v1/build_10k_cohort.py
```
