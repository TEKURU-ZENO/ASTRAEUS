# ASTRAEUS Phase 2 — Astronomical Reference Corpus v1.0 Audit Report
## Stage 1 (Days 91–105): Identity Resolution, Provenance Hygiene & Leakage Audit
**Programme:** ASTRAEUS — Autonomous Computational Universe & Scientific Intelligence Ecosystem
**Verification UTC:** `2026-11-09T12:48:40.319217+00:00` | **Corpus Version:** `1.0.0`
**Audit Gate Status:** PASSED (CERTIFIED FOR FOUNDATION MODEL TRAINING)
**Operational Axiom:** *An anomaly is not a discovery, a prediction is not an observation, and an agent's confidence is not scientific evidence.*

---
## 1. Executive Audit Summary
In accordance with the Phase 2 Stage 1 mandatory exit criteria, Astronomical Reference Corpus v1.0 has been constructed and subjected to independent 7-tier multi-modal verification:

- **Total Reference Objects Audited:** 283
- **Spatial Split Distribution:** Train: 190 (67.1%), Validation: 47 (16.6%), Held-Out Test: 46 (16.3%)
- **Spatial Boundary Buffer Quarantined Records:** 55 objects isolated within 5.0" of conflicting HEALPix partitions
- **Identity & Alias Leakage:** **0** cross-split overlaps (Zero-Leakage Invariant Satisfied)
- **Spatial Partition Overlap:** **0** overlapping pixels across splits
- **Temporal Hygiene:** **0** future-time observations in training split
- **Cryptographic Integrity & HMAC Signatures:** **0** signature failures (100% verified authentic)
- **Frozen Evaluation Protocol SHA-256:** `0db2447383abddfd3e93863d4576ca9d4630b51adbe997a16a435676835ee619`

---
## 2. Seven-Tier Audit Verification Scorecard
| Audit Gate | Evidentiary Standard | Observed Outcome | Gate Verdict |
| :--- | :--- | :---: | :---: |
| **Gate 1: Canonical Identity & Aliases** | Zero cross-split canonical ID or cross-catalogue alias leakage | 0 ID / 0 Alias / 0 Dups | **PASSED** |
| **Gate 2: Spatial HEALPix Isolation** | 0% pixel intersection between train, val, test ($N_{\mathrm{side}}=32$) | 0 overlapping pixels | **PASSED** |
| **Gate 3: Boundary Buffer Quarantine** | Strict quarantine for objects within 5.0" of conflicting boundaries | 0 unquarantined buffer violations | **PASSED** |
| **Gate 4: Temporal Stratification** | No training observations after $t_{\mathrm{split}} = 60000.0\,\mathrm{MJD}$ | 0 temporal violations | **PASSED** |
| **Gate 5: Origin Integrity** | Strict labeling of ARCHIVAL, SYNTHETIC, SIMULATED, INJECTED | 0 mislabeled origins | **PASSED** |
| **Gate 6: Cryptographic Authenticity** | SHA-256 integrity digest + HMAC-SHA256 authority signature verification | 0 invalid signatures | **PASSED** |
| **Gate 7: Class Balance & Selection Function** | Documented support, median SNR, and cadence across splits | All 8 classes balanced | **PASSED** |

---
## 3. Population Representation & Split Breakdown
### Class Distribution per Split (Support Counts):
| Astronomical Class | Origin Type | Train Count | Val Count | Test Count | Total Support | Median SNR | Median Cadence (hr) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARTIFACT_DIPOLE** | SYNTHETIC/SIM | 24 | 5 | 6 | 35 | 12.0 | 77.7 |
| **FBOT** | ARCHIVAL/SIM | 25 | 8 | 3 | 36 | 12.2 | 61.3 |
| **SLSN** | SYNTHETIC/SIM | 22 | 6 | 7 | 35 | 11.7 | 67.2 |
| **SN_II** | SYNTHETIC/SIM | 25 | 6 | 4 | 35 | 12.2 | 60.4 |
| **SN_Ia** | ARCHIVAL/SIM | 27 | 2 | 7 | 36 | 12.0 | 66.8 |
| **SN_Ibc** | SYNTHETIC/SIM | 21 | 8 | 6 | 35 | 11.8 | 57.0 |
| **TDE** | ARCHIVAL/SIM | 22 | 8 | 6 | 36 | 11.7 | 64.2 |
| **VARIABLE_STAR** | SYNTHETIC/SIM | 24 | 4 | 7 | 35 | 12.0 | 58.3 |

---
## 4. Canonical Archival Reference Objects in Corpus v1.0
1. **AT2018cow (`FBOT`):**
   - Coordinates: $\alpha = 244.000917^\circ, \delta = +22.268031^\circ$ (ICRS, J2018.5)
   - Discovery Source: ATLAS (`ATLAS18qqn`), Discovery MJD: 58285.25 (June 16.28, 2018 UTC)
   - Host Galaxy: CGCG 137-068 ($z = 0.014145$), projected offset $d_{\mathrm{proj}} = 1.73\,\mathrm{kpc}$
   - Photometric Calibration: Native $g, r, i$ passbands, $ZP = 27.5\,\mathrm{mag\ AB}$, standardized `optical_pseudoflux`
   - Epistemic Tier: `directly_observed` with full 11-point provenance seal.
2. **SN2011fe (`SN_Ia`):**
   - Coordinates: $\alpha = 210.80227^\circ, \delta = +54.27083^\circ$ (ICRS, J2011.6)
   - Discovery Source: PTF (`PTF11kly`), Discovery MJD: 55797.2 (August 24, 2011 UTC)
   - Host Galaxy: Messier 101 ($z = 0.000804$), projected offset $d_{\mathrm{proj}} = 5.4\,\mathrm{kpc}$
   - Photometric Calibration: Native PTF $g, r$, peak apparent mag $9.98\,\mathrm{mag}$
   - Epistemic Tier: `directly_observed` with full 11-point provenance seal.
3. **AT2018zr (`TDE`):**
   - Coordinates: $\alpha = 119.22721^\circ, \delta = +34.26211^\circ$ (ICRS, J2018.2)
   - Discovery Source: ZTF (`PS18kh` / `ZTF18aabtxvd`), Discovery MJD: 58175.0 (March 2, 2018 UTC)
   - Host Galaxy: SDSS J075654.53+341543.6 ($z = 0.071$), nuclear offset $d_{\mathrm{proj}} = 0.07\,\mathrm{kpc}$
   - Photometric Calibration: Native ZTF $g, r$, characteristic blue power-law decay.
   - Epistemic Tier: `directly_observed` with full 11-point provenance seal.

---
## 5. Frozen Benchmark Evaluation Protocol Contract
To prevent methodological data leakage and ensure that foundation model pretraining does not tune on evaluation data, the evaluation protocol is cryptographically frozen:
- **Protocol Hash (SHA-256):** `0db2447383abddfd3e93863d4576ca9d4630b51adbe997a16a435676835ee619`
- **Frozen Epoch (UTC):** `2026-11-09T12:48:39.318968+00:00`
- **Held-Out Validation Support:** 47 entities
- **Held-Out Test Support:** 46 entities

### Frozen Benchmark Tasks:
- **`task_1_temporal_reconstruction` — Irregular Masked Temporal Autoencoding:**
  - Target Split: `test`
  - Primary Metric: `CRPS_Gaussian` (Continuous Ranked Probability Score under Gaussian predictive distribution.)
  - Secondary Metrics: MAE_flux, NLL_uncertainty
  - Baselines to Beat: Interpolation_Linear, ASTROMER_Base, GaussianProcess
- **`task_2_cross_survey_transfer` — Zero-Shot Cross-Survey Property Estimation:**
  - Target Split: `test`
  - Primary Metric: `Macro_F1` (Unweighted mean of per-class F1 scores across standard transient classes.)
  - Secondary Metrics: Expected_Calibration_Error, Abstention_Recall
  - Baselines to Beat: RandomForest_Engineered, XGBoost_Passbands
- **`task_3_rare_event_triage` — Rare Transient Discovery & Ranking:**
  - Target Split: `test`
  - Primary Metric: `Precision_At_Budget_K10` (Fraction of genuine high-priority transients among top 10 ranked candidates.)
  - Secondary Metrics: Top_K_Recall_K20, Artifact_FDR
  - Baselines to Beat: IsolationForest_Uncalibrated, Mahalanobis_Distance

---
## 6. Formal Phase 2 Stage 1 Exit Sign-Off
- **Identity Resolution & Provenance:** **CERTIFIED**
- **Spatial Zero-Leakage & Boundary Quarantine:** **CERTIFIED**
- **Class Balance & Photometric Integrity:** **CERTIFIED**
- **Evaluation Protocol Freezing:** **FROZEN & VERIFIED**
- **Authorization for Stage 2 (Temporal Foundation Model):** **APPROVED TO PROCEED**
