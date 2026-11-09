# ASTRAEUS Phase 2 — Astronomical Reference Corpus v1.0.2 Audit & Reconciliation Report
## Stage 1 Final Certification: Exact Spherical Boundary Geometry, Single-Object Buffer Quarantine, Triple-Cohort Benchmark Separation & Derivative Exclusion
**Programme:** ASTRAEUS — Autonomous Computational Universe & Scientific Intelligence Ecosystem
**Verification UTC:** `2026-11-09T13:43:38.764727+00:00` | **Corpus Version:** `1.0.2`
**Audit Gate Status:** PASSED (FORMALLY CERTIFIED FOR FOUNDATION MODEL BENCHMARKING)
**Operational Axiom:** *An anomaly is not a discovery, a prediction is not an observation, and an agent's confidence is not scientific evidence.*

---
## 1. Executive Protocol Hash Harmonization & Genealogy (P0)
To ensure complete scientific provenance and methodological transparency across all Phase 2 development releases, we provide the authoritative genealogical trace of evaluation protocol hashes:

| Protocol Milestone | Release Git Commit | Active Partitions (Train / Val / Test / Quar) | SHA-256 Protocol Digest | Methodological Status |
| :--- | :---: | :---: | :---: | :---: |
| **Milestone 1 (Pre-Segregation)** | `8ca48f7` | $190 / 47 / 46 / 0$ | `0db2447383abddfd3e93863d4576ca9d4630b51adbe997a16a435676835ee619` | Superceded (unsegregated boundary buffer) |
| **Milestone 2 (Inscribed Clamp Proxy)** | `f15d128` | $155 / 37 / 36 / 55$ | `24150b79e9bceb0dfd67bc2ce59068c5c10e6ed59fdaae302c29bdab2a0e9931` | Superceded (inscribed circle corner clamp) |
| **Milestone 3 (Interim Exclusion Draft)** | `49fe23a` | $155 / 37 / 36 / 55$ | `90e388ba6e57624ebd90c11cb451d682fb5d1da339e33a94f3640c4c9eb98892` | Superceded (retained corner clamp) |
| **Milestone 4 (Authoritative v1.0.2)** | `HEAD` | **190 / 47 / 45 / 1** | `19ce6a78dcc1da48d112884fc65cc9793defb7b6ff62ae71f896a9c718eb4218` | **AUTHORITATIVE CERTIFIED FREEZE** |

### Reproducible Verification Standard:
- **Serialization Standard:** Strict RFC 8785 Canonical JSON (`separators=(',', ':')`, sorted dictionary keys, UTF-8 encoded).
- **Bound Invariants:** The protocol hash cryptographically binds all 3 benchmark task definitions, 6 evaluation metrics, 47 validation object IDs, 45 held-out test object IDs, 1 quarantined object ID, and the explicit Dataset B cross-corpus derivative exclusion invariant.
- **Authoritative Protocol Hash:** `19ce6a78dcc1da48d112884fc65cc9793defb7b6ff62ae71f896a9c718eb4218`
- **Reproducible CLI Verification Command:**
  ```bash
  python experiments/phase2_reference_corpus/verify_protocol_hash.py
  ```

---
## 2. Corrected Spherical Cell-Edge Geometry & Quarantine Reconciliation (P0)
### Elimination of the Inscribed-Circle Approximation:
In v1.0.0 and v1.0.1, the spatial partition engine approximated boundary proximity using the inscribed circular radius $R_{\mathrm{half}} = \theta_{\mathrm{pixel}} / 2 \approx 3298.07''$. Because HEALPix cells are spherical quadrilaterals whose corner vertices extend to $R_{\mathrm{corner}} \approx 4700''$, entities located in the outer quadrilateral regions ($d_{\mathrm{center}} > R_{\mathrm{half}}$) produced negative values for $R_{\mathrm{half}} - d_{\mathrm{center}}$, which were clamped to $0.1'' < 5.0''$. This incorrectly quarantined an entire peripheral band of 55 objects rather than true boundary-proximate entities.

### Vector-Based Spherical Distance Formulation:
In v1.0.2, ASTRAEUS computes the mathematically exact, shortest great-circle angular distance from candidate entity coordinate $\mathbf{p}$ to the spherical polygon boundary segments $\partial C = \bigcup_i [\mathbf{a}_i, \mathbf{b}_i]$ of the containing HEALPix pixel at $N_{\mathrm{side}}=32$:
$$d_{\mathrm{edge}}(\mathbf{p}) = \min_{i} d(\mathbf{p}, [\mathbf{a}_i, \mathbf{b}_i])$$
where for each great-circle segment with normal $\mathbf{n}_i = \frac{\mathbf{a}_i \times \mathbf{b}_i}{\|\mathbf{a}_i \times \mathbf{b}_i\|}$:
$$d(\mathbf{p}, [\mathbf{a}_i, \mathbf{b}_i]) = \begin{cases} \arcsin(|\mathbf{p} \cdot \mathbf{n}_i|) & \text{if orthogonal projection } \mathbf{p}_{\mathrm{proj}} \text{ lies between } \mathbf{a}_i \text{ and } \mathbf{b}_i \\ \min(\operatorname{angdist}(\mathbf{p}, \mathbf{a}_i), \operatorname{angdist}(\mathbf{p}, \mathbf{b}_i)) & \text{otherwise} \end{cases}$$

### The Rigorous Quarantine Predicate:
$$\mathcal{P}_{\mathrm{quarantine}}(x) \iff \mathrm{conflicting\_neighbor}(p) \land \left[ d_{\mathrm{edge}}(x) \le 5.0'' \lor d_{\mathrm{edge}}(x) \le 3.0 \cdot \sigma_{\mathrm{ast}} \right]$$
Under this exact geometry, exactly **1** reference object in the entire 283-object corpus falls within 5.0″ of a boundary adjacent to a conflicting split (`ASTRAEUS-VARIABLE_STAR-syn-00295`, $d_{\mathrm{edge}} = 3.90'' \le 5.0''$, conflicting neighbor = `True`).

### Quarantine Diagnostics Table (Illustrative Edge Cases + Closest Corpus Objects):
| Object Identifier | Distance to Actual Edge ($d_{\mathrm{edge}}$) | Astrometric Uncertainty ($\sigma_{\mathrm{ast}}$) | Conflicting Neighbour | Partition Decision | Operating Policy Code |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Case A (Illustrative)** | 2.00″ | 0.500″ | Yes | **Quarantine** | `SPHERICAL_CELL_EDGE_BUFFER_VIOLATION` |
| **Case B (Illustrative)** | 500.00″ | 0.500″ | Yes | **Retain** | `SAFE_INTERIOR` |
| **Case C (Illustrative)** | 8.00″ | 4.000″ | Yes | **Quarantine** | `ASTROMETRIC_UNCERTAINTY_CROSSING_RISK` |
| **Case D (Illustrative)** | 2.00″ | 0.500″ | No | **Retain** | `INTERNAL_SPLIT_BOUNDARY` |
| **ASTRAEUS-VARIABLE_STAR-syn-00295** | 3.90″ | 0.044″ | Yes | **Quarantine** | `SPHERICAL_CELL_EDGE_BUFFER_VIOLATION` |
| **ASTRAEUS-ARTIFACT_DIPOLE-syn-00260** | 5.67″ | 0.036″ | Yes | **Retain (val)** | `SAFE_INTERIOR` |
| **ASTRAEUS-SN_Ia-syn-00072** | 6.56″ | 0.022″ | Yes | **Retain (test)** | `SAFE_INTERIOR` |
| **ASTRAEUS-SN_II-syn-00096** | 6.61″ | 0.031″ | Yes | **Retain (train)** | `SAFE_INTERIOR` |
| **ASTRAEUS-ARTIFACT_DIPOLE-syn-00256** | 17.12″ | 0.045″ | Yes | **Retain (train)** | `SAFE_INTERIOR` |
| **ASTRAEUS-TDE-num-00222** | 22.09″ | 0.041″ | Yes | **Retain (val)** | `SAFE_INTERIOR` |

### Key Acceptance Test Verification:
- **Verification Guarantee:** An entity situated $3800''$ from the cell center ($> R_{\mathrm{half}} = 3298''$) but $\approx 425''$ from the actual cell edge evaluates $d_{\mathrm{edge}} = 424.87'' > 5.0''$. It is **RETAINED** in active partitions, formally confirming that the flawed inscribed-circle clamp is 100% eliminated.

---
## 3. Astrometric Reconciliation & SN 2011fe Projected Offset (P1)
### Rigorous Arithmetic Derivation:
- **Astrometric Parameters for SN 2011fe in Messier 101:**
  - Nuclear angular separation: $\theta = 162.0'' = 7.8540 \times 10^{-4}\,\mathrm{rad}$.
  - Host redshift: $z = 0.000804$ ($cz \approx 241\,\mathrm{km\,s^{-1}}$). Direct Hubble conversion yields $d = v/H_0 \approx 3.4\,\mathrm{Mpc}$ (distorted by Virgo cluster infall).
  - Adopted direct distance: Cepheid / TRGB independent distance $d_L = 6.4 \pm 0.5\,\mathrm{Mpc}$ (Shappee & Stanek 2011).
  - Angular-diameter distance: $D_A = d_L / (1+z)^2 = 6.40 / 1.0016 = 6.3897\,\mathrm{Mpc} \approx 6.39\,\mathrm{Mpc}$.
  - **Projected Physical Separation:**
    $$d_{\mathrm{proj}} = \theta_{\mathrm{rad}} \cdot D_A = 162.0'' \times \frac{6390\,\mathrm{kpc}}{206264.8''/\mathrm{rad}} = 5.015\,\mathrm{kpc} \approx \mathbf{5.02 \pm 0.39\,\mathrm{kpc}}$$
- **Purging of the Obsolete $5.4\,\mathrm{kpc}$ Figure:** The historical figure of $5.4\,\mathrm{kpc}$ (from an older $6.9\,\mathrm{Mpc}$ distance model) is completely purged.
- **AT2018zr Nuclear Offset Resolution:** $\Delta \theta = 0.05''$ at $z=0.071$ yields $d_{\mathrm{proj}} = 0.07\,\mathrm{kpc}$. Documented that within the $1.0''$ survey PSF and $20\,\mathrm{mas}$ astrometric uncertainty, this is formally classified as consistent with nuclear centering.

---
## 4. Photometric Calibration & Native Linear Flux Error (P1)
### Separation of Nominal Survey Scales and Observation-Level Zero Points:
- `survey_nominal_zero_point`: Catalog nominal standard ($ZP_{\mathrm{ZTF}}=26.3$, $ZP_{\mathrm{PS1}}=25.0$, $ZP_{\mathrm{PTF}}=27.0$, $ZP_{\mathrm{ATLAS}}=27.5$).
- `observation_zero_point`: Exposure-specific calibration tracked per observation epoch (SN 2011fe PTF $ZP_{\mathrm{obs}}=27.02$, AT2018cow ATLAS $ZP_{\mathrm{obs}}=27.48$, AT2018zr ZTF $ZP_{\mathrm{obs}}=26.31$).
- Standardized internal pseudoflux scale remains $ZP=27.5\,\mathrm{mag\ AB}$.

### Negative Difference Flux & Native Uncertainty Preservation:
$$\sigma_F = \begin{cases} 0.921 \cdot F \cdot \sigma_m & \text{if } F > 0 \text{ (positive forced photometry)} \\ \sigma_{F, \mathrm{native}} & \text{if } F \le 0 \text{ (difference imaging linear subtraction noise)} \end{cases}$$
Verified in schema and numerical tests across all three cases: (a) positive photometry, (b) negative noisy difference flux, and (c) $5\sigma$ censored upper limits.

---
## 5. Dataset B Cross-Corpus Exclusion Contract (P1)
To prevent subtle data contamination between pretraining and held-out evaluation:
- **Hard Cross-Corpus Invariant:**
  $$A_{\mathrm{test}} \cap B_{\mathrm{pretrain}} = \varnothing$$
- **Exclusion Scope:** Extends beyond canonical IDs to include IAU aliases, coordinate proximity ($< 5.0''$), and all transformed, augmented, or injected derivatives (e.g. `AUGMENTED_SHIFT_*`, `INJECTED_FROM_*`, `DERIVATIVE_OF_*`).
- **Split Boundary Clarification:** Validation objects ($N=47$) may be utilized for downstream architecture selection and supervised tuning, but never for self-supervised foundation model pretraining.
- **Evaluation Hierarchy:**
  1. *Level 1 (Unseen Observations):* Held-out temporal epochs of known physical sources.
  2. *Level 2 (Unseen Objects):* Disjoint physical objects evaluated across disjoint spatial partitions.
  3. *Level 3 (Unseen Surveys):* Zero-shot cross-survey representation transfer.
  4. *Level 4 (Strict Joint Invariant):* Jointly unseen physical objects AND unseen surveys (Task 2).

---
## 6. Triple-Cohort Benchmark Separation & Statistical Disclosures (P1)
### Decoupling Evaluation Cohorts:
| Evaluation Cohort | Population Count | Contributing Populations | Scientific Evaluation Role |
| :--- | :---: | :--- | :--- |
| **1. Genuine Archival Observations** | **3** | `AT2018cow` (FBOT), `SN2011fe` (SN Ia), `AT2018zr` (TDE) | Real telescope data with empirical PSFs, atmospheric extinction, and instrumental noise |
| **2. Numerically Simulated Objects** | **105** | SLSNe (35), TDEs (35), FBOTs (35) | Radiative-hydrodynamic expanding atmosphere models |
| **3. Standardized Synthetic Objects** | **175** | SNe Ia (35), SNe II (35), SNe Ibc (35), Artifacts (35), Variables (35) | Controlled parameter-space and noise-injection benchmarking |

### Held-Out Test Set Support ($N_{\mathrm{test}}=45$) & Exact 95% Clopper-Pearson Intervals:
| Transient Class | Test Support ($N$) | 100% Recall 95% CI | 1-Error Recall 95% CI | Statistical Characterization |
| :--- | :---: | :---: | :---: | :--- |
| **ARTIFACT_DIPOLE** | 6 | $[54.1\%, 100.0\%]$ | $[35.9\%, 99.6\%]$ | Moderate test cohort support |
| **FBOT** | 3 | $[29.2\%, 100.0\%]$ | $[9.4\%, 99.2\%]$ | Small sample; 1 error = 25% recall change |
| **SLSN** | 7 | $[59.0\%, 100.0\%]$ | $[42.1\%, 99.6\%]$ | Moderate test cohort support |
| **SN_II** | 4 | $[39.8\%, 100.0\%]$ | $[19.4\%, 99.4\%]$ | Small sample; 1 error = 25% recall change |
| **SN_Ia** | 7 | $[59.0\%, 100.0\%]$ | $[42.1\%, 99.6\%]$ | Moderate test cohort support |
| **SN_Ibc** | 6 | $[54.1\%, 100.0\%]$ | $[35.9\%, 99.6\%]$ | Moderate test cohort support |
| **TDE** | 6 | $[54.1\%, 100.0\%]$ | $[35.9\%, 99.6\%]$ | Moderate test cohort support |
| **VARIABLE_STAR** | 6 | $[54.1\%, 100.0\%]$ | $[35.9\%, 99.6\%]$ | Moderate test cohort support |

- **Scientific Limitation Disclosure:** A high Macro-F1 across the held-out test cohort ($N=45$) serves as a controlled validation benchmark of representation learning under simulated and archival conditions, but does NOT constitute broad observational discovery proof across real-time sky surveys.

---
## 7. HMAC Authenticity vs Cryptographic Non-Repudiation
- **Symmetric HMAC-SHA256:** Provides tamper-evident integrity and authorized-origin verification within the trusted ASTRAEUS pipeline authority environment.
- **Non-Repudiation Boundary:** HMAC does not provide asymmetric digital signature non-repudiation (which requires asymmetric key pairs such as Ed25519). ASTRAEUS explicitly documents this boundary.

---
## 8. Final Population Accounting & Seven-Tier Audit Scorecard
### Population Reconciliation:
$$\mathbf{N_{\mathrm{total}} = N_{\mathrm{train}} + N_{\mathrm{val}} + N_{\mathrm{test}} + N_{\mathrm{quar}} = 190 + 47 + 45 + 1 = 283}$$

| Audit Gate | Evidentiary Standard | Observed Outcome | Gate Verdict |
| :--- | :--- | :---: | :---: |
| **Gate 1: Canonical Identity & Aliases** | Zero cross-split canonical ID or alias leakage | 0 ID / 0 Alias / 0 Dups | **PASSED** |
| **Gate 2: Spatial HEALPix Isolation** | 0% pixel intersection between train, val, test ($N_{\mathrm{side}}=32$) | 0 overlapping pixels | **PASSED** |
| **Gate 3: Host Galaxy Cluster Isolation** | Objects associated with same host cannot bridge splits | 0 cross-split host associations | **PASSED** |
| **Gate 4: Observation & Survey Isolation** | Repeated measurements and cross-survey IDs cannot leak across splits | 0 cross-split observation links | **PASSED** |
| **Gate 5: Quarantine Segregation** | 100% of boundary buffer objects quarantined into `split="quarantine"` | 0 buffer segregation violations | **PASSED** |
| **Gate 6: Temporal Stratification** | No training observations after $t_{\mathrm{split}} = 60000.0\,\mathrm{MJD}$ | 0 temporal violations | **PASSED** |
| **Gate 7: Cryptographic Authenticity** | SHA-256 integrity digest + HMAC-SHA256 MAC tag verification | 0 invalid signatures (283/283 authentic) | **PASSED** |

---
## 9. Stage 1 Final Certification Sign-Off & Stage 2 Authorization
- **P0 Quarantine Geometry:** RESOLVED (Exact spherical cell-boundary distance $d_{\mathrm{edge}}$ implemented).
- **P0 Single-Object Quarantine Recalculated:** RESOLVED ($N_{\mathrm{quar}}=1$, Train: 190, Val: 47, Test: 45).
- **P0 Authoritative Protocol Hash:** RESOLVED & VERIFIED (`19ce6a78dcc1da48d112884fc65cc9793defb7b6ff62ae71f896a9c718eb4218`).
- **P1 Quarantine Diagnostics Published:** RESOLVED (Illustrative + empirical edge diagnostics published).
- **P1 Triple-Cohort Decoupling:** RESOLVED (Archival, simulated, and synthetic cohorts separated).
- **P1 Derivative Exclusion Enforced:** RESOLVED (Exclusion contract covers aliases and transformed derivatives).
- **Corpus Certification:** **ASTRONOMICAL REFERENCE CORPUS v1.0.2 FORMALLY CERTIFIED**
- **Stage 2 Temporal Foundation Pretraining:** **FORMALLY AUTHORIZED TO PROCEED**
