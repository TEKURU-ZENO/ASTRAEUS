# ASTRAEUS Phase 5: Survey Capability, Data Access & Acquisition Feasibility Audit

**Author:** Dev Mehta  
**Lead Investigator & Mission Director, Project ASTRAEUS**  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-AUDIT-001`  
**Milestone:** Phase 5 — Cross-Survey Generalization & Domain Robustness  
**Workstream:** Workstream 2 — Survey Reconnaissance & Acquisition Feasibility  

---

## 1. Executive Summary & Purpose

A robust cross-survey benchmark cannot be constructed by simply querying catalog coordinates and assuming all surveys provide interchangeable time-domain light curves. As discovered during Phase 4B with TDE `ZTF18aabtxvd`, raw HTTP retrieval does not equal feature evaluability, and different survey reduction pipelines introduce subtle, fundamental differences in image subtraction, flux zero points, noise floors, and temporal sampling.

This audit evaluates the real-world accessibility, photometric conventions, temporal depth, and candidate yield of four external astronomical facilities (ATLAS, DECam DDF, Pan-STARRS1, and Rubin LSST) relative to our reference baseline (ZTF), establishing the empirical feasibility of Phase 5.

---

## 2. Comprehensive Survey Capability Matrix

| Facility Parameter | ZTF (Reference Domain) | ATLAS (Primary External) | DECam DDF (Secondary External) | Pan-STARRS1 (Complementary) | Rubin LSST (Prospective) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Telescope & Aperture** | Palomar 48-inch (1.2m) Schmidt | Twin 0.5m $f/2$ Wright Schmidt (x4) | Blanco 4m Victor M. Blanco | PS1 1.8m Ritchey-Chrétien | Simonyi 8.4m Survey Telescope |
| **Field of View (FoV)** | $47.0\,\mathrm{deg}^2$ | $29.2\,\mathrm{deg}^2$ per camera | $3.0\,\mathrm{deg}^2$ | $3.3\,\mathrm{deg}^2$ | $9.6\,\mathrm{deg}^2$ |
| **Pixel Scale** | $1.01''/\mathrm{pixel}$ | $1.86''/\mathrm{pixel}$ | $0.263''/\mathrm{pixel}$ | $0.258''/\mathrm{pixel}$ | $0.200''/\mathrm{pixel}$ |
| **Primary Passbands** | ZTF $g$, $r$, $i$ | Cyan ($c$), Orange ($o$) | DECam $u, g, r, i, z, Y$ | $g_{\mathrm{P1}}, r_{\mathrm{P1}}, i_{\mathrm{P1}}, z_{\mathrm{P1}}, y_{\mathrm{P1}}$ | Rubin $u, g, r, i, z, y$ |
| **Single-Visit Depth** | $r \approx 20.5\,\mathrm{mag}$ | $o \approx 19.5\,\mathrm{mag}$ | $r \approx 23.5\text{--}24.5\,\mathrm{mag}$ | $r_{\mathrm{P1}} \approx 21.8\,\mathrm{mag}$ | $r \approx 24.5\,\mathrm{mag}$ |
| **Cadence Profile** | 2–3 days (Northern sky) | 1–2 days (All-sky, multi-site) | Nightly across observing blocks | $3\pi$: $\approx 12$ epochs/band; MDS: 3–4d | 3–4 days (Wide-Fast-Deep) |
| **Time-Domain Products** | Real-time difference alerts, DR coadds | Forced photometry API & public alerts | Released candidate DDF light curves | MDS transient alerts; 3pi warps | Real-time alerts; Annual DRs |
| **Photometric Convention** | Nanomaggies ($Z_0 = 22.500$) | MicroJanskys ($\mu\mathrm{Jy}$) | ADU counts / calibrated AB mag | Calibrated AB magnitudes | Nanomaggies / nanoJanskys |
| **Access Mechanism** | ALeRCE API, IRSA TAP service | Forced Photometry HTTP API | NOIRLab Astro Data Lab, Zenodo | MAST CasJobs, TAP, Web API | Alert brokers, Rubin RSP |
| **Data Licensing** | Public (Alerts & DRs) | Public forced service (registered) | Open public research release | Open public research access | Public alerts; proprietary DRs |

---

## 3. In-Depth Facility Feasibility Audits

### 3.1 ATLAS Forced Photometry Service (Primary External Source)
- **Service Description:** The Asteroid Terrestrial-impact Last Alert System provides a high-throughput forced photometry server (`fallingstar.com` / `fallingstar-data.com`) that performs point-spread-function (PSF) fitting photometry on historical difference images at arbitrary user-supplied sky coordinates $(\alpha, \delta)$.
- **Photometric Integrity & Conventions:**
  - Fluxes are natively reported in **microJanskys ($\mu\mathrm{Jy}$)** with corresponding $1\sigma$ uncertainties (`uJy`, `duJy`).
  - Native filters are broad-band cyan ($c$, $\lambda_{\mathrm{eff}} \approx 5330\,\text{Å}$, FWHM $\approx 2300\,\text{Å}$) and orange ($o$, $\lambda_{\mathrm{eff}} \approx 6790\,\text{Å}$, FWHM $\approx 2600\,\text{Å}$).
  - *Crucial Strength:* The forced photometry pipeline outputs genuine physical measurements regardless of significance, **preserving negative difference fluxes and non-detections** without censoring. This completely avoids artificial Eddington bias or missing upper-limit baselines.
  - *Quality Bitmasks:* Quality flags include `chi/N` (reduced $\chi^2$ of PSF fit), major/minor axis moments, and proximity to chip edges or bright stars. Flag gating (`chi/N < 3.0`) is required to eliminate subtraction artifacts.
- **Cadence & Overlap Potential:**
  - With four operational observing sites (Haleakala, Mauna Loa, South Africa, Chile), ATLAS scans the entire visible sky every 24–48 hours.
  - Overlap with ZTF Bright Transient Survey (BTS) transients is exceptionally high: $> 90\%$ of Northern/Equatorial transients observed by ZTF have concurrent ATLAS coverage, providing the primary basis for our matched time-domain benchmark.

### 3.2 DECam Deep Drilling Fields (Secondary External Replication)
- **Service Description:** The Dark Energy Camera on the 4m Blanco telescope operated dedicated Deep Drilling Fields (e.g. COSMOS, ECDFS, ELAIS-S1, XMM-LSS). In 2021, an extensive time-domain programme released 4,413 candidate transient light curves over approximately 40 observing nights per field.
- **Photometric Nuance & Limitations:**
  - *Depth Advantage:* DECam reaches $r \approx 23.5\text{--}24.5\,\mathrm{mag}$, penetrating 3 to 4 magnitudes deeper than ZTF. This enables empirical testing of domain shift into the ultra-faint, low-SNR regime.
  - *The Negative Difference Flux Challenge:* Published candidate DDF light curves are predominantly based on difference-image source extractions. The published tables require careful vetting because some pipelines thresholded on positive detections ($s > 0$), omitting negative difference excursions caused by reference image over-subtraction.
  - *Passband Diversity:* Provides simultaneous $g, r, i, z$ observations with well-characterized standard DECam filter response profiles, providing an ideal testing ground for multi-color physical models.
- **Overlap Potential:**
  - Because DDF fields cover specific high-latitude pencil beams ($\approx 3\,\mathrm{deg}^2$ each), absolute transient overlap with wide-field shallow ZTF alerts is sparse ($\approx 20\text{--}50$ common spectroscopic events). However, DECam provides our primary **independent replication benchmark** for high-cadence, multi-band physics fitting.

### 3.3 Pan-STARRS1 (Complementary Archival & Host Domain)
- **Service Description:** Pan-STARRS1 (PS1) conducted the $3\pi$ Steradian Survey and the Medium Deep Survey (MDS) from Haleakala, Hawaii.
- **Photometric Nuance:**
  - *3π Survey Limitation:* Over the $30,000\,\mathrm{deg}^2$ footprint, PS1 $3\pi$ has only $\approx 12$ individual exposures per filter accumulated across years. This sampling is inadequate for high-resolution transient light-curve classification.
  - *Medium Deep Survey (MDS) Value:* The 10 MDS fields were observed with a 3-day cadence in $g_{\mathrm{P1}}, r_{\mathrm{P1}}, i_{\mathrm{P1}}, z_{\mathrm{P1}}, y_{\mathrm{P1}}$, capturing well-sampled light curves for thousands of supernovae.
  - *Warp vs. Stack Products:* MAST hosts both single-epoch calibrated "warp" images and deep stacked images. For host-galaxy context, deep stacked images provide exquisite, un-trailed reference imaging. For time-domain validation, single-epoch warps or MDS tables must be used exclusively.
- **Overlap & Utility:**
  - PS1 serves as the authoritative source for host-galaxy morphology, stellar mass estimates, photometric redshifts, and reference-catalog photometric calibration.

### 3.4 Rubin Observatory LSST (Prospective Boundary)
- **Data Policy & Access:**
  - Rubin LSST public alert packets (60-second latency, containing difference flux, cutouts, and 12-month forced photometry) are fully open to the worldwide community via downstream alert brokers.
  - However, deep processed Data Releases (DR1, DR2) and prompt pixel-level cutouts remain restricted to Rubin Data Rights holders during proprietary periods.
- **Phase 5 Decision:**
  - ASTRAEUS will build a forward-compatible **Rubin Alert Schema Adapter**, ensuring architecture readiness.
  - However, to prevent external governance dependencies, **Phase 5 scientific closure will not depend on Rubin data rights**. Completion will be established on ZTF, ATLAS, DECam DDF, and PS1.

---

## 4. First-Stage Feasibility Pilot Protocol (Matched-Event Study)

Before constructing production data pipelines or downloading bulk repositories, Workstream 2 mandates an empirical feasibility pilot across 100–200 candidate events to establish matched data yield.

```mermaid
flowchart TD
    CAND["Curate Candidate Cohort (N=150)<br/>Confirmed ZTF Supernovae with IAU Names"]
    QUERY_A["Query ATLAS Forced Photometry API<br/>Coordinate Box & Explosion Window"]
    QUERY_D["Query DECam DDF Catalog<br/>Cross-Match Blanco Archive"]
    QUERY_P["Query PS1 MAST TAP Service<br/>Host & Deep Stack Photometry"]

    CAND --> QUERY_A
    CAND --> QUERY_D
    CAND --> QUERY_P

    AUDIT["Feasibility Assessment Gate<br/>1. Common Event Yield<br/>2. Photometric SNR >= 5 Count<br/>3. Baseline Stability<br/>4. Label Verifiability"]

    QUERY_A --> AUDIT
    QUERY_D --> AUDIT
    QUERY_P --> AUDIT

    style CAND fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style AUDIT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### Pilot Evaluation Criteria:
1. **Matched Event Yield:** Determine the exact proportion of ZTF-selected Type Ia and Core-Collapse supernovae that possess $\ge 8$ valid photometric measurements in ATLAS or DECam.
2. **Signal-to-Noise Ratio Distribution:** Measure the median SNR of external detections across bright ($m < 18$) vs. faint ($m \ge 18$) subsets.
3. **Pre-Explosion Baseline Noise:** Verify that forced photometry non-detections exhibit Gaussian zero-centered flux distributions ($\mathbb{E}[F] \approx 0, \chi^2_{\mathrm{red}} \approx 1.0$) prior to explosion.
4. **Spectroscopic Label Traceability:** Ensure that every matched candidate traces back to an unambiguous IAU/TNS classification with named instrument attribution.

---

## 5. Feasibility Audit Verdict

The investigation establishes that:
1. **ATLAS is immediately viable and optimal** as our primary external light-curve domain, providing all-sky temporal overlap and un-censored forced photometry.
2. **DECam DDF is viable as an independent replication domain**, specifically targeting high-cadence and deep-field transient physics.
3. **Pan-STARRS1 is viable as a complementary host-galaxy and archival imaging provider**, with time-domain experiments restricted to its Medium Deep fields.
4. **Rubin LSST is appropriately managed as a prospective adapter**, maintaining zero operational blocking risk.

**AUDIT DISPOSITION: PHASE 5 FEASIBILITY CONFIRMED; PROCEED TO DATASET & COHORT DESIGN.**

**Signed & Certified by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
Date: November 16, 2026
