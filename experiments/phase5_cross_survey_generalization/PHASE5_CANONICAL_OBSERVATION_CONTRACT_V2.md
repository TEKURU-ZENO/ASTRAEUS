# ASTRAEUS Phase 5: Canonical Observation Record Contract v2.0

**Author:** Dev Mehta  
**Lead Investigator & Mission Director, Project ASTRAEUS**  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-ARCH-002`  
**Milestone:** Phase 5 — System Architecture & Data Contracts  
**Governing Status:** Immutable Photometric Data Contract Specification  

---

## 1. Executive Summary & Purpose

The initial `CanonicalObservationRecord` established essential standardizations (such as microJansky units and negative difference flux preservation), but the Phase 5 scientific audit identified several critical gaps:
1. **Timestamp Conflation:** Rubin's `midpointMjdTai` (TAI scale) was assigned directly to `t_obs_bmjd` without scale conversion or barycentric correction.
2. **Fail-Open Adapter Defaults:** Missing or unrecognized filter names were silently defaulted (e.g., to `r_DECam` or `o_ATLAS`).
3. **Silent Uncertainty Flooring:** Non-positive uncertainties were floored using `max(err, 1e-3)` rather than being quarantined as corrupted or uncalibrated data.
4. **Conflation of Measurement Constructions:** Alert difference fluxes and forced difference fluxes were not rigorously distinguished from coadded total fluxes or aperture fluxes.

Contract v2.0 establishes rigorous, fail-closed data semantics across all surveys (ZTF, ATLAS, DECam, Pan-STARRS1, Rubin).

---

## 2. Complete Field Specification for Contract v2.0

```python
class CanonicalObservationRecordV2(BaseModel):
    # 1. Identity & Provenance
    entity_id: str                      # Permanent ASTRAEUS event ID (ASTRAEUS-EVT-XXXXX)
    survey_obs_id: str                  # Native exposure/observation record ID
    survey_id: str                      # 'ZTF', 'ATLAS', 'DECam', 'PS1', or 'Rubin'
    provenance_hash: str                # SHA-256 digest of original raw provider response

    # 2. Spatio-Temporal Coordinates & Physical Time Scales
    t_native: float                     # Native exposure midpoint as reported by archive
    native_time_scale: str              # 'UTC', 'TAI', 'TDB', or 'TT'
    exposure_midpoint_convention: str   # 'MIDPOINT', 'START_PLUS_HALF_EXPTIME', 'SHUTTER_OPEN'
    t_converted_bmjd: Optional[float]   # Barycentric Modified Julian Date (TDB scale), strictly None unless converted
    converted_time_scale: Optional[str] # 'BMJD_TDB' if barycentric conversion succeeded, else None
    time_conversion_method: str         # e.g. 'ASTROPY_BARYCORR_V1' or 'NATIVE_UNCONVERTED'
    t_obs_bmjd: Optional[float]         # Deprecated alias. Strictly None unless barycentric correction was performed
    t_err_days: float                   # Timing uncertainty or half-exposure duration

    @property
    def observation_time(self) -> float:
        """Returns t_converted_bmjd if computed; falls back to t_native without false BMJD claims."""
        return self.t_converted_bmjd if self.t_converted_bmjd is not None else self.t_native

    ra_deg: float                       # ICRS Right Ascension [0, 360)
    dec_deg: float                      # ICRS Declination [-90, +90]
    astrometric_sigma_arcsec: float     # 1-sigma positional uncertainty (strictly > 0)

    # 3. Photometric Measurement & Semantics
    native_passband: str                # Strictly validated against SURVEY_FILTER_METADATA
    obs_type: ObservationTypeV2         # Explicit observation construction type
    measured_flux_ujy: float            # Calibrated flux density in microJanskys (AB Z0 = 23.900)
    measurement_sigma_ujy: float        # 1-sigma uncertainty (strictly > 0; NaN/inf forbidden)
    upper_limit_ujy: Optional[float]    # 5-sigma limiting flux for non-detections
    is_positive_detection_snr5: bool    # True if flux > 0 and flux / sigma >= 5.0

    # 4. Calibration & Covariance Context
    zero_point_ab: Optional[float]      # Reference zero-point magnitude (default 23.900)
    zero_point_err: Optional[float]     # Zero-point uncertainty
    photometric_group_id: Optional[str] # Group ID for shared calibration covariance

    # 5. Quality, Masking & Governance
    native_flags: int                   # Survey-specific reduction bitmask
    is_clean: bool                      # True if all quality cuts pass
    quarantine_reason: Optional[str]    # None if clean; else 'UNKNOWN_FILTER', 'INVALID_SIGMA', etc.
```

---

## 3. Observation Construction Semantics (`ObservationTypeV2`)

Heterogeneous astronomical measurements must never be pooled without explicitly modeling their physical measurement process:

| Type Enum | Physical Definition | Allowed Flux Range | Treatment of Negative Flux |
| :--- | :--- | :---: | :--- |
| **`DIFFERENCE_FORCED`** | PSF flux measured at fixed celestial coordinates on a template-subtracted difference image. | $(-\infty, +\infty)$ | **Preserved natively.** Represents unbiased noise or negative fluctuations around baseline. |
| **`DIFFERENCE_ALERT`** | PSF flux of an alert-broker transient discovery on a difference image ($F > 5\sigma$). | $(0, +\infty)$ | Strictly positive by alert stream definition. |
| **`TOTAL_STACK`** | Coadded static-sky stack flux including host-galaxy background. | $[0, +\infty)$ | Un-subtracted total flux. Must not be directly merged with difference light curves. |
| **`CATALOG_APER`** | Fixed-aperture measurement on a single exposure. | $[0, +\infty)$ | Requires aperture correction and host subtraction before transient modeling. |
| **`UPPER_LIMIT_THRESHOLD`** | $5\sigma$ exposure sensitivity limit for non-detections. | $(0, +\infty)$ | Stored in `upper_limit_ujy` with `measured_flux_ujy = 0.0`. |
| **`MISSING_MEASUREMENT`** | Scheduled epoch that was clouded out or rejected during reduction. | `None` / `NaN` | Quarantined; used only for cadence and duty-cycle modeling. |

---

## 4. Rigorous Fail-Closed Adapter Rules

To prevent data corruption from propagating downstream into scientific models, all survey adapters must enforce the following fail-closed assertions:

### Rule 1: Unknown Filter Rejection
- If a raw exposure record reports an unrecognized filter string (e.g. empty, `"none"`, or a filter not present in `SURVEY_FILTER_METADATA`), the adapter **must NOT default to any filter**.
- The record must be rejected or assigned `is_clean = False` with `quarantine_reason = "UNKNOWN_FILTER"`.

### Rule 2: Invalid Uncertainty Rejection
- Measurement uncertainty `measurement_sigma_ujy` must be strictly positive, real, and finite:
  $$\sigma_F > 0.0, \quad \sigma_F \neq \text{NaN}, \quad \sigma_F \neq \infty$$
- Under no circumstances may an adapter silently clamp or floor invalid uncertainties using `max(err, 1e-3)`. Records violating this rule must be quarantined with `quarantine_reason = "INVALID_UNCERTAINTY"`.

### Rule 3: Native Time-Scale Preservation
- The adapter must store the raw timestamp in `t_native` and record its exact archive scale in `native_time_scale`:
  - **Rubin alerts:** reported in `TAI` (International Atomic Time) -> `native_time_scale = "TAI"`.
  - **ATLAS forced photometry:** reported in `MJD` (UTC) -> `native_time_scale = "UTC"`.
  - **ZTF alerts:** reported in `JD` / `MJD` (UTC) -> `native_time_scale = "UTC"`.
- Barycentric conversion to `BMJD(TDB)` requires:
  $$t_{\mathrm{BMJD}} = t_{\mathrm{MJD}} + \Delta t_{\mathrm{clock}} + \frac{\mathbf{r}_{\mathrm{geo}} \cdot \hat{\mathbf{n}}}{c} + \Delta t_{\mathrm{rel}}$$
  If the observatory position or precise coordinates are unavailable to compute this term accurately, `t_converted_bmjd` must be set to `None` and `time_conversion_method = "NATIVE_UNCONVERTED"`.
