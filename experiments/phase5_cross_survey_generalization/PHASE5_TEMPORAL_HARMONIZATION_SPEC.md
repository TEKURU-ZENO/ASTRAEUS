# ASTRAEUS Phase 5: Temporal Harmonization & Barycentric Correction Specification

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026 (Remediated Checkpoint)  
**Document Identifier:** `ASTRAEUS-P5-TIME-SPEC-001`  
**Milestone:** Phase 5 — Workstream 4 (Deliverable D4.1)  
**Governing Status:** Mathematical & Algorithmic Time-Standard Specification  

---

## 1. Executive Summary & Problem Formulation

In transient time-series modeling, comparing light curves across independent survey facilities requires high-precision temporal alignment. Different astronomical archives publish timestamps in divergent time scales and coordinate frames:
- **ZTF:** Modified Julian Date in UTC (`MJD_UTC`).
- **ATLAS:** Modified Julian Date in UTC (`MJD_UTC`).
- **Rubin / LSST Alerts:** International Atomic Time (`midpointMjdTai`).
- **Solar System / Orbital Ephemerides:** Barycentric Dynamical Time (`TDB`) or Terrestrial Time (`TT`).

Because the Earth revolves around the Sun at an average orbital speed of $\approx 30\,\mathrm{km/s}$, an observatory on Earth is displaced by up to $1\,\mathrm{AU} \approx 1.496 \times 10^8\,\mathrm{km}$ relative to the Solar System Barycenter (SSB). For an optical transient at celestial coordinate $\hat{\mathbf{n}}$, the geometric light-travel time difference reaches:
$$\Delta t_{\mathrm{bary}} = \frac{\mathbf{r}_{\mathrm{topo}} \cdot \hat{\mathbf{n}}}{c} \approx \pm 499\,\mathrm{seconds} \approx \pm 0.00578\,\mathrm{days}$$

If uncorrected, simultaneous observations from facilities separated in season or location (e.g. Palomar vs. Cerro Tololo or Haleakala) will incur an apparent temporal shift of up to $\approx 8.3\,\mathrm{minutes}$, distorting rise-time fits, shock-breakout searches, and cross-survey feature correlations.

This specification establishes:
1. Physical time-scale conversion pipelines ($\mathrm{UTC} \to \mathrm{TAI} \to \mathrm{TT} \to \mathrm{TDB}$).
2. Standardized observatory topocentric coordinates for 7 key Phase 5 survey facilities.
3. Library-verified barycentric corrections via Astropy's pinned builtin Solar System ephemeris (`ephemeris="builtin"`, IAU/SOFA numerical algorithms) yielding Barycentric Modified Julian Date on the TDB scale ($\mathrm{BMJD}_{\mathrm{TDB}}$).
4. Strict fail-closed fallback semantics guaranteeing that unregistered observatories or invalid scales are rejected with explicit errors, previous conversions are invalidated when inputs change, and unconverted native timestamps are never mistaken for barycentric coordinates.

---

## 2. Physical Time Scales & Conversion Chain

```mermaid
flowchart LR
    UTC["UTC<br/>(Coordinated Universal Time)<br/>Discontinuous Leap Seconds"] -->|+ Delta_AT (37s)| TAI["TAI<br/>(International Atomic Time)<br/>Uniform SI Seconds"]
    TAI -->|+ 32.184s| TT["TT<br/>(Terrestrial Time)<br/>Geocentric Coordinate Time"]
    TT -->|+ Relativistic Variations (approx 1.6ms)| TDB["TDB<br/>(Barycentric Dynamical Time)<br/>Solar System Barycenter"]
    TDB -->|+ Astropy Barycentric Delay (dt_bary)| BMJD["BMJD_TDB<br/>(Barycentric Modified Julian Date)"]

    style UTC fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style TAI fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style TT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style TDB fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style BMJD fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

### 2.1 The Exact Mathematical Relations:
1. **UTC to TAI (Leap Seconds):**
   $$\mathrm{TAI} = \mathrm{UTC} + \Delta\mathrm{AT}$$
   where $\Delta\mathrm{AT} = 37.0\,\mathrm{seconds}$ (for all observation epochs from January 1, 2017 to late 2026).
2. **TAI to TT (Fixed Offset):**
   $$\mathrm{TT} = \mathrm{TAI} + 32.184\,\mathrm{seconds} = \mathrm{UTC} + 69.184\,\mathrm{seconds}$$
3. **TT to TDB (Relativistic Periodic Variations):**
   $$\mathrm{TDB} \approx \mathrm{TT} + 0.001657\,\mathrm{s} \sin(g) + 0.000022\,\mathrm{s} \sin(2g)$$
   where $g \approx 357.53^\circ + 0.98560028^\circ \times (t_{\mathrm{TT}} - 2451545.0)$ is the Earth's mean anomaly. The amplitude of $(TDB - TT)$ is bounded by $< 1.7\,\mathrm{milliseconds}$.

---

## 3. Barycentric Correction Formulation & Library Attribution

For an observer at topocentric position vector $\mathbf{r}_{\mathrm{topo}}(t)$ relative to the Solar System Barycenter observing a celestial source in direction $\hat{\mathbf{n}} = (\cos\delta \cos\alpha, \cos\delta \sin\alpha, \sin\delta)^T$, the barycentric light-travel time correction $\Delta t_{\mathrm{bary}}$ is calculated via Astropy's `Time.light_travel_time(coord, 'barycentric', ephemeris='builtin')`.

### 3.1 Algorithmic Scope & Physical Model:
- **Pinned Ephemeris:** The calculation is pinned to Astropy's built-in Solar System ephemeris (`ephemeris="builtin"`), utilizing IAU/SOFA numerical routines.
- **Library Correction:** Astropy's `light_travel_time` implements the classical geometric Roemer delay:
  $$\Delta t_R = \frac{\mathbf{r}_{\mathrm{topo}}(t) \cdot \hat{\mathbf{n}}}{c}$$
  combined with first-order relativistic corrections. It accounts for Earth's orbital position, topocentric observatory location on the WGS-84 reference ellipsoid, and diurnal rotation.
- **Physical Bounds:** The absolute delay is strictly bounded by Earth's orbital radius: $|\Delta t_{\mathrm{bary}}| \le \frac{1\,\mathrm{AU}}{c} \approx 499\,\mathrm{s} \approx 0.00578\,\mathrm{days}$. Incorporating lunar and planetary perturbations, any computed value exceeding $|\Delta t_{\mathrm{bary}}| > 0.00585\,\mathrm{days}$ is flagged and rejected as unphysical.
- **Sub-Second Precision Target:** The standard conversion is validated against independently calculated reference benchmarks to ensure timing discrepancy $|\Delta t| < 0.01\,\mathrm{s} \approx 1.15 \times 10^{-7}\,\mathrm{days}$.

---

## 4. Standardized Observatory Geodetic Coordinates (WGS-84)

To calculate $\mathbf{r}_{\mathrm{topo}}(t)$ accurately, topocentric observatory positions are standardized into the `ObservatoryRegistry`:

| Facility Key | Observatory Name | Geodetic Latitude | Geodetic Longitude | Elevation ($h$) |
| :--- | :--- | :---: | :---: | :---: |
| **`PALOMAR`** | Palomar Observatory (Samuel Oschin) | $+33.3563^\circ\,\mathrm{N}$ | $-116.8650^\circ\,\mathrm{W}$ | $1706\,\mathrm{m}$ |
| **`HALEAKALA`** | Haleakala Observatory, Maui (Twin 0.5m / PS1) | $+20.7075^\circ\,\mathrm{N}$ | $-156.2570^\circ\,\mathrm{W}$ | $3055\,\mathrm{m}$ |
| **`MAUNA_LOA`** | Mauna Loa Observatory, Hawaii | $+19.5362^\circ\,\mathrm{N}$ | $-155.5763^\circ\,\mathrm{W}$ | $3394\,\mathrm{m}$ |
| **`SUTHERLAND`** | Sutherland Observatory, South Africa | $-32.3760^\circ\,\mathrm{S}$ | $+20.8108^\circ\,\mathrm{E}$ | $1798\,\mathrm{m}$ |
| **`EL_SAUCE`** | El Sauce Observatory, Chile | $-30.4726^\circ\,\mathrm{S}$ | $-70.7634^\circ\,\mathrm{W}$ | $1560\,\mathrm{m}$ |
| **`CERRO_TOLOLO`** | Cerro Tololo Inter-American Observatory (DECam) | $-30.1690^\circ\,\mathrm{S}$ | $-70.8065^\circ\,\mathrm{W}$ | $2207\,\mathrm{m}$ |
| **`CERRO_PACHON`** | Vera C. Rubin Observatory (Cerro Pachón) | $-30.2444^\circ\,\mathrm{S}$ | $-70.7494^\circ\,\mathrm{W}$ | $2663\,\mathrm{m}$ |

---

## 5. Software Architecture & Conversion Invariants

The `TemporalHarmonizer` class enforces five strict operational rules:
1. **Coupled Time Coordinates:** Calling `record.get_time_coordinate()` returns the explicit `(timestamp, time_scale)` tuple. Scientific consumers are barred from ingesting naked numeric timestamps.
2. **Fail-Closed Observatory & Scale Validation:** Passing an unregistered observatory or an unsupported time scale raises `ValueError` immediately. Silent fallbacks to Palomar or UTC are strictly eliminated.
3. **Stale-Result Invalidation:** If a previously converted record undergoes coordinate changes ($\alpha, \delta$), native time modification ($t_{\mathrm{native}}$), scale updates, or observatory re-assignment, the existing conversion is invalidated and recomputed.
4. **Non-Swallowed Error Reporting:** If conversion fails during batch processing, the record is tagged with `conversion_status = "FAILED"`, `t_converted_bmjd = None`, and `quarantine_reason` populated with the exact error. It never appears as a valid or untouched record.
5. **Exposure Midpoint Normalization:** Timestamps published as shutter-open times are shifted to midpoints ($t_{\mathrm{midpoint}} = t_{\mathrm{start}} + \frac{1}{2} t_{\mathrm{exp}}$). Recognized conventions (`MIDPOINT`, `START_PLUS_HALF_EXPTIME`, `SHUTTER_OPEN`) are strictly enforced.

---

**Approved by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026 (Remediated Checkpoint)
