# ASTRAEUS Phase 5: Temporal Harmonization & Barycentric Correction Specification

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026  
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

Because the Earth revolves around the Sun at an average orbital speed of $\approx 30\,\mathrm{km/s}$, an observatory on Earth is displaced by up to $1\,\mathrm{AU} \approx 1.496 \times 10^8\,\mathrm{km}$ relative to the Solar System Barycenter (SSB). For an optical transient at celestial coordinate $\hat{\mathbf{n}}$, the geometric light-travel time difference (the **Roemer delay**) reaches:
$$\Delta t_R = \frac{\mathbf{r}_{\mathrm{obs}} \cdot \hat{\mathbf{n}}}{c} \approx \pm 499\,\mathrm{seconds} \approx \pm 0.00578\,\mathrm{days}$$

If uncorrected, simultaneous observations from facilities separated in season or location (e.g. Palomar vs. Cerro Tololo or Haleakala) will incur an apparent temporal shift of up to $\approx 8.3\,\mathrm{minutes}$, distorting rise-time fits, shock-breakout searches, and cross-survey feature correlations.

This specification establishes:
1. Exact physical time-scale conversion pipelines ($\mathrm{UTC} \to \mathrm{TAI} \to \mathrm{TT} \to \mathrm{TDB}$).
2. Standardized observatory topocentric coordinates for all Phase 5 survey facilities.
3. Exact relativistic barycentric corrections yielding Barycentric Modified Julian Date on the TDB scale ($\mathrm{BMJD}_{\mathrm{TDB}}$).
4. Strict fail-closed fallback semantics guaranteeing that unconverted native timestamps are never mistaken for barycentric coordinates.

---

## 2. Physical Time Scales & Conversion Chain

```mermaid
flowchart LR
    UTC["UTC<br/>(Coordinated Universal Time)<br/>Discontinuous Leap Seconds"] -->|+ Delta_AT (37s)| TAI["TAI<br/>(International Atomic Time)<br/>Uniform SI Seconds"]
    TAI -->|+ 32.184s| TT["TT<br/>(Terrestrial Time)<br/>Geocentric Coordinate Time"]
    TT -->|+ Relativistic Variations (approx 1.6ms)| TDB["TDB<br/>(Barycentric Dynamical Time)<br/>Solar System Barycenter"]
    TDB -->|+ Roemer, Einstein & Shapiro Delays| BMJD["BMJD_TDB<br/>(Barycentric Modified Julian Date)"]

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

## 3. Barycentric Correction Formulation

For an observer at topocentric position vector $\mathbf{r}_{\mathrm{obs}}(t)$ relative to the Solar System Barycenter observing a celestial source in direction $\hat{\mathbf{n}} = (\cos\delta \cos\alpha, \cos\delta \sin\alpha, \sin\delta)^T$, the total barycentric time correction $\Delta t_{\mathrm{bary}}$ is:

$$\Delta t_{\mathrm{bary}} = \Delta t_R + \Delta t_E + \Delta t_S$$

### 3.1 Components of the Correction:
1. **Roemer Delay (Classical Geometric Delay):**
   $$\Delta t_R = \frac{\mathbf{r}_{\mathrm{obs}}(t) \cdot \hat{\mathbf{n}}}{c}$$
   Magnitude: up to $\approx \pm 499\,\mathrm{s}$ ($0.00578\,\mathrm{d}$).
2. **Einstein Delay (Gravitational Redshift & Time Dilation):**
   $$\Delta t_E = \frac{1}{c^2} \int \left( U(\mathbf{r}) + \frac{1}{2} v^2 \right) dt$$
   Magnitude: $\approx 1.6\,\mathrm{ms}$.
3. **Shapiro Delay (Gravitational Deflection in Solar Potential):**
   $$\Delta t_S = -\frac{2 G M_\odot}{c^3} \ln\left(1 + \cos\theta_\odot\right)$$
   where $\theta_\odot$ is the elongation angle between the target and the Sun. Outside the immediate solar limb ($\theta_\odot > 20^\circ$), $\Delta t_S < 0.1\,\mathrm{ms}$.

---

## 4. Standardized Observatory Geocentric Coordinates

To calculate $\mathbf{r}_{\mathrm{obs}}(t)$ accurately, topocentric observatory positions are standardized into the `ObservatoryRegistry`:

| Facility | Observatory Name | Geodetic Latitude | Geodetic Longitude | Elevation ($h$) |
| :--- | :--- | :---: | :---: | :---: |
| **ZTF** | Palomar Observatory (Samuel Oschin) | $+33.3563^\circ\,\mathrm{N}$ | $-116.8650^\circ\,\mathrm{W}$ | $1706\,\mathrm{m}$ |
| **ATLAS HK** | Haleakala Observatory, Maui (Twin 0.5m) | $+20.7075^\circ\,\mathrm{N}$ | $-156.2570^\circ\,\mathrm{W}$ | $3055\,\mathrm{m}$ |
| **ATLAS ML** | Mauna Loa Observatory, Hawaii | $+19.5362^\circ\,\mathrm{N}$ | $-155.5763^\circ\,\mathrm{W}$ | $3394\,\mathrm{m}$ |
| **ATLAS SA** | Sutherland Observatory, South Africa | $-32.3760^\circ\,\mathrm{S}$ | $+20.8108^\circ\,\mathrm{E}$ | $1798\,\mathrm{m}$ |
| **ATLAS CH** | El Sauce Observatory, Chile | $-30.4726^\circ\,\mathrm{S}$ | $-70.7634^\circ\,\mathrm{W}$ | $1560\,\mathrm{m}$ |
| **DECam** | Cerro Tololo Inter-American Observatory | $-30.1690^\circ\,\mathrm{S}$ | $-70.8065^\circ\,\mathrm{W}$ | $2207\,\mathrm{m}$ |
| **PS1** | Haleakala (Pan-STARRS1 1.8m) | $+20.7075^\circ\,\mathrm{N}$ | $-156.2570^\circ\,\mathrm{W}$ | $3055\,\mathrm{m}$ |
| **Rubin** | Vera C. Rubin Observatory (Cerro Pachón) | $-30.2444^\circ\,\mathrm{S}$ | $-70.7494^\circ\,\mathrm{W}$ | $2663\,\mathrm{m}$ |

---

## 5. Software Architecture & Conversion Invariants

The `TemporalHarmonizer` class enforces four strict operational rules:
1. **Coupled Time Coordinates:** Calling `record.get_time_coordinate()` returns the explicit `(timestamp, time_scale)` tuple.
2. **Ephemeris Validation:** Barycentric conversion utilizes the JPL DE430/DE440 numerical planetary ephemeris via `astropy.time.Time` and `astropy.coordinates.EarthLocation`.
3. **Midpoint Normalization:** Timestamps published as shutter-open times are normalized to exposure midpoints:
   $$t_{\mathrm{midpoint}} = t_{\mathrm{start}} + \frac{1}{2} t_{\mathrm{exp}}$$
4. **Fail-Closed Fallback:** If coordinates or observatory identification are unavailable, `t_converted_bmjd` remains strictly `None`, and `time_conversion_method = "NATIVE_UNCONVERTED"`.

---

**Approved by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026
