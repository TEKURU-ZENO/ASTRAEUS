# ASTRAEUS Phase 5: Photometric Harmonization & Synthetic Bandpass Specification

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026 (Remediated Checkpoint)  
**Document Identifier:** `ASTRAEUS-P5-PHOTOM-SPEC-001`  
**Milestone:** Phase 5 — Workstream 4 (Deliverable D4.2)  
**Governing Status:** Physical & Mathematical Filter Harmonization Charter  

---

## 1. Executive Summary & Filter Passband Architecture

Cross-survey astronomical inference requires rigorous translation between heterogeneous instrument passbands without distorting underlying spectral energy distributions (SEDs). 

Workstream 4 standardizes passbands across five optical/NIR survey systems:

| Passband Identifier | Survey Facility | Central $\lambda_{\mathrm{eff}}$ | Effective Bandwidth ($\Delta\lambda$) | Filter Class & Primary Function |
| :--- | :---: | :---: | :---: | :--- |
| **`ZTF_g`** | ZTF | $4722.7\,\text{Å}$ | $1290\,\text{Å}$ | Source reference blue band |
| **`ZTF_r`** | ZTF | $6339.6\,\text{Å}$ | $1380\,\text{Å}$ | Source reference red band |
| **`ZTF_i`** | ZTF | $7886.1\,\text{Å}$ | $1420\,\text{Å}$ | Source reference NIR band |
| **`ATLAS_c`** | ATLAS | $5330.0\,\text{Å}$ | $2280\,\text{Å}$ | Broad composite "cyan" ($g + r$) |
| **`ATLAS_o`** | ATLAS | $6790.0\,\text{Å}$ | $2570\,\text{Å}$ | Broad composite "orange" ($r + i$) |
| **`DECam_g`** | DECam | $4720.0\,\text{Å}$ | $1520\,\text{Å}$ | Deep Southern sky blue band |
| **`DECam_r`** | DECam | $6415.0\,\text{Å}$ | $1480\,\text{Å}$ | Deep Southern sky red band |
| **`DECam_i`** | DECam | $7835.0\,\text{Å}$ | $1470\,\text{Å}$ | Deep Southern sky NIR band |
| **`PS1_g`** | Pan-STARRS1 | $4866.0\,\text{Å}$ | $1166\,\text{Å}$ | Medium-Deep blue band |
| **`PS1_r`** | Pan-STARRS1 | $6215.0\,\text{Å}$ | $1315\,\text{Å}$ | Medium-Deep red band |
| **`LSST_r`** | Rubin | $6223.0\,\text{Å}$ | $1378\,\text{Å}$ | Prospective ultra-deep red band |

---

## 2. Synthetic Bandpass Photometry Formulation & Convention Audit

### 2.1 Total System Photon Throughput $T_P(\lambda)$
For any optical survey passband $P$, the total system photon transmission is defined as:
$$T_P(\lambda) = T_{\mathrm{atm}}(\lambda, X) \times T_{\mathrm{optics}}(\lambda) \times T_{\mathrm{filter}, P}(\lambda) \times \mathrm{QE}_{\mathrm{detector}}(\lambda)$$
Where:
- $T_{\mathrm{atm}}(\lambda, X)$ is the topocentric atmospheric transmission at airmass $X$.
- $T_{\mathrm{optics}}(\lambda)$ is the telescope optical train reflection and transmission efficiency.
- $T_{\mathrm{filter}, P}(\lambda)$ is the interference filter transmission curve.
- $\mathrm{QE}_{\mathrm{detector}}(\lambda)$ is the quantum efficiency (photoelectrons generated per incident photon).

$T_P(\lambda)$ is strictly a **dimensionless probability** representing the fraction of top-of-atmosphere photons at wavelength $\lambda$ recorded as photoelectrons.

### 2.2 Photon-Counting Detector Response & Band-Averaged Flux Density $\langle F_\nu \rangle$
Astronomical charge-coupled devices (CCDs) and CMOS sensors are **photon-counting detectors**, not energy-integrating bolometers. The number of recorded photoelectrons per unit telescope collecting area and exposure time is proportional to the photon arrival rate:
$$N_{e^-} \propto \int \frac{f_\lambda(\lambda, t)}{h\nu} \, T_P(\lambda) \, d\lambda = \frac{1}{hc} \int f_\lambda(\lambda, t) \, T_P(\lambda) \, \lambda \, d\lambda$$

In the standard monochromatic AB magnitude system (Oke & Gunn 1983; Fukugita et al. 1996), the band-averaged flux density $\langle F_\nu \rangle$ (in frequency units) detected by a photon-counting detector is defined as:
$$\langle F_\nu \rangle_P = \frac{\int f_\nu(\nu) \, T_P(\nu) \, \frac{d\nu}{h\nu}}{\int T_P(\nu) \, \frac{d\nu}{h\nu}} = \frac{\int f_\nu(\nu(\lambda)) \, T_P(\lambda) \, \frac{d\lambda}{\lambda}}{\int T_P(\lambda) \, \frac{d\lambda}{\lambda}}$$

Using the differential relation $f_\nu(\nu) = \frac{\lambda^2}{c} f_\lambda(\lambda)$, this yields the authoritative photon-weighted synthetic flux density:
$$\langle F_\nu \rangle_P = \frac{1}{c} \frac{\int_0^\infty f_\lambda(\lambda, t) \, T_P(\lambda) \, \lambda \, d\lambda}{\int_0^\infty T_P(\lambda) \, \lambda^{-1} \, d\lambda}$$

### 2.3 Dimensional Verification & Physical Units
- **Numerator:** $\int f_\lambda \, T_P(\lambda) \, \lambda \, d\lambda$ has dimensions:
  $$[\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\text{Å}^{-1}] \times [1] \times [\text{Å}] \times [\text{Å}] = \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\text{Å}$$
- **Denominator:** $\int T_P(\lambda) \, \lambda^{-1} \, d\lambda$ has dimensions:
  $$[1] \times [\text{Å}^{-1}] \times [\text{Å}] = \text{dimensionless}$$
- **Speed of Light:** $c = 2.99792458 \times 10^{18}\,\text{Å}\,\mathrm{s}^{-1}$.
- **Resulting Flux Density:**
  $$[\langle F_\nu \rangle_P] = \frac{\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\text{Å}}{\text{Å}\,\mathrm{s}^{-1}} = \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1} \equiv \mathrm{cgs\ frequency\ flux\ density}$$

In physical units of microJanskys ($\mu\mathrm{Jy}$), where $1\,\mathrm{Jy} = 10^{-23}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}$ and $1\,\mu\mathrm{Jy} = 10^{-29}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}$:
$$\langle F_\nu \rangle_P \, [\mu\mathrm{Jy}] = \frac{10^{29}}{c \, [\text{Å}\,\mathrm{s}^{-1}]} \frac{\int_0^\infty f_\lambda(\lambda, t) \, T_P(\lambda) \, \lambda \, d\lambda}{\int_0^\infty T_P(\lambda) \, \lambda^{-1} \, d\lambda}$$

### 2.4 Energy-Weighted Response Distinction
For comparison, a pure energy-integrating device (e.g. bolometer) measures the energy-weighted band-averaged flux density $\langle F_\lambda \rangle$:
$$\langle F_\lambda \rangle_P = \frac{\int f_\lambda(\lambda) \, T_P(\lambda) \, d\lambda}{\int T_P(\lambda) \, d\lambda}$$
ASTRAEUS enforces the photon-weighted $\langle F_\nu \rangle$ formulation for all optical survey sensors (ZTF, ATLAS, DECam, PS1, Rubin).

### 2.5 AB Reference Flux and Interstellar Extinction
- The AB zero-point reference flux density is immutable:
  $$F_{\nu, 0} = 3631\,\mathrm{Jy} = 3.631 \times 10^9\,\mu\mathrm{Jy} = 3.631 \times 10^{-20}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}$$
  $$m_{\mathrm{AB}} = -2.5 \log_{10} \left( \frac{\langle F_\nu \rangle}{F_{\nu, 0}} \right) = -2.5 \log_{10} \left( \frac{\langle F_\nu \rangle_{\mu\mathrm{Jy}}}{10^{(23.90 / 2.5)}} \right) = -2.5 \log_{10}(\langle F_\nu \rangle_{\mu\mathrm{Jy}}) + 23.900$$
- **Interstellar Extinction:** Top-of-atmosphere fluxes are dereddened using the Fitzpatrick (1999) dust attenuation curve with $R_V = 3.1$, anchored to the Schlafly & Finkbeiner (2011) re-calibrated dust maps:
  $$f_\lambda^{\mathrm{dered}}(\lambda) = f_\lambda^{\mathrm{obs}}(\lambda) \times 10^{0.4 \, A(\lambda)}, \quad A(\lambda) = E(B-V) \left[ R_V \frac{A(\lambda)}{A_V} \right]$$

---

## 3. Empirical Color Transformation & Full Covariance Architecture

For observations lacking full SEDs, empirical polynomials (Tonry et al. 2018; Smith et al. 2020) are permitted **only under State 2 conditions** ($F > 0$, $\mathrm{SNR} \ge 3.0$, $-0.4 \le (g - r) \le 2.0$):

### 3.1 Forward Transformation Equations (AB Magnitudes):
$$c = 0.593\,g + 0.407\,r + 0.016$$
$$o = 0.222\,g + 0.778\,r - 0.035$$

Transformation Jacobian matrix:
$$\mathbf{J} = \begin{pmatrix} 0.593 & 0.407 \\ 0.222 & 0.778 \end{pmatrix}$$

Condition number: $\kappa(\mathbf{J}) = \sigma_{\max} / \sigma_{\min} \approx 1.058 / 0.371 \approx 2.85$. The linear mapping is well-conditioned.

### 3.2 Exact Inversion Matrix $\mathbf{J}^{-1}$:
$$\begin{pmatrix} g \\ r \end{pmatrix} = \begin{pmatrix} 0.593 & 0.407 \\ 0.222 & 0.778 \end{pmatrix}^{-1} \begin{pmatrix} c - 0.016 \\ o + 0.035 \end{pmatrix} = \begin{pmatrix} 2.097035 & -1.097035 \\ -0.598383 & 1.598383 \end{pmatrix} \begin{pmatrix} c - 0.016 \\ o + 0.035 \end{pmatrix}$$

### 3.3 Full Magnitude Covariance Matrix:
The complete uncertainty model incorporates input measurement covariance, shared calibration/zero-point uncertainty, model residual dispersion, and temporal mismatch:

$$\mathbf{\Sigma}_{c,o}^{(\mathrm{mag})} = \mathbf{J} \mathbf{\Sigma}_{\mathrm{input}} \mathbf{J}^T + \mathbf{J} \mathbf{\Sigma}_{\mathrm{cal}} \mathbf{J}^T + \mathbf{\Sigma}_{\mathrm{model}} + \mathbf{\Sigma}_{\Delta t}$$

Where:
1. **Input Measurement Covariance:**
   $$\mathbf{\Sigma}_{\mathrm{input}} = \begin{pmatrix} \sigma_g^2 & \operatorname{Cov}(g, r) \\ \operatorname{Cov}(g, r) & \sigma_r^2 \end{pmatrix}$$
2. **Shared Calibration Covariance:**
   $$\mathbf{\Sigma}_{\mathrm{cal}} = \begin{pmatrix} \sigma_{\mathrm{zp}, g}^2 & \rho_{\mathrm{zp}} \sigma_{\mathrm{zp}, g} \sigma_{\mathrm{zp}, r} \\ \rho_{\mathrm{zp}} \sigma_{\mathrm{zp}, g} \sigma_{\mathrm{zp}, r} & \sigma_{\mathrm{zp}, r}^2 \end{pmatrix}$$
3. **Model Residual Covariance:**
   $$\mathbf{\Sigma}_{\mathrm{model}} = \begin{pmatrix} \sigma_{\mathrm{model}, c}^2 & \operatorname{Cov}_{\mathrm{model}}(c, o) \\ \operatorname{Cov}_{\mathrm{model}}(c, o) & \sigma_{\mathrm{model}, o}^2 \end{pmatrix}$$
4. **Intra-Epoch Temporal Mismatch Uncertainty:**
   For non-simultaneous observations with time separation $\Delta t = |t_g - t_r| \le 0.1\,\mathrm{days}$ and transient light-curve derivative $\dot{m} = dm/dt$:
   $$\mathbf{\Sigma}_{\Delta t} = \begin{pmatrix} (\dot{m} \Delta t)^2 & 0 \\ 0 & (\dot{m} \Delta t)^2 \end{pmatrix}$$

### 3.4 Propagation into Physical Flux Space ($\mu\mathrm{Jy}^2$):
Because physical difference-flux sequence models ingest flux densities $F$ rather than magnitudes $m$, errors must propagate through the nonlinear mapping $F = F_0 10^{-0.4 m}$:
$$\frac{\partial F}{\partial m} = -\frac{\ln 10}{2.5} F \approx -0.921034 \times F$$

The flux covariance matrix is:
$$\mathbf{\Sigma}_{F_c, F_o}^{(\mathrm{flux})} = \mathbf{D}_{\mathrm{flux}} \mathbf{\Sigma}_{c,o}^{(\mathrm{mag})} \mathbf{D}_{\mathrm{flux}}^T$$
Where:
$$\mathbf{D}_{\mathrm{flux}} = \begin{pmatrix} -\frac{\ln 10}{2.5} F_c & 0 \\ 0 & -\frac{\ln 10}{2.5} F_o \end{pmatrix}$$

The off-diagonal cross-filter flux covariance is:
$$\operatorname{Cov}(F_c, F_o) = \left( \frac{\ln 10}{2.5} \right)^2 F_c F_o \, \operatorname{Cov}(c, o)^{(\mathrm{mag})} > 0$$

Downstream machine-learning sequence models ingest the complete $2 \times 2$ flux covariance matrix. Assuming $\operatorname{Cov}(F_c, F_o) = 0$ is mathematically and physically invalid.

---

## 4. State 3: Native Passband Retention Rules

Whenever an observation exhibits:
1. Non-positive difference flux ($F \le 0.0\,\mu\mathrm{Jy}$),
2. Input flux with $\mathrm{SNR} < 3.0$,
3. Measured color $(g - r) < -0.4$ or $(g - r) > 2.0$, or
4. An isolated single-passband exposure without synchronous multi-band coverage ($\Delta t > 0.1\,\mathrm{d}$),

**The observation must remain strictly in its native passband.**  
The transformation state is tagged `NATIVE_PASSBAND_UNTRANSFORMED`. Downstream sequence models must use filter-specific positional encodings rather than transformed synthetic fluxes.

---

**Approved by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026 (Remediated Checkpoint)
