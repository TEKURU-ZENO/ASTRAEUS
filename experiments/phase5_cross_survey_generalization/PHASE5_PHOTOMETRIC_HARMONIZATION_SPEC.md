# ASTRAEUS Phase 5: Photometric Harmonization & Synthetic Bandpass Specification

**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Date:** November 16, 2026  
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
| **`ATLAS_c`** | ATLAS | $5330.0\,\text{Å}$ | $2000\,\text{Å}$ | Broad composite "cyan" ($g + r$) |
| **`ATLAS_o`** | ATLAS | $6790.0\,\text{Å}$ | $2300\,\text{Å}$ | Broad composite "orange" ($r + i$) |
| **`DECam_g`** | DECam | $4730.0\,\text{Å}$ | $1480\,\text{Å}$ | Deep Southern sky blue band |
| **`DECam_r`** | DECam | $6420.0\,\text{Å}$ | $1480\,\text{Å}$ | Deep Southern sky red band |
| **`DECam_i`** | DECam | $7840.0\,\text{Å}$ | $1470\,\text{Å}$ | Deep Southern sky NIR band |
| **`PS1_g`** | Pan-STARRS1 | $4810.0\,\text{Å}$ | $1053\,\text{Å}$ | Medium-Deep blue band |
| **`PS1_r`** | Pan-STARRS1 | $6170.0\,\text{Å}$ | $1252\,\text{Å}$ | Medium-Deep red band |
| **`LSST_r`** | Rubin | $6220.0\,\text{Å}$ | $1370\,\text{Å}$ | Prospective ultra-deep red band |

---

## 2. Synthetic Bandpass Photometry Formulation

When a calibrated spectral energy distribution $f_\lambda(\lambda, t)$ (in $\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\text{Å}^{-1}$) or $f_\nu(\nu, t)$ (in $\mu\mathrm{Jy}$) is available, the synthetic flux detected by a photon-counting detector through a filter transmission curve $R(\lambda)$ is governed by:

$$F_\nu = \frac{\int f_\nu(\nu(\lambda), t) \, R(\lambda) \, \lambda^{-1} \, d\lambda}{\int R(\lambda) \, \lambda^{-1} \, d\lambda}$$

In terms of specific flux density $f_\lambda$:
$$F_\nu = \frac{1}{c} \frac{\int f_\lambda(\lambda, t) \, R(\lambda) \, \lambda \, d\lambda}{\int R(\lambda) \, \lambda^{-1} \, d\lambda} \times 10^{29}\,\mu\mathrm{Jy}$$

---

## 3. Empirical Color Transformation & Covariance Propagation

For observations without full continuous SEDs, empirical stellar and supernova color transformations (Tonry et al. 2018; Smith et al. 2020) are permitted **only under State 2 conditions** ($F > 0$, $\mathrm{SNR} \ge 3.0$, $-0.4 \le (g - r) \le 2.0$):

### 3.1 Forward Transformation Equations (AB Magnitudes):
$$c = 0.593\,g + 0.407\,r + 0.016$$
$$o = 0.222\,g + 0.778\,r - 0.035$$

### 3.2 Exact Inversion Matrix $A^{-1}$:
$$\begin{pmatrix} g \\ r \end{pmatrix} = \begin{pmatrix} 0.593 & 0.407 \\ 0.222 & 0.778 \end{pmatrix}^{-1} \begin{pmatrix} c - 0.016 \\ o + 0.035 \end{pmatrix} = \begin{pmatrix} 2.0963 & -1.0966 \\ -0.5982 & 1.5978 \end{pmatrix} \begin{pmatrix} c - 0.016 \\ o + 0.035 \end{pmatrix}$$

### 3.3 Exact Propagated Variances on Independent Inputs:
$$\operatorname{Var}(c) = (0.593)^2 \sigma_g^2 + (0.407)^2 \sigma_r^2 + \sigma_{\mathrm{model}}^2$$
$$\operatorname{Var}(o) = (0.222)^2 \sigma_g^2 + (0.778)^2 \sigma_r^2 + \sigma_{\mathrm{model}}^2$$

### 3.4 Strictly Positive Cross-Filter Covariance:
$$\operatorname{Cov}(c, o) = (0.593 \times 0.222)\,\sigma_g^2 + (0.407 \times 0.778)\,\sigma_r^2 \approx 0.1316\,\sigma_g^2 + 0.3166\,\sigma_r^2 > 0$$

Downstream machine-learning sequence models must ingest the complete $2 \times 2$ photometric covariance matrix:
$$\Sigma_{\mathrm{phot}} = \begin{pmatrix} \operatorname{Var}(c) & \operatorname{Cov}(c, o) \\ \operatorname{Cov}(c, o) & \operatorname{Var}(o) \end{pmatrix}$$

Treating $c$ and $o$ as statistically independent observations when derived from the same $(g, r)$ epoch is mathematically flawed and strictly prohibited.

---

## 4. State 3: Native Passband Retention Rules

Whenever an observation exhibits:
1. Non-positive difference flux ($F \le 0.0\,\mu\mathrm{Jy}$),
2. Input flux with $\mathrm{SNR} < 3.0$,
3. Measured color $(g - r) < -0.4$ or $(g - r) > 2.0$, or
4. An isolated single-passband exposure without synchronous multi-band coverage ($\Delta t > 0.2\,\mathrm{d}$),

**The observation must remain strictly in its native passband.**  
The transformation state is tagged `NATIVE_PASSBAND_UNTRANSFORMED`. Downstream sequence models must use filter-specific positional encodings rather than transformed synthetic fluxes.

---

**Approved by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 16, 2026
