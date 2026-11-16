# ASTRAEUS Phase 5: Probabilistic Bayesian Cross-Matching & Stable Identity Registry Specification

**Author:** Dev Mehta  
**Lead Investigator & Mission Director, Project ASTRAEUS**  
**Date:** November 16, 2026  
**Document Identifier:** `ASTRAEUS-P5-XMATCH-002`  
**Milestone:** Phase 5 — Workstream 3 (Identity and Provenance Resolution)  
**Governing Status:** Formal Mathematical Specification & Implementation Charter  

---

## 1. Executive Summary & Problem Formulation

In astronomical cross-survey matching, naive nearest-neighbor search within an arbitrary fixed radius (e.g., $3''$ or $5''$) introduces severe systematic errors:
1. **False positive contamination** in dense stellar fields (e.g., low Galactic latitudes, inner host-galaxy disks).
2. **Failure to quantify candidate competition** when multiple plausible counterparts lie within the search region.
3. **Absence of an explicit no-counterpart hypothesis ($H_0$)**, leading to false associations for transients that were non-detections or below the survey limit.
4. **Instability of coordinate-derived primary keys**, where sub-arcsecond measurement noise across catalogs produces conflicting identifiers for the same astrophysical entity.

This specification establishes:
1. A rigorous **competing-hypothesis Bayesian generative model** based on Sutherland & Saunders (1992) and Budavári & Szalay (2008).
2. An exact **likelihood normalization framework** incorporating the full $2 \times 2$ astrometric covariance matrix $\Sigma$ and local background surface density $\rho_{\mathrm{local}}$.
3. An empirical **posterior probability calibration protocol** evaluated against positive controls, artificial coordinate perturbations, and negative control fields.
4. A **persistent Entity Registry architecture** decoupling immutable astrophysical object identities from transient observational measurements.

---

## 2. Mathematical Derivation of the Competing-Hypothesis Bayesian Matcher

Let a query transient event have celestial coordinate $\mathbf{x}_{\mathrm{query}} = (\alpha_0, \delta_0)$ with astrometric covariance $\Sigma_{\mathrm{query}}$. A spatial cone search with maximum radius $R_{\mathrm{max}}$ defines an evaluation region of solid angle:
$$\Omega = 2\pi (1 - \cos R_{\mathrm{max}}) \approx \pi R_{\mathrm{max}}^2$$

Suppose the target survey catalog contains $K$ candidate detections within $\Omega$ at positions $\mathbf{x}_j = (\alpha_j, \delta_j)$ with covariances $\Sigma_{\mathrm{cand}, j}$ for $j \in \{1, \dots, K\}$.

### 2.1 The Hypotheses Set:
We define a set of $K + 1$ mutually exclusive and exhaustive hypotheses:
- **$H_0$ (Null Hypothesis):** None of the $K$ candidates is the physical counterpart of the query event (i.e., all $K$ candidates are unrelated background sources, and the true counterpart was unobserved or below the detection threshold).
- **$H_j$ (Candidate $j$ Hypothesis):** Candidate $j$ is the true physical counterpart of the query event, while all other $K - 1$ candidates $k \neq j$ are unrelated background sources.

### 2.2 Prior Probabilities:
Let $Q \in (0, 1)$ denote the prior probability that the true counterpart of the query event is present in the target catalog within the search area $\Omega$ (the survey completeness prior, typically $Q \approx 0.90\text{--}0.95$ for bright confirmed SNe).
- Prior for the null hypothesis:
  $$\pi_0 = P(H_0) = 1 - Q$$
- Under the absence of candidate-specific prior information, the prior probability that candidate $j$ is the counterpart is:
  $$\pi_j = P(H_j) = \frac{Q}{K_{\mathrm{expected}}}$$
  where $K_{\mathrm{expected}} = \rho_{\mathrm{local}} \Omega + Q$. In the standard Sutherland & Saunders formulation, the relative prior ratio between any counterpart candidate and the background is governed by $Q / (1 - Q)$.

### 2.3 Astrometric Likelihood Model:
Under hypothesis $H_j$, the positional displacement $\Delta\mathbf{x}_j = \mathbf{x}_j - \mathbf{x}_{\mathrm{query}}$ follows a two-dimensional Gaussian distribution governed by the combined astrometric covariance matrix:
$$\Sigma_j = \Sigma_{\mathrm{query}} + \Sigma_{\mathrm{cand}, j}$$
$$\det \Sigma_j = (\sigma_{\alpha, \mathrm{tot}} \cos \delta_0)^2 \, \sigma_{\delta, \mathrm{tot}}^2 - \sigma_{\alpha\delta}^2$$

The positional likelihood is:
$$p(\Delta\mathbf{x}_j \mid H_j) = \frac{1}{2\pi \sqrt{\det \Sigma_j}} \exp\left(-\frac{1}{2} \Delta\mathbf{x}_j^T \Sigma_j^{-1} \Delta\mathbf{x}_j\right)$$

In the isotropic limit where $\sigma_j^2 = \sigma_{\mathrm{query}}^2 + \sigma_{\mathrm{cand}, j}^2$ and angular separation is $\psi_j = \|\Delta\mathbf{x}_j\|$:
$$p(\psi_j \mid H_j) = \frac{1}{2\pi \sigma_j^2} \exp\left(-\frac{\psi_j^2}{2\sigma_j^2}\right)$$

### 2.4 Background Likelihood Model:
Under the background hypothesis, sources are distributed as a spatial Poisson process with local surface density $\rho_{\mathrm{local}}$ (measured in $\text{arcsec}^{-2}$). Within solid angle $\Omega$:
$$p(\mathbf{x}_j \mid \text{background}) = \rho_{\mathrm{local}}$$

### 2.5 Likelihood Ratio Definition:
For each candidate $j \in \{1, \dots, K\}$, the dimensionless likelihood ratio comparing the counterpart hypothesis to the background hypothesis is:
$$\ell_j = \frac{p(\Delta\mathbf{x}_j \mid H_j)}{\rho_{\mathrm{local}}} = \frac{1}{2\pi \rho_{\mathrm{local}} \sqrt{\det \Sigma_j}} \exp\left(-\frac{1}{2} \Delta\mathbf{x}_j^T \Sigma_j^{-1} \Delta\mathbf{x}_j\right)$$

### 2.6 Exact Posterior Association Probabilities:
Applying Bayes' theorem over the exhaustive set of mutually exclusive hypotheses $\{H_0, H_1, \dots, H_K\}$:

$$P(H_j \mid D) = \frac{\pi_j p(D \mid H_j)}{\pi_0 p(D \mid H_0) + \sum_{k=1}^K \pi_k p(D \mid H_k)} = \frac{\ell_j}{\sum_{k=1}^K \ell_k + \frac{1 - Q}{Q}}$$

and the posterior probability of no counterpart in the region is:
$$P(H_0 \mid D) = \frac{\frac{1 - Q}{Q}}{\sum_{k=1}^K \ell_k + \frac{1 - Q}{Q}} = 1 - \sum_{j=1}^K P(H_j \mid D)$$

### 2.7 Key Mathematical Invariants of This Formulation:
1. **Strict Normalization:** $\sum_{j=0}^K P(H_j \mid D) = 1.0$ is guaranteed algebraically.
2. **Candidate Competition:** If two candidate counterparts $A$ and $B$ both lie close to the query position with large likelihood ratios $\ell_A \approx \ell_B \gg 1$, neither can achieve $P(H) > 0.50$. The algorithm naturally flags the association as ambiguous.
3. **Local Density Sensitivity:** In dense Galactic fields ($\rho_{\mathrm{local}} \gg 0.01\,\text{arcsec}^{-2}$), $\ell_j$ is severely penalized, preventing chance stellar alignments from masquerading as secure associations.
4. **Distance Penalty:** As separation $\psi \to R_{\mathrm{max}}$, $p(\psi \mid H_j) \to 0$, driving $\ell_j \to 0$ and $P(H_0 \mid D) \to 1.0$.

---

## 3. Multi-Modal Consistency & Factorization Rules

The raw positional likelihood ratio $\ell_j$ may be modulated by independent astrophysical evidence **only where conditional independence is physically and mathematically defensible**:

$$\ell_j^{\mathrm{augmented}} = \ell_j \times \mathcal{M}_{\mathrm{temporal}} \times \mathcal{M}_{\mathrm{redshift}}$$

### 3.1 Strict Conditioning Rules:
1. **Temporal Coincidence Factor ($\mathcal{M}_{\mathrm{temporal}}$):**
   - Transient emission is temporally bounded. An observation taken $> 150$ days before explosion or $> 300$ days post-explosion cannot represent transient flux from the same event (barring recurring novae).
   - We define $\mathcal{M}_{\mathrm{temporal}}$ as an explicit step-window prior:
     $$\mathcal{M}_{\mathrm{temporal}} = \begin{cases}
     1.0 & \text{if } t_{\mathrm{obs}} \in [t_{\mathrm{disc}} - 60\,\mathrm{d}, t_{\mathrm{disc}} + 180\,\mathrm{d}] \\
     \exp\left(-\frac{(|t_{\mathrm{obs}} - t_{\mathrm{disc}}| - 180)^2}{2 \times 30^2}\right) & \text{if } 180\,\mathrm{d} < |t_{\mathrm{obs}} - t_{\mathrm{disc}}| \le 300\,\mathrm{d} \\
     10^{-4} & \text{if } |t_{\mathrm{obs}} - t_{\mathrm{disc}}| > 300\,\mathrm{d} \text{ (quarantined)}
     \end{cases}$$
2. **Spectroscopic Redshift Consistency ($\mathcal{M}_{\mathrm{redshift}}$):**
   - If independent spectra are reported for the query and candidate, the host or transient redshifts must agree within instrument resolution:
     $$|\Delta z| \le 0.001 (1 + z)$$
   - Discrepant redshifts immediately force $\mathcal{M}_{\mathrm{redshift}} = 0.0$ (definitive rejection).
3. **No Unjustified Flux Multipliers:**
   - Multiplicative magnitude or color priors must **NOT** be multiplied into the likelihood ratio unless derived from a joint color-magnitude prior distribution $q(m, c)$ formally trained on external survey catalogs.

---

## 4. Association Dispositions & Thresholds

| Match Status Code | Posterior Probability Criterion | Candidate Competition Invariant | Scientific Action |
| :--- | :---: | :---: | :--- |
| **`SECURE_MATCH`** | **$P(H_j \mid D) \ge 0.90$** | $P(H_k \mid D) < 0.10$ for all $k \neq j$ | Admitted to primary cross-survey benchmark cohort. |
| **`PROBABLE_MATCH`** | **$0.70 \le P(H_j \mid D) < 0.90$** | No competing candidate with $P > 0.20$ | Excluded from primary benchmark; reserved for sensitivity analysis. |
| **`AMBIGUOUS_MATCH`** | **$P(H_j \mid D) \ge 0.20$** | Second candidate exists with $P(H_k \mid D) \ge 0.20$ | Sequestered in Population 4 (Identity Conflict Ledger). |
| **`NO_MATCH`** | **$P(H_0 \mid D) > 0.80$** | No candidate satisfies $\ell_j \ge 1.0$ | Classified as non-detection or catalog orphan. |

---

## 5. Empirical Calibration & Validation Protocol

The match probabilities $P(H_j \mid D)$ must be empirically calibrated. An algorithm that outputs $P = 0.95$ must demonstrate an empirical false match rate $\le 5\%$.

### Validation Tests:
1. **Positive Control Test ($0.0''$ Offset):**
   - High-SNR confirmed matches with known counterparts must yield $P(H_1 \mid D) \ge 0.95$.
2. **Artificial Coordinate Shift Test:**
   - Inject controlled offsets $\Delta \psi \in \{0.5'', 1.0'', 2.0'', 3.5'', 5.0''\}$ into known pairs.
   - Posterior $P(H_1 \mid D)$ must decrease monotonically with separation.
3. **Negative Control Test (Randomized Offsets):**
   - Query random sky positions offset by $> 30''$ in extragalactic fields.
   - The null posterior $P(H_0 \mid D)$ must exceed $0.98$ in $> 95\%$ of trials.
4. **Dense Field Competition Test:**
   - Inject a synthetic competing source at separation $1.0''$.
   - Confirm that candidate probabilities drop proportionally and status flips from `SECURE_MATCH` to `AMBIGUOUS_MATCH`.

---

## 6. Stable Entity Registry Architecture

To prevent coordinate precision or catalog discrepancies from fragmenting transient identities, Phase 5 implements the **Stable Entity Registry**:

```
ASTRAEUS Persistent Entity Record
├── astraeus_id:           ASTRAEUS-EVT-00000001 (Sequential/UUID Permanent Key)
├── iau_name:              SN 2020adv
├── ztf_id:                ZTF17aadlxmv
├── atlas_id:              ATLAS20adv
├── decam_id:              None
├── ps1_id:                PS1-20adv
├── discovery_mjd:         58863.2
├── reference_ra_deg:      127.44829
├── reference_dec_deg:     33.90633
├── identity_status:       SECURELY_ASSOCIATED
└── Entity Relationships:
    ├── Host Galaxy:       SDSS J082947.59+335422.8 (z = 0.062)
    └── Observations:      [OBS-0001, OBS-0002, ..., OBS-0054]
```

### Entity Hierarchy Rules:
1. **Permanent Internal Key:** Once assigned, `ASTRAEUS-EVT-XXXXX` is immutable. Refining the coordinate from new Gaia or HST astrometry updates `reference_ra_deg` but **never** alters `astraeus_id`.
2. **Coordinate Hashes as Secondary Candidate Keys:** Hashed spatial keys (e.g. `ASTRAEUS-P5-261116-XXXXXXXX`) are treated strictly as non-unique spatial indexing hashes, never as primary keys.
3. **Decoupling Entities from Exposures:** An event is an astrophysical transient phenomenon. An exposure is an instrumental observation at an epoch. An event possesses many observations across surveys.
