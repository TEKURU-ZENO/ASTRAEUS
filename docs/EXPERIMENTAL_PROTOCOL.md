# ASTRAEUS Experimental Protocol & Benchmarking Standards
## Data Splitting, Zero-Leakage Invariants, Baselines & Evaluation Metrics

**Document Identifier:** `ASTRAEUS-PROTO-001`  
**Classification:** Experimental Methodology & Verification Standard  
**Version:** 1.0.0  

---

## 1. Multi-Tier Data Isolation & Leakage Audit Protocol

Machine learning applied to astronomy frequently suffers from severe data leakage due to overlapping survey footprints, localized telescope point spread function (PSF) variations, atmospheric seeing correlations, light curve interpolation into unobserved future epochs, and uncatalogued object aliases.

ASTRAEUS distinguishes between **HEALPix assignment overlap** (a mathematical property of partition sets) and **source-level leakage** (astrophysical identity contamination). A manifest showing disjoint pixels does *not* constitute a universal zero-leakage guarantee.

ASTRAEUS enforces a 6-tier isolation and audit protocol:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   6-TIER LEAKAGE AUDIT PROTOCOL                         │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. HEALPix Partition Overlap      │ Disjoint spatial pixel index sets  │
│ 2. Spatial Boundary Buffer        │ Flag sources within 3-sigma error  │
│ 3. Canonical Object Identity      │ Cross-catalogue ID deduplication   │
│ 4. Cross-Catalogue Aliases        │ Check SIMBAD / NED / TNS aliases   │
│ 5. Host-Galaxy Association        │ Ensure host & transient co-split   │
│ 6. Temporal & Cadence Cutoff      │ Strict T_split chronological gate  │
└───────────────────────────────────┴────────────────────────────────────┘
```

1. **Spatial HEALPix Discretization ($N_{\mathrm{side}}=32$):**
   - Discretizes the celestial sphere into 12,288 equal-area pixels ($\approx 1.83^\circ$ diameter).
   - Entire pixels are allocated to **Train (70%)**, **Validation (15%)**, or **Test (15%)**, guaranteeing **zero HEALPix assignment overlap**.
2. **Spatial Boundary Uncertainty Buffer:**
   - Evaluates sources residing within $3\sigma_{\mathrm{astrometric}}$ of a HEALPix pixel boundary. Sources whose astrometric uncertainty ellipse crosses a split boundary are quarantined.
3. **Canonical Object Identity & Repeated Detections:**
   - Ensures distinct alert packets representing the same astrophysical transient or variable star cannot appear across multiple splits.
4. **Cross-Catalogue Alias Audit:**
   - Resolves multi-survey designations (e.g., ZTF20acvhrgm = AT2020xnd) to prevent cross-catalogue train/test leakage.
5. **Host-Galaxy Association Audit:**
   - Verifies that a transient candidate and its associated host galaxy do not reside in conflicting splits.
6. **Chronological Temporal Split ($T_{\mathrm{split}}$):**
   - Partitions time-series chronologically ($T_{\mathrm{split}} = \mathrm{MJD}\; 60000.0$). Models are trained on historical data up to $T_{\mathrm{split}}$ and evaluated on newly emerging transients occurring after $T_{\mathrm{split}}$.

---

## 2. Probabilistic Gaia DR3 Cross-Matching Protocol

Positional proximity within a fixed angular radius is insufficient for robust astronomical cross-matching. ASTRAEUS implements a Bayesian probabilistic matching framework accounting for:
- Propagated coordinates using Gaia proper motions:
  $$\alpha(t) = \alpha_0 + \frac{\mu_\alpha^*}{\cos \delta_0}(t - t_0), \quad \delta(t) = \delta_0 + \mu_\delta (t - t_0)$$
- Positional covariance ellipse convolution ($\mathbf{\Sigma} = \mathbf{\Sigma}_{\mathrm{alert}} + \mathbf{\Sigma}_{\mathrm{Gaia}}$).
- Local background stellar density $\rho_{\mathrm{local}}$ (sources per $\mathrm{arcsec}^2$).
- Chance coincidence probability:
  $$P_{\mathrm{chance}} = 1 - \exp\left(-\pi \Delta \theta^2 \rho_{\mathrm{local}}\right)$$
- Posterior match probability $P(\mathcal{M} \mid \Delta \theta, \rho)$.
- Tri-state association classification:
  - **`SECURE_MATCH`:** $P(\mathcal{M}) \ge 0.95$ with no secondary candidate with $P > 0.05$.
  - **`AMBIGUOUS_MATCH`:** Multiple candidate counterparts with overlapping probabilities.
  - **`NO_RELIABLE_MATCH`:** No counterpart within search radius or $P(\mathcal{M}) < 0.50$ (scientifically meaningful; indicates survey depth, high extinction, or extragalactic origin).

---

## 2. ASTRAEUS-Bench Core Benchmark Suites

| Benchmark Suite | Evaluated Scientific Domain | Baseline Models | Primary Quantitative Metrics |
| :--- | :--- | :--- | :--- |
| **AstroVision** | Galaxy morphology classification, strong gravitational lensing discovery. | Random Forest, ResNet-50, Swin-T, ViT. | Balanced Macro-F1, mAP, OOD latent entropy. |
| **AstroTemporal** | Variable star classification, light curve phase modeling. | Lomb-Scargle + RF, Gaussian Processes, ASTROMER, Conformer. | Balanced Macro-F1, AUROC, Continuous Ranked Probability Score (CRPS). |
| **AstroFusion** | Cross-modal image-spectra-photometry property regression. | Unimodal MLPs, Late Fusion, AstroCLIP, AION-1. | Cross-modal Retrieval Recall@K, $R^2$, Prediction Interval Width. |
| **AstroAnomaly** | Rare transient discovery, peculiar supernova triage. | Isolation Forest, Local Outlier Factor (LOF), Reconstruction VAE, ASTRAEUS Composite Score. | Precision-Recall AUC, Recall@Budget ($N \le 100$), False Discovery Rate (FDR). |
| **AstroGraph** | Evidence retrieval, scientific citation verification, provenance tracking. | Dense Vector RAG, Vanilla Graph RAG, Hybrid dense-sparse retrieval. | nDCG@K, Provenance Path Recall, Citation Hallucination Rate. |
| **AstroReason** | Hypothesis falsification, adversarial scientific debate consistency. | Single-agent LLM, unconstrained multi-agent debate, REGROUND protocol. | Falsification Success Rate, Expected Calibration Error (ECE) of claims. |

---

## 3. Baseline Implementations & Comparison Standards

Every novel scientific claim must be evaluated against three distinct baseline tiers:
1. **Classical Astrophysical Methods:**
   - Lomb-Scargle periodograms for periodicity extraction.
   - Bazin and Arnett analytical templates for light curve fitting.
   - Astrometric nearest-neighbor catalog association.
2. **Standard Machine Learning Methods:**
   - Random Forest classifiers with hand-crafted morphological/variability features.
   - Isolation Forest and Local Outlier Factor (LOF) for outlier detection.
3. **Contemporary AI / Foundation Model Methods:**
   - ASTROMER 1D Transformer for light-curve embeddings.
   - Standard Dense Vector RAG (Qdrant/Cosine similarity).

---

## 4. Acceptance Criteria for Phase 1 Validation

- **Data Integrity:** Ingestion of $\ge 5 \times 10^5$ light curves with $0\%$ cross-split leakage.
- **Classification Performance:** Baseline ASTROMER achieves balanced Macro-F1 $\ge 0.78$ on the 8 primary ZTF Bright Transient Survey classes under spatial cross-validation.
- **Anomaly Detection:** Composite score $s(x) = \alpha d_M + \beta u + \gamma r_{\mathrm{phys}}$ achieves $\ge 85\%$ Recall@Top-100 on confirmed rare transient test sets.
- **Conformal Calibration:** Conformal prediction intervals achieve empirical coverage $P(y \in \mathcal{C}(x)) \ge 1 - \alpha$ ($\ge 90\%$ coverage for $\alpha=0.10$) across both internal and cross-survey datasets.
- **Reproducibility:** 100% of pipeline executions output verified IVOA PROV-DM entity hashes and git commit fingerprints.
