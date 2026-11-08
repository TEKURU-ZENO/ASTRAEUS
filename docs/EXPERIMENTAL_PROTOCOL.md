# ASTRAEUS Experimental Protocol & Benchmarking Standards
## Data Splitting, Zero-Leakage Invariants, Baselines & Evaluation Metrics

**Document Identifier:** `ASTRAEUS-PROTO-001`  
**Classification:** Experimental Methodology & Verification Standard  
**Version:** 1.0.0  

---

## 1. Zero-Leakage Data Partitioning Protocol

Machine learning applied to astronomy frequently suffers from severe data leakage due to overlapping survey footprints, localized telescope point spread function (PSF) variations, atmospheric seeing correlations, and light curve interpolation into unobserved future epochs.

ASTRAEUS enforces a strict three-dimensional isolation protocol:

```
                          SKY PARTITIONING
              (HEALPix N_side = 32 -> 12,288 Pixels)
           ┌──────────────────────────────────────────┐
           │   Train: 70%   │   Val: 15%   │ Test: 15%│
           └──────────────────────────────────────────┘
                                 │
                                 ▼
                         TEMPORAL CUTOFF
                     (T_split = MJD 60000.0)
     Historical Light Curves (<= T_split) ──► Training & Validation
     Future Transients       (>  T_split) ──► Out-of-Distribution Test
                                 │
                                 ▼
                     CROSS-SURVEY ZERO SHOT
      Trained on: ZTF (Optical g, r)
      Evaluated on: Pan-STARRS (g, r, i, z, y) / DECam (u, g, r, i, z, Y)
```

1. **Spatial Sky Partitioning (HEALPix $N_{\mathrm{side}}=32$):**
   - Sky is discretized into 12,288 equal-area pixels ($\approx 1.83^\circ$ per pixel).
   - Entire pixels are allocated exclusively to **Train (70%)**, **Validation (15%)**, or **Test (15%)**.
   - No celestial object in the test set shares a HEALPix pixel with any training object.
2. **Temporal Split ($T_{\mathrm{split}}$):**
   - All light curves are partitioned chronologically.
   - Models are trained on data preceding $T_{\mathrm{split}}$ and evaluated on newly emerging transients occurring after $T_{\mathrm{split}}$.
3. **Cross-Survey Zero-Shot Generalization:**
   - Evaluated without retraining on independent survey pipelines (Pan-STARRS and DECam).

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
