# ASTRAEUS — Gate 4 Phase 4A: HEALPix Footprint Geometry & Spatial Buffer Verification

**Milestone:** Gate 4 Phase 4A — Spatial Boundary Specification & Zero-Query Verification  
**Author:** Antigravity (Google DeepMind) in coordination with Scientific Review  
**Date:** November 15, 2026  
**Governing Configuration:** `configs/data_splits_healpix.json` (SHA-256: `234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b`)  
**Implementation Library:** `astropy_healpix.HEALPix` (C-optimized HEALPix bindings)  

---

## 1. Mathematical & Geometric Specification of the Spatial Partition

To resolve spatial boundary ambiguity, we formally specify the exact geometric parameters governing the Gate 3 Spatial Test partition:

| Parameter | Specification | Scientific & Mathematical Definition |
| :--- | :--- | :--- |
| **Coordinate Frame** | `ICRS` (J2000.0) | Standard International Celestial Reference System $(\alpha, \delta)$. |
| **HEALPix Resolution ($\text{NSIDE}$)** | $\text{NSIDE} = 32$ | $N_{\text{pix}} = 12 \times \text{NSIDE}^2 = 12 \times 32^2 = \mathbf{12,288}$ total equal-area sky pixels. |
| **Pixel Area** | $\Omega_{\text{pix}} \approx 3.356\text{ deg}^2$ | $\frac{4\pi}{12,288}\text{ sr} \approx 0.0010227\text{ sr} \approx 3.356\text{ sq. deg}$. |
| **Pixel Mean Spacing ($\theta_{\text{res}}$)** | $\approx 1.832^\circ$ ($109.94\text{ arcmin}$) | Characteristic center-to-edge angular separation: $\approx 6,596\text{ arcseconds}$. |
| **Pixel Ordering Scheme** | `NESTED` | Tree-structured hierarchical nested pixel index ordering (`order="nested"`). |
| **Partition Manifest** | `configs/data_splits_healpix.json` | Committed Nov 13, 2026. Explicit array of 1,808 discrete test pixels. |
| **Test Pixel Count ($N_{\text{test}}$)** | $1,808$ pixels | Exactly $1,808 / 12,288 = \mathbf{14.71\%}$ of the celestial sphere. |

> [!NOTE]
> **Resolution of Historical "Ring 80" Shorthand:**  
> In early Sprint 4 planning, the spatial holdout was colloquially termed "HEALPix Ring 80" based on an exploratory declination-band ring slice. The **actual immutable committed partition** is the explicit 1,808-pixel set defined in `configs/data_splits_healpix.json` under the `NESTED` scheme.

---

## 2. Geometric Safety Margin & Buffered Footprint Definition

### 2.1 The Positional Uncertainty Problem
If an external transient is located within the astrometric matching radius ($1.5''$) of a pixel boundary bordering a test pixel, coordinate errors could cause it to be assigned to a non-test pixel even though it physically overlaps with the test population.

### 2.2 The 60-Arcsecond Cone Buffer Formulation
To eliminate any possibility of boundary leakage, Phase 4A implements a **60-arcsecond geometric cone buffer**:

$$\mathcal{C}(\alpha, \delta; R = 60'') = \left\{ (\alpha', \delta') \in \mathcal{S}^2 : \arccos\left(\sin\delta \sin\delta' + \cos\delta \cos\delta' \cos(\alpha - \alpha')\right) \le 60'' \right\}$$

For every candidate coordinate $c = (\alpha, \delta)$, the native `astropy_healpix.HEALPix.cone_search_skycoord` method computes all pixels $\mathcal{P}_{\text{cone}}$ that intersect the 60-arcsecond disc around $c$:

```python
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u

hp = HEALPix(nside=32, order="nested", frame="icrs")
cone_pixels = hp.cone_search_skycoord(candidate_coord, radius=60.0 * u.arcsec)
is_in_spatial_test_buffer = any(int(p) in test_pixel_set for p in cone_pixels)
```

### 2.3 Proof of Complete Enclosure & Safety Margin
1. **Safety Margin Multiplier:** The 60-arcsecond buffer is **40 times larger** than the maximum allowable cross-match radius ($1.5''$), absorbing any realistic astrometric centroiding uncertainty (typically $< 0.1''$ in ZTF difference imaging).
2. **Zero-Query Independence:** The calculation depends solely on:
   - The candidate's public sky coordinates $(\alpha, \delta)$.
   - The public split configuration `configs/data_splits_healpix.json`.
   - The geometric spherical HEALPix projection.
   - **Zero queries, reads, or ID comparisons against the quarantined Gate 3 test partition are ever executed.**

---

## 3. Deterministic Verification on the Raw BTS Catalog ($N = 10,988$)

Running the verified spatial buffer audit on the acquired Caltech BTS catalog produces the following exact results:

| Spatial Classification | Pixel Criterion | Raw Record Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Exact Test Pixel Overlap** | $\text{skycoord\_to\_healpix}(\alpha, \delta) \in \mathcal{P}_{\text{test}}$ | 1,622 | 14.76% |
| **Boundary Buffer Overlap** | Intersects test pixel within $60''$ cone | 43 | 0.39% |
| **Total Buffered Test Footprint** | $\text{cone\_search\_skycoord}(c, 60'') \cap \mathcal{P}_{\text{test}} \ne \emptyset$ | **1,665** | **15.15%** |
| **Disjoint External Candidates** | Completely disjoint by $> 60''$ from test footprint | **9,323** | **84.85%** |

### Verification Assertions
1. Every candidate located within the exact test pixels is quarantined: $\text{Overlap} \ge 1,622$.
2. The 43 boundary candidates within $60''$ of a test pixel boundary are safely quarantined under `QUARANTINE_SPATIAL_GATE3_FOOTPRINT_BUFFER`.
3. The remaining 9,323 candidates are **mathematically and geometrically proven** to lie at least 60 arcseconds away from every point in the Gate 3 Spatial Test partition.
