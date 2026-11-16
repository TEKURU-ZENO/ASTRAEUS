# ASTRAEUS Gate 4 — Work Package 4D: Early-Epoch Physics & SN Disambiguation Protocol

**Author:** Dev Mehta  
**Date:** 2026-11-16  
**Status:** COMPLETE — SUBMITTED FOR FORMAL CLOSURE REVIEW  
**Work Package:** Gate 4D — Early-Epoch Physics & SN_CC / SN_Ia Disambiguation  
**Deliverable ID:** D1 (`GATE_4D_EARLY_EPOCH_PHYSICS_PROTOCOL.md`)  

---

## 1. Scientific Objective & Hypotheses

The primary objective of Work Package 4D is to test the information content of astronomical light curves *before* peak brightness to determine whether early-epoch observations can physically distinguish Type Ia supernovae (thermonuclear white dwarf disruptions) from Core-Collapse supernovae (massive star gravitational collapse), and to establish the earliest horizon at which defensible discrimination is possible.

### Pre-Registered Hypotheses:
1. **Hypothesis H1 (Early Color Degeneracy):**  
   During the first 5–10 days post-explosion, both Type Ia and Type II supernovae exhibit hot, expanding photospheres with blue optical colors ($g - r \le 0.0\text{ mag}$), rendering color-based disambiguation ineffective without precise multi-band rise derivatives.
2. **Hypothesis H2 (Rise-Rate Divergence):**  
   Type Ia supernovae exhibit faster, more uniform rise rates ($dF/dt \propto t^2$ fireball phase, reaching peak in $\approx 15\text{--}19\text{ days}$) compared to Type IIP supernovae (which rise to a plateau or have diverse shock-breakout cooling tails).
3. **Hypothesis H3 (Horizon Threshold):**  
   Reliable discrimination requires observations past peak light ($N_{\text{obs}} \ge 20$) to sample the post-maximum decline rate ($\Delta m_{15}$) and secondary infrared/optical bump characteristic of iron-peak recombination in Type Ia.

---

## 2. Chronological Observation Policy & Leakage Prevention

To ensure strict scientific integrity and prevent data snooping:
1. **Fixed Chronological Prefixes:**  
   Observations for each transient are strictly sorted by MJD. Feature extraction is evaluated across four predefined prefix horizons:
   - **Horizon $N=5$:** First 5 valid observations.
   - **Horizon $N=10$:** First 10 valid observations.
   - **Horizon $N=20$:** First 20 valid observations.
   - **Horizon Full:** Complete available light curve.
2. **Absolute Zero-Future-Observation Invariant:**  
   Under no circumstances may any measurement, peak flux, temporal baseline, or upper limit with $\text{MJD} > \text{MJD}_{\text{horizon}}$ enter the feature extraction pipeline.

---

## 3. Physical Phase Reference & Uncertainty Bounds

In rolling survey streams, the exact explosion time $t_0$ is unknown. We track three distinct temporal references:
1. **First Observed Detection ($t_{\text{first}}$):** Epoch of first $5\sigma$ positive detection.
2. **Pre-Explosion Constraint ($t_{\text{last\_ul}}$):** Epoch of latest non-detection ($<5\sigma$) preceding $t_{\text{first}}$. The explosion epoch is bounded by $[t_{\text{last\_ul}}, t_{\text{first}}]$.
3. **Estimated Peak Epoch ($t_{\text{peak}}$):** Maximum flux epoch within the evaluated horizon.

Sensitivity to phase offset is evaluated over $\pm 2$ and $\pm 5$ days.

---

## 4. Evaluated Supernova Cohort

Evaluated strictly on the parent-transient partitioned Track 1 Supernova cohort:
- **Training Population:** $N = 1,453$ ($890$ Type Ia, $563$ Core-Collapse)
- **Validation Holdout:** $N = 322$ ($202$ Type Ia, $120$ Core-Collapse)
- **Sealed Test Set ($N=310$ SNe):** Strictly quarantined in accordance with Gate 3 invariants.
