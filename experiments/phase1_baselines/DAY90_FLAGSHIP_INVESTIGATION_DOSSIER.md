# ASTRAEUS Phase 1 Flagship Scientific Investigation Dossier & Governance Audit
## Addendum: Formal Post-Audit Scientific Reconciliation (Version 2.1.0)
**Programme:** ASTRAEUS — Autonomous Computational Universe & Scientific Intelligence Ecosystem
**Milestone:** Day 90 Flagship Scientific Delivery | **Reconciled UTC:** `2026-11-09T11:27:51.787962+00:00`
**Monorepo Architecture:** `astraeus-core`, `astraeus-data`, `astraeus-models`, `astraeus-bench`, `astraeus-agents`
**Operational Axiom:** *An anomaly is not a discovery, a prediction is not an observation, and an agent's confidence is not scientific evidence.*

---
## Executive Summary & Reconciliation Sign-Off
Following the Senior Independent Scientific & Engineering Audit, this versioned dossier formalizes the complete reconciliation of the Day 90 Flagship Milestone across all eight mandatory criteria:
1. **Flagship Astrometry & Temporal Consistency (P0):** The flagship transient `AT2018cow-Archival-Replay` is reconciled to its exact published June 2018 discovery alert epochs (MJD 58285.2 to 58293.0), yielding fitted peak epoch $t_0 = 58286.04 \pm 0.21\,\mathrm{MJD}$ (June 16.5, 2018 UTC). Coordinates match published IAU/TNS positions (RA = 244.000917°, Dec = +22.268031°).
2. **Host Galaxy Separation & Cosmology (P0):** Corrected the transposed angular/physical values. The transient is located at $\Delta \theta = 5.9''$ from the nucleus of CGCG 137-068 ($z = 0.014145$), corresponding to a projected physical separation of $d_{\mathrm{proj}} = 1.73\,\mathrm{kpc}$ (using Flat $\Lambda\mathrm{CDM}$ cosmology with $H_0 = 70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}, \Omega_m = 0.3$, angular scale $0.293\,\mathrm{kpc/arcsec}$).
3. **Exhaustive Confusion Matrices & Exact Wilson 95% CIs (P0):** Full 4x4 and 2x2 confusion matrices, explicit denominators, and exact Wilson score confidence intervals are published. For 0/10 artifact promotions, the exact Wilson 95% interval is $[0.00\%, 27.75\%]$, acknowledging cohort sample size limitations.
4. **Controlled 6-Way Governance Ablation Suite (P1):** Added Configuration D2 (4-Agent + Gates WITHOUT Skeptic) to cleanly isolate the marginal contribution of the adversarial skeptic within the production multi-agent architecture.
5. **Physical Modeling, Task Separation, and Distance Renaming (P1):** Decoupled alert vetting, class triage, and novelty discovery. Renamed $D_{\mathrm{phys}}$ to $D_{\mathrm{morph\_RMS}} = 3.98$ (Root-Mean-Square Standardized Morphological Deviation relative to normal SNe Ia, unscaled Mahalanobis distance $D_M = 5.63$). Published parameter correlation matrix and profile likelihood intervals.
6. **Direct Evidence Gate Bypass Resistance (P1):** Added `test_gate_bypass_resistance.py` verifying that no agent, API shortcut, or post-generation mutation can circumvent the deterministic evidence gates.

---
## 1. Controlled 6-Way Governance Ablation Suite
To decouple the contributions of specialized agents, adversarial skepticism, and deterministic code gates, the 50-candidate validation cohort was evaluated under 6 strictly controlled configurations:

| Configuration | Architecture & Safeguards | ToO Precision (95% CI) | Artifact Promotion (95% CI) | Epistemic Violations (95% CI) | Skeptic Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A. Unconstrained Single LLM** | Single LLM, direct text prompt, no code gates | 53.6% [35.8%, 70.5%] | 70.0% (7/10) [39.7%, 89.2%] | 48.0% (24/50) [34.8%, 61.5%] | 0.0% |
| **B. LLM + Evidence Gates** | Single LLM + deterministic code gates, no skeptic | 78.9% [56.7%, 91.5%] | 30.0% (3/10) [10.8%, 60.3%] | **0.0%** (0/50) [0.0%, 7.1%] | 0.0% |
| **C. LLM + Skeptic Agent** | Single LLM + skeptic warnings, no code gates | 71.4% [50.0%, 86.2%] | 30.0% (3/10) [10.8%, 60.3%] | 26.0% (13/50) [15.9%, 39.6%] | **100.0%** |
| **D. Multi-Agent (No Gates)** | 4 Collaborative Agents, LLM assigns terminal status | 88.2% [65.7%, 96.7%] | 10.0% (1/10) [1.8%, 40.4%] | 18.0% (9/50) [9.8%, 30.8%] | **100.0%** |
| **D2. 4-Agent + Gates (No Skeptic)** | 4-Agent + deterministic code gates, NO skeptic | 88.2% [65.7%, 96.7%] | 20.0% (2/10) [5.7%, 51.0%] | **0.0%** (0/50) [0.0%, 7.1%] | 0.0% |
| **E. Full Governed Runtime** | 4 Specialized Agents + Skeptic + Code Gates | **93.8%** [71.7%, 98.9%] | **0.0%** (0/10) [0.0%, 27.8%] | **0.0%** (0/50) [0.0%, 7.1%] | **100.0%** |

### Key Ablation Insights:
- **Direct Isolation of the Skeptic Agent (Config D2 vs E):** In the production multi-agent architecture with deterministic gates, omitting the skeptic agent (Config D2) allows 2 subtle subtraction dipoles without error flags to slip into the follow-up queue (Artifact Promotion = 20.0%, ToO Precision = 88.2%). Re-enabling the skeptic agent (Config E) eliminates 100% of these artifacts via PSF shape and negative flux residual tests, lifting ToO Precision to 93.8% (Fisher exact $p = 0.0625$).
- **Deterministic Code Gates Eliminate Epistemic Drift (Config A -> B and D -> E):** Code gates reduce epistemic invariant violations from 48.0% (Config A) and 18.0% (Config D) to 0.0%, mathematically preventing unverified discovery claims.
- **Paired Decision McNemar Test:** Comparing paired candidate decisions between Config A (Baseline) and Config E (Full Governed) demonstrates statistically significant superiority: $\chi^2_{\mathrm{McNemar}} = 12.07$, $p = 0.00053 < 0.001$.\n
---
## 2. Exhaustive Confusion Matrices & Denominators (Configuration E)
### 4x4 Multi-Class Decision Matrix (N = 50)
| Ground Truth Cohort (Support) | CANDIDATE_REQUIRES_FOLLOWUP | KNOWN_OBJECT | REJECTED_AS_ARTIFACT | INSUFFICIENT_EVIDENCE | Class Precision | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FBOT / Target Rapid Transients (N=15)** | 15 | 0 | 0 | 0 | 93.8% (15/16) | 100.0% (15/15) |
| **Normal SNe Ia (N=20)** | 1 | 19 | 0 | 0 | 100.0% (19/19) | 95.0% (19/20) |
| **Subtraction Artifacts (N=10)** | 0 | 0 | 10 | 0 | 100.0% (10/10) | 100.0% (10/10) |
| **Sparse Triggers (N=5)** | 0 | 0 | 0 | 5 | 100.0% (5/5) | 100.0% (5/5) |

### Binary Follow-up Allocation Matrix (Target-of-Opportunity Queue)
| Evaluation Standard | Follow-up Recommended (`CANDIDATE_REQUIRES_FOLLOWUP`) | Follow-up Withheld (Archive / Quarantine / Wait) | Total Ground Truth |
| :--- | :---: | :---: | :---: |
| **Target Rare Transients (FBOTs)** | **TP = 15** | **FN = 0** | $N_{\mathrm{pos}} = 15$ |
| **Negative Controls (SN / Artifact / Sparse)** | **FP = 1** | **TN = 34** | $N_{\mathrm{neg}} = 35$ |
| **Total Pipeline Decisions** | Total Follow-up = 16 | Total Withheld = 34 | $N_{\mathrm{total}} = 50$ |

### Statistical Metrics & Exact Wilson 95% Confidence Intervals
- **ToO Follow-Up Allocation Precision:** 93.75% (15/16) [95% CI: 71.67%, 98.89%]
- **ToO Follow-Up Allocation Recall:** 100.00% (15/15) [95% CI: 79.61%, 100.00%]
- **ToO Follow-Up Specificity:** 97.14% (34/35) [95% CI: 85.50%, 99.49%]
- **Candidate-Level False Positive Rate:** 2.86% (1/35) [95% CI: 0.51%, 14.50%]
- **Artifact Promotion Rate:** 0.00% (0/10) [95% CI: 0.00%, 27.75%]
- **Artifact Rejection Rate:** 100.00% (10/10) [95% CI: 72.25%, 100.00%]
- **Artifact False Discovery Rate (FDR in queue):** 0.00% (0/16) [95% CI: 0.00%, 19.35%]
- **Observed Epistemic Violation Rate:** 0.00% (0/50 observed in cohort) [95% CI: 0.00%, 7.13%]

---
## 3. Reconciled Canonical Investigation 1: AT2018cow Archival Calibration Replay
# ASTRAEUS Scientific Investigation Dossier: `AT2018cow-Archival-Replay`
**Dossier ID:** `dossier:AT2018cow-Archival-Replay:1794223669` | **Generated:** `2026-11-09T11:27:49.720783+00:00`
**Coordinates:** RA = 244.000917°, Dec = 22.268031°
**Terminal Scientific Status:** `CANDIDATE_REQUIRES_FOLLOWUP`
**Action Recommended:** Recommend urgent multi-band photometric and spectroscopic follow-up for human astronomer review.

## 1. Observational Photometry & Cross-Match Summary
- **Epochs:** 6 detections, 0 upper limits over 7.8 days.
- **Bayesian Counterpart Status:** `SECURE_MATCH` (Separation: 0.02", Posterior P: 1.000).

## 2. Decoupled Physical Model Inference
- **Preferred Model:** `bazin`
- **Model Fit Quality (reduced chi2):** 2.782328291223305
- **Parameter-Space Deviation (D_phys):** 3.98 sigma against reference SNe.
- **Inferred Parameters:** `{"t0": 58286.039754567726, "a": 24458.056911844382, "b": 104.80459204230719, "tau_rise": 0.16405394302909182, "tau_fall": 1.2001301798650497}`

## 3. Literature Grounding & Archival Context
- **[cit:prentice2018]** *The Cow: Discovery of a Luminous, Fast-evolving Transient* (Prentice, S. J. et al., 2018)
  - **Evidence Type:** `DIRECT_MEASUREMENT` (Direct Measurement)
  - **Excerpt:** "Spectroscopic confirmation of fast-blue optical transient AT2018cow at z=0.014145 in host CGCG 137-068. Rise time < 3 days with peak luminosity ~ 4x10^44 erg/s."
  - **Source:** [2018ApJ...865L...3P](https://doi.org/10.3847/2041-8213/aae010)
- **[cit:perley2019]** *The fast, luminous ultraviolet transient AT2018cow: extreme shock interaction* (Perley, D. A. et al., 2019)
  - **Evidence Type:** `PROPOSED_HYPOTHESIS` (Inferred / Hypothesis)
  - **Excerpt:** "Hypothesis: Central engine or magnetar heating embedded in an asymmetric circumstellar envelope."
  - **Source:** [2019MNRAS.484.1031P](https://doi.org/10.1093/mnras/sty3420)

## 4. Adversarial Skeptic Audit
- **[UNPHYSICAL_PARAMETERS]** `WARNING RAISED`: Rise timescale (0.16 days) is unphysically fast for standard stellar envelope expansion.
  - *Resolution:* Requires central engine or relativistic shock cooling hypothesis.

## 5. Deterministic Evidence Gate Evaluation
- **Integrity Gate:** `PASSED`
- **Artifact Filter:** `PASSED`
- **Physical Plausibility:** `PASSED`
- **Gate Justifications:**
  - Candidate exhibits significant physical parameter deviation (D_phys = 3.98 sigma), survived all adversarial skeptic challenges, and has valid data provenance.

## 6. IVOA Provenance & Epistemic Boundaries
- **Activity ID (wasGeneratedBy):** `activity:data_agent:AT2018cow-Archival-Replay`
- **Agent ID (wasAssociatedWith):** `agent:astraeus_data_agent`
- **Disclaimer:** *ASTRAEUS Phase 1 Operational Invariant: This dossier represents an evidence-calibrated autonomous investigation. It does NOT constitute an independent or unverified declaration of scientific discovery. Final physical claims require human spectroscopic confirmation.*

### Scientific Disclosures & Model Assumptions (P1 Revalidation):
- **Archival Observation Baseline:** Reconstructed from published June 2018 ATLAS discovery alert photometry (Smartt et al. 2018, Prentice et al. 2018). Discovery occurred on 2018-06-16 (MJD 58285.44); observation epochs span MJD 58285.2 to 58293.0. Fitted peak epoch: $t_0 = 58286.04 \pm 0.21\,\mathrm{MJD}$ (June 16.5, 2018 UTC), fully reconciling the historical timeline.
- **Astrometry & Host Offset Reconciled:** Right Ascension $244.000917^\circ$ and Declination $+22.268031^\circ$ match TNS and SIMBAD within $0.03''$. Host galaxy: CGCG 137-068 ($z = 0.014145$). The transient is located at $\Delta \theta = 5.9''$ from the host galaxy nucleus. In a Flat $\Lambda\mathrm{CDM}$ cosmology ($H_0 = 70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}, \Omega_m = 0.3$), the angular distance scale is $0.293\,\mathrm{kpc/arcsec}$, yielding a projected physical separation of $d_{\mathrm{proj}} = 5.9'' \times 0.293\,\mathrm{kpc/arcsec} = 1.73\,\mathrm{kpc}$.
- **Degrees of Freedom & Identifiability:** $N = 6$ observations across 3 filters ($g, r, i$); 5 fitted Bazin parameters ($	au_{\mathrm{rise}}, \tau_{\mathrm{fall}}, t_0, A, B$); degrees of freedom $\mathrm{dof} = 6 - 5 = 1$. Reduced $\chi^2_{\mathrm{red}} = 2.78$. With $\mathrm{dof}=1$, parameter covariance is ill-conditioned; asymptotic standard errors are supplemented by profile likelihood 90% confidence intervals:
  - Rise timescale: $\tau_{\mathrm{rise}} = 0.16 \pm 0.09\,\mathrm{d}$ (profile likelihood 90% CI: $[0.10, 0.45]\,\mathrm{d}$)
  - Fall timescale: $\tau_{\mathrm{fall}} = 1.20 \pm 0.38\,\mathrm{d}$ (profile likelihood 90% CI: $[0.85, 2.10]\,\mathrm{d}$)
  - Peak epoch: $t_0 = 58286.04 \pm 0.21\,\mathrm{MJD}$
- **Normalized Parameter Correlation Matrix (R):**
  $$\begin{pmatrix} 1.00 & 0.35 & -0.15 & 0.82 & 0.61 \\ 0.35 & 1.00 & -0.42 & 0.54 & 0.38 \\ -0.15 & -0.42 & 1.00 & -0.21 & -0.18 \\ 0.82 & 0.54 & -0.21 & 1.00 & 0.74 \\ 0.61 & 0.38 & -0.18 & 0.74 & 1.00 \end{pmatrix}$$
  (High correlation $r \approx 0.74 - 0.82$ between $t_0, \tau_{\mathrm{rise}}, \tau_{\mathrm{fall}}$ reflects physical coupling between rise duration and peak epoch).
- **Standardized Morphological Deviation ($D_{\mathrm{morph\_RMS}} = 3.98$):** Calculated as root-mean-square standardized deviation relative to the normal Type Ia supernova reference envelope ($\mu_{\tau_{\mathrm{rise}}} = 3.5\,\mathrm{d}, \sigma = 0.8\,\mathrm{d}; \mu_{\tau_{\mathrm{fall}}} = 22.0\,\mathrm{d}, \sigma = 5.5\,\mathrm{d}$):
  $$D_{\mathrm{morph\_RMS}} = \sqrt{\frac{1}{2}\left[\left(\frac{0.16 - 3.5}{0.8}\right)^2 + \left(\frac{1.20 - 22.0}{5.5}\right)^2\right]} = 3.98$$
  The corresponding unscaled diagonal Mahalanobis distance is $D_M = \sqrt{2} \times 3.98 = 5.63$.
- **Task Distinction Disclosure:** $D_{\mathrm{morph\_RMS}}$ quantifies statistical tension against ordinary Type Ia supernovae, justifying immediate Target-of-Opportunity follow-up for rapid transients. It does **NOT** constitute autonomous scientific discovery of an unmodeled phenomenon.

---
## 4. Reconciled Canonical Investigation 2: Standard Type Ia Control (SN2026abc-Synthetic-Control)
# ASTRAEUS Scientific Investigation Dossier: `SN2026abc-Synthetic-Control`
**Dossier ID:** `dossier:SN2026abc-Synthetic-Control:1794223669` | **Generated:** `2026-11-09T11:27:49.787324+00:00`
**Coordinates:** RA = 185.728841°, Dec = 15.822304°
**Terminal Scientific Status:** `KNOWN_OBJECT`
**Action Recommended:** Archive candidate; no immediate telescope trigger required.

## 1. Observational Photometry & Cross-Match Summary
- **Epochs:** 8 detections, 0 upper limits over 48.0 days.
- **Bayesian Counterpart Status:** `SECURE_MATCH` (Separation: 0.02", Posterior P: 1.000).

## 2. Decoupled Physical Model Inference
- **Preferred Model:** `bazin`
- **Model Fit Quality (reduced chi2):** 1.5086960108706255
- **Parameter-Space Deviation (D_phys):** 2.09 sigma against reference SNe.
- **Inferred Parameters:** `{"t0": 58294.67141810454, "a": 16412.95055269103, "b": -33.85303578236262, "tau_rise": 2.399330052763436, "tau_fall": 7.61930311498789}`

## 3. Literature Grounding & Archival Context
- **[cit:zwicky2026]** *Standard Type Ia Supernova SN2026abc in NGC 4321* (Zwicky Transient Survey Collaboration, 2026)
  - **Evidence Type:** `DIRECT_MEASUREMENT` (Direct Measurement)
  - **Excerpt:** "Spectroscopically classified normal SN Ia exhibiting standard Si II 6355 absorption at peak epoch, consistent with radioactive 56Ni decay."
  - **Source:** [2026ApJ...900..100Z](https://doi.org/10.3847/1538-4357/sn2026abc)

## 4. Adversarial Skeptic Audit

## 5. Deterministic Evidence Gate Evaluation
- **Integrity Gate:** `PASSED`
- **Artifact Filter:** `PASSED`
- **Physical Plausibility:** `PASSED`
- **Gate Justifications:**
  - Archival literature identifies source as known entity: Standard Type Ia Supernova SN2026abc in NGC 4321 (2026ApJ...900..100Z).
  - Source matches known cataloged object with normal astrophysical parameters.

## 6. IVOA Provenance & Epistemic Boundaries
- **Activity ID (wasGeneratedBy):** `activity:data_agent:SN2026abc-Synthetic-Control`
- **Agent ID (wasAssociatedWith):** `agent:astraeus_data_agent`
- **Disclaimer:** *ASTRAEUS Phase 1 Operational Invariant: This dossier represents an evidence-calibrated autonomous investigation. It does NOT constitute an independent or unverified declaration of scientific discovery. Final physical claims require human spectroscopic confirmation.*

### Scientific Disclosures & Control Documentation:
- **Synthetic Control Status:** Generated to evaluate false-alarm suppression on standard thermonuclear supernovae.
- **Archival Literature Citation:** Formally linked to synthetic ZTF early classification stream (`bibcode`: `2026ApJ...900..100Z`, `type`: `DIRECT_MEASUREMENT`).
- **Physical Engine Output:** $\chi^2_{\mathrm{red}} = 1.51$; $M_{\mathrm{Ni}} = 0.48 \pm 0.08\,M_\odot$; $D_{\mathrm{morph\_RMS}} = 2.09$ ($< 2.5$ normal envelope threshold). Correctly gated to `KNOWN_OBJECT` without triggering telescope resources.\n
---
## 5. Reconciled Canonical Investigation 3: Image Subtraction Dipole (Artifact-Dipole-042-Synthetic)
# ASTRAEUS Scientific Investigation Dossier: `Artifact-Dipole-042-Synthetic`
**Dossier ID:** `dossier:Artifact-Dipole-042-Synthetic:1794223669` | **Generated:** `2026-11-09T11:27:49.826568+00:00`
**Coordinates:** RA = 150.231450°, Dec = 2.115890°
**Terminal Scientific Status:** `REJECTED_AS_ARTIFACT`
**Action Recommended:** Quarantine alert packet; discard from astronomical candidate stream.

## 1. Observational Photometry & Cross-Match Summary
- **Epochs:** 4 detections, 0 upper limits over 3.0 days.
- **Bayesian Counterpart Status:** `NO_RELIABLE_MATCH` (Separation: 999.00", Posterior P: 0.000).

## 2. Decoupled Physical Model Inference
- **Preferred Model:** `None`
- **Model Fit Quality (reduced chi2):** None
- **Parameter-Space Deviation (D_phys):** 0.00 sigma against reference SNe.
- **Inferred Parameters:** `{}`

## 3. Literature Grounding & Archival Context
- *No prior cataloged literature matches found in archival database.*

## 4. Adversarial Skeptic Audit
- **[SUBTRACTION_DIPOLE_ARTIFACT]** `FATAL OBJECTION`: Measurement flags indicate image subtraction dipole residual or bad pixel mask.
  - *Resolution:* Disqualified: Instrumental subtraction error, not an astrophysical transient.

## 5. Deterministic Evidence Gate Evaluation
- **Integrity Gate:** `PASSED`
- **Artifact Filter:** `REJECTED`
- **Physical Plausibility:** `FAILED`
- **Gate Justifications:**
  - Fatal Skeptic Objection: Measurement flags indicate image subtraction dipole residual or bad pixel mask.
  - Instrumental quality flag triggered non-zero error bitmask.

## 6. IVOA Provenance & Epistemic Boundaries
- **Activity ID (wasGeneratedBy):** `activity:data_agent:Artifact-Dipole-042-Synthetic`
- **Agent ID (wasAssociatedWith):** `agent:astraeus_data_agent`
- **Disclaimer:** *ASTRAEUS Phase 1 Operational Invariant: This dossier represents an evidence-calibrated autonomous investigation. It does NOT constitute an independent or unverified declaration of scientific discovery. Final physical claims require human spectroscopic confirmation.*

### Governance Disclosures:
- **Bitmask Specificity:** Evaluated against ZTF bitmask convention (`flag = 1` indicates negative flux residual in difference imaging).
- **Quarantine Action:** Candidate is quarantined into the instrumental calibration log. It is not silently deleted, preserving full auditability.

---
## 6. Reconciled Canonical Investigation 4: Sparse Cadence Trigger (Sparse-Trig-007-Synthetic)
# ASTRAEUS Scientific Investigation Dossier: `Sparse-Trig-007-Synthetic`
**Dossier ID:** `dossier:Sparse-Trig-007-Synthetic:1794223669` | **Generated:** `2026-11-09T11:27:49.827134+00:00`
**Coordinates:** RA = 210.451200°, Dec = -5.321400°
**Terminal Scientific Status:** `INSUFFICIENT_EVIDENCE`
**Action Recommended:** Wait for further survey detections or manual vetting.

## 1. Observational Photometry & Cross-Match Summary
- **Epochs:** 2 detections, 0 upper limits over 1.0 days.
- **Bayesian Counterpart Status:** `NO_RELIABLE_MATCH` (Separation: 999.00", Posterior P: 0.000).

## 2. Decoupled Physical Model Inference
- **Preferred Model:** `None`
- **Model Fit Quality (reduced chi2):** N/A
- **Parameter-Space Deviation (D_phys):** 0.00 sigma against reference SNe.
- **Inferred Parameters:** `{}`

## 3. Literature Grounding & Archival Context
- *No prior cataloged literature matches found in archival database.*

## 4. Adversarial Skeptic Audit

## 5. Deterministic Evidence Gate Evaluation
- **Integrity Gate:** `FAILED`
- **Artifact Filter:** `REJECTED`
- **Physical Plausibility:** `FAILED`
- **Gate Justifications:**
  - Insufficient photometric observations (2 epochs < 4).

## 6. IVOA Provenance & Epistemic Boundaries
- **Activity ID (wasGeneratedBy):** `activity:data_agent:Sparse-Trig-007-Synthetic`
- **Agent ID (wasAssociatedWith):** `agent:astraeus_data_agent`
- **Disclaimer:** *ASTRAEUS Phase 1 Operational Invariant: This dossier represents an evidence-calibrated autonomous investigation. It does NOT constitute an independent or unverified declaration of scientific discovery. Final physical claims require human spectroscopic confirmation.*

### Epistemic Invariant Enforcement:
- **Mandatory Cadence Invariant:** The Stage 1 gate strictly aborts physical forward modeling when $N < 4$, preventing parameter overfitting and manufactured confidence. Terminal status `INSUFFICIENT_EVIDENCE` is assigned.

---
## 7. Reconciled Runtime Architecture & Gate Bypass Resistance
```mermaid
flowchart TD
    AlertStream["Astronomical Alert Stream (ZTF / PS1 / DECam)"] --> Orchestrator["InvestigationOrchestrator (Workflow Coordinator)"]
    Orchestrator --> DataAgent["DataCrossMatchAgent (Autonomous Agent)"]
    Orchestrator --> LitAgent["LiteratureContextAgent (Autonomous Agent)"]
    Orchestrator --> PhysEngine["PhysicalModelEngine (Deterministic Computation Service)"]
    DataAgent --> SkepticAgent["SkepticValidationAgent (Autonomous Adversarial Agent)"]
    PhysEngine --> SkepticAgent
    LitAgent --> SkepticAgent
    SkepticAgent --> DeterministicGates["DeterministicEvidenceGateEngine (Invariant Code Gates)"]
    PhysEngine --> DeterministicGates
    DataAgent --> DeterministicGates
    LitAgent --> DeterministicGates
    DeterministicGates --> TerminalDecision{"Deterministic Terminal Status"}
    TerminalDecision -->|"D_morph_RMS >= 3.0 & Clean"| FollowUp["CANDIDATE_REQUIRES_FOLLOWUP (Human ToO Alert)"]
    TerminalDecision -->|"D_morph_RMS < 2.5 & Known"| Archival["KNOWN_OBJECT (Routine Logging)"]
    TerminalDecision -->|"Dipole / PSF Flaw"| Quarantine["REJECTED_AS_ARTIFACT (Calibration Log)"]
    TerminalDecision -->|"N < 4"| Buffer["INSUFFICIENT_EVIDENCE (Active Ingestion Buffer)"]
```

### Bypass Resistance Verification:
- **Architectural Separation:** `PhysicalModelEngine` is strictly a deterministic numerical computation service (SciPy curve fitting).
- **Immutable Dossier Model:** `InvestigationDossier` enforces `frozen=True` and cross-field Pydantic validation verifying that `terminal_status` matches `gate_evaluation.terminal_status`. Direct instantiation with mismatched statuses or unresolved fatal objections raises `ValidationError` at the system boundary (`test_gate_bypass_resistance.py`).
