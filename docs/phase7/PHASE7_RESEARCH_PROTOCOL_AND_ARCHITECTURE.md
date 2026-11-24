# ASTRAEUS Phase 7: Closed-Loop Autonomous Scientific Intelligence
## Master Architecture, Threat Model & WS7.1 Ingestion Specification

**Document Identifier:** `ASTRAEUS-P7-ARCH-001`  
**Classification:** Strategic Architecture, Epistemic Threat Model & Engineering Protocol  
**Version:** 1.0.0 (Phase 7 Baseline Authorization)  
**Author:** Dev Mehta (Lead Investigator & Mission Director, Project ASTRAEUS)  
**Parent Checkpoint:** `5c8435c` (`PHASE6_1_INDEPENDENT_VALIDATION_AND_REPRODUCIBILITY`)  
**Active Working Branch:** `phase7-closed-loop-intelligence`  
**Date:** November 24, 2026  
**Status:** **ACTIVE & AUTHORIZED**

---

## 1. Executive Mandate & Strategic Scope

Phase 7 (*Closed-Loop Autonomous Scientific Intelligence*) marks the transition of Project ASTRAEUS from isolated mathematical engines and prototype agents into an **integrated, autonomous scientific discovery loop**.

The mission of Phase 7 is not merely to build LLM agents that execute tool calls, but to construct an auditable scientific system capable of:
1. Ingesting live or simulated multi-messenger astronomical events.
2. Performing automated triage and anomaly detection.
3. Conducting rigorous, multi-agent scientific investigations with competing hypotheses.
4. Orchestrating computational experiments while strictly respecting engine capability limits.
5. Optimizing follow-up observing requests via Information Gain.
6. Closing the loop by updating posteriors from new observations, preserving failed hypotheses, and producing bit-level reproducible scientific reports.

### 1.1 Non-Negotiable Epistemic Guardrails
As established in the Phase 6.1 Closure Decision:
> **Core Epistemic Invariant:**  
> A computational model does not become a scientific instrument merely because it is called by an autonomous agent. Phase 7 agents are strictly forbidden from retroactively promoting experimental prototypes into validated scientific instruments, or asserting confirmed astrophysical discoveries based on unvalidated surrogate predictions.

---

## 2. Epistemic Capability Registry & Operation Modes

Every computational engine integrated into Phase 7 must declare its capability tier under `ScientificCapabilityRegistry`:

| Engine Identifier | Architecture / Method | Formal Capability Tier | Operational Constraints in Phase 7 |
| :--- | :--- | :--- | :--- |
| `engines.hydrodynamics.pino3d` | 3D Fourier Neural Operator | `EXPERIMENTAL_PROTOTYPE` | IllustrisTNG held-out $L_2$ error is $0.9999$. Predictions carry mandatory experimental qualifiers; output cannot be cited as validated physical gas properties. |
| `engines.cosmology.differentiable` | Independent Model A $\ne$ Model B Parameter Recovery | `EXPERIMENTAL_PROTOTYPE` | Evaluated on $N=5$ mocks with Cramér-Rao lower bounds. Authorizes sensitivity exploration and proposal generation; cannot assert calibrated observational survey constraints. |
| `engines.experimental_design.ppo_scheduler` | Constrained Actor-Critic PPO Scheduler | `EXPERIMENTAL_PROTOTYPE` | Outperformed by analytical Greedy EIG oracle ($122.3$ vs $55.1 - 60.8$ EIG). Retained as research challenger; cannot replace analytical baselines as primary queue controller. |
| `engines.experimental_design.greedy_eig` | Analytical Expected Information Gain Scheduler | `VALIDATED_INSTRUMENT` | Primary reference scheduler for target follow-up prioritization and information-gain optimization. |
| `engines.cosmology.pm_solver` | KDK Symplectic Particle-Mesh Gravity Solver | `NUMERICAL_BENCHMARK_ONLY` | Verified numerical integrator; restricted to physical trajectory integration under specified potentials. |

---

## 3. Epistemic Threat Model & Systemic Failure Modes

Autonomous multi-agent scientific systems are vulnerable to cognitive and procedural failure modes that human research teams mitigate through institutional peer review. Phase 7 implements automated technical defenses:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PHASE 7 THREAT MODEL                                 │
├──────────────────────────┬─────────────────────────────┬─────────────────────────┤
│ Threat Vector            │ Potential Failure Impact    │ Technical Defense       │
├──────────────────────────┼─────────────────────────────┼─────────────────────────┤
│ 1. Hallucinated /        │ Agent cites an experimental │ ScientificCapability-   │
│    Unearned Epistemic    │ prototype or proxy model    │ Registry blocks any     │
│    Promotion             │ as confirmed physical truth.│ unearned promotion with │
│                          │                             │ EpistemicViolationError.│
├──────────────────────────┼─────────────────────────────┼─────────────────────────┤
│ 2. Gate 3 Observational  │ Autonomous search ingests   │ QuarantineFirewall      │
│    Quarantine Leakage    │ 1,808 quarantined HEALPix   │ dynamically blocks      │
│                          │ test pixels or candidates.  │ coordinates & catalog   │
│                          │                             │ file reads at runtime.  │
├──────────────────────────┼─────────────────────────────┼─────────────────────────┤
│ 3. Multi-Agent Sycophancy│ Personas converge on a      │ REGROUND debate mandates│
│    & Premature Consensus │ plausible but incorrect     │ a dedicated Red-Team /  │
│                          │ hypothesis without debate.  │ Skeptic with veto power.│
├──────────────────────────┼─────────────────────────────┼─────────────────────────┤
│ 4. Irreproducible        │ Non-deterministic LLM tool  │ InvestigationProvenance-│
│    Investigation         │ paths prevent independent   │ Record hashes all seeds,│
│    Trajectories          │ audit or replay.            │ inputs, and tool calls. │
└──────────────────────────┴─────────────────────────────┴─────────────────────────┘
```

---

## 4. Workstream Breakdown (WS7.1 – WS7.6)

Phase 7 is organized into six integrated engineering workstreams:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 7 CLOSED-LOOP DISCOVERY PIPELINE                        │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ WS7.1 — Scientific Event Ingestion: ZTF, Rubin, GW Notice Schemas & Firewalls │
  └──────────────────────────────────────┬────────────────────────────────────────┘
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ WS7.2 — Intelligent Triage: Cross-Matching, Host Context & Anomaly Scoring    │
  └──────────────────────────────────────┬────────────────────────────────────────┘
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ WS7.3 — Multi-Agent Investigation: REGROUND Multi-Hypothesis Debate Protocol  │
  └──────────────────────────────────────┬────────────────────────────────────────┘
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ WS7.4 — Computational Engine Orchestration: Registered Sandboxed Simulations  │
  └──────────────────────────────────────┬────────────────────────────────────────┘
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ WS7.5 — Experimental Design: Greedy EIG Oracle vs PPO Follow-Up Planning      │
  └──────────────────────────────────────┬────────────────────────────────────────┘
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ WS7.6 — Closed-Loop Verification: Posterior Update, Falsification & Prov-DM   │
  └───────────────────────────────────────────────────────────────────────────────┘
```

1. **WS7.1 — Scientific Event Ingestion & Normalization:**
   Standardizes streaming and batch astronomical notices across optical surveys (ZTF, Rubin/LSST diaSource/diaObject) and multi-messenger facilities (LIGO/Virgo/KAGRA GCN notices). Guarantees raw packet immutability, schema validation, and quarantine boundary protection.
2. **WS7.2 — Intelligent Triage and Prioritization:**
   Extracts light-curve features, associates host galaxies, evaluates transient rise/fade rates, detects rare astrophysical anomalies, and prioritizes events for investigation.
3. **WS7.3 — Multi-Agent Scientific Investigation (REGROUND):**
   Deploys 5 specialized agents:
   - *Astrophysicist:* Physical transient modeling (SNe, Kilonovae, TDEs).
   - *Cosmologist:* Large-scale structure, Hubble tension, and lensing context.
   - *Observational Astronomer:* Photometric calibration, extinction, and instrument systematics.
   - *Statistical Auditor:* Likelihood verification, model priors, and degrees of freedom.
   - *Skeptic / Red Team:* Adversarial falsification, premise challenging, and anomaly verification.
4. **WS7.4 — Computational Investigation Engine:**
   Connects agents to registered computational solvers under explicit capability boundaries.
5. **WS7.5 — Experimental Design & Follow-up Planning:**
   Computes Expected Information Gain across candidate telescope targets and filters.
6. **WS7.6 — Closed-Loop Verification & Reporting:**
   Ingests simulated or live follow-up photometry/spectroscopy, updates Bayesian hypothesis probabilities, preserves falsified hypotheses, and exports bit-level reproducible IVOA Prov-DM v1.0 reports.

---

## 5. First Phase 7 Acceptance Scenario: Autonomous Transient Investigation

The integration acceptance scenario tests the entire closed-loop discovery pipeline:
1. **Event Arrival:** Ingestion of a simulated optical transient (e.g. rapid blue transient / kilonova candidate).
2. **Validation & Firewall:** Schema validation and verification that the celestial coordinates do not lie in the Gate 3 quarantine footprint.
3. **Hypothesis Formulation:** Generation of at least 3 competing astrophysical hypotheses ($H_1$: Kilonova, $H_2$: Core-Collapse SN IIb, $H_3$: Active Galactic Nucleus flare).
4. **Evidence Retrieval:** Ingestion of contextual archival data (host redshift, extinction $E(B-V)$, past photometric history).
5. **Computational Simulation:** Invocation of registered engines with explicit experimental disclaimers recorded in provenance.
6. **Follow-Up Planning:** Greedy EIG recommendation for next-night multiband follow-up observation.
7. **Closed-Loop Update:** Ingestion of a simulated follow-up observation and Bayesian posterior revision.
8. **Red-Team Challenge & Report:** Skeptic challenges leading hypothesis; system compiles signed, reproducible investigation manifest.

---

## 6. WS7.1 Technical Specification

WS7.1 implements the foundational ingestion layer:
- **Rubin Alert Packet Schema:** LSST alert containing `diaObject`, `diaSource`, `diaForcedSource`, and 6-band ($u, g, r, i, z, y$) calibrated flux/magnitude.
- **Gravitational Wave Notice Schema:** LVK GCN notice containing event ID, false alarm rate (FAR), distance estimate, skymap URL, and source probabilities ($P_{\mathrm{BNS}}, P_{\mathrm{NSBH}}, P_{\mathrm{BBH}}, P_{\mathrm{Terrestrial}}$).
- **ZTF Alert Packet Schema:** Legacy and ongoing ZTF alert compatibility.
- **Unified Event Container:** `UnifiedScientificEvent` with coordinates, timestamps, raw packet hash, and quarantine clearance certificate.
- **Ingestion Engine:** `ScientificEventIngestionEngine` with idempotent caching and dead-letter quarantine routing.

---

**Certified by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 24, 2026
