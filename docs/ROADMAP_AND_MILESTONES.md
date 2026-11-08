# ASTRAEUS Roadmap & Milestone Verification Matrix
## Phased Multi-Year Development Programme & Gate Acceptance Criteria

**Document Identifier:** `ASTRAEUS-ROADMAP-001`  
**Classification:** Strategic Planning & Milestone Verification  
**Version:** 1.0.0  

---

## 1. 15-Month Master Programme Timeline

```
Months 1-2    Months 3-4     Months 5-6     Months 7-8     Months 9-10    Months 11-12   Months 13-14   Month 15
┌───────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ Phase 1:  │ │ Phase 2:   │ │ Phase 3:   │ │ Phase 4:   │ │ Phase 5:   │ │ Phase 6:   │ │ Phase 7:   │ │ Phase 8:   │
│ Scientific│ │ Multimodal │ │ Knowledge  │ │ Multi-Agent│ │ Differenti-│ │ Experiment │ │ Full Closed│ │ Open-Source│
│ Data      │ │ Latent     │ │ Fabric &   │ │ Adversarial│ │ able Cosmic│ │ Intelligence│ │ Loop       │ │ Research   │
│ Foundation│ │ Foundation │ │ Graph RAG  │ │ Reasoning  │ │ Surrogates │ │ & RL Sched │ │ Integration│ │ Ecosystem  │
└───────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

---

## 2. Milestone Acceptance Gates

| Phase / Milestone | Focus Area | Core Deliverables | Formal Acceptance Gate |
| :--- | :--- | :--- | :--- |
| **Phase 1 (M1-2)** | **Scientific Data Foundations** | Epistemic schemas, IVOA PROV-DM, Astroquery/MAST pipelines, baseline classifiers, composite anomaly triage. | Ingestion of $\ge 5 \times 10^5$ light curves with spatial HEALPix ($N_{\mathrm{side}}=32$) zero-leakage guarantee; Day 90 discovery report. |
| **Phase 2 (M3-4)** | **Multimodal Foundations** | Cross-modal contrastive encoders, masked autoencoders, missing-modality masks, conformal quantile heads. | Cross-survey zero-shot evaluation exceeds unimodal baselines by $\ge 15\%$ on property estimation ($R^2$ / CRPS). |
| **Phase 3 (M5-6)** | **Knowledge Fabric** | Neo4j property graph, Qdrant vector index, NASA ADS literature ingestion, citation verification engine. | $\ge 90\%$ accuracy on expert-curated citation verification; 0% hallucinated bibcodes. |
| **Phase 4 (M7-8)** | **Multi-Agent Reasoning** | LangGraph orchestration engine, 24 specialized agents, isolated sandboxes, REGROUND debate protocol. | REGROUND verification rejects 100% of injected false premises and instrumental artifacts in validation tests. |
| **Phase 5 (M9-10)** | **Computational Universe** | JAX `pmwd` integration, PINO hydrodynamic surrogates trained on CAMELS and IllustrisTNG. | Neural surrogate achieves $>10^3\times$ speedup over full hydrodynamics with $<3\%$ relative field error. |
| **Phase 6 (M11-12)** | **Experimental Intelligence** | Constrained PPO telescope scheduler, Bayesian optimal experimental design (BED) follow-up planner. | RL scheduler delivers $\ge 20\%$ higher scientific Expected Information Gain over greedy dispatch under weather noise. |
| **Phase 7 (M13-14)** | **Closed-Loop Integration** | End-to-end integration: alert ingestion $\to$ triage $\to$ debate $\to$ simulation $\to$ follow-up planning. | Automated end-to-end processing of simulated Rubin/GW alert bursts without manual pipeline intervention. |
| **Phase 8 (M15)** | **Community Release** | Open-source release under Apache-2.0, technical documentation, ASTRAEUS-Bench public leaderboard. | Reproducible installation runbooks, open-weight model weights, complete scientific benchmark suite. |

---

## 3. Phase 1 Sprint Milestones (Days 1–90)

- **Sprint 1 (Days 1–15):** Setup monorepo, author 6 master specifications, implement `astraeus-core` epistemic data structures and IVOA bindings. Pass unit test suite.
- **Sprint 2 (Days 16–30):** Implement data ingestion pipelines (`astraeus-data`) for ZTF and Gaia DR3. Partition data into spatial HEALPix and temporal cutoffs.
- **Sprint 3 (Days 31–45):** Train baseline temporal models (Lomb-Scargle + Random Forest, ASTROMER 1D Transformer). Log runs to MLflow.
- **Sprint 4 (Days 46–60):** Implement composite anomaly scoring ($d_M + u + r_{\mathrm{phys}}$) and conformal calibration. Validate on confirmed rare transients.
- **Sprint 5 (Days 61–75):** Run cross-survey zero-shot transfer tests on Pan-STARRS and DECam. Audit conformal prediction coverage.
- **Sprint 6 (Days 76–90):** Deploy 4-agent runtime (Orchestrator, Data Agent, Literature Agent, Skeptic Agent). Execute the Day 90 Reproducible Discovery Report demonstration.
