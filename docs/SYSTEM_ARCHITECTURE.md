# ASTRAEUS System Architecture
## Modular Subsystems, Microservice Boundaries & Data Invariant Infrastructure

**Document Identifier:** `ASTRAEUS-ARCH-001`  
**Classification:** Systems Engineering & Software Architecture  
**Version:** 1.0.0  

---

## 1. Monorepo Structural Blueprint

ASTRAEUS decouples scientific algorithms into isolated Python packages while enforcing centralized version control across schemas, benchmarks, and model registries:

```
ASTRAEUS/
├── docs/                             # Programmatic architecture, research agendas, protocols
├── packages/
│   ├── astraeus-core/                 # Epistemic schemas, IVOA PROV-DM, validation invariants
│   ├── astraeus-data/                 # Multi-survey ingestion, Kafka brokers, cross-match engine
│   ├── astraeus-models/               # ASTROMER, RF, multimodal cross-attention, conformal heads
│   ├── astraeus-agents/               # 4-agent Phase 1 runtime (Orchestrator, Data, Lit, Skeptic)
│   ├── astraeus-knowledge/            # Vector index (Qdrant), Graph schemas (Neo4j), ADS clients
│   └── astraeus-bench/                # Leakage validators, AstroTemporal, AstroAnomaly runners
├── services/
│   ├── ingestion-worker/              # Streaming alert consumer & Parquet partitioner
│   └── report-generator/              # Provenance-backed markdown/PDF scientific reporter
├── configs/
│   ├── base_workstation.yaml          # Local RTX 5050 hardware budget & batch profiles
│   └── data_splits_healpix.json       # Deterministic sky partitioning manifests
└── experiments/
    └── phase1_baselines/              # Baseline tracking, seed registries, MLflow configs
```

---

## 2. Microservice Layer & Inter-Service Invariant Gates

```
[Raw Sky Streams: ZTF/Fink, TESS, Gaia]
                  │
                  ▼
      [services/ingestion-worker]
                  │
                  ▼ (Epistemic Tier Gate: rejects untagged/unprovenanced data)
     [packages/astraeus-data]  ──► [Partitioned Parquet / PostgreSQL Metadata]
                  │
                  ▼ (Extracted Features & Light Curve Cadences)
    [packages/astraeus-models] ──► [MLflow Registry / Checkpoints]
                  │
                  ▼ (Composite Score: d_M + u + r_phys)
    [Candidate Anomaly Triage]
                  │
                  ▼
    [packages/astraeus-agents] (Orchestrator, Data Agent, Literature Agent, Skeptic Agent)
                  │
                  ▼ (Adversarial REGROUND Challenge: 100% false-prior rejection)
     [services/report-generator] ──► [Auditable Scientific Discovery Report]
```

### Invariant Rules
1. **The Invariant Provenance Rule:** No entry can enter the persistent database without an explicit IVOA `wasGeneratedBy` activity hash and primary instrument identifier.
2. **The Epistemic Isolation Rule:** Hypotheses ($\mathcal{E}_{\mathrm{hyp}}$) and simulated values ($\mathcal{E}_{\mathrm{sim}}$) cannot modify observational data tables ($\mathcal{E}_{\mathrm{obs}}$).
3. **The Conformal Uncertainty Gate:** All statistical predictions ($\mathcal{E}_{\mathrm{inf}}$) must output finite-sample calibrated prediction intervals $[q_{\alpha/2}, q_{1-\alpha/2}]$ conditioned on local latent density $\sigma_k(z)$.

---

## 3. Four-Agent Adversarial Runtime for Phase 1

To prevent hallucinations and premature consensus, Phase 1 establishes a 4-agent LangGraph workflow:

1. **Chief Scientific Orchestrator:**
   - Ingests high-priority anomaly triggers ($s(x) > \tau_{\mathrm{threshold}}$).
   - Coordinates task execution graphs, sets token and compute resource quotas.
2. **Observation Data Agent:**
   - Interfaces with `astraeus-data`, retrieves raw multi-epoch photometry, computes zero points, checks image flags.
3. **Scientific Literature Agent:**
   - Queries NASA ADS and arXiv via `astraeus-knowledge`, retrieves published decay curves, spectroscopic templates, and host galaxy properties.
4. **Skeptic Agent (REGROUND Protocol):**
   - Systematically challenges the discovery: checks for Solar System asteroid trails, satellite passes, diffraction spikes, bad pixels, and catalog cross-match mismatches.
   - Requires empirical evidence to eliminate the null hypothesis (instrumental artifact / known recurrent variable).

---

## 4. Hardware Optimization Architecture (RTX 5050 8 GB VRAM)

```
Total VRAM: 8,192 MB
├── Reserved for PyTorch Model Weights & Gradients: 1,800 MB (BF16 / FP16 mixed precision)
├── Activation Memory (Micro-batch 16, seq_len 256): 1,200 MB (FlashAttention / SDPA enabled)
├── Vector & Anomaly Scratchpad (Mahalanobis / CQR): 1,000 MB
├── CUDA Context & Driver Overhead:                  1,000 MB
└── Safety Headroom (Prevents OOM during inference): 3,192 MB
```
