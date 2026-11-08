# ASTRAEUS
## Autonomous Computational Universe & Scientific Intelligence Ecosystem

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

ASTRAEUS is an autonomous computational universe and scientific intelligence ecosystem designed to bridge observational astronomy, theoretical astrophysics, computational physics, and artificial intelligence.

The central ambition is to build an ecosystem of interoperable scientific models, agents, simulators, and data fabrics that rigorously distinguishes between:
1. **Directly Observed ($\mathcal{E}_{\mathrm{obs}}$)**
2. **Statistically Inferred ($\mathcal{E}_{\mathrm{inf}}$)**
3. **Physically Predicted ($\mathcal{E}_{\mathrm{pred}}$)**
4. **Numerically Simulated ($\mathcal{E}_{\mathrm{sim}}$)**
5. **Hypothesized ($\mathcal{E}_{\mathrm{hyp}}$)**

---

## Repository Architecture

```
ASTRAEUS/
├── docs/                 # Programme specifications, research agendas, and protocols
├── packages/
│   ├── astraeus-core/    # Epistemic schemas, IVOA PROV-DM, validation invariants
│   ├── astraeus-data/    # Ingestion pipelines (ZTF, TESS, Gaia DR3)
│   ├── astraeus-models/  # Baseline encoders (ASTROMER, RF) & conformal prediction
│   ├── astraeus-agents/  # Multi-agent adversarial runtime (LangGraph)
│   ├── astraeus-knowledge/ # Epistemic Graph RAG (Neo4j & Qdrant)
│   └── astraeus-bench/   # Benchmark suites & zero-leakage evaluation
├── services/             # Independent worker services (ingestion, reporting)
├── configs/              # Execution profiles (RTX 5050 8GB VRAM budgets)
└── tests/                # Unit, integration, and epistemic leakage tests
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/astraeus-universe/astraeus.git
cd ASTRAEUS

# Install in development mode
pip install -e packages/astraeus-core

# Run test suite
pytest
```

---

## Documentation

- [Master Specification](docs/ASTRAEUS_MASTER_SPEC.md)
- [Scientific Research Agenda](docs/SCIENTIFIC_RESEARCH_AGENDA.md)
- [Literature and Novelty Audit](docs/LITERATURE_AND_NOVELTY_AUDIT.md)
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Experimental Protocol](docs/EXPERIMENTAL_PROTOCOL.md)
- [Roadmap and Milestones](docs/ROADMAP_AND_MILESTONES.md)
