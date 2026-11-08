# ASTRAEUS Literature and Novelty Audit
## Prior Work Landscape, Methodological Gaps & Differentiating Contributions

**Document Identifier:** `ASTRAEUS-AUDIT-001`  
**Classification:** Research Governance & Literature Landscape  
**Version:** 1.0.0  

---

## 1. Prior Work Landscape & Methodological Gaps

The application of machine learning and autonomous systems to astrophysics is an active, evolving domain. To ensure scientific rigor and avoid overstated claims, ASTRAEUS maps existing research against specific, unaddressed gaps:

| Research Domain | State-of-the-Art Precedents | Methodological Failure Modes & Limitations | ASTRAEUS Target Contribution |
| :--- | :--- | :--- | :--- |
| **Astronomical Multimodal Learning** | AstroCLIP (Parker et al. 2024), AION-1 (Polymathic AI 2024), ASTROMER (Donoso-Oliva et al. 2023). | Models assume aligned or static cross-modal pairs; collapse under missing modalities; lack survey-adversarial invariant heads. | Cross-attention fusion with missing-modality masks and survey-adversarial gradient reversal for instrument-invariant representations. |
| **Transient Anomaly Detection** | Fink Broker (Möller et al. 2021), ALeRCE (Sánchez-Sáez et al. 2021), BTSbot (Perley et al. 2020). | Rely on isolated outlier metrics (Isolation Forest, LOF) resulting in high false discovery rates from detector artifacts and cosmic rays. | Composite triage score integrating latent distance ($d_M$), conformal epistemic uncertainty ($u$), and forward physical residuals ($r_{\mathrm{phys}}$). |
| **Evidence-Grounded Scientific RAG** | Standard Vector RAG, general Graph RAG (Edge et al. 2024). | Collapse citations into flat embeddings; suffer from hallucinations; cannot trace claims to primary IVOA detector observations. | Epistemic Graph RAG enforcing IVOA Provenance Data Model v1.0 entity-activity links and zero-hallucinated bibcode gates. |
| **Physics-Informed Surrogates** | FNO (Li et al. 2020), DeepONet (Lu et al. 2021), CAMELS multifield datasets (Villaescusa-Navarro et al.). | Standard operators overfit to specific subgrid recipes; violate conservation laws outside training regimes. | Physics-Informed Neural Operators (PINO) with hard conservation constraints (mass, momentum) delivering $10^4\times$ speedup over hydro solvers. |
| **Differentiable Cosmology** | pmwd (Li et al. 2022), DISCO-DJ (Hahn et al. 2023), jax-cosmo (Campagne et al. 2023). | Restricted to dark-matter particle-mesh dynamics; memory bottlenecks in backward passes across millions of particles. | Discrete adjoint state method using time-reversible symplectic leapfrog integration to eliminate intermediate trajectory caching. |
| **Autonomous Telescope Scheduling** | Greedy Dispatch, MILP, ROARS (2025), Ariel Space Telescope RL Scheduler (2026). | Classical methods fail under dynamic stochastic weather; unconstrained RL suffers from reward exploitation and mechanical violation. | Constrained Proximal Policy Optimization (CPPO) optimizing Expected Information Gain (EIG) under Lagrangian mechanical limits. |
| **Multi-Agent Scientific Reasoning** | AutoCog, AI Scientist (Sakana AI 2024), REGROUND (2025). | Cascading confirmation bias; agents agree on hallucinated premises; unconstrained natural-language tool use. | Epistemically stratified, isolated agent sandboxes with adversarial skeptic falsification protocols before knowledge integration. |

---

## 2. Six Primary Novelty Hypotheses

ASTRAEUS structures its academic contributions around six defensible research hypotheses:

1. **Unified Cross-Survey Astronomical Representation Learning:**
   - *Hypothesis:* Learning cross-modal representations while optimizing survey-adversarial and missing-modality objectives improves zero-shot transfer across unseen instruments and passbands compared to unimodal models.
2. **Evidence-Provenance Scientific Graph RAG:**
   - *Hypothesis:* Structuring astronomical knowledge graphs with IVOA provenance linkages ($\mathcal{E}_{\mathrm{obs}} \to \mathcal{E}_{\mathrm{inf}} \to \mathcal{E}_{\mathrm{hyp}}$) eliminates unsupported scientific claims compared to dense vector retrieval.
3. **Uncertainty-Aware Open-World Discovery Triage:**
   - *Hypothesis:* Coupling latent Mahalanobis density with conformal prediction and physical forward residuals reduces false discovery rates by $>40\%$ at fixed follow-up budgets.
4. **Differentiable Multiscale Astrophysical Digital Twins:**
   - *Hypothesis:* Integrating time-reversible discrete adjoint N-body solvers with PINO hydrodynamic surrogates achieves $>10^3\times$ acceleration while preserving field-level Bayesian parameter recovery fidelity.
5. **Adversarial Scientific Hypothesis Evaluation:**
   - *Hypothesis:* Isolating agent reasoning workspaces and requiring hypotheses to survive adversarial falsification by an independent skeptic agent eliminates hallucinated consensus in multi-agent workflows.
6. **Information-Theoretic Multi-Messenger Follow-Up Planning:**
   - *Hypothesis:* Constrained RL policies trained to maximize Expected Information Gain discover $>20\%$ more rapid optical counterparts to gravitational-wave events than greedy priority dispatchers.

---

## 3. Four Flagship Manuscripts

1. **Paper 01 (Perception):** *Towards Survey-Invariant Multimodal Representations for Astronomical Discovery* (Target: NeurIPS / Monthly Notices of the Royal Astronomical Society).
2. **Paper 02 (Reasoning):** *Evidence-Grounded Autonomous Scientific Investigation through Provenance-Aware Graph Reasoning* (Target: ICLR / The Astrophysical Journal).
3. **Paper 03 (Simulation):** *Uncertainty-Aware Neural Operators for Cosmological and Astrophysical Simulation Surrogates* (Target: Journal of Computational Physics / ICML).
4. **Paper 04 (Planning):** *Information-Theoretic Multi-Agent Reinforcement Learning for Autonomous Astronomical Experiment Planning* (Target: Nature Astronomy / Robotics & Autonomous Systems).
