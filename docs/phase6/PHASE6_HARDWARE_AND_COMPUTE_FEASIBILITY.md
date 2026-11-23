# ASTRAEUS Phase 6 Hardware Budget, Memory Scaling & Compute Feasibility Charter
## Operational Compute Tiers, VRAM Ceilings & Anti-Overclaiming Rules

**Document Identifier:** `ASTRAEUS-P6-COMPUTE-001`  
**Classification:** Engineering Specification & Resource Feasibility Audit  
**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Baseline Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB GDDR6 VRAM, 23 GB Host RAM)  
**Date:** November 23, 2026  
**Status:** APPROVED & BINDING  

---

## 1. Local Hardware Profiling & Memory Envelopes

Project ASTRAEUS development and automated CI testing run on the primary workstation:
- **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (Dedicated VRAM: 8,192 MB GDDR6)
- **Host Memory:** ~23.5 GB DDR5 System RAM
- **Compute Capability:** Ada Lovelace Architecture (CUDA 12.x / Compute 8.9)
- **Host Storage:** NVMe PCIe 4.0 SSD

### Memory Footprint Analysis for 3D Particle-Mesh (PM) Cosmological Simulations

In comoving Particle-Mesh simulations, memory consumption is dominated by 3D real grids, 3D complex Fourier transforms, and particle coordinate arrays:

| Mesh Grid Resolution ($N_{\mathrm{grid}}^3$) | Particle Count ($N_p = N_{\mathrm{grid}}^3$) | Real Density Grid (FP32) | Complex FFT Buffer (Complex64) | Particle State ($\mathbf{x}, \mathbf{v}$) | JAX AD Tape (30 steps, no chkpt) | Local Feasibility (RTX 5050 8GB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$32^3$ (32,768)** | 32,768 | 0.13 MB | 0.26 MB | 0.79 MB | ~25 MB | **Instantaneous (< 10 ms)** |
| **$64^3$ (262,144)** | 262,144 | 1.05 MB | 2.10 MB | 6.29 MB | ~200 MB | **Feasible (< 100 ms/step)** |
| **$128^3$ (2,097,152)** | 2,097,152 | 8.39 MB | 16.78 MB | 50.33 MB | ~1.6 GB | **Feasible (Full Local Benchmark)** |
| **$256^3$ (16,777,216)** | 16,777,216 | 67.11 MB | 134.22 MB | 402.65 MB | ~12.8 GB | **Exceeds VRAM without Remat** |
| **$512^3$ (134,217,728)** | 134,217,728 | 536.87 MB | 1,073.74 MB | 3,221.23 MB | > 80 GB | **Infeasible on Local Laptop** |

---

## 2. Three-Tier Operational Compute Model

To guarantee scientific rigor while maintaining realistic engineering constraints, Phase 6 formally establishes three decoupled compute tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: LOCAL WORKSTATION                       │
│    NVIDIA RTX 5050 (8 GB VRAM) · 23 GB RAM · Local Fast Prototyping   │
│  - WS1 Numerical reference solver & analytical benchmark tests        │
│  - $32^3$ to $128^3$ Particle-Mesh forward simulations                 │
│  - Unit tests, CI test collection, and mathematical validation         │
│  - RL telescope scheduling environment prototyping (Gymnasium)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Model Scaling & Deep Training
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 2: CLOUD GPU NODES                         │
│       NVIDIA A100 (40 GB / 80 GB VRAM) · Scalable Cloud Compute       │
│  - WS2 Differentiable gradient benchmarks with reverse-mode AD         │
│  - WS3 PINO Fourier Neural Operator training on CAMELS sub-volumes     │
│  - Constrained PPO telescope scheduler training ($10^6$ steps)         │
│  - Parameter posterior recovery MCMC/HMC chains                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Large-Scale Production
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                TIER 3: INSTITUTIONAL HPC & SUPERCOMPUTING              │
│       Multi-Node Distributed MPI / Slurm Clusters (Perlmutter/TNG)     │
│  - Full IllustrisTNG hydrodynamic data extraction & multi-scale fields │
│  - Large-box cosmological simulations ($N_{\mathrm{grid}} \ge 512^3$)  │
│  - Final cross-facility observational scheduler production runs        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Strict Scientific Anti-Overclaiming Rules

1. **Explicit Hardware Disclosure:**  
   Every reported throughput number, speedup factor, latency measurement, and training epoch duration must explicitly state the hardware platform (e.g., *"Evaluated on NVIDIA RTX 5050 Laptop GPU (8 GB)"* or *"Trained on Cloud A100-SXM4-80GB"*).
2. **Fair Baseline Comparison:**  
   When calculating speedup metrics ($\ge 1,000\times$), the neural surrogate and the reference numerical solver must be evaluated on identical problem dimensions (same box size $L$, same grid resolution $N^3$, and same target fields) and must account for input preprocessing and tensor transfer overheads.
3. **No Unsubstantiated Scale Claims:**  
   Under no circumstances shall publications or milestone reports claim that full survey-scale hydrodynamic simulations (e.g. TNG300-1) were produced locally on the workstation. All local experiments are strictly designated as numerical reference benchmarks or feasibility prototypes.

---

**Certified by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 23, 2026
