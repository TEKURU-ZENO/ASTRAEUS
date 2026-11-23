# ASTRAEUS Phase 6 Workstream 1: Cosmological Simulation & Numerical Foundation Specification
## Mathematical Formulations, Symplectic Integrators, Particle-Mesh Solvers & Analytical Benchmarks

**Document Identifier:** `ASTRAEUS-P6-NUM-SPEC-001`  
**Classification:** Mathematical & Numerical Specification  
**Author:** Dev Mehta  
**Role:** Lead Investigator & Mission Director, Project ASTRAEUS  
**Baseline Git Checkpoint:** `0601beb`  
**Date:** November 23, 2026  
**Status:** APPROVED & ACTIVE  

---

## 1. Background Cosmology & Comoving Expansion

The background spacetime is described by the flat Friedmann-Lemaître-Robertson-Walker (FLRW) metric:
$$ds^2 = c^2 dt^2 - a^2(t) \delta_{ij} dx^i dx^j$$
where $a(t)$ is the dimensionless scale factor normalized to $a(t_0) = 1$ at the present epoch $z = 0$.

### 1.1 Friedmann Equation
In a universe with matter density parameter $\Omega_{m,0}$, radiation density $\Omega_{r,0}$, dark energy $\Omega_{\Lambda,0}$, and dark energy equation-of-state $w(a) = w_0 + w_a(1-a)$, the Hubble expansion parameter $H(a) = \dot{a}/a$ satisfies:
$$H^2(a) = H_0^2 E^2(a)$$
$$E^2(a) = \Omega_{m,0} a^{-3} + \Omega_{r,0} a^{-4} + \Omega_{k,0} a^{-2} + \Omega_{\Lambda,0} a^{-3(1 + w_0 + w_a)} \exp\left[ 3 w_a (a - 1) \right]$$

### 1.2 Conformal Time
Conformal time $\tau$ satisfies $d\tau = dt / a$, which converts to scale factor derivatives via:
$$\frac{d\tau}{da} = \frac{1}{a^2 H(a)} = \frac{1}{H_0 a^2 E(a)}$$

### 1.3 Linear Growth Factor
In linear perturbation theory, the growing mode $D_1(a)$ of the matter density contrast $\delta(\mathbf{x}, a) = D_1(a) \delta_0(\mathbf{x})$ satisfies the differential equation:
$$\frac{d^2 D_1}{da^2} + \left( \frac{3}{a} + \frac{1}{E(a)} \frac{dE}{da} \right) \frac{d D_1}{da} - \frac{3 \Omega_{m,0}}{2 a^5 E^2(a)} D_1 = 0$$
Normalized such that $D_1(a=1) = 1$. The logarithmic growth rate is $f(a) \equiv \frac{d \ln D_1}{d \ln a} \approx \Omega_m(a)^{0.55}$.

---

## 2. Hamiltonian Dynamics in Comoving Coordinates

For an $N$-body system of collisionless dark matter particles in an expanding universe:
- **Comoving Position:** $\mathbf{x} = \mathbf{r} / a$
- **Peculiar Velocity:** $\mathbf{u} = a \frac{d\mathbf{x}}{dt} = a \dot{\mathbf{x}}$
- **Canonical Momentum:** $\mathbf{p} = m a^2 \dot{\mathbf{x}} = m a \mathbf{u}$

The comoving Hamiltonian $\mathcal{H}$ is:
$$\mathcal{H}(\mathbf{x}, \mathbf{p}, t) = \sum_{i=1}^N \frac{\mathbf{p}_i^2}{2 m_i a^2(t)} + m_i \phi(\mathbf{x}_i, t)$$

Transforming the time variable from cosmic time $t$ to scale factor $a$:
$$\frac{d\mathbf{x}_i}{da} = \frac{\mathbf{p}_i}{m_i a^3 H(a)}$$
$$\frac{d\mathbf{p}_i}{da} = -\frac{m_i \nabla_{\mathbf{x}} \phi(\mathbf{x}_i)}{a H(a)}$$

---

## 3. Comoving Poisson Equation & 3D Particle-Mesh (PM) Solver

### 3.1 Gravitational Potential Equation
In comoving coordinates, the Poisson equation for the peculiar gravitational potential $\phi$ is:
$$\nabla_{\mathbf{x}}^2 \phi(\mathbf{x}, a) = \frac{3}{2} \Omega_{m,0} H_0^2 \frac{1}{a} \delta(\mathbf{x}, a)$$
where the dimensionless density contrast is:
$$\delta(\mathbf{x}) \equiv \frac{\rho(\mathbf{x}) - \bar{\rho}_m}{\bar{\rho}_m}$$

### 3.2 Cloud-in-Cell (CIC) Mass Assignment
For a particle at comoving coordinate $\mathbf{x}_p$ within a cubic box of side length $L$ discretized into $N_{\mathrm{grid}}^3$ cells of spacing $\Delta x = L / N_{\mathrm{grid}}$:
Let normalized coordinates be $\mathbf{X}_p = \mathbf{x}_p / \Delta x$. The grid index of the lower-left node is $i = \lfloor X_p \rfloor, j = \lfloor Y_p \rfloor, k = \lfloor Z_p \rfloor$, and fractional cell offsets are $d_x = X_p - i, d_y = Y_p - j, d_z = Z_p - k$.

The trilinear Cloud-in-Cell assignment kernel deposits particle mass $m_p$ to the 8 adjacent nodes:
$$w_{i, j, k} = (1 - d_x)(1 - d_y)(1 - d_z)$$
$$w_{i+1, j, k} = d_x (1 - d_y)(1 - d_z)$$
$$w_{i, j+1, k} = (1 - d_x) d_y (1 - d_z)$$
$$w_{i+1, j+1, k} = d_x d_y (1 - d_z)$$
$$w_{i, j, k+1} = (1 - d_x)(1 - d_y) d_z$$
$$w_{i+1, j, k+1} = d_x (1 - d_y) d_z$$
$$w_{i, j+1, k+1} = (1 - d_x) d_y d_z$$
$$w_{i+1, j+1, k+1} = d_x d_y d_z$$
with periodic boundary wrapping: $i_{\mathrm{per}} = i \pmod{N_{\mathrm{grid}}}$.

**Invariant:** $\sum_{l=1}^8 w_l = 1$, guaranteeing exact machine-precision mass conservation:
$$\sum_{i,j,k} \rho_{i,j,k} \Delta x^3 = \sum_{p=1}^N m_p = M_{\mathrm{total}}$$

### 3.3 3D Discrete Fourier Transform (FFT) Poisson Solver
Applying discrete Fourier transforms on the periodic grid, the 7-point finite-difference discrete Laplacian yields the exact discrete Green's function:
$$\hat{\phi}(\mathbf{k}) = G(\mathbf{k}) \cdot \hat{\delta}(\mathbf{k})$$
$$G(\mathbf{k}) = \begin{cases} 
-\frac{3}{2} \Omega_{m,0} H_0^2 \frac{1}{a} \left[ \frac{\Delta x^2}{4 \left( \sin^2\frac{k_x \Delta x}{2} + \sin^2\frac{k_y \Delta x}{2} + \sin^2\frac{k_z \Delta x}{2} \right)} \right] & \mathbf{k} \neq \mathbf{0} \\
0 & \mathbf{k} = \mathbf{0}
\end{cases}$$
The zero-frequency mode $\mathbf{k} = \mathbf{0}$ is strictly set to zero, enforcing the cosmological boundary condition that the universe has zero mean gravitational acceleration.

### 3.4 Gravitational Acceleration on Grid & Force Interpolation
Gravitational acceleration $\mathbf{g} = -\nabla_{\mathbf{x}} \phi$ is computed on grid nodes using 2nd-order central differences:
$$g_x(i, j, k) = -\frac{\phi(i+1, j, k) - \phi(i-1, j, k)}{2 \Delta x}$$
Acceleration at arbitrary particle positions $\mathbf{g}(\mathbf{x}_p)$ is interpolated using the exact same CIC trilinear weights $w_l$, preventing self-forces and ensuring momentum conservation.

---

## 4. Symplectic Time Integration (Kick-Drift-Kick Leapfrog)

To preserve phase space volume (Liouville's theorem) and bound long-term energy drift, ASTRAEUS implements the comoving symplectic Kick-Drift-Kick (KDK) leapfrog scheme:

Given state $(\mathbf{x}_n, \mathbf{p}_n)$ at scale factor $a_n$ and step size $\Delta a = a_{n+1} - a_n$:
1. **Half-Step Kick:**
   $$\mathbf{p}_{n+1/2} = \mathbf{p}_n + \mathbf{f}(\mathbf{x}_n, a_n) \cdot \frac{\Delta a}{2}, \quad \text{where } \mathbf{f}(\mathbf{x}, a) = -\frac{m \nabla_{\mathbf{x}} \phi(\mathbf{x})}{a H(a)}$$
2. **Full-Step Drift:**
   $$a_{n+1/2} = a_n + \frac{\Delta a}{2}$$
   $$\mathbf{x}_{n+1} = \left[ \mathbf{x}_n + \frac{\mathbf{p}_{n+1/2}}{m a_{n+1/2}^3 H(a_{n+1/2})} \cdot \Delta a \right] \pmod L$$
3. **Potential Re-evaluation:**
   Compute $\rho(\mathbf{x}_{n+1})$ via CIC, solve $\nabla^2 \phi = \frac{3}{2}\Omega_m H_0^2 a_{n+1}^{-1}\delta$ via 3D FFT, and differentiate to obtain $\mathbf{f}(\mathbf{x}_{n+1}, a_{n+1})$.
4. **Final Half-Step Kick:**
   $$\mathbf{p}_{n+1} = \mathbf{p}_{n+1/2} + \mathbf{f}(\mathbf{x}_{n+1}, a_{n+1}) \cdot \frac{\Delta a}{2}$$

---

## 5. Physical Invariants & Numerical Diagnostics

Every ASTRAEUS simulation run monitors four fundamental physical conservation laws:

1. **Mass Conservation:**
   $$\left| \frac{\sum_{i,j,k} \rho_{i,j,k} \Delta x^3 - M_{\mathrm{total}}}{M_{\mathrm{total}}} \right| \le 10^{-14}$$
2. **Total Comoving Momentum Conservation:**
   $$\left\| \sum_{p=1}^N \mathbf{p}_p(a) - \sum_{p=1}^N \mathbf{p}_p(a_{\mathrm{init}}) \right\| \le 10^{-10} \cdot \|\mathbf{P}_0\|$$
3. **Symplectic Phase-Space Time Reversibility:**
   Reversing time $\Delta a \to -\Delta a$ from $a_{\mathrm{final}}$ must recover $\mathbf{x}_{\mathrm{init}}$ to relative error $\le 10^{-8}$.
4. **Cosmic Energy Equation (Layzer-Irvine Equation):**
   In an expanding comoving frame, the total kinetic energy $T = \frac{1}{2}\sum m u^2$ and potential energy $U = \frac{1}{2} \int \rho \phi \, d^3x$ satisfy:
   $$\frac{d}{da}(T + U) + \frac{2T + U}{a} = 0 \implies \mathcal{E}_{\mathrm{LI}}(a) \equiv a(T + U) - \int_{a_{\mathrm{init}}}^a U(a') da' = \text{const}$$

---

## 6. Exact Analytical Benchmark: 1D Zel'dovich Pancake Collapse

To validate the code against an exact analytical solution prior to shell crossing, ASTRAEUS uses the 1D Zel'dovich pancake solution:

In an Einstein-de Sitter (EdS, $\Omega_m=1$) comoving background, initial comoving coordinates $q \in [0, L)$ are perturbed along the $x$-axis by a single Fourier wave $k = 2\pi / L$:
$$x(q, a) = q - \frac{D_1(a) - D_1(a_{\mathrm{init}})}{k} \sin(k q)$$
$$\dot{x}(q, a) = -\frac{\dot{D}_1(a)}{k} \sin(k q)$$

Caustic formation (infinite density / shell crossing) occurs exactly when the Jacobian vanishes:
$$\frac{\partial x}{\partial q} = 1 - [D_1(a_c) - D_1(a_{\mathrm{init}})] \cos(k q) = 0 \implies D_1(a_c) = D_1(a_{\mathrm{init}}) + 1$$

For $a < a_c$, the exact analytical position and velocity of every particle are known analytically, enabling machine-precision verification of the Particle-Mesh force and integration solver.

---

**Certified by:**  
**Dev Mehta**  
Lead Investigator & Mission Director, Project ASTRAEUS  
November 23, 2026
