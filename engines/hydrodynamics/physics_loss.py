"""
ASTRAEUS Hydrodynamics Engine: Physics-Informed Conservation Loss (Track B).
Calculates exact periodic PDE conservation residuals for compressible cosmological hydrodynamics:
- Gas Mass Continuity: div(rho * v) = 0
- Momentum Euler Equation: div(rho * v (x) v) + grad(P) + rho * grad(Phi) = 0
- Solenoidal Magnetic Field: div(B) = 0
- Ideal Gas Equation of State: P = (gamma - 1) * rho * u = n * k_B * T / (mu * m_p)

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def periodic_central_diff(tensor: torch.Tensor, dim: int, dx: float) -> torch.Tensor:
    """Computes second-order central difference with periodic boundary conditions."""
    return (torch.roll(tensor, -1, dims=dim) - torch.roll(tensor, 1, dims=dim)) / (2.0 * dx)


def periodic_divergence_3d(vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor, dx: float) -> torch.Tensor:
    """Computes divergence div(v) = d(vx)/dx + d(vy)/dy + d(vz)/dz on periodic 3D grid."""
    # Spatial dimensions in tensor (B, 1, Nx, Ny, Nz) or (B, Nx, Ny, Nz) are -3, -2, -1
    dvx_dx = periodic_central_diff(vx, dim=-3, dx=dx)
    dvy_dy = periodic_central_diff(vy, dim=-2, dx=dx)
    dvz_dz = periodic_central_diff(vz, dim=-1, dx=dx)
    return dvx_dx + dvy_dy + dvz_dz


def periodic_poisson_solver_3d(density: torch.Tensor, dx: float, G: float = 1.0) -> torch.Tensor:
    """
    Solves periodic Poisson equation grad^2(Phi) = 4 * pi * G * (rho - <rho>) using 3D FFT.
    Returns gravitational potential Phi.
    density shape: (B, 1, Nx, Ny, Nz)
    """
    b, c, nx, ny, nz = density.shape
    device = density.device

    # Overdensity delta = (rho - mean_rho)
    mean_rho = torch.mean(density, dim=(-3, -2, -1), keepdim=True)
    delta_rho = density - mean_rho

    # Discrete Fourier frequencies
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=dx, device=device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=dx, device=device)
    kz = 2.0 * math.pi * torch.fft.rfftfreq(nz, d=dx, device=device)

    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    K_sq = KX ** 2 + KY ** 2 + KZ ** 2
    K_sq[0, 0, 0] = 1.0  # Avoid zero division at k=0 mode

    rho_k = torch.fft.rfftn(delta_rho, dim=[-3, -2, -1])
    # Poisson: -k^2 Phi_k = 4 * pi * G * rho_k  ==> Phi_k = -4 * pi * G * rho_k / k^2
    phi_k = -4.0 * math.pi * G * rho_k / K_sq
    phi_k[:, :, 0, 0, 0] = 0.0  # Zero mean potential

    phi = torch.fft.irfftn(phi_k, s=(nx, ny, nz), dim=[-3, -2, -1])
    return phi


class PhysicsConservationLoss(nn.Module):
    """
    Evaluates conservative PDE residuals and combines them with data loss for PINO training.
    """

    def __init__(
        self,
        dx: float = 1.0,
        gamma: float = 5.0 / 3.0,
        lambda_data: float = 1.0,
        lambda_mass: float = 0.05,
        lambda_momentum: float = 0.02,
        lambda_solenoidal: float = 0.01,
        G_newton: float = 1.0,
    ):
        super().__init__()
        self.dx = float(dx)
        self.gamma = float(gamma)
        self.lambda_data = float(lambda_data)
        self.lambda_mass = float(lambda_mass)
        self.lambda_momentum = float(lambda_momentum)
        self.lambda_solenoidal = float(lambda_solenoidal)
        self.G_newton = float(G_newton)

    def compute_mass_residual(self, rho: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor, vz: torch.Tensor) -> torch.Tensor:
        """
        Calculates mass continuity residual: div(rho * v)
        """
        j_x = rho * vx
        j_y = rho * vy
        j_z = rho * vz
        res = periodic_divergence_3d(j_x, j_y, j_z, self.dx)
        return torch.mean(res ** 2)

    def compute_momentum_residual(
        self,
        rho: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        vz: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates momentum Euler residual:
        div(rho * v (x) v) + grad(P) + rho * grad(Phi)
        where P = (gamma - 1) * rho * T (in normalized code units).
        """
        dx = self.dx
        # Pressure: P = rho * T
        pressure = rho * temperature

        # Gravitational potential
        phi = periodic_poisson_solver_3d(rho, dx=dx, G=self.G_newton)
        grad_phi_x = periodic_central_diff(phi, dim=-3, dx=dx)
        grad_phi_y = periodic_central_diff(phi, dim=-2, dx=dx)
        grad_phi_z = periodic_central_diff(phi, dim=-1, dx=dx)

        # 1. Momentum flux divergence along X:
        # div(rho * vx * v) + dP/dx + rho * dPhi/dx
        fx_x = rho * vx * vx
        fx_y = rho * vx * vy
        fx_z = rho * vx * vz
        div_fx = periodic_divergence_3d(fx_x, fx_y, fx_z, dx)
        dp_dx = periodic_central_diff(pressure, dim=-3, dx=dx)
        res_x = div_fx + dp_dx + rho * grad_phi_x

        # 2. Momentum flux divergence along Y:
        fy_x = rho * vy * vx
        fy_y = rho * vy * vy
        fy_z = rho * vy * vz
        div_fy = periodic_divergence_3d(fy_x, fy_y, fy_z, dx)
        dp_dy = periodic_central_diff(pressure, dim=-2, dx=dx)
        res_y = div_fy + dp_dy + rho * grad_phi_y

        # 3. Momentum flux divergence along Z:
        fz_x = rho * vz * vx
        fz_y = rho * vz * vy
        fz_z = rho * vz * vz
        div_fz = periodic_divergence_3d(fz_x, fz_y, fz_z, dx)
        dp_dz = periodic_central_diff(pressure, dim=-1, dx=dx)
        res_z = div_fz + dp_dz + rho * grad_phi_z

        return torch.mean(res_x ** 2 + res_y ** 2 + res_z ** 2)

    def compute_solenoidal_residual(self, bx: torch.Tensor, by: torch.Tensor, bz: torch.Tensor) -> torch.Tensor:
        """Calculates magnetic solenoidality residual: div(B) = 0."""
        div_b = periodic_divergence_3d(bx, by, bz, self.dx)
        return torch.mean(div_b ** 2)

    def forward(
        self,
        predicted: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite PINO loss:
        predicted: (batch, 5, Nx, Ny, Nz) -> [rho, vx, vy, vz, T]
        """
        rho = predicted[:, 0:1]
        vx = predicted[:, 1:2]
        vy = predicted[:, 2:3]
        vz = predicted[:, 3:4]
        T = predicted[:, 4:5]

        # 1. Physical conservation residuals
        res_mass = self.compute_mass_residual(rho, vx, vy, vz)
        res_mom = self.compute_momentum_residual(rho, vx, vy, vz, T)

        # 2. Data loss (if target provided)
        if target is not None:
            loss_data = F.mse_loss(predicted, target)
        else:
            loss_data = torch.tensor(0.0, device=predicted.device)

        total_loss = (
            self.lambda_data * loss_data
            + self.lambda_mass * res_mass
            + self.lambda_momentum * res_mom
        )

        metrics = {
            "loss_total": float(total_loss.detach().cpu().item()),
            "loss_data": float(loss_data.detach().cpu().item()),
            "res_mass": float(res_mass.detach().cpu().item()),
            "res_momentum": float(res_mom.detach().cpu().item()),
        }

        return total_loss, metrics
