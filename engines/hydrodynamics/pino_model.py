"""
ASTRAEUS Hydrodynamics Engine: 3D Fourier Neural Operator & PINO Architecture (Track B).
Implements spectral convolution layers in 3D using real FFTs, Fourier neural operator blocks,
and physics-preserving neural surrogates for multi-scale cosmological hydrodynamics.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution layer using Real-to-Complex Fast Fourier Transforms.
    Computes frequency domain multiplication against parameterized complex weights
    restricted to the lowest Fourier modes: modes1 x modes2 x modes3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Modes along dimension X
        self.modes2 = modes2  # Modes along dimension Y
        self.modes3 = modes3  # Modes along dimension Z (RFFT half spectrum)

        scale = 1.0 / (in_channels * out_channels)
        # 4 corners in Fourier space corresponding to combinations of positive and negative frequencies
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex tensor multiplication: (batch, in_channel, x, y, z) * (in_channel, out_channel, x, y, z)."""
        return torch.einsum("bixyz,ioxyz->boxyz", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: (batch, in_channels, size_x, size_y, size_z)
        """
        batchsize, in_dim, size_x, size_y, size_z = x.shape

        # 3D Real FFT: last dimension size is size_z // 2 + 1
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Output Fourier representation initialized to zeros
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            size_x,
            size_y,
            size_z // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, size_x // 2)
        m2 = min(self.modes2, size_y // 2)
        m3 = min(self.modes3, size_z // 2 + 1)

        # Corner 1: (+kx, +ky, +kz)
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3]
        )
        # Corner 2: (-kx, +ky, +kz)
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, :m1, :m2, :m3]
        )
        # Corner 3: (+kx, -ky, +kz)
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights3[:, :, :m1, :m2, :m3]
        )
        # Corner 4: (-kx, -ky, +kz)
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4[:, :, :m1, :m2, :m3]
        )

        # 3D Inverse Real FFT
        x_out = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=[-3, -2, -1])
        return x_out


class FourierBlock3D(nn.Module):
    """
    Building block of Fourier Neural Operator combining spectral convolution,
    local channel mixing bypass, and non-linear activation.
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        modes3: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.linear_shortcut = nn.Conv3d(width, width, kernel_size=1)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral_conv(x) + self.linear_shortcut(x)
        return self.activation(out)


class PINO3DModel(nn.Module):
    """
    Physics-Informed Neural Operator (PINO) in 3D for multi-channel cosmological hydrodynamics.
    Maps input fields (e.g. gravitational potential, dark matter density) to full hydrodynamic state
    (gas density, velocity vector, temperature, magnetic field).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 5,
        width: int = 32,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        num_layers: int = 3,
        include_grid: bool = True,
        positive_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.include_grid = include_grid
        self.positive_channels = positive_channels  # None by default for standardized targets

        # Uplifting layer: input (+ 3 spatial coordinates if include_grid) -> width
        input_dim = in_channels + (3 if include_grid else 0)
        self.lifting = nn.Conv3d(input_dim, width, kernel_size=1)

        # Fourier operator layers
        self.blocks = nn.ModuleList([
            FourierBlock3D(width, modes1, modes2, modes3)
            for _ in range(num_layers)
        ])

        # Projection / decoding head
        self.proj1 = nn.Conv3d(width, width * 2, kernel_size=1)
        self.proj2 = nn.Conv3d(width * 2, out_channels, kernel_size=1)

    def get_grid(self, shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """Constructs normalized periodic coordinate grid in [0, 1)^3."""
        nx, ny, nz = shape
        gx = torch.arange(nx, device=device, dtype=torch.float32) / float(nx)
        gy = torch.arange(ny, device=device, dtype=torch.float32) / float(ny)
        gz = torch.arange(nz, device=device, dtype=torch.float32) / float(nz)
        grid_x, grid_y, grid_z = torch.meshgrid(gx, gy, gz, indexing="ij")
        grid = torch.stack([grid_x, grid_y, grid_z], dim=0)  # (3, nx, ny, nz)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, nx, ny, nz)
        Returns: (batch, out_channels, nx, ny, nz)
        """
        batchsize, _, nx, ny, nz = x.shape

        if self.include_grid:
            grid = self.get_grid((nx, ny, nz), x.device).unsqueeze(0).repeat(batchsize, 1, 1, 1, 1)
            x_in = torch.cat([x, grid], dim=1)
        else:
            x_in = x

        h = self.lifting(x_in)

        for block in self.blocks:
            h = block(h)

        h = F.gelu(self.proj1(h))
        out = self.proj2(h)

        # Enforce physical positivity on requested channels (e.g. density, temperature)
        if self.positive_channels:
            out_list = []
            for ch_idx in range(self.out_channels):
                ch_slice = out[:, ch_idx : ch_idx + 1]
                if ch_idx in self.positive_channels:
                    # Positive definite: Softplus + epsilon
                    ch_slice = F.softplus(ch_slice) + 1e-4
                out_list.append(ch_slice)
            out = torch.cat(out_list, dim=1)

        return out

    def get_num_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
