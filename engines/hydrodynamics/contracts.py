"""
ASTRAEUS Hydrodynamics Engine: Contracts and Schemas (Track B).
Defines standardized data contracts, field representations, normalization pipelines,
and epistemic provenance for CAMELS and IllustrisTNG cosmological hydrodynamic suites.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

from astraeus_core.epistemic import EpistemicTier, EpistemicMetadata, ClaimType
from astraeus_core.provenance import ProvEntity, create_provenance_record


class HydroChannel(str, Enum):
    """Hydrodynamic field channels."""
    DENSITY = "density"                # Comoving gas density (M_sun / kpc^3 or scaled)
    VELOCITY_X = "velocity_x"          # Comoving gas velocity x-component (km/s)
    VELOCITY_Y = "velocity_y"          # Comoving gas velocity y-component (km/s)
    VELOCITY_Z = "velocity_z"          # Comoving gas velocity z-component (km/s)
    TEMPERATURE = "temperature"        # Gas temperature (Kelvin)
    NEUTRAL_HYDROGEN = "neutral_hi"    # Neutral hydrogen fraction n_HI / n_H
    MAGNETIC_B = "magnetic_b"          # Comoving magnetic field magnitude (microGauss)


class SimulationSuite(str, Enum):
    """Astrophysical simulation suite origin."""
    CAMELS = "CAMELS"
    ILLUSTRISTNG = "IllustrisTNG"
    SYNTHETIC_PROXY = "SyntheticProxy"


@dataclass
class HydroFieldMetadata:
    """Metadata specification for 3D hydrodynamic field cubes."""
    suite: SimulationSuite
    box_size_mpc_h: float
    grid_resolution: int
    redshift: float = 0.0
    omega_m: float = 0.30
    sigma_8: float = 0.80
    channels: List[HydroChannel] = field(default_factory=lambda: [
        HydroChannel.DENSITY,
        HydroChannel.VELOCITY_X,
        HydroChannel.VELOCITY_Y,
        HydroChannel.VELOCITY_Z,
        HydroChannel.TEMPERATURE,
    ])
    epistemic_tier: EpistemicTier = EpistemicTier.SIMULATED
    provenance_id: str = "astraeus:hydro:cube_001"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_epistemic_metadata(self) -> EpistemicMetadata:
        """Converts to astraeus-core epistemic contract."""
        return EpistemicMetadata(
            tier=self.epistemic_tier,
            claim_type=ClaimType.THEORETICAL_MODEL,
            verification_gate_passed=True,
            assumptions=[
                "Ideal gas equation of state gamma=5/3",
                f"Periodic boundary conditions in {self.box_size_mpc_h} Mpc/h box",
                "Sub-grid stellar and AGN feedback parameterized according to suite physics",
            ],
            validity_envelope={
                "box_size_mpc_h": self.box_size_mpc_h,
                "grid_resolution": self.grid_resolution,
                "redshift": self.redshift,
            },
        )

    def to_prov_entity(self, activity_id: str = "act:hydro_surrogate_training") -> ProvEntity:
        """Converts to IVOA Prov-DM v1.0 entity."""
        return create_provenance_record(
            entity_id=self.provenance_id,
            entity_type="hydrodynamic_field_cube",
            activity_id=activity_id,
            agent_id="agent:astraeus_researcher_dev_mehta",
            extra_metadata={
                "suite": self.suite.value,
                "box_size_mpc_h": self.box_size_mpc_h,
                "resolution": self.grid_resolution,
                "omega_m": self.omega_m,
                "sigma_8": self.sigma_8,
            },
        )


@dataclass
class HydroFieldCube:
    """
    Standardized container for multi-channel 3D cosmological hydrodynamic field grids.
    Shape of data: (num_channels, N, N, N).
    """
    data: np.ndarray  # Shape: (C, N, N, N)
    metadata: HydroFieldMetadata
    norm_stats: Optional[Dict[str, Tuple[float, float]]] = None  # channel -> (mean, std)

    def __post_init__(self):
        c, n1, n2, n3 = self.data.shape
        if n1 != n2 or n2 != n3:
            raise ValueError(f"HydroFieldCube must be cubic, got spatial dimensions ({n1}, {n2}, {n3})")
        if len(self.metadata.channels) != c:
            raise ValueError(
                f"Number of channels in metadata ({len(self.metadata.channels)}) "
                f"does not match tensor channels ({c})"
            )
        if self.metadata.grid_resolution != n1:
            raise ValueError(
                f"Metadata grid resolution {self.metadata.grid_resolution} != array shape {n1}"
            )

    @property
    def resolution(self) -> int:
        return self.metadata.grid_resolution

    @property
    def num_channels(self) -> int:
        return len(self.metadata.channels)

    def get_channel(self, channel: HydroChannel) -> np.ndarray:
        """Retrieves 3D array for specific channel."""
        idx = self.metadata.channels.index(channel)
        return self.data[idx]

    def normalize(self) -> Tuple[HydroFieldCube, Dict[str, Tuple[float, float]]]:
        """
        Normalizes fields for neural operator consumption:
        Log-transform for density, temperature, and magnetic fields;
        Standard z-score normalization across spatial grid.
        """
        norm_data = np.zeros_like(self.data, dtype=np.float32)
        stats: Dict[str, Tuple[float, float]] = {}

        for idx, ch in enumerate(self.metadata.channels):
            arr = self.data[idx]
            if ch in (HydroChannel.DENSITY, HydroChannel.TEMPERATURE, HydroChannel.MAGNETIC_B):
                # Positive definite: log10 transform
                arr_proc = np.log10(np.maximum(arr, 1e-10))
            else:
                arr_proc = arr

            m = float(np.mean(arr_proc))
            s = float(np.std(arr_proc)) + 1e-8
            stats[ch.value] = (m, s)
            norm_data[idx] = (arr_proc - m) / s

        cube_norm = HydroFieldCube(
            data=norm_data,
            metadata=self.metadata,
            norm_stats=stats,
        )
        return cube_norm, stats

    def denormalize(self, stats: Optional[Dict[str, Tuple[float, float]]] = None) -> HydroFieldCube:
        """Inverts normalization transform to restore physical units."""
        use_stats = stats or self.norm_stats
        if use_stats is None:
            raise ValueError("No normalization stats provided for denormalization.")

        raw_data = np.zeros_like(self.data, dtype=np.float32)
        for idx, ch in enumerate(self.metadata.channels):
            m, s = use_stats[ch.value]
            arr = self.data[idx] * s + m
            if ch in (HydroChannel.DENSITY, HydroChannel.TEMPERATURE, HydroChannel.MAGNETIC_B):
                raw_data[idx] = 10.0 ** arr
            else:
                raw_data[idx] = arr

        return HydroFieldCube(
            data=raw_data,
            metadata=self.metadata,
            norm_stats=None,
        )

    def to_torch(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Converts data array to PyTorch batch tensor of shape (1, C, N, N, N)."""
        tensor = torch.from_numpy(self.data).float().unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


def generate_synthetic_hydro_cube(
    grid_size: int = 16,
    box_size_mpc_h: float = 25.0,
    seed: int = 42,
    suite: SimulationSuite = SimulationSuite.SYNTHETIC_PROXY,
) -> HydroFieldCube:
    """
    Generates a physically consistent synthetic hydrodynamic cube mimicking CAMELS/IllustrisTNG
    with log-normal density, turbulent velocity, polytropic thermal structure, and magnetic field.
    """
    rng = np.random.RandomState(seed)
    N = grid_size
    L = box_size_mpc_h
    dx = L / N

    # 1. Fourier mesh for turbulent fluctuations
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K_sq = KX ** 2 + KY ** 2 + KZ ** 2
    K_mag = np.sqrt(K_sq)
    K_mag[0, 0, 0] = 1.0

    # Gaussian random field with P(k) ~ k^-2
    phi_k = (rng.normal(0, 1, (N, N, N)) + 1j * rng.normal(0, 1, (N, N, N))) / np.sqrt(2.0)
    phi_k[0, 0, 0] = 0.0
    field_k = phi_k * (K_mag ** (-1.0))
    field_real = np.fft.ifftn(field_k).real
    field_norm = (field_real - field_real.mean()) / (field_real.std() + 1e-10)

    # Gas density: log-normal with mean 1.0 (mean comoving density = 1.0 in code units)
    log_rho = 0.8 * field_norm
    density = np.exp(log_rho)
    density = density / density.mean()  # normalize mean density to 1.0

    # 2. Turbulent velocities: Kolmogorov spectrum P_v(k) ~ k^(-5/3) with divergence-free projection
    vels = []
    for _ in range(3):
        v_phase = (rng.normal(0, 1, (N, N, N)) + 1j * rng.normal(0, 1, (N, N, N))) / np.sqrt(2.0)
        v_phase[0, 0, 0] = 0.0
        v_k = v_phase * (K_mag ** (-5.0 / 6.0))
        vels.append(v_k)

    # Incompressible projection: v_sol = v - k(k.v)/k^2
    K_sq_safe = np.where(K_sq == 0.0, 1.0, K_sq)
    k_dot_v = (KX * vels[0] + KY * vels[1] + KZ * vels[2]) / K_sq_safe
    k_dot_v[0, 0, 0] = 0.0
    vx_k = vels[0] - KX * k_dot_v
    vy_k = vels[1] - KY * k_dot_v
    vz_k = vels[2] - KZ * k_dot_v

    # Scale to ~ 100 - 300 km/s
    vx = np.fft.ifftn(vx_k).real * 150.0
    vy = np.fft.ifftn(vy_k).real * 150.0
    vz = np.fft.ifftn(vz_k).real * 150.0

    # 3. Temperature: T ~ T0 * rho^(gamma-1) with gamma = 5/3, plus virialized halo heating
    T_base = 1.0e4 * (density ** (2.0 / 3.0))
    # Clump heating: where density > 2, add shock heating up to 1e7 K
    excess_density = np.maximum(density - 2.0, 0.0)
    halo_heating = np.where(density > 2.0, 5.0e6 * (excess_density / 2.0) ** 1.5, 0.0)
    temperature = np.clip(T_base + halo_heating, 1.0e3, 1.0e8)

    # 4. Neutral hydrogen: high in cold/dense gas, low in hot/ionized gas
    ion_factor = 1.0 / (1.0 + (temperature / 2.0e4) ** 2)
    n_hi = np.clip(density * ion_factor * 0.76, 1e-8, 1.0)

    # 5. Magnetic field: B ~ B0 * rho^(2/3) ~ 0.1 - 5.0 microGauss
    b_mag = np.clip(0.5 * (density ** (2.0 / 3.0)), 0.01, 20.0)

    channels = [
        HydroChannel.DENSITY,
        HydroChannel.VELOCITY_X,
        HydroChannel.VELOCITY_Y,
        HydroChannel.VELOCITY_Z,
        HydroChannel.TEMPERATURE,
        HydroChannel.NEUTRAL_HYDROGEN,
        HydroChannel.MAGNETIC_B,
    ]

    stacked_data = np.stack([density, vx, vy, vz, temperature, n_hi, b_mag], axis=0).astype(np.float32)

    meta = HydroFieldMetadata(
        suite=suite,
        box_size_mpc_h=box_size_mpc_h,
        grid_resolution=grid_size,
        redshift=0.0,
        omega_m=0.30,
        sigma_8=0.80,
        channels=channels,
        epistemic_tier=EpistemicTier.SIMULATED,
        provenance_id=f"astraeus:hydro:{suite.value.lower()}_{grid_size}cube",
    )

    return HydroFieldCube(data=stacked_data, metadata=meta)
