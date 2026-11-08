"""
ASTRAEUS Hardware Profiler & VRAM Benchmarking Module.
Profiles peak VRAM allocation, tensor memory, and throughput on NVIDIA RTX 5050 (8 GB)
under mixed precision (BF16/FP16) before freezing model architectures.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import time


class VramBenchmarkResult(BaseModel):
    batch_size: int
    seq_length: int
    precision: str
    allocated_mb: float
    reserved_mb: float
    throughput_samples_per_sec: float
    fits_in_8gb_envelope: bool


class HardwareBenchmarkRunner:
    """
    Simulates transformer sequence encoding workloads and measures physical hardware metrics.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)

    def benchmark_workload(
        self,
        batch_size: int = 16,
        seq_length: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        precision: str = "bf16"
    ) -> VramBenchmarkResult:
        """
        Executes forward + backward pass on a representative Transformer encoder block
        and measures peak GPU VRAM allocation.
        """
        dtype = torch.bfloat16 if (precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
            torch.float16 if (precision == "fp16" and torch.cuda.is_available()) else torch.float32
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dtype=dtype
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(batch_size, seq_length, d_model, device=self.device, dtype=dtype)

        # Warmup
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Timed pass
        start = time.perf_counter()
        n_iters = 5
        for _ in range(n_iters):
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        elapsed = time.perf_counter() - start

        throughput = float((batch_size * n_iters) / max(elapsed, 1e-6))

        if self.device.type == "cuda":
            allocated_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            reserved_mb = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
        else:
            # CPU estimation based on tensor bytes
            allocated_mb = float((x.element_size() * x.nelement() * num_layers * 4) / (1024 * 1024))
            reserved_mb = allocated_mb * 1.5

        # 8 GB envelope leaves 3 GB safety headroom -> max allowed is 5000 MB
        fits = reserved_mb < 5000.0

        return VramBenchmarkResult(
            batch_size=batch_size,
            seq_length=seq_length,
            precision=precision,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            throughput_samples_per_sec=throughput,
            fits_in_8gb_envelope=fits,
        )
