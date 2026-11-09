"""
ASTRAEUS 1D Astronomical Transformer (ASTROMER-compatible) Module.
Learns self-supervised representations from irregularly sampled multi-band photometric time series.
Optimized for memory efficiency on RTX 5050 (8 GB) hardware envelope.
"""

import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


class ContinuousTimePositionalEncoding(nn.Module):
    """
    Encodes continuous observation timestamps (MJD) into d_model dimensional vectors
    using sinusoidal frequency projections.
    Handles arbitrary irregular sampling intervals: Delta_t = t_j - t_{j-1}.
    """

    def __init__(self, d_model: int, max_period_days: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        # Frequencies spanning from hours (1/24 day) to years (1000 days)
        frequencies = torch.exp(
            torch.linspace(math.log(1.0 / max_period_days), math.log(24.0), half_dim)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, times_mjd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            times_mjd: [batch_size, seq_len] tensor of observation epochs.
        Returns:
            [batch_size, seq_len, d_model] continuous positional embeddings.
        """
        # Outer product: [batch_size, seq_len, 1] * [half_dim] -> [batch_size, seq_len, half_dim]
        angles = times_mjd.unsqueeze(-1) * self.frequencies
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        pe = torch.cat([sin_emb, cos_emb], dim=-1)
        if pe.shape[-1] < self.d_model:
            # Pad if odd d_model
            pad = torch.zeros(*pe.shape[:-1], self.d_model - pe.shape[-1], device=pe.device)
            pe = torch.cat([pe, pad], dim=-1)
        return pe


class AstromerEncoder(nn.Module):
    """
    1D Multi-Head Attention Astronomical Transformer Backbone.
    Processes (times, magnitudes, errors, passband_tokens) sequences.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_passbands: int = 4,  # e.g., 0=pad, 1=g, 2=r, 3=i
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.time_encoder = ContinuousTimePositionalEncoding(d_model=d_model)
        self.mag_proj = nn.Linear(2, d_model // 2)  # magnitude + error
        self.band_emb = nn.Embedding(num_passbands, d_model // 2)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction head for self-supervised masked pretraining
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # Classification / representation pooling
        self.classifier_head: Optional[nn.Module] = None

    def build_classifier_head(self, num_classes: int) -> None:
        """Attaches a task-specific classification head."""
        head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, num_classes),
        )
        try:
            device = next(self.parameters()).device
            head = head.to(device)
        except StopIteration:
            pass
        self.classifier_head = head

    def forward_representation(
        self,
        times_mjd: torch.Tensor,
        magnitudes: torch.Tensor,
        errors: torch.Tensor,
        band_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Encodes light curves into sequence latents H \in R^{B x T x D}.
        """
        # Time embedding
        time_emb = self.time_encoder(times_mjd)

        # Value embedding: combine (magnitude, error) with passband ID
        mag_err = torch.stack([magnitudes, errors], dim=-1)
        val_emb = self.mag_proj(mag_err)
        band_emb = self.band_emb(band_ids)
        combined_val = torch.cat([val_emb, band_emb], dim=-1)

        # Input representation
        x = self.input_norm(time_emb + combined_val)

        # Transformer self-attention
        latents = self.transformer(x, src_key_padding_mask=padding_mask)
        return latents

    def pool_representation(
        self,
        latents: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Mean-pools valid non-padded tokens to produce an object-level representation z \in R^{B x D}.
        """
        if padding_mask is not None:
            # Mask is True for padding tokens; invert for weight
            weights = (~padding_mask).unsqueeze(-1).float()
            pooled = (latents * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            pooled = latents.mean(dim=1)
        return pooled

    def forward_masked_pretraining(
        self,
        times_mjd: torch.Tensor,
        magnitudes: torch.Tensor,
        errors: torch.Tensor,
        band_ids: torch.Tensor,
        mask_prob: float = 0.15,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Self-supervised masked autoencoding: randomly masks a subset of magnitude values
        and computes the reconstruction MSE loss against unmasked true values.
        """
        # Create random mask over non-padded elements
        rand = torch.rand_like(magnitudes)
        mask = (rand < mask_prob)
        if padding_mask is not None:
            mask = mask & (~padding_mask)

        # Clone and mask magnitudes with zero/noise
        masked_mags = magnitudes.clone()
        masked_mags[mask] = 0.0

        latents = self.forward_representation(times_mjd, masked_mags, errors, band_ids, padding_mask)
        pred_mags = self.reconstruction_head(latents).squeeze(-1)

        # Compute loss only on masked tokens
        if mask.sum() > 0:
            loss = F.mse_loss(pred_mags[mask], magnitudes[mask])
        else:
            loss = torch.tensor(0.0, device=magnitudes.device, requires_grad=True)

        return loss, pred_mags, mask

    def forward_classification(
        self,
        times_mjd: torch.Tensor,
        magnitudes: torch.Tensor,
        errors: torch.Tensor,
        band_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extracts pooled latent representation and passes through classification head.
        """
        if self.classifier_head is None:
            raise RuntimeError("Classification head not built. Call build_classifier_head(num_classes) first.")

        latents = self.forward_representation(times_mjd, magnitudes, errors, band_ids, padding_mask)
        pooled = self.pool_representation(latents, padding_mask)
        logits = self.classifier_head(pooled)
        return logits
