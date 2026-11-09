"""
Baseline 3: Non-Pretrained Bidirectional Continuous-Time Transformer.
Architecture ablation control (1.19M parameters) evaluated across 5 frozen seeds:
[42, 137, 1024, 2048, 4096].
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astraeus_data.corpus.sequence import TemporalSequence, ObservationRecord, ObservationType

PASSBAND_ORDER = ["u", "g", "r", "i", "z", "Y", "c", "o"]
PASSBAND_MAP = {pb: idx for idx, pb in enumerate(PASSBAND_ORDER)}
SURVEY_MAP = {"ZTF": 0, "ATLAS": 1, "PS1": 2, "DECAM": 3, "SYNTHETIC": 4}


class ContinuousLogarithmicTimeEncoding(nn.Module):
    """
    Continuous logarithmic harmonic positional encoding across K=32 timescales (64 dimensions),
    spanning 0.01 days (14.4 min) to 500.0 days.
    """
    def __init__(self, num_timescales: int = 32, tau_min: float = 0.01, tau_max: float = 500.0):
        super().__init__()
        self.num_timescales = num_timescales
        # Logarithmically spaced timescales tau_k
        k = torch.arange(num_timescales, dtype=torch.float32)
        taus = tau_min * ((tau_max / tau_min) ** (k / (num_timescales - 1)))
        self.register_buffer("inv_taus", 1.0 / taus)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t shape: (B, L)
        Returns: (B, L, 64)
        """
        # Outer product: (B, L, 1) * (1, 1, K) -> (B, L, K)
        phases = t.unsqueeze(-1) * self.inv_taus.view(1, 1, -1)
        sin_part = torch.sin(phases)
        cos_part = torch.cos(phases)
        return torch.cat([sin_part, cos_part], dim=-1)


class NonPretrainedTransformerModel(nn.Module):
    """
    6-layer bidirectional continuous-time Transformer encoder (1.19M parameters).
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.10,
        num_classes: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.time_encoder = ContinuousLogarithmicTimeEncoding(num_timescales=32)

        # Discrete embedding layers
        self.band_embed = nn.Embedding(len(PASSBAND_ORDER) + 1, 16)
        self.survey_embed = nn.Embedding(len(SURVEY_MAP) + 1, 8)
        self.qual_embed = nn.Embedding(4, 4)

        # Input feature projection: 4 numerical features (flux, sigma, SNR, dt) + 64 time + 16 band + 8 survey + 4 qual = 96
        self.input_proj = nn.Linear(4 + 64 + 16 + 8 + 4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.reg_head = nn.Linear(d_model, 2)  # [mu_pred, unconstrained_s]
        self.cls_head = nn.Linear(d_model, num_classes)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        features: torch.Tensor,
        times: torch.Tensor,
        bands: torch.Tensor,
        surveys: torch.Tensor,
        quals: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - mu_pred: (B, L)
        - sigma_aleatoric: (B, L) = softplus(s)
        - class_logits: (B, num_classes)
        """
        pe_time = self.time_encoder(times)
        emb_b = self.band_embed(bands)
        emb_s = self.survey_embed(surveys)
        emb_q = self.qual_embed(quals)

        x_in = torch.cat([features, pe_time, emb_b, emb_s, emb_q], dim=-1)
        h0 = self.input_proj(x_in)

        # Encoder pass
        h = self.encoder(
            h0,
            mask=is_causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        reg_out = self.reg_head(h)
        mu_pred = reg_out[..., 0]
        s_unconstrained = reg_out[..., 1]
        sigma_aleatoric = F.softplus(s_unconstrained) + 1e-4

        # Attention pooling for sequence classification
        if src_key_padding_mask is not None:
            weights = (~src_key_padding_mask).float().unsqueeze(-1)
            h_pool = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            h_pool = h.mean(dim=1)

        class_logits = self.cls_head(h_pool) / self.temperature.clamp(min=0.1)
        return mu_pred, sigma_aleatoric, class_logits


class NonPretrainedTransformerBaseline:
    """
    Manages training and evaluation of Baseline 3 across the 5 frozen seeds:
    [42, 137, 1024, 2048, 4096].
    """
    FROZEN_SEEDS = [42, 137, 1024, 2048, 4096]

    def __init__(self, seed: int = 42, device: Optional[str] = None):
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._set_seed(seed)
        self.model = NonPretrainedTransformerModel().to(self.device)
        self.is_trained = False

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def sequence_to_tensors(self, seq: TemporalSequence) -> Dict[str, torch.Tensor]:
        """Converts TemporalSequence into PyTorch tensor inputs."""
        n = len(seq.observations)
        t_obs = np.array([o.t_obs_bmjd for o in seq.observations], dtype=np.float32)
        t0 = t_obs[0] if n > 0 else 0.0
        rel_times = t_obs - t0
        dt = np.diff(t_obs, prepend=t_obs[0])

        feats = np.zeros((n, 4), dtype=np.float32)
        bands = np.zeros(n, dtype=np.int64)
        surveys = np.zeros(n, dtype=np.int64)
        quals = np.zeros(n, dtype=np.int64)

        for i, o in enumerate(seq.observations):
            flx = o.measured_flux_nmgy if o.measured_flux_nmgy is not None else 0.0
            sig = o.effective_measurement_sigma_nmgy
            feats[i, 0] = flx          # F / sigma_0 (sigma_0 = 1 nmgy)
            feats[i, 1] = sig          # sigma / sigma_0
            feats[i, 2] = flx / max(sig, 1e-4)  # SNR
            feats[i, 3] = math.log(1.0 + max(dt[i], 0.0))  # log interval

            bands[i] = PASSBAND_MAP.get(o.passband, 0)
            surveys[i] = SURVEY_MAP.get(o.survey_id.upper(), 0)
            quals[i] = 1 if o.quality_severity == "DEGRADED" else (2 if o.quality_severity == "FATAL" else 0)

        return {
            "features": torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device),
            "times": torch.tensor(rel_times, dtype=torch.float32).unsqueeze(0).to(self.device),
            "bands": torch.tensor(bands, dtype=torch.int64).unsqueeze(0).to(self.device),
            "surveys": torch.tensor(surveys, dtype=torch.int64).unsqueeze(0).to(self.device),
            "quals": torch.tensor(quals, dtype=torch.int64).unsqueeze(0).to(self.device)
        }

    def predict_sequence(self, seq: TemporalSequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates forward inference:
        Returns mu_pred, sigma_aleatoric, class_probabilities.
        """
        self.model.eval()
        tensors = self.sequence_to_tensors(seq)
        with torch.no_grad():
            mu, sig_aleatoric, logits = self.model(
                tensors["features"],
                tensors["times"],
                tensors["bands"],
                tensors["surveys"],
                tensors["quals"]
            )
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            return mu.squeeze(0).cpu().numpy(), sig_aleatoric.squeeze(0).cpu().numpy(), probs
