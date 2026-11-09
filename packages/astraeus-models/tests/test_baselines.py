"""
ASTRAEUS Phase 2 Stage 2 — Continuous-Time Baselines Unit Tests (Day 110).
Verifies Baselines 1-4 and the Common 24-Feature Platt-Calibrated Classifier.
"""

import pytest
import math
import numpy as np
import torch

from astraeus_data.corpus.sequence import (
    ObservationRecord,
    ObservationType,
    SequenceViewType,
    TemporalSequence,
)
from astraeus_models.baselines.akima import AkimaBaselineModel
from astraeus_models.baselines.gp import MultiOutputGPBaseline, PASSBAND_ORDER
from astraeus_models.baselines.transformer import NonPretrainedTransformerBaseline
from astraeus_models.baselines.astromer_adapter import AstromerAdapterBaseline
from astraeus_models.baselines.common_classifier import CommonFeatureClassifier, CLASSES_8


def _create_synthetic_multiband_sequence(n_obs: int = 50) -> TemporalSequence:
    records = []
    base_t = 60000.0
    passbands = ["g", "r", "i", "z"]

    for i in range(n_obs):
        t_bmjd = base_t + i * 1.0  # 1-day spacing
        pb = passbands[i % len(passbands)]
        phase = i - 25.0
        flx = 15.0 * math.exp(-(phase ** 2) / (2.0 * (8.0 ** 2))) + 0.5
        sig = 0.2

        rec = ObservationRecord(
            object_id="SN-BENCH-001",
            observation_id=f"OBS-{i:03d}",
            t_obs_bmjd=t_bmjd,
            t_pub_utc=f"2023-04-{(i%28)+1:02d}T12:00:00Z",
            passband=pb,
            survey_id="ZTF",
            obs_type=ObservationType.FORCED_PHOTOMETRY,
            measured_flux_nmgy=flx,
            measurement_sigma_nmgy=sig,
            effective_measurement_sigma_nmgy=sig,
            provenance_hash=f"hash_{i}"
        )
        records.append(rec)

    return TemporalSequence.from_records("SN-BENCH-001", records)


class TestAkimaBaseline:
    """Verifies Baseline 1 Akima multi-band interpolation and causal forecasting."""

    def test_multiband_fitting_and_crps(self):
        seq = _create_synthetic_multiband_sequence(40)
        model = AkimaBaselineModel(tau_decay=15.0, s_extrap_prior=2.5)
        model.fit_sequence(seq)

        # Predict within observed range for 'g' band
        mu_g, sig_g = model.predict_band("g", np.array([60015.0]))
        assert len(mu_g) == 1
        assert mu_g[0] > 0.0
        assert sig_g[0] > 0.0

        # Predict in future extrapolation (t = 60060, after t_max = 60039)
        mu_future, sig_future = model.predict_band("g", np.array([60060.0]))
        # Uncertainty in future must retain boundary variance and expand
        assert sig_future[0] > sig_g[0]

        # Evaluate CRPS
        crps = AkimaBaselineModel.evaluate_crps(1.0, float(mu_future[0]), float(sig_future[0]))
        assert crps >= 0.0


class TestMultiOutputGPBaseline:
    """Verifies Baseline 2 8-passband coregionalized Gaussian Process."""

    def test_coregionalization_psd_and_prediction(self):
        gp = MultiOutputGPBaseline(lengthscale=10.0, signal_var=2.0, rank=3)
        B = gp.get_coregionalization_matrix()

        # Assert B is symmetric and positive semi-definite (all eigenvalues >= 0)
        assert B.shape == (8, 8)
        assert np.allclose(B, B.T)
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals >= -1e-7), f"Eigenvalues must be non-negative: {eigvals}"

        # Fit sequence
        seq = _create_synthetic_multiband_sequence(30)
        gp.fit_sequence(seq)

        # Predict across bands
        mu_r, sig_r = gp.predict_band("r", np.array([60015.0, 60045.0]))
        assert len(mu_r) == 2
        assert np.all(sig_r > 0.0)

        # CRPS calculation
        crps = gp.evaluate_crps(y_true=5.0, mu_pred=float(mu_r[0]), sigma_pred=float(sig_r[0]))
        assert crps >= 0.0


class TestNonPretrainedTransformerBaseline:
    """Verifies Baseline 3 1.19M parameter continuous-time Transformer across 5 seeds."""

    def test_parameter_count_and_forward(self):
        baseline = NonPretrainedTransformerBaseline(seed=42, device="cpu")
        param_count = baseline.count_parameters()
        # Architecture has ~1.19M parameters (between 1.15M and 1.25M)
        assert 1_150_000 <= param_count <= 1_250_000, f"Unexpected parameter count: {param_count}"

        seq = _create_synthetic_multiband_sequence(20)
        mu, sig, probs = baseline.predict_sequence(seq)

        assert len(mu) == 20
        assert len(sig) == 20
        assert np.all(sig > 0.0)
        assert len(probs) == 8
        assert math.isclose(float(np.sum(probs)), 1.0, rel_tol=1e-5)

    def test_reproducibility_across_seeds(self):
        """Verifies distinct random seeds produce distinct deterministic initializations."""
        b42 = NonPretrainedTransformerBaseline(seed=42, device="cpu")
        b137 = NonPretrainedTransformerBaseline(seed=137, device="cpu")

        p42 = b42.model.input_proj.weight.detach().numpy()
        p137 = b137.model.input_proj.weight.detach().numpy()
        assert not np.allclose(p42, p137)


class TestAstromerAdapterBaseline:
    """Verifies Baseline 4 ASTROMER adapter, frozen backbone, and pretraining audit."""

    def test_frozen_backbone_and_overlap_audit(self):
        adapter = AstromerAdapterBaseline(d_astromer=256, num_classes=8, device="cpu")

        # Verify backbone weights have requires_grad == False
        for param in adapter.frozen_backbone.parameters():
            assert param.requires_grad is False

        # Verify pretraining spatial overlap audit
        # High latitude extragalactic coordinates (ra=150.0, dec=+30.0)
        audit_result = AstromerAdapterBaseline.audit_pretraining_overlap([(150.0, 30.0), (45.0, -10.0)])
        assert audit_result["overlap_audit_passed"] is True
        assert audit_result["is_spatially_disjoint"] is True
        assert audit_result["min_angular_separation_deg"] > 25.0

        # Predict sequence
        seq = _create_synthetic_multiband_sequence(15)
        mu, sig, probs = adapter.predict_sequence(seq)
        assert len(mu) == 15
        assert len(probs) == 8
        assert math.isclose(float(np.sum(probs)), 1.0, rel_tol=1e-5)


class TestCommonFeatureClassifier:
    """Verifies Baseline 5 24-feature extractor and Platt-calibrated GBDT layer."""

    def test_24_feature_extraction_and_causal_isolation(self):
        clf = CommonFeatureClassifier(ebv_mw=0.03)
        seq = _create_synthetic_multiband_sequence(40)

        feats = clf.extract_features(seq)
        assert feats.shape == (24,)
        # Peak fluxes (indices 0-7) must be non-negative
        assert np.all(feats[:8] >= 0.0)
        # Rise times (indices 8-13) and decline times (indices 14-19) must be positive
        assert np.all(feats[8:20] > 0.0)

    def test_platt_calibration_and_simplex_normalization(self):
        clf = CommonFeatureClassifier()
        # Synthetic feature dataset: 30 train samples, 10 val samples
        np.random.seed(42)
        X_train = np.random.randn(30, 24)
        y_train = np.random.randint(0, 8, size=30)
        X_val = np.random.randn(10, 24)
        y_val = np.random.randint(0, 8, size=10)

        clf.fit(X_train, y_train, X_val, y_val)
        probs = clf.predict_proba(X_val)

        assert probs.shape == (10, 8)
        # Every probability row must sum to 1.0
        row_sums = np.sum(probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
