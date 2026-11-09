import pytest
import numpy as np
import torch

from astraeus_bench.metrics import (
    compute_expected_calibration_error,
    compute_crps_gaussian,
    compute_classification_metrics,
    compute_anomaly_budget_metrics,
)
from astraeus_models.astromer import AstromerEncoder, ContinuousTimePositionalEncoding
from astraeus_bench.astro_temporal import AstroTemporalBenchmark


def test_continuous_time_positional_encoding():
    pe = ContinuousTimePositionalEncoding(d_model=64)
    times = torch.tensor([[0.0, 1.5, 5.2, 10.0], [0.0, 0.5, 2.0, 30.0]])
    emb = pe(times)
    assert emb.shape == (2, 4, 64)
    # Ensure distinct epochs have distinct embeddings
    assert not torch.allclose(emb[0, 0], emb[0, 1])


def test_astromer_encoder_forward_and_pretraining():
    model = AstromerEncoder(d_model=32, nhead=2, num_layers=2, dim_feedforward=64)
    b, seq_len = 4, 16
    times = torch.rand(b, seq_len) * 50.0
    mags = torch.rand(b, seq_len) * 5.0 + 15.0
    errs = torch.full((b, seq_len), 0.05)
    bands = torch.randint(1, 3, (b, seq_len))

    # Test representation pooling
    latents = model.forward_representation(times, mags, errs, bands)
    assert latents.shape == (b, seq_len, 32)

    pooled = model.pool_representation(latents)
    assert pooled.shape == (b, 32)

    # Test masked pretraining loss
    loss, pred_mags, mask = model.forward_masked_pretraining(times, mags, errs, bands, mask_prob=0.25)
    assert loss.item() >= 0.0
    assert pred_mags.shape == (b, seq_len)

    # Test classification head
    model.build_classifier_head(num_classes=3)
    logits = model.forward_classification(times, mags, errs, bands)
    assert logits.shape == (b, 3)


def test_expected_calibration_error():
    # Perfectly calibrated probabilities
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.1, 0.9],
        [0.2, 0.8],
    ])
    labels = np.array([0, 0, 0, 1, 1])
    metrics = compute_expected_calibration_error(probs, labels, n_bins=5)
    assert metrics.expected_calibration_error < 0.25
    assert metrics.brier_score < 0.15


def test_crps_gaussian_calculation():
    y_true = np.array([10.0, 15.0, 20.0])
    mu_pred = np.array([10.0, 15.0, 20.0])
    sigma_pred = np.array([1.0, 1.0, 1.0])

    # When mu == y_true and sigma is small, CRPS should be close to sigma / sqrt(pi) * const
    crps = compute_crps_gaussian(y_true, mu_pred, sigma_pred)
    assert crps > 0.0
    assert crps < 1.0


def test_anomaly_budget_metrics():
    # 10 objects, 3 are true anomalies
    scores = np.array([0.95, 0.90, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])
    is_true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    # Budget of top 3: gets indices 0, 1, 2 (labels: 1, 1, 0 -> 2 found)
    metrics = compute_anomaly_budget_metrics(scores, is_true, budget_n=3)
    assert metrics.true_anomalies_found == 2
    assert metrics.recall_at_budget == pytest.approx(2.0 / 3.0)
    assert metrics.precision_at_budget == pytest.approx(2.0 / 3.0)
    assert metrics.false_discovery_rate == pytest.approx(1.0 / 3.0)


def test_astro_temporal_benchmark_runner():
    bench = AstroTemporalBenchmark(d_model=32, num_layers=2)
    classes = ["periodic", "transient"]

    # Generate synthetic training and testing sets
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for i in range(16):
        t = np.linspace(0, 15, 20)
        # Class 0: Periodic
        m_per = 17.5 + 0.4 * np.sin(2 * np.pi * t / 0.8) + np.random.normal(0, 0.03, len(t))
        e = np.full_like(t, 0.03)
        b = ["ZTF_g" if j % 2 == 0 else "ZTF_r" for j in range(len(t))]
        train_data.append((t, m_per, e, b))
        train_labels.append(0)

        # Class 1: Transient
        m_trans = 19.0 - 2.5 * np.exp(-t / 4.0) + np.random.normal(0, 0.03, len(t))
        train_data.append((t, m_trans, e, b))
        train_labels.append(1)

    for i in range(4):
        t = np.linspace(0, 15, 20)
        m_per = 17.5 + 0.4 * np.sin(2 * np.pi * t / 0.8) + np.random.normal(0, 0.03, len(t))
        e = np.full_like(t, 0.03)
        b = ["ZTF_g" if j % 2 == 0 else "ZTF_r" for j in range(len(t))]
        test_data.append((t, m_per, e, b))
        test_labels.append(0)

        m_trans = 19.0 - 2.5 * np.exp(-t / 4.0) + np.random.normal(0, 0.03, len(t))
        test_data.append((t, m_trans, e, b))
        test_labels.append(1)

    report = bench.run_benchmark(
        task_name="periodic_vs_transient_test",
        classes=classes,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        astromer_epochs=5,
    )

    assert report.num_train == 32
    assert report.num_test == 8
    assert report.random_forest_metrics.macro_f1 >= 0.70
    assert report.astromer_metrics.macro_f1 >= 0.0
    assert report.rf_latency_ms_per_sample > 0.0
    assert report.astromer_latency_ms_per_sample > 0.0
