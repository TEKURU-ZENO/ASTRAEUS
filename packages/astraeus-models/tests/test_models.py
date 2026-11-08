import pytest
import numpy as np

from astraeus_models.periodicity import LombScargleExtractor
from astraeus_models.features import AstronomicalFeatureExtractor
from astraeus_models.classifier import TaskDomain, TaskSeparatedClassifier
from astraeus_models.hardware_benchmark import HardwareBenchmarkRunner


def test_lomb_scargle_periodicity_recovery():
    # Synthetic periodic signal: period = 0.5 days
    true_period = 0.50
    times = np.sort(np.random.uniform(59000.0, 59030.0, size=50))
    magnitudes = 17.0 + 0.5 * np.sin(2.0 * np.pi * times / true_period) + np.random.normal(0, 0.05, size=50)
    errors = np.full_like(times, 0.05)

    extractor = LombScargleExtractor(min_period_days=0.1, max_period_days=5.0)
    res = extractor.extract(times, magnitudes, errors)

    assert res.is_periodic is True
    assert res.best_period_days == pytest.approx(true_period, rel=0.05)
    assert res.peak_power > 0.40


def test_feature_extraction():
    times = np.array([59000.0, 59002.0, 59005.0, 59010.0, 59020.0])
    mags = np.array([19.5, 17.2, 17.0, 18.2, 19.8])
    errs = np.array([0.05, 0.03, 0.03, 0.04, 0.06])
    bands = ["ZTF_g", "ZTF_r", "ZTF_g", "ZTF_r", "ZTF_g"]

    extractor = AstronomicalFeatureExtractor()
    feat = extractor.extract(times, mags, errs, filter_bands=bands)

    assert feat.num_epochs == 5
    assert feat.peak_mag == pytest.approx(17.0)
    assert feat.faintest_mag == pytest.approx(19.8)
    assert feat.amplitude == pytest.approx(2.8)
    assert feat.rise_rate_mag_per_day > 0.0


def test_task_separated_classifier():
    extractor = AstronomicalFeatureExtractor()
    train_features = []
    train_labels = []

    # Generate 20 periodic and 20 transient samples
    for i in range(20):
        t = np.linspace(0, 10, 20)
        # Periodic
        m_per = 18.0 + 0.4 * np.sin(2 * np.pi * t / 0.5)
        train_features.append(extractor.extract(t, m_per, np.full_like(t, 0.05)))
        train_labels.append("periodic")

        # Transient
        m_trans = 18.0 - 2.0 * np.exp(-t / 3.0)
        train_features.append(extractor.extract(t, m_trans, np.full_like(t, 0.05)))
        train_labels.append("transient")

    test_features = train_features[:6]
    test_labels = train_labels[:6]

    classifier = TaskSeparatedClassifier(task_domain=TaskDomain.PERIODIC_VARIABLE)
    res = classifier.train_and_evaluate(train_features, train_labels, test_features, test_labels)

    assert res.macro_f1 >= 0.80
    assert "amplitude" in res.feature_importances


def test_hardware_vram_profiler():
    runner = HardwareBenchmarkRunner()
    res = runner.benchmark_workload(
        batch_size=8,
        seq_length=128,
        d_model=128,
        nhead=2,
        num_layers=2,
        precision="fp16"
    )
    assert res.allocated_mb > 0.0
    assert res.fits_in_8gb_envelope is True
    assert res.throughput_samples_per_sec > 0.0
