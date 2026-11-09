"""
ASTRAEUS Phase 1 Baseline Benchmark Execution Script.
Executes head-to-head comparison between Feature-Engineered Baseline (Random Forest)
and Learned Representation Baseline (ASTROMER) across segregated scientific tasks.
Outputs: experiments/phase1_baselines/benchmark_results.json
"""

import json
from pathlib import Path
import numpy as np
import torch

from astraeus_data.synthetic import generate_synthetic_entity
from astraeus_bench.astro_temporal import AstroTemporalBenchmark


def run_experiment(output_json: str = "experiments/phase1_baselines/benchmark_results.json"):
    print("=" * 70)
    print("ASTRAEUS Phase 1: AstroTemporal Baseline Benchmark")
    print("Task: Segregated 3-Class Classification")
    print("Classes: ['rr_lyrae', 'supernova_ia', 'detector_artifact']")
    print("=" * 70)

    classes = ["rr_lyrae", "supernova_ia", "detector_artifact"]
    class_map = {name: idx for idx, name in enumerate(classes)}

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Generate 60 train and 18 test synthetic objects
    print("\n[1/3] Generating synthetic benchmark light curves with IVOA provenance...")
    for idx, cls_name in enumerate(classes):
        # 20 per class for training
        for i in range(20):
            e = generate_synthetic_entity(cls_name, seed=1000 + idx * 100 + i)
            times = np.array([obs.mjd for obs in e.observations])
            mags = np.array([obs.calibrated_flux.value for obs in e.observations])
            errs = np.array([obs.calibrated_flux.uncertainty for obs in e.observations])
            bands = [obs.filter_band for obs in e.observations]
            train_data.append((times, mags, errs, bands))
            train_labels.append(idx)

        # 6 per class for testing
        for i in range(6):
            e = generate_synthetic_entity(cls_name, seed=2000 + idx * 100 + i)
            times = np.array([obs.mjd for obs in e.observations])
            mags = np.array([obs.calibrated_flux.value for obs in e.observations])
            errs = np.array([obs.calibrated_flux.uncertainty for obs in e.observations])
            bands = [obs.filter_band for obs in e.observations]
            test_data.append((times, mags, errs, bands))
            test_labels.append(idx)

    print(f"Dataset generated: {len(train_data)} train samples, {len(test_data)} test samples.")

    # Run Benchmark
    print("\n[2/3] Executing head-to-head model comparison...")
    benchmark = AstroTemporalBenchmark(d_model=64, num_layers=2)
    report = benchmark.run_benchmark(
        task_name="astro_temporal_phase1_3class",
        classes=classes,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        astromer_epochs=20,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    print(f"Random Forest Macro-F1: {report.random_forest_metrics.macro_f1:.4f}")
    if report.random_forest_metrics.calibration:
        print(f"Random Forest ECE:      {report.random_forest_metrics.calibration.expected_calibration_error:.4f}")
    print(f"Random Forest Latency:  {report.rf_latency_ms_per_sample:.3f} ms/sample")
    print("-" * 70)
    print(f"ASTROMER Macro-F1:      {report.astromer_metrics.macro_f1:.4f}")
    if report.astromer_metrics.calibration:
        print(f"ASTROMER ECE:           {report.astromer_metrics.calibration.expected_calibration_error:.4f}")
    print(f"ASTROMER Latency:       {report.astromer_latency_ms_per_sample:.3f} ms/sample")
    print(f"Macro-F1 Delta:         {report.astromer_f1_delta:+.4f}")
    print("=" * 70)

    # Save to disk
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"\n[3/3] Benchmark report saved to: {out_path.resolve()}")
    return report


if __name__ == "__main__":
    run_experiment()
