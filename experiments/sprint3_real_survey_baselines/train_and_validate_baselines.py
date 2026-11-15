"""
ASTRAEUS Sprint 3: Real-Survey Baseline Training, Validation & Survey Ablation Pipeline.
Gate 2 Execution Script.

Scientific Governance Contract:
- Trains strictly on Track 1 Training partition (N = 2,808).
- Validates and tunes strictly on Track 1 Validation partition (N = 741).
- The Spatial Test partition (N = 586) is SEALED and strictly UNTOUCHED.
- Evaluates 3 baseline architectures: Random Forest (RF), HistGradientBoosting (HGBT), and Calibrated Ridge (CommonFeature).
- Evaluates 3 certified observation configurations: COMBINED, ALERCE_ONLY, ZTF_DR24_ONLY.
- Evaluates 2 feature representations: Full 32-Feature vs Purged 27-Feature Physical Observable.
- Computes Wilson score 95% confidence intervals on per-class metrics.
- Outputs comprehensive experiment artifacts to experiments/sprint3_real_survey_baselines/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from astraeus_bench.real_survey_benchmark_protocol import (
    REAL_SURVEY_CLASSES_8,
    TRACK1_SPLIT_COUNTS,
    compute_wilson_score_interval,
)
from astraeus_data.real_survey.observation_feature_pipeline import (
    FEATURE_NAMES_32,
    FEATURE_SPECIFICATION_32,
    CandidateObservationLoader,
    ObservationConfig,
    RealSurveyFeatureExtractor,
)

# 27 Physical Observable Features (Strictly excluding 5 survey presence/origin indicators)
SURVEY_ORIGIN_FEATURES_5 = [
    "has_alerce_observations",
    "has_dr24_observations",
    "n_alerce_epochs",
    "n_dr24_epochs",
    "survey_epoch_ratio",
]
PHYSICAL_FEATURES_27 = [f for f in FEATURE_NAMES_32 if f not in SURVEY_ORIGIN_FEATURES_5]


def compute_multiclass_ece(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error across multi-class predictions."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true_indices)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin

    return float(ece)


def compute_multiclass_brier(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 8,
) -> float:
    """Computes Multi-Class Brier Score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
    n_samples = len(y_true_indices)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_true_indices] = 1.0
    brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
    return float(brier)


class Sprint3BaselineHarness:
    """
    Certified Sprint 3 Real-Survey Baseline Training and Validation Runner.
    """

    def __init__(
        self,
        candidates_path: str = "data/processed/real_survey_10k_v1/candidates_10k.parquet",
        alerce_lc_dir: str = "data/raw/real_survey_v1/alerce/lightcurves",
        dr24_dir: str = "data/raw/real_survey_v1/ztf_dr24",
        output_dir: str = "experiments/sprint3_real_survey_baselines",
    ):
        self.candidates_path = Path(candidates_path)
        self.alerce_lc_dir = Path(alerce_lc_dir)
        self.dr24_dir = Path(dr24_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = CandidateObservationLoader(
            alerce_lc_dir=str(self.alerce_lc_dir),
            dr24_dir=str(self.dr24_dir),
        )

        self.classes = REAL_SURVEY_CLASSES_8
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        # Load Cohort Metadata
        self.df_cand = pd.read_parquet(self.candidates_path)
        self.t1_df = self.df_cand[self.df_cand["coarse_class"].isin(self.classes)].copy()

        self.train_df = self.t1_df[self.t1_df["assigned_split"] == "train"].copy()
        self.val_df = self.t1_df[self.t1_df["assigned_split"] == "val"].copy()

        assert len(self.train_df) == TRACK1_SPLIT_COUNTS["train"], f"Train count mismatch: {len(self.train_df)}"
        assert len(self.val_df) == TRACK1_SPLIT_COUNTS["val"], f"Val count mismatch: {len(self.val_df)}"

        print(f"Harness initialized with {len(self.train_df)} train candidates, {len(self.val_df)} val candidates.", flush=True)
        print("SPATIAL TEST SET (N=586) IS SEALED AND EXCLUDED.", flush=True)

        # Preload observations to guarantee 100% in-memory feature extraction speed
        self._obs_cache: Dict[str, Any] = {}
        all_cids = list(self.train_df["candidate_id"]) + list(self.val_df["candidate_id"])
        print(f"Preloading observations for {len(all_cids)} candidates...", flush=True)
        t_pre = time.time()
        for idx, cid in enumerate(all_cids):
            self._obs_cache[cid] = self.loader.load_candidate_observations(cid)
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(all_cids):
                print(f"  Preloaded {idx + 1}/{len(all_cids)} candidates in {time.time()-t_pre:.1f}s...", flush=True)

    def _get_observations(self, candidate_id: str):
        return self._obs_cache.get(candidate_id, [])

    def extract_feature_matrices(
        self,
        config: ObservationConfig,
        feature_subset: str = "full_32",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extracts feature matrices for train and validation splits under the specified survey configuration.
        """
        print(f"\nExtracting features: Config={config.value}, FeatureSet={feature_subset}...")
        t0 = time.time()
        extractor = RealSurveyFeatureExtractor(config=config)

        feat_names = FEATURE_NAMES_32 if feature_subset == "full_32" else PHYSICAL_FEATURES_27

        # 1. Train Features
        X_train_list = []
        y_train_list = []
        for _, row in self.train_df.iterrows():
            cid = row["candidate_id"]
            label_idx = self.class_to_idx[row["coarse_class"]]
            obs = self._get_observations(cid)
            feats_dict = extractor.extract_from_observations(obs)
            vec = [feats_dict[f] for f in feat_names]
            X_train_list.append(vec)
            y_train_list.append(label_idx)

        # 2. Validation Features
        X_val_list = []
        y_val_list = []
        for _, row in self.val_df.iterrows():
            cid = row["candidate_id"]
            label_idx = self.class_to_idx[row["coarse_class"]]
            obs = self._get_observations(cid)
            feats_dict = extractor.extract_from_observations(obs)
            vec = [feats_dict[f] for f in feat_names]
            X_val_list.append(vec)
            y_val_list.append(label_idx)

        X_train = np.array(X_train_list, dtype=np.float64)
        y_train = np.array(y_train_list, dtype=np.int64)
        X_val = np.array(X_val_list, dtype=np.float64)
        y_val = np.array(y_val_list, dtype=np.int64)

        # Check for NaNs or Infs and clean
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)

        print(f"  Extracted X_train {X_train.shape}, X_val {X_val.shape} in {time.time()-t0:.2f}s")
        return X_train, y_train, X_val, y_val, feat_names

    def train_and_evaluate_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Trains a specified baseline model on X_train/y_train and evaluates on X_val/y_val.
        Computes Macro-F1, per-class CIs, Balanced Accuracy, ECE, Brier score, and feature importances.
        """
        t0 = time.time()
        scaler = None

        if model_name == "RandomForest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
            cal_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3, n_jobs=1)
            cal_clf.fit(X_train, y_train)
            model_to_predict = cal_clf
            clf.fit(X_train, y_train)
            importances = clf.feature_importances_.tolist()

        elif model_name == "HistGradientBoosting":
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.08,
                max_depth=5,
                class_weight="balanced",
                random_state=seed,
            )
            cal_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3, n_jobs=1)
            cal_clf.fit(X_train, y_train)
            model_to_predict = cal_clf
            importances = []

        elif model_name == "CalibratedRidge":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            base_ridge = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=seed)
            cal_clf = CalibratedClassifierCV(estimator=base_ridge, method="sigmoid", cv=3, n_jobs=1)
            cal_clf.fit(X_train_scaled, y_train)
            model_to_predict = cal_clf
            importances = []
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        fit_time = time.time() - t0

        # Predict on validation set
        if scaler is not None:
            X_val_eval = scaler.transform(X_val)
        else:
            X_val_eval = X_val

        y_prob = model_to_predict.predict_proba(X_val_eval)
        y_pred = np.argmax(y_prob, axis=1)

        # Performance Metrics
        macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        accuracy = float(accuracy_score(y_val, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_val, y_pred))
        ece = compute_multiclass_ece(y_val, y_prob)
        brier = compute_multiclass_brier(y_val, y_prob)
        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(self.classes)))).tolist()

        # Per-Class Metrics with Wilson Score Intervals
        per_class_results = {}
        for idx, cname in enumerate(self.classes):
            c_true = (y_val == idx)
            c_pred = (y_pred == idx)
            support = int(np.sum(c_true))

            tp = int(np.sum(c_true & c_pred))
            fp = int(np.sum((~c_true) & c_pred))
            fn = int(np.sum(c_true & (~c_pred)))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / support) if support > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            # Wilson score CIs
            _, prec_low, prec_high = compute_wilson_score_interval(tp, tp + fp)
            _, rec_low, rec_high = compute_wilson_score_interval(tp, support)

            per_class_results[cname] = {
                "support": support,
                "precision": round(prec, 4),
                "precision_ci_95": [round(prec_low, 4), round(prec_high, 4)],
                "recall": round(rec, 4),
                "recall_ci_95": [round(rec_low, 4), round(rec_high, 4)],
                "f1": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "is_rare_class": (support < 30),
            }

        # Feature Importance Mapping (for tree models)
        feat_imp_dict = {}
        if importances and len(importances) == len(feature_names):
            feat_imp_dict = {
                f: round(float(imp), 5)
                for f, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            }

        result = {
            "model_name": model_name,
            "seed": seed,
            "fit_time_seconds": round(fit_time, 2),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "expected_calibration_error": round(ece, 4),
            "brier_score": round(brier, 4),
            "per_class": per_class_results,
            "confusion_matrix": cm,
            "feature_importances_top10": dict(list(feat_imp_dict.items())[:10]),
        }

        return result

    def run_full_suite(self) -> Dict[str, Any]:
        """
        Executes the complete Sprint 3 experiment matrix across configurations,
        feature representations, and models.
        """
        print("\n" + "=" * 80)
        print("ASTRAEUS Sprint 3: Running Complete Real-Survey Baseline Suite")
        print("=" * 80)

        suite_results: Dict[str, Any] = {
            "milestone": "ASTRAEUS Sprint 3 Real-Survey Baselines",
            "execution_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "governance_lock": "ACTIVE",
            "spatial_test_sealed": True,
            "taxonomy": self.classes,
            "train_size": len(self.train_df),
            "val_size": len(self.val_df),
            "configurations": {},
            "survey_origin_shortcut_audit": {},
            "multi_seed_stability": {},
        }

        configs = [
            ObservationConfig.COMBINED,
            ObservationConfig.ALERCE_ONLY,
            ObservationConfig.ZTF_DR24_ONLY,
        ]
        models = ["RandomForest", "HistGradientBoosting", "CalibratedRidge"]

        # Run primary sweep across configurations
        for cfg in configs:
            cfg_key = cfg.value
            suite_results["configurations"][cfg_key] = {
                "full_32_features": {},
                "physical_27_features": {},
            }

            # 1. Full 32 Features
            X_tr, y_tr, X_va, y_va, fnames = self.extract_feature_matrices(cfg, "full_32")
            for m in models:
                res = self.train_and_evaluate_model(m, X_tr, y_tr, X_va, y_va, fnames, seed=42)
                suite_results["configurations"][cfg_key]["full_32_features"][m] = res
                print(f"  [{cfg_key} | Full-32 | {m:20s}] Macro-F1 = {res['macro_f1']:.4f}, Bal-Acc = {res['balanced_accuracy']:.4f}, ECE = {res['expected_calibration_error']:.4f}")

            # 2. Physical 27 Features (Purged of survey presence shortcuts)
            X_tr_p, y_tr_p, X_va_p, y_va_p, fnames_p = self.extract_feature_matrices(cfg, "physical_27")
            for m in models:
                res_p = self.train_and_evaluate_model(m, X_tr_p, y_tr_p, X_va_p, y_va_p, fnames_p, seed=42)
                suite_results["configurations"][cfg_key]["physical_27_features"][m] = res_p
                print(f"  [{cfg_key} | Phys-27 | {m:20s}] Macro-F1 = {res_p['macro_f1']:.4f}, Bal-Acc = {res_p['balanced_accuracy']:.4f}, ECE = {res_p['expected_calibration_error']:.4f}")

        # -------------------------------------------------------------
        # Shortcut Audit: Quantify delta between Full-32 and Physical-27
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("Auditing Survey Shortcut Impact (Full-32 vs Physical-27)...")
        print("=" * 80)
        shortcut_audit = {}
        for cfg in configs:
            cfg_key = cfg.value
            shortcut_audit[cfg_key] = {}
            for m in models:
                f32_f1 = suite_results["configurations"][cfg_key]["full_32_features"][m]["macro_f1"]
                p27_f1 = suite_results["configurations"][cfg_key]["physical_27_features"][m]["macro_f1"]
                delta = round(f32_f1 - p27_f1, 4)
                pct_drop = round((delta / f32_f1) * 100, 2) if f32_f1 > 0 else 0.0
                shortcut_audit[cfg_key][m] = {
                    "full_32_macro_f1": f32_f1,
                    "physical_27_macro_f1": p27_f1,
                    "shortcut_delta_macro_f1": delta,
                    "percentage_drop_when_purged": pct_drop,
                }
                print(f"  [{cfg_key:12s} | {m:20s}] Full: {f32_f1:.4f} -> Purged: {p27_f1:.4f} (Delta: {delta:+.4f}, Drop: {pct_drop:.1f}%)")

        suite_results["survey_origin_shortcut_audit"] = shortcut_audit

        # -------------------------------------------------------------
        # Multi-Seed Stability Check on Champion Architecture (COMBINED, Physical-27)
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("Running Multi-Seed Stability Check (Seeds 42, 123, 777)...")
        print("=" * 80)
        X_tr_c, y_tr_c, X_va_c, y_va_c, fnames_c = self.extract_feature_matrices(ObservationConfig.COMBINED, "physical_27")
        seed_results = []
        for s in [42, 123, 777]:
            res_s = self.train_and_evaluate_model("RandomForest", X_tr_c, y_tr_c, X_va_c, y_va_c, fnames_c, seed=s)
            seed_results.append(res_s)
            print(f"  [RandomForest | Seed {s:3d}] Macro-F1 = {res_s['macro_f1']:.4f}, Bal-Acc = {res_s['balanced_accuracy']:.4f}")

        f1_vals = [r["macro_f1"] for r in seed_results]
        suite_results["multi_seed_stability"]["RandomForest_COMBINED_Physical27"] = {
            "seeds": [42, 123, 777],
            "macro_f1_values": f1_vals,
            "mean_macro_f1": round(float(np.mean(f1_vals)), 4),
            "std_macro_f1": round(float(np.std(f1_vals)), 4),
        }

        # Save primary result JSON
        res_file = self.output_dir / "sprint3_baseline_results.json"
        with open(res_file, "w") as f:
            json.dump(suite_results, f, indent=2)

        print(f"\nSaved full baseline benchmark results to: {res_file}")
        return suite_results


if __name__ == "__main__":
    harness = Sprint3BaselineHarness()
    harness.run_full_suite()
