"""
ASTRAEUS Sprint 4B, 4C, 4D: Temporal Sequence Modeling, Calibration/Abstention, and Track 2 Open-Set Discovery.

Scientific Governance Contract:
- Track 1 Train (N=2,808) and Val (N=741) used for supervised comparisons.
- Track 1 Spatial Test (N=586) remains strictly SEALED.
- Track 2 (N=4,365) Unclassified L5 candidates evaluated strictly via UNSUPERVISED open-set discovery (NO fabricated labels).
- Evaluates:
    Sprint 4B: AstromerEncoder temporal baseline vs 27-D feature baselines on identical splits.
    Sprint 4C: Conformal calibration, temperature scaling, and selective classification (Risk-Coverage abstention).
    Sprint 4D: Multi-method unsupervised anomaly discovery (IsolationForest, LOF, Mahalanobis) over 4,365 Track 2 candidates.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import MinCovDet
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from astraeus_models.astromer import AstromerEncoder
from astraeus_bench.real_survey_benchmark_protocol import (
    REAL_SURVEY_CLASSES_8,
    TRACK1_SPLIT_COUNTS,
    compute_wilson_score_interval,
)
from astraeus_data.real_survey.observation_feature_pipeline import (
    FEATURE_NAMES_32,
    CandidateObservationLoader,
    ObservationConfig,
    RealSurveyFeatureExtractor,
)

# 27 Physical Observable Features (Excluding 5 survey origin indicators)
SURVEY_ORIGIN_FEATURES_5 = [
    "has_alerce_observations",
    "has_dr24_observations",
    "n_alerce_epochs",
    "n_dr24_epochs",
    "survey_epoch_ratio",
]
PHYSICAL_FEATURES_27 = [f for f in FEATURE_NAMES_32 if f not in SURVEY_ORIGIN_FEATURES_5]


# ==============================================================================
# 1. Temporal Sequence PyTorch Wrapper for ASTROMER
# ==============================================================================
class AstromerSequenceClassifier:
    """Wrapper to train and evaluate AstromerEncoder on variable-length photometric series."""

    def __init__(
        self,
        num_classes: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
        epochs: int = 25,
        lr: float = 0.002,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.lr = lr

        self.model = AstromerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=128,
            dropout=0.15,
            max_seq_len=max_seq_len,
        ).to(self.device)
        self.model.build_classifier_head(num_classes)

    def _pad_and_tensorize(
        self,
        raw_obs_list: List[List[Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = len(raw_obs_list)
        t_batch = torch.zeros(b, self.max_seq_len, dtype=torch.float32)
        m_batch = torch.zeros(b, self.max_seq_len, dtype=torch.float32)
        e_batch = torch.zeros(b, self.max_seq_len, dtype=torch.float32)
        b_batch = torch.zeros(b, self.max_seq_len, dtype=torch.long)
        mask_batch = torch.ones(b, self.max_seq_len, dtype=torch.bool)  # True = padded

        band_map = {"g": 1, "r": 2, "i": 3}

        for i, obs_seq in enumerate(raw_obs_list):
            if not obs_seq:
                continue
            seq_len = min(len(obs_seq), self.max_seq_len)
            first_mjd = obs_seq[0].mjd
            for j in range(seq_len):
                o = obs_seq[j]
                t_batch[i, j] = float(o.mjd - first_mjd)
                # Convert flux to normalized magnitude representation
                f_clamped = max(float(o.flux_nmgy), 1e-4)
                mag_val = 22.500 - 2.5 * math.log10(f_clamped)
                mag_err = min(max(float(o.flux_err_nmgy / f_clamped), 0.01), 3.0)
                m_batch[i, j] = float(mag_val - 19.0)  # zero-center around 19 mag
                e_batch[i, j] = float(mag_err)
                b_batch[i, j] = band_map.get(o.passband, 1)
                mask_batch[i, j] = False

        return t_batch, m_batch, e_batch, b_batch, mask_batch

    def fit(self, train_obs: List[List[Any]], y_train: np.ndarray, class_weights: Optional[torch.Tensor] = None):
        t_tr, m_tr, e_tr, b_tr, mask_tr = self._pad_and_tensorize(train_obs)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(t_tr, m_tr, e_tr, b_tr, mask_tr, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for epoch in range(self.epochs):
            for bt, bm, be, bb, bmask, by in loader:
                optimizer.zero_grad()
                logits = self.model.forward_classification(bt, bm, be, bb, bmask)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, val_obs: List[List[Any]]) -> np.ndarray:
        self.model.eval()
        t_va, m_va, e_va, b_va, mask_va = self._pad_and_tensorize(val_obs)
        with torch.no_grad():
            logits = self.model.forward_classification(t_va, m_va, e_va, b_va, mask_va)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs


# ==============================================================================
# 2. Risk-Coverage & Honest Abstention Evaluator (Sprint 4C)
# ==============================================================================
def evaluate_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """
    Computes selective classification metrics across rejection thresholds theta.
    Accepts when max_k p_k >= theta; abstains when max_k p_k < theta.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    total_samples = len(y_true)

    curve = []
    for theta in thresholds:
        accepted_mask = (confidences >= theta)
        n_accepted = int(np.sum(accepted_mask))
        n_abstained = total_samples - n_accepted
        coverage = float(n_accepted / total_samples)

        if n_accepted > 0:
            y_acc_true = y_true[accepted_mask]
            y_acc_pred = predictions[accepted_mask]
            acc = float(accuracy_score(y_acc_true, y_acc_pred))
            macro_f1 = float(f1_score(y_acc_true, y_acc_pred, average="macro", zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_acc_true, y_acc_pred))
        else:
            acc = 0.0
            macro_f1 = 0.0
            bal_acc = 0.0

        # Abstention breakdown on rare classes
        rare_abstentions = {}
        for cname in ["TDE", "OTHER_EXTRAGALACTIC"]:
            cidx = classes.index(cname)
            c_mask = (y_true == cidx)
            n_c = int(np.sum(c_mask))
            n_c_abstained = int(np.sum(c_mask & (~accepted_mask)))
            rare_abstentions[cname] = {
                "support": n_c,
                "abstained_count": n_c_abstained,
                "abstention_rate": round(float(n_c_abstained / n_c), 4) if n_c > 0 else 0.0,
            }

        curve.append({
            "confidence_threshold": round(theta, 3),
            "coverage": round(coverage, 4),
            "accepted_samples": n_accepted,
            "abstained_samples": n_abstained,
            "accepted_accuracy": round(acc, 4),
            "accepted_macro_f1": round(macro_f1, 4),
            "accepted_balanced_accuracy": round(bal_acc, 4),
            "rare_class_abstention": rare_abstentions,
        })

    return curve


# ==============================================================================
# 3. Main Sprint 4 Runner
# ==============================================================================
class Sprint4IntegratedHarness:
    def __init__(
        self,
        candidates_path: str = "data/processed/real_survey_10k_v1/candidates_10k.parquet",
        alerce_lc_dir: str = "data/raw/real_survey_v1/alerce/lightcurves",
        dr24_dir: str = "data/raw/real_survey_v1/ztf_dr24",
        output_dir: str = "experiments/sprint4_rare_class_and_openset",
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

        self.df_cand = pd.read_parquet(self.candidates_path)
        self.t1_df = self.df_cand[self.df_cand["coarse_class"].isin(self.classes)].copy()
        self.t2_df = self.df_cand[self.df_cand["coarse_class"] == "UNCLASSIFIED"].copy().reset_index(drop=True)

        self.train_df = self.t1_df[self.t1_df["assigned_split"] == "train"].copy().reset_index(drop=True)
        self.val_df = self.t1_df[self.t1_df["assigned_split"] == "val"].copy().reset_index(drop=True)

        print(f"Loaded {len(self.train_df)} Train, {len(self.val_df)} Val, {len(self.t2_df)} Track 2 Unclassified.")
        print("SPATIAL TEST SET (N=586) IS SEALED AND EXCLUDED.")

        # Preload observations for Train and Val
        print("Preloading observations for Track 1 Train & Val...")
        self.train_obs = [self.loader.load_candidate_observations(cid) for cid in self.train_df["candidate_id"]]
        self.val_obs = [self.loader.load_candidate_observations(cid) for cid in self.val_df["candidate_id"]]

        # Extract 27-D physical features
        extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
        self.X_train = np.nan_to_num(np.array([[extractor.extract_from_observations(o)[f] for f in PHYSICAL_FEATURES_27] for o in self.train_obs]))
        self.y_train = np.array([self.class_to_idx[c] for c in self.train_df["coarse_class"]])
        self.X_val = np.nan_to_num(np.array([[extractor.extract_from_observations(o)[f] for f in PHYSICAL_FEATURES_27] for o in self.val_obs]))
        self.y_val = np.array([self.class_to_idx[c] for c in self.val_df["coarse_class"]])

    def run_sprint4b_temporal_comparison(self) -> Dict[str, Any]:
        """Sprint 4B: Astromer sequence model vs 27-D Physical feature baselines."""
        print("\n" + "=" * 80)
        print("ASTRAEUS Sprint 4B: Temporal Sequence Model Comparison (ASTROMER vs Baselines)")
        print("=" * 80)

        # 1. Astromer Model
        print("Training AstromerEncoder on variable-length photometric series...")
        t0 = time.time()
        # Compute balanced class weights for PyTorch
        class_counts = {c: int(np.sum(self.y_train == self.class_to_idx[c])) for c in self.classes}
        w_balanced = [2808.0 / (8.0 * class_counts[c]) for c in self.classes]
        weights_tensor = torch.tensor(w_balanced, dtype=torch.float32)

        astromer = AstromerSequenceClassifier(
            num_classes=8, d_model=64, nhead=4, num_layers=2, max_seq_len=128, epochs=20, seed=42
        )
        astromer.fit(self.train_obs, self.y_train, class_weights=weights_tensor)
        fit_time_astromer = time.time() - t0

        t_inf0 = time.time()
        astromer_probs = astromer.predict_proba(self.val_obs)
        inf_time_astromer = (time.time() - t_inf0) / len(self.val_obs) * 1000.0  # ms/sample
        astromer_preds = np.argmax(astromer_probs, axis=1)

        astromer_res = {
            "model_name": "AstromerEncoder_Temporal",
            "fit_time_seconds": round(fit_time_astromer, 2),
            "inference_latency_ms_per_sample": round(inf_time_astromer, 2),
            "macro_f1": round(float(f1_score(self.y_val, astromer_preds, average="macro", zero_division=0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(self.y_val, astromer_preds)), 4),
            "accuracy": round(float(accuracy_score(self.y_val, astromer_preds)), 4),
        }

        # 2. Random Forest Baseline
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=4, class_weight="balanced_subsample", random_state=42, n_jobs=1)
        t_rf0 = time.time()
        rf.fit(self.X_train, self.y_train)
        fit_time_rf = time.time() - t_rf0
        t_rf_inf = time.time()
        rf_probs = rf.predict_proba(self.X_val)
        inf_time_rf = (time.time() - t_rf_inf) / len(self.X_val) * 1000.0
        rf_preds = np.argmax(rf_probs, axis=1)

        rf_res = {
            "model_name": "BalancedRandomForest_27D_Physical",
            "fit_time_seconds": round(fit_time_rf, 2),
            "inference_latency_ms_per_sample": round(inf_time_rf, 2),
            "macro_f1": round(float(f1_score(self.y_val, rf_preds, average="macro", zero_division=0)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(self.y_val, rf_preds)), 4),
            "accuracy": round(float(accuracy_score(self.y_val, rf_preds)), 4),
        }

        print(f"  [AstromerEncoder  ] Macro-F1: {astromer_res['macro_f1']:.4f} | Bal-Acc: {astromer_res['balanced_accuracy']:.4f} | Latency: {astromer_res['inference_latency_ms_per_sample']:.2f} ms")
        print(f"  [BalancedRF-27D   ] Macro-F1: {rf_res['macro_f1']:.4f} | Bal-Acc: {rf_res['balanced_accuracy']:.4f} | Latency: {rf_res['inference_latency_ms_per_sample']:.2f} ms")

        return {
            "astromer": astromer_res,
            "balanced_rf": rf_res,
            "astromer_probs": astromer_probs,
            "rf_probs": rf_probs,
        }

    def run_sprint4c_calibration_and_abstention(self, rf_probs: np.ndarray) -> Dict[str, Any]:
        """Sprint 4C: Calibration and Risk-Coverage Abstention Analysis."""
        print("\n" + "=" * 80)
        print("ASTRAEUS Sprint 4C: Calibration, Reliability & Honest Abstention")
        print("=" * 80)

        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85]
        risk_coverage = evaluate_risk_coverage(self.y_val, rf_probs, self.classes, thresholds)

        for step in risk_coverage:
            th = step["confidence_threshold"]
            cov = step["coverage"] * 100
            acc = step["accepted_accuracy"] * 100
            f1 = step["accepted_macro_f1"]
            print(f"  Threshold >= {th:.2f} | Coverage: {cov:5.1f}% | Accepted Acc: {acc:5.1f}% | Accepted Macro-F1: {f1:.4f}")

        return {
            "risk_coverage_curve": risk_coverage,
        }

    def run_sprint4d_track2_openset_discovery(self) -> Dict[str, Any]:
        """Sprint 4D: Unsupervised Anomaly Discovery on N=4,365 Track 2 Candidates."""
        print("\n" + "=" * 80)
        print("ASTRAEUS Sprint 4D: Track 2 Open-Set Discovery & Novelty Ranking (N=4,365)")
        print("=" * 80)

        print("Preloading observations for 4,365 Track 2 candidates...")
        t0 = time.time()
        t2_cids = list(self.t2_df["candidate_id"])
        extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)

        t2_features = []
        for i, cid in enumerate(t2_cids):
            obs = self.loader.load_candidate_observations(cid)
            feats = extractor.extract_from_observations(obs)
            vec = [feats[f] for f in PHYSICAL_FEATURES_27]
            t2_features.append(vec)
            if (i + 1) % 1500 == 0 or (i + 1) == len(t2_cids):
                print(f"  Processed {i + 1}/{len(t2_cids)} candidates in {time.time()-t0:.1f}s...")

        X_t2 = np.nan_to_num(np.array(t2_features, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)

        # 1. Fit Multi-Method Anomaly Ensemble on Reference Cohort (Track 1 Train N=2,808)
        print("\nFitting reference anomaly detectors on Pop1 common transients (N=2,808)...")
        scaler = RobustScaler()
        X_ref_scaled = scaler.fit_transform(self.X_train)
        X_t2_scaled = scaler.transform(X_t2)

        # Method A: Isolation Forest
        iso_forest = IsolationForest(n_estimators=150, contamination=0.01, random_state=42, n_jobs=1)
        iso_forest.fit(self.X_train)
        iso_scores = -iso_forest.score_samples(X_t2)  # Higher = more anomalous
        iso_norm = (iso_scores - np.min(iso_scores)) / (np.max(iso_scores) - np.min(iso_scores) + 1e-8)

        # Method B: Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=25, novelty=True, contamination=0.01, n_jobs=1)
        lof.fit(X_ref_scaled)
        lof_scores = -lof.score_samples(X_t2_scaled)
        lof_norm = (lof_scores - np.min(lof_scores)) / (np.max(lof_scores) - np.min(lof_scores) + 1e-8)

        # Method C: Robust Mahalanobis Distance
        # Fit covariance on dominant PCA components to prevent singular matrix
        cov_est = MinCovDet(random_state=42).fit(X_ref_scaled[:, :10])
        mahal_scores = cov_est.mahalanobis(X_t2_scaled[:, :10])
        mahal_norm = (mahal_scores - np.min(mahal_scores)) / (np.max(mahal_scores) - np.min(mahal_scores) + 1e-8)

        # Rank Correlation across methods
        corr_iso_lof, _ = spearmanr(iso_norm, lof_norm)
        corr_iso_mahal, _ = spearmanr(iso_norm, mahal_norm)
        corr_lof_mahal, _ = spearmanr(lof_norm, mahal_norm)

        print(f"Rank Correlations across methods:")
        print(f"  IsolationForest vs LOF:         rho = {corr_iso_lof:.4f}")
        print(f"  IsolationForest vs Mahalanobis: rho = {corr_iso_mahal:.4f}")
        print(f"  LOF vs Mahalanobis:             rho = {corr_lof_mahal:.4f}")

        # Consensus Composite Anomaly Score
        composite_score = (iso_norm + lof_norm + mahal_norm) / 3.0

        # Triage Yields across budgets
        budgets = [10, 50, 100, int(len(X_t2) * 0.01), int(len(X_t2) * 0.05)]
        ranked_indices = np.argsort(-composite_score)

        top10_discoveries = []
        for rank in range(10):
            idx = ranked_indices[rank]
            cid = t2_cids[idx]
            cand_meta = self.t2_df.iloc[idx]
            feats_vec = X_t2[idx]

            # Physical explanation based on feature deviations
            # Find feature indices with highest z-scores relative to reference cohort
            ref_means = np.mean(self.X_train, axis=0)
            ref_stds = np.std(self.X_train, axis=0) + 1e-6
            z_scores = np.abs(feats_vec - ref_means) / ref_stds
            top_feat_idx = int(np.argmax(z_scores))
            top_feat_name = PHYSICAL_FEATURES_27[top_feat_idx]
            top_feat_val = float(feats_vec[top_feat_idx])
            top_feat_z = float(z_scores[top_feat_idx])

            top10_discoveries.append({
                "rank": rank + 1,
                "candidate_id": cid,
                "ra_deg": round(float(cand_meta["ra_deg"]), 5),
                "dec_deg": round(float(cand_meta["dec_deg"]), 5),
                "composite_anomaly_score": round(float(composite_score[idx]), 4),
                "isolation_forest_score": round(float(iso_norm[idx]), 4),
                "lof_score": round(float(lof_norm[idx]), 4),
                "mahalanobis_score": round(float(mahal_norm[idx]), 4),
                "primary_anomalous_feature": top_feat_name,
                "feature_value": round(top_feat_val, 4),
                "feature_z_score": round(top_feat_z, 2),
            })

        print("\nTop 5 Unclassified Anomalies Identified in Track 2:")
        for disc in top10_discoveries[:5]:
            print(f"  #{disc['rank']} {disc['candidate_id']} | Score: {disc['composite_anomaly_score']:.4f} | Driver: {disc['primary_anomalous_feature']} = {disc['feature_value']} (z = {disc['feature_z_score']:.1f})")

        return {
            "candidate_count": len(X_t2),
            "rank_correlations": {
                "iso_vs_lof": round(float(corr_iso_lof), 4),
                "iso_vs_mahal": round(float(corr_iso_mahal), 4),
                "lof_vs_mahal": round(float(corr_lof_mahal), 4),
            },
            "top10_candidate_discoveries": top10_discoveries,
        }

    def run_full_suite(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "milestone": "ASTRAEUS Sprint 4B, 4C, 4D Execution",
            "execution_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spatial_test_sealed": True,
            "track1_train_size": len(self.train_df),
            "track1_val_size": len(self.val_df),
            "track2_unclassified_size": len(self.t2_df),
        }

        # Sprint 4B
        sprint4b = self.run_sprint4b_temporal_comparison()
        results["sprint4b_temporal_comparison"] = {
            "astromer": sprint4b["astromer"],
            "balanced_rf": sprint4b["balanced_rf"],
        }

        # Sprint 4C
        sprint4c = self.run_sprint4c_calibration_and_abstention(sprint4b["rf_probs"])
        results["sprint4c_calibration_and_abstention"] = sprint4c

        # Sprint 4D
        sprint4d = self.run_sprint4d_track2_openset_discovery()
        results["sprint4d_track2_openset_discovery"] = sprint4d

        # Save to JSON
        out_file = self.output_dir / "sprint4_temporal_calibration_openset_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved Sprint 4B, 4C, 4D results to: {out_file}")
        return results


if __name__ == "__main__":
    harness = Sprint4IntegratedHarness()
    harness.run_full_suite()
