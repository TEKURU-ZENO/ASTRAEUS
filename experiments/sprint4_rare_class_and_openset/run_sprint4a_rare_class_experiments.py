"""
ASTRAEUS Sprint 4A: Controlled Rare-Class Imbalance Experiments & Trade-Off Analysis.

Scientific Governance Contract:
- Training strictly on Track 1 Training partition (N = 2,808).
- Model selection, hyperparameter tuning, and weighting ablations performed via Stratified 5-Fold Cross-Validation on the training partition to prevent validation-set selection bias.
- Final evaluation conducted once on the Track 1 Validation partition (N = 741) as the frozen validation holdout.
- Spatial Test partition (N = 586) remains strictly SEALED.
- Experiments evaluate:
    1. Control Unweighted (RF, HGBT)
    2. Balanced Class Weights (RF, HGBT)
    3. Effective-Number Class Weighting (Cui et al., CVPR 2019, beta in {0.99, 0.999})
    4. Balanced Random Forest (balanced bootstrap per tree)
    5. Focal Loss Classifier (PyTorch MLP with multiclass focal loss, gamma=2.0)
    6. Physical-Feature SMOTE (controlled ablation strictly inside training folds)
- Granular per-class precision, recall, F1, false positive counts, and Wilson 95% CIs.
- Detailed failure analysis on rare classes: TDE (N=24 val) and OTHER_EXTRAGALACTIC (N=18 val).
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 1. Effective Number Class Weights (Cui et al., CVPR 2019)
# ==============================================================================
def compute_effective_num_weights(
    class_counts: Dict[str, int],
    classes: List[str],
    beta: float = 0.99,
) -> Dict[str, float]:
    """
    Computes class weights based on the effective number of samples:
    E_n = (1 - beta^n) / (1 - beta), Weight = 1 / E_n
    Normalized so sum of weights equals number of classes.
    """
    weights = {}
    for c in classes:
        n = class_counts.get(c, 1)
        eff_n = (1.0 - beta**n) / (1.0 - beta)
        weights[c] = 1.0 / max(eff_n, 1e-6)

    mean_w = np.mean([weights[c] for c in classes])
    return {c: float(weights[c] / mean_w) for c in classes}


# ==============================================================================
# 2. Physics-Aware In-Fold SMOTE for Rare Classes
# ==============================================================================
class PhysicalFeatureSMOTE:
    """
    Synthetic Minority Over-sampling strictly for continuous physical features.
    Enforces non-negativity and boundary plausibility constraints.
    Applied strictly within training folds; NEVER across validation splits.
    """

    def __init__(
        self,
        target_classes: List[str],
        class_to_idx: Dict[str, int],
        feature_names: List[str],
        target_ratio: float = 0.5,
        k_neighbors: int = 4,
        random_state: int = 42,
    ):
        self.target_classes = target_classes
        self.class_to_idx = class_to_idx
        self.feature_names = feature_names
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors
        self.rng = np.random.RandomState(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_res = list(X)
        y_res = list(y)

        # Majority class count
        class_counts = {idx: np.sum(y == idx) for idx in np.unique(y)}
        max_count = max(class_counts.values())

        for cname in self.target_classes:
            cidx = self.class_to_idx[cname]
            mask = (y == cidx)
            X_c = X[mask]
            n_c = len(X_c)

            if n_c < 2:
                continue

            target_n = int(max_count * self.target_ratio)
            n_to_generate = max(0, target_n - n_c)
            if n_to_generate <= 0:
                continue

            k = min(self.k_neighbors, n_c - 1)
            nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            nn_model.fit(X_c)
            indices = nn_model.kneighbors(X_c, return_distance=False)

            for _ in range(n_to_generate):
                idx = self.rng.randint(0, n_c)
                neighbor_idx = self.rng.choice(indices[idx, 1:])
                diff = X_c[neighbor_idx] - X_c[idx]
                lam = self.rng.uniform(0.0, 1.0)
                synth = X_c[idx] + lam * diff

                # Enforce physical validity constraints
                for fi, fname in enumerate(self.feature_names):
                    if "days" in fname or "baseline" in fname:
                        synth[fi] = max(0.0, synth[fi])
                    elif "fraction" in fname:
                        synth[fi] = np.clip(synth[fi], 0.0, 1.0)
                    elif "std" in fname:
                        synth[fi] = max(0.0, synth[fi])

                X_res.append(synth)
                y_res.append(cidx)

        return np.array(X_res, dtype=np.float64), np.array(y_res, dtype=np.int64)


# ==============================================================================
# 3. Multiclass Focal Loss PyTorch Classifier
# ==============================================================================
class MulticlassFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1.0 - p_t) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        return loss.mean()


class PyTorchFocalLossClassifier:
    """Multi-layer perceptron trained with multiclass focal loss."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        alpha_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        epochs: int = 60,
        lr: float = 0.003,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        ).to(self.device)

        alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(self.device) if alpha_weights else None
        self.criterion = MulticlassFocalLoss(alpha=alpha_tensor, gamma=gamma)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> PyTorchFocalLossClassifier:
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        self.model.train()

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# ==============================================================================
# 4. Calibration & Metric Helpers
# ==============================================================================
def compute_multiclass_ece(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

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
    n_classes: int = 8,
) -> float:
    n_samples = len(y_true_indices)
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_true_indices] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


# ==============================================================================
# 5. Sprint 4A Experiment Harness
# ==============================================================================
class Sprint4ARareClassHarness:
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

        # Load Cohort Metadata
        self.df_cand = pd.read_parquet(self.candidates_path)
        self.t1_df = self.df_cand[self.df_cand["coarse_class"].isin(self.classes)].copy()

        self.train_df = self.t1_df[self.t1_df["assigned_split"] == "train"].copy()
        self.val_df = self.t1_df[self.t1_df["assigned_split"] == "val"].copy()

        assert len(self.train_df) == TRACK1_SPLIT_COUNTS["train"]
        assert len(self.val_df) == TRACK1_SPLIT_COUNTS["val"]

        print(f"Harness initialized with {len(self.train_df)} train candidates, {len(self.val_df)} val candidates.", flush=True)
        print("SPATIAL TEST SET (N=586) IS SEALED AND EXCLUDED.", flush=True)

        # Preload observations into memory
        self._obs_cache: Dict[str, Any] = {}
        all_cids = list(self.train_df["candidate_id"]) + list(self.val_df["candidate_id"])
        print(f"Preloading observations for {len(all_cids)} candidates...", flush=True)
        t_pre = time.time()
        for idx, cid in enumerate(all_cids):
            self._obs_cache[cid] = self.loader.load_candidate_observations(cid)
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(all_cids):
                print(f"  Preloaded {idx + 1}/{len(all_cids)} candidates in {time.time()-t_pre:.1f}s...", flush=True)

        # Extract features for COMBINED Physical-27
        self.X_train, self.y_train = self._extract_features(self.train_df)
        self.X_val, self.y_val = self._extract_features(self.val_df)
        print(f"Extracted X_train {self.X_train.shape}, X_val {self.X_val.shape}")

        # Compute train class counts
        self.train_counts = {c: int(np.sum(self.y_train == self.class_to_idx[c])) for c in self.classes}
        print(f"Train class distribution: {self.train_counts}")

    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
        X_list = []
        y_list = []
        for _, row in df.iterrows():
            cid = row["candidate_id"]
            label_idx = self.class_to_idx[row["coarse_class"]]
            obs = self._obs_cache.get(cid, [])
            feats_dict = extractor.extract_from_observations(obs)
            vec = [feats_dict[f] for f in PHYSICAL_FEATURES_27]
            X_list.append(vec)
            y_list.append(label_idx)

        X = np.nan_to_num(np.array(X_list, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def _build_model(
        self,
        strategy: str,
        seed: int = 42,
    ) -> Any:
        """Constructs model corresponding to the specified imbalance strategy."""
        if strategy == "Control_Unweighted_RF":
            base = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                class_weight=None, random_state=seed, n_jobs=1
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        elif strategy == "Balanced_Weights_RF":
            base = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                class_weight="balanced", random_state=seed, n_jobs=1
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        elif strategy == "Effective_Number_Beta99_RF":
            w_eff = compute_effective_num_weights(self.train_counts, self.classes, beta=0.99)
            base = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                class_weight={self.class_to_idx[c]: w for c, w in w_eff.items()},
                random_state=seed, n_jobs=1
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        elif strategy == "Effective_Number_Beta999_RF":
            w_eff = compute_effective_num_weights(self.train_counts, self.classes, beta=0.999)
            base = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                class_weight={self.class_to_idx[c]: w for c, w in w_eff.items()},
                random_state=seed, n_jobs=1
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        elif strategy == "Balanced_Random_Forest":
            # Balanced bootstrap subsampling: computes balanced weights per bootstrap sample
            base = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        elif strategy == "Balanced_Weights_HGBT":
            base = HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.08, max_depth=5,
                class_weight="balanced", random_state=seed
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        elif strategy == "Focal_Loss_PyTorch":
            w_eff = compute_effective_num_weights(self.train_counts, self.classes, beta=0.99)
            alpha_list = [w_eff[c] for c in self.classes]
            return PyTorchFocalLossClassifier(
                input_dim=len(PHYSICAL_FEATURES_27),
                num_classes=len(self.classes),
                alpha_weights=alpha_list,
                gamma=2.0,
                epochs=60,
                seed=seed,
            )

        elif strategy == "SMOTE_Physical_Ablation_RF":
            base = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                class_weight="balanced", random_state=seed, n_jobs=1
            )
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def run_stratified_cv_training(
        self,
        strategy: str,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes Stratified 5-Fold Cross-Validation on the 2,808 Training Set only.
        Guarantees that model selection is NOT biased by the validation set.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_macro_f1s = []
        fold_bal_accs = []
        oof_preds = np.zeros(len(self.y_train), dtype=np.int64)
        oof_probs = np.zeros((len(self.y_train), len(self.classes)), dtype=np.float64)

        for fold, (train_idx, val_fold_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_tr, y_tr = self.X_train[train_idx], self.y_train[train_idx]
            X_vf, y_vf = self.X_train[val_fold_idx], self.y_train[val_fold_idx]

            # If SMOTE is requested: apply ONLY inside the training fold
            if "SMOTE" in strategy:
                smote = PhysicalFeatureSMOTE(
                    target_classes=["TDE", "OTHER_EXTRAGALACTIC"],
                    class_to_idx=self.class_to_idx,
                    feature_names=PHYSICAL_FEATURES_27,
                    target_ratio=0.3,
                    random_state=seed + fold,
                )
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

            model = self._build_model(strategy, seed=seed + fold)
            model.fit(X_tr, y_tr)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vf)
                preds = np.argmax(probs, axis=1)
            else:
                preds = model.predict(X_vf)
                probs = np.zeros((len(preds), len(self.classes)))
                probs[np.arange(len(preds)), preds] = 1.0

            oof_preds[val_fold_idx] = preds
            oof_probs[val_fold_idx] = probs

            mf1 = f1_score(y_vf, preds, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(y_vf, preds)
            fold_macro_f1s.append(mf1)
            fold_bal_accs.append(bacc)

        oof_macro_f1 = float(f1_score(self.y_train, oof_preds, average="macro", zero_division=0))
        oof_bal_acc = float(balanced_accuracy_score(self.y_train, oof_preds))

        return {
            "strategy": strategy,
            "cv_folds": n_splits,
            "cv_macro_f1_mean": round(float(np.mean(fold_macro_f1s)), 4),
            "cv_macro_f1_std": round(float(np.std(fold_macro_f1s)), 4),
            "cv_bal_acc_mean": round(float(np.mean(fold_bal_accs)), 4),
            "cv_bal_acc_std": round(float(np.std(fold_bal_accs)), 4),
            "oof_macro_f1": round(oof_macro_f1, 4),
            "oof_bal_acc": round(oof_bal_acc, 4),
        }

    def evaluate_on_validation_holdout(
        self,
        strategy: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Fits on full Train (N=2,808) and evaluates once on frozen Validation Holdout (N=741).
        """
        t0 = time.time()
        X_tr, y_tr = self.X_train, self.y_train

        if "SMOTE" in strategy:
            smote = PhysicalFeatureSMOTE(
                target_classes=["TDE", "OTHER_EXTRAGALACTIC"],
                class_to_idx=self.class_to_idx,
                feature_names=PHYSICAL_FEATURES_27,
                target_ratio=0.3,
                random_state=seed,
            )
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        model = self._build_model(strategy, seed=seed)
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(self.X_val)
            preds = np.argmax(probs, axis=1)
        else:
            preds = model.predict(self.X_val)
            probs = np.zeros((len(preds), len(self.classes)))
            probs[np.arange(len(preds)), preds] = 1.0

        macro_f1 = float(f1_score(self.y_val, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(self.y_val, preds, average="weighted", zero_division=0))
        acc = float(accuracy_score(self.y_val, preds))
        bal_acc = float(balanced_accuracy_score(self.y_val, preds))
        ece = compute_multiclass_ece(self.y_val, probs)
        brier = compute_multiclass_brier(self.y_val, probs)
        cm = confusion_matrix(self.y_val, preds, labels=list(range(len(self.classes)))).tolist()

        # Per-class granular breakdown
        per_class = {}
        for idx, cname in enumerate(self.classes):
            c_true = (self.y_val == idx)
            c_pred = (preds == idx)
            support = int(np.sum(c_true))
            tp = int(np.sum(c_true & c_pred))
            fp = int(np.sum((~c_true) & c_pred))
            fn = int(np.sum(c_true & (~c_pred)))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / support) if support > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            _, p_low, p_high = compute_wilson_score_interval(tp, tp + fp)
            _, r_low, r_high = compute_wilson_score_interval(tp, support)

            per_class[cname] = {
                "support": support,
                "precision": round(prec, 4),
                "precision_ci_95": [round(p_low, 4), round(p_high, 4)],
                "recall": round(rec, 4),
                "recall_ci_95": [round(r_low, 4), round(r_high, 4)],
                "f1": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "is_rare_class": (support < 30),
            }

        return {
            "strategy": strategy,
            "fit_time_seconds": round(fit_time, 2),
            "val_macro_f1": round(macro_f1, 4),
            "val_weighted_f1": round(weighted_f1, 4),
            "val_accuracy": round(acc, 4),
            "val_balanced_accuracy": round(bal_acc, 4),
            "val_ece": round(ece, 4),
            "val_brier_score": round(brier, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
        }

    def run_all_experiments(self) -> Dict[str, Any]:
        strategies = [
            "Control_Unweighted_RF",
            "Balanced_Weights_RF",
            "Effective_Number_Beta99_RF",
            "Effective_Number_Beta999_RF",
            "Balanced_Random_Forest",
            "Balanced_Weights_HGBT",
            "Focal_Loss_PyTorch",
            "SMOTE_Physical_Ablation_RF",
        ]

        print("\n" + "=" * 80)
        print("ASTRAEUS Sprint 4A: Running Controlled Rare-Class Imbalance Experiments")
        print("=" * 80)

        results: Dict[str, Any] = {
            "milestone": "ASTRAEUS Sprint 4A Rare-Class Learning",
            "execution_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spatial_test_sealed": True,
            "train_size": len(self.train_df),
            "val_size": len(self.val_df),
            "taxonomy": self.classes,
            "train_class_distribution": self.train_counts,
            "cross_validation_results": {},
            "validation_holdout_results": {},
            "rare_class_tradeoff_analysis": {},
        }

        # 1. Stratified 5-Fold Cross-Validation on Train
        print("\n--- Phase 1: Stratified 5-Fold CV on Training Set (N=2,808) ---")
        for strat in strategies:
            cv_res = self.run_stratified_cv_training(strat, n_splits=5, seed=42)
            results["cross_validation_results"][strat] = cv_res
            print(f"  [{strat:28s}] CV Macro-F1: {cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f} | Bal-Acc: {cv_res['cv_bal_acc_mean']:.4f}")

        # 2. Validation Holdout Evaluation
        print("\n--- Phase 2: Frozen Validation Holdout Evaluation (N=741) ---")
        for strat in strategies:
            val_res = self.evaluate_on_validation_holdout(strat, seed=42)
            results["validation_holdout_results"][strat] = val_res
            tde_f1 = val_res["per_class"]["TDE"]["f1"]
            oe_f1 = val_res["per_class"]["OTHER_EXTRAGALACTIC"]["f1"]
            print(f"  [{strat:28s}] Val Macro-F1: {val_res['val_macro_f1']:.4f} | Bal-Acc: {val_res['val_balanced_accuracy']:.4f} | TDE F1: {tde_f1:.4f} | Other F1: {oe_f1:.4f} | ECE: {val_res['val_ece']:.4f}")

        # 3. Rare-Class Trade-Off Analysis
        control_acc = results["validation_holdout_results"]["Control_Unweighted_RF"]["val_accuracy"]
        control_f1 = results["validation_holdout_results"]["Control_Unweighted_RF"]["val_macro_f1"]
        for strat in strategies:
            val_res = results["validation_holdout_results"][strat]
            oe_stats = val_res["per_class"]["OTHER_EXTRAGALACTIC"]
            tde_stats = val_res["per_class"]["TDE"]
            results["rare_class_tradeoff_analysis"][strat] = {
                "overall_accuracy_delta": round(val_res["val_accuracy"] - control_acc, 4),
                "macro_f1_delta": round(val_res["val_macro_f1"] - control_f1, 4),
                "other_extragalactic_fp": oe_stats["false_positives"],
                "other_extragalactic_tp": oe_stats["true_positives"],
                "other_extragalactic_f1": oe_stats["f1"],
                "tde_fp": tde_stats["false_positives"],
                "tde_tp": tde_stats["true_positives"],
                "tde_f1": tde_stats["f1"],
            }

        # Save to JSON
        out_file = self.output_dir / "sprint4a_rare_class_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved Sprint 4A experimental results to: {out_file}")
        return results


if __name__ == "__main__":
    harness = Sprint4ARareClassHarness()
    harness.run_all_experiments()
