"""
Common Feature Classifier (Baseline 5): Level-Playing-Field Multi-Class Benchmark Layer.
Extracts 24 standardized light-curve features strictly from cutoff-visible observations and applies
1-parameter global temperature-scaled OvR Ridge classification across the 8-class taxonomy
(with 16-parameter OvR Platt retired from default use and retained only as a legacy diagnostic),
plus an in-domain feature-distance/entropy OOD abstention module.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np

from astraeus_data.corpus.sequence import TemporalSequence, ObservationRecord

PASSBANDS_ALL_8 = ["u", "g", "r", "i", "z", "Y", "c", "o"]
PASSBANDS_PRIMARY_6 = ["u", "g", "r", "i", "z", "Y"]
CLASSES_8 = ["SN_Ia", "SN_II", "SN_Ibc", "SLSN", "TDE", "FBOT", "VARIABLE_STAR", "ARTIFACT_DIPOLE"]

# Galactic extinction coefficients A_lambda / E(B-V) for Fitzpatrick 1999 (R_V = 3.1)
EXTINCTION_COEFFS = {
    "u": 4.858,
    "g": 3.793,
    "r": 2.603,
    "i": 1.954,
    "z": 1.488,
    "Y": 1.263,
    "c": 3.198,  # ATLAS cyan (approx g+r)
    "o": 2.278   # ATLAS orange (approx r+i)
}


class CommonFeatureClassifier:
    """
    Standardized 24-feature extractor and calibrated classifier.
    Features:
    1-8:   Peak flux in all 8 passbands (nmgy)
    9-14:  Rise timescale t_10%->90% in primary 6 passbands (days)
    15-20: Decline timescale t_90%->50% in primary 6 passbands (days)
    21-23: MW-dereddened peak colors: (g-r), (r-i), (c-o)
    24:    Early color slope d(g-r)/dt during early rise (mag/day)
    """
    def __init__(self, ebv_mw: float = 0.02, calibration_mode: str = "temperature_scaled"):
        if calibration_mode not in ("temperature_scaled", "ovr_platt_legacy"):
            raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")
        self.ebv_mw = ebv_mw
        self.calibration_mode = calibration_mode
        self.num_classes = len(CLASSES_8)
        self.is_fitted = False
        # Legacy One-vs-Rest Platt scaling parameters (slope, intercept); retired from default inference
        self.platt_weights: np.ndarray = np.ones((self.num_classes, 2))
        self.global_temperature: float = 1.0
        self.val_size: int = 0
        self.val_per_class_support: Dict[str, int] = {}
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.class_centroids: Optional[np.ndarray] = None
        self.inv_pooled_cov: Optional[np.ndarray] = None
        self.ensemble_weights: List[np.ndarray] = []
        # Post-hoc OOD abstention thresholds calibrated strictly on Dataset A Validation
        self.ood_val_threshold_95: float = 0.0
        self.ood_val_entropy_95: float = 0.0
        self.ood_val_mahalanobis_95: float = 0.0
        self.ood_val_disagreement_95: float = 0.0
        self.ood_val_conformal_95: float = 0.0
        self.val_l2_mean: float = 0.85
        self.val_l2_std: float = 0.35
        self.conformal_q95: float = 1.0

    def extract_features(self, sequence: TemporalSequence) -> np.ndarray:
        """
        Extracts 24 standardized features strictly from observations present in the input sequence.
        If sequence is a CausalForecastView, all features reflect strictly cutoff-visible history!
        """
        feats = np.zeros(24, dtype=np.float64)

        by_band: Dict[str, List[Tuple[float, float]]] = {}
        for obs in sequence.observations:
            if obs.measured_flux_nmgy is not None and obs.passband in PASSBANDS_ALL_8:
                by_band.setdefault(obs.passband, []).append((obs.t_obs_bmjd, obs.measured_flux_nmgy))

        # 1-8: Peak flux per passband (nanomaggies)
        peak_fluxes: Dict[str, float] = {}
        peak_times: Dict[str, float] = {}
        for idx, pb in enumerate(PASSBANDS_ALL_8):
            if pb in by_band and by_band[pb]:
                t_arr = np.array([pt[0] for pt in by_band[pb]])
                f_arr = np.array([pt[1] for pt in by_band[pb]])
                max_idx = np.argmax(f_arr)
                p_flx = float(f_arr[max_idx])
                p_tim = float(t_arr[max_idx])
                peak_fluxes[pb] = p_flx
                peak_times[pb] = p_tim
                feats[idx] = max(p_flx, 0.0)
            else:
                feats[idx] = 0.0

        # 9-14: Rise timescales in 6 primary passbands (days)
        for idx, pb in enumerate(PASSBANDS_PRIMARY_6):
            feat_idx = 8 + idx
            if pb in by_band and len(by_band[pb]) >= 3 and peak_fluxes.get(pb, 0.0) > 0.5:
                t_arr = np.array([pt[0] for pt in by_band[pb]])
                f_arr = np.array([pt[1] for pt in by_band[pb]])
                p_flx = peak_fluxes[pb]
                p_tim = peak_times[pb]

                # Points prior to peak
                pre_mask = t_arr <= p_tim
                if np.sum(pre_mask) >= 2:
                    t_pre = t_arr[pre_mask]
                    f_pre = f_arr[pre_mask]
                    # 10% and 90% crossing times
                    t10 = float(np.interp(0.10 * p_flx, f_pre, t_pre))
                    t90 = float(np.interp(0.90 * p_flx, f_pre, t_pre))
                    t_rise = max(t90 - t10, 0.5)
                    feats[feat_idx] = min(t_rise, 100.0)
                else:
                    feats[feat_idx] = 15.0  # Median default
            else:
                feats[feat_idx] = 15.0      # Median default

        # 15-20: Decline timescales in 6 primary passbands (days)
        for idx, pb in enumerate(PASSBANDS_PRIMARY_6):
            feat_idx = 14 + idx
            if pb in by_band and len(by_band[pb]) >= 3 and peak_fluxes.get(pb, 0.0) > 0.5:
                t_arr = np.array([pt[0] for pt in by_band[pb]])
                f_arr = np.array([pt[1] for pt in by_band[pb]])
                p_flx = peak_fluxes[pb]
                p_tim = peak_times[pb]

                # Points after peak
                post_mask = t_arr >= p_tim
                if np.sum(post_mask) >= 2:
                    t_post = t_arr[post_mask]
                    f_post = f_arr[post_mask]
                    # Decay from 90% to 50%
                    t50 = float(np.interp(-0.50 * p_flx, -f_post, t_post))
                    t_decay = max(t50 - p_tim, 1.0)
                    feats[feat_idx] = min(t_decay, 200.0)
                else:
                    feats[feat_idx] = 30.0  # Median default
            else:
                feats[feat_idx] = 30.0      # Median default

        # 21-23: MW-Dereddened Peak Colors: (g-r), (r-i), (c-o)
        def _calc_dered_color(pb1: str, pb2: str) -> float:
            f1 = peak_fluxes.get(pb1, 0.0)
            f2 = peak_fluxes.get(pb2, 0.0)
            if f1 > 0.1 and f2 > 0.1:
                # Deredden fluxes
                a1 = EXTINCTION_COEFFS[pb1] * self.ebv_mw
                a2 = EXTINCTION_COEFFS[pb2] * self.ebv_mw
                f1_dered = f1 * (10.0 ** (0.4 * a1))
                f2_dered = f2 * (10.0 ** (0.4 * a2))
                return -2.5 * math.log10(f1_dered / f2_dered)
            return 0.0

        feats[20] = _calc_dered_color("g", "r")  # Feature 21
        feats[21] = _calc_dered_color("r", "i")  # Feature 22
        feats[22] = _calc_dered_color("c", "o")  # Feature 23

        # 24: Early Color Evolution Slope d(g-r)/dt (mag/day)
        if "g" in by_band and "r" in by_band and len(by_band["g"]) >= 2 and len(by_band["r"]) >= 2:
            t_g = np.array([pt[0] for pt in by_band["g"]])
            f_g = np.array([pt[1] for pt in by_band["g"]])
            t_r = np.array([pt[0] for pt in by_band["r"]])
            f_r = np.array([pt[1] for pt in by_band["r"]])

            t_start = min(t_g[0], t_r[0])
            t_end = t_start + 10.0

            # Interpolate to common early grid
            t_eval = np.linspace(t_start, t_end, 5)
            f_g_interp = np.interp(t_eval, t_g, f_g)
            f_r_interp = np.interp(t_eval, t_r, f_r)

            valid_color_mask = (f_g_interp > 0.1) & (f_r_interp > 0.1)
            if np.sum(valid_color_mask) >= 2:
                colors = -2.5 * np.log10(f_g_interp[valid_color_mask] / f_r_interp[valid_color_mask])
                time_span = t_eval[valid_color_mask][-1] - t_eval[valid_color_mask][0]
                slope = (colors[-1] - colors[0]) / max(time_span, 1.0)
                feats[23] = max(-1.0, min(1.0, float(slope)))
            else:
                feats[23] = 0.0
        else:
            feats[23] = 0.0

        return feats

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "CommonFeatureClassifier":
        """
        Fits common feature classifier with One-vs-Rest (OvR) Platt scaling calibration on validation split.
        """
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0) + 1e-4

        X_train_norm = (X_train - self.feature_means) / self.feature_stds
        X_val_norm = (X_val - self.feature_means) / self.feature_stds

        # Linear OvR Decision Weights W in R^{8x24}
        np.random.seed(42)
        W_raw = np.zeros((self.num_classes, 24), dtype=np.float64)
        for c in range(self.num_classes):
            y_bin = (y_train == c).astype(np.float64)
            # Regularized least-squares decision vector
            reg_eye = 1.0 * np.eye(24)
            w_c = np.linalg.solve(X_train_norm.T @ X_train_norm + reg_eye, X_train_norm.T @ y_bin)
            W_raw[c] = w_c

        self.W_raw = W_raw

        # Fit Platt scaling parameters on validation set (slope a_c, intercept b_c)
        val_logits = X_val_norm @ self.W_raw.T
        for c in range(self.num_classes):
            val_y_bin = (y_val == c).astype(np.float64)
            z_c = val_logits[:, c]

            # Numerical Platt scaling via logistic loss minimization over grid
            best_a = 1.0
            best_b = 0.0
            min_loss = 1e9
            for a_cand in np.linspace(0.2, 4.0, 25):
                for b_cand in np.linspace(-3.0, 3.0, 25):
                    s = 1.0 / (1.0 + np.exp(-np.clip(a_cand * z_c + b_cand, -15.0, 15.0)))
                    eps = 1e-12
                    bce = -np.mean(val_y_bin * np.log(s + eps) + (1.0 - val_y_bin) * np.log(1.0 - s + eps))
                    if bce < min_loss:
                        min_loss = bce
                        best_a = float(a_cand)
                        best_b = float(b_cand)

            self.platt_weights[c, 0] = best_a
            self.platt_weights[c, 1] = best_b

        self.val_size = int(len(y_val))
        self.val_per_class_support = {
            CLASSES_8[c]: int(np.sum(y_val == c)) for c in range(self.num_classes)
        }

        # 1-parameter global temperature scaling baseline over validation logits
        best_temp = 1.0
        best_nll = 1e9
        for temp in np.linspace(0.05, 5.0, 200):
            scaled = val_logits / temp
            scaled -= np.max(scaled, axis=1, keepdims=True)
            exp_s = np.exp(scaled)
            probs_t = exp_s / np.maximum(np.sum(exp_s, axis=1, keepdims=True), 1e-12)
            nll = -float(np.mean(np.log(np.clip(probs_t[np.arange(len(y_val)), y_val], 1e-12, 1.0))))
            if nll < best_nll:
                best_nll = nll
                best_temp = float(temp)
        self.global_temperature = best_temp
        self.is_fitted = True

        # Fit class centroids & shrinkage-regularized inverse pooled covariance for Mahalanobis OOD detection
        centroids = np.zeros((self.num_classes, 24), dtype=np.float64)
        residuals = np.zeros_like(X_train_norm)
        for c in range(self.num_classes):
            mask_c = y_train == c
            if np.any(mask_c):
                mu_c = np.mean(X_train_norm[mask_c], axis=0)
                centroids[c] = mu_c
                residuals[mask_c] = X_train_norm[mask_c] - mu_c
        self.class_centroids = centroids
        n_tr = max(len(X_train_norm) - self.num_classes, 1)
        emp_cov = (residuals.T @ residuals) / float(n_tr)
        shrinkage = 0.25
        reg_cov = (1.0 - shrinkage) * emp_cov + shrinkage * np.eye(24)
        self.inv_pooled_cov = np.linalg.inv(reg_cov)

        # Fit 20-member bootstrap Ridge ensemble for epistemic disagreement OOD scoring
        rng_ens = np.random.default_rng(109)
        self.ensemble_weights = []
        reg_eye = 1.0 * np.eye(24)
        for _ in range(20):
            b_idx = rng_ens.integers(0, len(X_train_norm), size=len(X_train_norm))
            Xb = X_train_norm[b_idx]
            yb = y_train[b_idx]
            W_b = np.zeros((self.num_classes, 24), dtype=np.float64)
            gram_b = Xb.T @ Xb + reg_eye
            for c in range(self.num_classes):
                y_bin = (yb == c).astype(np.float64)
                W_b[c] = np.linalg.solve(gram_b, Xb.T @ y_bin)
            self.ensemble_weights.append(W_b)

        # Calibrate post-hoc OOD abstention thresholds & split-conformal quantile strictly on Dataset A Validation
        val_scores = self.compute_ood_scores(X_val)
        val_z_rms = val_scores["feature_l2_rms"]
        self.val_l2_mean = float(np.mean(val_z_rms))
        self.val_l2_std = max(float(np.std(val_z_rms)), 0.15)
        self.ood_val_threshold_95 = float(np.percentile(val_z_rms, 95.0))
        self.ood_val_entropy_95 = float(np.percentile(val_scores["predictive_entropy"], 95.0))
        self.ood_val_mahalanobis_95 = float(np.percentile(val_scores["class_conditional_mahalanobis"], 95.0))
        self.ood_val_disagreement_95 = float(np.percentile(val_scores["ensemble_disagreement"], 95.0))
        self.ood_val_conformal_95 = float(np.percentile(val_scores["conformal_nonconformity"], 95.0))

        # Finite-sample split-conformal threshold: ceil((n_val + 1)*(1 - alpha)) / n_val quantile of 1 - p_true
        val_probs_all = self.predict_temp_scaled_proba(X_val)
        n_v = max(len(y_val), 1)
        val_nonconf_true = 1.0 - val_probs_all[np.arange(n_v), y_val]
        q_level = min(1.0, math.ceil((n_v + 1) * 0.95) / float(n_v))
        self.conformal_q95 = float(np.quantile(val_nonconf_true, q_level))

        return self

    def predict_raw_logits(self, X: np.ndarray) -> np.ndarray:
        """Returns uncalibrated OvR ridge decision logits of shape (N, num_classes)."""
        if not self.is_fitted or self.feature_means is None:
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            return np.zeros((n_samples, self.num_classes), dtype=np.float64)
        X_norm = (X - self.feature_means) / self.feature_stds
        if len(X_norm.shape) == 1:
            X_norm = X_norm[None, :]
        return X_norm @ self.W_raw.T

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12)

    def predict_uncalibrated_softmax(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated softmax(logits) without Platt scaling or temperature scaling."""
        return self._softmax(self.predict_raw_logits(X))

    def predict_temp_scaled_proba(self, X: np.ndarray) -> np.ndarray:
        """1-parameter global temperature-scaled softmax(logits / T) (default in-domain baseline)."""
        if not self.is_fitted or self.feature_means is None:
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            return np.full((n_samples, self.num_classes), 1.0 / self.num_classes)
        return self._softmax(self.predict_raw_logits(X) / max(self.global_temperature, 1e-3))

    def predict_legacy_platt_proba(self, X: np.ndarray) -> np.ndarray:
        """
        RETIRED legacy 16-parameter One-vs-Rest Platt scaling + simplex normalization.
        Retained strictly for historical ablation / diagnostic comparison.
        """
        if not self.is_fitted or self.feature_means is None:
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            return np.full((n_samples, self.num_classes), 1.0 / self.num_classes)

        raw_logits = self.predict_raw_logits(X)  # shape: (N, 8)
        calibrated_ovr = 1.0 / (1.0 + np.exp(-np.clip(self.platt_weights[:, 0] * raw_logits + self.platt_weights[:, 1], -15.0, 15.0)))
        prob_sum = np.sum(calibrated_ovr, axis=-1, keepdims=True)
        return calibrated_ovr / np.maximum(prob_sum, 1e-9)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities on the probability simplex Delta^7.
        Defaults to 1-parameter global temperature scaling (retiring 16-parameter OvR Platt);
        explicitly NOT guaranteed calibrated on OOD / cross-survey Track 3 inputs.
        """
        if self.calibration_mode == "ovr_platt_legacy":
            return self.predict_legacy_platt_proba(X)
        return self.predict_temp_scaled_proba(X)

    def compute_ood_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes 5 post-hoc OOD detectors relative to training/validation distributions:
        1. feature_l2_rms: standardized feature RMS-L2 distance from training mean
        2. predictive_entropy: Shannon entropy H(p) of temperature-scaled probabilities
        3. class_conditional_mahalanobis: min_c sqrt((z - mu_c)^T Sigma_reg^{-1} (z - mu_c) / D)
        4. ensemble_disagreement: 20-member bootstrap Ridge BALD mutual information H(mean_p) - mean H(p_m)
        5. conformal_nonconformity: LAC score (1 - max_c p_c)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        n = X_arr.shape[0]
        if not self.is_fitted or self.feature_means is None or self.feature_stds is None:
            return {
                "feature_l2_rms": np.zeros(n, dtype=np.float64),
                "predictive_entropy": np.full(n, math.log(self.num_classes), dtype=np.float64),
                "class_conditional_mahalanobis": np.zeros(n, dtype=np.float64),
                "ensemble_disagreement": np.zeros(n, dtype=np.float64),
                "conformal_nonconformity": np.full(n, 1.0 - 1.0 / self.num_classes, dtype=np.float64),
                "max_prob": np.full(n, 1.0 / self.num_classes, dtype=np.float64),
                "abstain_l2": np.zeros(n, dtype=bool),
            }

        X_norm = (X_arr - self.feature_means) / self.feature_stds
        n_dims = float(max(X_norm.shape[1], 1))
        z_l2_rms = np.linalg.norm(X_norm, axis=1) / math.sqrt(n_dims)

        probs = self.predict_proba(X_arr)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        max_prob = np.max(probs, axis=1)
        conformal_nonconf = 1.0 - max_prob

        # Class-conditional shrinkage Mahalanobis distance (min across classes)
        if self.class_centroids is not None and self.inv_pooled_cov is not None:
            per_class_dists = np.empty((n, self.num_classes), dtype=np.float64)
            for c in range(self.num_classes):
                diff_c = X_norm - self.class_centroids[c]
                quad_c = np.sum((diff_c @ self.inv_pooled_cov) * diff_c, axis=1)
                per_class_dists[:, c] = np.sqrt(np.maximum(quad_c, 0.0) / n_dims)
            maha_min = np.min(per_class_dists, axis=1)
        else:
            maha_min = z_l2_rms.copy()

        # Bootstrap ensemble BALD disagreement: H(mean_p) - mean_m H(p_m)
        if self.ensemble_weights:
            temp = max(self.global_temperature, 1e-3)
            ens_probs = np.empty((len(self.ensemble_weights), n, self.num_classes), dtype=np.float64)
            for m_idx, W_m in enumerate(self.ensemble_weights):
                ens_probs[m_idx] = self._softmax((X_norm @ W_m.T) / temp)
            mean_ens_p = np.mean(ens_probs, axis=0)
            h_mean = -np.sum(mean_ens_p * np.log(np.clip(mean_ens_p, 1e-12, 1.0)), axis=1)
            h_members = -np.sum(ens_probs * np.log(np.clip(ens_probs, 1e-12, 1.0)), axis=2)
            disagreement = np.maximum(h_mean - np.mean(h_members, axis=0), 0.0)
        else:
            disagreement = np.zeros(n, dtype=np.float64)

        abstain_l2 = z_l2_rms > self.ood_val_threshold_95
        return {
            "feature_l2_rms": z_l2_rms,
            "predictive_entropy": entropy,
            "class_conditional_mahalanobis": maha_min,
            "ensemble_disagreement": disagreement,
            "conformal_nonconformity": conformal_nonconf,
            "max_prob": max_prob,
            "abstain_l2": abstain_l2,
        }

    def predict_blind_novelty_score(
        self,
        X: np.ndarray,
        model_probs: Optional[np.ndarray] = None,
    ) -> float:
        """
        Computes a strictly blind observable novelty signal in [0.0, 1.0] for T4-B open-set ranking.
        Uses ZERO object_id or ground-truth class inspection: relies solely on causal observable
        standardized feature anomaly (RMS-L2 relative to Dataset A Val distribution) and normalized
        predictive entropy H(p) / ln(8).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        ood = self.compute_ood_scores(X_arr)
        l2_val = float(ood["feature_l2_rms"][0])
        z_anom = max(0.0, (l2_val - self.val_l2_mean) / max(self.val_l2_std, 0.15))
        s_anom = min(1.0, z_anom / 3.0)

        if model_probs is not None:
            p_arr = np.asarray(model_probs, dtype=np.float64).ravel()
            p_arr = np.clip(p_arr, 1e-12, 1.0)
            p_arr = p_arr / max(float(np.sum(p_arr)), 1e-12)
            ent = -float(np.sum(p_arr * np.log(p_arr)))
        else:
            ent = float(ood["predictive_entropy"][0])
        h_norm = min(1.0, max(0.0, ent / math.log(float(self.num_classes))))

        return float(0.5 * s_anom + 0.5 * h_norm)

    @staticmethod
    def _auroc_and_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
        """Computes exact rank-sum AUROC and FPR@95%TPR (higher score => OOD)."""
        n_id = len(id_scores)
        n_ood = len(ood_scores)
        if n_id == 0 or n_ood == 0:
            return 0.5, 1.0
        # Pairwise comparison for exact AUROC with ties = 0.5
        diff = ood_scores[:, None] - id_scores[None, :]
        auroc = float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)))
        # Threshold giving >=95% TPR on OOD (5th percentile of OOD scores)
        thr_95tpr = float(np.percentile(ood_scores, 5.0))
        fpr95 = float(np.mean(id_scores >= thr_95tpr))
        return round(auroc, 4), round(fpr95, 4)

    def evaluate_ood_abstention(
        self,
        X_id: np.ndarray,
        y_id: np.ndarray,
        X_ood: np.ndarray,
        y_ood_in_tax: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates 5 post-hoc OOD detectors (ID Dataset A Test vs OOD Track 3 Archival),
        risk-versus-coverage curves (100% down to 50% coverage), and selective abstention.
        Explicitly notes finite-sample conformal quantile step size 1/(N_val+1) and clarifies
        that these are diagnostic OOD baselines, NOT a deployed safety gate.
        """
        id_s = self.compute_ood_scores(X_id)
        ood_s = self.compute_ood_scores(X_ood)
        id_probs = self.predict_proba(X_id)
        id_correct = (np.argmax(id_probs, axis=1) == y_id).astype(np.float64)

        detector_keys = [
            "feature_l2_rms",
            "predictive_entropy",
            "class_conditional_mahalanobis",
            "ensemble_disagreement",
            "conformal_nonconformity",
        ]
        val_thresholds = {
            "feature_l2_rms": self.ood_val_threshold_95,
            "predictive_entropy": self.ood_val_entropy_95,
            "class_conditional_mahalanobis": self.ood_val_mahalanobis_95,
            "ensemble_disagreement": self.ood_val_disagreement_95,
            "conformal_nonconformity": self.ood_val_conformal_95,
        }

        detectors_summary: Dict[str, Dict[str, Any]] = {}
        risk_coverage_curves: Dict[str, List[Dict[str, Any]]] = {}
        coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

        for det in detector_keys:
            auroc_d, fpr95_d = self._auroc_and_fpr95(id_s[det], ood_s[det])
            thr_d = val_thresholds[det]
            det_id_abst = float(np.mean(id_s[det] > thr_d))
            det_ood_abst = float(np.mean(ood_s[det] > thr_d))
            detectors_summary[det] = {
                "auroc": auroc_d,
                "fpr_at_95tpr": fpr95_d,
                "val_95_threshold": round(float(thr_d), 4),
                "id_mean": round(float(np.mean(id_s[det])), 4),
                "ood_mean": round(float(np.mean(ood_s[det])), 4),
                "id_abstention_rate_at_val95": round(det_id_abst, 4),
                "ood_abstention_rate_at_val95": round(det_ood_abst, 4),
            }

            # Risk-vs-coverage curve on pooled ID+OOD detection OR ID selective classification risk (1 - accuracy)
            order_id = np.argsort(id_s[det])  # Lowest OOD score retained first
            n_id_tot = len(id_s[det])
            curve_pts = []
            for cov in coverages:
                k_keep = max(1, int(round(cov * n_id_tot)))
                idx_keep = order_id[:k_keep]
                sel_acc = float(np.mean(id_correct[idx_keep]))
                # Also fraction of OOD passing under that same quantile threshold
                tau_cov = float(id_s[det][idx_keep[-1]])
                ood_pass = float(np.mean(ood_s[det] <= tau_cov)) if len(ood_s[det]) > 0 else 0.0
                curve_pts.append({
                    "target_id_coverage": cov,
                    "actual_id_coverage": round(k_keep / max(n_id_tot, 1), 4),
                    "id_selective_risk": round(1.0 - sel_acc, 4),
                    "id_selective_accuracy": round(sel_acc, 4),
                    "ood_false_acceptance_at_tau": round(ood_pass, 4),
                })
            risk_coverage_curves[det] = curve_pts

        keep_id = ~id_s["abstain_l2"]
        keep_ood = ~ood_s["abstain_l2"]
        id_cov = float(np.mean(keep_id))
        id_abst = float(np.mean(id_s["abstain_l2"]))
        ood_abst = float(np.mean(ood_s["abstain_l2"]))
        id_sel_acc = float(np.mean(id_correct[keep_id])) if np.any(keep_id) else 0.0

        ood_sel_acc = None
        if y_ood_in_tax is not None and len(y_ood_in_tax) == len(X_ood):
            ood_probs = self.predict_proba(X_ood)
            ood_correct = (np.argmax(ood_probs, axis=1) == y_ood_in_tax).astype(np.float64)
            ood_sel_acc = round(float(np.mean(ood_correct[keep_ood])), 4) if np.any(keep_ood) else None

        n_val = max(self.val_size, 1)
        return {
            "detector_method": "five_detector_posthoc_ood_benchmark",
            "detectors": detectors_summary,
            "risk_coverage_curves": risk_coverage_curves,
            "threshold_calibration_split": f"DatasetA_Val_N{self.val_size}_95th_percentile",
            "finite_sample_conformal_note": (
                f"Conformal quantile step granularity is 1/(N_val+1) = 1/{n_val + 1} "
                f"({100.0 / (n_val + 1):.2f}%); q95 threshold = {self.conformal_q95:.4f} on 1 - p_true. "
                "Diagnostic baseline only; NOT a certified safety gate."
            ),
            "feature_l2_threshold_95": round(self.ood_val_threshold_95, 4),
            "entropy_threshold_95": round(self.ood_val_entropy_95, 4),
            "mahalanobis_threshold_95": round(self.ood_val_mahalanobis_95, 4),
            "disagreement_threshold_95": round(self.ood_val_disagreement_95, 4),
            "conformal_nonconf_threshold_95": round(self.ood_val_conformal_95, 4),
            "split_conformal_q95_nonconf": round(self.conformal_q95, 4),
            "id_split": "Cohort_1_DatasetA_Test",
            "ood_split": "Cohort_2_Track3_Archival",
            "n_id": int(len(X_id)),
            "n_ood": int(len(X_ood)),
            "id_feature_l2_mean": round(float(np.mean(id_s["feature_l2_rms"])), 4),
            "ood_feature_l2_mean": round(float(np.mean(ood_s["feature_l2_rms"])), 4),
            "id_entropy_mean": round(float(np.mean(id_s["predictive_entropy"])), 4),
            "ood_entropy_mean": round(float(np.mean(ood_s["predictive_entropy"])), 4),
            "auroc_feature_l2": detectors_summary["feature_l2_rms"]["auroc"],
            "fpr_at_95tpr_feature_l2": detectors_summary["feature_l2_rms"]["fpr_at_95tpr"],
            "auroc_predictive_entropy": detectors_summary["predictive_entropy"]["auroc"],
            "fpr_at_95tpr_predictive_entropy": detectors_summary["predictive_entropy"]["fpr_at_95tpr"],
            "auroc_class_conditional_mahalanobis": detectors_summary["class_conditional_mahalanobis"]["auroc"],
            "fpr_at_95tpr_class_conditional_mahalanobis": detectors_summary["class_conditional_mahalanobis"]["fpr_at_95tpr"],
            "auroc_ensemble_disagreement": detectors_summary["ensemble_disagreement"]["auroc"],
            "fpr_at_95tpr_ensemble_disagreement": detectors_summary["ensemble_disagreement"]["fpr_at_95tpr"],
            "auroc_conformal_nonconformity": detectors_summary["conformal_nonconformity"]["auroc"],
            "fpr_at_95tpr_conformal_nonconformity": detectors_summary["conformal_nonconformity"]["fpr_at_95tpr"],
            "id_coverage_at_val95": round(id_cov, 4),
            "id_abstention_rate_at_val95": round(id_abst, 4),
            "id_overall_accuracy": round(float(np.mean(id_correct)), 4),
            "id_selective_accuracy_retained": round(id_sel_acc, 4),
            "ood_abstention_rate_at_val95": round(ood_abst, 4),
            "ood_retained_count": int(np.sum(keep_ood)),
            "ood_selective_accuracy_retained": ood_sel_acc,
            "is_safety_gate": False,
            "ood_calibration_disclaimer": (
                "Global temperature scaling is an in-domain baseline only; Track 3 OOD "
                "probabilities are distribution-shifted and MUST NOT be interpreted as calibrated "
                "or treated as an operational safety gate."
            ),
        }

    @staticmethod
    def _ece_from_arrays(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
        n = max(len(correct), 1)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (confs >= lo) & (confs <= hi if b == n_bins - 1 else confs < hi)
            cnt = int(np.sum(mask))
            if cnt > 0:
                ece += (cnt / n) * abs(float(np.mean(correct[mask])) - float(np.mean(confs[mask])))
        return float(ece)

    @classmethod
    def _calibration_summary(
        cls,
        probs: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        n = len(y_true)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        correct = (preds == y_true).astype(np.float64)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y_true] = 1.0

        per_sample_nll = -np.log(np.clip(probs[np.arange(n), y_true], 1e-12, 1.0))
        per_sample_brier = np.sum((probs - one_hot) ** 2, axis=1)
        nll = float(np.mean(per_sample_nll))
        brier = float(np.mean(per_sample_brier))

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bins = []
        for b in range(n_bins):
            lo, hi = float(edges[b]), float(edges[b + 1])
            mask = (confs >= lo) & (confs <= hi if b == n_bins - 1 else confs < hi)
            cnt = int(np.sum(mask))
            if cnt > 0:
                mean_conf = float(np.mean(confs[mask]))
                emp_acc = float(np.mean(correct[mask]))
                ece += (cnt / max(n, 1)) * abs(emp_acc - mean_conf)
            else:
                mean_conf = 0.0
                emp_acc = 0.0
            bins.append({
                "bin_range": f"[{lo:.1f}, {hi:.1f}{']' if b == n_bins - 1 else ')'}",
                "count": cnt,
                "mean_confidence": round(mean_conf, 4),
                "empirical_accuracy": round(emp_acc, 4),
            })

        # Bootstrap 95% CIs on NLL, Brier, ECE (B=1000)
        rng = np.random.default_rng(seed)
        boot_nll = np.empty(n_boot, dtype=np.float64)
        boot_brier = np.empty(n_boot, dtype=np.float64)
        boot_ece = np.empty(n_boot, dtype=np.float64)
        for b_idx in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_nll[b_idx] = float(np.mean(per_sample_nll[idx]))
            boot_brier[b_idx] = float(np.mean(per_sample_brier[idx]))
            boot_ece[b_idx] = cls._ece_from_arrays(confs[idx], correct[idx], n_bins=n_bins)

        # Class-wise reliability breakdown
        per_class_reliability = {}
        for c_idx, cls_name in enumerate(CLASSES_8):
            y_bin = (y_true == c_idx).astype(np.float64)
            sup = int(np.sum(y_true == c_idx))
            p_c = probs[:, c_idx]
            per_class_reliability[cls_name] = {
                "support": sup,
                "mean_pred_prob": round(float(np.mean(p_c)), 4),
                "empirical_freq": round(float(np.mean(y_bin)), 4),
                "class_brier": round(float(np.mean((p_c - y_bin) ** 2)), 4),
                "mean_prob_on_positives": round(float(np.mean(p_c[y_true == c_idx])), 4) if sup > 0 else None,
            }

        return {
            "accuracy": round(float(np.mean(correct)), 4),
            "nll": round(nll, 4),
            "nll_ci_95": [round(float(np.percentile(boot_nll, 2.5)), 4), round(float(np.percentile(boot_nll, 97.5)), 4)],
            "brier": round(brier, 4),
            "brier_ci_95": [round(float(np.percentile(boot_brier, 2.5)), 4), round(float(np.percentile(boot_brier, 97.5)), 4)],
            "ece": round(float(ece), 4),
            "ece_ci_95": [round(float(np.percentile(boot_ece, 2.5)), 4), round(float(np.percentile(boot_ece, 97.5)), 4)],
            "bootstrap_resamples": n_boot,
            "reliability_bins": bins,
            "per_class_reliability": per_class_reliability,
        }

    def compute_calibration_diagnostics(self, X: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """Compares default 1-param global temperature scaling against retired 16-param OvR Platt and uncalibrated softmax."""
        platt_summary = self._calibration_summary(self.predict_legacy_platt_proba(X), y_true, n_bins=n_bins)
        temp_summary = self._calibration_summary(self.predict_temp_scaled_proba(X), y_true, n_bins=n_bins)
        uncal_summary = self._calibration_summary(self.predict_uncalibrated_softmax(X), y_true, n_bins=n_bins)
        return {
            "n_eval": int(len(y_true)),
            "val_size": self.val_size,
            "val_per_class_support": self.val_per_class_support,
            "default_calibration_mode": self.calibration_mode,
            "ovr_platt_retired_from_default": True,
            "platt_param_count": int(self.platt_weights.size),
            "platt_a_range": [round(float(np.min(self.platt_weights[:, 0])), 4), round(float(np.max(self.platt_weights[:, 0])), 4)],
            "platt_b_range": [round(float(np.min(self.platt_weights[:, 1])), 4), round(float(np.max(self.platt_weights[:, 1])), 4)],
            "global_temperature_T": round(self.global_temperature, 4),
            "global_temp_1param": temp_summary,
            "ovr_platt_16param_legacy": platt_summary,
            "uncalibrated_softmax": uncal_summary,
        }

    def compute_temperature_stability(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_boot: int = 1000,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Quantifies finite-sample sensitivity of 1-parameter global temperature scaling via:
        1. Bootstrap resampling (B=1000) of Dataset A Validation (N=24) holding W_raw fixed.
        2. 5-fold stratified cross-validation across pooled Dataset A Train + Val (N=104).
        """
        rng = np.random.default_rng(seed)
        val_logits = self.predict_raw_logits(X_val)
        n_val = len(y_val)
        temp_grid = np.linspace(0.05, 5.0, 200)

        # Precompute log-probs over grid once or fast vectorized check for B=1000 resamples
        scaled_grid = val_logits[None, :, :] / temp_grid[:, None, None]  # (200, N_val, 8)
        scaled_grid -= np.max(scaled_grid, axis=2, keepdims=True)
        exp_grid = np.exp(scaled_grid)
        probs_grid = exp_grid / np.maximum(np.sum(exp_grid, axis=2, keepdims=True), 1e-12)
        nll_per_sample = -np.log(np.clip(probs_grid[:, np.arange(n_val), y_val], 1e-12, 1.0))  # (200, N_val)

        boot_temps = np.empty(n_boot, dtype=np.float64)
        for b_idx in range(n_boot):
            idx = rng.integers(0, n_val, size=n_val)
            mean_nll_b = np.mean(nll_per_sample[:, idx], axis=1)
            boot_temps[b_idx] = float(temp_grid[int(np.argmin(mean_nll_b))])

        # 5-fold stratified CV on pooled Train + Val
        X_pool = np.vstack([X_train, X_val])
        y_pool = np.concatenate([y_train, y_val])
        fold_ids = np.zeros(len(y_pool), dtype=int)
        for c in range(self.num_classes):
            c_idx = np.where(y_pool == c)[0]
            rng.shuffle(c_idx)
            for pos, orig_i in enumerate(c_idx):
                fold_ids[orig_i] = pos % n_splits

        fold_temps: List[float] = []
        fold_metrics: List[Dict[str, Any]] = []
        oof_probs = np.zeros((len(y_pool), self.num_classes), dtype=np.float64)

        for k in range(n_splits):
            tr_mask = fold_ids != k
            te_mask = fold_ids == k
            clf_k = CommonFeatureClassifier(ebv_mw=self.ebv_mw, calibration_mode="temperature_scaled")
            clf_k.fit(X_pool[tr_mask], y_pool[tr_mask], X_pool[te_mask], y_pool[te_mask])
            p_k = clf_k.predict_temp_scaled_proba(X_pool[te_mask])
            oof_probs[te_mask] = p_k
            confs_k = np.max(p_k, axis=1)
            corr_k = (np.argmax(p_k, axis=1) == y_pool[te_mask]).astype(np.float64)
            nll_k = float(np.mean(-np.log(np.clip(p_k[np.arange(len(y_pool[te_mask])), y_pool[te_mask]], 1e-12, 1.0))))
            fold_temps.append(round(clf_k.global_temperature, 4))
            fold_metrics.append({
                "fold": k + 1,
                "n_train": int(np.sum(tr_mask)),
                "n_val": int(np.sum(te_mask)),
                "temperature_T": round(clf_k.global_temperature, 4),
                "accuracy": round(float(np.mean(corr_k)), 4),
                "nll": round(nll_k, 4),
                "ece": round(self._ece_from_arrays(confs_k, corr_k, n_bins=10), 4),
            })

        oof_confs = np.max(oof_probs, axis=1)
        oof_corr = (np.argmax(oof_probs, axis=1) == y_pool).astype(np.float64)
        oof_one_hot = np.zeros_like(oof_probs)
        oof_one_hot[np.arange(len(y_pool)), y_pool] = 1.0
        oof_nll = float(np.mean(-np.log(np.clip(oof_probs[np.arange(len(y_pool)), y_pool], 1e-12, 1.0))))
        oof_brier = float(np.mean(np.sum((oof_probs - oof_one_hot) ** 2, axis=1)))
        oof_ece = self._ece_from_arrays(oof_confs, oof_corr, n_bins=10)

        return {
            "bootstrap_val_n24": {
                "n_val": int(n_val),
                "bootstrap_resamples": n_boot,
                "fitted_T": round(float(self.global_temperature), 4),
                "boot_mean_T": round(float(np.mean(boot_temps)), 4),
                "boot_median_T": round(float(np.median(boot_temps)), 4),
                "boot_std_T": round(float(np.std(boot_temps)), 4),
                "boot_iqr_T": [
                    round(float(np.percentile(boot_temps, 25.0)), 4),
                    round(float(np.percentile(boot_temps, 75.0)), 4),
                ],
                "boot_ci_95_T": [
                    round(float(np.percentile(boot_temps, 2.5)), 4),
                    round(float(np.percentile(boot_temps, 97.5)), 4),
                ],
            },
            "stratified_5fold_cv_pool_n104": {
                "n_total": int(len(y_pool)),
                "n_splits": n_splits,
                "fold_temperatures": fold_temps,
                "mean_T": round(float(np.mean(fold_temps)), 4),
                "std_T": round(float(np.std(fold_temps)), 4),
                "range_T": [round(float(min(fold_temps)), 4), round(float(max(fold_temps)), 4)],
                "oof_accuracy": round(float(np.mean(oof_corr)), 4),
                "oof_nll": round(oof_nll, 4),
                "oof_brier": round(oof_brier, 4),
                "oof_ece": round(oof_ece, 4),
                "folds": fold_metrics,
            },
        }
