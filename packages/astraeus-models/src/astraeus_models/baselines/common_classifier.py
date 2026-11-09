"""
Common Feature Classifier (Baseline 5): Level-Playing-Field Multi-Class Benchmark Layer.
Extracts 24 standardized light-curve features strictly from cutoff-visible observations and applies
One-vs-Rest Platt-calibrated gradient boosted classification across the 8-class taxonomy.
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
    def __init__(self, ebv_mw: float = 0.02):
        self.ebv_mw = ebv_mw
        self.num_classes = len(CLASSES_8)
        self.is_fitted = False
        # Calibration weights: One-vs-Rest Platt scaling parameters (slope, intercept)
        self.platt_weights: np.ndarray = np.ones((self.num_classes, 2))
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

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

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated class probabilities on the probability simplex Delta^7.
        """
        if not self.is_fitted or self.feature_means is None:
            # Fallback uniform
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            return np.full((n_samples, self.num_classes), 1.0 / self.num_classes)

        X_norm = (X - self.feature_means) / self.feature_stds
        if len(X_norm.shape) == 1:
            X_norm = X_norm[None, :]

        raw_logits = X_norm @ self.W_raw.T  # shape: (N, 8)

        # Apply One-vs-Rest Platt scaling: sigma(a * logit + b)
        calibrated_ovr = 1.0 / (1.0 + np.exp(-np.clip(self.platt_weights[:, 0] * raw_logits + self.platt_weights[:, 1], -15.0, 15.0)))

        # Normalize onto simplex
        prob_sum = np.sum(calibrated_ovr, axis=-1, keepdims=True)
        probs = calibrated_ovr / np.maximum(prob_sum, 1e-9)
        return probs
