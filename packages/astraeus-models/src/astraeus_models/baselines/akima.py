"""
Baseline 1: Local Piecewise Akima Splines with LOO Uncertainty & Calibrated Extrapolation.
Evaluates Tasks T1 (Reconstruction) and T2 (Forecasting) without future knot leakage.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
from scipy.interpolate import Akima1DInterpolator

from astraeus_data.corpus.sequence import TemporalSequence, ObservationRecord, ObservationType


class AkimaBaselineModel:
    """
    Multi-band Akima spline baseline with Leave-One-Out (LOO) empirical residual-error scaling
    and causal exponential decay extrapolation.
    """
    def __init__(
        self,
        tau_decay: float = 15.0,
        s_extrap_prior: float = 3.0,
        quiescent_flux: float = 0.0,
        kernel_bandwidth: float = 5.0
    ):
        self.tau_decay = tau_decay
        self.s_extrap_prior = s_extrap_prior
        self.quiescent_flux = quiescent_flux
        self.kernel_bandwidth = kernel_bandwidth
        self.models_by_band: Dict[str, Dict[str, Any]] = {}

    def fit_sequence(self, sequence: TemporalSequence) -> "AkimaBaselineModel":
        """
        Fits independent per-passband Akima models using only observations present in the input sequence.
        For causal forecasting, the input sequence MUST be a CausalForecastView (t_available <= T_cutoff).
        """
        self.models_by_band.clear()
        by_band: Dict[str, List[ObservationRecord]] = {}
        for obs in sequence.observations:
            by_band.setdefault(obs.passband, []).append(obs)

        for pb, obs_list in by_band.items():
            # Filter to observations with measured flux (forced photometry or detections)
            valid_pts = [
                obs for obs in obs_list
                if obs.measured_flux_nmgy is not None
            ]
            if not valid_pts:
                continue

            times = np.array([obs.t_obs_bmjd for obs in valid_pts], dtype=np.float64)
            fluxes = np.array([obs.measured_flux_nmgy for obs in valid_pts], dtype=np.float64)
            sigmas = np.array([obs.effective_measurement_sigma_nmgy for obs in valid_pts], dtype=np.float64)

            sort_idx = np.argsort(times)
            t_s = times[sort_idx]
            y_s = fluxes[sort_idx]
            sig_s = sigmas[sort_idx]

            n = len(t_s)
            spline = None
            loo_res = []

            if n >= 4:
                try:
                    spline = Akima1DInterpolator(t_s, y_s)
                except Exception:
                    spline = None

                # Compute Leave-One-Out (LOO) cross-validation errors on knots
                for i in range(n):
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    try:
                        sub_spline = Akima1DInterpolator(t_s[mask], y_s[mask])
                        pred_i = float(sub_spline(t_s[i]))
                        if math.isnan(pred_i):
                            # Linear extrapolation for boundary knots
                            if i == 0:
                                slope = (y_s[mask][1] - y_s[mask][0]) / (t_s[mask][1] - t_s[mask][0])
                                pred_i = y_s[mask][0] + slope * (t_s[0] - t_s[mask][0])
                            else:
                                slope = (y_s[mask][-1] - y_s[mask][-2]) / (t_s[mask][-1] - t_s[mask][-2])
                                pred_i = y_s[mask][-1] + slope * (t_s[-1] - t_s[mask][-1])
                        loo_res.append(float(y_s[i] - pred_i))
                    except Exception:
                        loo_res.append(float(sig_s[i]))
            elif n in (2, 3):
                # Linear fallback
                loo_res = [float(s) for s in sig_s]
            else:
                # 1 observation
                loo_res = [float(sig_s[0])]

            self.models_by_band[pb] = {
                "t": t_s,
                "y": y_s,
                "sigma": sig_s,
                "n": n,
                "spline": spline,
                "loo_residuals": np.array(loo_res, dtype=np.float64) if loo_res else np.array([1.0])
            }
        return self

    def predict_band(
        self,
        passband: str,
        query_times: np.ndarray,
        meas_sigmas: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and predictive standard deviation at query_times for a given passband.
        Retains boundary uncertainty into extrapolation.
        """
        q_t = np.asarray(query_times, dtype=np.float64)
        n_queries = len(q_t)

        if passband not in self.models_by_band:
            # Unobserved passband: predict quiescent flux with prior uncertainty
            prior_sig = math.sqrt(self.s_extrap_prior ** 2 + (0.5 ** 2))
            return np.full(n_queries, self.quiescent_flux, dtype=np.float64), np.full(n_queries, prior_sig, dtype=np.float64)

        band_data = self.models_by_band[passband]
        t_knots = band_data["t"]
        y_knots = band_data["y"]
        n_knots = band_data["n"]
        spline = band_data["spline"]
        loo_res = band_data["loo_residuals"]

        t_min = t_knots[0]
        t_max = t_knots[-1]
        last_flux = y_knots[-1]

        means = []
        sigmas = []

        for idx, t in enumerate(q_t):
            meas_sig = float(meas_sigmas[idx]) if meas_sigmas is not None else 0.2

            if t <= t_max:
                if t >= t_min:
                    if spline is not None:
                        mu = float(spline(t))
                        if math.isnan(mu):
                            mu = float(np.interp(t, t_knots, y_knots))
                    else:
                        mu = float(np.interp(t, t_knots, y_knots))
                else:
                    # Extrapolation prior to earliest observation
                    dt = t_min - t
                    mu = self.quiescent_flux + (y_knots[0] - self.quiescent_flux) * math.exp(-dt / self.tau_decay)

                # Local residual error via kernel regression over LOO residuals
                dists = t - t_knots
                weights = np.exp(-(dists ** 2) / (2.0 * (self.kernel_bandwidth ** 2)))
                denom = np.sum(weights)
                if denom > 1e-9:
                    s_local_sq = np.sum(weights * (loo_res ** 2)) / denom
                else:
                    s_local_sq = float(np.mean(loo_res ** 2))
                var_total = s_local_sq + (meas_sig ** 2)
            else:
                # Causal future extrapolation
                dt = t - t_max
                decay_factor = math.exp(-dt / self.tau_decay)
                mu = self.quiescent_flux + (last_flux - self.quiescent_flux) * decay_factor

                # Retain boundary uncertainty + expand prior extrapolation variance
                dists_edge = t_max - t_knots
                weights_edge = np.exp(-(dists_edge ** 2) / (2.0 * (self.kernel_bandwidth ** 2)))
                denom_edge = np.sum(weights_edge)
                s_edge_sq = np.sum(weights_edge * (loo_res ** 2)) / max(denom_edge, 1e-9)

                var_total = s_edge_sq + (self.s_extrap_prior ** 2) * (1.0 - decay_factor) + (meas_sig ** 2)

            means.append(mu)
            sigmas.append(math.sqrt(max(var_total, 1e-6)))

        return np.array(means, dtype=np.float64), np.array(sigmas, dtype=np.float64)

    @staticmethod
    def evaluate_crps(y_true: float, mu_pred: float, sigma_pred: float) -> float:
        """Exact closed-form Gaussian CRPS."""
        if sigma_pred <= 1e-9:
            return abs(y_true - mu_pred)
        z = (y_true - mu_pred) / sigma_pred
        phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        Phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        crps = sigma_pred * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - (1.0 / math.sqrt(math.pi)))
        return float(crps)
