"""
Baseline 2: Heteroscedastic Multi-Output Gaussian Process with Rank-3 Coregionalization.
Implements the 8-passband Intrinsic Coregionalization Model (ICM) with Matérn-3/2 kernel.
"""

from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from astraeus_data.corpus.sequence import TemporalSequence, ObservationRecord, ObservationType

PASSBAND_ORDER = ["u", "g", "r", "i", "z", "Y", "c", "o"]
PASSBAND_TO_INDEX = {pb: idx for idx, pb in enumerate(PASSBAND_ORDER)}


class MultiOutputGPBaseline:
    """
    Multi-Output Gaussian Process for irregular multi-band light curve regression.
    Parameterized via rank-3 coregionalization matrix B in R^{8x8}:
    B = W W^T + diag(kappa), with W in R^{8x3}, kappa in R^{8+}.
    """
    def __init__(
        self,
        lengthscale: float = 12.0,
        signal_var: float = 1.0,
        rank: int = 3
    ):
        self.lengthscale = max(lengthscale, 1.0)
        self.signal_var = max(signal_var, 0.1)
        self.rank = rank
        self.num_passbands = len(PASSBAND_ORDER)

        # Initialize rank-3 W matrix with physical cross-band correlation priors
        np.random.seed(42)
        self.W = np.random.randn(self.num_passbands, self.rank) * 0.3
        # Enhance adjacent filter correlations (g with r, r with i, c with g+r, o with r+i)
        for i in range(self.num_passbands):
            self.W[i, 0] = 0.8  # Shared achromatic luminosity component
        self.kappa = np.full(self.num_passbands, 0.1, dtype=np.float64)

        # Observation cache
        self.t_train: np.ndarray = np.array([], dtype=np.float64)
        self.p_train: np.ndarray = np.array([], dtype=np.int32)
        self.y_train: np.ndarray = np.array([], dtype=np.float64)
        self.var_meas_train: np.ndarray = np.array([], dtype=np.float64)
        self.L_chol: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None

    def get_coregionalization_matrix(self) -> np.ndarray:
        """Computes PSD coregionalization matrix B = W W^T + diag(kappa)."""
        B = self.W @ self.W.T + np.diag(np.abs(self.kappa) + 1e-5)
        return B

    def matern32_kernel(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Stationary Matérn-3/2 temporal kernel."""
        # Pairwise absolute temporal distance matrix
        dists = np.abs(t1[:, None] - t2[None, :])
        sqrt3_d_over_l = math.sqrt(3.0) * dists / self.lengthscale
        k = self.signal_var * (1.0 + sqrt3_d_over_l) * np.exp(-sqrt3_d_over_l)
        return k

    def fit_sequence(self, sequence: TemporalSequence) -> "MultiOutputGPBaseline":
        """
        Fits multi-output GP using measured-flux observations in the input sequence.
        Censored upper limits lacking measured flux are excluded from conditioning.
        """
        valid_obs = [
            obs for obs in sequence.observations
            if obs.measured_flux_nmgy is not None and obs.passband in PASSBAND_TO_INDEX
        ]
        if not valid_obs:
            self.t_train = np.array([], dtype=np.float64)
            self.L_chol = None
            return self

        times = [obs.t_obs_bmjd for obs in valid_obs]
        p_indices = [PASSBAND_TO_INDEX[obs.passband] for obs in valid_obs]
        fluxes = [obs.measured_flux_nmgy for obs in valid_obs]
        meas_vars = [obs.effective_measurement_sigma_nmgy ** 2 for obs in valid_obs]

        self.t_train = np.array(times, dtype=np.float64)
        self.p_train = np.array(p_indices, dtype=np.int32)
        self.y_train = np.array(fluxes, dtype=np.float64)
        self.var_meas_train = np.array(meas_vars, dtype=np.float64)

        n = len(self.t_train)
        B = self.get_coregionalization_matrix()
        K_time = self.matern32_kernel(self.t_train, self.t_train)

        # Coregionalized covariance: K_ij = B[p_i, p_j] * K_time[i, j]
        B_sub = B[self.p_train[:, None], self.p_train[None, :]]
        K_total = B_sub * K_time + np.diag(self.var_meas_train + 1e-4)

        # Cholesky decomposition
        try:
            self.L_chol = np.linalg.cholesky(K_total)
            # alpha = K^{-1} y
            self.alpha = np.linalg.solve(self.L_chol.T, np.linalg.solve(self.L_chol, self.y_train))
        except np.linalg.LinAlgError:
            # Fallback with jitter
            K_total += np.eye(n) * 1e-2
            self.L_chol = np.linalg.cholesky(K_total)
            self.alpha = np.linalg.solve(self.L_chol.T, np.linalg.solve(self.L_chol, self.y_train))

        return self

    def predict_band(
        self,
        passband: str,
        query_times: np.ndarray,
        meas_sigmas: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes closed-form Gaussian posterior mean and total standard deviation
        at query_times for a given passband.
        """
        q_t = np.asarray(query_times, dtype=np.float64)
        m = len(q_t)

        if self.L_chol is None or len(self.t_train) == 0:
            # Fallback to prior
            prior_sig = math.sqrt(self.signal_var + 0.2 ** 2)
            return np.zeros(m, dtype=np.float64), np.full(m, prior_sig, dtype=np.float64)

        p_idx = PASSBAND_TO_INDEX.get(passband, 1)  # Default to 'g' if unknown
        B = self.get_coregionalization_matrix()

        # Cross-covariance k_* between query points and training points
        K_time_star = self.matern32_kernel(q_t, self.t_train)
        B_cross = B[p_idx, self.p_train][None, :]  # shape: (1, n_train)
        K_star = K_time_star * B_cross              # shape: (m, n_train)

        # Posterior mean: mu_* = K_* alpha
        mu_star = K_star @ self.alpha

        # Posterior variance: var_* = k(t_*, t_*) - v^T v + sigma_meas^2
        # where v = L^{-1} K_*^T
        v = np.linalg.solve(self.L_chol, K_star.T)  # shape: (n_train, m)
        v_sq = np.sum(v ** 2, axis=0)                # shape: (m,)

        prior_var_diag = B[p_idx, p_idx] * self.signal_var
        post_var = prior_var_diag - v_sq

        # Add target measurement variance
        meas_var_q = (np.asarray(meas_sigmas, dtype=np.float64) ** 2) if meas_sigmas is not None else np.full(m, 0.04)
        total_var = np.maximum(post_var + meas_var_q, 1e-6)
        total_sigma = np.sqrt(total_var)

        return mu_star, total_sigma

    @staticmethod
    def evaluate_crps(y_true: float, mu_pred: float, sigma_pred: float) -> float:
        """Exact analytical Gaussian CRPS."""
        if sigma_pred <= 1e-9:
            return abs(y_true - mu_pred)
        z = (y_true - mu_pred) / sigma_pred
        phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        Phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        crps = sigma_pred * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - (1.0 / math.sqrt(math.pi)))
        return float(crps)
