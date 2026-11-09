"""
ASTRAEUS Physics Fit Status Module.
Establishes first-class outcomes for physical model fitting, preventing fit failures
or model non-applicability from being conflated with astrophysical anomalies.
"""

from enum import Enum


class FitStatus(str, Enum):
    """
    Formal outcomes of a physical forward-fitting attempt.
    """
    FIT_SUCCESS = "FIT_SUCCESS"
    FIT_POOR_QUALITY = "FIT_POOR_QUALITY"                   # Converged, but chi2_red > threshold or unphysical parameters
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"           # Candidate failed applicability gate (e.g. wrong phase or class)
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS" # Fewer than minimum required photometric epochs (N < 5)
    NON_IDENTIFIABLE_PARAMETERS = "NON_IDENTIFIABLE_PARAMETERS" # Covariance matrix singular or unconstrained
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"             # Numerical solver failed to converge
