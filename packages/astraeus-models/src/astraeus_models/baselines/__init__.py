"""
ASTRAEUS Stage 2 Continuous-Time Baselines & Level-Playing-Field Benchmark Protocol.
Implements Baselines 1-4 and the common 24-feature calibrated classifier.
"""

from astraeus_models.baselines.akima import AkimaBaselineModel
from astraeus_models.baselines.gp import MultiOutputGPBaseline
from astraeus_models.baselines.transformer import NonPretrainedTransformerBaseline
from astraeus_models.baselines.astromer_adapter import AstromerAdapterBaseline
from astraeus_models.baselines.common_classifier import CommonFeatureClassifier

__all__ = [
    "AkimaBaselineModel",
    "MultiOutputGPBaseline",
    "NonPretrainedTransformerBaseline",
    "AstromerAdapterBaseline",
    "CommonFeatureClassifier",
]
