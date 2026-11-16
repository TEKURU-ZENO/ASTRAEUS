"""
ASTRAEUS Phase 5: Multi-Survey Photometry Adapters.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
"""

from astraeus_data.adapters.canonical import CanonicalObservationRecord, ObservationType
from astraeus_data.adapters.atlas import AtlasForcedPhotometryAdapter
from astraeus_data.adapters.decam_ddf import DecamDdfPhotometryAdapter
from astraeus_data.adapters.rubin import RubinAlertInterfaceAdapter

__all__ = [
    "CanonicalObservationRecord",
    "ObservationType",
    "AtlasForcedPhotometryAdapter",
    "DecamDdfPhotometryAdapter",
    "RubinAlertInterfaceAdapter",
]
