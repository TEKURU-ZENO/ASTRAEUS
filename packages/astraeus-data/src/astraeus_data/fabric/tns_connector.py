"""
ASTRAEUS Scientific Data Fabric - Transient Name Server (TNS) Connector Module (WS7.4).
Implements delayed ground-truth linkage, spectroscopic certainty verification,
and strict offline learning gates preventing online backpropagation on unconfirmed alerts.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import time
import math
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


class TnsClassificationCertainty(str, Enum):
    """Scientific certainty hierarchy for transient ground-truth classifications."""
    SPECTROSCOPIC = "SPECTROSCOPIC"   # Verified with optical/IR spectrum, instrument, and redshift
    PHOTOMETRIC = "PHOTOMETRIC"       # Inferred from light-curve template fitting only
    UNCLASSIFIED = "UNCLASSIFIED"     # Reported transient without formal physical classification


class UnconfirmedAlertTrainingProhibitedError(Exception):
    """Raised when an attempt is made to update model parameters on unconfirmed/streaming alerts."""
    pass


class TnsRecord(BaseModel):
    """Canonical representation of an IAU Transient Name Server (TNS) object record."""
    iau_name: str = Field(..., description="IAU designation, e.g. 'SN 2023wtm' or 'AT 2024abc'")
    internal_name: Optional[str] = Field(default=None, description="Survey internal identifier, e.g. 'ZTF17aacpbmv'")
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    discovery_date_mjd: float = Field(..., description="Epoch of initial detection / reporting in MJD")
    discovery_mag: Optional[float] = Field(default=None)
    redshift: Optional[float] = Field(default=None)
    redshift_err: Optional[float] = Field(default=None)
    classification: Optional[str] = Field(default=None, description="Physical transient class, e.g. 'SN Ia', 'Kilonova', 'TDE'")
    certainty: TnsClassificationCertainty = Field(default=TnsClassificationCertainty.UNCLASSIFIED)
    classification_date_mjd: Optional[float] = Field(
        default=None,
        description="Epoch when spectroscopic confirmation or official classification was registered."
    )
    spectroscopy_instrument: Optional[str] = Field(default=None, description="Spectrograph used, e.g. 'SEDM', 'DBSP', 'LRIS'")
    spectroscopy_telescope: Optional[str] = Field(default=None, description="Telescope used, e.g. 'P200', 'Keck I', 'NOT'")
    reporting_group: Optional[str] = Field(default=None)
    raw_source_hash: Optional[str] = Field(default=None)

    def is_spectroscopically_confirmed(self) -> bool:
        return self.certainty == TnsClassificationCertainty.SPECTROSCOPIC and self.classification is not None


class OfflineLearningGate:
    """
    Architectural governance firewall enforcing the separation between
    Live Scientific Intelligence and Offline Scientific Learning.

    Core Scientific Rule:
    Never backpropagate directly on unconfirmed streaming alerts.
    Parameter updates are strictly restricted to verified spectroscopic ground-truth
    processed in offline, audited release cycles.
    """

    def __init__(self, audit_log_path: Optional[Union[str, Path]] = None):
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self._training_records: List[Dict[str, Any]] = []

    def verify_training_candidate(
        self,
        record: TnsRecord,
        allow_photometric_ablation: bool = False,
    ) -> bool:
        """
        Validates whether a candidate transient is authorized for model training.

        Raises:
            UnconfirmedAlertTrainingProhibitedError if the candidate is unclassified
            or purely photometric without explicit experimental ablation authorization.
        """
        if record.certainty == TnsClassificationCertainty.UNCLASSIFIED or not record.classification:
            raise UnconfirmedAlertTrainingProhibitedError(
                f"Gate 3 / Governance Violation: Attempted to train on unconfirmed alert '{record.iau_name}' "
                f"({record.internal_name}). Online backpropagation on unconfirmed alerts is strictly prohibited."
            )

        if record.certainty == TnsClassificationCertainty.PHOTOMETRIC and not allow_photometric_ablation:
            raise UnconfirmedAlertTrainingProhibitedError(
                f"Governance Warning: Candidate '{record.iau_name}' only has photometric classification. "
                f"Spectroscopic ground-truth is required for production training updates."
            )

        # Log authorized training event
        auth_entry = {
            "iau_name": record.iau_name,
            "internal_name": record.internal_name,
            "classification": record.classification,
            "certainty": record.certainty.value,
            "spectroscopy_instrument": record.spectroscopy_instrument,
            "timestamp_utc": time.time(),
            "authorized": True,
        }
        self._training_records.append(auth_entry)
        return True

    def get_authorized_training_count(self) -> int:
        return len(self._training_records)


class TnsClient:
    """
    Transient Name Server client managing catalog retrieval, delayed ground-truth linkage,
    and Gate 3 Quarantine validation.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        offline_mode: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/tns_cache")
        self.offline_mode = offline_mode
        self._records_by_iau: Dict[str, TnsRecord] = {}
        self._records_by_internal: Dict[str, TnsRecord] = {}
        self.learning_gate = OfflineLearningGate()

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().upper().replace(" ", "").replace("-", "").replace("_", "")

    def add_record(self, record: TnsRecord) -> None:
        """Indexes a TnsRecord into memory with normalized lookup keys."""
        norm_iau = self._normalize_key(record.iau_name)
        self._records_by_iau[norm_iau] = record
        self._records_by_iau[record.iau_name.upper()] = record
        if record.internal_name:
            norm_internal = self._normalize_key(record.internal_name)
            self._records_by_internal[norm_internal] = record
            self._records_by_internal[record.internal_name.upper()] = record

    def load_from_records(self, records: List[TnsRecord]) -> int:
        """Bulk indexes a list of TnsRecords."""
        count = 0
        for r in records:
            self.add_record(r)
            count += 1
        return count

    def get_record(self, identifier: str) -> Optional[TnsRecord]:
        """Looks up a TNS record by IAU name or internal survey name."""
        clean_id = identifier.strip().upper()
        norm_id = self._normalize_key(identifier)
        if norm_id in self._records_by_iau:
            return self._records_by_iau[norm_id]
        if clean_id in self._records_by_iau:
            return self._records_by_iau[clean_id]
        if norm_id in self._records_by_internal:
            return self._records_by_internal[norm_id]
        if clean_id in self._records_by_internal:
            return self._records_by_internal[clean_id]
        return None

    def link_ground_truth(
        self,
        identifier: str,
        as_of_mjd: float,
    ) -> Optional[TnsRecord]:
        """
        Delayed Ground-Truth Linkage:
        Returns the ground truth as it was known at epoch `as_of_mjd`.

        Guarantees No Lookahead Bias:
        If `as_of_mjd < classification_date_mjd`, the classification and spectroscopy details
        are masked, returning a record with `certainty = UNCLASSIFIED`.
        """
        record = self.get_record(identifier)
        if record is None:
            return None

        # Check if classification was known as of the requested MJD
        if record.classification_date_mjd is not None and as_of_mjd < record.classification_date_mjd:
            # Mask future classification to prevent lookahead bias
            masked_record = record.model_copy(
                update={
                    "classification": None,
                    "certainty": TnsClassificationCertainty.UNCLASSIFIED,
                    "spectroscopy_instrument": None,
                    "spectroscopy_telescope": None,
                }
            )
            return masked_record

        return record

    def query_cone(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 3.0,
    ) -> List[TnsRecord]:
        """
        Finds TNS records coincident within radius_arcsec with Gate 3 Quarantine check.
        """
        QuarantineFirewall.assert_not_quarantined(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            identifier=f"tns_cone_{ra_deg:.4f}_{dec_deg:.4f}",
            context="tns_cone_query",
        )

        radius_deg = radius_arcsec / 3600.0
        matches = []
        for r in self._records_by_iau.values():
            dra = (r.ra_deg - ra_deg) * math.cos(math.radians(dec_deg))
            ddec = r.dec_deg - dec_deg
            sep = math.hypot(dra, ddec)
            if sep <= radius_deg:
                matches.append(r)

        return matches
