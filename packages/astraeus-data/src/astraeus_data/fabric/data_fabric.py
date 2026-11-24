"""
ASTRAEUS Scientific Data Fabric - Central Orchestration Fabric (WS7.4).
Coordinates multi-archive data retrieval, Gate 3 Zero-Touch Quarantine Firewall enforcement,
rate limiting, query audit logging, and two-tier operating modes (Live vs. Offline).

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
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_data.fabric.broker_connector import (
    BrokerType,
    AlertQueryFilter,
    FinkBrokerClient,
    LasairBrokerClient,
)
from astraeus_data.fabric.tns_connector import (
    TnsClient,
    TnsRecord,
    TnsClassificationCertainty,
    OfflineLearningGate,
    UnconfirmedAlertTrainingProhibitedError,
)
from astraeus_data.fabric.archive_connectors import (
    GaiaTapClient,
    GaiaSourceRecord,
    GraceDbClient,
    GraceDbSuperevent,
    AstroSatHeasarcClient,
    MultiWavelengthRecord,
)


class OperatingMode(str, Enum):
    """
    Two-Tier Operational Modes separating real-time alerting from supervised learning:
    - LIVE_INTELLIGENCE: Streaming triage, provisional hypotheses, human-reviewed proposals.
                         No model parameter updates permitted.
    - OFFLINE_LEARNING: Controlled model retraining using audited, verified spectroscopic datasets.
    """
    LIVE_INTELLIGENCE = "LIVE_INTELLIGENCE"
    OFFLINE_LEARNING = "OFFLINE_LEARNING"


class TransientEnvironmentContext(BaseModel):
    """Unified astrophysical context assembled across all fabric archives."""
    target_id: str
    ra_deg: float
    dec_deg: float
    epoch_mjd: float
    healpix_pixel_nside32: int
    quarantine_cleared: bool = True

    # Astrometric Star Vetting
    is_galactic_star: bool = False
    gaia_match: Optional[Dict[str, Any]] = None

    # Ground-Truth Status (as of query epoch)
    tns_record: Optional[Dict[str, Any]] = None
    tns_certainty: str = "UNCLASSIFIED"
    is_ground_truth_confirmed: bool = False

    # Multi-Messenger Context
    coincident_gw_events: List[Dict[str, Any]] = Field(default_factory=list)
    has_gw_association: bool = False

    # Multi-Wavelength Context
    high_energy_observations: List[Dict[str, Any]] = Field(default_factory=list)
    has_xray_detection: bool = False
    xray_flux_upper_limit: Optional[float] = None

    # Provenance
    fabric_query_hash: str
    query_timestamp_utc: float


class ScientificDataFabric:
    """
    Central Scientific Data Fabric uniting live brokers, delayed ground-truth archives,
    and multi-wavelength catalogs with Gate 3 Quarantine Firewall protection.
    """

    def __init__(
        self,
        mode: OperatingMode = OperatingMode.LIVE_INTELLIGENCE,
        offline_mode: bool = True,
        audit_log_dir: Optional[Union[str, Path]] = None,
    ):
        self.mode = mode
        self.offline_mode = offline_mode
        self.audit_log_dir = Path(audit_log_dir) if audit_log_dir else Path("data/fabric_audit")
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize archive sub-clients
        self.fink = FinkBrokerClient(offline_mode=offline_mode)
        self.lasair = LasairBrokerClient(offline_mode=offline_mode)
        self.tns = TnsClient(offline_mode=offline_mode)
        self.gaia = GaiaTapClient(offline_mode=offline_mode)
        self.gracedb = GraceDbClient(offline_mode=offline_mode)
        self.heasarc = AstroSatHeasarcClient(offline_mode=offline_mode)
        self.learning_gate = OfflineLearningGate()

        self._query_counter: int = 0
        self._audit_entries: List[Dict[str, Any]] = []

    def set_operating_mode(self, mode: OperatingMode) -> None:
        """Switches the operating mode between LIVE_INTELLIGENCE and OFFLINE_LEARNING."""
        self.mode = mode

    def assert_safe_coordinates(self, ra_deg: float, dec_deg: float, context: str = "fabric_query") -> int:
        """
        Enforces Gate 3 Zero-Touch Quarantine Firewall.
        Raises QuarantineViolationError if target falls in the 1,808 frozen pixels.
        """
        return QuarantineFirewall.assert_not_quarantined(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            identifier=f"fabric_{context}_{ra_deg:.4f}_{dec_deg:.4f}",
            context=f"data_fabric_{context}",
        )

    def crossmatch_transient_environment(
        self,
        target_id: str,
        ra_deg: float,
        dec_deg: float,
        epoch_mjd: float,
        gaia_search_radius_arcsec: float = 2.0,
        gw_time_window_days: float = 1.0,
        heasarc_time_window_days: float = 7.0,
    ) -> TransientEnvironmentContext:
        """
        Aggregates multi-archive environmental intelligence around an alert candidate.
        Guarantees:
        1. Quarantine firewall verification on coordinates before any archive is queried.
        2. Delayed ground truth masking: prevents future classification from leaking.
        3. Gaia astrometric star vetting to eliminate Galactic stellar contamination.
        4. Gravitational-wave and high-energy multi-messenger coincidence check.
        """
        # 1. Gate 3 Quarantine check
        hp_pixel = self.assert_safe_coordinates(ra_deg, dec_deg, context=f"env_crossmatch_{target_id}")

        self._query_counter += 1
        now_ts = time.time()
        q_raw = f"{target_id}:{ra_deg:.6f}:{dec_deg:.6f}:{epoch_mjd:.4f}:{self._query_counter}"
        q_hash = hashlib.sha256(q_raw.encode()).hexdigest()

        # 2. Gaia Astrometry & Star Vetting
        is_star, gaia_src = self.gaia.check_galactic_star_contamination(
            ra_deg, dec_deg, search_radius_arcsec=gaia_search_radius_arcsec
        )

        # 3. Delayed TNS Ground-Truth Linkage (respecting replay/alert epoch)
        tns_record = self.tns.link_ground_truth(target_id, as_of_mjd=epoch_mjd)
        if tns_record is None:
            # Try cone search
            cone_matches = self.tns.query_cone(ra_deg, dec_deg, radius_arcsec=3.0)
            if cone_matches:
                tns_record = self.tns.link_ground_truth(cone_matches[0].iau_name, as_of_mjd=epoch_mjd)

        tns_dict = tns_record.model_dump() if tns_record else None
        tns_cert = tns_record.certainty.value if tns_record else "UNCLASSIFIED"
        is_confirmed = bool(
            tns_record and tns_record.certainty == TnsClassificationCertainty.SPECTROSCOPIC and tns_record.classification
        )

        # 4. GraceDB Multi-Messenger GW Association
        gw_matches = self.gracedb.query_events_in_window(
            mjd_min=epoch_mjd - gw_time_window_days,
            mjd_max=epoch_mjd + gw_time_window_days,
        )
        has_gw = len(gw_matches) > 0

        # 5. AstroSat / HEASARC High-Energy Upper Limits / Detections
        he_matches = self.heasarc.query_counterparts(
            ra_deg, dec_deg, epoch_mjd=epoch_mjd, time_window_days=heasarc_time_window_days
        )
        has_xray = any(m.is_detection for m in he_matches)
        xray_ul = None
        for m in he_matches:
            if not m.is_detection and m.flux_upper_limit_3sigma is not None:
                if xray_ul is None or m.flux_upper_limit_3sigma < xray_ul:
                    xray_ul = m.flux_upper_limit_3sigma

        env = TransientEnvironmentContext(
            target_id=target_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            epoch_mjd=epoch_mjd,
            healpix_pixel_nside32=hp_pixel,
            quarantine_cleared=True,
            is_galactic_star=is_star,
            gaia_match=gaia_src.model_dump() if gaia_src else None,
            tns_record=tns_dict,
            tns_certainty=tns_cert,
            is_ground_truth_confirmed=is_confirmed,
            coincident_gw_events=[e.model_dump() for e in gw_matches],
            has_gw_association=has_gw,
            high_energy_observations=[o.model_dump() for o in he_matches],
            has_xray_detection=has_xray,
            xray_flux_upper_limit=xray_ul,
            fabric_query_hash=q_hash,
            query_timestamp_utc=now_ts,
        )

        # Record audit ledger
        self._audit_entries.append({
            "query_hash": q_hash,
            "target_id": target_id,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "epoch_mjd": epoch_mjd,
            "operating_mode": self.mode.value,
            "timestamp_utc": now_ts,
            "healpix_pixel": hp_pixel,
            "is_galactic_star": is_star,
            "has_gw": has_gw,
        })

        return env

    def verify_and_authorize_training_event(
        self,
        record: TnsRecord,
        allow_photometric_ablation: bool = False,
    ) -> bool:
        """
        Authorizes a transient event for offline supervised training.
        Strictly enforces:
        1. Mode must be OFFLINE_LEARNING.
        2. Record must have verified SPECTROSCOPIC certainty (unless ablation authorized).
        3. Never updates directly on streaming, unconfirmed alerts.
        """
        if self.mode == OperatingMode.LIVE_INTELLIGENCE:
            raise UnconfirmedAlertTrainingProhibitedError(
                "Fabric Operational Violation: System is in LIVE_INTELLIGENCE mode. "
                "Model parameters are strictly frozen. Online parameter updates are prohibited."
            )

        return self.learning_gate.verify_training_candidate(
            record, allow_photometric_ablation=allow_photometric_ablation
        )

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        return list(self._audit_entries)
