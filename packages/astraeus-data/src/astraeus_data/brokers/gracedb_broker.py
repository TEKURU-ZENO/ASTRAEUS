"""
ASTRAEUS GraceDB Gravitational-Wave Broker Adapter (WS10.1).
Connects to LVC/LVK GraceDB event notices, parses GCN Circulars and VOEvents,
extracts 3D localization parameters, false-alarm rates, and astrophysical classifications,
and feeds electromagnetic follow-up planning.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import json
import hashlib
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from astraeus_data.schemas.gw_notice import GwNoticePacket
from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    EventIngestionResult,
    IngestionStatus,
)


class GraceDBBrokerConfig(BaseModel):
    broker_id: str = "GRACEDB_LVK_NOTICE_BROKER"
    max_far_hz: float = 1e-4  # Maximum false-alarm rate (~1 per 3 months)
    min_has_remnant_prob: float = 0.01  # Minimum non-zero ejecta probability
    require_localization: bool = True


class GraceDBBrokerAdapter:
    """Production broker adapter for Gravitational-Wave alerts from GraceDB."""

    def __init__(
        self,
        config: Optional[GraceDBBrokerConfig] = None,
        ingestion_engine: Optional[ScientificEventIngestionEngine] = None,
    ) -> None:
        self.config = config or GraceDBBrokerConfig()
        self.engine = ingestion_engine or ScientificEventIngestionEngine()
        self.ingested_notices = 0
        self.filtered_terrestrial_count = 0

    def parse_notice(self, notice_dict: Dict[str, Any]) -> GwNoticePacket:
        """Parses a dictionary representation of an LVK alert notice."""
        return GwNoticePacket(**notice_dict)

    def process_notice(self, notice: GwNoticePacket) -> EventIngestionResult:
        """Evaluates significance and ingests GW notices."""
        # Check false-alarm rate
        if notice.false_alarm_rate_hz > self.config.max_far_hz:
            return EventIngestionResult(
                status=IngestionStatus.SCHEMA_VALIDATION_ERROR,
                event_id=f"evt:gw:{notice.superevent_id}",
                raw_hash=notice.compute_payload_hash(),
                error_message=f"HIGH_FAR_REJECTED: FAR {notice.false_alarm_rate_hz:.2e} Hz > {self.config.max_far_hz:.2e} Hz",
            )

        # Check terrestrial noise
        if notice.prob_terrestrial >= 0.50:
            self.filtered_terrestrial_count += 1
            return EventIngestionResult(
                status=IngestionStatus.SCHEMA_VALIDATION_ERROR,
                event_id=f"evt:gw:{notice.superevent_id}",
                raw_hash=notice.compute_payload_hash(),
                error_message=f"TERRESTRIAL_NOISE: prob_terrestrial={notice.prob_terrestrial:.2f} >= 0.50",
            )

        res = self.engine.ingest_gw_notice(notice)
        if res.status == IngestionStatus.INGESTED:
            self.ingested_notices += 1
        return res
