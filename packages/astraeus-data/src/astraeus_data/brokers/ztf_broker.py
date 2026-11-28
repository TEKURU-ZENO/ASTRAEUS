"""
ASTRAEUS ZTF Alert Broker Adapter (WS10.1).
Connects to ZTF alert streams (e.g., Kafka / IPAC / Lasair / Pitt-Google),
decodes alert packets, validates photometric quality and Real-Bogus flags,
and routes valid detections into the scientific ingestion engine.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Iterator, Callable
from pydantic import BaseModel, Field

from astraeus_data.schemas.ztf_alert import ZtfAlertPacket, ZtfCandidate
from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    EventIngestionResult,
    IngestionStatus,
)


class ZtfBrokerConfig(BaseModel):
    broker_id: str = "ZTF_KAFKA_BROKER_PROD"
    stream_topic: str = "ztf_public_alerts"
    min_real_bogus_score: float = 0.50
    min_snr: float = 5.0
    max_retries: int = 3
    retry_backoff_sec: float = 0.5


class ZtfBrokerAdapter:
    """Production broker adapter for the Zwicky Transient Facility."""

    def __init__(
        self,
        config: Optional[ZtfBrokerConfig] = None,
        ingestion_engine: Optional[ScientificEventIngestionEngine] = None,
    ) -> None:
        self.config = config or ZtfBrokerConfig()
        self.engine = ingestion_engine or ScientificEventIngestionEngine()
        self.ingested_count = 0
        self.filtered_bogus_count = 0
        self.dead_letter_count = 0

    def parse_raw_packet(self, raw_bytes: bytes) -> Optional[ZtfAlertPacket]:
        """Parses raw JSON or Avro bytes into a validated ZtfAlertPacket."""
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            return ZtfAlertPacket(**data)
        except Exception as exc:
            self.dead_letter_count += 1
            return None

    def process_alert(self, packet: ZtfAlertPacket) -> EventIngestionResult:
        """Applies scientific quality filters and ingests valid alerts."""
        cand = packet.candidate

        # Real-Bogus filter (reject clear artifacts)
        rb_score = cand.drb if cand.drb is not None else (cand.rb or 1.0)
        if rb_score < self.config.min_real_bogus_score:
            self.filtered_bogus_count += 1
            return EventIngestionResult(
                status=IngestionStatus.SCHEMA_VALIDATION_ERROR,
                event_id=f"evt:ztf:{cand.candid}",
                raw_hash=packet.compute_payload_hash(),
                error_message=f"BOGUS_FILTER_REJECTED: Real-bogus score {rb_score:.3f} < {self.config.min_real_bogus_score}",
            )

        # Ingest into scientific ingestion engine with Gate 3 quarantine verification
        res = self.engine.ingest_ztf_packet(packet)
        if res.status == IngestionStatus.INGESTED:
            self.ingested_count += 1
        return res

    def process_stream(self, stream_iterator: Iterator[bytes]) -> List[EventIngestionResult]:
        """Consumes a stream of raw alert packets with automatic retry on transient error."""
        results = []
        for raw_bytes in stream_iterator:
            retries = 0
            while retries <= self.config.max_retries:
                try:
                    packet = self.parse_raw_packet(raw_bytes)
                    if packet is None:
                        results.append(EventIngestionResult(
                            status=IngestionStatus.SCHEMA_VALIDATION_ERROR,
                            event_id="unknown_corrupt",
                            raw_hash=hashlib.sha256(raw_bytes).hexdigest(),
                            error_message="Raw packet decoding failed",
                        ))
                        break
                    res = self.process_alert(packet)
                    results.append(res)
                    break
                except Exception as exc:
                    retries += 1
                    if retries > self.config.max_retries:
                        self.dead_letter_count += 1
                        results.append(EventIngestionResult(
                            status=IngestionStatus.SCHEMA_VALIDATION_ERROR,
                            event_id="retry_exhausted",
                            raw_hash=hashlib.sha256(raw_bytes).hexdigest(),
                            error_message=f"Max retries exceeded: {exc}",
                        ))
                        break
                    time.sleep(self.config.retry_backoff_sec)
        return results
