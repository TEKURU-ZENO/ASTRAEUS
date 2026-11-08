"""
ASTRAEUS Alert Ingestion Engine.
Implements idempotent replay, raw packet preservation, schema-version tracking,
and dead-letter quarantine for streaming and batch alerts.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Set, List, Union
import json
import hashlib
from datetime import datetime, timezone

from astraeus_data.schemas.ztf_alert import ZtfAlertPacket, ZtfCandidate
from astraeus_core.epistemic import EpistemicTier
from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.observations import ObservationRecord, CalibratedMeasurement
from astraeus_core.provenance import ProvEntity, create_provenance_record
from astraeus_core.entity import AstronomicalEntity


class IngestionStatus(str, Enum):
    INGESTED = "ingested"
    DUPLICATE_SKIPPED = "duplicate_skipped"
    QUARANTINED = "quarantined"
    ERROR = "error"


class IngestionResult:
    def __init__(
        self,
        status: IngestionStatus,
        candid: int,
        entity_id: Optional[str] = None,
        raw_hash: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        self.status = status
        self.candid = candid
        self.entity_id = entity_id
        self.raw_hash = raw_hash
        self.error_message = error_message


class AlertIngestionEngine:
    """
    Ingests and normalizes streaming astronomical alerts while guaranteeing idempotency
    and raw packet preservation.
    """

    def __init__(self, storage_root: Union[str, Path] = "data/alerts"):
        self.storage_root = Path(storage_root)
        self.raw_dir = self.storage_root / "raw"
        self.quarantine_dir = self.storage_root / "quarantine"
        self.entities_dir = self.storage_root / "entities"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.entities_dir.mkdir(parents=True, exist_ok=True)

        self.seen_hashes: Set[str] = set()

    def ingest_raw_dict(self, raw_dict: Dict[str, Any]) -> IngestionResult:
        """
        Ingests a raw dictionary, validating schema and routing malformed packets to quarantine.
        """
        candid = raw_dict.get("candid", 0)
        try:
            alert = ZtfAlertPacket.model_validate(raw_dict)
            return self.ingest_packet(alert)
        except Exception as e:
            quarantine_path = self.quarantine_dir / f"{candid}_schema_error.json"
            with open(quarantine_path, "w") as f:
                json.dump({"raw": raw_dict, "validation_error": str(e)}, f, indent=2)
            return IngestionResult(
                status=IngestionStatus.QUARANTINED,
                candid=candid,
                error_message=f"Schema validation error: {str(e)}",
            )

    def ingest_packet(self, alert: ZtfAlertPacket) -> IngestionResult:
        """
        Processes an individual alert packet.
        Guarantees:
        1. Raw packet preservation.
        2. Idempotent deduplication (replays do not duplicate records).
        3. Quarantine of malformed or invalid packets.
        4. Normalization into an IVOA-provenanced AstronomicalEntity.
        """
        payload_hash = alert.compute_payload_hash()

        # Check idempotency
        if payload_hash in self.seen_hashes or (self.raw_dir / f"{alert.candid}.json").exists():
            return IngestionResult(
                status=IngestionStatus.DUPLICATE_SKIPPED,
                candid=alert.candid,
                entity_id=f"astraeus:entity:ztf:{alert.objectId}",
                raw_hash=payload_hash,
            )

        # Preserve verbatim raw packet
        try:
            raw_path = self.raw_dir / f"{alert.candid}.json"
            with open(raw_path, "w") as f:
                f.write(alert.model_dump_json(indent=2))
        except Exception as e:
            return IngestionResult(
                status=IngestionStatus.ERROR,
                candid=alert.candid,
                error_message=f"Failed to write raw packet: {str(e)}",
            )

        # Validate astrometry & telemetry
        cand = alert.candidate
        if not (0.0 <= cand.ra < 360.0 and -90.0 <= cand.dec <= 90.0):
            # Route to quarantine
            quarantine_path = self.quarantine_dir / f"{alert.candid}_invalid_coords.json"
            with open(quarantine_path, "w") as f:
                json.dump({"alert": alert.model_dump(), "error": "Coordinates out of bounds"}, f, indent=2)
            return IngestionResult(
                status=IngestionStatus.QUARANTINED,
                candid=alert.candid,
                error_message=f"Astrometry out of physical bounds: ra={cand.ra}, dec={cand.dec}",
            )

        # Normalize into AstronomicalEntity
        try:
            coords = CelestialCoordinates(
                ra_deg=cand.ra,
                dec_deg=cand.dec,
                uncertainty_mas={"ra": 100.0, "dec": 100.0},
            )
            coords.compute_healpix_index(nside=32)

            activity_id = f"act:ztf_ingest:{alert.candid}"
            observations: List[ObservationRecord] = []

            # Combine current candidate with valid prior candidates
            all_candidates = [cand] + alert.prv_candidates
            for idx, c in enumerate(all_candidates):
                if not c.is_detection:
                    # Non-detection upper limit (magpsf is None)
                    # Recorded with limiting magnitude and flag 2 (upper limit)
                    meas = CalibratedMeasurement(
                        quantity_name="limiting_magnitude",
                        value=float(c.diffmaglim),
                        uncertainty=0.0,
                        unit="mag_AB",
                        filter_band=c.filter_name,
                        quality_flags=2,  # Flag 2 = upper limit
                        epistemic_tier=EpistemicTier.OBSERVED,
                    )
                else:
                    meas = CalibratedMeasurement(
                        quantity_name="apparent_magnitude",
                        value=float(c.magpsf),
                        uncertainty=float(c.sigmapsf),
                        unit="mag_AB",
                        filter_band=c.filter_name,
                        quality_flags=0 if (c.rb is None or c.rb >= 0.5) else 1,
                        epistemic_tier=EpistemicTier.OBSERVED,
                    )

                obs = ObservationRecord(
                    observation_id=f"astraeus:obs:ztf:{alert.candid}_{idx}",
                    survey="ZTF",
                    instrument="ZTF-Camera",
                    filter_band=c.filter_name,
                    mjd=float(c.mjd),
                    exposure_time_sec=30.0,
                    calibrated_flux=meas,
                    quality_flags=meas.quality_flags,
                    ivoa_provenance_id=activity_id,
                )
                observations.append(obs)

            entity_id = f"astraeus:entity:ztf:{alert.objectId}"
            prov = create_provenance_record(
                entity_id=entity_id,
                entity_type="transient_candidate",
                activity_id=activity_id,
                agent_id="agent:pipeline:ztf_ingester",
                extra_metadata={
                    "candid": alert.candid,
                    "raw_hash": payload_hash,
                    "schema_version": alert.schema_version,
                }
            )

            entity = AstronomicalEntity(
                entity_id=entity_id,
                entity_type="transient_candidate",
                epistemic_tier=EpistemicTier.OBSERVED,
                coordinates=coords,
                observations=observations,
                available_modalities=["photometric_time_series"],
                provenance=prov,
            )

            # Record seen hash
            self.seen_hashes.add(payload_hash)

            return IngestionResult(
                status=IngestionStatus.INGESTED,
                candid=alert.candid,
                entity_id=entity_id,
                raw_hash=payload_hash,
            )

        except Exception as e:
            # Route to quarantine on normalization error
            quarantine_path = self.quarantine_dir / f"{alert.candid}_exception.json"
            with open(quarantine_path, "w") as f:
                json.dump({"alert": alert.model_dump(), "exception": str(e)}, f, indent=2)
            return IngestionResult(
                status=IngestionStatus.QUARANTINED,
                candid=alert.candid,
                error_message=f"Normalization failed: {str(e)}",
            )
