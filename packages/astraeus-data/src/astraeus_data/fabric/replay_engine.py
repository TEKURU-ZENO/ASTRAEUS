"""
ASTRAEUS Scientific Data Fabric - Historical Streaming Replay Engine (WS7.4).
Simulates realistic chronological alert streaming from historical surveys without lookahead bias,
integrating delayed TNS ground-truth linkage and triage evaluation.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import time
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel

from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    UnifiedScientificEvent,
    EventIngestionResult,
)
from astraeus_data.triage.event_triage_engine import (
    EventTriageEngine,
    TriageDecision,
    AnomalyClass,
)
from astraeus_data.fabric.data_fabric import ScientificDataFabric, OperatingMode
from astraeus_data.fabric.tns_connector import TnsRecord, TnsClassificationCertainty


@dataclass
class ReplayAlertEpoch:
    """An individual transient detection epoch in chronological order."""
    alert_id: str
    target_id: str
    mjd: float
    ra_deg: float
    dec_deg: float
    filter_band: str
    mag_ab: float
    mag_err: float
    survey: str = "ZTF"
    is_trigger: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayEvaluationResult:
    """Evaluation record comparing live triage decision with delayed ground truth."""
    alert_id: str
    target_id: str
    triage_mjd: float
    predicted_class: str
    predicted_urgency: str
    predicted_score: float
    ground_truth_class: Optional[str] = None
    ground_truth_certainty: str = "UNCLASSIFIED"
    classification_release_mjd: Optional[float] = None
    lead_time_days: Optional[float] = None
    is_correct: Optional[bool] = None


class HistoricalStreamingReplayEngine:
    """
    Chronological replay engine evaluating triage accuracy and decision lead time
    against independent delayed ground-truth.
    """

    def __init__(
        self,
        fabric: ScientificDataFabric,
        triage_engine: Optional[EventTriageEngine] = None,
        ingestion_engine: Optional[ScientificEventIngestionEngine] = None,
    ):
        self.fabric = fabric
        self.triage_engine = triage_engine or EventTriageEngine()
        self.ingestion_engine = ingestion_engine or ScientificEventIngestionEngine(storage_root="data/events_replay")
        self.current_mjd: float = 0.0
        self.replay_history: List[ReplayEvaluationResult] = []

    def run_replay(
        self,
        alert_stream: List[ReplayAlertEpoch],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> List[ReplayEvaluationResult]:
        """
        Executes chronological streaming replay over a series of historical alert epochs.
        Guarantees that at epoch t, only observations and classifications released at <= t
        are visible to the triage algorithms.
        """
        # Sort alert stream strictly chronologically
        sorted_stream = sorted(alert_stream, key=lambda a: a.mjd)
        total_alerts = len(sorted_stream)
        results = []

        # Ensure fabric is in LIVE_INTELLIGENCE mode during streaming
        self.fabric.set_operating_mode(OperatingMode.LIVE_INTELLIGENCE)

        # Map to track light curves per target during stream
        lightcurves_by_target: Dict[str, List[Dict[str, Any]]] = {}

        for idx, alert in enumerate(sorted_stream):
            self.current_mjd = alert.mjd

            # Accumulate photometry
            if alert.target_id not in lightcurves_by_target:
                lightcurves_by_target[alert.target_id] = []
            
            phot_pt = {
                "mjd": alert.mjd,
                "filter": alert.filter_band,
                "filter_band": alert.filter_band,
                "mag": alert.mag_ab,
                "mag_ab": alert.mag_ab,
                "magerr": alert.mag_err,
                "mag_err": alert.mag_err,
                "flux_njy": 10.0 ** ((alert.mag_ab - 8.90) / -2.5) * 1e9 if alert.mag_ab > 0 else 0.0,
                "flux_err_njy": 0.0,
                "snr": 1.0857 / alert.mag_err if alert.mag_err > 0 else 0.0,
            }
            lightcurves_by_target[alert.target_id].append(phot_pt)

            # Check environment context as of CURRENT MJD (delayed ground truth masking)
            env_context = self.fabric.crossmatch_transient_environment(
                target_id=alert.target_id,
                ra_deg=alert.ra_deg,
                dec_deg=alert.dec_deg,
                epoch_mjd=self.current_mjd,
            )

            # Synchronize Gaia star context into triage cross matcher
            if env_context.is_galactic_star and env_context.gaia_match:
                gm = env_context.gaia_match
                pm = math.hypot(gm.get("pmra_mas_yr") or 0.0, gm.get("pmdec_mas_yr") or 0.0)
                self.triage_engine.cross_matcher.add_reference_star(
                    source_id=str(gm.get("source_id", "gaia_star")),
                    ra=alert.ra_deg,
                    dec=alert.dec_deg,
                    pm_mas_yr=pm,
                    parallax_mas=gm.get("parallax_mas") or 0.0,
                    is_variable=True,
                )

            # Construct normalized scientific event for triage
            event_id = f"replay:{alert.target_id}:{alert.alert_id}"
            unified_event = UnifiedScientificEvent(
                event_id=event_id,
                survey_origin="ZTF",
                ra_deg=alert.ra_deg,
                dec_deg=alert.dec_deg,
                healpix_pixel_nside32=env_context.healpix_pixel_nside32,
                event_time_mjd=alert.mjd,
                photometry_points=list(lightcurves_by_target[alert.target_id]),
                event_metadata=alert.metadata,
                raw_payload_hash=f"hash_{alert.alert_id}",
                quarantine_cleared=True,
                is_too_promising=(alert.mag_ab < 18.5 and alert.mag_err < 0.08),
            )

            # Perform live triage
            triage_decision = self.triage_engine.triage_event(unified_event)

            # Retrieve final ground truth (as known after all observations) for benchmarking
            final_record = self.fabric.tns.get_record(alert.target_id)
            if final_record is None:
                cone_matches = self.fabric.tns.query_cone(alert.ra_deg, alert.dec_deg, radius_arcsec=3.0)
                if cone_matches:
                    final_record = cone_matches[0]
            gt_class = final_record.classification if final_record else None
            gt_cert = final_record.certainty.value if final_record else "UNCLASSIFIED"
            rel_mjd = final_record.classification_date_mjd if final_record else None

            lead_time = (rel_mjd - alert.mjd) if rel_mjd is not None else None

            # Accuracy assessment (standardizing labels)
            is_corr = None
            pred_class = triage_decision.primary_anomaly_class.value
            if gt_class:
                is_corr = self._check_class_match(pred_class, gt_class)

            eval_res = ReplayEvaluationResult(
                alert_id=alert.alert_id,
                target_id=alert.target_id,
                triage_mjd=alert.mjd,
                predicted_class=pred_class,
                predicted_urgency=triage_decision.priority_tier.value,
                predicted_score=triage_decision.priority_score,
                ground_truth_class=gt_class,
                ground_truth_certainty=gt_cert,
                classification_release_mjd=rel_mjd,
                lead_time_days=lead_time,
                is_correct=is_corr,
            )
            results.append(eval_res)

            if progress_callback:
                progress_callback(idx + 1, total_alerts, alert.mjd)

        self.replay_history = results
        return results

    @staticmethod
    def _check_class_match(predicted: str, ground_truth: str) -> bool:
        """Harmonizes taxonomic labels between model predictions and IAU TNS classes."""
        pred = predicted.upper()
        gt = ground_truth.upper()

        if pred == "RAPID_OPTICAL_TRANSIENT_KN" and ("KILONOVA" in gt or "KN" in gt or "FAST" in gt):
            return True
        if pred == "SUPERLUMINOUS_SN_TDE" and ("TDE" in gt or "SLSN" in gt or "SUPERLUMINOUS" in gt):
            return True
        if pred == "EXTRAGALACTIC_SN" and ("SN" in gt or "SUPERNOVA" in gt):
            return True
        if pred == "STELLAR_FLARE_VARIABLE" and ("VARIABLE" in gt or "STAR" in gt or "FLARE" in gt or "CV" in gt):
            return True
        if pred == "MULTI_MESSENGER_GW_COUNTERPART" and ("GW" in gt or "BNS" in gt or "KILONOVA" in gt):
            return True

        return pred.lower() in gt.lower() or gt.lower() in pred.lower()
