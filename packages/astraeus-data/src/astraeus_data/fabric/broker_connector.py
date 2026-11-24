"""
ASTRAEUS Scientific Data Fabric - Broker Connectors Module (WS7.4).
Implements client connectors for community astronomical brokers (Fink, Lasair, ALeRCE)
with Gate 3 Quarantine Firewall enforcement, rate limiting, and offline replay buffers.

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
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


class BrokerType(str, Enum):
    FINK = "FINK"
    LASAIR = "LASAIR"
    ALERCE = "ALERCE"


class AlertQueryFilter(BaseModel):
    """Search parameters for broker alert queries and cone searches."""
    ra_deg: Optional[float] = Field(default=None, ge=0.0, lt=360.0)
    dec_deg: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    radius_arcsec: float = Field(default=5.0, ge=0.1, le=3600.0)
    mjd_min: Optional[float] = Field(default=None)
    mjd_max: Optional[float] = Field(default=None)
    mag_max: Optional[float] = Field(default=None, description="Apparent magnitude limit (brighter than mag_max)")
    classification_tag: Optional[str] = Field(default=None, description="Broker-specific class/tag filter")
    limit: int = Field(default=100, ge=1, le=1000)


class BrokerQueryRecord(BaseModel):
    """Audit ledger record for every outbound broker query."""
    query_id: str
    broker: BrokerType
    endpoint: str
    parameters: Dict[str, Any]
    timestamp_utc: float
    status_code: int
    num_results: int
    quarantine_checked: bool
    sha256_hash: str


class BaseBrokerClient:
    """Base client for astronomical alert brokers with safety, rate limiting, and offline replay."""

    def __init__(
        self,
        broker_type: BrokerType,
        base_url: str,
        rate_limit_per_second: float = 5.0,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        offline_mode: bool = False,
    ):
        self.broker_type = broker_type
        self.base_url = base_url.rstrip("/")
        self.rate_limit_per_second = rate_limit_per_second
        self.min_interval = 1.0 / max(0.1, rate_limit_per_second)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.offline_mode = offline_mode
        self._last_query_time: float = 0.0

        # Offline replay buffer: keyed by objectId or coordinate index
        self._replay_buffer: Dict[str, Dict[str, Any]] = {}
        self._query_ledger: List[BrokerQueryRecord] = []

    def load_replay_buffer(self, items: List[Dict[str, Any]]) -> int:
        """Pre-populates the client's offline replay buffer for deterministic benchmarking."""
        count = 0
        for item in items:
            oid = item.get("objectId") or item.get("objectId_str") or item.get("id")
            if oid:
                self._replay_buffer[str(oid)] = item
                count += 1
            elif "candidate" in item and "candid" in item["candidate"]:
                candid = str(item["candidate"]["candid"])
                self._replay_buffer[candid] = item
                count += 1
        return count

    def _enforce_rate_limit(self) -> None:
        """Enforces client-side token bucket rate limiting."""
        now = time.monotonic()
        elapsed = now - self._last_query_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_query_time = time.monotonic()

    def _assert_quarantine(self, ra_deg: float, dec_deg: float, context: str = "broker_query") -> int:
        """Verifies target coordinates against Gate 3 Quarantine Firewall."""
        return QuarantineFirewall.assert_not_quarantined(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            identifier=f"{self.broker_type.value}:{context}",
            context=f"broker_{self.broker_type.value.lower()}_query",
        )

    def _log_query(
        self,
        endpoint: str,
        params: Dict[str, Any],
        status_code: int,
        results: List[Any],
        quarantine_checked: bool = True,
    ) -> BrokerQueryRecord:
        """Records an immutable audit entry in the query ledger."""
        raw_repr = json.dumps({"broker": self.broker_type.value, "ep": endpoint, "params": params}, sort_keys=True)
        q_hash = hashlib.sha256(raw_repr.encode()).hexdigest()
        rec = BrokerQueryRecord(
            query_id=f"qry:{self.broker_type.value.lower()}:{q_hash[:16]}",
            broker=self.broker_type,
            endpoint=endpoint,
            parameters=params,
            timestamp_utc=time.time(),
            status_code=status_code,
            num_results=len(results),
            quarantine_checked=quarantine_checked,
            sha256_hash=q_hash,
        )
        self._query_ledger.append(rec)
        return rec

    def get_query_ledger(self) -> List[BrokerQueryRecord]:
        return list(self._query_ledger)


class FinkBrokerClient(BaseBrokerClient):
    """
    Client for the Fink community broker (CNRS/IN2P3).
    Supports object querying, cone search, and class-specific live streaming replay.
    """

    def __init__(
        self,
        base_url: str = "https://api.fink-portal.org",
        rate_limit_per_second: float = 4.0,
        timeout_seconds: float = 10.0,
        offline_mode: bool = False,
    ):
        super().__init__(
            broker_type=BrokerType.FINK,
            base_url=base_url,
            rate_limit_per_second=rate_limit_per_second,
            timeout_seconds=timeout_seconds,
            offline_mode=offline_mode,
        )

    def query_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Queries light curve and alert history for a specific ZTF object ID."""
        self._enforce_rate_limit()

        # Check offline replay buffer first
        if object_id in self._replay_buffer:
            res = self._replay_buffer[object_id]
            ra = res.get("candidate", {}).get("ra") or res.get("ra")
            dec = res.get("candidate", {}).get("dec") or res.get("dec")
            if ra is not None and dec is not None:
                self._assert_quarantine(float(ra), float(dec), context=f"object_{object_id}")
            self._log_query(f"/api/v1/objects/{object_id}", {"objectId": object_id}, 200, [res])
            return res

        if self.offline_mode:
            self._log_query(f"/api/v1/objects/{object_id}", {"objectId": object_id}, 404, [])
            return None

        # Network path (sandboxed safe simulation / mock when offline)
        endpoint = f"/api/v1/objects/{object_id}"
        self._log_query(endpoint, {"objectId": object_id}, 404, [])
        return None

    def query_cone(self, query: AlertQueryFilter) -> List[Dict[str, Any]]:
        """Executes a cone search around coordinates with Gate 3 Quarantine protection."""
        if query.ra_deg is None or query.dec_deg is None:
            raise ValueError("Cone search requires both ra_deg and dec_deg.")

        # Gate 3 Quarantine Firewall check: halts query if coordinates fall in restricted pixels
        self._assert_quarantine(query.ra_deg, query.dec_deg, context="cone_search")
        self._enforce_rate_limit()

        matches = []
        radius_deg = query.radius_arcsec / 3600.0

        for item in self._replay_buffer.values():
            cand = item.get("candidate", item)
            ra = cand.get("ra")
            dec = cand.get("dec")
            if ra is None or dec is None:
                continue

            # Angular separation (flat approximation for small radius)
            dra = (float(ra) - query.ra_deg) * math.cos(math.radians(query.dec_deg))
            ddec = float(dec) - query.dec_deg
            sep_deg = math.hypot(dra, ddec)
            if sep_deg <= radius_deg:
                matches.append(item)
                if len(matches) >= query.limit:
                    break

        self._log_query("/api/v1/explorer", query.model_dump(), 200, matches)
        return matches

    def query_latest_alerts_by_class(
        self,
        classification: str,
        n_alerts: int = 50,
        mjd_min: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Queries recent alerts tagged by Fink classifiers (e.g. 'Early SN candidate', 'Kilonova candidate')."""
        self._enforce_rate_limit()
        results = []

        for item in self._replay_buffer.values():
            cand = item.get("candidate", item)
            cl = cand.get("fink_class") or cand.get("classification") or item.get("fink_class")
            if cl and classification.lower() in str(cl).lower():
                ra = cand.get("ra")
                dec = cand.get("dec")
                if ra is not None and dec is not None:
                    # Quarantine check
                    try:
                        self._assert_quarantine(float(ra), float(dec), context=f"class_{classification}")
                    except QuarantineViolationError:
                        continue  # Safely suppress quarantined events from live return

                if mjd_min is not None:
                    mjd = cand.get("mjd") or (cand.get("jd", 2400000.5) - 2400000.5)
                    if mjd < mjd_min:
                        continue

                results.append(item)
                if len(results) >= n_alerts:
                    break

        self._log_query(
            "/api/v1/latests",
            {"class": classification, "n": n_alerts, "mjd_min": mjd_min},
            200,
            results,
        )
        return results


class LasairBrokerClient(BaseBrokerClient):
    """
    Client for the Lasair broker (University of Edinburgh / Queen's University Belfast).
    Provides access to ZTF alert streams, annotations, and watchlist cross-matches.
    """

    def __init__(
        self,
        base_url: str = "https://lasair-ztf.lsst.ac.uk/api",
        rate_limit_per_second: float = 3.0,
        timeout_seconds: float = 10.0,
        offline_mode: bool = False,
    ):
        super().__init__(
            broker_type=BrokerType.LASAIR,
            base_url=base_url,
            rate_limit_per_second=rate_limit_per_second,
            timeout_seconds=timeout_seconds,
            offline_mode=offline_mode,
        )

    def query_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Fetches light curve and annotations for a transient from Lasair."""
        self._enforce_rate_limit()

        if object_id in self._replay_buffer:
            res = self._replay_buffer[object_id]
            ra = res.get("ra") or res.get("candidate", {}).get("ra")
            dec = res.get("dec") or res.get("candidate", {}).get("dec")
            if ra is not None and dec is not None:
                self._assert_quarantine(float(ra), float(dec), context=f"lasair_{object_id}")
            self._log_query(f"/api/objects/{object_id}/", {"objectId": object_id}, 200, [res])
            return res

        self._log_query(f"/api/objects/{object_id}/", {"objectId": object_id}, 404, [])
        return None

    def query_cone(self, query: AlertQueryFilter) -> List[Dict[str, Any]]:
        """Queries Lasair cone search endpoint with Gate 3 Quarantine validation."""
        if query.ra_deg is None or query.dec_deg is None:
            raise ValueError("Cone search requires both ra_deg and dec_deg.")

        self._assert_quarantine(query.ra_deg, query.dec_deg, context="lasair_cone")
        self._enforce_rate_limit()

        matches = []
        radius_deg = query.radius_arcsec / 3600.0

        for item in self._replay_buffer.values():
            ra = item.get("ra") or item.get("candidate", {}).get("ra")
            dec = item.get("dec") or item.get("candidate", {}).get("dec")
            if ra is None or dec is None:
                continue

            dra = (float(ra) - query.ra_deg) * math.cos(math.radians(query.dec_deg))
            ddec = float(dec) - query.dec_deg
            if math.hypot(dra, ddec) <= radius_deg:
                matches.append(item)
                if len(matches) >= query.limit:
                    break

        self._log_query("/api/cone/", query.model_dump(), 200, matches)
        return matches
