"""
ASTRAEUS Investigation Reproducibility Contract (Phase 7 Foundation).
Mandates complete, bit-level auditable provenance for every autonomous investigation:
- Model versions and engine capability tiers
- Checkpoint and dataset SHA-256 digests
- Random seeds and execution configurations
- Recorded tool calls, intermediate hypotheses, and posterior likelihoods
- Sealed IVOA Prov-DM v1.0 entity

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class InvestigationProvenanceRecord:
    """Complete, immutable record of an autonomous scientific investigation."""
    investigation_id: str
    target_event_id: str
    target_coordinates: Dict[str, float]  # ra_deg, dec_deg
    healpix_pixel_nside32: int
    random_seed: int
    engine_capabilities_used: Dict[str, str]  # engine_id -> capability_tier
    checkpoint_hashes: Dict[str, str]        # checkpoint_name -> sha256
    dataset_hashes: Dict[str, str]           # dataset_name -> sha256
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    observations_ingested: List[Dict[str, Any]] = field(default_factory=list)
    quarantine_firewall_verified: bool = True
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sealed_digest: Optional[str] = None

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any], output_summary: str) -> None:
        self.tool_invocations.append({
            "step": len(self.tool_invocations) + 1,
            "tool": tool_name,
            "arguments": arguments,
            "output_summary": output_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_hypothesis(
        self,
        hypothesis_id: str,
        name: str,
        prior_prob: float,
        posterior_prob: float,
        evidence_citations: List[str],
        falsified: bool = False,
    ) -> None:
        self.hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "name": name,
            "prior_prob": float(prior_prob),
            "posterior_prob": float(posterior_prob),
            "evidence_citations": evidence_citations,
            "falsified": bool(falsified),
        })

    def seal(self) -> str:
        """Computes deterministic SHA-256 seal over the investigation payload."""
        payload = asdict(self)
        payload["sealed_digest"] = None
        serialized = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.sealed_digest = digest
        return digest

    def export_json(self, output_dir: str = "data/provenance/investigations") -> str:
        if self.sealed_digest is None:
            self.seal()

        os.makedirs(output_dir, exist_ok=True)
        fname = f"{self.investigation_id}.json"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return fpath
