"""
ASTRAEUS Scientific Knowledge Fabric - Claims & Knowledge Provenance (WS8.1).
Defines structured traceable scientific claims, equation extraction, contradiction detection,
and literature-to-observation evidence linkage.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field


class ReplicationStatus(str, Enum):
    """Formal replication status of a scientific claim."""
    UNVERIFIED = "UNVERIFIED"
    REPLICATED_INDEPENDENTLY = "REPLICATED_INDEPENDENTLY"
    REPLICATED_WITH_REVISIONS = "REPLICATED_WITH_REVISIONS"
    CONTESTED = "CONTESTED"
    FALSIFIED = "FALSIFIED"


class ClaimType(str, Enum):
    """Epistemological category of scientific claim."""
    THEORETICAL_PREDICTION = "THEORETICAL_PREDICTION"
    OBSERVATIONAL_MEASUREMENT = "OBSERVATIONAL_MEASUREMENT"
    EMPIRICAL_SCALING_RELATION = "EMPIRICAL_SCALING_RELATION"
    CONSERVATION_LAW = "CONSERVATION_LAW"
    PHENOMENOLOGICAL_MODEL = "PHENOMENOLOGICAL_MODEL"


class ExtractedEquation(BaseModel):
    """Mathematical equation extracted from theoretical literature or empirical fits."""
    latex_str: str = Field(..., description="LaTeX representation of the equation")
    sympy_expr_str: str = Field(..., description="Machine-readable symbolic formula")
    dependent_variable: str = Field(..., description="Primary output or response variable")
    independent_variables: List[str] = Field(default_factory=list, description="Independent physical variables")
    parameters: Dict[str, float] = Field(default_factory=dict, description="Numerical parameters or scale factors")
    units: Dict[str, str] = Field(default_factory=dict, description="Physical units for each variable")
    dimensions: Dict[str, str] = Field(default_factory=dict, description="Dimensional formula (e.g. M L^2 T^-2)")


class ScientificClaim(BaseModel):
    """
    Structured, traceable scientific claim with full provenance, mathematical specification,
    domain of validity, and supporting/contradicting evidence linkage.
    """
    claim_id: str = Field(..., description="Deterministic or unique claim identifier")
    claim_type: ClaimType = Field(..., description="Epistemological category")
    statement: str = Field(..., description="Precise prose statement of the claim")
    source_publication: str = Field(..., description="ADS Bibcode, DOI, or arXiv identifier")
    domain: str = Field(..., description="Astrophysical domain, e.g. TRANSIENT_ASTROPHYSICS")
    assumptions: List[str] = Field(default_factory=list, description="Explicit theoretical and physical assumptions")
    domain_validity_range: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Physical parameter ranges where claim holds, e.g. {'ejecta_mass_msun': (0.001, 0.1)}"
    )
    equations: List[ExtractedEquation] = Field(default_factory=list, description="Extracted mathematical equations")
    constants: Dict[str, float] = Field(default_factory=dict, description="Physical constants utilized")
    units: Dict[str, str] = Field(default_factory=dict, description="Dimensional units for claim variables")
    uncertainty_bounds: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Confidence intervals / uncertainty bounds on reported constants/parameters"
    )
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="Linked observation or test IDs")
    contradicting_evidence_ids: List[str] = Field(default_factory=list, description="Contradicting observation IDs")
    replication_status: ReplicationStatus = Field(default=ReplicationStatus.UNVERIFIED)
    superseded_by: Optional[str] = Field(default=None, description="Claim ID that supersedes this claim")
    supersedes: Optional[str] = Field(default=None, description="Previous Claim ID superseded by this claim")
    evidence_hashes: List[str] = Field(default_factory=list, description="Cryptographic hashes of linked evidence")
    sha256_hash: str = Field(default="", description="Cryptographic integrity hash of claim specification")
    created_at_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 fingerprint of the claim content."""
        payload = {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type.value,
            "statement": self.statement.strip(),
            "source_publication": self.source_publication.strip(),
            "domain": self.domain,
            "assumptions": sorted(self.assumptions),
            "domain_validity_range": {k: list(v) for k, v in sorted(self.domain_validity_range.items())},
            "equations": [eq.latex_str for eq in self.equations],
            "constants": {k: self.constants[k] for k in sorted(self.constants.keys())},
            "supersedes": self.supersedes,
        }
        raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        if not self.sha256_hash:
            self.sha256_hash = self.compute_hash()


class ClaimContradiction(BaseModel):
    """Identified theoretical or empirical contradiction between scientific claims."""
    claim_id_a: str
    claim_id_b: str
    incompatibility_reason: str
    domain_overlap: Dict[str, Tuple[float, float]]
    discrepancy_significance_sigma: float
    detected_at_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScientificKnowledgeFabric:
    """
    Central repository and query fabric for traceable scientific knowledge.
    Enforces cryptographic evidence provenance, detects contradictions across claims,
    and maintains versioned replication lineages.
    """

    def __init__(self) -> None:
        self._claims: Dict[str, ScientificClaim] = {}
        self._evidence_to_claims: Dict[str, Set[str]] = {}
        self._contradictions: List[ClaimContradiction] = []
        self._lineage_graph: Dict[str, List[str]] = {}

    def register_claim(self, claim: ScientificClaim) -> str:
        """Registers a structured scientific claim into the fabric."""
        # Ensure hash is computed
        if not claim.sha256_hash:
            claim.sha256_hash = claim.compute_hash()

        self._claims[claim.claim_id] = claim

        # Register evidence linkages
        for evid in claim.supporting_evidence_ids + claim.contradicting_evidence_ids:
            if evid not in self._evidence_to_claims:
                self._evidence_to_claims[evid] = set()
            self._evidence_to_claims[evid].add(claim.claim_id)

        # Check contradictions against existing claims in domain
        contradictions = self.detect_contradictions(claim)
        self._contradictions.extend(contradictions)

        return claim.claim_id

    def get_claim(self, claim_id: str) -> Optional[ScientificClaim]:
        """Retrieves claim by ID."""
        return self._claims.get(claim_id)

    def list_claims(self, domain: Optional[str] = None) -> List[ScientificClaim]:
        """Returns all registered claims, optionally filtered by domain."""
        if domain:
            return [c for c in self._claims.values() if c.domain == domain]
        return list(self._claims.values())

    def link_evidence(
        self,
        claim_id: str,
        evidence_id: str,
        supports: bool,
        evidence_hash: str,
    ) -> None:
        """Links an empirical observation or experimental result to a claim."""
        claim = self._claims.get(claim_id)
        if not claim:
            raise KeyError(f"Scientific claim {claim_id} not registered in fabric")

        if supports:
            if evidence_id not in claim.supporting_evidence_ids:
                claim.supporting_evidence_ids.append(evidence_id)
        else:
            if evidence_id not in claim.contradicting_evidence_ids:
                claim.contradicting_evidence_ids.append(evidence_id)

        if evidence_hash and evidence_hash not in claim.evidence_hashes:
            claim.evidence_hashes.append(evidence_hash)

        if evidence_id not in self._evidence_to_claims:
            self._evidence_to_claims[evidence_id] = set()
        self._evidence_to_claims[evidence_id].add(claim_id)

    def detect_contradictions(self, candidate: ScientificClaim) -> List[ClaimContradiction]:
        """
        Detects physical or parameter conflicts between the candidate claim and existing claims
        in the same domain.
        """
        contradictions: List[ClaimContradiction] = []

        for existing_id, existing in self._claims.items():
            if existing_id == candidate.claim_id:
                continue
            if existing.domain != candidate.domain:
                continue
            if existing.superseded_by is not None:
                continue

            # Check parameter overlap conflicts
            shared_params = set(candidate.domain_validity_range.keys()) & set(existing.domain_validity_range.keys())
            if not shared_params:
                continue

            # Calculate parameter domain overlap
            overlap: Dict[str, Tuple[float, float]] = {}
            has_domain_overlap = True
            for p in shared_params:
                c_min, c_max = candidate.domain_validity_range[p]
                e_min, e_max = existing.domain_validity_range[p]
                ov_min = max(c_min, e_min)
                ov_max = min(c_max, e_max)
                if ov_min <= ov_max:
                    overlap[p] = (ov_min, ov_max)
                else:
                    has_domain_overlap = False
                    break

            if not has_domain_overlap:
                continue

            # Check uncertainty bounds discrepancy if shared quantities exist
            shared_bounds = set(candidate.uncertainty_bounds.keys()) & set(existing.uncertainty_bounds.keys())
            for q in shared_bounds:
                c_low, c_high = candidate.uncertainty_bounds[q]
                e_low, e_high = existing.uncertainty_bounds[q]
                c_val = 0.5 * (c_low + c_high)
                e_val = 0.5 * (e_low + e_high)
                c_sigma = max(1e-6, (c_high - c_low) / 2.0)
                e_sigma = max(1e-6, (e_high - e_low) / 2.0)

                sigma_comb = (c_sigma**2 + e_sigma**2)**0.5
                discrepancy = abs(c_val - e_val) / sigma_comb

                # Flag statistical contradiction at > 3 sigma discrepancy
                if discrepancy > 3.0:
                    contradictions.append(ClaimContradiction(
                        claim_id_a=candidate.claim_id,
                        claim_id_b=existing.claim_id,
                        incompatibility_reason=(
                            f"Statistical tension on quantity '{q}': {c_val:.4f} vs {e_val:.4f} "
                            f"(discrepancy={discrepancy:.2f} sigma)"
                        ),
                        domain_overlap=overlap,
                        discrepancy_significance_sigma=round(discrepancy, 2),
                    ))

        return contradictions

    def supersede_claim(self, old_claim_id: str, new_claim: ScientificClaim) -> str:
        """
        Supersedes an existing claim with an updated claim, preserving immutable lineage.
        """
        old_claim = self._claims.get(old_claim_id)
        if not old_claim:
            raise KeyError(f"Cannot supersede non-existent claim: {old_claim_id}")

        new_claim.supersedes = old_claim_id
        old_claim.superseded_by = new_claim.claim_id

        # Update replication status if old claim was falsified or revised
        if old_claim.contradicting_evidence_ids:
            old_claim.replication_status = ReplicationStatus.FALSIFIED
        else:
            old_claim.replication_status = ReplicationStatus.REPLICATED_WITH_REVISIONS

        self.register_claim(new_claim)

        # Record lineage
        if old_claim_id not in self._lineage_graph:
            self._lineage_graph[old_claim_id] = []
        self._lineage_graph[old_claim_id].append(new_claim.claim_id)

        return new_claim.claim_id

    def update_replication_status(
        self,
        claim_id: str,
        new_status: ReplicationStatus,
        rationale: str,
    ) -> None:
        """Updates the replication status of a claim with auditable rationale."""
        claim = self._claims.get(claim_id)
        if not claim:
            raise KeyError(f"Claim {claim_id} not found in fabric")
        claim.replication_status = new_status

    def export_knowledge_summary(self) -> Dict[str, Any]:
        """Produces comprehensive summary of claims, contradictions, and replications."""
        status_counts = {st.value: 0 for st in ReplicationStatus}
        for c in self._claims.values():
            status_counts[c.replication_status.value] += 1

        return {
            "total_claims": len(self._claims),
            "total_contradictions": len(self._contradictions),
            "replication_status_breakdown": status_counts,
            "claims": [c.model_dump() for c in self._claims.values()],
            "contradictions": [ct.model_dump() for ct in self._contradictions],
            "lineage_graph": self._lineage_graph,
        }
