"""
ASTRAEUS Unified Scientific Evidence Fabric (WS10.3).
Provides an immutable, tamper-evident directed acyclic graph (DAG)
recording complete scientific reasoning lineage:
Observation -> Measurement -> Claim -> Hypothesis -> Model ->
Prediction -> Experiment -> Falsification -> Result -> Replication -> Decision.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field


class EvidenceNodeType(str, Enum):
    OBSERVATION = "OBSERVATION"
    MEASUREMENT = "MEASUREMENT"
    CLAIM = "CLAIM"
    HYPOTHESIS = "HYPOTHESIS"
    MODEL = "MODEL"
    PREDICTION = "PREDICTION"
    EXPERIMENT = "EXPERIMENT"
    FALSIFICATION = "FALSIFICATION"
    RESULT = "RESULT"
    REPLICATION = "REPLICATION"
    DECISION = "DECISION"


class EvidenceNode(BaseModel):
    """Immutable node in the unified scientific evidence fabric."""
    node_id: str
    node_type: EvidenceNodeType
    parent_node_ids: List[str] = Field(default_factory=list)
    payload: Dict[str, Any]
    payload_hash: str
    node_hash: str
    timestamp_epoch_sec: float
    timestamp_iso: str
    author_or_agent: str
    custodian_signature_hex: Optional[str] = None


class UnifiedEvidenceFabric:
    """
    Unified evidence fabric DAG manager.
    Tracks end-to-end scientific lineage with SHA-256 cryptographic chaining,
    enabling deterministic replay and tamper audit.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, EvidenceNode] = {}
        self.children: Dict[str, List[str]] = {}

    def add_node(
        self,
        node_type: EvidenceNodeType,
        payload: Dict[str, Any],
        parent_node_ids: Optional[List[str]] = None,
        author_or_agent: str = "ASTRAEUS_SCIENTIFIC_COLLECTIVE",
        custom_node_id: Optional[str] = None,
        custodian_signature_hex: Optional[str] = None,
    ) -> EvidenceNode:
        """
        Creates and registers a new evidence node, mathematically binding its parents
        and contents into its node_hash.
        """
        parents = parent_node_ids or []
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"PARENT NOT FOUND: Parent node '{p}' does not exist in evidence fabric.")

        now_dt = datetime.now(timezone.utc)
        epoch_sec = now_dt.timestamp()
        iso_str = now_dt.isoformat()

        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        node_id = custom_node_id or f"{node_type.value.lower()}_{len(self.nodes) + 1}_{payload_hash[:8]}"

        # Cryptographically link parents and payload into node hash
        sorted_parents = sorted(parents)
        chain_bytes = f"{node_id}:{node_type.value}:{':'.join(sorted_parents)}:{payload_hash}:{epoch_sec}".encode("utf-8")
        node_hash = hashlib.sha256(chain_bytes).hexdigest()

        node = EvidenceNode(
            node_id=node_id,
            node_type=node_type,
            parent_node_ids=sorted_parents,
            payload=payload,
            payload_hash=payload_hash,
            node_hash=node_hash,
            timestamp_epoch_sec=epoch_sec,
            timestamp_iso=iso_str,
            author_or_agent=author_or_agent,
            custodian_signature_hex=custodian_signature_hex,
        )

        self.nodes[node_id] = node
        for p in sorted_parents:
            self.children.setdefault(p, []).append(node_id)

        return node

    def get_ancestors_lineage(self, node_id: str) -> List[EvidenceNode]:
        """
        Traverses upward to retrieve all ancestor nodes reaching back to raw observations.
        Returns ancestors in topological order (earliest root ancestors first).
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in evidence fabric.")

        visited: Set[str] = set()
        topo_order: List[str] = []

        def dfs(curr_id: str):
            if curr_id in visited:
                return
            visited.add(curr_id)
            node = self.nodes[curr_id]
            for pid in node.parent_node_ids:
                dfs(pid)
            topo_order.append(curr_id)

        dfs(node_id)
        return [self.nodes[nid] for nid in topo_order]

    def verify_lineage_integrity(self, target_node_id: str) -> Tuple[bool, List[str]]:
        """
        Verifies cryptographic integrity of the entire lineage path leading to target_node_id:
        1. Checks payload_hash equality against stored payload.
        2. Checks node_hash recomputation against parents, payload, and timestamp.
        3. Confirms that all ancestor links are valid.
        """
        lineage = self.get_ancestors_lineage(target_node_id)
        notes = []
        is_valid = True

        for node in lineage:
            # Recompute payload hash
            pl_bytes = json.dumps(node.payload, sort_keys=True).encode("utf-8")
            recomputed_pl_hash = hashlib.sha256(pl_bytes).hexdigest()
            if recomputed_pl_hash != node.payload_hash:
                is_valid = False
                notes.append(f"PAYLOAD TAMPERING at node {node.node_id}: {recomputed_pl_hash} != {node.payload_hash}")

            # Recompute node hash
            chain_bytes = f"{node.node_id}:{node.node_type.value}:{':'.join(node.parent_node_ids)}:{node.payload_hash}:{node.timestamp_epoch_sec}".encode("utf-8")
            recomputed_node_hash = hashlib.sha256(chain_bytes).hexdigest()
            if recomputed_node_hash != node.node_hash:
                is_valid = False
                notes.append(f"NODE HASH TAMPERING at node {node.node_id}: {recomputed_node_hash} != {node.node_hash}")

        return is_valid, notes

    def export_mermaid_flowchart(self) -> str:
        """Generates Mermaid flowchart diagram for visualization in reports."""
        lines = ["flowchart TD"]
        for nid, node in self.nodes.items():
            safe_type = node.node_type.value
            lines.append(f'    {nid}["{safe_type}: {nid}"]')
            for pid in node.parent_node_ids:
                lines.append(f"    {pid} --> {nid}")
        return "\n".join(lines)
