"""
ASTRAEUS Autonomous Research Planning Engine (WS9.3).
Maintains long-horizon research programmes as Directed Acyclic Graphs (DAGs),
optimizes multi-resource allocation, and dynamically replans upon falsification.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field


class ResearchNodeType(str, Enum):
    """Categorical types of scientific research tasks in the DAG."""
    HYPOTHESIS_FORMULATION = "HYPOTHESIS_FORMULATION"
    COMPUTATIONAL_SIMULATION = "COMPUTATIONAL_SIMULATION"
    OBSERVATIONAL_PROPOSAL = "OBSERVATIONAL_PROPOSAL"
    STATISTICAL_INFERENCE = "STATISTICAL_INFERENCE"
    INDEPENDENT_REPLICATION_GATE = "INDEPENDENT_REPLICATION_GATE"
    SCIENTIFIC_SYNTHESIS = "SCIENTIFIC_SYNTHESIS"


class ResearchNodeStatus(str, Enum):
    """Execution state of a research DAG node."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FALSIFIED_OR_FAILED = "FALSIFIED_OR_FAILED"
    PRUNED = "PRUNED"


class ResearchDAGNode(BaseModel):
    """Individual scientific node in the autonomous research programme DAG."""
    node_id: str
    node_type: ResearchNodeType
    title: str
    description: str
    hypothesis_id: Optional[str] = None
    dependency_node_ids: List[str] = Field(default_factory=list)
    resource_cost_gpu_hours: float = 0.0
    resource_cost_telescope_hours: float = 0.0
    status: ResearchNodeStatus = ResearchNodeStatus.PENDING
    result_summary: Optional[str] = None
    falsification_evidence: Optional[str] = None
    completed_at_iso: Optional[str] = None


class ResearchProgrammeDAG(BaseModel):
    """Complete versioned research programme graph."""
    programme_id: str
    scientific_goal: str
    nodes: Dict[str, ResearchDAGNode]
    available_gpu_hours: float = 1000.0
    available_telescope_hours: float = 50.0
    used_gpu_hours: float = 0.0
    used_telescope_hours: float = 0.0
    replanning_iterations: int = 0
    created_at_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AutonomousResearchPlanner:
    """
    Orchestrates long-horizon research programmes.
    Constructs dependency DAGs, tracks resource budgets, and executes automated replanning.
    """

    def create_transient_research_programme(
        self,
        programme_id: str,
        target_phenomenon: str,
    ) -> ResearchProgrammeDAG:
        """
        Synthesizes a structured multi-stage discovery research programme DAG.
        """
        nodes: Dict[str, ResearchDAGNode] = {}

        # Stage 1: Formulate Competing Hypotheses
        n1 = ResearchDAGNode(
            node_id="node_hyp_csm",
            node_type=ResearchNodeType.HYPOTHESIS_FORMULATION,
            title="Formulate CSM Shock Interaction Hypothesis",
            description="Derive shock breakout conditions and dense wind light curve models.",
            hypothesis_id="hyp_csm_breakout",
            resource_cost_gpu_hours=2.0,
        )
        n2 = ResearchDAGNode(
            node_id="node_hyp_kilonova",
            node_type=ResearchNodeType.HYPOTHESIS_FORMULATION,
            title="Formulate Radioactive Kilonova Hypothesis",
            description="Derive r-process heating and opacity models for BNS merger.",
            hypothesis_id="hyp_radioactive_kilonova",
            resource_cost_gpu_hours=2.0,
        )
        nodes[n1.node_id] = n1
        nodes[n2.node_id] = n2

        # Stage 2: Computational Forward Simulations
        n3 = ResearchDAGNode(
            node_id="node_sim_csm",
            node_type=ResearchNodeType.COMPUTATIONAL_SIMULATION,
            title="Simulate CSM Grid and Light Curves",
            description="Compute multi-band light curves across ejecta and CSM mass parameter space.",
            dependency_node_ids=[n1.node_id],
            hypothesis_id=n1.hypothesis_id,
            resource_cost_gpu_hours=25.0,
        )
        n4 = ResearchDAGNode(
            node_id="node_sim_kilonova",
            node_type=ResearchNodeType.COMPUTATIONAL_SIMULATION,
            title="Simulate Radiative Transfer Kilonova Grid",
            description="Execute multi-band Monte Carlo photon transport on 2-component kilonova.",
            dependency_node_ids=[n2.node_id],
            hypothesis_id=n2.hypothesis_id,
            resource_cost_gpu_hours=30.0,
        )
        nodes[n3.node_id] = n3
        nodes[n4.node_id] = n4

        # Stage 3: Discriminating Observation
        n5 = ResearchDAGNode(
            node_id="node_obs_discriminating",
            node_type=ResearchNodeType.OBSERVATIONAL_PROPOSAL,
            title="Discriminating Multi-Facility Follow-Up",
            description="Execute VLT NIR spectroscopy and VLA 5GHz radio pointings.",
            dependency_node_ids=[n3.node_id, n4.node_id],
            resource_cost_telescope_hours=3.5,
        )
        nodes[n5.node_id] = n5

        # Stage 4: Independent Replication & Scientific Synthesis
        n6 = ResearchDAGNode(
            node_id="node_replication_gate",
            node_type=ResearchNodeType.INDEPENDENT_REPLICATION_GATE,
            title="Independent Replication and Skeptic Audit",
            description="Audit exact numerical reproducibility and verify evidence integrity.",
            dependency_node_ids=[n5.node_id],
            resource_cost_gpu_hours=10.0,
        )
        n7 = ResearchDAGNode(
            node_id="node_scientific_synthesis",
            node_type=ResearchNodeType.SCIENTIFIC_SYNTHESIS,
            title="Scientific Publication Dossier Synthesis",
            description="Synthesize final discovery dossier and update knowledge fabric.",
            dependency_node_ids=[n6.node_id],
            resource_cost_gpu_hours=5.0,
        )
        nodes[n6.node_id] = n6
        nodes[n7.node_id] = n7

        return ResearchProgrammeDAG(
            programme_id=programme_id,
            scientific_goal=f"Determine physical engine powering {target_phenomenon}",
            nodes=nodes,
            available_gpu_hours=500.0,
            available_telescope_hours=20.0,
        )

    def trigger_falsification_replanning(
        self,
        dag: ResearchProgrammeDAG,
        falsified_node_id: str,
        falsification_evidence: str,
    ) -> ResearchProgrammeDAG:
        """
        Dynamically replans the research DAG upon empirical falsification:
        1. Marks the failed node as FALSIFIED_OR_FAILED.
        2. Prunes downstream nodes uniquely dependent on the falsified branch.
        3. Generates a new refined hypothesis node addressing the falsification evidence.
        4. Reconnects dependencies to preserve research progress.
        """
        updated_dag = copy.deepcopy(dag)
        updated_dag.replanning_iterations += 1

        failed_node = updated_dag.nodes.get(falsified_node_id)
        if not failed_node:
            raise KeyError(f"Node {falsified_node_id} not found in research DAG")

        failed_node.status = ResearchNodeStatus.FALSIFIED_OR_FAILED
        failed_node.falsification_evidence = falsification_evidence

        # Identify all nodes downstream of this failed node
        pruned_nodes: Set[str] = set()
        for nid, node in updated_dag.nodes.items():
            if falsified_node_id in node.dependency_node_ids and node.node_type == ResearchNodeType.COMPUTATIONAL_SIMULATION:
                node.status = ResearchNodeStatus.PRUNED
                pruned_nodes.add(nid)

        # Create refined hypothesis node
        mutant_id = f"{falsified_node_id}_refined_v2"
        mutant_node = ResearchDAGNode(
            node_id=mutant_id,
            node_type=ResearchNodeType.HYPOTHESIS_FORMULATION,
            title=f"Refined Hypothesis: {failed_node.title} (v2)",
            description=f"Refined formulation incorporating falsification evidence: {falsification_evidence}",
            hypothesis_id=f"{failed_node.hypothesis_id}_v2",
            resource_cost_gpu_hours=failed_node.resource_cost_gpu_hours,
            status=ResearchNodeStatus.PENDING,
        )
        updated_dag.nodes[mutant_id] = mutant_node

        # Reroute general downstream nodes (e.g. discriminating observation) to depend on survivor or mutant
        for nid, node in updated_dag.nodes.items():
            if falsified_node_id in node.dependency_node_ids:
                node.dependency_node_ids = [
                    mutant_id if dep == falsified_node_id else dep
                    for dep in node.dependency_node_ids
                ]

        return updated_dag

    def get_ready_nodes(self, dag: ResearchProgrammeDAG) -> List[ResearchDAGNode]:
        """Returns all nodes whose dependencies have successfully completed."""
        ready = []
        for node in dag.nodes.values():
            if node.status != ResearchNodeStatus.PENDING:
                continue

            all_deps_done = True
            for dep_id in node.dependency_node_ids:
                dep_node = dag.nodes.get(dep_id)
                if not dep_node or dep_node.status != ResearchNodeStatus.COMPLETED:
                    all_deps_done = False
                    break

            if all_deps_done:
                ready.append(node)
        return ready
