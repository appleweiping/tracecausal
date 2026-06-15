from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class EvidenceTier(str, Enum):
    PAPER_RESULT = "paper_result"
    OFFICIAL = "official"
    DIAGNOSTIC = "diagnostic"
    PILOT = "pilot"


@dataclass(frozen=True)
class ProjectContract:
    slug: str
    required_docs: tuple[str, ...]
    required_config_markers: tuple[str, ...]


TRACECAUSAL_CONTRACT = ProjectContract(
    slug="tracecausal",
    required_docs=(
        "docs/research_brief.md",
        "docs/experiment_protocol.md",
        "docs/baseline_contract.md",
        "docs/claim_evidence_matrix.md",
        "docs/paper_claims_status.md",
        "docs/idea_synthesis.md",
        "docs/research_plan.md",
        "docs/active_todo.md",
        "docs/milestones.md",
        "docs/data_and_evaluation_plan.md",
        "docs/motivation_ablation_hparam_plan.md",
        "docs/risks_and_blockers.md",
        "docs/definition_of_done.md",
        "docs/aris_research_refine_audit.md",
        "docs/server_runbook.md",
    ),
    required_config_markers=(
        "seeds:",
        "baselines:",
        "metrics:",
        "gates:",
        "authorized: false",
    ),
)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return contract violations for a run or design manifest."""
    errors: list[str] = []
    if manifest.get("project") != TRACECAUSAL_CONTRACT.slug:
        errors.append("project slug must be tracecausal")
    if manifest.get("server", {}).get("authorized") is not False:
        errors.append("initial scaffold must not authorize server execution")
    if manifest.get("seeds", {}).get("paper_minimum", 0) < 20:
        errors.append("paper_minimum seeds must be at least 20")
    if len(manifest.get("baselines", [])) < 3:
        errors.append("at least three baseline families are required")
    return errors


def required_paths(root: Path) -> list[Path]:
    return [root / item for item in TRACECAUSAL_CONTRACT.required_docs]
