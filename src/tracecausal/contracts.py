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
        "docs/pre_registration.md",
        "docs/statistical_analysis_plan.md",
        "docs/compute_budget.md",
        "docs/reproducibility_ledger.md",
        "docs/intervention_protocol.md",
        "docs/literature_boundary.md",
        "docs/aris_experiment_plan_review.md",
        "reports/schema_validation/trace_schema.json",
        "reports/experiment_plan/aris_plan.md",
        "reports/adversarial_review/round1_response.md",
        "reports/aris_9_8_scorecard.md",
        "schemas/trace_manifest.schema.json",
        "configs/seeds/paper_20.txt",
        "configs/compute/first_gate_budget.yaml",
        "configs/experiments/first_gate.yaml",
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


# Experiment configs whose raw text must carry the ``authorized: false`` marker
# whenever it is supplied to ``validate_manifest`` (config-text enforcement). The
# hardened v4 plan (``redesign_v4_ar_lead.yaml``) is enforced alongside the formal
# config so the v4 lock cannot ship a config that flips ``server.authorized`` on.
AUTHORIZATION_GUARDED_CONFIGS: tuple[str, ...] = (
    "formal_tracecausal.yaml",
    "redesign_v4_ar_lead.yaml",
    "redesign_v5_ar_lead.yaml",
)


def validate_manifest(
    manifest: dict[str, Any],
    *,
    config_texts: dict[str, str] | None = None,
) -> list[str]:
    """Return contract violations for a run or design manifest.

    When ``config_texts`` is supplied (a mapping of config filename -> raw text),
    every guarded experiment config in :data:`AUTHORIZATION_GUARDED_CONFIGS` that
    appears in the mapping is required to carry the literal ``authorized: false``
    marker. This extends the ``server.authorized`` guard from the parsed manifest
    to the on-disk hardened v4 plan (``redesign_v4_ar_lead.yaml``), not only the
    formal config, so a v4-lock config cannot silently re-authorize a run. Configs
    not present in the mapping are not checked here (the caller decides which texts
    to hand in); a guarded config present but flipped to ``authorized: true`` fails.
    """
    errors: list[str] = []
    if manifest.get("project") != TRACECAUSAL_CONTRACT.slug:
        errors.append("project slug must be tracecausal")
    if manifest.get("server", {}).get("authorized") is not False:
        errors.append("initial scaffold must not authorize server execution")
    if manifest.get("seeds", {}).get("paper_minimum", 0) < 20:
        errors.append("paper_minimum seeds must be at least 20")
    if len(manifest.get("baselines", [])) < 3:
        errors.append("at least three baseline families are required")

    if config_texts:
        for name in AUTHORIZATION_GUARDED_CONFIGS:
            text = config_texts.get(name)
            if text is None:
                continue
            if "authorized: false" not in text:
                errors.append(
                    f"{name} must carry 'authorized: false' (server execution "
                    "must not be authorized)"
                )
            if "authorized: true" in text:
                errors.append(f"{name} must not set 'authorized: true'")
    return errors


def required_paths(root: Path) -> list[Path]:
    return [root / item for item in TRACECAUSAL_CONTRACT.required_docs]
