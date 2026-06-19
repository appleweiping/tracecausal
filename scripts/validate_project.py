from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecausal.contracts import TRACECAUSAL_CONTRACT, required_paths


def require_markers(path: Path, markers: list[str], label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        for marker in missing:
            print(f"missing {label} marker: {marker}")
        return False
    return True


def main() -> int:
    missing = [path for path in required_paths(ROOT) if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing required doc: {path}")
        return 1

    config = ROOT / "configs" / "experiments" / "formal_tracecausal.yaml"
    text = config.read_text(encoding="utf-8")
    missing_markers = [
        marker for marker in TRACECAUSAL_CONTRACT.required_config_markers if marker not in text
    ]
    if missing_markers:
        for marker in missing_markers:
            print(f"missing config marker: {marker}")
        return 1

    seeds = [
        line.strip()
        for line in (ROOT / "configs" / "seeds" / "paper_20.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if seeds != [str(index) for index in range(20)]:
        print("paper seed manifest must be exactly 0..19")
        return 1

    if not require_markers(
        ROOT / "configs" / "compute" / "first_gate_budget.yaml",
        ["buffer_percent: 30", "wall_clock_hours:", "storage_gb:"],
        "compute",
    ):
        return 1

    if not require_markers(
        ROOT / "configs" / "baselines" / "baseline_registry.yaml",
        [
            "TraceDet",
            "TDGNet",
            "SelfCheckGPT",
            "INSIDE",
            "semantic_entropy",
            "reasoning_consistency_detector",
            "paper_url:",
            "implementation_source:",
            "tuning_grid:",
            "input_access:",
            "fairness:",
        ],
        "baseline provenance",
    ):
        return 1

    if not require_markers(
        ROOT / "configs" / "experiments" / "first_gate.yaml",
        [
            "negative_controls:",
            "evaluator_leakage:",
            "authorized: false",
            "causal_margin_abs:",
            "utility_drop_abs:",
            "leakage_check:",
        ],
        "first-gate",
    ):
        return 1

    # The hardened v4 plan must also keep server execution unauthorized.
    if not require_markers(
        ROOT / "configs" / "experiments" / "redesign_v4_ar_lead.yaml",
        ["authorized: false"],
        "redesign-v4",
    ):
        return 1

    # The frozen v5 lead plan (the current source of truth: repair-transfer
    # certification + selective inference + Axis X') must likewise keep server
    # execution unauthorized AND carry the corrected v5 inference-plan markers, so a
    # silent regression to v4-style statistics (e.g. the symmetric (4/n_eff)*zeta_1
    # shorthand the kernel no longer uses) is caught by the local validator.
    if not require_markers(
        ROOT / "configs" / "experiments" / "redesign_v5_ar_lead.yaml",
        [
            "authorized: false",
            "status: design_frozen_stage1_RR",
            # headline v5 repair-transfer certification surface
            "g9_repair_transfer:",
            "g9_novelty:",
            "two_way_source_target_cluster_bootstrap",
            # corrected ordered-kernel two-projection variance (findings 4, 10): the
            # v5 plan must NOT carry the superseded (4/n_eff)*zeta_1 wording.
            "zeta_10/n_source + zeta_01/n_target",
            "r_power_repair",
            # the sign-flip null is a DIAGNOSTIC, not the confirmatory test (G9-FIX)
            "class_block_source_block_signflip_diagnostic",
            # selective-inference fold + repair-margin attenuation (Eq. m-R)
            "selection_split: required",
            "alpha_1' = 0.05 / (m' * K_bin * K_op)",
        ],
        "redesign-v5",
    ):
        return 1

    if not require_markers(
        ROOT / "docs" / "intervention_protocol.md",
        [
            "random_non_causal_segment",
            "shuffled_trace_segment",
            "no_op_intervention",
            "evaluator leakage",
        ],
        "intervention protocol",
    ):
        return 1

    if not require_markers(
        ROOT / "schemas" / "trace_manifest.schema.json",
        ["\"server_authorized\"", "\"const\": false", "\"trace_segments\"", "\"split_hash\""],
        "trace schema",
    ):
        return 1

    print("TraceCausal local contract validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
