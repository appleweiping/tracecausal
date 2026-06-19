"""Regression tests for the v5 lead-plan integrity (config + queue manifest).

These lock in three final-gate fixes:

1. ``configs/experiments/redesign_v5_ar_lead.yaml`` is covered by the contract
   guard (``validate_manifest`` config-text enforcement) AND carries the corrected
   v5 inference-plan markers -- not only v4/formal config. The frozen v5 plan must
   stay ``authorized: false`` and must NOT carry the superseded symmetric
   ``(4/n_eff)*zeta_1`` variance shorthand the kernel abandoned (findings 4, 10).

2. ``scripts/_gen_v5_manifest.py`` regenerates ``experiments/queue_manifest.yaml``
   **byte-for-byte** (idempotency). A previously-stale generator carried v4-style
   statistical wording (``hajek_projection_zeta_1`` / ``(4/n_eff)*zeta_1`` /
   ``zeta_1: DATA_NEEDED``) that would REGRESS the committed v5 manifest if rerun;
   this pins that they agree so the regression cannot recur silently.

3. The committed manifest itself carries the corrected v5 ordered-kernel
   two-projection wording and never the superseded shorthand.

All pure Python / pure text; no model, no GPU, no run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecausal.contracts import (  # noqa: E402
    AUTHORIZATION_GUARDED_CONFIGS,
    validate_manifest,
)

V5_CONFIG = ROOT / "configs" / "experiments" / "redesign_v5_ar_lead.yaml"
MANIFEST = ROOT / "experiments" / "queue_manifest.yaml"
GENERATOR = ROOT / "scripts" / "_gen_v5_manifest.py"

# The superseded v4-style statistical wording that must not be *used* as the active
# variance identity in the v5 plan or the committed manifest (the kernel explicitly
# abandoned it; findings 4, 10). We check for active *use* (the shorthand appearing
# as the value of a ``sigma_r_decomposition:`` key, or as the crosscheck token), not
# for mere mentions in explanatory comments that say the shorthand is NOT used.
STALE_DECOMPOSITION = "Var(R_hat) = (4/n_eff)*zeta_1"
STALE_CROSSCHECK = "variance_crosscheck: hajek_projection_zeta_1"


def _active_lines(text: str) -> list[str]:
    """Config/manifest lines that are NOT pure comments (a ``key: value`` payload).

    A leading-``#`` line is an explanatory comment; the abandoned-shorthand string is
    allowed there (the comment documents that it is NOT used). Active YAML payload
    lines are the ones that must never carry the superseded shorthand.
    """
    out: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
    return out

# The corrected ordered-kernel two-projection wording the v5 plan must carry.
V5_INFERENCE_MARKERS = (
    "zeta_10/n_source + zeta_01/n_target",
    "two_way_source_target_cluster_bootstrap",
    "class_block_source_block_signflip_diagnostic",
    "r_power_repair",
)


def _ok_manifest():
    return {
        "project": "tracecausal",
        "server": {"authorized": False},
        "seeds": {"paper_minimum": 20},
        "baselines": ["random_segment", "output_entropy", "semantic_entropy"],
    }


def test_v5_config_is_an_authorization_guarded_config():
    # The frozen v5 lead plan must be in the guard set (not only v4 / formal).
    assert "redesign_v5_ar_lead.yaml" in AUTHORIZATION_GUARDED_CONFIGS


def test_v5_config_text_enforced_authorized_false():
    # A v5 config flipped to authorized: true must fail the contract.
    errors = validate_manifest(
        _ok_manifest(),
        config_texts={"redesign_v5_ar_lead.yaml": "server:\n  authorized: true\n"},
    )
    assert any("redesign_v5_ar_lead.yaml" in e for e in errors)

    # The real on-disk v5 config text must pass the guard.
    clean = validate_manifest(
        _ok_manifest(),
        config_texts={"redesign_v5_ar_lead.yaml": V5_CONFIG.read_text(encoding="utf-8")},
    )
    assert clean == []


def test_v5_config_carries_corrected_inference_markers_not_stale_shorthand():
    text = V5_CONFIG.read_text(encoding="utf-8")
    assert "authorized: false" in text
    assert "status: design_frozen_stage1_RR" in text
    for marker in V5_INFERENCE_MARKERS:
        assert marker in text, f"v5 config missing inference marker: {marker}"
    active = "\n".join(_active_lines(text))
    assert STALE_DECOMPOSITION not in active, (
        "v5 config uses the superseded (4/n_eff)*zeta_1 variance identity"
    )
    assert STALE_CROSSCHECK not in active, (
        "v5 config uses the superseded hajek_projection_zeta_1 crosscheck"
    )


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_v5_manifest", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_generator_regenerates_committed_manifest_byte_for_byte(tmp_path, monkeypatch):
    # Run the generator into a temp file (NOT the committed path) and require the
    # output to equal the committed manifest exactly. This catches a stale generator
    # that would regress the v5 manifest (the original final-gate finding).
    committed = MANIFEST.read_text(encoding="utf-8")

    out_file = tmp_path / "queue_manifest.yaml"
    real_open = open

    def _redirect_open(path, *args, **kwargs):
        # The generator writes to the committed manifest path; redirect that single
        # write to the temp file so the test never mutates the repo artifact.
        if str(path).replace("\\", "/").endswith("experiments/queue_manifest.yaml"):
            return real_open(out_file, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    mod = _load_generator()
    monkeypatch.setattr("builtins.open", _redirect_open)
    mod.main()

    regenerated = out_file.read_text(encoding="utf-8")
    assert regenerated == committed, (
        "scripts/_gen_v5_manifest.py no longer regenerates the committed "
        "experiments/queue_manifest.yaml byte-for-byte; a stale generator would "
        "regress the v5 inference plan."
    )


def test_committed_manifest_has_v5_wording_not_stale_shorthand():
    text = MANIFEST.read_text(encoding="utf-8")
    assert "zeta_10/n_source + zeta_01/n_target" in text
    active = "\n".join(_active_lines(text))
    assert STALE_DECOMPOSITION not in active, (
        "committed manifest uses the superseded (4/n_eff)*zeta_1 variance identity"
    )
    assert STALE_CROSSCHECK not in active, (
        "committed manifest uses the superseded hajek_projection_zeta_1 crosscheck"
    )
