"""Shared do-not-run guard + plan/checkpoint helpers for the v5 entrypoints.

These helpers are imported by every ``scripts/<stage>.py`` CLI wrapper. They are
deliberately pure-Python and import NO heavy/GPU dependency at module load time.

The single most important behaviour here is the authorization GUARD
(:func:`require_authorization`). A stage performs heavy work (model load, GPU,
forwards) ONLY when BOTH of the following hold:

  1. the resolved experiment config has ``server.authorized: true``; AND
  2. the operator passes the explicit ``--i-have-authorization`` CLI flag.

When either is missing the stage stays in DRY-RUN: it prints the fully-resolved
plan and exits 0, having loaded no model and touched no GPU. This mirrors
``configs/...`` ``server.authorized: false`` and the contract in
``src/tracecausal/contracts.py`` (``validate_manifest`` /
``AUTHORIZATION_GUARDED_CONFIGS``).

Resumability is backed by a per-job ``STATUS.json`` checkpoint (see
``reports/run_packet.md`` section 6 and ``experiments/queue_manifest.yaml``
``state_model``). ``--resume`` consults it; a job whose ``state == done`` is
skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root = parent of this scripts/ directory; make src/ importable for the
# pure-Python kernel modules. (No kernel is imported here at top level.)
ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

STATUS_FILENAME = "STATUS.json"

# Config keys whose presence + ``authorized: false`` is enforced by the contract.
# Re-exported for the scripts' resolved-plan printouts.
DEFAULT_CONFIG = "configs/experiments/redesign_v5_ar_lead.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Config loading (tolerant; default-deny on the authorization flag)
# --------------------------------------------------------------------------- #
def _read_config_text(config_path: Path) -> str:
    if not config_path.exists():
        raise SystemExit(
            f"config not found: {config_path} (run from the repo root; "
            f"default is {DEFAULT_CONFIG})"
        )
    return config_path.read_text(encoding="utf-8")


def load_config(config_path: Path) -> dict[str, Any]:
    """Parse a YAML config. Falls back to ``{}`` if PyYAML is unavailable.

    A parse failure or a missing PyYAML never *grants* authorization: the
    server-authorized check below defaults to ``False`` when the value cannot be
    read, so the safe (dry-run) branch is always taken on ambiguity.
    """
    text = _read_config_text(config_path)
    try:
        import yaml  # local import: not required for the default dry-run path
    except Exception:
        return {}
    try:
        parsed = yaml.safe_load(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def config_server_authorized(config_path: Path) -> bool:
    """Return ``True`` only if the config explicitly sets ``server.authorized: true``.

    Default-deny: any parse ambiguity, missing key, or non-``True`` value yields
    ``False``. As a belt-and-braces check we also require the literal
    ``authorized: false`` marker to be ABSENT from the raw text before trusting a
    parsed ``True`` (a config that still carries the frozen marker is treated as
    unauthorized regardless of parse).
    """
    cfg = load_config(config_path)
    server = cfg.get("server") if isinstance(cfg, dict) else None
    parsed_true = bool(isinstance(server, dict) and server.get("authorized") is True)
    if not parsed_true:
        return False
    text = _read_config_text(config_path)
    if "authorized: false" in text:
        return False
    return True


# --------------------------------------------------------------------------- #
# STATUS.json checkpoint (resumability)
# --------------------------------------------------------------------------- #
def read_status(output_dir: Path) -> dict[str, Any] | None:
    status_path = output_dir / STATUS_FILENAME
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_status(output_dir: Path, **fields: Any) -> Path:
    """Write/refresh the per-job STATUS.json. Creates ``output_dir`` if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / STATUS_FILENAME
    base: dict[str, Any] = {
        "state": "pending",
        "started_at": None,
        "finished_at": None,
        "output_hash": None,
        "row_count": None,
        "git_commit": None,
        "stage": None,
        "depends_on": [],
        "updated_at": _utc_now(),
    }
    existing = read_status(output_dir)
    if existing:
        base.update(existing)
    base.update(fields)
    base["updated_at"] = _utc_now()
    status_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status_path


def job_is_done(output_dir: Path) -> bool:
    status = read_status(output_dir)
    return bool(status and status.get("state") == "done")


# --------------------------------------------------------------------------- #
# The shared argparse scaffold
# --------------------------------------------------------------------------- #
def base_parser(description: str) -> argparse.ArgumentParser:
    """An argparse parser pre-seeded with the flags every stage shares."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="frozen experiment config (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory for this job's artifacts + STATUS.json",
    )
    parser.add_argument(
        "--i-have-authorization",
        action="store_true",
        help=(
            "explicit operator acknowledgement that an authorized run is intended. "
            "Heavy work also requires server.authorized==true in --config; "
            "without BOTH this is a dry-run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip this job if its STATUS.json reports state==done",
    )
    return parser


@dataclass
class GuardDecision:
    authorized: bool
    config_authorized: bool
    flag_present: bool
    config_path: Path
    reasons: list[str] = field(default_factory=list)


def evaluate_guard(args: argparse.Namespace) -> GuardDecision:
    config_path = (ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    config_authorized = config_server_authorized(config_path)
    flag_present = bool(getattr(args, "i_have_authorization", False))
    reasons: list[str] = []
    if not config_authorized:
        reasons.append(f"server.authorized is not true in {config_path.name}")
    if not flag_present:
        reasons.append("--i-have-authorization flag not passed")
    return GuardDecision(
        authorized=config_authorized and flag_present,
        config_authorized=config_authorized,
        flag_present=flag_present,
        config_path=config_path,
        reasons=reasons,
    )


def print_plan(stage: str, args: argparse.Namespace, decision: GuardDecision, plan: dict[str, Any]) -> None:
    """Emit the fully-resolved, do-nothing plan for this stage as JSON."""
    payload = {
        "stage": stage,
        "mode": "AUTHORIZED_RUN" if decision.authorized else "DRY_RUN",
        "config": str(decision.config_path),
        "config_server_authorized": decision.config_authorized,
        "i_have_authorization_flag": decision.flag_present,
        "guard_blocks": [] if decision.authorized else decision.reasons,
        "output": args.output,
        "resume": bool(getattr(args, "resume", False)),
        "loads_model_or_gpu": False if not decision.authorized else True,
        "plan": plan,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def run_stage(stage: str, args: argparse.Namespace, plan: dict[str, Any], heavy):
    """Standard stage driver.

    1. If ``--resume`` and the job is already ``done``, skip and exit 0.
    2. Always print the resolved plan.
    3. If NOT authorized (the default), exit 0 without loading anything.
    4. If authorized, mark STATUS running, call ``heavy(args, plan)`` (which
       lazy-imports GPU deps), then mark STATUS done.

    ``heavy`` is a callable taking ``(args, plan)``. It is NEVER called in the
    default dry-run path.
    """
    output_dir = Path(args.output).resolve() if args.output else None

    if getattr(args, "resume", False) and output_dir is not None and job_is_done(output_dir):
        sys.stdout.write(
            json.dumps({"stage": stage, "mode": "RESUME_SKIP",
                        "reason": "STATUS.json state==done", "output": str(output_dir)},
                       indent=2, sort_keys=True) + "\n"
        )
        return 0

    decision = evaluate_guard(args)
    print_plan(stage, args, decision, plan)

    if not decision.authorized:
        # Dry-run: print the plan (above), record an inert checkpoint, load nothing.
        if output_dir is not None:
            write_status(output_dir, stage=stage, state="pending",
                         note="dry-run plan only; not authorized; no model/GPU loaded")
        return 0

    # ---- AUTHORIZED branch (only reachable with both gates satisfied) --------
    if output_dir is None:
        raise SystemExit("authorized run requires --output for the STATUS.json checkpoint")
    write_status(output_dir, stage=stage, state="running", started_at=_utc_now())
    try:
        result = heavy(args, plan)  # heavy() does its own lazy imports of torch/transformers
    except Exception as exc:  # keep the job in the ledger; do not silently drop
        write_status(output_dir, stage=stage, state="failed",
                     finished_at=_utc_now(), error=repr(exc))
        raise
    write_status(output_dir, stage=stage, state="done", finished_at=_utc_now(),
                 output_hash=(result or {}).get("output_hash"),
                 row_count=(result or {}).get("row_count"))
    return 0
