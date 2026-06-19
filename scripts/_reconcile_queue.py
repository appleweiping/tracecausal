"""Manifest <-> checkpoint reconciliation for ``eval_gates.py --reconcile-queue``.

Pure, import-light bookkeeping (no model, no GPU, no network). Rebuilds the
**pending job set** from ``experiments/queue_manifest.yaml`` reconciled against the
on-disk per-job ``STATUS.json`` checkpoints, and orders it by the v5 pipeline
dependency structure -- the behaviour ``reports/run_packet.md`` section 6.2 promises
("emits the remaining pending set, respecting v5 dependency order"). It authorizes
nothing and never loads a kernel.

Dependency model (read from the manifest itself, not hard-coded):

* Each job advertises a coarse ``stage`` (ordered by the manifest's ``pipeline_order``)
  and an ``arm`` aggregate token. Downstream jobs list upstream **arm tokens** (and a
  few external gate tokens such as ``server_authorized_true`` / ``hashes_pinned`` /
  ``preflight_pass``) in ``depends_on``.
* An arm token is **satisfied** iff *every* job advertising that ``arm`` is ``done``.
* The manifest's seed-fan-out jobs advertise a per-(family,dataset) ``arm`` (e.g.
  ``extract__ar_lead_qwen__triviaqa``) while downstream stages depend on the
  **aggregate-over-seeds** token ``<arm>_all_seeds``. An ``_all_seeds`` token is
  therefore satisfied iff every job whose ``arm`` equals the base (suffix-stripped)
  arm is ``done`` -- it is NOT an external precondition.
* External gate tokens (tokens that are neither an ``arm`` nor a per-job id) are
  treated as **preconditions**: satisfied only when listed in ``satisfied_external``
  (default: none -- so a fresh resume honestly reports them as blocking until the
  operator re-runs preflight). They are surfaced explicitly, never silently assumed.

A pending job is **ready** iff all its dependencies are satisfied, else **blocked**.
The emitted pending set is sorted by ``(pipeline_order index, id)`` so the queue is
re-issued in dependency order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# These external (non-job) depends_on tokens are run-level preconditions, not jobs.
# They are reported separately and treated as UNSATISFIED unless the caller passes
# them in ``satisfied_external`` (default-deny: a resume must re-establish them).
KNOWN_EXTERNAL_TOKENS = (
    "server_authorized_true",
    "hashes_pinned",
    "preflight_pass",
)

DONE_STATE = "done"

# Downstream stages depend on an aggregate-over-seeds token formed by appending this
# suffix to a base (family,dataset) arm; e.g. depends_on ``extract__..._all_seeds``
# means "every job with arm ``extract__...`` is done".
ALL_SEEDS_SUFFIX = "_all_seeds"
# A per-seed dependency token is ``<arm>__seed<N>`` and resolves to the single job
# with that ``arm`` and ``seed`` (e.g. ``extract__ar_lead_qwen__triviaqa__seed0``).
PER_SEED_SEP = "__seed"


def load_manifest_jobs(manifest_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(jobs, pipeline_order)`` parsed from the manifest.

    Falls back to an empty result (rather than raising) if PyYAML is unavailable or
    the manifest cannot be parsed, so the reconcile mode degrades to a clear,
    non-crashing report. ``pipeline_order`` is the manifest's declared stage order if
    present, else the stage first-appearance order over the jobs.
    """
    if not manifest_path.exists():
        return [], []
    try:
        import yaml  # local import: reconcile is pure bookkeeping
    except Exception:
        return [], []
    try:
        parsed = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    if not isinstance(parsed, Mapping):
        return [], []
    raw_jobs = parsed.get("jobs")
    jobs = [dict(j) for j in raw_jobs if isinstance(j, Mapping)] if isinstance(raw_jobs, list) else []
    pipeline_order = parsed.get("pipeline_order")
    if not (isinstance(pipeline_order, list) and all(isinstance(s, str) for s in pipeline_order)):
        # derive first-appearance stage order
        seen: list[str] = []
        for j in jobs:
            s = j.get("stage")
            if isinstance(s, str) and s not in seen:
                seen.append(s)
        pipeline_order = seen
    return jobs, list(pipeline_order)


def _job_output_dir(job: Mapping[str, Any], out_root: Path | None) -> Path | None:
    """Resolve the directory holding a job's ``STATUS.json``.

    The job's ``output_artifact`` is a *file* path (e.g.
    ``traces/<arm>__seed/trace_manifest.json``); its parent directory holds the
    ``STATUS.json`` checkpoint. Resolved under ``out_root`` when given.
    """
    artifact = job.get("output_artifact")
    if not isinstance(artifact, str) or not artifact:
        return None
    rel_parent = Path(artifact).parent
    if out_root is not None:
        return (out_root / rel_parent)
    return rel_parent


def _stage_rank(stage: Any, pipeline_order: Sequence[str]) -> int:
    if isinstance(stage, str) and stage in pipeline_order:
        return pipeline_order.index(stage)
    # unknown / aggregate stages sort after the known pipeline
    return len(pipeline_order)


def reconcile(
    jobs: Sequence[Mapping[str, Any]],
    pipeline_order: Sequence[str],
    *,
    out_root: Path | None,
    status_reader,
    satisfied_external: Iterable[str] = (),
) -> dict[str, Any]:
    """Reconcile manifest jobs against on-disk STATUS.json; emit the ordered pending set.

    ``status_reader`` is a callable ``output_dir -> dict | None`` (the per-job
    ``STATUS.json`` payload, or ``None`` if absent). Injected so the core logic is
    testable without touching disk.

    Returns a JSON-serializable payload with per-state job-id buckets, the **ordered
    pending set** (``ready`` first, then ``blocked``, each in dependency order),
    per-job blocking reasons, and the unsatisfied external preconditions.
    """
    satisfied_external = set(satisfied_external)

    # 1. Resolve each job's current state from its checkpoint.
    job_state: dict[str, str] = {}
    arm_of: dict[str, str] = {}
    jobs_by_arm: dict[str, list[str]] = {}
    jobs_by_arm_seed: dict[tuple[str, str], list[str]] = {}
    job_by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        jid = str(job.get("id"))
        job_by_id[jid] = dict(job)
        out_dir = _job_output_dir(job, out_root)
        status = status_reader(out_dir) if out_dir is not None else None
        state = (status or {}).get("state", "pending") if isinstance(status, Mapping) else "pending"
        if state not in ("pending", "running", "done", "failed", "oom_retry", "blocked"):
            state = "pending"
        job_state[jid] = state
        arm = job.get("arm")
        if isinstance(arm, str):
            arm_of[jid] = arm
            jobs_by_arm.setdefault(arm, []).append(jid)
            seed = job.get("seed")
            if seed is not None:
                jobs_by_arm_seed.setdefault((arm, str(seed)), []).append(jid)

    # 2. An arm token is satisfied iff every job advertising that arm is done.
    arm_satisfied: dict[str, bool] = {
        arm: all(job_state.get(j) == DONE_STATE for j in members)
        for arm, members in jobs_by_arm.items()
    }
    all_job_ids = set(job_by_id)

    def _resolve_per_seed_members(token: str) -> list[str] | None:
        """Resolve a ``<arm>__seed<N>`` token to its specific job id(s), if any."""
        if PER_SEED_SEP not in token:
            return None
        base, _, seed = token.rpartition(PER_SEED_SEP)
        members = jobs_by_arm_seed.get((base, seed))
        return members if members else None

    def _resolve_arm_token(token: str) -> str | None:
        """Map a depends_on token to a known arm, resolving the ``_all_seeds`` aggregate."""
        if token in jobs_by_arm:
            return token
        if token.endswith(ALL_SEEDS_SUFFIX):
            base = token[: -len(ALL_SEEDS_SUFFIX)]
            if base in jobs_by_arm:
                return base
        return None

    def _is_job_or_arm_token(token: str) -> bool:
        return (
            token in all_job_ids
            or _resolve_arm_token(token) is not None
            or _resolve_per_seed_members(token) is not None
        )

    def _dep_satisfied(token: str) -> bool:
        arm = _resolve_arm_token(token)
        if arm is not None:
            return arm_satisfied[arm]
        per_seed = _resolve_per_seed_members(token)
        if per_seed is not None:
            return all(job_state.get(j) == DONE_STATE for j in per_seed)
        if token in all_job_ids:
            return job_state.get(token) == DONE_STATE
        # external precondition / unknown token: default-deny unless declared satisfied
        return token in satisfied_external

    # 3. Bucket jobs; classify pending ones as ready vs blocked with reasons.
    buckets: dict[str, list[str]] = {
        "done": [], "running": [], "failed": [], "pending": [], "blocked": [],
    }
    blocking_reasons: dict[str, list[str]] = {}
    external_required: set[str] = set()

    for jid, job in job_by_id.items():
        state = job_state[jid]
        if state == DONE_STATE:
            buckets["done"].append(jid)
            continue
        if state == "running":
            buckets["running"].append(jid)
            continue
        if state == "failed":
            buckets["failed"].append(jid)
            # failed jobs are pending re-work; continue to dependency analysis below

        deps = job.get("depends_on") or []
        unmet: list[str] = []
        for token in deps:
            if not isinstance(token, str):
                continue
            if not _is_job_or_arm_token(token):
                external_required.add(token)
            if not _dep_satisfied(token):
                unmet.append(token)
        if unmet:
            blocking_reasons[jid] = sorted(set(unmet))
            buckets["blocked"].append(jid)
        else:
            buckets["pending"].append(jid)

    # 4. Order the pending set: ready (unblocked) first, then blocked, each by
    #    (pipeline-stage rank, id) so the queue is re-issued in dependency order.
    def _key(jid: str):
        return (_stage_rank(job_by_id[jid].get("stage"), pipeline_order), jid)

    ready_ordered = sorted(buckets["pending"], key=_key)
    blocked_ordered = sorted(buckets["blocked"], key=_key)
    pending_set = ready_ordered + blocked_ordered

    unsatisfied_external = sorted(t for t in external_required if t not in satisfied_external)

    return {
        "total_jobs": len(job_by_id),
        "pipeline_order": list(pipeline_order),
        "state_counts": {k: len(v) for k, v in buckets.items()},
        "done": sorted(buckets["done"], key=_key),
        "running": sorted(buckets["running"], key=_key),
        "failed": sorted(buckets["failed"], key=_key),
        # the rebuilt pending set, in dependency order (ready then blocked)
        "remaining_pending_set": pending_set,
        "ready_jobs": ready_ordered,
        "blocked_jobs": blocked_ordered,
        "blocking_reasons": blocking_reasons,
        "external_preconditions_required": unsatisfied_external,
        "satisfied_external": sorted(satisfied_external),
    }
