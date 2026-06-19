"""Tests for the manifest <-> checkpoint queue reconciliation (resumability fix).

These lock in the behaviour ``reports/run_packet.md`` section 6.2 promises and that
the prior ``eval_gates.py --reconcile-queue`` failed to deliver: the pending set is
**rebuilt from the manifest** (not an empty set when no output dir exists) and is
ordered by the v5 dependency structure (``depends_on`` arm/per-seed/aggregate tokens
plus external preconditions).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _reconcile_queue import load_manifest_jobs, reconcile  # noqa: E402

_MANIFEST = Path(__file__).resolve().parents[1] / "experiments" / "queue_manifest.yaml"

EXTERNALS = [
    "server_authorized_true",
    "hashes_pinned",
    "preflight_pass",
    "v5_kernel_harness_green",
]


def _none_reader(_out_dir):
    return None


def test_reconcile_rebuilds_full_pending_set_from_manifest_when_no_output_exists():
    # The core regression: with NO output dir, the pending set is the WHOLE manifest,
    # not empty (the old behaviour returned []).
    jobs, order = load_manifest_jobs(_MANIFEST)
    assert jobs, "manifest must parse into a non-empty job list"
    res = reconcile(jobs, order, out_root=None, status_reader=_none_reader)
    assert res["total_jobs"] == len(jobs)
    # every job is pending-or-blocked (none done); the pending set covers all jobs.
    assert len(res["remaining_pending_set"]) == len(jobs)
    assert res["state_counts"]["done"] == 0


def test_reconcile_blocks_on_external_preconditions_by_default():
    # default-deny: a fresh resume reports the run-level gate tokens as blocking and
    # surfaces ONLY genuine external tokens (arm / per-seed / _all_seeds resolve to jobs).
    jobs, order = load_manifest_jobs(_MANIFEST)
    res = reconcile(jobs, order, out_root=None, status_reader=_none_reader)
    assert set(res["external_preconditions_required"]) == set(EXTERNALS)
    # with externals unmet nothing is ready
    assert res["ready_jobs"] == []
    assert len(res["blocked_jobs"]) == len(jobs)


def test_reconcile_pending_set_is_dependency_ordered():
    # extract_traces is the first pipeline stage; its jobs must sort ahead of any
    # downstream-stage job in the rebuilt pending set.
    jobs, order = load_manifest_jobs(_MANIFEST)
    res = reconcile(jobs, order, out_root=None, status_reader=_none_reader)
    by_id = {j["id"]: j for j in jobs}
    pending = res["remaining_pending_set"]
    first_extract = next(i for i, jid in enumerate(pending)
                         if by_id[jid]["stage"] == "extract_traces")
    first_gates = next((i for i, jid in enumerate(pending)
                        if by_id[jid]["stage"] == "eval_gates"), None)
    if first_gates is not None:
        assert first_extract < first_gates


def test_reconcile_unblocks_downstream_when_upstream_done():
    # Mark every extract_traces job done; with externals satisfied, the immediately
    # downstream jobs whose deps are now met become READY (and the pending set shrinks).
    jobs, order = load_manifest_jobs(_MANIFEST)
    done_dirs = {
        str(Path(j["output_artifact"]).parent)
        for j in jobs
        if j["stage"] == "extract_traces"
    }

    def reader(out_dir):
        if out_dir is not None and str(out_dir) in done_dirs:
            return {"state": "done"}
        return None

    res = reconcile(jobs, order, out_root=None, status_reader=reader,
                    satisfied_external=EXTERNALS)
    n_extract = sum(1 for j in jobs if j["stage"] == "extract_traces")
    assert res["state_counts"]["done"] == n_extract
    # at least one downstream job must now be ready (binning / nuisance fan-in)
    assert len(res["ready_jobs"]) >= 1
    ready_stages = {by["stage"] for by in jobs if by["id"] in set(res["ready_jobs"])}
    assert "extract_traces" not in ready_stages  # extracts are done, not pending
    # the pending set strictly shrank versus the all-pending baseline
    assert len(res["remaining_pending_set"]) == len(jobs) - n_extract


def test_reconcile_all_seeds_and_per_seed_tokens_resolve_to_jobs_not_externals():
    # The aggregate '<arm>_all_seeds' and per-seed '<arm>__seedN' depends_on tokens
    # must resolve against real jobs, NOT be misreported as external preconditions.
    jobs, order = load_manifest_jobs(_MANIFEST)
    res = reconcile(jobs, order, out_root=None, status_reader=_none_reader)
    ext = set(res["external_preconditions_required"])
    assert not any(t.endswith("_all_seeds") for t in ext)
    assert not any("__seed" in t for t in ext)


def test_reconcile_degrades_cleanly_on_missing_manifest(tmp_path):
    jobs, order = load_manifest_jobs(tmp_path / "does_not_exist.yaml")
    assert jobs == []
    res = reconcile(jobs, order, out_root=None, status_reader=_none_reader)
    assert res["total_jobs"] == 0
    assert res["remaining_pending_set"] == []
