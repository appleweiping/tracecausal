from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Paradigm = Literal["autoregressive", "reasoning_trace", "diffusion_lm"]


@dataclass(frozen=True)
class TraceStep:
    step_id: str
    paradigm: Paradigm
    text_span: tuple[int, int]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceSegment:
    segment_id: str
    step_ids: tuple[str, ...]
    selector: str
    score: float
    intervention: str


@dataclass(frozen=True)
class TraceManifest:
    project: str
    dataset_id: str
    model_id: str
    split: str
    trace_steps: tuple[TraceStep, ...]
    candidate_segments: tuple[TraceSegment, ...]
    seed: int
    server_authorized: bool = False
    # --- common-trace-schema reconciliation (trace_manifest.schema.json) ----
    # Added as defaulted trailing fields so existing positional constructions
    # remain valid, while validate_trace_manifest enforces the schema's
    # `required` + constraints (split_hash minLength 8; trace_segments minItems 1).
    #
    # split_hash pins the frozen train/valid/test split identity (schema:10/22).
    split_hash: str = ""
    # trace_segments are the realised trace segments (schema:12, minItems 1),
    # distinct from the proposed ``candidate_segments``.
    trace_segments: tuple[TraceSegment, ...] = ()


def validate_trace_manifest(manifest: TraceManifest) -> list[str]:
    errors: list[str] = []
    if manifest.project != "tracecausal":
        errors.append("project must be tracecausal")
    if manifest.server_authorized:
        errors.append("server_authorized must remain false in local manifests")
    if manifest.split not in {"train", "valid", "test", "diagnostic"}:
        errors.append("split must be train, valid, test, or diagnostic")
    # split_hash: required, minLength 8 (trace_manifest.schema.json:22).
    if not manifest.split_hash or len(manifest.split_hash.strip()) < 8:
        errors.append("split_hash must be a non-empty hash of length >= 8")
    if not manifest.trace_steps:
        errors.append("trace_steps must not be empty")
    if not manifest.trace_segments:
        errors.append("trace_segments must not be empty")
    if not manifest.candidate_segments:
        errors.append("candidate_segments must not be empty")

    step_ids = {step.step_id for step in manifest.trace_steps}
    if len(step_ids) != len(manifest.trace_steps):
        errors.append("trace step ids must be unique")
    for step in manifest.trace_steps:
        start, end = step.text_span
        if start < 0 or end <= start:
            errors.append(f"invalid text_span for step {step.step_id}")
        if not 0.0 <= step.score <= 1.0:
            errors.append(f"step score must be in [0, 1] for {step.step_id}")
    # Both realised trace_segments and proposed candidate_segments must reference
    # only declared steps and carry in-range scores.
    for label, segments in (
        ("trace", manifest.trace_segments),
        ("candidate", manifest.candidate_segments),
    ):
        for segment in segments:
            if not segment.step_ids:
                errors.append(f"{label} segment {segment.segment_id} has no steps")
            missing = [s for s in segment.step_ids if s not in step_ids]
            if missing:
                errors.append(
                    f"{label} segment {segment.segment_id} references missing steps: {missing}"
                )
            if not 0.0 <= segment.score <= 1.0:
                errors.append(
                    f"{label} segment score must be in [0, 1] for {segment.segment_id}"
                )
    return errors

