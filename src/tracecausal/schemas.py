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


def validate_trace_manifest(manifest: TraceManifest) -> list[str]:
    errors: list[str] = []
    if manifest.project != "tracecausal":
        errors.append("project must be tracecausal")
    if manifest.server_authorized:
        errors.append("server_authorized must remain false in local manifests")
    if manifest.split not in {"train", "valid", "test", "diagnostic"}:
        errors.append("split must be train, valid, test, or diagnostic")
    if not manifest.trace_steps:
        errors.append("trace_steps must not be empty")
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
    for segment in manifest.candidate_segments:
        if not segment.step_ids:
            errors.append(f"segment {segment.segment_id} has no steps")
        missing = [step_id for step_id in segment.step_ids if step_id not in step_ids]
        if missing:
            errors.append(f"segment {segment.segment_id} references missing steps: {missing}")
        if not 0.0 <= segment.score <= 1.0:
            errors.append(f"segment score must be in [0, 1] for {segment.segment_id}")
    return errors

