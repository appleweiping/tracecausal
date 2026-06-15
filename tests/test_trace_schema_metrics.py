from tracecausal.metrics import intervention_effect, passes_intervention_gate
from tracecausal.schemas import TraceManifest, TraceSegment, TraceStep, validate_trace_manifest


def test_trace_manifest_rejects_local_server_authorization():
    manifest = TraceManifest(
        project="tracecausal",
        dataset_id="diagnostic",
        model_id="model",
        split="diagnostic",
        trace_steps=(
            TraceStep(
                step_id="s1",
                paradigm="autoregressive",
                text_span=(0, 4),
                score=0.7,
            ),
        ),
        candidate_segments=(
            TraceSegment(
                segment_id="seg1",
                step_ids=("s1",),
                selector="causal",
                score=0.8,
                intervention="patch",
            ),
        ),
        seed=0,
        server_authorized=True,
    )
    assert "server_authorized must remain false in local manifests" in validate_trace_manifest(manifest)


def test_intervention_gate_requires_margin_and_utility():
    assert intervention_effect(0.4, 0.52) == 0.12
    assert passes_intervention_gate(0.12, 0.03, 0.01)
    assert not passes_intervention_gate(0.06, 0.03, 0.01)
    assert not passes_intervention_gate(0.12, 0.03, 0.03)

