from tracecausal.contracts import EvidenceTier, validate_manifest


def test_evidence_tiers_are_explicit():
    assert EvidenceTier.PAPER_RESULT.value == "paper_result"
    assert EvidenceTier.PILOT.value == "pilot"


def test_manifest_rejects_toy_seed_count():
    manifest = {
        "project": "tracecausal",
        "server": {"authorized": False},
        "seeds": {"paper_minimum": 3},
        "baselines": ["random_segment", "output_entropy", "semantic_entropy"],
    }
    errors = validate_manifest(manifest)
    assert "paper_minimum seeds must be at least 20" in errors


def test_manifest_requires_multiple_baselines():
    manifest = {
        "project": "tracecausal",
        "server": {"authorized": False},
        "seeds": {"paper_minimum": 20},
        "baselines": ["random_segment"],
    }
    errors = validate_manifest(manifest)
    assert "at least three baseline families are required" in errors

