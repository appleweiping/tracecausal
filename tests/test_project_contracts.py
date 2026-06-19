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


def _ok_manifest():
    return {
        "project": "tracecausal",
        "server": {"authorized": False},
        "seeds": {"paper_minimum": 20},
        "baselines": ["random_segment", "output_entropy", "semantic_entropy"],
    }


def test_manifest_config_text_enforces_redesign_v4_authorized_false():
    # The hardened v4 plan must carry 'authorized: false'.
    errors = validate_manifest(
        _ok_manifest(),
        config_texts={"redesign_v4_ar_lead.yaml": "server:\n  authorized: true\n"},
    )
    assert any("redesign_v4_ar_lead.yaml" in e for e in errors)

    clean = validate_manifest(
        _ok_manifest(),
        config_texts={
            "redesign_v4_ar_lead.yaml": "server:\n  authorized: false\n",
            "formal_tracecausal.yaml": "server:\n  authorized: false\n",
        },
    )
    assert clean == []


def test_manifest_config_text_default_is_back_compatible():
    # Without config_texts the public single-arg behaviour is unchanged.
    assert validate_manifest(_ok_manifest()) == []

