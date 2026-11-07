"""Tests for domain schemas and models."""

import pytest
from pydantic import ValidationError
from app.schemas import (
    SIGN_DESCRIPTIONS,
    ExtractRequest,
    ExtractResponse,
    Finding,
    FindingValue,
    SignId,
)


def test_allowed_signs_count():
    """Verify exactly the 9 allowed signs exist."""
    assert len(SignId) == 9
    assert len(SIGN_DESCRIPTIONS) == 9
    expected_signs = {
        "reduced_appetite",
        "weakness",
        "swelling",
        "rapid_breathing",
        "cloudy_eye",
        "froth_nose_mouth",
        "red_urine",
        "unable_to_stand",
        "ticks",
    }
    actual_signs = {s.value for s in SignId}
    assert actual_signs == expected_signs


def test_finding_valid_creation():
    """Verify valid Finding instantiation."""
    finding = Finding(
        sign_id=SignId.RAPID_BREATHING,
        value=FindingValue.YES,
        quote="breathing very fast",
    )
    assert finding.sign_id == SignId.RAPID_BREATHING
    assert finding.value == FindingValue.YES
    assert finding.quote == "breathing very fast"


def test_finding_invalid_sign_id_raises():
    """Verify invalid sign_id cannot be parsed into Finding."""
    with pytest.raises(ValidationError):
        Finding(
            sign_id="coughing",  # Not in allowed list
            value=FindingValue.YES,
            quote="coughing",
        )


def test_finding_invalid_value_raises():
    """Verify invalid value cannot be parsed into Finding."""
    with pytest.raises(ValidationError):
        Finding(
            sign_id=SignId.TICKS,
            value="maybe",  # Not in allowed list
            quote="ticks",
        )


def test_extract_request_validation():
    """Verify request validation rules."""
    req = ExtractRequest(message="My cow is sick.")
    assert req.message == "My cow is sick."

    # Empty string should fail
    with pytest.raises(ValidationError):
        ExtractRequest(message="")


def test_extract_response_serialization():
    """Verify ExtractResponse serialization format."""
    resp = ExtractResponse(
        findings=[
            Finding(
                sign_id=SignId.UNABLE_TO_STAND,
                value=FindingValue.YES,
                quote="can't get up",
            )
        ],
        escalate=True,
        reason="Urgent veterinary attention required.",
    )
    data = resp.model_dump()
    assert data["escalate"] is True
    assert data["reason"] == "Urgent veterinary attention required."
    assert len(data["findings"]) == 1
    assert data["findings"][0]["sign_id"] == "unable_to_stand"
