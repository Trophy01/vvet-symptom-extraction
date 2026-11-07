"""Tests for rule-based escalation logic and reason generation."""

import pytest
from app.escalation import evaluate_escalation
from app.schemas import Finding, FindingValue, SignId


@pytest.mark.parametrize("sign", [
    SignId.FROTH_NOSE_MOUTH,
    SignId.UNABLE_TO_STAND,
    SignId.CLOUDY_EYE,
])
@pytest.mark.parametrize("val", [FindingValue.YES, FindingValue.NOT_SURE])
def test_escalation_single_urgent_sign(sign, val):
    """Emergency signs with 'yes' or 'not_sure' MUST escalate."""
    finding = Finding(sign_id=sign, value=val, quote="sample quote")
    escalate, reason = evaluate_escalation([finding])

    assert escalate is True
    assert sign.value in reason
    assert val.value in reason


@pytest.mark.parametrize("sign", [
    SignId.FROTH_NOSE_MOUTH,
    SignId.UNABLE_TO_STAND,
    SignId.CLOUDY_EYE,
])
def test_escalation_urgent_sign_with_no_does_not_escalate(sign):
    """Emergency signs with explicit 'no' MUST NOT escalate."""
    finding = Finding(sign_id=sign, value=FindingValue.NO, quote="sample quote")
    escalate, reason = evaluate_escalation([finding])

    assert escalate is False
    assert "No urgent signs detected" in reason


def test_escalation_multiple_trigger_signs():
    """When multiple emergency signs trigger, reason must mention all of them."""
    findings = [
        Finding(sign_id=SignId.FROTH_NOSE_MOUTH, value=FindingValue.YES, quote="froth from mouth"),
        Finding(sign_id=SignId.UNABLE_TO_STAND, value=FindingValue.NOT_SURE, quote="cannot stand"),
        Finding(sign_id=SignId.REDUCED_APPETITE, value=FindingValue.YES, quote="not eating"),
    ]
    escalate, reason = evaluate_escalation(findings)

    assert escalate is True
    assert "froth_nose_mouth (yes)" in reason
    assert "unable_to_stand (not_sure)" in reason


@pytest.mark.parametrize("sign", [
    SignId.REDUCED_APPETITE,
    SignId.WEAKNESS,
    SignId.SWELLING,
    SignId.RAPID_BREATHING,
    SignId.RED_URINE,
    SignId.TICKS,
])
def test_non_urgent_signs_do_not_escalate(sign):
    """Non-emergency signs (even if 'yes') must not trigger escalation."""
    finding = Finding(sign_id=sign, value=FindingValue.YES, quote="sample quote")
    escalate, reason = evaluate_escalation([finding])

    assert escalate is False
    assert "No urgent signs detected" in reason
    assert sign.value in reason


def test_empty_findings_does_not_escalate():
    """Empty findings list must not escalate and return a clear message."""
    escalate, reason = evaluate_escalation([])
    assert escalate is False
    assert "No clinical signs detected in message." in reason
