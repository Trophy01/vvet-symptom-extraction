"""Deterministic escalation logic for livestock symptoms.

Decides whether a case requires emergency veterinary triage based on a fixed set
of critical signs, independent of the LLM.
"""

from typing import List, Tuple
from app.schemas import Finding, FindingValue, SignId

# Signs that trigger emergency escalation if present ('yes') or suspected ('not_sure')
ESCALATION_SIGNS = {
    SignId.FROTH_NOSE_MOUTH,
    SignId.UNABLE_TO_STAND,
    SignId.CLOUDY_EYE,
}

TRIGGER_VALUES = {FindingValue.YES, FindingValue.NOT_SURE}


def evaluate_escalation(findings: List[Finding]) -> Tuple[bool, str]:
    """Evaluate whether the findings require emergency escalation and generate a clinical reason.

    Escalation rule:
        escalate is True if any of froth_nose_mouth, unable_to_stand, or cloudy_eye
        has value 'yes' or 'not_sure'.

    Args:
        findings: Validated list of clinical findings.

    Returns:
        (escalate, reason): A tuple containing the boolean escalation flag and
        a human-readable reason string naming the triggering signs.
    """
    triggering_findings = [
        f for f in findings
        if f.sign_id in ESCALATION_SIGNS and f.value in TRIGGER_VALUES
    ]

    if triggering_findings:
        details = ", ".join(
            f"{f.sign_id.value} ({f.value.value})" for f in triggering_findings
        )
        reason = f"Urgent veterinary attention required due to critical signs: {details}."
        return True, reason

    # When not escalating
    if not findings:
        return False, "No clinical signs detected in message."

    non_escalating_signs = [f.sign_id.value for f in findings if f.value in TRIGGER_VALUES]
    if non_escalating_signs:
        signs_str = ", ".join(non_escalating_signs)
        return False, f"No urgent signs detected. Reported non-urgent signs: {signs_str}."

    return False, "No urgent signs detected."
