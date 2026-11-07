"""Pydantic schemas and domain models for symptom extraction."""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class SignId(str, Enum):
    """The 9 allowed livestock clinical signs."""
    REDUCED_APPETITE = "reduced_appetite"
    WEAKNESS = "weakness"
    SWELLING = "swelling"
    RAPID_BREATHING = "rapid_breathing"
    CLOUDY_EYE = "cloudy_eye"
    FROTH_NOSE_MOUTH = "froth_nose_mouth"
    RED_URINE = "red_urine"
    UNABLE_TO_STAND = "unable_to_stand"
    TICKS = "ticks"


SIGN_DESCRIPTIONS = {
    SignId.REDUCED_APPETITE: "eating less, or has stopped eating or grazing",
    SignId.WEAKNESS: "dull, weak, lying down more than usual",
    SignId.SWELLING: "a lump below the ear, under the jaw, or in front of the shoulder (farmers may call it 'mbandabvu')",
    SignId.RAPID_BREATHING: "breathing fast while resting",
    SignId.CLOUDY_EYE: "the eye looks white, grey or milky",
    SignId.FROTH_NOSE_MOUTH: "foam or froth from the nose or mouth",
    SignId.RED_URINE: "urine that is red, brown, or the colour of strong tea",
    SignId.UNABLE_TO_STAND: "cannot get up or stay standing",
    SignId.TICKS: "ticks can be seen on the animal",
}


class FindingValue(str, Enum):
    """Allowed values for detected clinical signs."""
    YES = "yes"
    NO = "no"
    NOT_SURE = "not_sure"


class Finding(BaseModel):
    """A validated symptom finding extracted from the farmer message."""
    sign_id: SignId = Field(
        ...,
        description="The recognized sign identifier from the 9 valid signs.",
    )
    value: FindingValue = Field(
        ...,
        description="Whether the sign is present ('yes'), absent ('no'), or uncertain ('not_sure').",
    )
    quote: str = Field(
        ...,
        description="Exact quote or phrase from the farmer message indicating this sign.",
    )


class ExtractRequest(BaseModel):
    """Request payload for symptom extraction."""
    message: str = Field(
        ...,
        min_length=1,
        description="WhatsApp message from the farmer describing their animal.",
        examples=["My cow has not been eating since yesterday and she is breathing very fast."],
    )


class ExtractResponse(BaseModel):
    """Structured response containing findings, escalation decision, and rationale."""
    findings: List[Finding] = Field(
        default_factory=list,
        description="List of validated findings extracted from the message.",
    )
    escalate: bool = Field(
        ...,
        description="True if any urgent signs require emergency veterinary escalation.",
    )
    reason: str = Field(
        ...,
        description="Clinical rationale explaining the escalation decision.",
    )
