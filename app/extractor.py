"""Symptom extraction pipeline with validation, quote verification, and retry logic."""

import logging
from typing import Any, Dict, List, Optional, Set
from app.escalation import evaluate_escalation
from app.llm import LLMProvider, PermanentLLMError
from app.schemas import ExtractResponse, Finding, FindingValue, SignId

logger = logging.getLogger(__name__)

# Valid sign IDs set for quick membership checking
VALID_SIGN_IDS: Set[str] = {s.value for s in SignId}
VALID_VALUES: Set[str] = {v.value for v in FindingValue}


class ExtractionError(Exception):
    """Raised when extraction fails upstream (e.g. LLM failure or invalid response)."""
    pass


def validate_and_fix_quote(quote: Optional[str], message: str) -> Optional[str]:
    """Verify that the quote actually appears in the message, fixing minor case/formatting differences.

    Rules:
    - If quote appears verbatim in the message, return it.
    - If quote matches case-insensitively, return the exact substring from the message.
    - If quote was enclosed in extra quotes, strip them and recheck.
    - If quote does not appear in the message at all, return None (drop it).
    """
    if not quote or not isinstance(quote, str):
        return None

    cleaned = quote.strip()
    if not cleaned:
        return None

    # 1. Exact match
    if cleaned in message:
        return cleaned

    # 2. Match without outer quotation marks if model added them
    unquoted = cleaned.strip("\"'").strip()
    if unquoted and unquoted in message:
        return unquoted

    # 3. Case-insensitive match -> fix to exact substring from the original message
    lower_message = message.lower()
    target = unquoted.lower() if unquoted else cleaned.lower()
    idx = lower_message.find(target)
    if idx != -1:
        return message[idx : idx + len(target)]

    # Quote does not appear in the original text
    return None


class SymptomExtractor:
    """Orchestrates symptom extraction, validation, quote checking, and escalation."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def extract_with_retry(self, message: str) -> List[Dict[str, Any]]:
        """Call the LLM provider with retry on failure or invalid output.

        Retries once upon encountering an exception or invalid format.
        """
        attempts = 0
        max_attempts = 2
        last_error: Optional[Exception] = None

        while attempts < max_attempts:
            attempts += 1
            try:
                raw_findings = self.llm_provider.extract_raw_findings(message)
                if not isinstance(raw_findings, list):
                    raise ValueError(
                        f"LLM output must be a list of findings, got {type(raw_findings).__name__}"
                    )
                return raw_findings
            except PermanentLLMError as exc:
                raise ExtractionError(f"Permanent extraction error: {exc}") from exc
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Extraction attempt %d failed: %s. %s",
                    attempts,
                    exc,
                    "Retrying..." if attempts < max_attempts else "No more retries.",
                )

        raise ExtractionError(
            f"Extraction failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def validate_findings(
        self, raw_findings: List[Dict[str, Any]], message: str
    ) -> List[Finding]:
        """Validate raw findings against domain constraints.

        Constraints:
        - Drop any sign_id not in the 9 allowed signs.
        - Value must be one of: 'yes', 'no', 'not_sure'.
        - Remove duplicate signs (keep first occurrence).
        - Drop findings whose quote does not appear in the farmer's message.
        """
        validated: List[Finding] = []
        seen_signs: Set[SignId] = set()

        for item in raw_findings:
            if not isinstance(item, dict):
                continue

            raw_sign = item.get("sign_id")
            raw_value = item.get("value")
            raw_quote = item.get("quote")

            # Validate sign_id
            if not raw_sign or raw_sign not in VALID_SIGN_IDS:
                logger.debug("Dropping invalid sign_id: %s", raw_sign)
                continue

            sign_id = SignId(raw_sign)

            # Deduplicate signs (keep first occurrence)
            if sign_id in seen_signs:
                logger.debug("Dropping duplicate sign_id: %s", sign_id)
                continue

            # Validate value
            if not raw_value or raw_value not in VALID_VALUES:
                logger.debug("Dropping invalid value '%s' for sign '%s'", raw_value, sign_id)
                continue

            value = FindingValue(raw_value)

            # Validate and fix quote
            fixed_quote = validate_and_fix_quote(raw_quote, message)
            if not fixed_quote:
                logger.debug(
                    "Dropping sign '%s' because quote '%s' does not appear in message",
                    sign_id,
                    raw_quote,
                )
                continue

            validated.append(
                Finding(
                    sign_id=sign_id,
                    value=value,
                    quote=fixed_quote,
                )
            )
            seen_signs.add(sign_id)

        return validated

    def process_message(self, message: str) -> ExtractResponse:
        """Full pipeline: extract via LLM (with retry), validate findings, decide escalation."""
        raw_findings = self.extract_with_retry(message)
        findings = self.validate_findings(raw_findings, message)
        escalate, reason = evaluate_escalation(findings)

        return ExtractResponse(
            findings=findings,
            escalate=escalate,
            reason=reason,
        )
