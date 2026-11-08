"""Tests for symptom extractor, quote validation, and retry logic."""

import pytest
from app.extractor import ExtractionError, SymptomExtractor, validate_and_fix_quote
from app.schemas import FindingValue, SignId
from tests.conftest import MockLLMProvider


def test_quote_validation_exact_match():
    """Exact quote in message is accepted as-is."""
    msg = "My cow has rapid breathing and ticks on ears."
    quote = "ticks on ears"
    assert validate_and_fix_quote(quote, msg) == "ticks on ears"


def test_quote_validation_case_insensitive_fix():
    """Case difference is corrected to match the exact casing in the message."""
    msg = "The animal has Mbandabvu under the ear."
    quote = "mbandabvu under the ear"
    fixed = validate_and_fix_quote(quote, msg)
    assert fixed == "Mbandabvu under the ear"


def test_quote_validation_stripped_outer_quotes():
    """Model enclosing quote in quotation marks is handled cleanly."""
    msg = "She has not been eating since yesterday."
    quote = '"not been eating"'
    assert validate_and_fix_quote(quote, msg) == "not been eating"


def test_quote_validation_not_in_message_dropped():
    """Hallucinated quote not present in the message returns None."""
    msg = "My cow is weak and lying down."
    quote = "frothing at mouth"
    assert validate_and_fix_quote(quote, msg) is None


def test_quote_validation_empty_or_whitespace():
    """Empty or whitespace quote returns None."""
    msg = "My cow is sick."
    assert validate_and_fix_quote("", msg) is None
    assert validate_and_fix_quote("   ", msg) is None
    assert validate_and_fix_quote(None, msg) is None


def test_extractor_drops_invalid_sign_id():
    """Any sign_id not in the 9 allowed signs must be discarded."""
    message = "Cow has high fever and cough."
    raw = [
        {"sign_id": "fever", "value": "yes", "quote": "high fever"},
        {"sign_id": "cough", "value": "yes", "quote": "cough"},
    ]
    provider = MockLLMProvider(responses=raw)
    extractor = SymptomExtractor(provider)
    findings = extractor.validate_findings(raw, message)
    assert len(findings) == 0


def test_extractor_drops_invalid_value():
    """Any finding with value other than yes/no/not_sure must be discarded."""
    message = "Cow has ticks everywhere."
    raw = [
        {"sign_id": "ticks", "value": "definitely", "quote": "ticks"},
    ]
    provider = MockLLMProvider(responses=raw)
    extractor = SymptomExtractor(provider)
    findings = extractor.validate_findings(raw, message)
    assert len(findings) == 0


def test_extractor_deduplicates_signs():
    """Duplicate mentions of the same sign_id are deduplicated (first retained)."""
    message = "Cow has ticks on ears and ticks on tail."
    raw = [
        {"sign_id": "ticks", "value": "yes", "quote": "ticks on ears"},
        {"sign_id": "ticks", "value": "yes", "quote": "ticks on tail"},
    ]
    provider = MockLLMProvider(responses=raw)
    extractor = SymptomExtractor(provider)
    findings = extractor.validate_findings(raw, message)

    assert len(findings) == 1
    assert findings[0].sign_id == SignId.TICKS
    assert findings[0].quote == "ticks on ears"


def test_extractor_retry_success():
    """If provider fails on first attempt, extractor retries and succeeds."""
    message = "My cow cannot stand up."
    expected_findings = [
        {"sign_id": "unable_to_stand", "value": "yes", "quote": "cannot stand up"}
    ]
    # First call raises an exception, second call succeeds
    provider = MockLLMProvider(
        sequential_responses=[
            ConnectionError("Temporary network hiccup"),
            expected_findings,
        ]
    )
    extractor = SymptomExtractor(provider)
    response = extractor.process_message(message)

    assert provider.call_count == 2
    assert response.escalate is True
    assert len(response.findings) == 1
    assert response.findings[0].sign_id == SignId.UNABLE_TO_STAND


def test_extractor_retry_failure_raises_extraction_error():
    """If provider fails twice, ExtractionError is raised."""
    message = "My cow is weak."
    provider = MockLLMProvider(
        sequential_responses=[
            RuntimeError("Provider failure 1"),
            RuntimeError("Provider failure 2"),
        ]
    )
    extractor = SymptomExtractor(provider)

    with pytest.raises(ExtractionError) as exc_info:
        extractor.process_message(message)

    assert provider.call_count == 2
    assert "Extraction failed after 2 attempts" in str(exc_info.value)


def test_extractor_invalid_output_retries_and_fails():
    """Non-list output triggers retry and raises ExtractionError on consecutive failures."""
    message = "My cow is sick."
    provider = MockLLMProvider(
        sequential_responses=[
            "invalid string output",
            "second invalid output",
        ]
    )
    extractor = SymptomExtractor(provider)

    with pytest.raises(ExtractionError) as exc_info:
        extractor.process_message(message)

    assert provider.call_count == 2
    assert "LLM output must be a list" in str(exc_info.value)


def test_extractor_permanent_error_no_retry():
    """PermanentLLMError fails immediately without retrying."""
    from app.llm import PermanentLLMError

    message = "My cow is weak."
    provider = MockLLMProvider(
        sequential_responses=[
            PermanentLLMError("API key invalid"),
        ]
    )
    extractor = SymptomExtractor(provider)

    with pytest.raises(ExtractionError) as exc_info:
        extractor.process_message(message)

    assert provider.call_count == 1
    assert "Permanent extraction error" in str(exc_info.value)

