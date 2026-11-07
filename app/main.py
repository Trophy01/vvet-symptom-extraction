"""FastAPI application for V-Vet livestock symptom extraction."""

from fastapi import Depends, FastAPI, HTTPException, status
from app.extractor import ExtractionError, SymptomExtractor
from app.llm import GeminiLLMProvider, LLMProvider
from app.schemas import ExtractRequest, ExtractResponse

app = FastAPI(
    title="V-Vet Livestock Symptom Extraction Service",
    description=(
        "Extracts clinical findings from Zimbabwean farmer WhatsApp messages "
        "(English, Shona, or code-switched mixes) and evaluates emergency escalation."
    ),
    version="1.0.0",
)


def get_llm_provider() -> LLMProvider:
    """Dependency provider for the LLM client. Can be overridden in tests."""
    return GeminiLLMProvider()


def get_extractor(
    provider: LLMProvider = Depends(get_llm_provider),
) -> SymptomExtractor:
    """Dependency provider for the symptom extraction pipeline."""
    return SymptomExtractor(llm_provider=provider)


@app.get(
    "/health",
    tags=["Health"],
    summary="Service health check",
)
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "vvet-symptom-extraction"}


@app.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract symptoms from farmer message and evaluate escalation",
)
def extract_symptoms(
    request: ExtractRequest,
    extractor: SymptomExtractor = Depends(get_extractor),
) -> ExtractResponse:
    """Extract clinical signs from a WhatsApp message, validate findings, and evaluate escalation.

    Returns:
        ExtractResponse with validated findings, escalation boolean, and rationale.

    Raises:
        HTTPException 502: If the upstream LLM call fails or returns unrecoverable invalid output.
    """
    try:
        return extractor.process_message(request.message)
    except ExtractionError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream extraction failed: {err}",
        )
