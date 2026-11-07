"""Pytest fixtures and mock providers for testing."""

from typing import Any, Callable, Dict, List, Optional, Union
import pytest
from fastapi.testclient import TestClient
from app.llm import LLMProvider
from app.main import app, get_llm_provider


class MockLLMProvider(LLMProvider):
    """Configurable mock LLM provider for deterministic offline testing."""

    def __init__(
        self,
        responses: Optional[Union[List[Dict[str, Any]], Callable[[str], List[Dict[str, Any]]]]] = None,
        sequential_responses: Optional[List[Union[List[Dict[str, Any]], Exception]]] = None,
    ):
        self.responses = responses
        self.sequential_responses = sequential_responses or []
        self.call_count = 0
        self.recorded_messages: List[str] = []

    def extract_raw_findings(self, message: str) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.recorded_messages.append(message)

        if self.sequential_responses:
            resp = self.sequential_responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

        if callable(self.responses):
            return self.responses(message)

        if self.responses is not None:
            return self.responses

        return []


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    with TestClient(app) as test_client:
        yield test_client
