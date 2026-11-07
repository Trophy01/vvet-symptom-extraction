"""Integration tests for the FastAPI POST /extract endpoint and sample messages."""

import pytest
from app.main import app, get_llm_provider
from tests.conftest import MockLLMProvider


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Ensure dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


def test_health_check(client):
    """Verify service health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "vvet-symptom-extraction"}


def test_sample_1_reduced_appetite_and_rapid_breathing(client):
    """Sample 1: 'My cow has not been eating since yesterday and she is breathing very fast.'

    Expected:
    - reduced_appetite: yes
    - rapid_breathing: yes
    - escalate: False
    """
    message = "My cow has not been eating since yesterday and she is breathing very fast."
    canned_llm_output = [
        {
            "sign_id": "reduced_appetite",
            "value": "yes",
            "quote": "not been eating since yesterday",
        },
        {
            "sign_id": "rapid_breathing",
            "value": "yes",
            "quote": "breathing very fast",
        },
    ]

    mock_provider = MockLLMProvider(responses=canned_llm_output)
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    response = client.post("/extract", json={"message": message})
    assert response.status_code == 200
    data = response.json()

    assert data["escalate"] is False
    assert len(data["findings"]) == 2

    signs = {f["sign_id"]: f for f in data["findings"]}
    assert signs["reduced_appetite"]["value"] == "yes"
    assert signs["reduced_appetite"]["quote"] == "not been eating since yesterday"
    assert signs["rapid_breathing"]["value"] == "yes"
    assert signs["rapid_breathing"]["quote"] == "breathing very fast"
    assert "No urgent signs detected" in data["reason"]


def test_sample_2_froth_and_unable_to_stand_escalation(client):
    """Sample 2: 'There's foam coming from her nose and she can't get up.'

    Expected:
    - froth_nose_mouth: yes
    - unable_to_stand: yes
    - escalate: True (both are emergency signs)
    """
    message = "There's foam coming from her nose and she can't get up."
    canned_llm_output = [
        {
            "sign_id": "froth_nose_mouth",
            "value": "yes",
            "quote": "foam coming from her nose",
        },
        {
            "sign_id": "unable_to_stand",
            "value": "yes",
            "quote": "can't get up",
        },
    ]

    mock_provider = MockLLMProvider(responses=canned_llm_output)
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    response = client.post("/extract", json={"message": message})
    assert response.status_code == 200
    data = response.json()

    assert data["escalate"] is True
    assert len(data["findings"]) == 2
    assert "froth_nose_mouth (yes)" in data["reason"]
    assert "unable_to_stand (yes)" in data["reason"]


def test_sample_3_cloudy_eye_unsure_and_mbandabvu_swelling(client):
    """Sample 3: 'I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear.'

    Expected:
    - cloudy_eye: not_sure
    - swelling: yes
    - escalate: True (cloudy_eye with 'not_sure' triggers escalation)
    """
    message = "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear."
    canned_llm_output = [
        {
            "sign_id": "cloudy_eye",
            "value": "not_sure",
            "quote": "eyes look a bit cloudy but I'm not sure",
        },
        {
            "sign_id": "swelling",
            "value": "yes",
            "quote": "mbandabvu under the ear",
        },
    ]

    mock_provider = MockLLMProvider(responses=canned_llm_output)
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    response = client.post("/extract", json={"message": message})
    assert response.status_code == 200
    data = response.json()

    assert data["escalate"] is True
    assert len(data["findings"]) == 2
    assert "cloudy_eye (not_sure)" in data["reason"]


def test_sample_4_ticks_and_fine_calf(client):
    """Sample 4: 'The calf is fine, just lots of ticks on the ears.'

    Expected:
    - ticks: yes
    - No findings for unmentioned signs despite 'calf is fine'
    - escalate: False
    """
    message = "The calf is fine, just lots of ticks on the ears."
    canned_llm_output = [
        {
            "sign_id": "ticks",
            "value": "yes",
            "quote": "lots of ticks on the ears",
        }
    ]

    mock_provider = MockLLMProvider(responses=canned_llm_output)
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    response = client.post("/extract", json={"message": message})
    assert response.status_code == 200
    data = response.json()

    assert data["escalate"] is False
    assert len(data["findings"]) == 1
    assert data["findings"][0]["sign_id"] == "ticks"
    assert data["findings"][0]["quote"] == "lots of ticks on the ears"
    assert "No urgent signs detected" in data["reason"]


def test_sample_5_shona_code_switching(client):
    """Sample 5: 'Mombe yangu haisi kudya, and her urine is dark like tea.'

    Expected:
    - reduced_appetite: yes (from 'Mombe yangu haisi kudya')
    - red_urine: yes (from 'urine is dark like tea')
    - escalate: False
    """
    message = "Mombe yangu haisi kudya, and her urine is dark like tea."
    canned_llm_output = [
        {
            "sign_id": "reduced_appetite",
            "value": "yes",
            "quote": "Mombe yangu haisi kudya",
        },
        {
            "sign_id": "red_urine",
            "value": "yes",
            "quote": "urine is dark like tea",
        },
    ]

    mock_provider = MockLLMProvider(responses=canned_llm_output)
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    response = client.post("/extract", json={"message": message})
    assert response.status_code == 200
    data = response.json()

    assert data["escalate"] is False
    assert len(data["findings"]) == 2
    signs = {f["sign_id"]: f for f in data["findings"]}
    assert signs["reduced_appetite"]["quote"] == "Mombe yangu haisi kudya"
    assert signs["red_urine"]["quote"] == "urine is dark like tea"


def test_api_upstream_error_returns_502(client):
    """When the LLM fails repeatedly, API returns HTTP 502 Bad Gateway."""
    message = "Cow is sick."
    mock_provider = MockLLMProvider(
        sequential_responses=[
            RuntimeError("Connection timeout to Gemini API"),
            RuntimeError("Connection timeout on retry"),
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    response = client.post("/extract", json={"message": message})
    assert response.status_code == 502
    data = response.json()
    assert "Upstream extraction failed" in data["detail"]


def test_api_invalid_request_payload(client):
    """Missing or empty message field returns HTTP 422."""
    response = client.post("/extract", json={})
    assert response.status_code == 422

    response = client.post("/extract", json={"message": ""})
    assert response.status_code == 422
