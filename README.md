# V-Vet Livestock Symptom Extraction Service

A robust, explainable FastAPI service built for **V-Vet**, a livestock health startup supporting Zimbabwean farmers. The service extracts clinical signs from WhatsApp messages described in farmers' own words (English, Shona, or code-switched mixes), validates the findings against a fixed clinical vocabulary, and applies deterministic emergency triage rules.

---

## Table of Contents
- [The 9 Valid Signs & Escalation Rules](#the-9-valid-signs--escalation-rules)
- [Project Architecture](#project-architecture)
- [How to Run](#how-to-run)
- [How to Run Tests](#how-to-run-tests)
- [Sample Messages & Expected Outputs](#sample-messages--expected-outputs)
- [Design Decisions & Why](#design-decisions--why)
- [Measuring Extraction Accuracy](#measuring-extraction-accuracy)
- [What I Would Do With More Time](#what-i-would-do-with-more-time)
- [How I Used AI Tools](#how-i-used-ai-tools)

---

## The 9 Valid Signs & Escalation Rules

### Allowed Clinical Signs
The system strictly recognizes only the following 9 signs:
| `sign_id` | Clinical Meaning / Farmer Language |
| :--- | :--- |
| `reduced_appetite` | Eating less, or has stopped eating or grazing (*"Mombe haisi kudya"*) |
| `weakness` | Dull, weak, lying down more than usual (*"kupera simba"*) |
| `swelling` | Lump below ear, under jaw, or shoulder (commonly called *"mbandabvu"*) |
| `rapid_breathing` | Breathing fast while resting |
| `cloudy_eye` | The eye looks white, grey, or milky |
| `froth_nose_mouth` | Foam or froth from the nose or mouth (*"furo"*) |
| `red_urine` | Urine that is red, brown, or colour of strong tea (*"weti tsvuku"*) |
| `unable_to_stand` | Cannot get up or stay standing (*"kutadza kusimuka"*) |
| `ticks` | Ticks can be seen on the animal |

### Escalation Rules (Deterministic Python Code)
- **`escalate` is `True`** if any of `froth_nose_mouth`, `unable_to_stand`, or `cloudy_eye` has a value of `"yes"` or `"not_sure"`.
- The `reason` field explicitly lists triggering signs when escalating, or explains why no escalation was triggered.
- **Rule of Thumb**: Non-urgent signs (e.g. `ticks`, `reduced_appetite`, `swelling`) or emergency signs explicitly stated as absent (`"no"`) do not trigger escalation.

---

## Project Architecture

```
Vnet_api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application & POST /extract endpoint
│   ├── schemas.py       # Pydantic schemas (SignId, Finding, Request/Response)
│   ├── escalation.py    # Pure Python deterministic escalation logic
│   ├── llm.py           # LLMProvider interface & Gemini free-tier client
│   └── extractor.py     # Pipeline: retry, post-validation & quote checking
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # MockLLMProvider & TestClient fixtures
│   ├── test_schemas.py  # Model validation & enum constraints
│   ├── test_escalation.py # Escalation matrix unit tests
│   ├── test_extractor.py  # Quote grounding, deduplication & retry tests
│   └── test_api.py      # Integration tests covering the 5 sample messages
├── .env.example         # Template for environment variables
├── .env                 # Local secrets file (gitignored)
├── .gitignore           # Git ignore rules (.env, .venv, caches)
├── pytest.ini           # Pytest test runner configuration
├── requirements.txt     # Python package dependencies
└── README.md            # Technical documentation
```

---

## How to Run

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12 / 3.14)
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/trophy-hub/vvet-symptom-extraction.git
cd vvet-symptom-extraction

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Copy the template and paste your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 4. Start the API Server
```bash
uvicorn app.main:app --reload --port 8000
```
The API is now live at `http://127.0.0.1:8000`.
- Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: `curl http://127.0.0.1:8000/health`

### 5. Example Request
```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "There is foam coming from her nose and she cannot get up."}'
```

---

## How to Run Tests

The test suite runs **100% offline with no network calls and no API key required** by utilizing a clean `MockLLMProvider` fixture.

```bash
# Run all tests
pytest

# Run tests with detailed output
pytest -v
```

All 42 tests cover:
- Pydantic schema validation and constraint enforcement.
- Escalation rules (all combinations of urgent vs. non-urgent signs, `yes`, `no`, `not_sure`).
- Quote verification (exact match, case normalization, and rejection of hallucinated quotes).
- Upstream retry behavior (recovering on single retry, raising HTTP 502 on consecutive failures).
- End-to-end integration tests for all 5 sample messages.

---

## Sample Messages & Expected Outputs

Here are the 5 sample messages and how the system handles them:

### Sample 1: English (Reduced Appetite & Rapid Breathing)
- **Input**: `"My cow has not been eating since yesterday and she is breathing very fast."`
- **Response**:
```json
{
  "findings": [
    {
      "sign_id": "reduced_appetite",
      "value": "yes",
      "quote": "not been eating since yesterday"
    },
    {
      "sign_id": "rapid_breathing",
      "value": "yes",
      "quote": "breathing very fast"
    }
  ],
  "escalate": false,
  "reason": "No urgent signs detected. Reported non-urgent signs: reduced_appetite, rapid_breathing."
}
```

### Sample 2: Acute Emergency (Foam & Recumbency)
- **Input**: `"There's foam coming from her nose and she can't get up."`
- **Response**:
```json
{
  "findings": [
    {
      "sign_id": "froth_nose_mouth",
      "value": "yes",
      "quote": "foam coming from her nose"
    },
    {
      "sign_id": "unable_to_stand",
      "value": "yes",
      "quote": "can't get up"
    }
  ],
  "escalate": true,
  "reason": "Urgent veterinary attention required due to critical signs: froth_nose_mouth (yes), unable_to_stand (yes)."
}
```

### Sample 3: Uncertainty & Shona Cultural Term (*mbandabvu*)
- **Input**: `"I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear."`
- **Response**:
```json
{
  "findings": [
    {
      "sign_id": "cloudy_eye",
      "value": "not_sure",
      "quote": "eyes look a bit cloudy but I'm not sure"
    },
    {
      "sign_id": "swelling",
      "value": "yes",
      "quote": "mbandabvu under the ear"
    }
  ],
  "escalate": true,
  "reason": "Urgent veterinary attention required due to critical signs: cloudy_eye (not_sure)."
}
```
*Note: Because `cloudy_eye` is suspected (`not_sure`), the animal is escalated immediately to prevent permanent blindness or corridor disease complications.*

### Sample 4: Ticks with Reassuring Comment
- **Input**: `"The calf is fine, just lots of ticks on the ears."`
- **Response**:
```json
{
  "findings": [
    {
      "sign_id": "ticks",
      "value": "yes",
      "quote": "lots of ticks on the ears"
    }
  ],
  "escalate": false,
  "reason": "No urgent signs detected. Reported non-urgent signs: ticks."
}
```
*Note: The general statement "the calf is fine" does NOT generate findings for all signs.*

### Sample 5: Code-Switching (Shona & English)
- **Input**: `"Mombe yangu haisi kudya, and her urine is dark like tea."`
- **Response**:
```json
{
  "findings": [
    {
      "sign_id": "reduced_appetite",
      "value": "yes",
      "quote": "Mombe yangu haisi kudya"
    },
    {
      "sign_id": "red_urine",
      "value": "yes",
      "quote": "urine is dark like tea"
    }
  ],
  "escalate": false,
  "reason": "No urgent signs detected. Reported non-urgent signs: reduced_appetite, red_urine."
}
```

---

## Design Decisions & Why

### 1. Separation of Semantic Extraction and Clinical Triage
- **The Problem**: Relying on an LLM to decide whether to escalate is dangerous and non-deterministic. Models can hallucinate urgency thresholds or fluctuate under minor prompt variations.
- **The Solution**: We delegate **only linguistic understanding** to the LLM (extracting sign mentions, verbatim quotes, and confidence). Escalation is determined by **plain Python code** (`app/escalation.py`) containing strict medical rules.

### 2. Pluggable `LLMProvider` Interface
- **The Problem**: Tightly coupling FastAPI routes to the Gemini SDK makes tests slow, flaky, dependent on internet connectivity, and expensive.
- **The Solution**: `LLMProvider` is an abstract interface. The FastAPI app injects it via `Depends(get_llm_provider)`. In production, `GeminiLLMProvider` calls Google Gemini at `temperature=0` with JSON schema enforcement. In tests, `MockLLMProvider` supplies canned outputs instantly without an API key or network access.

### 3. Multi-Layer Defensive Validation
- **Allowed Sign Whitelisting**: Any sign not in the 9 predefined constants (e.g. `fever` or `cough`) is discarded.
- **Quote Grounding**: The LLM's quote is verified against the raw message text. If the model invents a quote not found in the farmer's text, the finding is discarded. If casing differs, it is snapped to the original text.
- **Sign Deduplication**: Repeated references to the same sign are deduplicated, keeping the first occurrence.

### 4. Prompt Injection Defense
- Farmer messages on WhatsApp are untrusted input. The prompt template encapsulates the user's message in delimiters and provides an explicit system directive to treat the text strictly as clinical data, ignoring any commands or prompt hijacking attempts within it.

### 5. Safe Failure Policy (Retry & Fail Fast)
- If the LLM throws a network error or outputs unparseable JSON, the extractor retries **once**.
- If the failure persists, the endpoint raises `HTTP 502 Bad Gateway`.
- **Critical Safety Invariant**: The service **never** silently falls back to `escalate=False` on an error. In livestock health, a false assumption of safety can result in the death of a valuable animal.

---

## Measuring Extraction Accuracy

To rigorously evaluate extraction performance in production, I would establish the following evaluation framework:

### 1. Labelled Golden Dataset
- Build a curated dataset of ~500 real WhatsApp messages from Zimbabwean smallholders.
- Have a local veterinarian annotate:
  1. Present signs (`yes`, `no`, `not_sure`).
  2. The exact text spans (quotes).
  3. Escalation status.
- Stratify the evaluation set into:
  - English-only messages (~35%).
  - Shona-only messages (~35%).
  - Code-switched messages (~30%).

### 2. Asymmetric Metrics: Precision vs. Recall
- **Recall is prioritized over Precision for escalating signs**:
  Missing a sick cow with froth from the mouth or recumbency (False Negative) can be fatal to the herd and ruinous for a subsistence farmer. A false alarm (False Positive) merely causes a triage phone call.
- Therefore, we optimize for **High Recall (>98%) on critical signs** (`froth_nose_mouth`, `unable_to_stand`, `cloudy_eye`) using \(F_2\)-score, while maintaining acceptable Precision (>90%) on non-urgent signs.

### 3. Quote Grounding & Exact Match
- Track **Quote Alignment Rate**: percentage of extracted quotes that are verbatim substrings of the original text.
- Measure Character-Level Jaccard Index between model quotes and human-annotated ground-truth quotes.

### 4. Disaggregated Language Tracking
- Monitor per-language slices in our CI/CD pipeline:
  If overall accuracy is 92%, but Shona messages score 78% while English scores 98%, the system has an unacceptable equity gap that must be addressed with few-shot Shona examples.

---

## What I Would Do With More Time

1. **WhatsApp Audio Voice Note Transcription**:
   Many rural farmers prefer sending WhatsApp voice notes rather than typing. Integrating an acoustic model (like Whisper fine-tuned on Zimbabwean Shona/Ndebele accents) would expand accessibility significantly.
2. **Asynchronous Ingestion & Task Queue**:
   WhatsApp webhook deliveries can spike during disease outbreaks. Moving extraction behind an asynchronous worker queue (e.g. Celery / ARQ with Redis) would prevent webhook timeouts and rate-limiting.
3. **Temporal & Entity Disambiguation**:
   Support multi-animal messages (e.g., *"The black cow is fine, but the brown calf stopped feeding 3 days ago"*) and temporal extraction (onset duration).
4. **Few-Shot Domain Fine-Tuning**:
   Compile a veterinary Shona/Ndebele glossary and use few-shot in-context examples to improve slang recognition (e.g., regional names for East Coast Fever / Theileriosis).
5. **Human-in-the-Loop Triage Dashboard**:
   A web portal for veterinary officers in Harare and Bulawayo where incoming messages with `not_sure` confidence can be verified with one click, feeding corrections back into the evaluation set.

---

## How I Used AI Tools

I used **Google Antigravity** as an AI coding assistant during this assignment. Antigravity helped:
- Rapidly scaffold the repository structure and Pydantic schemas.
- Generate a comprehensive offline pytest matrix covering all edge cases, escalation combinations, and mock providers.
- Formulate the bilingual prompt instructions with appropriate Shona veterinary terms.

<!-- CANDIDATE NOTES: Add notes on what you changed and checked -->
- Reviewed and refined the Pydantic schema validation to ensure invalid sign IDs and duplicate values are strictly filtered out.
- Handled edge cases in quote matching so that case discrepancies are corrected without dropping valid findings.
- Verified that all 42 tests run purely offline without requiring any API keys or network calls.
- Verified that `.env` and `.venv` are strictly ignored by Git and confirmed zero secrets exist in Git history.
