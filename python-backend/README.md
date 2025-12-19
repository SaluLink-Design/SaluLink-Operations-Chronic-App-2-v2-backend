# SaluLink Authi Python Backend

This backend service provides ClinicalBERT analysis for the SaluLink Chronic Treatment App.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python main.py
```

Or use uvicorn directly:

```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /analyze` - Analyze clinical note

### Example Request

```json
POST /analyze
{
  "clinical_note": "Patient presents with wheezing and shortness of breath..."
}
```

### Example Response

```json
{
  "extracted_keywords": ["patient", "wheezing", "breath", ...],
  "matched_conditions": [
    {
      "condition": "Asthma",
      "icd_code": "J45.9",
      "icd_description": "Asthma, unspecified",
      "similarity_score": 0.85
    }
  ]
}
```
