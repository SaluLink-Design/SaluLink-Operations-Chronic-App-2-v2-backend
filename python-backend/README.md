# SaluLink Authi Python Backend

This backend service implements **Authi 1.0**, an AI-powered diagnostic coding automation system for the SaluLink Chronic Treatment App.

## Authi 1.0 Architecture

### AI Components

1. **ClinicalBERT** (fine-tuned for chronic conditions)
   - Extracts symptoms, diagnostic descriptions, and clinical terminology from free-text notes
   - Produces contextualized keyword embeddings for semantic matching
   - Model: `emilyalsentzer/Bio_ClinicalBERT`
   - Filters out stop words and irrelevant terms to focus on clinical content

2. **Authi 1.0 Matching System**
   - Maps extracted keywords to chronic condition entries using cosine similarity
   - Returns **3–5 chronic condition suggestions** with ICD-10 codes
   - Uses intelligent scoring: combines maximum similarity (70%) and average similarity (30%)
   - Adaptive threshold adjustment to ensure minimum 3 results when possible
   - Default similarity threshold: 0.65

### Key Features

- ✅ Returns exactly 3-5 condition matches per analysis
- ✅ Improved accuracy through semantic similarity matching
- ✅ Context-aware keyword extraction
- ✅ ICD-10 code automation
- ✅ PMB (Prescribed Minimum Benefits) compliance support

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
  "extracted_keywords": ["patient", "wheezing", "breath", "shortness", "respiratory", ...],
  "matched_conditions": [
    {
      "condition": "Asthma",
      "icd_code": "J45.9",
      "icd_description": "Asthma, unspecified",
      "similarity_score": 0.87
    },
    {
      "condition": "Asthma",
      "icd_code": "J45.0",
      "icd_description": "Predominantly allergic asthma",
      "similarity_score": 0.82
    },
    {
      "condition": "Chronic obstructive pulmonary disease",
      "icd_code": "J44.9",
      "icd_description": "Chronic obstructive pulmonary disease, unspecified",
      "similarity_score": 0.76
    }
  ]
}
```

**Note:** The API will return between 3-5 matched conditions, ranked by similarity score.

## How It Works

1. **Input Processing**: Clinical note is received via POST request
2. **Keyword Extraction**: ClinicalBERT tokenizes and extracts clinical keywords with embeddings
3. **Condition Matching**: Authi 1.0 compares keyword embeddings against 209 chronic condition entries
4. **Scoring & Ranking**: Conditions are scored using cosine similarity and ranked
5. **Response**: Top 3-5 conditions returned with ICD codes and confidence scores

## Performance

- **Processing Time**: ~1-3 seconds per clinical note
- **Accuracy**: Enhanced through semantic similarity and contextualized embeddings
- **Coverage**: 209 chronic condition entries with ICD-10 codes
- **Model Size**: ~440MB (Bio_ClinicalBERT)
