# SaluLink Chronic Treatment App

A comprehensive healthcare application for managing chronic conditions, automating diagnostic coding, and ensuring PMB (Prescribed Minimum Benefits) compliance. Built with Next.js, TypeScript, Tailwind CSS, and powered by ClinicalBERT AI.

## Features

### Core Functionality

- **AI-Powered Analysis**: Uses ClinicalBERT to analyze clinical notes and identify chronic conditions
- **Authi 1.0 Integration**: Automated matching of conditions to ICD-10 codes, treatment protocols, and medications
- **Complete Workflow Management**: From initial diagnosis through ongoing management and referrals
- **PMB Compliance**: Ensures all treatments and medications comply with medical scheme rules

### Key Workflows

1. **Initial Diagnosis Workflow**
   - Clinical note input and AI analysis
   - Condition identification with confidence scores
   - ICD-10 code selection
   - Diagnostic basket management
   - Medication prescription with plan-based filtering
   - PDF export for claims

2. **Ongoing Management**
   - Add and document ongoing treatments
   - Track procedure limits
   - Update case status

3. **Medication Reports**
   - Review current medications
   - Document follow-up results
   - Prescribe new medications with motivation letters
   - Export comprehensive medication reports

4. **Specialist Referrals**
   - Generate referral documentation
   - Include complete case summary
   - Set urgency levels
   - Export referral PDFs

### Medical Plan Support

- Core
- Priority
- Saver
- Executive
- Comprehensive

## Technology Stack

### Frontend

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Zustand** - State management with persistence
- **jsPDF** - PDF generation
- **Papa Parse** - CSV parsing

### Backend

- **Python/FastAPI** - REST API for AI analysis
- **ClinicalBERT** - Medical NLP model
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face model integration

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.9+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Chronic App 2"
```

### 2. Frontend Setup

```bash
# Install dependencies
npm install

# The CSV files should already be in the public folder
# If not, copy them:
# cp "Chronic Conditions.csv" "Medicine List.csv" "Treatment Basket.csv" public/
```

### 3. Python Backend Setup

```bash
# Navigate to Python backend
cd python-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Start the Python Backend (Terminal 1)

```bash
cd python-backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The API will start on `http://localhost:8000`

### Start the Next.js Frontend (Terminal 2)

```bash
npm run dev
```

The app will start on `http://localhost:3000`

## Usage Guide

### Creating a New Case

1. **Enter Clinical Note**: Paste or type specialist notes in the text area
2. **Analyze**: Click the "Analyze Note" button to process with ClinicalBERT
3. **Select Condition**: Choose the most appropriate chronic condition from AI suggestions
4. **Select ICD-10 Code**: Pick the specific diagnosis code
5. **Add Diagnostic Tests**: Select required tests from the diagnostic basket
6. **Document Results**: Add findings and upload images for each test
7. **Prescribe Medication**: Filter by medical plan and select appropriate medications
8. **Add Registration Note**: Document the rationale for medication prescription
9. **Save & Export**: Save the case and/or export as PDF for claims submission

### Managing Existing Cases

1. Click the **menu icon** (top right) to open the sidebar
2. Select a saved case to load it
3. Choose from available actions:
   - **Ongoing Management**: Add follow-up treatments
   - **Medication Report**: Update medication status or prescribe new medications
   - **Create Referral**: Generate specialist referral documentation

### Exporting Documents

All workflows support PDF export:

- Initial diagnosis claims
- Ongoing management reports
- Medication reports with motivation letters
- Specialist referrals with case summaries

## Data Files

The application uses three CSV datasets:

1. **Chronic Conditions.csv** - ICD-10 codes and condition descriptions
2. **Medicine List.csv** - Approved medications by condition and plan
3. **Treatment Basket.csv** - Diagnostic and ongoing management protocols

These files should be located in the `public/` folder for the frontend to access.

## Architecture

### Frontend Architecture

```
app/
├── api/           # Next.js API routes
├── globals.css    # Global styles
├── layout.tsx     # Root layout
└── page.tsx       # Main application page

components/
├── ClinicalNoteInput.tsx
├── ConditionSelection.tsx
├── IcdCodeSelection.tsx
├── DiagnosticBasket.tsx
├── MedicationSelection.tsx
├── OngoingManagement.tsx
├── MedicationReport.tsx
├── Referral.tsx
├── CaseActions.tsx
└── Sidebar.tsx

lib/
├── store.ts       # Zustand state management
├── dataService.ts # CSV data handling
└── pdfExport.ts   # PDF generation
```

### Backend Architecture

```
python-backend/
├── main.py           # FastAPI application
└── requirements.txt  # Python dependencies
```

### Data Flow

1. User enters clinical note
2. Frontend calls `/api/analyze` endpoint
3. Next.js API route forwards to Python backend
4. Python backend uses ClinicalBERT to analyze text
5. Authi 1.0 logic matches keywords to conditions
6. Results returned to frontend
7. User completes workflow
8. Data saved locally (Zustand persist)
9. PDFs generated client-side

## API Endpoints

### Python Backend

#### `GET /`

Health check endpoint

#### `GET /health`

Returns model and data loading status

#### `POST /analyze`

Analyzes clinical notes and returns matched conditions

**Request:**

```json
{
  "clinical_note": "Patient presents with wheezing..."
}
```

**Response:**

```json
{
  "extracted_keywords": ["patient", "wheezing", ...],
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

### Next.js API

#### `POST /api/analyze`

Proxies requests to Python backend

## Troubleshooting

### Python Backend Won't Start

- Ensure Python 3.9+ is installed: `python --version`
- Check virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`
- Try port 8001 if 8000 is busy: `uvicorn main:app --port 8001`

### Frontend Can't Connect to Backend

- Verify Python backend is running on port 8000
- Check console for CORS errors
- Update `app/api/analyze/route.ts` if using different port

### CSV Files Not Loading

- Ensure CSV files are in `public/` folder
- Check browser console for fetch errors
- Verify file names match exactly (case-sensitive)

### Model Loading Takes Too Long

- First load downloads ClinicalBERT model (~440MB)
- Subsequent loads use cached model
- Allow 1-2 minutes for initial startup

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- Initial model load: ~1-2 minutes
- Analysis time: ~2-5 seconds
- PDF generation: Instant (client-side)
- Case loading: Instant (local storage)

## Security Notes

- All patient data stored locally in browser
- No data sent to external servers (except Python backend)
- Python backend runs locally
- CORS restricted to localhost:3000

## Future Enhancements

- [ ] Cloud deployment options
- [ ] Multi-user support with authentication
- [ ] Database integration for persistent storage
- [ ] Real-time collaboration features
- [ ] Mobile app version
- [ ] Integration with medical scheme APIs
- [ ] Enhanced AI models for better accuracy

## License

Proprietary - SaluLink Operations

## Support

For technical support or questions, contact the SaluLink development team.

---

**Version**: 1.0.0  
**Last Updated**: December 2025
