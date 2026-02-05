"""
SaluLink Chronic App - Python Backend API
Implements Authi 1.0 AI Components for Diagnostic Coding Automation

AI Components:
1. ClinicalBERT (fine-tuned for chronic conditions)
   - Extracts symptoms, diagnostic descriptions, and clinical terminology
   - Produces keyword set for condition matching
   
2. Authi 1.0 Matching System
   - Maps extracted keywords to chronic condition entries
   - Returns 3–5 chronic condition suggestions with ICD codes
   - Uses cosine similarity with intelligent scoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="SaluLink Authi API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
tokenizer = None
model = None
chronic_condition_embeddings = []


class AnalysisRequest(BaseModel):
    clinical_note: str


class KeywordMatch(BaseModel):
    keyword: str
    similarity_score: float


class NoteQualityScore(BaseModel):
    completeness_score: int  # 0-100
    missing_elements: List[str]
    warnings: List[str]


class MatchedConditionResponse(BaseModel):
    condition: str
    icd_code: str
    icd_description: str
    similarity_score: float
    is_confirmed: bool = False  # True if condition is explicitly mentioned in the note
    triggering_keywords: List[KeywordMatch] = []  # Top keywords that triggered this match
    match_explanation: str = ""  # Explanation of how the match was made
    suggested_icd_code: Optional[str] = None  # Most relevant ICD code
    icd_confidence: Optional[float] = None  # Confidence in ICD suggestion
    alternative_icd_codes: List[str] = []  # Other valid ICD options


class AnalysisResponse(BaseModel):
    extracted_keywords: List[str]
    matched_conditions: List[MatchedConditionResponse]
    confirmed_count: int = 0  # Number of conditions directly mentioned in note
    note_quality: NoteQualityScore


def load_model():
    """Initialize ClinicalBERT model and tokenizer"""
    global tokenizer, model
    
    print("Loading ClinicalBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()
    print("Model loaded successfully!")


def load_chronic_conditions():
    """Load and process chronic conditions with embeddings"""
    global chronic_condition_embeddings
    
    print("Loading chronic conditions...")
    # Use local CSV file (copied for Railway deployment)
    csv_path = Path(__file__).parent / "Chronic Conditions.csv"
    
    if not csv_path.exists():
        # Fallback to parent directory for local development
        csv_path = Path(__file__).parent.parent / "Chronic Conditions.csv"
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    chronic_condition_embeddings = []
    
    for _, row in df.iterrows():
        description = row['ICD-Code Description']
        condition = row['CHRONIC CONDITIONS']
        icd_code = row['ICD-Code']
        
        # Extract keywords and embeddings
        _, embeddings = extract_keywords_clinicalbert(description)
        
        if embeddings.nelement() > 0:
            averaged_embedding = torch.mean(embeddings, dim=0)
        else:
            averaged_embedding = None
        
        chronic_condition_embeddings.append({
            'condition': condition,
            'icd_code': icd_code,
            'icd_description': description,
            'embedding': averaged_embedding
        })
    
    print(f"Loaded {len(chronic_condition_embeddings)} chronic condition entries")


def extract_keywords_clinicalbert(text: str):
    """
    ClinicalBERT (fine-tuned for chronic conditions)
    Responsible for:
    - Extracting symptoms, diagnostic descriptions, and clinical terminology
    - Producing the keyword set for condition matching
    
    Processes clinical text and extracts meaningful keywords with embeddings
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings
    last_hidden_state = outputs.last_hidden_state
    
    # Get tokens
    input_ids = inputs['input_ids'].squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    extracted_keywords = []
    keyword_embeddings = []
    current_word = ""
    current_embedding_indices = []
    
    # Clinical stop words to filter out (common non-diagnostic terms)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                  'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    # Process tokens
    for i, token in enumerate(tokens):
        # Skip special tokens
        if token in tokenizer.all_special_tokens:
            if current_word and len(current_word) > 2 and current_word.lower() not in stop_words:
                avg_embedding = torch.mean(last_hidden_state[0, current_embedding_indices, :], dim=0)
                extracted_keywords.append(current_word)
                keyword_embeddings.append(avg_embedding)
            current_word = ""
            current_embedding_indices = []
            continue
        
        # Reassemble subword tokens
        if token.startswith('##'):
            current_word += token[2:]
            current_embedding_indices.append(i)
        else:
            if current_word and len(current_word) > 2 and current_word.lower() not in stop_words:
                avg_embedding = torch.mean(last_hidden_state[0, current_embedding_indices, :], dim=0)
                extracted_keywords.append(current_word)
                keyword_embeddings.append(avg_embedding)
            current_word = token
            current_embedding_indices = [i]
    
    # Add last word
    if current_word and len(current_word) > 2 and current_word.lower() not in stop_words:
        avg_embedding = torch.mean(last_hidden_state[0, current_embedding_indices, :], dim=0)
        extracted_keywords.append(current_word)
        keyword_embeddings.append(avg_embedding)
    
    embeddings_tensor = torch.stack(keyword_embeddings) if keyword_embeddings else torch.tensor([])
    return extracted_keywords, embeddings_tensor


def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    embedding1 = embedding1.squeeze()
    embedding2 = embedding2.squeeze()
    
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)
    
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)


def validate_note_completeness(clinical_text: str) -> Dict:
    """
    Validate the completeness of a clinical note
    
    Checks for:
    - Clinical indicators (symptoms, diagnoses, patient history)
    - Measurements (vitals, lab values)
    - Temporal information (duration, onset, frequency)
    - Severity markers (mild/moderate/severe, quantitative descriptions)
    
    Returns:
        Dictionary with completeness_score (0-100), missing_elements, and warnings
    """
    import re
    
    clinical_text_lower = clinical_text.lower()
    score = 0
    max_score = 100
    missing_elements = []
    warnings = []
    
    # Check for clinical indicators (25 points)
    clinical_indicators = {
        'symptoms': ['pain', 'swelling', 'fever', 'cough', 'dyspnea', 'wheezing', 'fatigue', 
                    'nausea', 'vomiting', 'diarrhea', 'headache', 'dizziness', 'weakness',
                    'shortness of breath', 'chest pain', 'abdominal pain', 'symptoms', 'presenting'],
        'diagnoses': ['diagnosed', 'diagnosis', 'condition', 'disease', 'disorder', 'syndrome'],
        'history': ['history', 'previous', 'prior', 'past', 'chronic', 'long-term', 'ongoing']
    }
    
    has_symptoms = any(term in clinical_text_lower for term in clinical_indicators['symptoms'])
    has_diagnosis = any(term in clinical_text_lower for term in clinical_indicators['diagnoses'])
    has_history = any(term in clinical_text_lower for term in clinical_indicators['history'])
    
    if has_symptoms:
        score += 10
    else:
        missing_elements.append("symptoms or presenting complaints")
        warnings.append("Consider documenting: patient symptoms or presenting complaints")
    
    if has_diagnosis:
        score += 10
    else:
        missing_elements.append("diagnosis or condition mention")
    
    if has_history:
        score += 5
    else:
        warnings.append("Consider adding: patient medical history")
    
    # Check for measurements/vitals (30 points)
    vital_patterns = {
        'blood_pressure': [r'\d+/\d+\s*(mmhg|mm hg)?', r'bp:?\s*\d+/\d+', r'blood pressure'],
        'heart_rate': [r'\d+\s*bpm', r'hr:?\s*\d+', r'heart rate', r'pulse'],
        'temperature': [r'\d+\.?\d*\s*(°c|°f|celsius|fahrenheit)', r'temp:?\s*\d+', r'temperature'],
        'glucose': [r'glucose:?\s*\d+', r'blood sugar', r'bg:?\s*\d+', r'bsl:?\s*\d+'],
        'hba1c': [r'hba1c:?\s*\d+\.?\d*%?', r'glycated hemoglobin', r'glycohemoglobin'],
        'lab_values': [r'\d+\.?\d*\s*(mg/dl|mmol/l|g/dl)', r'lab results', r'laboratory']
    }
    
    vitals_found = 0
    for vital_type, patterns in vital_patterns.items():
        if any(re.search(pattern, clinical_text_lower) for pattern in patterns):
            vitals_found += 1
    
    # Score based on number of vitals/measurements found
    if vitals_found >= 3:
        score += 30
    elif vitals_found == 2:
        score += 20
        warnings.append("Consider adding more measurements: vitals or lab values")
    elif vitals_found == 1:
        score += 10
        missing_elements.append("comprehensive vital signs or lab values")
        warnings.append("Consider documenting: blood pressure, heart rate, temperature, or relevant lab values")
    else:
        missing_elements.append("vital signs and measurements")
        warnings.append("Consider documenting: blood pressure, heart rate, temperature, glucose, or other relevant measurements")
    
    # Check for temporal information (20 points)
    temporal_patterns = {
        'duration': [r'\d+\s*(day|week|month|year)', r'for\s+\d+', r'since\s+\d+', 
                    r'duration', r'ongoing', r'chronic'],
        'onset': [r'onset', r'started', r'began', r'first noticed', r'initially'],
        'frequency': [r'daily', r'weekly', r'monthly', r'frequently', r'occasionally', 
                     r'intermittent', r'continuous', r'constant', r'times per']
    }
    
    temporal_found = 0
    for temporal_type, patterns in temporal_patterns.items():
        if any(re.search(pattern, clinical_text_lower) for pattern in patterns):
            temporal_found += 1
    
    if temporal_found >= 2:
        score += 20
    elif temporal_found == 1:
        score += 10
        warnings.append("Consider adding: symptom duration or frequency information")
    else:
        missing_elements.append("temporal information (duration, onset, frequency)")
        warnings.append("Consider documenting: when symptoms started, how long they've lasted, or how often they occur")
    
    # Check for severity markers (15 points)
    severity_patterns = [
        r'mild', r'moderate', r'severe', r'critical', r'acute', r'chronic',
        r'grade\s+\d', r'stage\s+\d', r'class\s+\d',
        r'worsening', r'improving', r'stable', r'deteriorating',
        r'significantly', r'markedly', r'slightly', r'minimally'
    ]
    
    has_severity = any(re.search(pattern, clinical_text_lower) for pattern in severity_patterns)
    
    if has_severity:
        score += 15
    else:
        missing_elements.append("severity indicators")
        warnings.append("Consider documenting: severity level (mild/moderate/severe) or disease progression")
    
    # Check for treatment/medication information (10 points)
    treatment_patterns = [
        r'treatment', r'medication', r'therapy', r'prescribed', r'taking',
        r'drug', r'medicine', r'dose', r'dosage', r'mg', r'mcg'
    ]
    
    has_treatment = any(re.search(pattern, clinical_text_lower) for pattern in treatment_patterns)
    
    if has_treatment:
        score += 10
    else:
        warnings.append("Consider adding: current medications or treatments if applicable")
    
    # Ensure score is within 0-100
    score = min(max(score, 0), max_score)
    
    return {
        'completeness_score': score,
        'missing_elements': missing_elements,
        'warnings': warnings
    }


def detect_negation_context(clinical_text, condition_term):
    """
    Detect if a condition mention is negated (e.g., "no diabetes", "denies hypertension")
    Returns: (is_negated: bool, context: str)
    
    This is critical to avoid false positives when conditions are mentioned but ruled out.
    
    ENHANCED: Also checks for symptom-based negations (e.g., "denies seizures" for epilepsy)
    """
    import re
    
    # Condition-specific symptom keywords that indicate the condition
    condition_symptom_map = {
        'epilepsy': ['seizure', 'seizures', 'convulsion', 'convulsions', 'fits', 'epileptic'],
        'diabetes': ['hyperglycemia', 'hyperglycemic', 'polyuria', 'polydipsia'],
        'hypertension': ['elevated blood pressure', 'high blood pressure', 'elevated bp', 'hypertensive'],
        'asthma': ['wheezing', 'bronchospasm', 'asthmatic'],
        'cardiac failure': ['heart failure', 'chf', 'cardiac decompensation'],
        'haemophilia': ['bleeding disorder', 'clotting disorder', 'factor deficiency'],
        'hypothyroidism': ['thyroid deficiency', 'underactive thyroid'],
        'copd': ['chronic bronchitis', 'emphysema'],
        'chronic renal disease': ['kidney failure', 'renal failure', 'kidney disease'],
        'cardiomyopathy': ['cardiomyopathic'],
        'hyperlipidaemia': ['high cholesterol', 'dyslipidemia']
    }
    
    clinical_text_lower = clinical_text.lower()
    condition_term_lower = condition_term.lower()
    
    # Negation patterns - comprehensive list of medical negation terms
    # Allow up to 10 words between negation term and condition (e.g., "denies symptoms, conditions, or X")
    negation_patterns = [
        r'\bno\s+(?:history\s+(?:of\s+)?)?' + re.escape(condition_term),
        r'\bdenies\s+(?:any\s+)?(?:history\s+(?:of\s+)?)?(?:\w+[,\s]+){0,10}?' + re.escape(condition_term),
        r'\brules?\s+out\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\br/o\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bruled\s+out\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bnegative\s+for\s+(?:\w+[,\s]+){0,10}?' + re.escape(condition_term),
        r'\bwithout\s+(?:evidence\s+(?:of\s+)?)?(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\babsent\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bnot\s+consistent\s+with\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bno\s+signs?\s+of\s+(?:\w+[,\s]+){0,10}?' + re.escape(condition_term),
        r'\bunlikely\s+(?:to\s+be\s+)?(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bdiscontinued\s+(?:\w+[,\s]+){0,3}?' + re.escape(condition_term),
        r'\bresolved\s+(?:\w+[,\s]+){0,3}?' + re.escape(condition_term),
        r'\bfree\s+(?:of|from)\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bexclude[ds]?\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bnot\s+(?:have|has|having)\s+(?:\w+[,\s]+){0,5}?' + re.escape(condition_term),
        r'\bno\s+longer\s+(?:has|have|having)\s+(?:\w+[,\s]+){0,3}?' + re.escape(condition_term)
    ]
    
    # Check direct condition term negation
    for pattern in negation_patterns:
        match = re.search(pattern, clinical_text_lower, re.IGNORECASE)
        if match:
            return (True, match.group())
    
    # Check symptom-based negation (e.g., "denies seizures" for epilepsy)
    # Find the base condition name for lookup
    for base_condition, symptoms in condition_symptom_map.items():
        if base_condition in condition_term_lower or condition_term_lower in base_condition:
            # Check if any of the symptoms are negated
            for symptom in symptoms:
                # Allow up to 10 intervening words for lists (e.g., "denies symptoms, conditions, or X")
                symptom_negation_patterns = [
                    r'\bno\s+(?:history\s+(?:of\s+)?)?(?:\w+[,\s]+){0,10}?' + re.escape(symptom),
                    r'\bdenies\s+(?:any\s+)?(?:history\s+(?:of\s+)?)?(?:\w+[,\s]+){0,10}?' + re.escape(symptom),
                    r'\bnegative\s+for\s+(?:\w+[,\s]+){0,10}?' + re.escape(symptom),
                    r'\bwithout\s+(?:\w+[,\s]+){0,10}?' + re.escape(symptom),
                    r'\bno\s+signs?\s+of\s+(?:\w+[,\s]+){0,10}?' + re.escape(symptom),
                    r'\brules?\s+out\s+(?:\w+[,\s]+){0,5}?' + re.escape(symptom),
                    r'\babsent\s+(?:\w+[,\s]+){0,5}?' + re.escape(symptom)
                ]
                
                for pattern in symptom_negation_patterns:
                    match = re.search(pattern, clinical_text_lower, re.IGNORECASE)
                    if match:
                        return (True, match.group())
    
    return (False, None)


def get_condition_symptom_indicators():
    """
    Returns symptom patterns that strongly indicate specific conditions.
    This improves semantic matching accuracy by recognizing condition-specific terminology.
    """
    return {
        'Asthma': [
            'wheezing', 'bronchospasm', 'shortness of breath on exertion',
            'nocturnal cough', 'dyspnea', 'chest tightness', 'peak flow',
            'inhaler use', 'albuterol', 'bronchodilator', 'salbutamol',
            'reactive airway', 'bronchial hyperresponsiveness'
        ],
        'Diabetes Mellitus Type 1': [
            'polyuria', 'polydipsia', 'polyphagia', 'weight loss',
            'ketoacidosis', 'dka', 'insulin therapy', 'hyperglycemia',
            'blood glucose', 'a1c', 'hba1c', 'insulin pump',
            'continuous glucose monitor', 'cgm', 'diabetic ketoacidosis'
        ],
        'Diabetes Mellitus Type 2': [
            'polyuria', 'polydipsia', 'hyperglycemia', 'metformin',
            'oral hypoglycemic', 'insulin resistance', 'metabolic syndrome',
            'elevated glucose', 'a1c', 'hba1c', 'prediabetes',
            'glyburide', 'glipizide', 'acarbose', 'sitagliptin'
        ],
        'Hypertension': [
            'elevated bp', 'systolic pressure', 'diastolic pressure',
            'hypertensive crisis', 'headache', 'antihypertensive',
            'amlodipine', 'lisinopril', 'losartan', 'blood pressure',
            'enalapril', 'valsartan', 'ramipril', 'elevated blood pressure'
        ],
        'Cardiac Failure': [
            'dyspnea on exertion', 'orthopnea', 'paroxysmal nocturnal dyspnea',
            'pnd', 'edema', 'peripheral edema', 'pulmonary edema',
            'jugular venous distension', 'jvd', 'rales', 'crackles',
            'ejection fraction', 'bnp', 'furosemide', 'lasix',
            'reduced ejection fraction', 'systolic dysfunction', 'biventricular'
        ],
        'Chronic Renal Disease': [
            'elevated creatinine', 'decreased egfr', 'proteinuria',
            'hematuria', 'uremia', 'dialysis', 'renal replacement',
            'fluid retention', 'anemia of ckd', 'hyperkalemia',
            'uremic', 'azotemia', 'chronic kidney disease', 'renal insufficiency'
        ],
        'Cardiomyopathy': [
            'reduced ejection fraction', 'lvef', 'left ventricular dysfunction',
            'dilated heart', 'ventricular hypertrophy', 'wall motion abnormality',
            'diastolic dysfunction', 'systolic dysfunction', 'cardiomegaly'
        ],
        'Hyperlipidaemia': [
            'elevated cholesterol', 'high ldl', 'low hdl', 'triglycerides',
            'lipid panel', 'statin', 'atorvastatin', 'simvastatin',
            'pravastatin', 'rosuvastatin', 'hypercholesterolemia', 'dyslipidemia'
        ],
        'Haemophilia': [
            'prolonged bleeding', 'factor deficiency', 'spontaneous bleeding',
            'hemarthrosis', 'easy bruising', 'bleeding disorder',
            'clotting disorder', 'factor viii', 'factor ix', 'christmas disease'
        ],
        'Chronic Obstructive Pulmonary Disease': [
            'chronic cough', 'sputum production', 'dyspnea', 'barrel chest',
            'prolonged expiration', 'smoking history', 'fev1', 'spirometry',
            'tiotropium', 'ipratropium', 'oxygen therapy', 'pack year',
            'chronic bronchitis', 'emphysema', 'airflow limitation'
        ],
        'Epilepsy': [
            'seizure activity', 'loss of consciousness', 'tonic clonic',
            'aura', 'postictal', 'anticonvulsant', 'antiepileptic',
            'levetiracetam', 'phenytoin', 'valproic acid', 'eeg',
            'convulsions', 'epileptic', 'seizure disorder', 'grand mal', 'petit mal'
        ],
        'Hypothyroidism': [
            'elevated tsh', 'low t4', 'low t3', 'fatigue', 'cold intolerance',
            'weight gain', 'bradycardia', 'constipation', 'dry skin',
            'levothyroxine', 'synthroid', 'thyroid replacement',
            'myxedema', 'hashimoto', 'underactive thyroid', 'thyroid deficiency'
        ]
    }


def find_direct_condition_matches(clinical_text):
    """
    Direct condition name matching - checks if condition names appear in clinical text
    This ensures we catch conditions that are explicitly mentioned (CONFIRMED conditions)
    
    Uses multiple matching strategies:
    1. Exact condition name match (highest confidence - CONFIRMED)
    2. Common medical term aliases (e.g., "diabetic" -> Diabetes, "hypertensive" -> Hypertension)
    3. ICD description keyword matching (for specific subtypes)
    
    Returns conditions with is_confirmed=True for explicit mentions
    """
    import re
    
    clinical_text_lower = clinical_text.lower()
    direct_matches = {}
    
    # Get unique condition names
    condition_names = set()
    for entry in chronic_condition_embeddings:
        condition_names.add(entry['condition'].lower())
    
    # Define condition aliases for common medical variations (EXPANDED for better accuracy)
    condition_aliases = {
        'diabetes mellitus type 1': [
            'type 1 diabetes', 'type i diabetes', 't1dm', 'type1 diabetes',
            'insulin-dependent diabetes', 'insulin dependent diabetes', 'iddm',
            'juvenile diabetes', 'autoimmune diabetes', 'brittle diabetes'
        ],
        'diabetes mellitus type 2': [
            'type 2 diabetes', 'type ii diabetes', 't2dm', 'type2 diabetes',
            'non-insulin-dependent diabetes', 'non insulin dependent diabetes', 'niddm',
            'adult-onset diabetes', 'metabolic diabetes', 'insulin resistance',
            'non-insulin dependent diabetes'
        ],
        'hypertension': [
            'high blood pressure', 'elevated blood pressure', 'hypertensive', 'htn',
            'bp elevated', 'raised blood pressure', 'systolic hypertension',
            'diastolic hypertension', 'malignant hypertension', 'resistant hypertension',
            'stage 1 hypertension', 'stage 2 hypertension', 'essential hypertension',
            'primary hypertension', 'secondary hypertension'
        ],
        'asthma': [
            'asthmatic', 'bronchial asthma', 'reactive airway disease', 'rad',
            'allergic asthma', 'exercise-induced asthma', 'occupational asthma',
            'severe asthma', 'status asthmaticus', 'bronchospasm',
            'extrinsic asthma', 'intrinsic asthma'
        ],
        'cardiac failure': [
            'heart failure', 'congestive heart failure', 'chf',
            'left ventricular failure', 'right heart failure', 'cardiac decompensation',
            'systolic heart failure', 'diastolic heart failure', 'hfpef', 'hfref',
            'acute heart failure', 'chronic heart failure', 'decompensated heart failure',
            'biventricular failure', 'ventricular dysfunction', 'congestive cardiac failure'
        ],
        'chronic renal disease': [
            'chronic kidney disease', 'ckd', 'renal failure', 'kidney failure',
            'nephropathy', 'renal insufficiency', 'kidney disease', 'esrd',
            'end stage renal disease', 'chronic kidney failure', 'renal impairment',
            'stage 1 ckd', 'stage 2 ckd', 'stage 3 ckd', 'stage 4 ckd', 'stage 5 ckd',
            'glomerulonephritis', 'pyelonephritis', 'diabetic nephropathy',
            'chronic renal failure', 'chronic renal insufficiency'
        ],
        'cardiomyopathy': [
            'cardiomyopathic', 'dilated cardiomyopathy', 'hypertrophic cardiomyopathy',
            'restrictive cardiomyopathy', 'dcm', 'hcm', 'ischaemic cardiomyopathy',
            'ischemic cardiomyopathy', 'alcoholic cardiomyopathy', 'viral cardiomyopathy',
            'idiopathic cardiomyopathy', 'hypertrophic obstructive cardiomyopathy', 'hocm'
        ],
        'hyperlipidaemia': [
            'hyperlipidemia', 'high cholesterol', 'dyslipidemia', 'dyslipidaemia',
            'elevated cholesterol', 'hypercholesterolemia', 'hypercholesterolaemia',
            'hypertriglyceridemia', 'mixed hyperlipidemia', 'familial hypercholesterolemia',
            'elevated ldl', 'low hdl', 'lipid disorder', 'hyperlipemia'
        ],
        'haemophilia': [
            'hemophilia', 'factor viii deficiency', 'factor ix deficiency',
            'bleeding disorder', 'haemophilia a', 'haemophilia b', 'hemophilia a',
            'hemophilia b', 'christmas disease', 'clotting disorder', 'coagulation disorder'
        ],
        'chronic obstructive pulmonary disease': [
            'copd', 'emphysema', 'chronic bronchitis', 'obstructive lung disease',
            'obstructive airway disease', 'chronic obstructive airway disease',
            'coad', 'chronic airflow limitation', 'chronic airflow obstruction',
            'chronic obstructive lung disease'
        ],
        'epilepsy': [
            'seizure disorder', 'seizures', 'epileptic', 'convulsions', 'fits',
            'focal seizures', 'generalized seizures', 'tonic clonic seizures',
            'petit mal', 'grand mal', 'absence seizures', 'status epilepticus',
            'refractory epilepsy', 'temporal lobe epilepsy', 'partial seizures'
        ],
        'hypothyroidism': [
            'underactive thyroid', 'low thyroid', 'thyroid deficiency',
            'myxedema', 'myxoedema', 'hashimoto', 'hashimoto thyroiditis',
            'hashimotos disease', 'primary hypothyroidism', 'secondary hypothyroidism',
            'subclinical hypothyroidism', 'thyroid insufficiency', 'hashimotos thyroiditis'
        ]
    }
    
    # Strategy 1: Direct condition name matching (CONFIRMED)
    for condition_name in condition_names:
        # Create word boundary regex pattern for accurate matching
        pattern = r'\b' + re.escape(condition_name) + r'\b'
        if re.search(pattern, clinical_text_lower):
            # Check for negation before confirming
            is_negated, negation_context = detect_negation_context(clinical_text, condition_name)
            if is_negated:
                print(f"   ⚠ Skipping negated condition: {condition_name} (context: '{negation_context}')")
                continue  # Skip this condition - it's explicitly ruled out
            
            # Find all matching entries for this condition
            for entry in chronic_condition_embeddings:
                if entry['condition'].lower() == condition_name:
                    condition_key = (entry['condition'], entry['icd_code'])
                    if condition_key not in direct_matches:
                        direct_matches[condition_key] = {
                            'condition': entry['condition'],
                            'icd_code': entry['icd_code'],
                            'icd_description': entry['icd_description'],
                            'similarity_score': 0.98,  # High score for direct matches
                            'match_type': 'confirmed',
                            'is_confirmed': True  # CONFIRMED - explicitly mentioned
                        }
    
    # Strategy 2: Check for condition aliases (also CONFIRMED)
    for canonical_condition, aliases in condition_aliases.items():
        alias_found = False
        for alias in aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, clinical_text_lower):
                # Check for negation before confirming
                is_negated, negation_context = detect_negation_context(clinical_text, alias)
                if is_negated:
                    print(f"   ⚠ Skipping negated alias: {alias} -> {canonical_condition} (context: '{negation_context}')")
                    alias_found = True  # Mark as found but negated
                    continue  # Skip this alias - it's negated
                
                # Find the canonical condition in our database
                for entry in chronic_condition_embeddings:
                    if entry['condition'].lower() == canonical_condition:
                        condition_key = (entry['condition'], entry['icd_code'])
                        if condition_key not in direct_matches:
                            direct_matches[condition_key] = {
                                'condition': entry['condition'],
                                'icd_code': entry['icd_code'],
                                'icd_description': entry['icd_description'],
                                'similarity_score': 0.95,  # High score for alias matches
                                'match_type': 'confirmed',
                                'is_confirmed': True  # CONFIRMED - alias explicitly mentioned
                            }
                alias_found = True
                break  # Stop checking aliases once found (whether negated or confirmed)
        
        # If we found a negated alias, skip checking other aliases for this condition
        if alias_found:
            continue
    
    # Strategy 3: Check for specific ICD description terms (for subtypes of confirmed conditions)
    # Only use this to find specific ICD codes for already-confirmed conditions
    confirmed_condition_names = set(match['condition'] for match in direct_matches.values())
    
    for entry in chronic_condition_embeddings:
        # Only check ICD descriptions for conditions we've already confirmed
        if entry['condition'] not in confirmed_condition_names:
            continue
            
        icd_desc_lower = entry['icd_description'].lower()
        
        # Extract significant medical terms
        medical_terms = [word for word in re.findall(r'\b[a-z]{3,}\b', icd_desc_lower) 
                        if word not in {'with', 'without', 'and', 'the', 'disease', 'syndrome', 
                                       'other', 'unspecified', 'disorder', 'complicating', 
                                       'specified', 'due', 'mellitus', 'type', 'related'}]
        
        # Check if any significant medical term appears in the clinical text
        for term in medical_terms[:5]:
            if len(term) >= 6:  # Check specific medical terms
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, clinical_text_lower):
                    condition_key = (entry['condition'], entry['icd_code'])
                    if condition_key not in direct_matches:
                        direct_matches[condition_key] = {
                            'condition': entry['condition'],
                            'icd_code': entry['icd_code'],
                            'icd_description': entry['icd_description'],
                            'similarity_score': 0.90,
                            'match_type': 'confirmed',
                            'is_confirmed': True  # Still confirmed - specific subtype
                        }
                    break
    
    return list(direct_matches.values())


def suggest_icd_code(condition_match: Dict, clinical_text: str) -> Dict:
    """
    Enhanced context-aware ICD code suggestion based on clinical note content
    
    Uses condition-specific rules to select the most appropriate ICD code
    based on complications, severity, and specific terminology mentioned.
    
    Args:
        condition_match: Dictionary containing condition, icd_code, icd_description
        clinical_text: The clinical note text
        
    Returns:
        Dictionary with suggested_icd_code, icd_confidence, and alternative_icd_codes
    """
    import re
    
    condition_name = condition_match['condition']
    current_icd = condition_match['icd_code']
    clinical_text_lower = clinical_text.lower()
    
    # Get all ICD codes for this condition
    condition_icd_codes = [
        entry for entry in chronic_condition_embeddings 
        if entry['condition'] == condition_name
    ]
    
    if len(condition_icd_codes) <= 1:
        # Only one ICD code available, return it with high confidence
        return {
            'suggested_icd_code': current_icd,
            'icd_confidence': 0.95 if condition_match.get('is_confirmed', False) else 0.80,
            'alternative_icd_codes': []
        }
    
    # Context-based ICD selection rules - condition-specific patterns
    context_rules = {
        'Diabetes Mellitus Type 1': {
            'with coma|comatose': ['E10.0', 'E12.0'],
            'ketoacidosis|dka|diabetic ketoacidosis': ['E10.1', 'E12.1'],
            'renal|kidney|nephropathy': ['E10.2'],
            'ophthalmic|eye|retinopathy|vision|cataract': ['E10.3'],
            'neuropathy|nerve|polyneuropathy|mononeuropathy': ['E10.4'],
            'peripheral|circulatory|angiopathy': ['E10.5'],
            'multiple complication': ['E10.7'],
            'without complication|uncomplicated|unspecified': ['E10.9']
        },
        'Diabetes Mellitus Type 2': {
            'with coma|comatose': ['E11.0'],
            'ketoacidosis|dka': ['E11.1'],
            'renal|kidney|nephropathy': ['E11.2'],
            'ophthalmic|eye|retinopathy|vision|cataract': ['E11.3'],
            'neuropathy|nerve|polyneuropathy|mononeuropathy': ['E11.4'],
            'peripheral|circulatory|angiopathy': ['E11.5'],
            'multiple complication': ['E11.7'],
            'without complication|uncomplicated|unspecified': ['E11.9']
        },
        'Hypertension': {
            'heart failure|congestive': ['I11.0', 'I13.0', 'I13.2'],
            'renal|kidney|renal failure': ['I12.0', 'I13.1', 'I13.2'],
            'pregnancy|childbirth|gravid': ['O10.'],
            'secondary|endocrine': ['I15.'],
            'essential|primary': ['I10']
        },
        'Cardiac Failure': {
            'congestive|chf': ['I50.0'],
            'left ventricular|lvef|left heart': ['I50.1'],
            'hypertensive': ['I11.0', 'I13.0']
        },
        'Chronic Renal Disease': {
            'stage 5|esrd|end stage|end-stage': ['N18.0'],
            'stage 4': ['N18.4'],
            'stage 3': ['N18.3'],
            'stage 2': ['N18.2'],
            'stage 1': ['N18.1'],
            'hypertensive': ['I12.0', 'I13.1'],
            'glomerulo|glomerulonephritis': ['N03.'],
            'pyelonephritis|tubulo': ['N11.']
        },
        'Asthma': {
            'allergic|extrinsic|atopic': ['J45.0'],
            'nonallergic|intrinsic|non-allergic': ['J45.1'],
            'mixed': ['J45.8'],
            'status asthmaticus|severe|acute': ['J46'],
            'unspecified': ['J45.9']
        },
        'Chronic Obstructive Pulmonary Disease': {
            'acute|exacerbation|infection|acute lower respiratory': ['J44.0', 'J44.1'],
            'emphysema|panlobular|centrilobular': ['J43.'],
            'unspecified': ['J44.9']
        },
        'Epilepsy': {
            'focal|partial|localization': ['G40.0', 'G40.1', 'G40.2'],
            'generalized|idiopathic': ['G40.3', 'G40.4'],
            'grand mal|tonic.clonic': ['G40.6'],
            'status epilepticus': ['G41.'],
            'unspecified': ['G40.9']
        },
        'Hypothyroidism': {
            'congenital|neonatal': ['E03.0', 'E03.1'],
            'myxedema|myxoedema|coma': ['E03.5'],
            'drug|medication|medicament': ['E03.2'],
            'postprocedural|post.surgical': ['E89.0'],
            'iodine': ['E01.8', 'E02'],
            'unspecified': ['E03.9']
        },
        'Cardiomyopathy': {
            'ischaemic|ischemic|coronary': ['I25.5'],
            'dilated|dcm': ['I42.0'],
            'hypertrophic|hcm|obstructive': ['I42.1', 'I42.2'],
            'restrictive': ['I42.5'],
            'alcoholic|alcohol': ['I42.6'],
            'unspecified': ['I42.9']
        },
        'Hyperlipidaemia': {
            'pure.*cholesterol|hypercholesterol': ['E78.0'],
            'triglyceride|hyperglycerid': ['E78.1'],
            'mixed|combined': ['E78.2'],
            'unspecified': ['E78.5']
        },
        'Haemophilia': {
            'factor viii|factor 8|haemophilia a|hemophilia a': ['D66'],
            'factor ix|factor 9|christmas|haemophilia b|hemophilia b': ['D67']
        }
    }
    
    # Score each ICD code
    icd_scores = []
    suggested_icd = None
    confidence = 0.6
    
    for icd_entry in condition_icd_codes:
        icd_description_lower = icd_entry['icd_description'].lower()
        score = 0.0
        matched_terms = []
        
        # Check context rules for this condition
        if condition_name in context_rules:
            for context_pattern, preferred_icds in context_rules[condition_name].items():
                if re.search(context_pattern, clinical_text_lower):
                    # Check if this ICD code matches the preferred pattern
                    for pref in preferred_icds:
                        if icd_entry['icd_code'].startswith(pref.rstrip('.')):
                            score += 10.0  # High score for context match
                            matched_terms.append(f"context:{context_pattern}")
                            break
        
        # Extract significant medical terms from ICD description
        icd_terms = [
            word for word in re.findall(r'\b[a-z]{5,}\b', icd_description_lower)
            if word not in {'without', 'disease', 'syndrome', 'disorder', 'unspecified', 
                           'other', 'specified', 'mellitus', 'chronic', 'complication'}
        ]
        
        # Check which terms appear in clinical text
        for term in icd_terms:
            if term in clinical_text_lower:
                score += 1.0
                matched_terms.append(term)
        
        # Bonus for high-priority specific keywords
        specific_keywords = {
            'ketoacidosis': 3.0, 'coma': 3.0, 'stage 5': 3.0, 'stage 4': 2.5,
            'nephropathy': 2.0, 'neuropathy': 2.0, 'retinopathy': 2.0,
            'gangrene': 2.0, 'status asthmaticus': 2.5,
            'complications': 1.5, 'ulcer': 1.5,
            'hypertensive': 1.5, 'renal': 1.5, 'cardiac': 1.5
        }
        
        for keyword, bonus in specific_keywords.items():
            if keyword in icd_description_lower and keyword in clinical_text_lower:
                score += bonus
                if keyword not in matched_terms:
                    matched_terms.append(keyword)
        
        # Calculate confidence based on match strength
        conf = min(score / 10.0, 1.0) if score > 0 else 0.5
        
        icd_scores.append({
            'icd_code': icd_entry['icd_code'],
            'icd_description': icd_entry['icd_description'],
            'score': score,
            'confidence': conf,
            'matched_terms': matched_terms
        })
    
    # Sort by score
    icd_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # If top score is 0, look for "unspecified" ICD as fallback
    if icd_scores[0]['score'] == 0:
        for icd in condition_icd_codes:
            if 'unspecified' in icd['icd_description'].lower():
                suggested_icd = icd['icd_code']
                confidence = 0.65
                break
        
        if suggested_icd is None:
            suggested_icd = current_icd
            confidence = 0.60
        
        alternatives = [entry['icd_code'] for entry in icd_scores[:4] 
                       if entry['icd_code'] != suggested_icd]
        
        return {
            'suggested_icd_code': suggested_icd,
            'icd_confidence': confidence,
            'alternative_icd_codes': alternatives
        }
    
    # Return top suggestion with alternatives
    best_match = icd_scores[0]
    alternatives = [entry['icd_code'] for entry in icd_scores[1:5] 
                   if entry['icd_code'] != best_match['icd_code']]
    
    return {
        'suggested_icd_code': best_match['icd_code'],
        'icd_confidence': min(best_match['confidence'] * 0.95, 0.98),  # Cap at 0.98
        'alternative_icd_codes': alternatives
    }


def calculate_enhanced_confidence(match: Dict, clinical_text: str, keyword_matches: List[Dict]) -> float:
    """
    Calculate enhanced confidence score based on multiple factors
    
    Factors considered:
    1. Is it directly mentioned (confirmed)?
    2. Number of supporting keywords
    3. Presence of condition-specific measurements/tests
    4. Quality of keyword matches
    
    Args:
        match: Condition match dictionary
        clinical_text: Clinical note text
        keyword_matches: List of keyword matches for this condition
        
    Returns:
        Enhanced confidence score (0.0 to 0.98)
    """
    base_score = match.get('similarity_score', 0.7)
    
    # Factor 1: Is it directly mentioned? (highest confidence)
    if match.get('is_confirmed', False):
        confidence = 0.95
    else:
        confidence = base_score
    
    # Factor 2: Number of supporting keywords
    keyword_count = len(keyword_matches)
    if keyword_count >= 5:
        confidence *= 1.10  # 10% boost for many supporting keywords
    elif keyword_count >= 3:
        confidence *= 1.05  # 5% boost for multiple keywords
    elif keyword_count == 1:
        confidence *= 0.95  # Slight penalty for single keyword match
    
    # Factor 3: Presence of specific measurements/tests
    condition = match['condition']
    clinical_lower = clinical_text.lower()
    
    measurement_indicators = {
        'Diabetes Mellitus Type 1': ['hba1c', 'blood glucose', 'insulin', 'glucose level', 'a1c'],
        'Diabetes Mellitus Type 2': ['hba1c', 'blood glucose', 'metformin', 'glucose level', 'a1c'],
        'Hypertension': ['bp', 'blood pressure', 'systolic', 'diastolic', 'mmhg'],
        'Cardiac Failure': ['ejection fraction', 'bnp', 'echocardiogram', 'echo', 'lvef'],
        'Chronic Renal Disease': ['creatinine', 'egfr', 'proteinuria', 'gfr', 'urea'],
        'Hypothyroidism': ['tsh', 't4', 'thyroid', 't3', 'thyroid function'],
        'Asthma': ['peak flow', 'spirometry', 'fev1', 'pefr'],
        'Chronic Obstructive Pulmonary Disease': ['spirometry', 'fev1', 'oxygen', 'spo2', 'o2'],
        'Hyperlipidaemia': ['cholesterol', 'ldl', 'hdl', 'lipid panel', 'triglyceride'],
        'Cardiomyopathy': ['ejection fraction', 'echo', 'lvef', 'echocardiogram'],
        'Epilepsy': ['eeg', 'electroencephalogram', 'seizure frequency'],
        'Haemophilia': ['factor level', 'factor viii', 'factor ix', 'aptt', 'ptt']
    }
    
    if condition in measurement_indicators:
        measurement_found = any(
            indicator in clinical_lower 
            for indicator in measurement_indicators[condition]
        )
        if measurement_found:
            confidence *= 1.15  # 15% boost for objective measurements
    
    # Factor 4: Quality of keyword matches (average similarity)
    if keyword_matches and not match.get('is_confirmed', False):
        avg_keyword_score = sum(kw.get('similarity_score', 0) for kw in keyword_matches) / len(keyword_matches)
        if avg_keyword_score >= 0.85:
            confidence *= 1.08  # High quality matches
        elif avg_keyword_score < 0.70:
            confidence *= 0.92  # Lower quality matches
    
    # Factor 5: Check for symptom indicators
    symptom_indicators = get_condition_symptom_indicators()
    if condition in symptom_indicators:
        symptom_count = sum(
            1 for symptom in symptom_indicators[condition]
            if symptom.lower() in clinical_lower
        )
        if symptom_count >= 3:
            confidence *= 1.12  # Multiple symptoms present
        elif symptom_count >= 2:
            confidence *= 1.06  # Some symptoms present
    
    # Cap at 0.98 (never 100% certain without human review)
    return min(confidence, 0.98)


def validate_keyword_quality(keyword_matches: List[Dict], condition_name: str, clinical_text: str) -> bool:
    """
    Validate that a condition has sufficient high-quality keyword matches.
    
    Requirements:
    1. At least 3 distinct clinical keywords
    2. Keywords must be condition-specific, not generic terms
    3. Average keyword similarity should be reasonable (>0.70)
    
    Args:
        keyword_matches: List of keyword matches for this condition
        condition_name: Name of the condition being validated
        clinical_text: Original clinical note text
        
    Returns:
        True if keyword quality is sufficient, False otherwise
    """
    if not keyword_matches or len(keyword_matches) < 3:
        return False
    
    # Generic medical terms that don't indicate specific conditions
    generic_terms = {
        'patient', 'diagnosis', 'diagnosed', 'medication', 'treatment', 
        'disease', 'disorder', 'condition', 'symptoms', 'history',
        'medical', 'clinical', 'therapy', 'medicine', 'drug',
        'present', 'presents', 'reported', 'reports', 'noted',
        'complaint', 'episode', 'episodes', 'current', 'recent',
        'direct_mention'  # This is okay for confirmed conditions
    }
    
    # Get condition-specific indicators
    symptom_indicators = get_condition_symptom_indicators()
    condition_specific_terms = set()
    if condition_name in symptom_indicators:
        condition_specific_terms = {term.lower() for term in symptom_indicators[condition_name]}
    
    # Count valid (non-generic) keywords
    valid_keywords = []
    clinical_lower = clinical_text.lower()
    
    for kw in keyword_matches:
        keyword = kw['keyword'].lower()
        
        # Direct mention is always valid for confirmed conditions
        if keyword == 'direct_mention':
            valid_keywords.append(kw)
            continue
        
        # Skip generic terms unless they're condition-specific
        if keyword in generic_terms and keyword not in condition_specific_terms:
            continue
        
        # Check if keyword actually appears in clinical text (validates it's real)
        if len(keyword) >= 4 and keyword in clinical_lower:
            valid_keywords.append(kw)
        # Or if it's a known condition-specific term
        elif keyword in condition_specific_terms:
            valid_keywords.append(kw)
    
    # Require at least 3 valid keywords
    if len(valid_keywords) < 3:
        return False
    
    # Check average similarity of valid keywords
    avg_similarity = sum(kw['similarity_score'] for kw in valid_keywords) / len(valid_keywords)
    if avg_similarity < 0.70:
        return False
    
    return True


def match_conditions(clinical_keywords, clinical_keyword_embeddings, clinical_text="", threshold=0.65):
    """
    Authi 1.0 - Condition Matching Component
    
    ENHANCED ACCURACY LOGIC:
    - If conditions are explicitly mentioned in the note (CONFIRMED), prioritize those
    - Only suggest additional conditions if they are RELATED to confirmed conditions
    - If no confirmed conditions found, use semantic matching to suggest possible conditions
    - STRICT FILTERING: High confidence matches (>95%) require others to be >90%
    - ADAPTIVE THRESHOLD: Threshold increases based on top match confidence
    - MINIMUM KEYWORDS: Requires 3+ distinct clinical keywords per condition
    - Returns 1-5 conditions based on clinical evidence strength
    - Tracks triggering keywords for transparency
    
    Responsible for:
    - Mapping extracted keywords to chronic condition entries
    - Returning clinically accurate condition suggestions (not just similar words)
    - Marking confirmed vs suggested conditions
    - Tracking which keywords triggered each match
    """
    # First, check for direct condition name matches (CONFIRMED conditions)
    direct_matches = find_direct_condition_matches(clinical_text) if clinical_text else []
    
    # Use condition NAME only as key to avoid duplicate conditions with different ICD codes
    confirmed_conditions = {}
    suggested_conditions = {}
    condition_scores = {}
    condition_keyword_matches = {}  # Track which keywords matched which conditions
    
    # Add confirmed matches (explicitly mentioned in note)
    for match in direct_matches:
        condition_name = match['condition']
        if condition_name not in confirmed_conditions or match['similarity_score'] > confirmed_conditions[condition_name]['similarity_score']:
            confirmed_conditions[condition_name] = match
            # For confirmed matches, the triggering "keyword" is the direct mention
            condition_keyword_matches[condition_name] = [
                {'keyword': 'direct_mention', 'similarity_score': match['similarity_score']}
            ]
            print(f"   ✓ CONFIRMED condition found: {match['condition']}")
    
    # Get the count of unique confirmed conditions
    confirmed_count = len(confirmed_conditions)
    
    # If we have confirmed conditions, we should primarily return those
    # Only use semantic matching to potentially find related/comorbid conditions
    if confirmed_count > 0:
        print(f"   Found {confirmed_count} confirmed condition(s). Limiting semantic suggestions.")
        
        # Define related condition pairs (comorbidities often found together)
        # Updated with more medically accurate relationships based on clinical evidence
        related_conditions = {
            'Hypertension': [
                'Cardiac Failure', 'Chronic Renal Disease', 'Cardiomyopathy', 
                'Diabetes Mellitus Type 2', 'Hypothyroidism', 'Hyperlipidaemia'
            ],
            'Diabetes Mellitus Type 1': [
                'Chronic Renal Disease', 'Hypertension', 'Hyperlipidaemia', 
                'Hypothyroidism', 'Cardiac Failure'
            ],
            'Diabetes Mellitus Type 2': [
                'Hypertension', 'Hyperlipidaemia', 'Chronic Renal Disease', 
                'Cardiac Failure', 'Hypothyroidism', 'Chronic Obstructive Pulmonary Disease'
            ],
            'Cardiac Failure': [
                'Hypertension', 'Cardiomyopathy', 'Chronic Renal Disease', 
                'Hypothyroidism', 'Diabetes Mellitus Type 2', 'Hyperlipidaemia',
                'Chronic Obstructive Pulmonary Disease'
            ],
            'Cardiomyopathy': [
                'Cardiac Failure', 'Hypertension', 'Diabetes Mellitus Type 2',
                'Hyperlipidaemia'
            ],
            'Chronic Renal Disease': [
                'Hypertension', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 
                'Cardiac Failure', 'Hyperlipidaemia', 'Hypothyroidism'
            ],
            'Hyperlipidaemia': [
                'Hypertension', 'Diabetes Mellitus Type 2', 'Hypothyroidism',
                'Cardiac Failure', 'Cardiomyopathy', 'Chronic Renal Disease'
            ],
            'Asthma': [
                'Chronic Obstructive Pulmonary Disease', 'Hyperlipidaemia'  # Overlap syndrome
            ],
            'Haemophilia': [],  # Typically standalone condition
            'Chronic Obstructive Pulmonary Disease': [
                'Cardiac Failure', 'Hypertension', 'Diabetes Mellitus Type 2',
                'Asthma', 'Hyperlipidaemia'  # COPD often found with cardiovascular and metabolic issues
            ],
            'Epilepsy': [
                'Hypothyroidism'  # Thyroid disorders can affect seizure control
            ],
            'Hypothyroidism': [
                'Hypertension', 'Diabetes Mellitus Type 2', 'Hyperlipidaemia',
                'Cardiac Failure', 'Chronic Renal Disease'  # Common metabolic associations
            ]
        }
        
        # Get related conditions for the confirmed ones
        allowed_suggestions = set()
        for confirmed_name in confirmed_conditions.keys():
            if confirmed_name in related_conditions:
                allowed_suggestions.update(related_conditions[confirmed_name])
        
        # Remove already confirmed conditions from suggestions
        allowed_suggestions -= set(confirmed_conditions.keys())
        
        # Only look for semantic matches that are related to confirmed conditions
        # AND only if we need more conditions to reach 3-5
        if confirmed_count < 5 and len(allowed_suggestions) > 0:
            for i, keyword_embedding in enumerate(clinical_keyword_embeddings):
                current_keyword = clinical_keywords[i] if i < len(clinical_keywords) else f"keyword_{i}"
                
                for condition_data in chronic_condition_embeddings:
                    # Only consider related conditions
                    if condition_data['condition'] not in allowed_suggestions:
                        continue
                    
                    condition_embedding = condition_data['embedding']
                    if condition_embedding is None:
                        continue
                    
                    similarity = calculate_cosine_similarity(keyword_embedding, condition_embedding)
                    
                    # Higher threshold for suggested conditions when we have confirmed ones
                    if similarity >= 0.75:  # Stricter threshold
                        condition_name = condition_data['condition']
                        
                        if condition_name not in condition_scores:
                            condition_scores[condition_name] = []
                        condition_scores[condition_name].append(similarity.item())
                        
                        # Track keyword matches
                        if condition_name not in condition_keyword_matches:
                            condition_keyword_matches[condition_name] = []
                        condition_keyword_matches[condition_name].append({
                            'keyword': current_keyword,
                            'similarity_score': similarity.item()
                        })
                        
                        if condition_name not in suggested_conditions or similarity.item() > suggested_conditions[condition_name]['similarity_score']:
                            suggested_conditions[condition_name] = {
                                'condition': condition_data['condition'],
                                'icd_code': condition_data['icd_code'],
                                'icd_description': condition_data['icd_description'],
                                'similarity_score': similarity.item(),
                                'match_type': 'suggested',
                                'is_confirmed': False
                            }
    else:
        # No confirmed conditions - use semantic matching to suggest possible conditions
        print(f"   No confirmed conditions found. Using semantic analysis to suggest conditions.")
        
        for i, keyword_embedding in enumerate(clinical_keyword_embeddings):
            current_keyword = clinical_keywords[i] if i < len(clinical_keywords) else f"keyword_{i}"
            best_match = None
            highest_similarity = -1.0
            
            for condition_data in chronic_condition_embeddings:
                condition_embedding = condition_data['embedding']
                if condition_embedding is None:
                    continue
                
                similarity = calculate_cosine_similarity(keyword_embedding, condition_embedding)
                
                if similarity >= threshold:
                    condition_name = condition_data['condition']
                    
                    if condition_name not in condition_scores:
                        condition_scores[condition_name] = []
                    condition_scores[condition_name].append(similarity.item())
                    
                    # Track keyword matches
                    if condition_name not in condition_keyword_matches:
                        condition_keyword_matches[condition_name] = []
                    condition_keyword_matches[condition_name].append({
                        'keyword': current_keyword,
                        'similarity_score': similarity.item()
                    })
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = {
                            'condition': condition_data['condition'],
                            'icd_code': condition_data['icd_code'],
                            'icd_description': condition_data['icd_description'],
                            'similarity_score': highest_similarity.item(),
                            'match_type': 'suggested',
                            'is_confirmed': False
                        }
            
            if best_match:
                condition_name = best_match['condition']
                if condition_name not in suggested_conditions or best_match['similarity_score'] > suggested_conditions[condition_name]['similarity_score']:
                    suggested_conditions[condition_name] = best_match
    
    # Calculate average score for suggested conditions to improve ranking
    for condition_name, match in suggested_conditions.items():
        if condition_name in condition_scores:
            avg_score = sum(condition_scores[condition_name]) / len(condition_scores[condition_name])
            match['similarity_score'] = (match['similarity_score'] * 0.7) + (avg_score * 0.3)
    
    # Build result list: confirmed conditions first, then suggestions
    result_list = list(confirmed_conditions.values())
    
    # Sort suggestions by score and add them after confirmed conditions
    sorted_suggestions = sorted(suggested_conditions.values(), key=lambda x: x['similarity_score'], reverse=True)
    
    # Add suggestions only if we have room (max 5 total) and they're strong matches
    remaining_slots = 5 - len(result_list)
    if remaining_slots > 0:
        for suggestion in sorted_suggestions[:remaining_slots]:
            # Only add suggestions with reasonably high scores
            if suggestion['similarity_score'] >= 0.70:
                result_list.append(suggestion)
                print(f"   → Suggested related condition: {suggestion['condition']} (score: {suggestion['similarity_score']:.3f})")
    
    # Attach triggering keywords to each condition
    for condition_match in result_list:
        condition_name = condition_match['condition']
        if condition_name in condition_keyword_matches:
            # Get top 5 keywords sorted by similarity
            keyword_matches = sorted(
                condition_keyword_matches[condition_name],
                key=lambda x: x['similarity_score'],
                reverse=True
            )[:5]
            condition_match['triggering_keywords'] = keyword_matches
        else:
            condition_match['triggering_keywords'] = []
        
        # Add match explanation
        if condition_match.get('is_confirmed', False):
            condition_match['match_explanation'] = "Direct mention in clinical note"
        else:
            condition_match['match_explanation'] = "Semantic match based on clinical terminology"
    
    # Add ICD code suggestions for each condition
    for condition_match in result_list:
        icd_suggestion = suggest_icd_code(condition_match, clinical_text)
        condition_match.update(icd_suggestion)
    
    # Apply enhanced confidence scoring to refine similarity scores
    for condition_match in result_list:
        condition_name = condition_match['condition']
        keyword_matches = condition_match.get('triggering_keywords', [])
        
        # Calculate enhanced confidence
        enhanced_score = calculate_enhanced_confidence(
            condition_match, 
            clinical_text, 
            keyword_matches
        )
        
        # Update the similarity score with enhanced confidence
        condition_match['similarity_score'] = enhanced_score
        
        # Also update match explanation to reflect confidence factors
        if condition_match.get('is_confirmed', False):
            condition_match['match_explanation'] = "Direct mention in clinical note (high confidence)"
        else:
            factors = []
            if len(keyword_matches) >= 3:
                factors.append("multiple keyword matches")
            
            # Check for measurements
            condition = condition_match['condition']
            clinical_lower = clinical_text.lower()
            measurement_indicators = {
                'Diabetes Mellitus Type 1': ['hba1c', 'blood glucose', 'insulin'],
                'Diabetes Mellitus Type 2': ['hba1c', 'blood glucose', 'metformin'],
                'Hypertension': ['bp', 'blood pressure', 'systolic', 'diastolic'],
                'Cardiac Failure': ['ejection fraction', 'bnp', 'echocardiogram'],
                'Chronic Renal Disease': ['creatinine', 'egfr', 'proteinuria'],
                'Hypothyroidism': ['tsh', 't4', 'thyroid'],
                'Asthma': ['peak flow', 'spirometry', 'fev1'],
                'Chronic Obstructive Pulmonary Disease': ['spirometry', 'fev1', 'oxygen'],
                'Hyperlipidaemia': ['cholesterol', 'ldl', 'hdl'],
                'Cardiomyopathy': ['ejection fraction', 'echo', 'lvef'],
                'Epilepsy': ['eeg', 'electroencephalogram'],
                'Haemophilia': ['factor level', 'factor viii', 'factor ix']
            }
            if condition in measurement_indicators:
                if any(ind in clinical_lower for ind in measurement_indicators[condition]):
                    factors.append("objective measurements present")
            
            if factors:
                condition_match['match_explanation'] = f"Semantic match based on clinical terminology ({', '.join(factors)})"
            else:
                condition_match['match_explanation'] = "Semantic match based on clinical terminology"
    
    # Sort final result by enhanced score
    result_list.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # ============================================================================
    # ENHANCED ACCURACY FILTERING
    # ============================================================================
    
    # Step 1: Validate keyword quality for each condition AND check for negations
    validated_conditions = []
    for condition_match in result_list:
        condition_name = condition_match['condition']
        keyword_matches = condition_match.get('triggering_keywords', [])
        
        # Confirmed conditions (direct mention) always pass validation
        if condition_match.get('is_confirmed', False):
            validated_conditions.append(condition_match)
            continue
        
        # Check for negation even in suggested conditions
        is_negated, negation_context = detect_negation_context(clinical_text, condition_name)
        if is_negated:
            print(f"   ⚠ Filtered out {condition_name}: condition is negated in note ('{negation_context}')")
            continue
        
        # Suggested conditions must have quality keywords
        if validate_keyword_quality(keyword_matches, condition_name, clinical_text):
            validated_conditions.append(condition_match)
        else:
            print(f"   ⚠ Filtered out {condition_name}: insufficient keyword evidence (only {len(keyword_matches)} keywords)")
    
    if not validated_conditions:
        print(f"   ⚠ No conditions passed keyword validation")
        return []
    
    # Step 2: Apply strict filtering based on top match confidence
    top_score = validated_conditions[0]['similarity_score']
    filtered_conditions = [validated_conditions[0]]  # Always keep top match
    
    # Strict filtering thresholds
    if top_score >= 0.95:
        # Very high confidence top match - only keep others if they're also very high (>0.90)
        min_secondary_score = 0.90
        print(f"   🎯 High confidence match ({top_score:.3f}) - applying strict filter (≥{min_secondary_score})")
    elif top_score >= 0.85:
        # High confidence - require others to be strong (>0.80)
        min_secondary_score = 0.80
        print(f"   🎯 Strong match ({top_score:.3f}) - applying moderate filter (≥{min_secondary_score})")
    elif top_score >= 0.75:
        # Moderate confidence - allow reasonable matches (>0.70)
        min_secondary_score = 0.70
    else:
        # Lower confidence - use adaptive threshold based on gap
        min_secondary_score = max(0.65, top_score - 0.10)
    
    # Add secondary matches that meet the threshold
    for condition in validated_conditions[1:]:
        if condition['similarity_score'] >= min_secondary_score:
            filtered_conditions.append(condition)
        else:
            print(f"   ⚠ Filtered out {condition['condition']}: score {condition['similarity_score']:.3f} below threshold {min_secondary_score:.3f}")
    
    # Step 3: Apply adaptive result count
    # If top match is very strong (>0.95), prefer returning fewer conditions
    if top_score >= 0.95 and len(filtered_conditions) == 1:
        print(f"   ✓ Returning single high-confidence match")
        return filtered_conditions[:1]
    
    # Step 4: Check for related conditions (comorbidities)
    # If we have multiple conditions, verify they make clinical sense together
    if len(filtered_conditions) > 1:
        related_conditions = {
            'Hypertension': [
                'Cardiac Failure', 'Chronic Renal Disease', 'Cardiomyopathy', 
                'Diabetes Mellitus Type 2', 'Hypothyroidism', 'Hyperlipidaemia'
            ],
            'Diabetes Mellitus Type 1': [
                'Chronic Renal Disease', 'Hypertension', 'Hyperlipidaemia', 
                'Hypothyroidism', 'Cardiac Failure'
            ],
            'Diabetes Mellitus Type 2': [
                'Hypertension', 'Hyperlipidaemia', 'Chronic Renal Disease', 
                'Cardiac Failure', 'Hypothyroidism', 'Chronic Obstructive Pulmonary Disease'
            ],
            'Cardiac Failure': [
                'Hypertension', 'Cardiomyopathy', 'Chronic Renal Disease', 
                'Hypothyroidism', 'Diabetes Mellitus Type 2', 'Hyperlipidaemia',
                'Chronic Obstructive Pulmonary Disease'
            ],
            'Cardiomyopathy': [
                'Cardiac Failure', 'Hypertension', 'Diabetes Mellitus Type 2',
                'Hyperlipidaemia'
            ],
            'Chronic Renal Disease': [
                'Hypertension', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 
                'Cardiac Failure', 'Hyperlipidaemia', 'Hypothyroidism'
            ],
            'Hyperlipidaemia': [
                'Hypertension', 'Diabetes Mellitus Type 2', 'Hypothyroidism',
                'Cardiac Failure', 'Cardiomyopathy', 'Chronic Renal Disease'
            ],
            'Asthma': [
                'Chronic Obstructive Pulmonary Disease', 'Hyperlipidaemia'
            ],
            'Chronic Obstructive Pulmonary Disease': [
                'Cardiac Failure', 'Hypertension', 'Diabetes Mellitus Type 2',
                'Asthma', 'Hyperlipidaemia'
            ],
            'Epilepsy': [
                'Hypothyroidism'
            ],
            'Hypothyroidism': [
                'Hypertension', 'Diabetes Mellitus Type 2', 'Hyperlipidaemia',
                'Cardiac Failure', 'Chronic Renal Disease'
            ],
            'Haemophilia': []  # Typically standalone
        }
        
        # Keep top match and related conditions
        top_condition_name = filtered_conditions[0]['condition']
        clinically_valid = [filtered_conditions[0]]
        
        for condition in filtered_conditions[1:]:
            condition_name = condition['condition']
            
            # Check if this condition is related to the top match
            if top_condition_name in related_conditions:
                if condition_name in related_conditions[top_condition_name]:
                    clinically_valid.append(condition)
                else:
                    print(f"   ⚠ Filtered out {condition_name}: not clinically related to {top_condition_name}")
            else:
                # If top condition has no common comorbidities, be strict
                if condition['similarity_score'] >= 0.90:
                    clinically_valid.append(condition)
                else:
                    print(f"   ⚠ Filtered out {condition_name}: no clinical relationship established")
        
        filtered_conditions = clinically_valid
    
    # Return validated, filtered conditions (1-5 based on evidence)
    final_count = min(len(filtered_conditions), 5)
    print(f"   ✓ Returning {final_count} condition(s) after comprehensive filtering")
    
    return filtered_conditions[:final_count]


@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup"""
    load_model()
    load_chronic_conditions()


@app.get("/")
async def root():
    import datetime
    return {
        "message": "SaluLink Authi API is running",
        "version": "v2.1-accuracy-improved",
        "loaded_at": datetime.datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "conditions_loaded": len(chronic_condition_embeddings) > 0
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_clinical_note(request: AnalysisRequest):
    """
    Analyze a clinical note and return 3-5 matched chronic conditions
    
    This endpoint implements the complete Authi 1.0 workflow:
    1. ClinicalBERT extracts keywords and clinical terminology from the note
    2. Authi 1.0 matches keywords to chronic conditions
    3. CONFIRMED conditions (explicitly mentioned) are prioritized
    4. Only related conditions are suggested when confirmed conditions exist
    5. Returns 3-5 conditions with is_confirmed flag and similarity scores
    
    Args:
        request: AnalysisRequest containing the clinical note text
        
    Returns:
        AnalysisResponse with extracted keywords, matched conditions, and confirmed count
    """
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate note completeness first
        note_quality = validate_note_completeness(request.clinical_note)
        
        # Extract keywords
        keywords, embeddings = extract_keywords_clinicalbert(request.clinical_note)
        
        if embeddings.nelement() == 0:
            return AnalysisResponse(
                extracted_keywords=[],
                matched_conditions=[],
                confirmed_count=0,
                note_quality=NoteQualityScore(
                    completeness_score=note_quality['completeness_score'],
                    missing_elements=note_quality['missing_elements'],
                    warnings=note_quality['warnings']
                )
            )
        
        # Match conditions using Authi 1.0 algorithm
        # Returns 3-5 conditions with confirmed conditions prioritized
        matches = match_conditions(keywords, embeddings, request.clinical_note)
        
        # Count confirmed conditions
        confirmed_count = sum(1 for m in matches if m.get('is_confirmed', False))
        
        # Log results for monitoring
        print(f"\n{'='*60}")
        print(f"Note Quality Score: {note_quality['completeness_score']}/100")
        print(f"Analysis completed: {len(keywords)} keywords extracted")
        print(f"Conditions found: {len(matches)} total, {confirmed_count} CONFIRMED")
        print(f"{'='*60}")
        for i, match in enumerate(matches, 1):
            status = "✓ CONFIRMED" if match.get('is_confirmed', False) else "→ Suggested"
            suggested_icd = f" [Suggested: {match.get('suggested_icd_code', 'N/A')}]" if match.get('suggested_icd_code') else ""
            print(f"  {i}. [{status}] {match['condition']} ({match['icd_code']}) - Score: {match['similarity_score']:.3f}{suggested_icd}")
        print(f"{'='*60}\n")
        
        return AnalysisResponse(
            extracted_keywords=keywords[:30],  # Return up to 30 most relevant keywords
            matched_conditions=[
                MatchedConditionResponse(
                    condition=m['condition'],
                    icd_code=m['icd_code'],
                    icd_description=m['icd_description'],
                    similarity_score=m['similarity_score'],
                    is_confirmed=m.get('is_confirmed', False),
                    triggering_keywords=[
                        KeywordMatch(
                            keyword=kw['keyword'],
                            similarity_score=kw['similarity_score']
                        ) for kw in m.get('triggering_keywords', [])
                    ],
                    match_explanation=m.get('match_explanation', ''),
                    suggested_icd_code=m.get('suggested_icd_code'),
                    icd_confidence=m.get('icd_confidence'),
                    alternative_icd_codes=m.get('alternative_icd_codes', [])
                )
                for m in matches
            ],
            confirmed_count=confirmed_count,
            note_quality=NoteQualityScore(
                completeness_score=note_quality['completeness_score'],
                missing_elements=note_quality['missing_elements'],
                warnings=note_quality['warnings']
            )
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

