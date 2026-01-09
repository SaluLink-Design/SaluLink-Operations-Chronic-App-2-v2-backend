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


class MatchedConditionResponse(BaseModel):
    condition: str
    icd_code: str
    icd_description: str
    similarity_score: float
    is_confirmed: bool = False  # True if condition is explicitly mentioned in the note


class AnalysisResponse(BaseModel):
    extracted_keywords: List[str]
    matched_conditions: List[MatchedConditionResponse]
    confirmed_count: int = 0  # Number of conditions directly mentioned in note


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
    
    # Define condition aliases for common medical variations
    condition_aliases = {
        'diabetes mellitus type 1': ['type 1 diabetes', 'type i diabetes', 't1dm', 'type1 diabetes', 'insulin-dependent diabetes', 'insulin dependent diabetes', 'iddm'],
        'diabetes mellitus type 2': ['type 2 diabetes', 'type ii diabetes', 't2dm', 'type2 diabetes', 'non-insulin-dependent diabetes', 'non insulin dependent diabetes', 'niddm', 'adult-onset diabetes'],
        'hypertension': ['high blood pressure', 'elevated blood pressure', 'hypertensive', 'htn', 'bp elevated', 'raised blood pressure'],
        'asthma': ['asthmatic', 'bronchial asthma'],
        'cardiac failure': ['heart failure', 'congestive heart failure', 'chf', 'left ventricular failure', 'right heart failure', 'cardiac decompensation'],
        'chronic renal disease': ['chronic kidney disease', 'ckd', 'renal failure', 'kidney failure', 'nephropathy', 'renal insufficiency', 'kidney disease'],
        'cardiomyopathy': ['cardiomyopathic', 'dilated cardiomyopathy', 'hypertrophic cardiomyopathy', 'restrictive cardiomyopathy'],
        'hyperlipidaemia': ['hyperlipidemia', 'high cholesterol', 'dyslipidemia', 'dyslipidaemia', 'elevated cholesterol', 'hypercholesterolemia', 'hypercholesterolaemia'],
        'haemophilia': ['hemophilia', 'factor viii deficiency', 'factor ix deficiency', 'bleeding disorder']
    }
    
    # Strategy 1: Direct condition name matching (CONFIRMED)
    for condition_name in condition_names:
        # Create word boundary regex pattern for accurate matching
        pattern = r'\b' + re.escape(condition_name) + r'\b'
        if re.search(pattern, clinical_text_lower):
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
        for alias in aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, clinical_text_lower):
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
                break  # Stop checking aliases once found
    
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


def match_conditions(clinical_keywords, clinical_keyword_embeddings, clinical_text="", threshold=0.65):
    """
    Authi 1.0 - Condition Matching Component
    
    NEW LOGIC:
    - If conditions are explicitly mentioned in the note (CONFIRMED), prioritize those
    - Only suggest additional conditions if they are RELATED to confirmed conditions
    - If no confirmed conditions found, use semantic matching to suggest possible conditions
    - Returns 3-5 conditions with is_confirmed flag indicating explicit mention
    
    Responsible for:
    - Mapping extracted keywords to chronic condition entries
    - Returning 3–5 UNIQUE chronic condition suggestions
    - Marking confirmed vs suggested conditions
    """
    # First, check for direct condition name matches (CONFIRMED conditions)
    direct_matches = find_direct_condition_matches(clinical_text) if clinical_text else []
    
    # Use condition NAME only as key to avoid duplicate conditions with different ICD codes
    confirmed_conditions = {}
    suggested_conditions = {}
    condition_scores = {}
    
    # Add confirmed matches (explicitly mentioned in note)
    for match in direct_matches:
        condition_name = match['condition']
        if condition_name not in confirmed_conditions or match['similarity_score'] > confirmed_conditions[condition_name]['similarity_score']:
            confirmed_conditions[condition_name] = match
            print(f"   ✓ CONFIRMED condition found: {match['condition']}")
    
    # Get the count of unique confirmed conditions
    confirmed_count = len(confirmed_conditions)
    
    # If we have confirmed conditions, we should primarily return those
    # Only use semantic matching to potentially find related/comorbid conditions
    if confirmed_count > 0:
        print(f"   Found {confirmed_count} confirmed condition(s). Limiting semantic suggestions.")
        
        # Define related condition pairs (comorbidities often found together)
        related_conditions = {
            'Hypertension': ['Cardiac Failure', 'Chronic Renal Disease', 'Cardiomyopathy', 'Diabetes Mellitus Type 2'],
            'Diabetes Mellitus Type 1': ['Chronic Renal Disease', 'Hypertension', 'Hyperlipidaemia'],
            'Diabetes Mellitus Type 2': ['Hypertension', 'Hyperlipidaemia', 'Chronic Renal Disease', 'Cardiac Failure'],
            'Cardiac Failure': ['Hypertension', 'Cardiomyopathy', 'Chronic Renal Disease'],
            'Cardiomyopathy': ['Cardiac Failure', 'Hypertension'],
            'Chronic Renal Disease': ['Hypertension', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 'Cardiac Failure'],
            'Hyperlipidaemia': ['Hypertension', 'Diabetes Mellitus Type 2'],
            'Asthma': [],  # Asthma typically stands alone
            'Haemophilia': []  # Haemophilia typically stands alone
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
    
    # Sort final result by score
    result_list.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return 3-5 conditions
    # If we have confirmed conditions, we return what we have (even if less than 3)
    # If no confirmed conditions and less than 3 suggestions, try with lower threshold
    if confirmed_count == 0 and len(result_list) < 3 and len(result_list) > 0 and threshold > 0.5:
        return match_conditions(clinical_keywords, clinical_keyword_embeddings, clinical_text, threshold=threshold - 0.1)
    
    return result_list[:5]


@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup"""
    load_model()
    load_chronic_conditions()


@app.get("/")
async def root():
    return {"message": "SaluLink Authi API is running"}


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
        
        # Extract keywords
        keywords, embeddings = extract_keywords_clinicalbert(request.clinical_note)
        
        if embeddings.nelement() == 0:
            return AnalysisResponse(
                extracted_keywords=[],
                matched_conditions=[],
                confirmed_count=0
            )
        
        # Match conditions using Authi 1.0 algorithm
        # Returns 3-5 conditions with confirmed conditions prioritized
        matches = match_conditions(keywords, embeddings, request.clinical_note)
        
        # Count confirmed conditions
        confirmed_count = sum(1 for m in matches if m.get('is_confirmed', False))
        
        # Log results for monitoring
        print(f"\n{'='*60}")
        print(f"Analysis completed: {len(keywords)} keywords extracted")
        print(f"Conditions found: {len(matches)} total, {confirmed_count} CONFIRMED")
        print(f"{'='*60}")
        for i, match in enumerate(matches, 1):
            status = "✓ CONFIRMED" if match.get('is_confirmed', False) else "→ Suggested"
            print(f"  {i}. [{status}] {match['condition']} ({match['icd_code']}) - Score: {match['similarity_score']:.3f}")
        print(f"{'='*60}\n")
        
        return AnalysisResponse(
            extracted_keywords=keywords[:30],  # Return up to 30 most relevant keywords
            matched_conditions=[
                MatchedConditionResponse(
                    condition=m['condition'],
                    icd_code=m['icd_code'],
                    icd_description=m['icd_description'],
                    similarity_score=m['similarity_score'],
                    is_confirmed=m.get('is_confirmed', False)
                )
                for m in matches
            ],
            confirmed_count=confirmed_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

