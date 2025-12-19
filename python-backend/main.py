"""
SaluLink Chronic App - Python Backend API
Handles ClinicalBERT analysis using the Authi 1.0 logic
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


class AnalysisResponse(BaseModel):
    extracted_keywords: List[str]
    matched_conditions: List[MatchedConditionResponse]


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
    Processes clinical text and extracts keywords with embeddings
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
    
    # Process tokens
    for i, token in enumerate(tokens):
        # Skip special tokens
        if token in tokenizer.all_special_tokens:
            if current_word:
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
            if current_word:
                avg_embedding = torch.mean(last_hidden_state[0, current_embedding_indices, :], dim=0)
                extracted_keywords.append(current_word)
                keyword_embeddings.append(avg_embedding)
            current_word = token
            current_embedding_indices = [i]
    
    # Add last word
    if current_word:
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


def match_conditions(clinical_keywords, clinical_keyword_embeddings, threshold=0.7):
    """
    Match clinical keywords to chronic conditions
    """
    matched_conditions = {}
    
    for i, keyword_embedding in enumerate(clinical_keyword_embeddings):
        best_match = None
        highest_similarity = -1.0
        
        for condition_data in chronic_condition_embeddings:
            condition_embedding = condition_data['embedding']
            if condition_embedding is None:
                continue
            
            similarity = calculate_cosine_similarity(keyword_embedding, condition_embedding)
            
            if similarity > highest_similarity and similarity >= threshold:
                highest_similarity = similarity
                best_match = {
                    'condition': condition_data['condition'],
                    'icd_code': condition_data['icd_code'],
                    'icd_description': condition_data['icd_description'],
                    'similarity_score': highest_similarity.item()
                }
        
        if best_match:
            condition_key = (best_match['condition'], best_match['icd_code'])
            if condition_key not in matched_conditions or \
               best_match['similarity_score'] > matched_conditions[condition_key]['similarity_score']:
                matched_conditions[condition_key] = best_match
    
    result_list = list(matched_conditions.values())
    result_list.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return result_list[:5]  # Return top 5 matches


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
    Analyze a clinical note and return matched conditions
    """
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Extract keywords
        keywords, embeddings = extract_keywords_clinicalbert(request.clinical_note)
        
        if embeddings.nelement() == 0:
            return AnalysisResponse(
                extracted_keywords=[],
                matched_conditions=[]
            )
        
        # Match conditions
        matches = match_conditions(keywords, embeddings)
        
        return AnalysisResponse(
            extracted_keywords=keywords[:20],  # Return first 20 keywords
            matched_conditions=[
                MatchedConditionResponse(
                    condition=m['condition'],
                    icd_code=m['icd_code'],
                    icd_description=m['icd_description'],
                    similarity_score=m['similarity_score']
                )
                for m in matches
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

