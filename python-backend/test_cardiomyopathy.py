"""
Quick test for cardiomyopathy detection
This is the specific test case mentioned by the user
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_cardiomyopathy():
    """Test the exact case the user mentioned"""
    
    clinical_note = "patient has been diagnosed with cardiomyopathy"
    
    print("=" * 80)
    print("CARDIOMYOPATHY DETECTION TEST")
    print("=" * 80)
    print(f"\nClinical Note: '{clinical_note}'")
    print()
    
    # Check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ Backend health check failed")
            print("\nPlease start the backend first:")
            print("   cd python-backend")
            print("   python main.py")
            return
        print("✅ Backend is running")
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        print(f"   Error: {e}")
        print("\nPlease start the backend first:")
        print("   cd python-backend")
        print("   python main.py")
        return
    
    # Test the analysis
    try:
        print("\nSending request to /analyze endpoint...")
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"clinical_note": clinical_note},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n📋 Extracted Keywords: {', '.join(data['extracted_keywords'][:10])}")
            print(f"\n🔍 Matched Conditions ({len(data['matched_conditions'])} found):")
            print()
            
            cardiomyopathy_found = False
            
            for i, condition in enumerate(data['matched_conditions'], 1):
                is_cardio = condition['condition'] == 'Cardiomyopathy'
                marker = "✅" if is_cardio else "  "
                
                print(f"{marker} {i}. {condition['condition']}")
                print(f"      ICD Code: {condition['icd_code']}")
                print(f"      Description: {condition['icd_description']}")
                print(f"      Similarity Score: {condition['similarity_score']:.3f}")
                print()
                
                if is_cardio:
                    cardiomyopathy_found = True
            
            print("=" * 80)
            if cardiomyopathy_found:
                print("✅ SUCCESS: Cardiomyopathy was correctly identified!")
                print("\nThe improved backend now uses:")
                print("  1. Direct condition name matching (word boundary detection)")
                print("  2. Medical keyword matching from ICD descriptions")
                print("  3. Semantic similarity using ClinicalBERT embeddings")
            else:
                print("❌ FAILED: Cardiomyopathy was NOT identified")
                print("\nDetected conditions:", [c['condition'] for c in data['matched_conditions']])
            print("=" * 80)
            
        else:
            print(f"❌ API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    test_cardiomyopathy()


