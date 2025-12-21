"""
Test script for Authi 1.0 improvements
Demonstrates the 3-5 condition output and improved accuracy
"""

import requests
import json

# Backend URL (adjust if needed)
BASE_URL = "http://localhost:8000"

def test_authi_analysis():
    """Test the Authi 1.0 analysis with sample clinical notes"""
    
    # Sample clinical notes to test
    test_cases = [
        {
            "name": "Asthma Case",
            "note": "Patient presents with persistent wheezing, shortness of breath, and difficulty breathing especially at night. History of allergic reactions. Requires bronchodilator use multiple times per day."
        },
        {
            "name": "Diabetes Case",
            "note": "Patient reports increased thirst, frequent urination, and unexplained weight loss over the past 3 months. Blood glucose levels elevated. Family history of type 2 diabetes."
        },
        {
            "name": "Hypertension Case",
            "note": "Patient has persistent elevated blood pressure readings above 140/90. Complains of headaches and dizziness. Cardiovascular risk factors present."
        },
        {
            "name": "COPD Case",
            "note": "Chronic productive cough with sputum production. Progressive dyspnea on exertion. Long history of smoking (30 pack-years). Spirometry shows airflow limitation."
        }
    ]
    
    print("=" * 80)
    print("AUTHI 1.0 - TEST RESULTS")
    print("Testing ClinicalBERT + Authi 1.0 Matching System")
    print("=" * 80)
    print()
    
    # First check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Backend is running and healthy")
            print(f"   Status: {health.json()}")
            print()
        else:
            print("❌ Backend health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        print(f"   Error: {e}")
        print()
        print("Please ensure the backend is running:")
        print("   cd python-backend")
        print("   python main.py")
        return
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"\nClinical Note:")
        print(f"  {test_case['note']}")
        print()
        
        try:
            # Make API request
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"clinical_note": test_case['note']},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display extracted keywords
                print(f"Extracted Keywords ({len(data['extracted_keywords'])} total):")
                keywords_display = ', '.join(data['extracted_keywords'][:15])
                if len(data['extracted_keywords']) > 15:
                    keywords_display += f"... (+{len(data['extracted_keywords']) - 15} more)"
                print(f"  {keywords_display}")
                print()
                
                # Display matched conditions
                conditions = data['matched_conditions']
                print(f"Matched Conditions: {len(conditions)} conditions found")
                print()
                
                if len(conditions) >= 3 and len(conditions) <= 5:
                    print("✅ PASS: Returned 3-5 conditions as expected")
                elif len(conditions) < 3:
                    print(f"⚠️  WARNING: Only {len(conditions)} conditions returned (expected 3-5)")
                else:
                    print(f"⚠️  WARNING: {len(conditions)} conditions returned (expected 3-5)")
                print()
                
                for j, condition in enumerate(conditions, 1):
                    score_bar = "█" * int(condition['similarity_score'] * 20)
                    print(f"{j}. {condition['condition']}")
                    print(f"   ICD Code: {condition['icd_code']}")
                    print(f"   Description: {condition['icd_description']}")
                    print(f"   Similarity Score: {condition['similarity_score']:.3f} {score_bar}")
                    print()
                
            else:
                print(f"❌ Error: API returned status code {response.status_code}")
                print(f"   Response: {response.text}")
        
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("\n✅ Authi 1.0 Components:")
    print("   • ClinicalBERT: Extracts clinical keywords with embeddings")
    print("   • Authi 1.0: Maps keywords to chronic conditions")
    print("   • Output: 3-5 ranked condition suggestions with ICD codes")
    print()
    print("📊 Improvements:")
    print("   • Stop word filtering for better keyword quality")
    print("   • Weighted scoring (70% max + 30% average similarity)")
    print("   • Adaptive threshold for consistent 3-5 results")
    print("   • Enhanced logging and monitoring")
    print()

if __name__ == "__main__":
    test_authi_analysis()

