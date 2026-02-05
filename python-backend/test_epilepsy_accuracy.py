"""
Test the epilepsy note accuracy improvements
This test verifies that the comprehensive filtering approach works correctly
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_epilepsy_note():
    """Test the exact epilepsy note provided by the user"""
    
    epilepsy_note = """Chief Complaint
Recurrent seizure episodes with recent increase in frequency.

⸻

History of Present Illness
Patient is a 24-year-old female with a known diagnosis of generalized epilepsy diagnosed at age 16. She reports experiencing two seizure episodes in the past month, compared to her usual baseline of one episode every 6–8 months.

Seizures described as sudden loss of consciousness followed by generalized tonic-clonic movements lasting approximately 1–2 minutes. Postictal confusion and fatigue lasting 30–45 minutes reported. No tongue biting reported during recent episodes, but urinary incontinence occurred during the most recent seizure.

Patient admits to missing several doses of sodium valproate over the past two months due to medication stock issues. Reports increased academic stress and irregular sleep patterns. Denies recent head trauma, fever, or substance use."""
    
    print("\n" + "="*80)
    print("EPILEPSY ACCURACY TEST")
    print("="*80)
    print("\nClinical Note:")
    print(epilepsy_note)
    print("\n" + "-"*80)
    
    # Check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("❌ Backend is not running! Please start it first.")
            print("   Run: cd python-backend && python main.py")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Run: cd python-backend && python main.py")
        return False
    
    # Send request
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"clinical_note": epilepsy_note}
    )
    
    if response.status_code != 200:
        print(f"❌ ERROR: {response.status_code} - {response.text}")
        return False
    
    result = response.json()
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nKeywords Extracted: {len(result['extracted_keywords'])}")
    print(f"Conditions Detected: {len(result['matched_conditions'])}")
    print(f"Confirmed Conditions: {result['confirmed_count']}")
    print(f"Note Quality Score: {result['note_quality']['completeness_score']}/100")
    
    print("\n" + "-"*80)
    print("DETECTED CONDITIONS:")
    print("-"*80)
    
    for i, condition in enumerate(result['matched_conditions'], 1):
        status = "✓ CONFIRMED" if condition['is_confirmed'] else "→ SUGGESTED"
        print(f"\n{i}. [{status}] {condition['condition']}")
        print(f"   ICD Code: {condition['icd_code']}")
        print(f"   Confidence: {condition['similarity_score']:.3f} ({condition['similarity_score']*100:.1f}%)")
        print(f"   Explanation: {condition['match_explanation']}")
        
        if condition.get('triggering_keywords'):
            keywords = [kw['keyword'] for kw in condition['triggering_keywords'][:5]]
            print(f"   Triggering Keywords: {', '.join(keywords)}")
    
    # Validation
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    detected_conditions = [c['condition'] for c in result['matched_conditions']]
    
    # Check for epilepsy
    epilepsy_found = any('Epilepsy' in c for c in detected_conditions)
    if epilepsy_found:
        epilepsy_condition = [c for c in result['matched_conditions'] if 'Epilepsy' in c['condition']][0]
        epilepsy_score = epilepsy_condition['similarity_score']
        print(f"✅ PASS: Epilepsy detected with {epilepsy_score*100:.1f}% confidence")
    else:
        print("❌ FAIL: Epilepsy not detected!")
        return False
    
    # Check for hypertension (should NOT be present)
    hypertension_found = any('Hypertension' in c for c in detected_conditions)
    if hypertension_found:
        hypertension_condition = [c for c in result['matched_conditions'] if 'Hypertension' in c['condition']][0]
        hypertension_score = hypertension_condition['similarity_score']
        print(f"❌ FAIL: Hypertension incorrectly detected ({hypertension_score*100:.1f}% confidence)")
        print("   This note has NO blood pressure readings or hypertension symptoms!")
        return False
    else:
        print("✅ PASS: Hypertension correctly filtered out")
    
    # Check result count (should be 1 for this clear case)
    if len(result['matched_conditions']) == 1:
        print(f"✅ PASS: Returned only 1 condition (as expected for clear case)")
    elif len(result['matched_conditions']) <= 2:
        print(f"⚠️  PARTIAL: Returned {len(result['matched_conditions'])} conditions (acceptable)")
    else:
        print(f"⚠️  WARNING: Returned {len(result['matched_conditions'])} conditions (expected 1)")
    
    # Check for any unrelated conditions
    unrelated_conditions = [c for c in detected_conditions if 'Epilepsy' not in c]
    if unrelated_conditions:
        print(f"\n⚠️  NOTE: Additional conditions detected: {', '.join(unrelated_conditions)}")
        for cond_name in unrelated_conditions:
            cond = [c for c in result['matched_conditions'] if c['condition'] == cond_name][0]
            print(f"   - {cond_name}: {cond['similarity_score']*100:.1f}% confidence")
    
    print("\n" + "="*80)
    print("✅ TEST PASSED - Accuracy improvements working correctly!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    print("\n🚀 Testing Epilepsy Note Accuracy Improvements...")
    print("Make sure the backend is running: python python-backend/main.py\n")
    
    import time
    time.sleep(1)
    
    success = test_epilepsy_note()
    
    if success:
        print("\n🎉 All tests passed! The backend now accurately identifies conditions.")
    else:
        print("\n❌ Tests failed. Review the output above for details.")
