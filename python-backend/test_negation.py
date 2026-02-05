"""
Test negation detection specifically
"""

import requests

BASE_URL = "http://localhost:8000"

def test_negation_case(note: str, denied_condition: str):
    """Test that a specific condition is correctly negated"""
    
    print(f"\n{'='*80}")
    print(f"Testing negation of: {denied_condition}")
    print(f"{'='*80}")
    print(f"Note: {note[:150]}...")
    
    response = requests.post(f"{BASE_URL}/analyze", json={"clinical_note": note})
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        return False
    
    result = response.json()
    detected = [c['condition'] for c in result['matched_conditions']]
    
    print(f"\nDetected: {', '.join(detected) if detected else 'None'}")
    
    # Check if denied condition appears
    if any(denied_condition.lower() in d.lower() for d in detected):
        print(f"❌ FAIL: {denied_condition} was detected despite being negated")
        for c in result['matched_conditions']:
            if denied_condition.lower() in c['condition'].lower():
                print(f"   - Score: {c['similarity_score']*100:.1f}%")
                print(f"   - Explanation: {c['match_explanation']}")
                print(f"   - Is Confirmed: {c['is_confirmed']}")
        return False
    else:
        print(f"✅ PASS: {denied_condition} correctly excluded")
        return True


if __name__ == "__main__":
    print("\n🧪 Testing Negation Detection...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Denies seizures
    tests_total += 1
    if test_negation_case(
        "Patient with confirmed hypothyroidism. TSH 12.5, T4 low. On levothyroxine 100mcg. " 
        "Reports fatigue, cold intolerance, and weight gain. Thyroid exam shows mild enlargement. "
        "Denies cardiac symptoms, seizures, or bleeding disorders.",
        "Epilepsy"
    ):
        tests_passed += 1
    
    # Test 2: No history of diabetes
    tests_total += 1
    if test_negation_case(
        "Patient presents with chest pain. No history of diabetes or hypertension. "
        "Blood pressure 120/75, normal. Glucose 95 mg/dL, normal.",
        "Diabetes"
    ):
        tests_passed += 1
    
    # Test 3: Denies wheezing
    tests_total += 1
    if test_negation_case(
        "Patient with chronic cough. Denies wheezing, chest tightness, or shortness of breath. "
        "No history of asthma. Physical exam unremarkable.",
        "Asthma"
    ):
        tests_passed += 1
    
    print(f"\n{'='*80}")
    print(f"NEGATION TEST SUMMARY: {tests_passed}/{tests_total} passed")
    print(f"{'='*80}\n")
