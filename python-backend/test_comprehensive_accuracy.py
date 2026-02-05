"""
Comprehensive accuracy test suite for ClinicalBERT improvements
Tests multiple scenarios to ensure filtering works correctly
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_scenario(test_name: str, clinical_note: str, expected_conditions: list, unexpected_conditions: list = None):
    """Test a specific clinical scenario"""
    
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Clinical Note (excerpt): {clinical_note[:200]}...")
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"clinical_note": clinical_note}
    )
    
    if response.status_code != 200:
        print(f"❌ ERROR: {response.status_code}")
        return False
    
    result = response.json()
    detected = [c['condition'] for c in result['matched_conditions']]
    
    print(f"\nDetected {len(detected)} condition(s):")
    for c in result['matched_conditions']:
        print(f"  - {c['condition']}: {c['similarity_score']*100:.1f}%")
    
    # Check expected conditions
    all_expected_found = True
    for expected in expected_conditions:
        if any(expected.lower() in d.lower() for d in detected):
            print(f"✅ Expected: {expected} found")
        else:
            print(f"❌ Expected: {expected} NOT found")
            all_expected_found = False
    
    # Check unexpected conditions
    if unexpected_conditions:
        no_unexpected_found = True
        for unexpected in unexpected_conditions:
            if any(unexpected.lower() in d.lower() for d in detected):
                print(f"❌ Unexpected: {unexpected} found (should be filtered)")
                no_unexpected_found = False
            else:
                print(f"✅ Correctly filtered: {unexpected}")
        
        return all_expected_found and no_unexpected_found
    
    return all_expected_found


def run_all_tests():
    """Run comprehensive test suite"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ACCURACY TEST SUITE")
    print("="*80)
    
    # Check backend health
    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("❌ Backend not running!")
            return
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        return
    
    tests_passed = 0
    tests_failed = 0
    
    # TEST 1: Clear single condition (Epilepsy)
    result = test_scenario(
        "Single Clear Condition - Epilepsy",
        """Patient is a 24-year-old female with a known diagnosis of generalized epilepsy. 
        She reports experiencing two seizure episodes in the past month. Seizures described 
        as sudden loss of consciousness followed by generalized tonic-clonic movements 
        lasting approximately 1–2 minutes. Postictal confusion reported. On sodium valproate.""",
        expected_conditions=["Epilepsy"],
        unexpected_conditions=["Hypertension", "Diabetes"]
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 2: Clear single condition (Asthma)
    result = test_scenario(
        "Single Clear Condition - Asthma",
        """28-year-old male presents with wheezing and dyspnea. Diagnosed with asthma 
        at age 10. Uses albuterol inhaler PRN. Peak flow measurements show 65% of predicted. 
        Reports nocturnal cough and chest tightness with exercise. No cardiac symptoms.""",
        expected_conditions=["Asthma"],
        unexpected_conditions=["Cardiac Failure", "COPD"]
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 3: Related comorbidities (should show multiple)
    result = test_scenario(
        "Related Comorbidities - Diabetes + Hypertension",
        """Patient with Type 2 diabetes mellitus and hypertension. HbA1c 8.2%, 
        BP 155/92 mmHg. On metformin and lisinopril. Reports polyuria and polydipsia. 
        Blood pressure consistently elevated over past year.""",
        expected_conditions=["Diabetes", "Hypertension"],
        unexpected_conditions=[]
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 4: Unrelated conditions (should filter)
    result = test_scenario(
        "Unrelated Conditions Should Filter",
        """Patient with confirmed hypothyroidism. TSH 12.5, T4 low. On levothyroxine 100mcg. 
        Reports fatigue, cold intolerance, and weight gain. Thyroid exam shows mild enlargement. 
        Denies cardiac symptoms, seizures, or bleeding disorders.""",
        expected_conditions=["Hypothyroidism"],
        unexpected_conditions=["Epilepsy", "Haemophilia", "Cardiac Failure"]
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 5: High confidence with related condition
    result = test_scenario(
        "High Confidence + Related Condition",
        """Patient with chronic heart failure, NYHA class III. Ejection fraction 30%. 
        History of hypertension. On furosemide, carvedilol, and lisinopril. 
        Reports dyspnea on exertion, orthopnea, and peripheral edema. BP 142/88.""",
        expected_conditions=["Cardiac Failure", "Hypertension"],
        unexpected_conditions=["Diabetes", "Asthma"]
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 6: Symptoms only (no direct mention)
    result = test_scenario(
        "Symptom-Based Detection",
        """45-year-old presents with persistent cough and progressive dyspnea. 
        30 pack-year smoking history. Spirometry shows FEV1 55% predicted. 
        Prolonged expiration noted. Uses tiotropium inhaler. Barrel chest on exam.""",
        expected_conditions=["Chronic Obstructive"],  # Will match "Chronic Obstructive Pulmonary Disease"
        unexpected_conditions=[]
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # SUMMARY
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    total = tests_passed + tests_failed
    print(f"Tests Passed: {tests_passed}/{total}")
    print(f"Tests Failed: {tests_failed}/{total}")
    print(f"Success Rate: {(tests_passed/total*100):.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Accuracy improvements working perfectly.")
    elif tests_passed / total >= 0.80:
        print("\n✅ MOST TESTS PASSED! System is performing well.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review results above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🚀 Running Comprehensive Accuracy Test Suite...")
    print("Ensure backend is running: python3 python-backend/main.py\n")
    
    import time
    time.sleep(1)
    
    run_all_tests()
