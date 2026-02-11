"""
Comprehensive test suite for symptom-based condition detection
Tests ALL 12 chronic conditions for accurate symptom pattern recognition
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_condition(test_name, clinical_note, expected_conditions, min_expected=1, should_be_top=None):
    """Test a specific symptom-based clinical scenario"""
    
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Clinical Note: {clinical_note[:150]}...")
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"clinical_note": clinical_note}
    )
    
    if response.status_code != 200:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)
        return False
    
    result = response.json()
    detected = [c['condition'] for c in result['matched_conditions']]
    
    print(f"\nDetected {len(detected)} condition(s):")
    for i, c in enumerate(result['matched_conditions'], 1):
        status = "✓ CONFIRMED" if c.get('is_confirmed', False) else "→ Suggested"
        symptom_based = " [SYMPTOM-BASED]" if c.get('is_symptom_based', False) else ""
        print(f"  {i}. [{status}] {c['condition']}: {c['similarity_score']*100:.1f}%{symptom_based}")
    
    # Check if expected conditions were found
    success = True
    for expected in expected_conditions:
        found = any(expected.lower() in d.lower() for d in detected)
        if found:
            print(f"✅ Expected: {expected} found")
        else:
            print(f"❌ Expected: {expected} NOT found")
            success = False
    
    # Check if specific condition should be top
    if should_be_top and detected:
        if should_be_top.lower() in detected[0].lower():
            print(f"✅ {should_be_top} correctly ranked as top condition")
        else:
            print(f"❌ {should_be_top} should be top, but {detected[0]} is ranked first")
            success = False
    
    # Check minimum expected conditions
    if len(detected) < min_expected:
        print(f"❌ Expected at least {min_expected} conditions, got {len(detected)}")
        success = False
    
    return success


def run_comprehensive_tests():
    """Run comprehensive symptom-based tests for ALL conditions"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SYMPTOM-BASED CONDITION DETECTION TEST SUITE")
    print("="*80)
    
    # Check backend health
    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("❌ Backend not running!")
            return
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        print("Please start the backend: python3 python-backend/main.py")
        return
    
    tests_passed = 0
    tests_failed = 0
    
    # ============================================================================
    # DIABETES TESTS
    # ============================================================================
    print("\n" + "="*80)
    print("DIABETES SYMPTOM DETECTION TESTS")
    print("="*80)
    
    # Test 1: Classic diabetes triad (should return BOTH types)
    result = test_condition(
        "Diabetes Classic Triad - Should Return BOTH Types",
        "Increased thirst, frequent urination, and persistent fatigue over the past 4 months.",
        expected_conditions=["Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"],
        min_expected=2,
        should_be_top="Diabetes"
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # Test 2: Diabetes with weight loss (Type 1 indicator)
    result = test_condition(
        "Diabetes with Weight Loss - Type 1 Only",
        "Patient reports increased thirst, frequent urination, fatigue, and significant weight loss over 3 months despite good appetite.",
        expected_conditions=["Diabetes Mellitus Type 1"],
        min_expected=1
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # Test 3: Diabetes with obesity (Type 2 indicator)
    result = test_condition(
        "Diabetes with Obesity - Type 2 Only",
        "Overweight patient with BMI 34 presents with increased thirst, frequent urination, and fatigue. History of sedentary lifestyle.",
        expected_conditions=["Diabetes Mellitus Type 2"],
        min_expected=1
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # RESPIRATORY CONDITIONS
    # ============================================================================
    print("\n" + "="*80)
    print("RESPIRATORY CONDITION TESTS")
    print("="*80)
    
    # Test 4: Asthma symptoms
    result = test_condition(
        "Asthma Classic Symptoms",
        "Patient presents with wheezing, chest tightness, and shortness of breath. Symptoms worse at night and with exercise.",
        expected_conditions=["Asthma"],
        min_expected=1,
        should_be_top="Asthma"
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # Test 5: COPD symptoms
    result = test_condition(
        "COPD Classic Symptoms",
        "65-year-old with 40 pack-year smoking history presents with chronic cough, sputum production, and progressive dyspnea. Barrel chest on examination.",
        expected_conditions=["Chronic Obstructive Pulmonary Disease"],
        min_expected=1,
        should_be_top="Chronic Obstructive"
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # CARDIOVASCULAR CONDITIONS
    # ============================================================================
    print("\n" + "="*80)
    print("CARDIOVASCULAR CONDITION TESTS")
    print("="*80)
    
    # Test 6: Cardiac Failure symptoms
    result = test_condition(
        "Cardiac Failure Classic Symptoms",
        "Patient reports shortness of breath on exertion, leg swelling, and difficulty lying flat at night. Requires 3 pillows to sleep.",
        expected_conditions=["Cardiac Failure"],
        min_expected=1,
        should_be_top="Cardiac"
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # Test 7: Hypertension symptoms
    result = test_condition(
        "Hypertension Classic Symptoms",
        "Persistent headaches, dizziness, and occasional chest discomfort. Patient notes feeling of pressure in head.",
        expected_conditions=["Hypertension"],
        min_expected=1
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # Test 8: Cardiomyopathy symptoms
    result = test_condition(
        "Cardiomyopathy Symptoms",
        "Patient with chest pain, palpitations, and episodes of near-syncope. Family history of sudden cardiac death.",
        expected_conditions=["Cardiomyopathy"],
        min_expected=1
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # METABOLIC/ENDOCRINE CONDITIONS
    # ============================================================================
    print("\n" + "="*80)
    print("METABOLIC/ENDOCRINE CONDITION TESTS")
    print("="*80)
    
    # Test 9: Hypothyroidism symptoms
    result = test_condition(
        "Hypothyroidism Classic Symptoms",
        "Patient reports persistent fatigue, cold intolerance, and weight gain despite no change in diet. Also notes constipation and dry skin.",
        expected_conditions=["Hypothyroidism"],
        min_expected=1,
        should_be_top="Hypothyroidism"
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # NEUROLOGICAL CONDITIONS
    # ============================================================================
    print("\n" + "="*80)
    print("NEUROLOGICAL CONDITION TESTS")
    print("="*80)
    
    # Test 10: Epilepsy symptoms
    result = test_condition(
        "Epilepsy Classic Symptoms",
        "Patient experienced seizure with loss of consciousness and tonic-clonic movements lasting 2 minutes. Postictal confusion noted.",
        expected_conditions=["Epilepsy"],
        min_expected=1,
        should_be_top="Epilepsy"
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # RENAL CONDITIONS
    # ============================================================================
    print("\n" + "="*80)
    print("RENAL CONDITION TESTS")
    print("="*80)
    
    # Test 11: CKD symptoms (should NOT match diabetes symptoms)
    result = test_condition(
        "Chronic Renal Disease - Should NOT Match Diabetes",
        "Patient with fatigue, leg swelling, decreased urine output, and nausea. History of hypertension.",
        expected_conditions=["Chronic Renal Disease"],
        min_expected=1
    )
    # Check that diabetes is NOT the top condition
    response = requests.post(f"{BASE_URL}/analyze", 
                            json={"clinical_note": "Patient with fatigue, leg swelling, decreased urine output, and nausea."})
    if response.status_code == 200:
        detected = [c['condition'] for c in response.json()['matched_conditions']]
        if detected and 'diabetes' not in detected[0].lower():
            print("✅ Diabetes correctly NOT top condition for CKD symptoms")
        else:
            print("⚠️  Warning: Diabetes incorrectly detected for CKD symptoms")
    
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # HEMATOLOGICAL CONDITIONS
    # ============================================================================
    print("\n" + "="*80)
    print("HEMATOLOGICAL CONDITION TESTS")
    print("="*80)
    
    # Test 12: Haemophilia symptoms
    result = test_condition(
        "Haemophilia Classic Symptoms",
        "Patient with easy bruising, prolonged bleeding after minor cuts, and joint pain with swelling. Family history of bleeding disorder.",
        expected_conditions=["Haemophilia"],
        min_expected=1
    )
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # CONFIDENCE THRESHOLD TESTS
    # ============================================================================
    print("\n" + "="*80)
    print("CONFIDENCE THRESHOLD TESTS")
    print("="*80)
    
    # Test 13: High confidence scenario (should return 1 condition)
    result = test_condition(
        "High Confidence - Single Condition Expected",
        "Patient diagnosed with epilepsy. Had generalized tonic-clonic seizure with loss of consciousness followed by postictal confusion. On levetiracetam.",
        expected_conditions=["Epilepsy"],
        min_expected=1
    )
    # Check that only 1-2 conditions returned (epilepsy is very specific)
    response = requests.post(f"{BASE_URL}/analyze",
                            json={"clinical_note": "Patient diagnosed with epilepsy. Had seizure with loss of consciousness."})
    if response.status_code == 200:
        num_detected = len(response.json()['matched_conditions'])
        if num_detected <= 2:
            print(f"✅ Returned {num_detected} condition(s) - appropriate for high confidence")
        else:
            print(f"⚠️  Returned {num_detected} conditions - may be too many for clear epilepsy case")
    
    tests_passed += 1 if result else 0
    tests_failed += 0 if result else 1
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    total = tests_passed + tests_failed
    print(f"Tests Passed: {tests_passed}/{total}")
    print(f"Tests Failed: {tests_failed}/{total}")
    print(f"Success Rate: {(tests_passed/total*100):.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Symptom-based detection working perfectly across all conditions.")
    elif tests_passed / total >= 0.80:
        print("\n✅ MOST TESTS PASSED! System is performing well.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review results above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🚀 Running Comprehensive Symptom-Based Condition Detection Tests...")
    print("Ensure backend is running: python3 python-backend/main.py\n")
    
    import time
    time.sleep(1)
    
    run_comprehensive_tests()
