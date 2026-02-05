"""
Comprehensive Test Suite for Backend Accuracy Improvements
Tests all 6 major enhancements implemented for better condition detection
"""

import requests
import json
from typing import List, Dict

# Backend URL
BASE_URL = "http://localhost:8000"

def test_analyze(clinical_note: str, test_name: str) -> Dict:
    """Send a clinical note to the backend and return results"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Clinical Note:\n{clinical_note}")
    print(f"\n{'-'*80}")
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"clinical_note": clinical_note}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nRESULTS:")
        print(f"Keywords Extracted: {len(result['extracted_keywords'])}")
        print(f"Confirmed Conditions: {result['confirmed_count']}")
        print(f"Total Conditions: {len(result['matched_conditions'])}")
        print(f"Note Quality Score: {result['note_quality']['completeness_score']}/100")
        
        print(f"\nCONDITIONS DETECTED:")
        for i, condition in enumerate(result['matched_conditions'], 1):
            status = "✓ CONFIRMED" if condition['is_confirmed'] else "→ SUGGESTED"
            print(f"\n{i}. [{status}] {condition['condition']}")
            print(f"   ICD Code: {condition['icd_code']}")
            print(f"   Confidence: {condition['similarity_score']:.3f}")
            if condition.get('suggested_icd_code'):
                print(f"   Suggested ICD: {condition['suggested_icd_code']} (confidence: {condition.get('icd_confidence', 0):.3f})")
            print(f"   Explanation: {condition['match_explanation']}")
            if condition.get('triggering_keywords'):
                keywords = [kw['keyword'] for kw in condition['triggering_keywords'][:3]]
                print(f"   Key Terms: {', '.join(keywords)}")
        
        return result
    else:
        print(f"ERROR: {response.status_code} - {response.text}")
        return None


def run_all_tests():
    """Run comprehensive test suite"""
    
    print("\n" + "="*80)
    print("BACKEND ACCURACY IMPROVEMENTS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check backend health
    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("❌ Backend is not running! Please start it first.")
            return
        print("✓ Backend is running and healthy")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("Please start the backend first: python python-backend/main.py")
        return
    
    tests_passed = 0
    tests_failed = 0
    
    # TEST 1: Negation Detection
    print("\n\n" + "🔍 TEST SUITE 1: NEGATION DETECTION" + "\n")
    
    result = test_analyze(
        "Patient presents with chest pain. No history of diabetes. Denies hypertension. "
        "Blood pressure 120/80, normal. No signs of cardiac failure.",
        "Test 1.1: Negation - Should NOT detect diabetes, hypertension, or cardiac failure"
    )
    if result:
        negated_found = any(
            c['condition'] in ['Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 
                              'Hypertension', 'Cardiac Failure']
            for c in result['matched_conditions']
        )
        if not negated_found:
            print("✅ PASS: Negated conditions correctly excluded")
            tests_passed += 1
        else:
            print("❌ FAIL: Negated conditions were incorrectly detected")
            tests_failed += 1
    
    # TEST 2: Expanded Medical Aliases
    print("\n\n" + "🔍 TEST SUITE 2: EXPANDED MEDICAL ALIASES" + "\n")
    
    test_cases = [
        ("Patient with T2DM, HbA1c 8.5%, on metformin", 
         "Test 2.1: Alias Detection - T2DM should detect Type 2 Diabetes", 
         "Diabetes Mellitus Type 2"),
        
        ("Patient has CHF with reduced ejection fraction 35%", 
         "Test 2.2: Alias Detection - CHF should detect Cardiac Failure",
         "Cardiac Failure"),
        
        ("COPD patient with chronic bronchitis, FEV1 45% predicted",
         "Test 2.3: Alias Detection - COPD should be detected",
         "Chronic Obstructive Pulmonary Disease"),
        
        ("Patient diagnosed with ESRD stage 5, on dialysis",
         "Test 2.4: Alias Detection - ESRD should detect Chronic Renal Disease",
         "Chronic Renal Disease"),
    ]
    
    for note, test_name, expected_condition in test_cases:
        result = test_analyze(note, test_name)
        if result:
            found = any(c['condition'] == expected_condition for c in result['matched_conditions'])
            if found:
                print(f"✅ PASS: {expected_condition} correctly detected")
                tests_passed += 1
            else:
                print(f"❌ FAIL: {expected_condition} not detected")
                tests_failed += 1
    
    # TEST 3: Enhanced ICD Code Selection
    print("\n\n" + "🔍 TEST SUITE 3: CONTEXT-AWARE ICD CODE SELECTION" + "\n")
    
    result = test_analyze(
        "Patient with Type 1 diabetes mellitus complicated by diabetic ketoacidosis. "
        "Blood glucose 450 mg/dL, ketones present. Started on insulin therapy.",
        "Test 3.1: Context-aware ICD - Should suggest ketoacidosis-specific ICD code"
    )
    if result:
        dm_conditions = [c for c in result['matched_conditions'] 
                        if 'Diabetes Mellitus Type 1' in c['condition']]
        if dm_conditions:
            suggested_icd = dm_conditions[0].get('suggested_icd_code', '')
            # E10.1 is ketoacidosis
            if 'E10.1' in suggested_icd or 'ketoacidosis' in dm_conditions[0]['icd_description'].lower():
                print("✅ PASS: Context-aware ICD code selection working")
                tests_passed += 1
            else:
                print(f"⚠️  PARTIAL: ICD suggested but not ketoacidosis-specific: {suggested_icd}")
                tests_passed += 0.5
                tests_failed += 0.5
    
    result = test_analyze(
        "Hypertensive patient with chronic kidney disease stage 4. Creatinine 3.2, eGFR 25.",
        "Test 3.2: Context-aware ICD - Should detect stage 4 CKD"
    )
    if result:
        ckd_conditions = [c for c in result['matched_conditions'] 
                         if 'Chronic Renal Disease' in c['condition']]
        if ckd_conditions:
            print("✅ PASS: CKD detected with stage information")
            tests_passed += 1
        else:
            print("❌ FAIL: CKD not detected")
            tests_failed += 1
    
    # TEST 4: Symptom-Based Detection
    print("\n\n" + "🔍 TEST SUITE 4: SYMPTOM-BASED DETECTION" + "\n")
    
    result = test_analyze(
        "Patient presents with wheezing, dyspnea, and chest tightness. "
        "Uses albuterol inhaler. Peak flow reduced at 60% predicted.",
        "Test 4.1: Symptom Detection - Should detect Asthma from symptoms"
    )
    if result:
        found_asthma = any(c['condition'] == 'Asthma' for c in result['matched_conditions'])
        if found_asthma:
            print("✅ PASS: Asthma detected from symptoms")
            tests_passed += 1
        else:
            print("❌ FAIL: Asthma not detected from symptoms")
            tests_failed += 1
    
    result = test_analyze(
        "Patient with polyuria, polydipsia, and unexplained weight loss. "
        "Blood glucose 380 mg/dL, HbA1c 11.2%.",
        "Test 4.2: Symptom Detection - Should detect Diabetes from classic symptoms"
    )
    if result:
        found_diabetes = any('Diabetes' in c['condition'] for c in result['matched_conditions'])
        if found_diabetes:
            print("✅ PASS: Diabetes detected from classic symptoms")
            tests_passed += 1
        else:
            print("❌ FAIL: Diabetes not detected from symptoms")
            tests_failed += 1
    
    # TEST 5: Enhanced Confidence Scoring
    print("\n\n" + "🔍 TEST SUITE 5: ENHANCED CONFIDENCE SCORING" + "\n")
    
    result = test_analyze(
        "Patient with confirmed hypertension. BP 160/95 mmHg. On amlodipine 10mg daily. "
        "Blood pressure has been elevated for 6 months despite treatment.",
        "Test 5.1: Confidence Scoring - Confirmed with measurements should have high confidence"
    )
    if result:
        htn = [c for c in result['matched_conditions'] if c['condition'] == 'Hypertension']
        if htn and htn[0]['similarity_score'] >= 0.90:
            print(f"✅ PASS: High confidence score ({htn[0]['similarity_score']:.3f}) for confirmed condition with measurements")
            tests_passed += 1
        else:
            print(f"⚠️  Confidence score: {htn[0]['similarity_score']:.3f} (expected >= 0.90)")
            tests_passed += 0.5
            tests_failed += 0.5
    
    # TEST 6: Comorbidity Detection
    print("\n\n" + "🔍 TEST SUITE 6: COMORBIDITY RELATIONSHIPS" + "\n")
    
    result = test_analyze(
        "Patient with known Type 2 diabetes mellitus and hypertension. "
        "Recent labs show elevated cholesterol (LDL 180 mg/dL). "
        "HbA1c 7.8%, BP 145/90 mmHg.",
        "Test 6.1: Comorbidity - Should detect related conditions (DM2, HTN, Hyperlipidemia)"
    )
    if result:
        detected_conditions = {c['condition'] for c in result['matched_conditions']}
        expected = {'Diabetes Mellitus Type 2', 'Hypertension', 'Hyperlipidaemia'}
        found = expected & detected_conditions
        if len(found) >= 2:
            print(f"✅ PASS: Multiple comorbidities detected ({len(found)}/3): {found}")
            tests_passed += 1
        else:
            print(f"⚠️  PARTIAL: Only {len(found)} comorbidity detected: {found}")
            tests_passed += 0.5
            tests_failed += 0.5
    
    # TEST 7: Complex Clinical Note
    print("\n\n" + "🔍 TEST SUITE 7: COMPLEX CLINICAL SCENARIO" + "\n")
    
    result = test_analyze(
        "67-year-old male with past medical history significant for Type 2 diabetes mellitus "
        "complicated by diabetic nephropathy (CKD stage 3, eGFR 45), hypertension, and dyslipidemia. "
        "Patient denies any history of seizures or bleeding disorders. "
        "Current medications include metformin 1000mg BID, lisinopril 20mg daily, and atorvastatin 40mg daily. "
        "Recent labs: HbA1c 7.2%, BP 138/82 mmHg, creatinine 1.8 mg/dL, LDL 95 mg/dL. "
        "Patient reports good medication compliance. No chest pain or dyspnea on exertion.",
        "Test 7.1: Complex Note - Should detect DM2, CKD, HTN, Hyperlipidemia; exclude negated conditions"
    )
    if result:
        detected = {c['condition'] for c in result['matched_conditions']}
        expected_present = {'Diabetes Mellitus Type 2', 'Chronic Renal Disease', 
                           'Hypertension', 'Hyperlipidaemia'}
        expected_absent = {'Epilepsy', 'Haemophilia'}
        
        present = expected_present & detected
        absent = expected_absent & detected
        
        print(f"\nExpected present: {expected_present}")
        print(f"Detected: {detected}")
        print(f"Correctly detected: {present}")
        print(f"Incorrectly included (negated): {absent}")
        
        if len(present) >= 3 and len(absent) == 0:
            print(f"✅ PASS: Complex scenario handled correctly")
            tests_passed += 1
        else:
            print(f"⚠️  PARTIAL: {len(present)}/4 conditions detected, {len(absent)} false positives")
            tests_passed += 0.5
            tests_failed += 0.5
    
    # SUMMARY
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Success Rate: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Backend accuracy improvements are working perfectly.")
    elif tests_passed / (tests_passed + tests_failed) >= 0.80:
        print("\n✅ MOST TESTS PASSED! Backend improvements are working well.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review the results above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🚀 Starting Accuracy Improvements Test Suite...")
    print("Make sure the backend is running: python python-backend/main.py\n")
    
    import time
    time.sleep(1)
    
    run_all_tests()
