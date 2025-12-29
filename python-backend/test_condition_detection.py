"""
Comprehensive test script to verify all chronic conditions can be detected
Tests both direct condition name matching and semantic matching
"""

import requests
import json
import sys

# Backend URL
BASE_URL = "http://localhost:8000"

def test_condition_detection():
    """Test detection of all major chronic conditions"""
    
    # Test cases covering all conditions in the database
    test_cases = [
        {
            "condition": "Cardiomyopathy",
            "notes": [
                "Patient has been diagnosed with cardiomyopathy",
                "Patient presents with dilated cardiomyopathy and shortness of breath",
                "History of hypertrophic cardiomyopathy with chest pain",
                "Ischaemic cardiomyopathy confirmed by echocardiogram"
            ]
        },
        {
            "condition": "Asthma",
            "notes": [
                "Patient diagnosed with asthma",
                "Patient has allergic asthma with frequent wheezing",
                "Status asthmaticus requiring emergency treatment"
            ]
        },
        {
            "condition": "Diabetes Mellitus Type 1",
            "notes": [
                "Patient diagnosed with type 1 diabetes mellitus",
                "Insulin-dependent diabetes mellitus with ketoacidosis",
                "Patient has diabetes type 1 requiring daily insulin"
            ]
        },
        {
            "condition": "Diabetes Mellitus Type 2",
            "notes": [
                "Patient diagnosed with type 2 diabetes",
                "Non-insulin-dependent diabetes mellitus poorly controlled",
                "Patient has diabetes type 2 managed with metformin"
            ]
        },
        {
            "condition": "Hypertension",
            "notes": [
                "Patient diagnosed with hypertension",
                "Essential hypertension with elevated blood pressure readings",
                "Patient has high blood pressure requiring treatment"
            ]
        },
        {
            "condition": "Chronic Renal Disease",
            "notes": [
                "Patient diagnosed with chronic renal disease",
                "Chronic kidney disease stage 3 with declining GFR",
                "End-stage renal disease requiring dialysis"
            ]
        },
        {
            "condition": "Cardiac Failure",
            "notes": [
                "Patient diagnosed with cardiac failure",
                "Congestive heart failure with reduced ejection fraction",
                "Patient has heart failure with leg swelling"
            ]
        },
        {
            "condition": "Hyperlipidaemia",
            "notes": [
                "Patient diagnosed with hyperlipidaemia",
                "Mixed hyperlipidaemia with elevated cholesterol and triglycerides",
                "Patient has high cholesterol requiring statin therapy"
            ]
        },
        {
            "condition": "Haemophilia",
            "notes": [
                "Patient diagnosed with haemophilia",
                "Hereditary factor VIII deficiency causing bleeding disorder",
                "Patient has hemophilia with recurrent joint bleeds"
            ]
        }
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE CONDITION DETECTION TEST")
    print("Testing all major chronic conditions")
    print("=" * 80)
    print()
    
    # Check backend health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Backend is running and healthy")
            print()
        else:
            print("❌ Backend health check failed")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        print(f"   Error: {e}")
        print()
        print("Please ensure the backend is running:")
        print("   cd python-backend")
        print("   python main.py")
        return False
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    # Test each condition
    for test_case in test_cases:
        condition_name = test_case['condition']
        print(f"\n{'=' * 80}")
        print(f"Testing: {condition_name}")
        print(f"{'=' * 80}")
        
        condition_detected = False
        
        for i, note in enumerate(test_case['notes'], 1):
            total_tests += 1
            print(f"\nTest {i}/{len(test_case['notes'])}: {note[:70]}...")
            
            try:
                response = requests.post(
                    f"{BASE_URL}/analyze",
                    json={"clinical_note": note},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    conditions = data['matched_conditions']
                    
                    # Check if the expected condition is in the results
                    found = False
                    for matched in conditions:
                        if matched['condition'] == condition_name:
                            found = True
                            condition_detected = True
                            print(f"   ✅ PASS - {condition_name} detected!")
                            print(f"      ICD Code: {matched['icd_code']}")
                            print(f"      Score: {matched['similarity_score']:.3f}")
                            passed_tests += 1
                            break
                    
                    if not found:
                        print(f"   ❌ FAIL - {condition_name} NOT detected")
                        print(f"      Detected conditions: {[c['condition'] for c in conditions]}")
                        failed_tests.append({
                            'condition': condition_name,
                            'note': note,
                            'detected': [c['condition'] for c in conditions]
                        })
                else:
                    print(f"   ❌ ERROR - API returned status {response.status_code}")
                    failed_tests.append({
                        'condition': condition_name,
                        'note': note,
                        'error': f"Status {response.status_code}"
                    })
                    
            except Exception as e:
                print(f"   ❌ ERROR - {e}")
                failed_tests.append({
                    'condition': condition_name,
                    'note': note,
                    'error': str(e)
                })
        
        if condition_detected:
            print(f"\n✅ Overall: {condition_name} detection SUCCESSFUL")
        else:
            print(f"\n❌ Overall: {condition_name} detection FAILED in all test cases")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal tests run: {total_tests}")
    print(f"Tests passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Tests failed: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")
    
    if failed_tests:
        print(f"\n⚠️  Failed Tests:")
        for i, failed in enumerate(failed_tests, 1):
            print(f"\n{i}. Condition: {failed['condition']}")
            print(f"   Note: {failed['note'][:70]}...")
            if 'detected' in failed:
                print(f"   Detected instead: {failed['detected']}")
            if 'error' in failed:
                print(f"   Error: {failed['error']}")
    else:
        print(f"\n🎉 All tests passed! All conditions can be detected correctly.")
    
    print("\n" + "=" * 80)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = test_condition_detection()
    sys.exit(0 if success else 1)

