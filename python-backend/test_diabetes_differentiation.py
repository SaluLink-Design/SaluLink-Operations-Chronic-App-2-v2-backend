"""
Test script to verify that the backend correctly differentiates between
Type 1 and Type 2 diabetes based on clinical context
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_diabetes_scenario(scenario_name, clinical_note, expected_type, unexpected_type):
    """Test a specific diabetes scenario"""
    print(f"\n{'='*80}")
    print(f"TEST: {scenario_name}")
    print(f"{'='*80}")
    print(f"Clinical Note:\n{clinical_note}\n")
    
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
    
    print(f"Detected {len(detected)} condition(s):")
    for c in result['matched_conditions']:
        status = "✓ CONFIRMED" if c.get('is_confirmed', False) else "→ Suggested"
        print(f"  - [{status}] {c['condition']}: {c['similarity_score']*100:.1f}%")
    
    # Check if expected type is found
    expected_found = any(expected_type.lower() in d.lower() for d in detected)
    unexpected_found = any(unexpected_type.lower() in d.lower() for d in detected)
    
    if expected_found:
        print(f"✅ Expected: {expected_type} found")
    else:
        print(f"❌ Expected: {expected_type} NOT found")
    
    if unexpected_found:
        print(f"❌ Unexpected: {unexpected_type} found (should NOT be present)")
        return False
    else:
        print(f"✅ Correctly excluded: {unexpected_type}")
    
    return expected_found and not unexpected_found


def run_diabetes_tests():
    """Run comprehensive diabetes differentiation tests"""
    
    print("\n" + "="*80)
    print("DIABETES TYPE DIFFERENTIATION TEST SUITE")
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
    
    # TEST 1: Clear Type 1 indicators (DKA, insulin therapy, young age)
    result = test_diabetes_scenario(
        "Clear Type 1 Diabetes - Ketoacidosis",
        """24-year-old female patient diagnosed with Type 1 diabetes mellitus. 
        Presents with diabetic ketoacidosis (DKA). HbA1c 11.2%. 
        On intensive insulin therapy with insulin pump. 
        Requires continuous glucose monitoring.""",
        expected_type="Diabetes Mellitus Type 1",
        unexpected_type="Diabetes Mellitus Type 2"
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 2: Clear Type 2 indicators (metformin, older age, obesity)
    result = test_diabetes_scenario(
        "Clear Type 2 Diabetes - Metformin",
        """58-year-old male with Type 2 diabetes mellitus and obesity (BMI 34). 
        HbA1c 8.5%. Managed with metformin 1000mg twice daily. 
        Also on lisinopril for hypertension. 
        Reports good medication compliance.""",
        expected_type="Diabetes Mellitus Type 2",
        unexpected_type="Diabetes Mellitus Type 1"
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 3: Generic "diabetes" mention - should detect indicators
    result = test_diabetes_scenario(
        "Ambiguous - Generic diabetes with Type 2 indicators",
        """Patient with diabetes. On metformin. HbA1c 7.8%. 
        BMI 32. No history of ketoacidosis. 
        Adult-onset diabetes diagnosed 5 years ago.""",
        expected_type="Diabetes Mellitus Type 2",
        unexpected_type="Diabetes Mellitus Type 1"
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 4: Insulin-dependent Type 2 (should still be Type 2)
    result = test_diabetes_scenario(
        "Type 2 on Insulin",
        """67-year-old with longstanding Type 2 diabetes, now insulin-requiring. 
        Previously managed with metformin and glyburide. 
        Now on basal-bolus insulin regimen. 
        No ketoacidosis history.""",
        expected_type="Diabetes Mellitus Type 2",
        unexpected_type="Diabetes Mellitus Type 1"
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 5: Young person with Type 1 (IDDM)
    result = test_diabetes_scenario(
        "Juvenile Diabetes - Clear Type 1",
        """16-year-old with insulin-dependent diabetes mellitus (IDDM). 
        Diagnosed at age 8. History of diabetic ketoacidosis. 
        On insulin pump therapy. Requires frequent blood glucose monitoring.""",
        expected_type="Diabetes Mellitus Type 1",
        unexpected_type="Diabetes Mellitus Type 2"
    )
    if result:
        tests_passed += 1
    else:
        tests_failed += 1
    
    # TEST 6: Just "diabetes" mentioned - ambiguous case
    result = test_diabetes_scenario(
        "Ambiguous - Just 'diabetes' mentioned",
        """Patient has diabetes. HbA1c 9.1%. 
        Reports polyuria and polydipsia.""",
        expected_type="Diabetes",  # Should detect at least one type
        unexpected_type=""  # No specific exclusion in this case
    )
    # This test is informational - just print what was detected
    print(f"\n   ℹ️  This is an AMBIGUOUS case - system should make best guess based on limited info")
    
    # SUMMARY
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    total = tests_passed + tests_failed
    print(f"Tests Passed: {tests_passed}/{total}")
    print(f"Tests Failed: {tests_failed}/{total}")
    print(f"Success Rate: {(tests_passed/total*100):.1f}%")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Diabetes differentiation working perfectly.")
    elif tests_passed / total >= 0.80:
        print("\n✅ MOST TESTS PASSED! System is performing well.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Diabetes differentiation needs improvement.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🚀 Running Diabetes Differentiation Test Suite...")
    print("Ensure backend is running: python3 python-backend/main.py\n")
    
    import time
    time.sleep(1)
    
    run_diabetes_tests()
