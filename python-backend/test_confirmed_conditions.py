"""
Test script to verify the CONFIRMED vs SUGGESTED condition detection logic

This tests that:
1. Conditions explicitly mentioned in the note are marked as CONFIRMED
2. When confirmed conditions exist, only related conditions are suggested
3. Unrelated conditions are NOT suggested when confirmed conditions exist
"""

import requests
import json
import sys

# Backend URL
BASE_URL = "http://localhost:8000"

def test_confirmed_conditions():
    """Test the new confirmed condition detection logic"""
    
    test_cases = [
        # Test 1: Single confirmed condition - should only return that condition and related ones
        {
            "name": "Single condition - Hypertension explicit mention",
            "note": "Patient diagnosed with Hypertension. Blood pressure consistently elevated at 160/100.",
            "expected_confirmed": ["Hypertension"],
            "should_not_include": ["Asthma", "Haemophilia", "Diabetes Mellitus Type 1"]
        },
        
        # Test 2: Two confirmed conditions mentioned
        {
            "name": "Two conditions - Hypertension and Diabetes Type 2",
            "note": "Patient has Hypertension and Type 2 Diabetes. Currently on metformin and lisinopril.",
            "expected_confirmed": ["Hypertension", "Diabetes Mellitus Type 2"],
            "should_not_include": ["Asthma", "Haemophilia"]
        },
        
        # Test 3: Single condition with alias - should still be confirmed
        {
            "name": "Alias detection - Heart Failure (alias for Cardiac Failure)",
            "note": "Patient presents with congestive heart failure, shortness of breath, and leg edema.",
            "expected_confirmed": ["Cardiac Failure"],
            "should_not_include": ["Asthma", "Haemophilia"]
        },
        
        # Test 4: Multiple aliases
        {
            "name": "Multiple aliases - High blood pressure and CKD",
            "note": "Patient has high blood pressure and chronic kidney disease stage 3.",
            "expected_confirmed": ["Hypertension", "Chronic Renal Disease"],
            "should_not_include": ["Asthma", "Haemophilia"]
        },
        
        # Test 5: No explicit condition - should suggest based on symptoms
        {
            "name": "No explicit condition - Should use semantic matching",
            "note": "Patient presents with wheezing, shortness of breath, and chest tightness that worsens at night.",
            "expected_confirmed": [],  # No confirmed, but should suggest Asthma
            "may_suggest": ["Asthma"]
        },
        
        # Test 6: Single specific condition - should NOT suggest unrelated
        {
            "name": "Single specific - Asthma (should not suggest diabetes, hypertension)",
            "note": "Patient has Asthma and requires daily inhaler use.",
            "expected_confirmed": ["Asthma"],
            "should_not_include": ["Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2", "Hypertension", "Cardiac Failure"]
        },
        
        # Test 7: Cardiomyopathy - should only suggest related cardiac conditions
        {
            "name": "Cardiomyopathy - should only suggest related conditions",
            "note": "Patient diagnosed with dilated cardiomyopathy on echocardiogram.",
            "expected_confirmed": ["Cardiomyopathy"],
            "should_not_include": ["Asthma", "Haemophilia", "Diabetes Mellitus Type 1"]
        },
        
        # Test 8: Complex case with multiple explicit conditions
        {
            "name": "Complex - Hypertension, Diabetes Type 2, and CKD",
            "note": "Patient has Hypertension, Type 2 Diabetes, and Chronic Renal Disease stage 4.",
            "expected_confirmed": ["Hypertension", "Diabetes Mellitus Type 2", "Chronic Renal Disease"],
            "should_not_include": ["Asthma", "Haemophilia"]
        }
    ]
    
    print("=" * 80)
    print("CONFIRMED vs SUGGESTED CONDITION DETECTION TEST")
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
        print("\nPlease ensure the backend is running:")
        print("   cd python-backend")
        print("   python main.py")
        return False
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 80}")
        print(f"Note: {test['note'][:100]}...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"clinical_note": test['note']},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                conditions = data['matched_conditions']
                confirmed_count = data.get('confirmed_count', 0)
                
                print(f"\nResults:")
                print(f"  Total conditions returned: {len(conditions)}")
                print(f"  Confirmed count: {confirmed_count}")
                
                # Separate confirmed and suggested
                confirmed = [c for c in conditions if c.get('is_confirmed', False)]
                suggested = [c for c in conditions if not c.get('is_confirmed', False)]
                
                print(f"\n  Confirmed conditions ({len(confirmed)}):")
                for c in confirmed:
                    print(f"    ✓ {c['condition']} ({c['icd_code']}) - Score: {c['similarity_score']:.3f}")
                
                print(f"\n  Suggested conditions ({len(suggested)}):")
                for c in suggested:
                    print(f"    → {c['condition']} ({c['icd_code']}) - Score: {c['similarity_score']:.3f}")
                
                # Verify expected confirmed conditions
                test_passed = True
                confirmed_names = [c['condition'] for c in confirmed]
                all_condition_names = [c['condition'] for c in conditions]
                
                # Check expected confirmed conditions are present
                for expected in test.get('expected_confirmed', []):
                    if expected not in confirmed_names:
                        print(f"\n  ❌ FAIL: Expected '{expected}' to be CONFIRMED but it wasn't")
                        test_passed = False
                
                # Check that unwanted conditions are not present
                for unwanted in test.get('should_not_include', []):
                    if unwanted in all_condition_names:
                        print(f"\n  ❌ FAIL: '{unwanted}' should NOT be in results but was found")
                        test_passed = False
                
                # Check suggested conditions if specified
                for may_suggest in test.get('may_suggest', []):
                    if may_suggest in all_condition_names:
                        print(f"\n  ✓ Good: '{may_suggest}' was suggested as expected")
                
                if test_passed:
                    print(f"\n  ✅ TEST PASSED")
                    passed += 1
                else:
                    print(f"\n  ❌ TEST FAILED")
                    failed += 1
                    
            else:
                print(f"  ❌ ERROR - API returned status {response.status_code}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR - {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Confirmed condition logic is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the output above.")
    
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = test_confirmed_conditions()
    sys.exit(0 if success else 1)

