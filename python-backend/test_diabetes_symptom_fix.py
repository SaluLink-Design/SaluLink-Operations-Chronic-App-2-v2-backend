"""
Quick validation test for the original diabetes symptom detection issue
Tests the specific case that was failing before improvements
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_original_problem():
    """Test the original problem: diabetes symptoms not being detected"""
    
    print("\n" + "="*80)
    print("TESTING ORIGINAL PROBLEM: Diabetes Symptoms Not Detected")
    print("="*80)
    
    clinical_note = "Increased thirst, frequent urination, and persistent fatigue over the past 4 months"
    
    print(f"\nClinical Note: {clinical_note}")
    print("\nExpected: Diabetes should be the top condition(s)")
    print("Previous Result: Chronic Renal Disease and Hypertension (incorrect)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"clinical_note": clinical_note}
        )
        
        if response.status_code != 200:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        detected = [c['condition'] for c in result['matched_conditions']]
        
        print(f"\n{'='*80}")
        print("RESULTS")
        print("="*80)
        print(f"Detected {len(detected)} condition(s):\n")
        
        for i, c in enumerate(result['matched_conditions'], 1):
            status = "✓ CONFIRMED" if c.get('is_confirmed', False) else "→ Suggested"
            symptom_based = " [SYMPTOM-BASED]" if c.get('is_symptom_based', False) else ""
            print(f"{i}. [{status}] {c['condition']}")
            print(f"   Confidence: {c['similarity_score']*100:.1f}%{symptom_based}")
            print(f"   ICD: {c['icd_code']} - {c['icd_description'][:60]}...")
            print()
        
        # Validation checks
        print("="*80)
        print("VALIDATION")
        print("="*80)
        
        success = True
        
        # Check 1: Diabetes should be in top 3
        diabetes_found = False
        diabetes_position = None
        for i, condition in enumerate(detected[:3], 1):
            if 'diabetes' in condition.lower():
                diabetes_found = True
                diabetes_position = i
                break
        
        if diabetes_found:
            print(f"✅ Diabetes detected in top 3 (position {diabetes_position})")
        else:
            print(f"❌ Diabetes NOT detected in top 3")
            success = False
        
        # Check 2: Both diabetes types should be present (type ambiguous from symptoms)
        type1_found = any('type 1' in c.lower() for c in detected)
        type2_found = any('type 2' in c.lower() for c in detected)
        
        if type1_found and type2_found:
            print(f"✅ Both Diabetes Type 1 and Type 2 returned (correct for ambiguous symptoms)")
        elif type1_found or type2_found:
            print(f"⚠️  Only one diabetes type returned (acceptable but both preferred)")
        else:
            print(f"❌ No diabetes types detected")
            success = False
        
        # Check 3: Should return at least 3 conditions
        if len(detected) >= 3:
            print(f"✅ Returned {len(detected)} conditions (meets 3-5 requirement)")
        else:
            print(f"⚠️  Only {len(detected)} conditions returned (expected 3-5)")
        
        # Check 4: Chronic Renal Disease should NOT be top
        if detected and 'chronic renal' not in detected[0].lower():
            print(f"✅ Chronic Renal Disease correctly NOT ranked first")
        else:
            print(f"❌ Chronic Renal Disease incorrectly ranked first (old behavior)")
            success = False
        
        print("\n" + "="*80)
        if success:
            print("🎉 SUCCESS! Diabetes symptom detection is now working correctly!")
        else:
            print("⚠️  Some issues remain. See validation checks above.")
        print("="*80 + "\n")
        
        return success
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend!")
        print("Please ensure the backend is running:")
        print("  cd python-backend && python3 main.py")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Testing Original Diabetes Symptom Detection Issue...")
    print("This test validates that the improvements are working correctly.\n")
    
    import time
    time.sleep(0.5)
    
    test_original_problem()
