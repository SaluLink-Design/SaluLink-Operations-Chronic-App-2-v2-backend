"""
Test edge cases where the backend might confuse Type 1 and Type 2 diabetes
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_scenario(test_name, clinical_note):
    """Test a specific diabetes scenario and report what's detected"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Clinical Note:\n{clinical_note}\n")
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"clinical_note": clinical_note}
    )
    
    if response.status_code != 200:
        print(f"❌ ERROR: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    detected = [c['condition'] for c in result['matched_conditions']]
    
    print(f"Detected {len(detected)} condition(s):")
    for c in result['matched_conditions']:
        status = "✓ CONFIRMED" if c.get('is_confirmed', False) else "→ Suggested"
        print(f"  - [{status}] {c['condition']}: {c['similarity_score']*100:.1f}%")
        print(f"    ICD: {c['icd_code']} - {c['icd_description'][:80]}...")
    
    # Check if both types are detected
    has_type1 = any('type 1' in d.lower() for d in detected)
    has_type2 = any('type 2' in d.lower() for d in detected)
    
    if has_type1 and has_type2:
        print("\n⚠️  WARNING: BOTH Type 1 AND Type 2 detected!")
    elif has_type1:
        print("\n✓ Only Type 1 detected")
    elif has_type2:
        print("\n✓ Only Type 2 detected")
    else:
        print("\n⚠️  No diabetes type detected")


def run_edge_case_tests():
    """Run edge case tests to identify confusion"""
    
    print("\n" + "="*80)
    print("DIABETES EDGE CASE TESTS - Identifying Confusion Points")
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
    
    # EDGE CASE 1: Just "diabetes" mentioned - no type specified
    test_scenario(
        "EDGE CASE 1: Generic 'diabetes' only",
        """Patient with diabetes. Reports fatigue and increased thirst."""
    )
    
    # EDGE CASE 2: "Diabetic" as adjective - no type
    test_scenario(
        "EDGE CASE 2: 'Diabetic patient' - no type specified",
        """Diabetic patient presents with foot ulcer. Blood glucose elevated."""
    )
    
    # EDGE CASE 3: "DM" abbreviation without type
    test_scenario(
        "EDGE CASE 3: 'DM' abbreviation",
        """History of DM, hypertension, and hyperlipidemia."""
    )
    
    # EDGE CASE 4: Diabetes complications without type
    test_scenario(
        "EDGE CASE 4: Diabetic complications - no type",
        """Patient with diabetic nephropathy and diabetic retinopathy. 
        HbA1c 10.2%. On multiple medications."""
    )
    
    # EDGE CASE 5: Insulin mention (could be either type)
    test_scenario(
        "EDGE CASE 5: Patient on insulin (ambiguous)",
        """Patient on insulin for diabetes control. HbA1c 8.9%."""
    )
    
    # EDGE CASE 6: Both keywords present (Type 1 + Type 2 keywords)
    test_scenario(
        "EDGE CASE 6: Mixed keywords - has both T1DM and metformin",
        """Patient with diabetes. Was on metformin. Now has ketoacidosis 
        and requires insulin therapy. History of poor control."""
    )
    
    # EDGE CASE 7: Elderly patient with "diabetes" (should likely be Type 2)
    test_scenario(
        "EDGE CASE 7: Elderly + diabetes (likely Type 2)",
        """82-year-old with diabetes, managed at home. 
        Reports polyuria. On oral medications."""
    )
    
    # EDGE CASE 8: Young patient with "diabetes" (could be either)
    test_scenario(
        "EDGE CASE 8: Young patient + diabetes (ambiguous)",
        """19-year-old presents with diabetes. Recently diagnosed. 
        Weight loss and fatigue noted."""
    )
    
    print("\n" + "="*80)
    print("EDGE CASE ANALYSIS COMPLETE")
    print("="*80)
    print("\nℹ️  Review the results above to identify where confusion occurs.")
    print("   Key questions:")
    print("   1. Are both types ever detected simultaneously?")
    print("   2. Is the default choice appropriate for ambiguous cases?")
    print("   3. Are keywords being weighted correctly?")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🔍 Running Diabetes Edge Case Analysis...")
    print("This test identifies specific scenarios where Type 1/Type 2 confusion occurs.\n")
    
    run_edge_case_tests()
