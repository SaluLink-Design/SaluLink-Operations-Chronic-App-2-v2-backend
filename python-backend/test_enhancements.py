"""
Test script for Authi 1.0 Trust & Transparency Enhancements

Tests the three new features:
1. Clinical Note Completeness Validation
2. AI Transparency - Keyword Match Details
3. Automatic ICD Code Suggestions
"""

import requests
import json
from typing import Dict

BASE_URL = "http://localhost:8000"

# Test cases with different quality levels
TEST_CASES = [
    {
        "name": "High Quality Note - Diabetes with Ketoacidosis",
        "clinical_note": """
Patient presents with diagnosed diabetes mellitus type 1 with ketoacidosis.
Blood pressure: 140/90 mmHg, Heart rate: 88 bpm, Temperature: 37.2°C
Blood glucose: 18.5 mmol/L, HbA1c: 9.2%
Symptoms started 2 weeks ago and have been worsening progressively.
Patient reports severe thirst, frequent urination, and moderate fatigue.
Previous history of diabetes for 5 years, poorly controlled.
        """.strip(),
        "expected_quality_score": ">= 80",
        "expected_confirmed": True,
        "expected_condition": "Diabetes Mellitus Type 1"
    },
    {
        "name": "Medium Quality Note - Asthma",
        "clinical_note": """
Patient with asthma presents with wheezing and shortness of breath.
Symptoms worsen with exercise and cold air exposure.
History of allergic rhinitis.
        """.strip(),
        "expected_quality_score": "50-79",
        "expected_confirmed": False,
        "expected_condition": "Asthma"
    },
    {
        "name": "Low Quality Note - Hypertension",
        "clinical_note": """
Patient has high blood pressure.
        """.strip(),
        "expected_quality_score": "< 50",
        "expected_confirmed": False,
        "expected_condition": "Hypertension"
    },
    {
        "name": "Comprehensive Note - Multiple Conditions",
        "clinical_note": """
Patient presents with hypertension and chronic renal disease.
BP: 165/95 mmHg, HR: 72 bpm, Temp: 36.8°C
Serum creatinine elevated at 2.8 mg/dL, eGFR: 35 ml/min
Patient reports fatigue and reduced urine output for the past 3 months.
Diagnosed with Stage 3 chronic kidney disease.
History of uncontrolled hypertension for 8 years.
Current medications: ACE inhibitor, diuretic.
        """.strip(),
        "expected_quality_score": ">= 80",
        "expected_confirmed": True,
        "expected_condition": "Hypertension"
    },
    {
        "name": "Cardiac Failure with Specifics",
        "clinical_note": """
Patient diagnosed with cardiac failure and cardiomyopathy.
Presenting with severe dyspnea, peripheral edema, and chest discomfort.
BP: 110/70 mmHg, HR: 110 bpm, elevated JVP
Echocardiogram shows left ventricular ejection fraction of 30%.
BNP levels significantly elevated at 850 pg/ml.
Symptoms have been progressively worsening over 6 months.
        """.strip(),
        "expected_quality_score": ">= 80",
        "expected_confirmed": True,
        "expected_condition": "Cardiac Failure"
    }
]


def print_separator(title: str = ""):
    """Print a formatted separator"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
    else:
        print(f"{'='*80}")


def analyze_note(clinical_note: str) -> Dict:
    """Send clinical note to backend for analysis"""
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"clinical_note": clinical_note},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error analyzing note: {e}")
        return None


def evaluate_quality_score(score: int, expected: str) -> bool:
    """Check if quality score matches expectations"""
    if ">=" in expected:
        threshold = int(expected.split(">=")[1].strip())
        return score >= threshold
    elif "<" in expected and "=" not in expected:
        threshold = int(expected.split("<")[1].strip())
        return score < threshold
    elif "-" in expected:
        low, high = map(int, expected.split("-"))
        return low <= score <= high
    return False


def test_note_completeness(result: Dict, expected_quality: str) -> bool:
    """Test Feature #1: Note Completeness Validation"""
    print("\n📋 Testing Note Completeness Validation:")
    
    if not result or "note_quality" not in result:
        print("  ❌ Note quality data missing")
        return False
    
    quality = result["note_quality"]
    score = quality["completeness_score"]
    missing = quality["missing_elements"]
    warnings = quality["warnings"]
    
    print(f"  Score: {score}/100")
    print(f"  Missing Elements: {len(missing)}")
    print(f"  Warnings: {len(warnings)}")
    
    if missing:
        print(f"    Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
    if warnings:
        print(f"    Warnings: {warnings[0] if warnings else ''}")
    
    passed = evaluate_quality_score(score, expected_quality)
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}: Quality score matches expectation ({expected_quality})")
    
    return passed


def test_keyword_transparency(result: Dict) -> bool:
    """Test Feature #2: AI Transparency with Keyword Matches"""
    print("\n🔍 Testing Keyword Transparency:")
    
    if not result or "matched_conditions" not in result:
        print("  ❌ Matched conditions data missing")
        return False
    
    conditions = result["matched_conditions"]
    if not conditions:
        print("  ❌ No conditions matched")
        return False
    
    passed = True
    for i, condition in enumerate(conditions[:2], 1):  # Check first 2 conditions
        has_explanation = bool(condition.get("match_explanation"))
        has_keywords = bool(condition.get("triggering_keywords"))
        
        print(f"\n  Condition {i}: {condition['condition']}")
        print(f"    Match Type: {'CONFIRMED' if condition.get('is_confirmed') else 'SUGGESTED'}")
        print(f"    Explanation: {condition.get('match_explanation', 'N/A')}")
        
        if has_keywords and condition["triggering_keywords"]:
            keywords = condition["triggering_keywords"][:3]
            print(f"    Top Keywords: {', '.join([f\"{kw['keyword']} ({kw['similarity_score']:.2f})\" for kw in keywords])}")
        else:
            print(f"    Keywords: None provided")
        
        if not has_explanation:
            print(f"    ⚠️ Missing match explanation")
            passed = False
    
    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Transparency data available")
    return passed


def test_icd_suggestions(result: Dict, expected_confirmed: bool) -> bool:
    """Test Feature #3: Automatic ICD Code Suggestions"""
    print("\n💊 Testing ICD Code Auto-Suggestion:")
    
    if not result or "matched_conditions" not in result:
        print("  ❌ Matched conditions data missing")
        return False
    
    conditions = result["matched_conditions"]
    if not conditions:
        print("  ❌ No conditions matched")
        return False
    
    passed = True
    for i, condition in enumerate(conditions[:2], 1):
        has_suggestion = bool(condition.get("suggested_icd_code"))
        confidence = condition.get("icd_confidence", 0)
        alternatives = condition.get("alternative_icd_codes", [])
        
        print(f"\n  Condition {i}: {condition['condition']}")
        print(f"    Current ICD: {condition['icd_code']}")
        print(f"    Suggested ICD: {condition.get('suggested_icd_code', 'N/A')}")
        print(f"    Confidence: {confidence:.2%}" if confidence else "    Confidence: N/A")
        print(f"    Alternatives: {len(alternatives)} options")
        
        if not has_suggestion:
            print(f"    ⚠️ No ICD suggestion provided")
            passed = False
        elif confidence < 0.5:
            print(f"    ⚠️ Low confidence in suggestion")
    
    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: ICD suggestions available")
    return passed


def run_test_case(test_case: Dict) -> Dict:
    """Run a single test case"""
    print_separator(f"TEST: {test_case['name']}")
    
    print(f"\nClinical Note Preview:")
    preview = test_case['clinical_note'][:150] + "..." if len(test_case['clinical_note']) > 150 else test_case['clinical_note']
    print(f"  {preview}")
    
    # Analyze the note
    print(f"\n⚙️  Analyzing note...")
    result = analyze_note(test_case['clinical_note'])
    
    if not result:
        return {"passed": False, "reason": "Analysis failed"}
    
    # Test all three features
    results = {
        "completeness": test_note_completeness(result, test_case['expected_quality_score']),
        "transparency": test_keyword_transparency(result),
        "icd_suggestion": test_icd_suggestions(result, test_case['expected_confirmed'])
    }
    
    all_passed = all(results.values())
    
    # Summary
    print(f"\n{'='*40}")
    print(f"  Test Summary: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    print(f"{'='*40}")
    
    return {"passed": all_passed, "details": results}


def main():
    """Run all test cases"""
    print_separator("AUTHI 1.0 TRUST & TRANSPARENCY ENHANCEMENTS - TEST SUITE")
    
    print("\n🎯 Testing Backend Endpoint Health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend: {e}")
        print(f"   Make sure the Python backend is running on {BASE_URL}")
        return
    
    # Run all test cases
    results = []
    for test_case in TEST_CASES:
        result = run_test_case(test_case)
        results.append({
            "name": test_case["name"],
            "passed": result["passed"]
        })
    
    # Final summary
    print_separator("FINAL TEST RESULTS")
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {sum(1 for r in results if r['passed'])}")
    print(f"Failed: {sum(1 for r in results if not r['passed'])}")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status}: {result['name']}")
    
    print_separator()
    
    all_passed = all(r["passed"] for r in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Enhancements are working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review the output above for details.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
