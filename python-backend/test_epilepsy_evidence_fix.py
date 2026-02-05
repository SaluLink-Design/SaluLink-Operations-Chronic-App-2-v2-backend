"""
Test the Evidence-Based Filtering Fix
Tests the specific epilepsy case that was triggering false hypertension detection
"""

import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

# The epilepsy clinical note that was causing false positives
EPILEPSY_NOTE = """Chief Complaint:
Recurrent seizure episodes over the past 8 months.

History of Present Illness:
Patient reports experiencing sudden episodes of loss of consciousness associated with generalized body stiffening followed by rhythmic jerking movements lasting approximately 1–2 minutes. Episodes are occasionally preceded by an aura described as a rising sensation in the stomach and brief confusion. Postictal phase includes fatigue, headache, and disorientation lasting up to 30 minutes.

Seizure frequency has increased from one episode every three months to approximately two episodes per month. Most recent seizure occurred at home and was witnessed by a family member who reported tongue biting and urinary incontinence.

Patient denies recent head trauma, fever, or substance use. Reports poor sleep and increased academic stress as possible triggers."""


def test_epilepsy_note():
    """Test that the epilepsy note correctly excludes hypertension"""
    print("\n" + "="*80)
    print("TESTING: Evidence-Based Filtering - Epilepsy Note")
    print("="*80)
    print("\nClinical Note (excerpt):")
    print(EPILEPSY_NOTE[:200] + "...")
    print("\n" + "-"*80)
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"clinical_note": EPILEPSY_NOTE},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Analysis completed successfully")
            print(f"Keywords Extracted: {len(result['extracted_keywords'])}")
            print(f"Confirmed Conditions: {result['confirmed_count']}")
            print(f"Total Conditions Detected: {len(result['matched_conditions'])}")
            print(f"Note Quality Score: {result['note_quality']['completeness_score']}/100")
            
            print(f"\n{'-'*80}")
            print("CONDITIONS DETECTED:")
            print(f"{'-'*80}\n")
            
            epilepsy_found = False
            hypertension_found = False
            
            for i, condition in enumerate(result['matched_conditions'], 1):
                status = "✓ CONFIRMED" if condition['is_confirmed'] else "→ SUGGESTED"
                evidence = condition.get('evidence_level', 'unknown')
                evidence_score = condition.get('evidence_score', 0.0)
                
                print(f"{i}. [{status}] {condition['condition']}")
                print(f"   Confidence: {condition['similarity_score']:.3f}")
                print(f"   Evidence Level: {evidence} (score: {evidence_score:.2f})")
                print(f"   ICD Code: {condition['icd_code']}")
                
                if condition.get('missing_evidence'):
                    print(f"   Missing Evidence: {', '.join(condition['missing_evidence'][:3])}")
                
                print(f"   Explanation: {condition['match_explanation']}\n")
                
                # Check for specific conditions
                if 'Epilepsy' in condition['condition']:
                    epilepsy_found = True
                if 'Hypertension' in condition['condition']:
                    hypertension_found = True
            
            # Validate results
            print(f"\n{'='*80}")
            print("TEST RESULTS:")
            print(f"{'='*80}\n")
            
            if epilepsy_found:
                print("✅ PASS: Epilepsy correctly detected")
            else:
                print("❌ FAIL: Epilepsy was NOT detected")
            
            if not hypertension_found:
                print("✅ PASS: Hypertension correctly excluded (no BP readings in note)")
            else:
                print("❌ FAIL: Hypertension was incorrectly detected (FALSE POSITIVE)")
            
            print(f"\n{'='*80}")
            
            if epilepsy_found and not hypertension_found:
                print("🎉 SUCCESS: Evidence-based filtering is working correctly!")
                print("   - Epilepsy detected (strong evidence: seizures, postictal, etc.)")
                print("   - Hypertension excluded (insufficient evidence: no BP readings)")
                return True
            else:
                print("⚠️  ISSUE: Evidence-based filtering needs adjustment")
                return False
                
        else:
            print(f"❌ ERROR: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend")
        print("Please start the backend first: python3 main.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_hypertension_with_evidence():
    """Test that hypertension IS detected when BP readings are present"""
    print("\n\n" + "="*80)
    print("CONTROL TEST: Hypertension WITH BP Readings")
    print("="*80)
    
    htn_note = """Patient presents with elevated blood pressure. 
    BP reading: 165/95 mmHg. Patient reports headaches and dizziness.
    Previous BP readings also elevated (160/92, 158/90).
    Started on amlodipine 5mg daily."""
    
    print("\nClinical Note:")
    print(htn_note)
    print("\n" + "-"*80)
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"clinical_note": htn_note},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Analysis completed")
            print(f"Conditions Detected: {len(result['matched_conditions'])}\n")
            
            hypertension_found = False
            
            for condition in result['matched_conditions']:
                if 'Hypertension' in condition['condition']:
                    hypertension_found = True
                    evidence = condition.get('evidence_level', 'unknown')
                    evidence_score = condition.get('evidence_score', 0.0)
                    
                    print(f"✓ Hypertension detected")
                    print(f"  Confidence: {condition['similarity_score']:.3f}")
                    print(f"  Evidence Level: {evidence} (score: {evidence_score:.2f})")
                    print(f"  Status: {'CONFIRMED' if condition['is_confirmed'] else 'SUGGESTED'}")
            
            print(f"\n{'='*80}")
            if hypertension_found:
                print("✅ CONTROL TEST PASS: Hypertension detected WITH BP evidence")
                return True
            else:
                print("❌ CONTROL TEST FAIL: Hypertension NOT detected despite BP readings")
                return False
        else:
            print(f"❌ ERROR: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EVIDENCE-BASED FILTERING TEST SUITE")
    print("Testing the fix for false comorbidity detection")
    print("="*80)
    
    # Check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("\n❌ Backend health check failed!")
            print("Please start the backend: cd python-backend && python3 main.py")
            exit(1)
        print("\n✓ Backend is running and healthy\n")
    except:
        print("\n❌ Cannot connect to backend!")
        print("Please start the backend: cd python-backend && python3 main.py")
        exit(1)
    
    # Run tests
    test1_passed = test_epilepsy_note()
    test2_passed = test_hypertension_with_evidence()
    
    # Summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Test 1 (Epilepsy - exclude HTN): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (HTN with BP readings): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Evidence-based filtering is working correctly.")
        print("\nKey improvements:")
        print("  ✓ False positives eliminated (HTN not detected in epilepsy notes)")
        print("  ✓ True positives preserved (HTN detected when BP readings present)")
        print("  ✓ Transparent evidence levels shown for all conditions")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review results above")
    
    print("="*80 + "\n")
