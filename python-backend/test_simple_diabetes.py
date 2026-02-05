import requests

note = "Patient presents with chest pain. No history of diabetes or hypertension. Blood pressure 120/75, normal. Glucose 95 mg/dL, normal."

print("Testing note:")
print(note)
print()

response = requests.post("http://localhost:8000/analyze", json={"clinical_note": note})
result = response.json()

print(f"Detected {len(result['matched_conditions'])} conditions:")
for c in result['matched_conditions']:
    print(f"  - {c['condition']}: {c['similarity_score']*100:.1f}% ({'CONFIRMED' if c['is_confirmed'] else 'suggested'})")

print("\nExpected: Neither Diabetes nor Hypertension should appear (both are negated)")
