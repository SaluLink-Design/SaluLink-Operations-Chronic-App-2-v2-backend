# Testing Guide - Condition Detection

## Quick Start

### 1. Start the Backend
```bash
cd python-backend
python main.py
```

The backend will start on `http://localhost:8000`

### 2. Run Tests

#### Option A: Test the Specific Cardiomyopathy Issue
```bash
# In a new terminal
cd python-backend
python test_cardiomyopathy.py
```

**Expected Output:**
```
✅ SUCCESS: Cardiomyopathy was correctly identified!
```

#### Option B: Comprehensive Test (All Conditions)
```bash
cd python-backend
python test_condition_detection.py
```

**Expected Output:**
```
Total tests run: 27
Tests passed: 27 (100.0%)
🎉 All tests passed! All conditions can be detected correctly.
```

### 3. Test via API Directly

#### Using curl:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"clinical_note": "patient has been diagnosed with cardiomyopathy"}'
```

#### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"clinical_note": "patient has been diagnosed with cardiomyopathy"}
)

data = response.json()
print(f"Conditions: {[c['condition'] for c in data['matched_conditions']]}")
```

## Test Cases for Each Condition

### Cardiomyopathy
```
"patient has been diagnosed with cardiomyopathy"
"dilated cardiomyopathy with reduced ejection fraction"
"hypertrophic cardiomyopathy causing chest pain"
```

### Asthma
```
"patient diagnosed with asthma"
"allergic asthma with wheezing"
"status asthmaticus requiring treatment"
```

### Diabetes Type 1
```
"patient has type 1 diabetes mellitus"
"insulin-dependent diabetes with ketoacidosis"
"diagnosed with diabetes type 1"
```

### Diabetes Type 2
```
"patient diagnosed with type 2 diabetes"
"non-insulin-dependent diabetes mellitus"
"diabetes type 2 managed with metformin"
```

### Hypertension
```
"patient diagnosed with hypertension"
"essential hypertension, elevated blood pressure"
"high blood pressure requiring treatment"
```

### Chronic Renal Disease
```
"patient diagnosed with chronic renal disease"
"chronic kidney disease stage 3"
"end-stage renal disease requiring dialysis"
```

### Cardiac Failure
```
"patient diagnosed with cardiac failure"
"congestive heart failure with leg swelling"
"heart failure with reduced ejection fraction"
```

### Hyperlipidaemia
```
"patient diagnosed with hyperlipidaemia"
"mixed hyperlipidaemia with high cholesterol"
"elevated cholesterol requiring statin"
```

### Haemophilia
```
"patient diagnosed with haemophilia"
"factor VIII deficiency causing bleeding"
"hemophilia with joint bleeds"
```

## Verifying the Fix

### The Original Issue:
```python
# This was failing before:
Input: "patient has been diagnosed with cardiomyopathy"
Result: ❌ Cardiomyopathy NOT in results
```

### After the Fix:
```python
# Now working:
Input: "patient has been diagnosed with cardiomyopathy"
Result: ✅ Cardiomyopathy detected with score 0.95
```

## What to Look For

### ✅ Success Indicators:
1. The condition mentioned is in the top 3-5 results
2. High similarity score (0.85-0.95 for direct matches)
3. Correct ICD codes returned
4. Console logs show "Direct match found" or "Keyword match found"

### ❌ Failure Indicators:
1. Mentioned condition not in results
2. Wrong conditions returned
3. Empty results
4. API errors

## Troubleshooting

### Backend Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Reinstall dependencies
pip install -r requirements.txt
```

### Model Download Issues
```bash
# The first run downloads ClinicalBERT (~400MB)
# This may take a few minutes
# Check console for "Model loaded successfully!"
```

### Import Errors
```bash
# Ensure you're in the right directory
cd python-backend

# Check Python version (requires 3.8+)
python --version

# Verify all packages installed
pip list | grep -E "fastapi|torch|transformers"
```

## Integration Testing

### Test with Frontend
1. Start backend: `cd python-backend && python main.py`
2. Start frontend: `cd .. && npm run dev`
3. Open `http://localhost:3000`
4. Enter: "patient has been diagnosed with cardiomyopathy"
5. Click "Analyze with AI"
6. Verify Cardiomyopathy appears in results

## Performance Notes

- First analysis: ~5-10 seconds (model initialization)
- Subsequent analyses: ~1-2 seconds
- Memory usage: ~2GB (ClinicalBERT model)
- CPU: Efficient on standard hardware (no GPU required)

## Deployment Testing

### Railway Deployment
```bash
# After deploying to Railway, test the live endpoint:
curl -X POST https://your-app.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"clinical_note": "patient has been diagnosed with cardiomyopathy"}'
```

### Health Check
```bash
# Verify backend is healthy
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "conditions_loaded": true
}
```

## Monitoring

### Check Logs
The backend logs each analysis with details:
```
Analysis completed: 8 keywords extracted, 5 conditions matched
   Direct match found: Cardiomyopathy
  1. Cardiomyopathy (I42.0) - Score: 0.950
  2. Cardiac Failure (I50.0) - Score: 0.782
  3. Hypertension (I10) - Score: 0.711
```

Look for:
- "Direct match found" for explicit condition mentions
- "Keyword match found" for medical terminology
- Score values (higher = better confidence)

## Summary

✅ **The fix ensures that:**
1. All explicitly mentioned conditions are detected
2. Medical terminology is recognized
3. Semantic understanding works for descriptions
4. Results are consistent and reliable

🚀 **Ready to test!** Run `python test_cardiomyopathy.py` to verify the fix.



