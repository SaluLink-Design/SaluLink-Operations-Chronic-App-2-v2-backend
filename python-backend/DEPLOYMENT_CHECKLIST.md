# ✅ Railway Deployment Checklist

## Pre-Deployment Setup (COMPLETED ✓)

- [x] **CSV File Copied**: `Chronic Conditions.csv` is now in `python-backend/`
- [x] **Code Updated**: `main.py` now looks for CSV in local directory first
- [x] **Railway Config**: `railway.json`, `Procfile`, and `runtime.txt` created
- [x] **Port Configuration**: App reads PORT from environment variable
- [x] **Fallback Logic**: Code works both locally and on Railway

## Files Ready for Deployment

```
python-backend/
├── Chronic Conditions.csv      ✓ (23 KB)
├── main.py                      ✓ (Updated with local CSV path)
├── requirements.txt             ✓ (All dependencies listed)
├── railway.json                 ✓ (Railway configuration)
├── Procfile                     ✓ (Start command)
├── runtime.txt                  ✓ (Python 3.11)
├── .gitignore                   ✓ (Ensures CSV is tracked)
└── README.md                    ✓ (Documentation)
```

## Deploy to Railway - Next Steps

### Option 1: GitHub Deployment (Recommended)

1. **Commit and push your changes**:
   ```bash
   git add python-backend/
   git commit -m "Add Railway deployment configuration and CSV file"
   git push origin main
   ```

2. **Deploy on Railway**:
   - Go to [railway.app/new](https://railway.app/new)
   - Click "Deploy from GitHub repo"
   - Select your repository
   - **Important**: Set **Root Directory** to `python-backend`
   - Click "Deploy"

3. **Monitor deployment**:
   - Watch the build logs
   - First deployment will take 5-10 minutes (downloading ClinicalBERT model)
   - Look for "Loading chronic conditions..." and "Model loaded successfully!"

4. **Get your URL**:
   - Railway will provide a URL like: `https://your-app.up.railway.app`
   - Test it: `https://your-app.up.railway.app/health`

### Option 2: Railway CLI

```bash
# Install CLI (if not already installed)
npm install -g @railway/cli

# Login
railway login

# Navigate to backend
cd python-backend

# Initialize and deploy
railway init
railway up

# Get your URL
railway domain
```

## Testing Your Deployment

Once deployed, test these endpoints:

```bash
# Replace YOUR_URL with your actual Railway URL

# 1. Health check
curl https://YOUR_URL/health

# Expected response:
# {"status":"healthy","model_loaded":true,"conditions_loaded":true}

# 2. Root endpoint
curl https://YOUR_URL/

# Expected response:
# {"message":"SaluLink Authi API is running"}

# 3. Analysis endpoint
curl -X POST https://YOUR_URL/analyze \
  -H "Content-Type: application/json" \
  -d '{"clinical_note":"Patient presents with chest pain and shortness of breath"}'

# Expected: JSON with extracted_keywords and matched_conditions
```

## Environment Variables (Optional)

If you need to add any environment variables in Railway:

1. Go to your service in Railway dashboard
2. Click on "Variables" tab
3. Add variables as needed (currently none required)

## Connecting Your Next.js Frontend

Once deployed, update your frontend to use the Railway backend:

### 1. Create/Update `.env.local` in your Next.js root:

```env
NEXT_PUBLIC_API_URL=https://your-app.up.railway.app
```

### 2. Update your API calls:

```javascript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Example usage
const analyzeNote = async (clinicalNote) => {
  const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clinical_note: clinicalNote })
  });
  return response.json();
};
```

## Troubleshooting

### If deployment fails:

1. **Check build logs** in Railway dashboard
2. **Verify Root Directory** is set to `python-backend`
3. **Check CSV file** is in the repository and tracked by git
4. **Memory issues**: Your Railway plan should handle this (you mentioned you have a plan)

### If app crashes on startup:

1. Check Railway logs for errors
2. Look for "Loading CSV from:" message
3. Verify model download completed successfully

### If CSV not found:

```bash
# Verify CSV is in git
git ls-files python-backend/

# Should show:
# python-backend/Chronic Conditions.csv
```

## Expected Startup Logs

When your app starts successfully, you should see:

```
Loading ClinicalBERT model...
Model loaded successfully!
Loading chronic conditions...
Loading CSV from: /app/Chronic Conditions.csv
Loaded 123 chronic condition entries  # (actual number may vary)
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:PORT
```

## Performance Notes

- **Cold Start**: First request after idle may take 10-30 seconds
- **Model Loading**: Happens once on startup (~30 seconds)
- **CSV Loading**: Happens once on startup (~1-2 seconds)
- **Subsequent Requests**: Should be fast (<1 second)

---

## 🎉 You're Ready to Deploy!

Everything is configured and ready. Just push to GitHub and deploy on Railway!

**Questions?** Check the detailed `RAILWAY_DEPLOYMENT.md` guide or let me know if you need help!
