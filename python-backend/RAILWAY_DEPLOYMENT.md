# 🚀 Railway Deployment Guide

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### Option 1: Deploy from GitHub (Recommended)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Go to Railway Dashboard**:
   - Visit [railway.app/new](https://railway.app/new)
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect it's a Python project

3. **Configure the Service**:
   - Railway should auto-detect your `requirements.txt`
   - Set the **Root Directory** to `python-backend` (important!)
   - Railway will use the `Procfile` and `railway.json` automatically

4. **Add Environment Variables** (if needed):
   - Go to your service → Variables tab
   - Add any custom environment variables your app needs

5. **Deploy**:
   - Railway will automatically build and deploy
   - You'll get a public URL like `https://your-app.railway.app`

### Option 2: Deploy using Railway CLI

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   # or
   brew install railway
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize Railway project** (from python-backend directory):
   ```bash
   cd python-backend
   railway init
   ```

4. **Deploy**:
   ```bash
   railway up
   ```

5. **Get your deployment URL**:
   ```bash
   railway domain
   ```

## Important Configuration Notes

### 1. CSV Files
Your backend needs access to `Chronic Conditions.csv` which is in the parent directory. You have two options:

**Option A: Copy CSV to python-backend** (Simplest)
```bash
cp "../Chronic Conditions.csv" ./
```
Then update `main.py` line 65:
```python
csv_path = Path(__file__).parent / "Chronic Conditions.csv"
```

**Option B: Include parent directory in deployment**
- Deploy from the root directory instead
- Update Railway root directory setting
- Keep the current path logic

### 2. Model Download
The ClinicalBERT model (`emilyalsentzer/Bio_ClinicalBERT`) will be downloaded on first startup. This means:
- First deployment will take longer (~5-10 minutes)
- Requires ~500MB-1GB of storage
- Railway's free tier should handle this

### 3. Memory Requirements
Your app uses PyTorch and transformers, which are memory-intensive:
- **Minimum**: 2GB RAM
- **Recommended**: 4GB RAM
- Railway's free tier provides 512MB-1GB (may need to upgrade)

### 4. CORS Configuration
Your backend already has CORS enabled for all origins:
```python
allow_origins=["*"]
```

For production, update this to your specific frontend URL:
```python
allow_origins=["https://your-frontend-domain.com"]
```

## Testing Your Deployment

Once deployed, test your API:

```bash
# Health check
curl https://your-app.railway.app/health

# Test analysis endpoint
curl -X POST https://your-app.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"clinical_note": "Patient presents with chest pain and shortness of breath"}'
```

## Connecting Your Frontend

Update your Next.js frontend to use the Railway backend URL:

```javascript
// In your Next.js app
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://your-backend.railway.app';

const response = await fetch(`${API_URL}/analyze`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ clinical_note: note })
});
```

Add to your `.env.local`:
```
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

## Monitoring & Logs

- View logs in Railway dashboard
- Monitor resource usage
- Set up alerts for errors

## Cost Estimates

Railway pricing (as of 2024):
- **Free Tier**: $5 credit/month (limited resources)
- **Hobby Plan**: $5/month base + usage
- **Pro Plan**: $20/month base + usage

Your app will likely need the Hobby plan due to memory requirements.

## Troubleshooting

### Build Fails
- Check Railway build logs
- Ensure `requirements.txt` is correct
- Verify Python version compatibility

### App Crashes on Startup
- Check if CSV file is accessible
- Monitor memory usage (may need upgrade)
- Review startup logs for model loading errors

### Slow Response Times
- First request after idle will be slow (cold start)
- Model loading takes time on startup
- Consider keeping service warm with periodic health checks

## Alternative: Copy CSV File Approach

For the simplest deployment, I recommend copying the CSV file:

```bash
cd python-backend
cp "../Chronic Conditions.csv" ./
```

This avoids path issues during deployment.

---

**Need help?** Let me know which deployment option you'd like to use, and I can guide you through it step by step!
