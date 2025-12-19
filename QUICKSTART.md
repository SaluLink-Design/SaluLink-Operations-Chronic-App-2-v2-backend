# Quick Start Guide - SaluLink Chronic App

## 🚀 Get Started in 5 Minutes

### Step 1: Install Node.js Dependencies

```bash
npm install
```

### Step 2: Setup Python Backend

```bash
# Navigate to Python backend folder
cd python-backend

# Create virtual environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# On Windows use:
# venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Go back to root
cd ..
```

### Step 3: Start Both Servers

**Terminal 1 - Python Backend:**
```bash
cd python-backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py
```

Wait for: `INFO: Application startup complete.`

**Terminal 2 - Next.js Frontend:**
```bash
npm run dev
```

Wait for: `Ready - started server on 0.0.0.0:3000`

### Step 4: Open the App

Open your browser to: **http://localhost:3000**

---

## 🎯 Try It Out

### Test with Sample Clinical Note

```
Patient presents with persistent wheezing and shortness of breath that worsens with exercise. 
History of allergic rhinitis. Chest tightness noted during physical examination. 
Symptoms have been occurring for the past 3 months, particularly during cold weather exposure.
Peak flow measurements show reduced lung capacity.
```

### Workflow Steps:

1. **Paste the note** into the Clinical Note text area
2. Click **"Analyze Note"** (wait ~3-5 seconds)
3. **Select "Asthma"** from the matched conditions
4. **Choose an ICD-10 code** (e.g., J45.9 - Asthma, unspecified)
5. **Add diagnostic tests** from the basket (e.g., Peak flow, Flow volume test)
6. **Document results** for each test
7. **Select a medical plan** (e.g., Core)
8. **Choose medications** from the filtered list
9. **Write a registration note**
10. Click **"Save Case"** and enter patient details
11. Click **"Export PDF"** to generate the claim

---

## 📋 Common Issues & Solutions

### Issue: "Failed to analyze note"
**Solution**: Make sure Python backend is running on port 8000
```bash
# Check if running
curl http://localhost:8000/health
```

### Issue: CSV files not loading
**Solution**: Verify files are in public folder
```bash
ls public/*.csv
# Should show:
# public/Chronic Conditions.csv
# public/Medicine List.csv
# public/Treatment Basket.csv
```

### Issue: Module not found errors
**Solution**: Reinstall dependencies
```bash
rm -rf node_modules
npm install
```

---

## 🎨 Key Features to Try

### 1. Case Management
- Click the **menu icon** (top right) to view saved cases
- Load a saved case to see the full history
- Try the case actions: Ongoing Management, Medication Report, Referral

### 2. Medical Plan Filtering
- Switch between different plans (Core, Executive, etc.)
- Notice how medication options and CDA amounts change

### 3. PDF Export
- Export at any stage to get claim documentation
- Each workflow type (diagnostic, ongoing, referral) has its own PDF format

### 4. AI Analysis
- Try different clinical notes to see AI matching
- The system shows confidence scores for each match
- Top 5 most relevant conditions are displayed

---

## 📊 Project Structure

```
Chronic App 2/
├── app/                    # Next.js app
│   ├── api/               # API routes
│   ├── page.tsx           # Main app page
│   └── globals.css        # Global styles
├── components/            # React components
├── lib/                   # Utilities
├── python-backend/        # Python/FastAPI backend
├── public/               # Static files (CSVs)
└── types/                # TypeScript types
```

---

## 🔧 Development Scripts

```bash
# Frontend only
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Python backend
npm run python-api  # or: cd python-backend && python main.py
```

---

## ✅ Feature Checklist

- [x] AI-powered clinical note analysis
- [x] Condition matching with ClinicalBERT
- [x] ICD-10 code selection
- [x] Diagnostic basket management
- [x] Medication prescription with plan filtering
- [x] Case saving and management
- [x] PDF export for all workflows
- [x] Ongoing management tracking
- [x] Medication reports
- [x] Specialist referrals
- [x] Local data persistence

---

## 📞 Need Help?

1. Check the **README.md** for detailed documentation
2. Review the **Troubleshooting** section
3. Ensure both servers are running
4. Check browser console for errors (F12)

---

## 🎉 You're All Set!

The app is now ready to use. Try creating your first chronic condition case and explore all the features!

