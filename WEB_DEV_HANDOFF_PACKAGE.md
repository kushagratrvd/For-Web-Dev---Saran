# Web Developer Handoff Package

## Complete Package for Dashboard Deployment

---

## What You're Getting

### **Files in This Package:**

```
web_deployment/
├── MODELS (Download from Kaggle after running save script):
│   ├── random_forest_production.pkl         ⭐ BEST (5.14% sMAPE)
│   ├── tuned_lightgbm_production.txt        ⭐ BEST (5.22% sMAPE)
│   ├── lightgbm_model.txt                   (Original, 8.11% sMAPE)
│   └── hybrid_model.pkl                     (Backup, 10.35% sMAPE)
│
├── DATA (Download from Kaggle):
│   ├── consumer_behavior_electronics_processed_complete.csv
│   └── brand_product_catalog.csv
│
├── CODE:
│   ├── electronics_dashboard.py             Main dashboard
│   └── requirements.txt                     Dependencies
│
└── DOCUMENTATION:
    ├── WEB_DEV_HANDOFF_PACKAGE.md          This file
    └── CLIENT_PRESENTATION_SUMMARY.md       For reference
```

---

## What Client Wants (Requirements)

### **1. Brand Search Functionality** ✓
- **Input Field**: User types brand name
  - Options: Samsung, Apple, Sony, LG, HP
  - Dropdown or text search

### **2. Timeline Selection** ✓
- **Dropdown**: 1, 2, or 3 months
- User selects forecast horizon

### **3. Predictions Display** ✓
- Show ALL products for selected brand
- For each product show:
  - Product name
  - Predictions for next N months
  - Trend indicator (up/down)
  - Visual charts

### **Example User Flow:**
```
User opens dashboard
  ↓
Types: "Samsung" (or selects from dropdown)
  ↓
Selects: "3 months"
  ↓
Clicks: "PREDICT" button
  ↓
Sees: 
  - Samsung Galaxy S24: [25,000 → 26,000 → 24,000] units
  - Samsung QLED TV: [8,000 → 8,500 → 7,500] units
  - Samsung Tab: [6,000 → 6,200 → 5,800] units
  - (Charts and visualizations)
```

---

## Models to Deploy

### **Option A: Deploy Both Best Models** ⭐ RECOMMENDED

**Why deploy both:**
- Random Forest: Best accuracy (5.14% sMAPE)
- Tuned LightGBM: Best direction (88.63%), faster predictions
- Let users choose or show both predictions

**Implementation:**
```python
# In dashboard, add model selector
model_choice = st.selectbox(
    "Select Model",
    ["Random Forest (Best Accuracy)", 
     "Tuned LightGBM (Best Direction)",
     "Show Both"]
)
```

### **Option B: Deploy Only Random Forest**

**Simplest approach:**
- Single model, best overall performance
- 5.14% sMAPE, 87.63% direction
- Easiest for users

---

## Installation Instructions

### **Step 1: Set Up Environment**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Prepare Files**

```
project/
├── electronics_dashboard.py
├── requirements.txt
├── models/
│   ├── random_forest_production.pkl
│   └── tuned_lightgbm_production.txt
└── data/
    ├── consumer_behavior_electronics_processed_complete.csv
    └── brand_product_catalog.csv
```

### **Step 3: Test Locally**

```bash
streamlit run electronics_dashboard.py
```

Open browser to `http://localhost:8501`

---

## Dashboard Modifications Needed

### **Current Dashboard:**
The provided `electronics_dashboard.py` has basic structure.

### **You Need to Update:**

**1. Model Loading Section:**
```python
@st.cache_resource
def load_models():
    models = {}
    
    # Load Random Forest
    with open('models/random_forest_production.pkl', 'rb') as f:
        models['random_forest'] = pickle.load(f)
    
    # Load Tuned LightGBM
    import lightgbm as lgb
    models['tuned_lgb'] = lgb.Booster(model_file='models/tuned_lightgbm_production.txt')
    
    return models
```

**2. Prediction Function:**
```python
def make_predictions(brand, months, model_type='random_forest'):
    # Get products for brand
    brand_products = df[df['brand'] == brand]['product'].unique()
    
    predictions = []
    for product in brand_products:
        # Get features for this product
        product_data = df[(df['brand'] == brand) & (df['product'] == product)]
        
        # Use latest data point features
        latest_features = product_data[feature_cols].iloc[-1:].values
        
        # Predict using selected model
        if model_type == 'random_forest':
            pred = models['random_forest'].predict(latest_features)[0]
        else:
            pred = models['tuned_lgb'].predict(latest_features)[0]
        
        predictions.append({
            'product': product,
            'prediction': pred
        })
    
    return predictions
```

**3. User Interface:**
```python
# Sidebar inputs
st.sidebar.header("Prediction Parameters")

brand = st.sidebar.selectbox(
    "Select Brand",
    ["Samsung", "Apple", "Sony", "LG", "HP"]
)

timeline = st.sidebar.selectbox(
    "Forecast Timeline",
    [1, 2, 3],
    format_func=lambda x: f"{x} month(s)"
)

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Tuned LightGBM", "Both"]
)

if st.sidebar.button("GENERATE PREDICTIONS", type="primary"):
    # Make predictions and display
    ...
```

---

## Deployment Options

### **Option 1: Streamlit Cloud** ⭐ EASIEST

**Steps:**
1. Push code to GitHub repository
2. Go to https://streamlit.io/cloud
3. Connect GitHub account
4. Select repository
5. Choose `electronics_dashboard.py`
6. Deploy

**Requirements:**
- GitHub repository (public or private)
- Model files must be < 100MB each
- Free tier available

**URL:** You get `https://yourapp.streamlit.app`

---

### **Option 2: AWS/Azure** (Production)

**AWS Elastic Beanstalk:**
```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 electronics-dashboard

# Create environment
eb create electronics-dashboard-env

# Deploy
eb deploy
```

**Azure App Service:**
```bash
# Install Azure CLI
az login

# Create app
az webapp up --name electronics-dashboard --runtime "PYTHON:3.10"
```

---

### **Option 3: Docker** (Any Platform)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "electronics_dashboard.py", "--server.port=8501"]
```

**Run:**
```bash
docker build -t electronics-dashboard .
docker run -p 8501:8501 electronics-dashboard
```

---

## Testing Checklist

Before deploying, test:

- [ ] Dashboard loads without errors
- [ ] All brands appear in dropdown
- [ ] Timeline selection works (1, 2, 3 months)
- [ ] Predictions generate when button clicked
- [ ] All products for brand show up
- [ ] Charts display correctly
- [ ] No negative predictions
- [ ] Predictions are reasonable (not 0 or millions)
- [ ] Model selector works (if using both models)
- [ ] Page loads in < 5 seconds

---

## Performance Specifications

### **Models:**

| Model | sMAPE | Direction | Speed | Recommendation |
|-------|-------|-----------|-------|----------------|
| Random Forest | 5.14% | 87.63% | Medium | Best accuracy |
| Tuned LightGBM | 5.22% | 88.63% | Fast | Best direction, faster |
| Original LightGBM | 8.11% | 85.95% | Fast | Backup |

### **Expected Prediction Time:**
- Single brand (5 products): < 1 second
- All brands: < 3 seconds

---

## Troubleshooting

### **Issue: Models not loading**
```python
# Check file paths
import os
print(os.listdir('models/'))

# Check model files exist
print(os.path.exists('models/random_forest_production.pkl'))
```

### **Issue: Predictions return NaN**
```python
# Check feature columns match
print("Model features:", len(feature_cols))
print("Data features:", len(df.columns))
```

### **Issue: Dashboard crashes**
```python
# Check Streamlit version
pip show streamlit

# Reinstall if needed
pip install streamlit==1.31.0
```

---

## Support & Contact

### **For Technical Issues:**
- Check model files are in correct location
- Verify all dependencies installed
- Review error messages in terminal

### **For Dashboard Functionality:**
- Refer to Streamlit documentation: https://docs.streamlit.io
- Example apps: https://streamlit.io/gallery

### **For Model Questions:**
- Random Forest: scikit-learn.org
- LightGBM: lightgbm.readthedocs.io

---

## Success Criteria

Dashboard is ready when:

✓ User can search by brand
✓ User can select timeline (1-3 months)
✓ Predictions display for all brand products
✓ Charts show trends
✓ Performance < 5 seconds
✓ No errors in console
✓ Works on mobile/tablet
✓ Deployed and accessible via URL

---

## Timeline

**Day 1:** Set up environment, test locally
**Day 2:** Modify dashboard for client requirements
**Day 3:** Test all functionality
**Day 4:** Deploy to Streamlit Cloud
**Day 5:** User acceptance testing

**Total:** 1 week for complete deployment

---

## Files Summary

**What you MUST download from Kaggle:**
1. `random_forest_production.pkl` (after running save script)
2. `tuned_lightgbm_production.txt` (after running save script)
3. `consumer_behavior_electronics_processed_complete.csv`
4. `brand_product_catalog.csv`

**What you already have:**
1. `electronics_dashboard.py`
2. `requirements.txt`

**Total deployment package:** 6 files

---

**Ready to deploy!** Any questions, refer back to this document.
