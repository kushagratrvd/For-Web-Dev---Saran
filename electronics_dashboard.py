"""
Electronics Consumer Behavior Prediction Dashboard
Modern UI Design with Enhanced Visual Appeal
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Electronics Prediction Service", layout="wide")

# ============================================================================
# CUSTOM CSS FOR MODERN DESIGN
# ============================================================================

st.markdown("""
<style>
    .black-header {
        color: black !important;
    }
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #2596be 0%, #92cbdf 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(131, 224, 241) 0%, rgba(131, 206, 241) 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
            
    .main-header h1{
        color:#003a6c; 
        font-size:2.5rem; 
        margin-bottom:0.5rem; 
        font-weight:700;
    }
    
    .main-header p {
        color: #666;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Tab styling */
    .tab-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Content card */
    .content-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .special-title {
        color:#003a6c; 
        font-size:2.5rem; 
        margin-bottom:0.5rem; 
        font-weight:700;
    }
            
    .section-title {
        color: #333;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Quick test buttons */
    .quick-test-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .quick-test-title {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* ========================================= */
    /* METRIC CARD STYLING                       */
    /* ========================================= */
    .metric-card {
        background: linear-gradient(135deg, #003c70 0%, #012a4d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }

    /* SPECIFICITY FIX: Target elements explicitly to beat the global rule */
    /* Rule Score: 0-1-2 (Class + Tag + Tag) vs Global 0-1-1 */
    div.metric-card h3,
    div.metric-card p,
    div.metric-card span,
    div.metric-card div {
        color: white !important;
    }

    /* Fix the SVG Icon color inside the header link */
    div.metric-card svg {
        fill: white !important;
        stroke: white !important;
    }
    
    /* Input fields */
    .stNumberInput {
        background: white;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #003c70 0%, #012a4d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgb(0, 60, 112);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(253, 221, 232) 0%, rgb(253, 221, 232) 100%);
    }
    
    [data-testid="stSidebar"] .stButton > button {
        margin-top: 1rem;
    }
            
    /* ========== GLOBAL BLACK TEXT FIXES ========== */

    /* Sidebar: labels, markdown, text, inputs */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p {
        color: black !important;
    }

    /* Tabs text */
    .stTabs [data-baseweb="tab"] {
        color: black !important;
        font-weight: 600 !important;
    }
            
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        font-size: 1.7rem !important;
        font-weight: 600;   /* ← change size here  */
        color: black !important;         /* optional */
    }
            
    /* Style to force Left-Center-Right tab alignment */
.stTabs [data-baseweb="tab-list"] {
    /* Step 1: Force the container to take full width and justify its children */
    gap: 0px !important; /* Remove any default gaps */
    justify-content: space-between !important; /* Distributes items evenly */
    width: 100% !important;
}

.stTabs [data-baseweb="tab"] {
    /* Step 2: Ensure each tab title is centered within its spaced area */
    justify-content: center !important; 
    flex-grow: 1 !important; /* Allow tabs to grow and fill the space */
    width: 33.33% !important; /* Explicitly set width for consistent spacing */
    
    /* Re-add the appearance styles from your original file */
    margin-right: 0px !important; 
    padding-left: 0px !important;
    padding-right: 0px !important;
}

    /* Expander title text */
    .streamlit-expanderHeader {
        color: black !important;
    }

    /* Markdown text inside main content */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: black !important;
    }

    /* Table headers + row text */
    [data-testid="stDataFrame"] table,
    [data-testid="stDataFrame"] tbody tr td,
    [data-testid="stDataFrame"] thead tr th {
        color: black !important;
    }

    /* Metric label text */
    .css-1xarl3l, .css-17eq0hr, .css-1ht1j8u {
        color: black !important;
    }

    /* Text inside number inputs */
    .stNumberInput input {
        color: black !important;
    }

    /* Dropdown menu background */
    ul[role="listbox"] {
        background: white !important;
        border-radius: 10px !important;
    }

    /* Dropdown option text */
    ul[role="listbox"] li {
        color: black !important;
        background: white !important;
    }

    ul[role="listbox"] li:hover {
        background: #e6e6e6 !important;
    }      
            
    div[data-testid="stWidgetLabel"] {
    background: transparent !important;
    padding: 0 !important;
    }

    div[data-testid="stWidgetLabel"] p {
        color: black !important;
    }

    .stButton > button,
    .stButton > button * {
        color: white !important;
    }

    /* Remove the white bar behind Selectbox labels */
label[data-testid="stWidgetLabel"] > div[class*="st-emotion-cache"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
            
    /* Target the Selectbox container (Closed State) */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        color: black !important;
    }

    /* Force text color to Black so it is visible on the White background */
    div[data-baseweb="select"] span {
        color: black !important;
    }

    /* Change the dropdown arrow icon to black */
    div[data-baseweb="select"] svg {
        fill: black !important;
    }

    /* Ensure the Dropdown Menu (Open State) is also white */
    ul[data-baseweb="menu"] {
        background-color: white !important;
    }

    /* Force the text and arrow of the UP delta (green/normal) to black */
[data-testid="stMetricDelta"] [data-testid="stMetricDelta"] {
    color: black !important;
}

/* Specific styling for the UP arrow (using delta_color="normal") */
[data-testid="stMetricDelta"] svg[fill="#4589ff"] { /* Targeting the default green/blue color associated with 'normal' */
    fill: black !important;
}

/* Specific styling for the DOWN arrow (using delta_color="inverse") */
[data-testid="stMetricDelta"] svg[fill="#ef553b"] { /* Targeting the default red color associated with 'inverse' */
    fill: black !important;
}

/* Ensure the value text remains black (sometimes needed for high specificity) */
[data-testid="stMetricDelta"] {
    color: black !important;
}
            
/* 1. Force White Text & Font Size (High Specificity override) */
    div[data-testid="stMarkdownContainer"] div.static-button {
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        line-height: 1.5 !important; /* Ensure vertical centering looks right */
    }

    /* 2. Base Layout & Styling */
    .static-button {
        background: linear-gradient(135deg, #003c70 0%, #012a4d 100%);
        border-radius: 10px;
        /* Match standard button padding */
        padding: 0.75rem 1rem; 
        text-align: center;
        /* Make it behave like a block element to fill the column */
        display: block;
        width: 100%;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }
            
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. LOAD DATA AND MODELS
# ============================================================================

@st.cache_data
def load_data():
    # Load data from the 'data/' directory
    df = pd.read_csv('data/consumer_behavior_electronics_processed_complete.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_models():
    models = {}
    
    # Load Random Forest (Best Accuracy: 5.14% sMAPE)
    try:
        with open('models/random_forest_production.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model 'models/random_forest_production.pkl' not found.")
        models['Random Forest'] = None

    # Load Tuned LightGBM (Best Direction: 88.63%)
    try:
        models['Tuned LightGBM'] = lgb.Booster(model_file='models/tuned_lightgbm_production.txt')
    except Exception as e:
        models['Tuned LightGBM'] = None
    
    return models

# Load resources
df = load_data()
models = load_models()

# Identify feature columns
non_feature_cols = ['date', 'brand', 'product', 'sales_units', 'avg_price', 'year_month']
feature_cols = [c for c in df.columns if c not in non_feature_cols]

# ============================================================================
# 2. PREDICTION ENGINE
# ============================================================================

def get_predictions(brand, months_ahead, model_name):
    """
    Generates predictions using the selected ML model.
    Uses the latest available data point features for prediction.
    """
    model = models.get(model_name)
    if not model:
        return [], df['date'].max()

    brand_products = df[df['brand'] == brand]['product'].unique()
    predictions_data = []
    latest_date = df['date'].max()

    for product in brand_products:
        product_data = df[(df['brand'] == brand) & (df['product'] == product)].copy()
        
        if len(product_data) < 12:
            continue
            
        latest_features = product_data[feature_cols].iloc[-1:].values
        
        if model_name == 'Random Forest':
            base_pred = model.predict(latest_features)[0]
        else:
            base_pred = model.predict(latest_features)[0]
            
        forecasts = []
        for i in range(1, months_ahead + 1):
            future_date = latest_date + timedelta(days=30 * i)
            
            seasonal_factor = 1.0
            if future_date.month in [11, 12]: seasonal_factor = 1.2
            elif future_date.month == 1: seasonal_factor = 0.85
            
            trend_factor = 1.0 + (np.random.normal(0, 0.01)) 
            final_pred = base_pred * seasonal_factor * trend_factor
            forecasts.append(int(max(final_pred, 100)))
            base_pred = final_pred 

        current_sales = product_data.iloc[-1]['sales_units']
        trend = 'up' if forecasts[0] > current_sales else 'down'
        
        predictions_data.append({
            'product': product,
            'predictions': forecasts,
            'current': int(current_sales),
            'trend': trend
        })
        
    return predictions_data, latest_date

# ============================================================================
# 3. USER INTERFACE
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <div class="special-title">
        Cloud-Integrated Electronics Prediction System
    </div>
    <p>Advanced AI-powered platform for real-time sales forecasting and consumer behavior analysis</p>
</div>
""", unsafe_allow_html=True)

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["Current Predictions", "Multi-Month Trend", "Model Performance"])

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown('<div class="section-title">Prediction Parameters</div>', unsafe_allow_html=True)
    st.markdown("")
    
    # Brand Selection
    available_brands = sorted(df['brand'].unique().tolist())
    selected_brand = st.selectbox(
        "Select Brand",
        options=available_brands,
        help="Choose the electronics brand for prediction"
    )
    
    st.markdown("")
    
    # Timeline Selection
    timeline = st.selectbox(
        "Forecast Timeline",
        options=[1, 2, 3],
        format_func=lambda x: f"{x} month(s)",
        help="Select prediction horizon"
    )
    
    st.markdown("")
    
    # Model Selection
    model_choice = st.selectbox(
        "AI Model",
        ["Random Forest", "Tuned LightGBM"],
        help="Random Forest (Best Accuracy) | LightGBM (Best Direction)"
    )
    
    predict_button = st.button("Generate Predictions")
    
    st.markdown("---")
    st.info("Select parameters above and click 'Generate Predictions' to start forecasting.")

# ============================================================================
# TAB 1: CURRENT PREDICTIONS
# ============================================================================

with tab1:    
    # Quick Test Section
    st.markdown("""
        <div class="section-title"> Quick Test - Sample Brands</div>
    """, unsafe_allow_html=True)

    st.markdown("")
    
    # Sample brand buttons
    sample_brands = available_brands[:min(5, len(available_brands))]
    cols = st.columns(len(sample_brands))
    for idx, brand in enumerate(sample_brands):
        with cols[idx]:
            if st.button(brand, key=f"sample_{brand}"):
                selected_brand = brand
                predict_button = True
    
    st.markdown("---")
    
    if predict_button and selected_brand:
        # Section Title
        st.markdown(f'<div class="section-title">Forecast Analysis for {selected_brand}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-subtitle">Model: {model_choice} | Forecast Period: Next {timeline} month(s)</div>', unsafe_allow_html=True)
        
        preds, start_date = get_predictions(selected_brand, timeline, model_choice)
        
        if not preds:
            st.warning(f"Insufficient data to generate predictions for {selected_brand}.")
        else:
            # Summary Metrics
            total_sales = sum([sum(p['predictions']) for p in preds])
            avg_growth = sum([((p['predictions'][0] - p['current']) / p['current'] * 100) for p in preds]) / len(preds)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{:,}</h3>
                    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Total Volume</p>
                </div>
                """.format(total_sales), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{}</h3>
                    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Products Analyzed</p>
                </div>
                """.format(len(preds)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{} Mo</h3>
                    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Forecast Horizon</p>
                </div>
                """.format(timeline), unsafe_allow_html=True)
            
            with col4:
                growth_color = "#4CAF50" if avg_growth > 0 else "#f44336"
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem; color:{};">{:+.1f}%</h3>
                    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Avg Growth</p>
                </div>
                """.format(growth_color, avg_growth), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Product-wise Predictions
            for p in preds:
                with st.expander(f"📱 **{p['product']}**", expanded=True):
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        # Metrics
                        trend_icon = "📈" if p['trend'] == 'up' else "📉"
                        trend_color = "green" if p['trend'] == 'up' else "red"
                        change_pct = ((p['predictions'][0] - p['current']) / p['current'] * 100)
                        
                        st.metric(
                            "Next Month Projection",
                            f"{p['predictions'][0]:,} units",
                            f"{change_pct:+.1f}%",
                            delta_color="normal" if p['trend'] == 'up' else "inverse"
                        )
                        
                        st.markdown(f"**Current Sales:** {p['current']:,} units")
                        st.markdown(f"**Trend:** {trend_icon} :{trend_color}[**{p['trend'].upper()}**]")
                        
                        st.markdown("---")
                        st.markdown("**📅 Monthly Breakdown:**")
                        for i, pred in enumerate(p['predictions'], 1):
                            future_date = start_date + timedelta(days=30 * i)
                            st.write(f"• {future_date.strftime('%b %Y')}: **{pred:,}** units")
                    
                    with c2:
                        # Chart
                        dates = [(start_date + timedelta(days=30*i)).strftime("%b %Y") for i in range(1, timeline+1)]
                        
                        # Determine bar color based on timeline
                        if timeline == 1:
                            bar_color = '#00e03c' # Light green for single month
                            show_scale = False
                        else:
                            bar_color = p['predictions'] # Use values for color scale
                            show_scale = False

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=dates,
                            y=p['predictions'],
                            marker=dict(
                                color=bar_color,
                                colorscale='Viridis' if timeline > 1 else None,
                                showscale=show_scale
                            ),
                            text=[f"{val:,}" for val in p['predictions']],
                            textposition='outside',
                            name='Predicted Sales'
                        ))
                        
                        fig.update_layout(
                            title=f"{p['product']} Sales Forecast",
                            xaxis_title="Month",
                            yaxis_title="Sales Units",
                            height=350,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Arial", size=12),
                            margin=dict(t=50, b=40, l=40, r=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
<div style='text-align: center; color: white;'>
    <strong>👈 Select a brand from the sidebar and click 'Generate Predictions' to view forecasts. </strong>
</div>
""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: MULTI-MONTH TREND
# ============================================================================

with tab2:
    st.markdown('<div class="section-title">Multi-Month Trend Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Compare predictions across different time horizons</div>', unsafe_allow_html=True)
    
    if predict_button and selected_brand:
        preds, start_date = get_predictions(selected_brand, 3, model_choice)
        
        if preds:
            # Aggregate trend chart
            dates = [(start_date + timedelta(days=30*i)).strftime("%b %Y") for i in range(1, 4)]
            
            fig = go.Figure()
            
            for p in preds[:5]:  # Top 5 products
                y_values = p['predictions'][:3]
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=y_values,
                    mode='lines+markers',
                    name=p['product'],
                    line=dict(width=3),
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Top Products - 3-Month Forecast Comparison",
                xaxis_title="Month",
                yaxis_title="Predicted Sales (Units)",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align: center; color: white; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px;'>
            <strong>Generate predictions in the 'Current Predictions' tab to view multi-month trends.</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================

with tab3:
    st.markdown('<div class="section-title"> Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Detailed accuracy and performance statistics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Random Forest")
        st.success("✅ **Best for Accuracy**")
        st.metric("sMAPE Score", "5.14%", "-0.08%", delta_color="inverse")
        st.metric("Trend Accuracy", "87.63%")
        st.markdown("""
        **Strengths:**
        - Highest numerical accuracy
        - Robust to outliers
        - Good generalization
        """)
    
    with col2:
        st.markdown("### Tuned LightGBM")
        st.success("✅ **Best for Trend Direction**")
        st.metric("sMAPE Score", "5.22%")
        st.metric("Trend Accuracy", "88.63%", "+1.0%")
        st.markdown("""
        **Strengths:**
        - Best directional accuracy
        - Fast inference time
        - Efficient memory usage
        """)
    
    st.markdown("---")
    
    st.markdown("### Performance Comparison Table")
    perf_df = pd.DataFrame({
        "Model": ["Random Forest", "Tuned LightGBM"],
        "Accuracy (sMAPE)": ["5.14% ⭐", "5.22%"],
        "Trend Accuracy": ["87.63%", "88.63% ⭐"],
        "Training Time": ["Medium", "Fast"],
        "Inference Speed": ["Fast", "Very Fast"]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Available Brands")
    # Chunk brands into groups of 5 to match the layout of Tab 1
    chunks = [available_brands[i:i + 5] for i in range(0, len(available_brands), 5)]

    for chunk in chunks:
        # Create a new row of 5 columns for each chunk
        cols = st.columns(5)
        for i, brand in enumerate(chunk):
            with cols[i]:
                 # Use <div> instead of <span> for block-level width
                 st.markdown(f'<div class="static-button">{brand}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px;'>
    <strong>Consumer Behavior Prediction Service</strong> | Electronics Domain | Powered by Machine Learning<br>
    © 2024 AI Forecasting Solutions
</div>
""", unsafe_allow_html=True)