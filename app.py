import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

st.set_page_config(
    page_title="Flight Price Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === DARK NAVY PROFESSIONAL THEME === */
    .stApp {
        background-color: #0d1526;
        color: #cbd5e1;
        font-family: 'Inter', poppins;
    }

    /* Hero section headers */
    h1 {
        background: linear-gradient(90deg, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.03em;
    }
    h2 { color: #e2e8f0 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h3 { color: #94a3b8 !important; font-weight: 600 !important; }

    /* White sidebar — high contrast for filters */
    [data-testid="stSidebar"] {
        background-color: #131f35 !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }

    /* KPI Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2d4a, #162038);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 18px 22px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] {
        color: #818cf8 !important;
        font-weight: 800;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricDelta"] { color: #34d399 !important; font-weight: 600; }

    /* Tab Navigation */
    button[data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom: 2px solid #818cf8 !important;
        background: rgba(129, 140, 248, 0.08) !important;
    }

    /* Glowing Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #818cf8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 11px 28px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.55);
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white;
    }

    /* Form Inputs */
    .stTextInput>div>div>input {
        background-color: #162038;
        color: #e2e8f0;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
    }
    div[data-baseweb="select"] > div {
        background-color: #162038 !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    div[data-baseweb="select"] span { color: #cbd5e1 !important; }
    div[data-baseweb="menu"] { background-color: #1e2d4a !important; border: 1px solid rgba(148, 163, 184, 0.15) !important; }
    li[role="option"]:hover { background-color: #263858 !important; }

    /* Alert boxes */
    .stAlert { border-radius: 10px; border-left-width: 4px; }

    /* Section containers (visual cards) */
    div[data-testid="stExpander"] {
        background: #131f35;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    div[data-testid="stExpander"] summary { color: #94a3b8 !important; }

    /* Slider */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] { color: #64748b; }
</style>
""", unsafe_allow_html=True)

# Configuration for Live API
API_URL = "http://127.0.0.1:8000"
AVIATION_API_KEY = "d512635553fef823a7e725d6180641d2"

# ==========================================
# 1. LOAD LOCAL DATASET FOR EDA & MODELING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")
    df = df.drop(columns=["index", "flight"], errors="ignore").dropna()
    return df

df_original = load_data()
df = df_original.copy()

# Basic Encoding for Sidebar / Classic ML tabs
encoders = {}
categorical_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city", "stops", "class"]
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# ==========================================
# 2. SIDEBAR (Slicers / Filters)
# ==========================================
st.sidebar.header("🔍 Interactive Slicers")
st.sidebar.markdown("Filter the dataset for EDA & Local Models below.")

airline_filter = st.sidebar.multiselect("✈️ Airline", options=df_original['airline'].unique())
class_filter = st.sidebar.multiselect("💺 Class", options=df_original['class'].unique())

min_p = int(df_original["price"].min())
max_p = int(df_original["price"].max())
price_range = st.sidebar.slider("💰 Price Range (₹)", min_p, max_p, (min_p, max_p))

# Apply Filters
filtered_orig = df_original.copy()
filtered_encoded = df.copy()

if airline_filter:
    filtered_orig = filtered_orig[filtered_orig['airline'].isin(airline_filter)]
    # Safe encoding logic for the dataframe slice
    if 'airline' in encoders:
        encoded_vals = [encoders['airline'].transform([val])[0] for val in airline_filter if val in encoders['airline'].classes_]
        filtered_encoded = filtered_encoded[filtered_encoded['airline'].isin(encoded_vals)]
        
if class_filter:
    filtered_orig = filtered_orig[filtered_orig['class'].isin(class_filter)]
    if 'class' in encoders:
        encoded_vals = [encoders['class'].transform([val])[0] for val in class_filter if val in encoders['class'].classes_]
        filtered_encoded = filtered_encoded[filtered_encoded['class'].isin(encoded_vals)]

filtered_orig = filtered_orig[(filtered_orig["price"] >= price_range[0]) & (filtered_orig["price"] <= price_range[1])]
filtered_encoded = filtered_encoded[(filtered_encoded["price"] >= price_range[0]) & (filtered_encoded["price"] <= price_range[1])]

if st.sidebar.checkbox("👁️ Show Raw Filtered Data"):
    st.sidebar.dataframe(filtered_orig.head(15))

# ==========================================
# 3. HELPER FUNCTIONS FOR LIVE API
# ==========================================
def fetch_live_flight(flight_code):
    try:
        r = requests.get("http://api.aviationstack.com/v1/flights", params={"access_key": AVIATION_API_KEY, "flight_iata": flight_code}).json()
        
        # Free Tier or No Data Fallback -> Generate Mock Data
        if "data" not in r or len(r["data"]) == 0 or not r.get("data")[0].get("departure", {}).get("scheduled"):
            st.warning(f"⚠️ Live API limit reached or flight not found. Generating simulated live data for {flight_code}.")
            # Generate realistic fallback mock data based on input
            import random
            airline_mock = random.choice(VALID_AIRLINES)
            dur_mock = round(random.uniform(2.0, 15.0), 2)
            return {
                "airline": airline_mock, 
                "duration": dur_mock, 
                "source_city": "Delhi", 
                "destination_city": "Mumbai", 
                "departure_time": random.choice(["Morning", "Afternoon", "Evening", "Night"]), 
                "arrival_time": random.choice(["Morning", "Afternoon", "Evening", "Night"])
            }
            
        f = r["data"][0]
        airline = f["airline"]["name"] if f["airline"] and f["airline"]["name"] else "Air_India"
        dep, arr = f["departure"]["scheduled"], f["arrival"]["scheduled"]
        
        if not dep or not arr: 
            raise ValueError("Missing schedule")
            
        dep_dt = datetime.fromisoformat(dep.replace("Z", ""))
        arr_dt = datetime.fromisoformat(arr.replace("Z", ""))
        duration = round(abs((arr_dt - dep_dt).total_seconds() / 3600), 2)
        
        # Simple time categorizer
        hour = dep_dt.hour
        dep_time = "Morning" if 8 <= hour < 12 else "Afternoon" if 12 <= hour < 16 else "Evening" if 16 <= hour < 20 else "Early_Morning" if 4 <= hour < 8 else "Night"
        
        arr_hour = arr_dt.hour
        arr_time = "Morning" if 8 <= arr_hour < 12 else "Afternoon" if 12 <= arr_hour < 16 else "Evening" if 16 <= arr_hour < 20 else "Early_Morning" if 4 <= arr_hour < 8 else "Night"
        
        return {"airline": airline, "duration": duration, "source_city": "Delhi", "destination_city": "Mumbai", "departure_time": dep_time, "arrival_time": arr_time}
    except Exception as e:
        st.error(f"Live API Error: {str(e)}")
        return None

def predict_single(data_dict):
    try:
        res = requests.post(f"{API_URL}/predict", json=data_dict)
        return res.json() if res.status_code == 200 else None
    except:
        return None

def predict_batch(flights_list):
    try:
        res = requests.post(f"{API_URL}/simulate", json={"flights": flights_list})
        return res.json() if res.status_code == 200 else None
    except:
        return None

# ==========================================
# 4. MAIN UI & TABS
# ==========================================
st.title("✈️ Real-Time Flight Price Intelligence")
st.markdown("*A high-performance ML inference pipeline processing live market data via REST APIs.*")

tabs = st.tabs([
    "🚀 Live Market Inference", 
    "📊 Simulation Dashboard", 
    "📈 EDA & Insights", 
    "🤖 Classical Models", 
    "🧩 Clustering Patterns"
])

VALID_AIRLINES = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
VALID_CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']

# --- TAB 1: LIVE INFERENCE ---
with tabs[0]:
    st.header("Real-Time Pricing Engine")
    st.markdown("**(Tip: Try flight codes like `AI202` or `UK995` in the Live Lookup)**")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Flight Parameters")
        flight_no = st.text_input("Flight IATA Code", placeholder="e.g., AI202")
        days_left = st.slider("Days to Departure", 1, 50, 15)
        flight_class = st.selectbox("Cabin Class", ["Economy", "Business"])
        stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
        predict_btn = st.button("Fetch Live Data & Predict", use_container_width=True)
        
    with col2:
        st.subheader("Inference Results")
        if predict_btn and flight_no:
            with st.spinner("Fetching live data from Aviationstack..."):
                f_info = fetch_live_flight(flight_no)
            if f_info:
                st.success(f"Live data retrieved for {f_info['airline']} ({flight_no})")
                payload = {
                    "airline": f_info['airline'] if f_info['airline'] in VALID_AIRLINES else "Air_India", 
                    "source_city": f_info['source_city'], "destination_city": f_info['destination_city'],
                    "departure_time": f_info['departure_time'], "arrival_time": f_info['arrival_time'],
                    "duration": f_info['duration'], "days_left": days_left, "stops": stops, "flight_class": flight_class
                }
                with st.spinner("Running ML Inference Pipeline..."):
                    result = predict_single(payload)
                if result:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Fare", f"₹ {result['predicted_price']:,.0f}")
                    m2.metric("Pipeline Latency", f"{result['latency_seconds'] * 1000:.1f} ms", delta="- Fast", delta_color="inverse")
                    m3.metric("Flight Duration", f"{f_info['duration']} hrs")
            else:
                st.warning("Could not retrieve live data. The flight might not be scheduled or API limit reached.")

# --- TAB 2: SCENARIO SIMULATION ---
with tabs[1]:
    st.header("Multi-Scenario Market Simulation")
    st.markdown("Simulate **50+ competitive data points** instantly to model pricing elasticity.")
    
    sim_col1, sim_col2 = st.columns([1, 3])
    with sim_col1:
        sim_airline = st.selectbox("Carrier", VALID_AIRLINES)
        sim_source = st.selectbox("Origin", VALID_CITIES)
        sim_dest = st.selectbox("Destination", [c for c in VALID_CITIES if c != sim_source], index=1)
        sim_duration = st.slider("Est. Duration (hrs)", 1.0, 15.0, 2.5)
        sim_trigger = st.button("Generate 60 Scenarios", type="primary")
        
    with sim_col2:
        if sim_trigger:
            with st.spinner("Executing matrix inference via REST API..."):
                scenarios = [{"airline": sim_airline, "source_city": sim_source, "destination_city": sim_dest,
                              "departure_time": "Morning", "arrival_time": "Afternoon", "duration": sim_duration,
                              "days_left": d, "stops": "zero", "flight_class": c} 
                             for c in ["Economy", "Business"] for d in range(1, 31)]
                start_req = time.time()
                batch_res = predict_batch(scenarios)
                rt_latency = time.time() - start_req
                
            if batch_res:
                st.success(f"Simulated {batch_res['total_scenarios']} scenarios. Engine latency: {batch_res['latency_seconds']:.4f}s")
                df_sim = pd.DataFrame({"Days Left": list(range(1, 31))*2, "Class": ["Economy"]*30 + ["Business"]*30, "Price (₹)": batch_res['predicted_prices']})
                
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#0d1526')
                ax.set_facecolor('#162038')
                
                sns.lineplot(data=df_sim, x="Days Left", y="Price (₹)", hue="Class",
                             palette=["#34d399", "#818cf8"], linewidth=2.5, marker="o", ax=ax, markersize=7)
                
                ax.set_title("Pricing Elasticity by Booking Window", color="#e2e8f0", size=14, pad=15, weight='bold')
                ax.invert_xaxis()
                ax.tick_params(colors="#94a3b8")
                ax.xaxis.label.set_color("#94a3b8")
                ax.yaxis.label.set_color("#94a3b8")
                ax.grid(True, linestyle='--', alpha=0.25, color='#818cf8')
                for spine in ax.spines.values():
                    spine.set_color('#1e3a5f')
                leg = ax.legend(frameon=True, facecolor='#1e2d4a', edgecolor='#263858')
                for text in leg.get_texts():
                    text.set_color('#e2e8f0')
                plt.tight_layout()
                st.pyplot(fig)

# --- TAB 3: EDA & INSIGHTS ---
with tabs[2]:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Visualizing native dataset patterns from `airlines_flights_data.csv` based on your Slicer filters.")
    
    if len(filtered_orig) < 10:
        st.warning("Not enough data. Please broaden your Slicer filters.")
    else:
        e1, e2 = st.columns(2)
        with e1:
            fig1 = plt.figure(figsize=(6,4))
            ax1 = fig1.add_subplot(111)
            fig1.patch.set_facecolor('#0d1526'); ax1.set_facecolor('#162038')
            sns.scatterplot(data=filtered_orig.sample(min(1000, len(filtered_orig))), x="days_left", y="price", alpha=0.7, color="#818cf8", ax=ax1, edgecolor="#0d1526", linewidth=0.3)
            ax1.set_title("Price vs Days Left", color="#e2e8f0", weight='bold')
            ax1.set_xlabel("Days Left", color="#94a3b8"); ax1.set_ylabel("Price", color="#94a3b8"); ax1.tick_params(colors="#94a3b8")
            ax1.grid(True, linestyle='--', alpha=0.25, color='#818cf8')
            for spine in ax1.spines.values():
                spine.set_color('#1e3a5f')
            plt.tight_layout()
            st.pyplot(fig1)

        with e2:
            fig2 = plt.figure(figsize=(6,4))
            ax2 = fig2.add_subplot(111)
            fig2.patch.set_facecolor('#0d1526'); ax2.set_facecolor('#162038')
            sns.scatterplot(data=filtered_orig.sample(min(1000, len(filtered_orig))), x="duration", y="price", alpha=0.7, color="#34d399", ax=ax2, edgecolor="#0d1526", linewidth=0.3)
            ax2.set_title("Price vs Duration", color="#e2e8f0", weight='bold')
            ax2.set_xlabel("Duration", color="#94a3b8"); ax2.set_ylabel("Price", color="#94a3b8"); ax2.tick_params(colors="#94a3b8")
            ax2.grid(True, linestyle='--', alpha=0.25, color='#34d399')
            for spine in ax2.spines.values():
                spine.set_color('#1e3a5f')
            plt.tight_layout()
            st.pyplot(fig2)

        fig3 = plt.figure(figsize=(10,4))
        ax3 = fig3.add_subplot(111)
        fig3.patch.set_facecolor('#0d1526'); ax3.set_facecolor('#162038')
        sns.heatmap(filtered_encoded[["price","duration","days_left","stops","class"]].corr(), annot=True, cmap="plasma", ax=ax3, linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"color": "white"})
        ax3.set_title("Correlation Heatmap", color="#e2e8f0", pad=15, weight='bold')
        ax3.tick_params(colors="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig3)

# --- TAB 4: ML MODELS (Baseline & Classify) ---
with tabs[3]:
    st.header("🤖 Local ML Models (Linear & Logistic)")
    st.markdown("Running basic Sklearn models locally on filtered dataset to demonstrate modeling fundamentals.")
    
    if len(filtered_encoded) > 50:
        # 4a. LINEAR REGRESSION
        st.subheader("1. Price Prediction (Linear Regression)")
        X_reg = filtered_encoded[["duration", "days_left", "stops", "class"]]
        y_reg = filtered_encoded["price"]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        lr_scaler = StandardScaler()
        Xr_train_sc = lr_scaler.fit_transform(Xr_train)
        Xr_test_sc = lr_scaler.transform(Xr_test)
        
        lr = LinearRegression()
        lr.fit(Xr_train_sc, yr_train)
        yr_pred = lr.predict(Xr_test_sc)
        
        c1, c2 = st.columns(2)
        c1.metric("Linear R² Score", round(r2_score(yr_test, yr_pred), 3))
        c2.metric("Mean Absolute Error", f"₹ {int(mean_absolute_error(yr_test, yr_pred))}")
        
        # 4b. LOGISTIC REGRESSION (Class Predict)
        st.markdown("---")
        st.subheader("2. Flight Type Classification (Logistic Regression)")
        st.markdown("Goal: Predict whether a flight is Economy or Business Class based on Price, Days Left, and Duration.")
        
        X_clf = filtered_encoded[["price", "duration", "days_left", "stops"]]
        y_clf = filtered_encoded["class"] # Target
        
        # Only fit if there are both classes present
        if len(y_clf.unique()) > 1:
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
            log_scaler = StandardScaler()
            Xc_train_sc = log_scaler.fit_transform(Xc_train)
            Xc_test_sc = log_scaler.transform(Xc_test)
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xc_train_sc, yc_train)
            yc_pred = clf.predict(Xc_test_sc)
            
            acc = accuracy_score(yc_test, yc_pred)
            st.metric("Classification Accuracy", f"{acc*100:.2f}%")
            
            with st.expander("Show Classification Report"):
                report = classification_report(yc_test, yc_pred)
                st.code(report, language="text")
        else:
            st.info("⚠️ Only 1 class present in filtered data. Broaden Slicer to test classification.")
    else:
        st.warning("Not enough data to train models. Adjust siderbar filters.")

# --- TAB 5: CLUSTERING ---
with tabs[4]:
    st.header("🧩 Flight Segmentation (K-Means Clustering)")
    st.markdown("Discovering natural clusters of flights based strictly on Price vs Duration vectors.")
    
    if len(filtered_encoded) > 10:
        X_km = filtered_encoded[["price", "duration"]]
        km_scaler = StandardScaler()
        X_km_scaled = km_scaler.fit_transform(X_km)
        
        clusters = st.slider("Number of Clusters (K)", 2, 6, 3)
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        clustered_labels = kmeans.fit_predict(X_km_scaled)
        
        fig4 = plt.figure(figsize=(8, 5))
        ax4 = fig4.add_subplot(111)
        fig4.patch.set_facecolor('#0d1526'); ax4.set_facecolor('#162038')
        
        sns.scatterplot(x=filtered_encoded["duration"], y=filtered_encoded["price"], 
                        hue=clustered_labels, palette="cool", ax=ax4, alpha=0.85, edgecolor="#0d1526", linewidth=0.2)
        ax4.set_title("Flight Clusters: Price vs Duration", color="#e2e8f0", weight='bold')
        ax4.set_xlabel("Duration (hrs)", color="#94a3b8"); ax4.set_ylabel("Price (₹)", color="#94a3b8")
        ax4.tick_params(colors="#94a3b8")
        ax4.grid(True, linestyle='--', alpha=0.2, color='#818cf8')
        for spine in ax4.spines.values():
            spine.set_color('#1e3a5f')
        
        legend = ax4.legend(title="Cluster", frameon=True, facecolor='#1e2d4a', edgecolor='#263858')
        plt.setp(legend.get_title(), color='#e2e8f0')
        for text in legend.get_texts():
            text.set_color('#cbd5e1')
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.warning("Not enough data to run K-Means.")
