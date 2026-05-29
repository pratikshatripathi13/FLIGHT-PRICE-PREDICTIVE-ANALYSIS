import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
import base64
from pathlib import Path
from datetime import datetime, date, timedelta
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

st.set_page_config(page_title="SkyPrice — Smart Flight Fare Predictor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* ─── Core page styling ─── */
    .stApp {
        background-color: #111111;
        color: #e8e8e8;
        font-family: 'Outfit', sans-serif;
    }

    /* ─── Hero section ─── */
    .hero-wrapper {
        position: relative;
        width: 100%;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    }
    .hero-wrapper img {
        width: 100%;
        display: block;
        object-fit: cover;
        max-height: 340px;
        filter: brightness(0.72) contrast(1.08) saturate(0.5);
    }
    .hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(
            to bottom,
            rgba(10,10,10,0.15) 0%,
            rgba(10,10,10,0.45) 60%,
            rgba(10,10,10,0.88) 100%
        );
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 2.5rem 3rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin: 0 0 0.4rem 0;
        color: #f5f5f5;
        text-shadow: 0 2px 20px rgba(0,0,0,0.8);
    }
    .hero-title span {
        background: linear-gradient(90deg, #d4d4d4, #f5f5f5, #a8a8a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9a9a9a;
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.01em;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #b0b0b0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        width: fit-content;
    }

    /* ─── Headers & Text ─── */
    h1 {
        color: #e8e8e8 !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem !important;
    }
    h2 { color: #d4d4d4 !important; font-weight: 700 !important; }
    h3 { color: #a8a8a8 !important; font-weight: 600 !important; }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #707070 !important;
    }

    /* ─── Metric Cards ─── */
    [data-testid="stMetric"] {
        background: rgba(30,30,30,0.7) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 16px;
        padding: 20px 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(200,200,200,0.2) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stMetricValue"] {
        color: #d4d4d4 !important;
        font-weight: 800;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #686868 !important;
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricDelta"] {
        color: #a8a8a8 !important;
        font-weight: 600;
    }

    /* ─── Tabs ─── */
    button[data-baseweb="tab"] {
        color: #686868 !important;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 12px 24px !important;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #c8c8c8 !important;
        background: rgba(255,255,255,0.04) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e8e8e8 !important;
        border-bottom: 2px solid #c0c0c0 !important;
        background: rgba(255,255,255,0.06) !important;
    }

    /* ─── Form Buttons ─── */
    .stButton>button {
        background: linear-gradient(135deg, #3a3a3a, #2a2a2a) !important;
        color: #e8e8e8 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px;
        padding: 14px 30px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
        background: linear-gradient(135deg, #505050, #3a3a3a) !important;
        border-color: rgba(255,255,255,0.22) !important;
        color: #ffffff !important;
    }
    .stButton>button:active {
        transform: translateY(1px);
        background: #222222 !important;
        color: #e8e8e8 !important;
    }
    .stButton>button:focus {
        color: #e8e8e8 !important;
        outline: none !important;
    }
    .stButton>button p, .stButton>button span {
        color: #e8e8e8 !important;
    }

    /* ─── Dropdowns ─── */
    div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #d4d4d4 !important;
        padding: 4px 6px !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(200,200,200,0.3) !important;
    }
    div[data-baseweb="select"] span { color: #d4d4d4 !important; }
    div[data-baseweb="select"] svg { fill: #686868 !important; }

    div[data-baseweb="popover"], div[role="listbox"], ul[role="listbox"], div[data-baseweb="menu"] {
        background-color: #1a1a1a !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.7) !important;
    }
    li[role="option"], div[role="option"] {
        color: #b0b0b0 !important;
        background-color: transparent !important;
        padding: 12px 16px !important;
        transition: background-color 0.15s ease, color 0.15s ease;
        font-weight: 500;
    }
    li[role="option"]:hover, div[role="option"]:hover,
    li[role="option"][aria-selected="true"]:hover, div[role="option"][aria-selected="true"]:hover {
        background-color: #333333 !important;
        color: #f5f5f5 !important;
        cursor: pointer;
    }
    li[role="option"][aria-selected="true"], div[role="option"][aria-selected="true"] {
        background-color: rgba(255,255,255,0.08) !important;
        color: #d4d4d4 !important;
        font-weight: 600;
    }

    /* ─── Inputs ─── */
    .stTextInput input, div[data-testid="stDateInput"] input {
        background-color: #1a1a1a !important;
        color: #d4d4d4 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus, div[data-testid="stDateInput"] input:focus {
        border-color: rgba(200,200,200,0.35) !important;
        box-shadow: 0 0 0 2px rgba(200,200,200,0.1) !important;
    }
    .stTextInput input::placeholder { color: #404040 !important; }

    /* ─── Labels & P tags ─── */
    label {
        color: #686868 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    p { color: #a8a8a8 !important; }

    /* ─── Sliders ─── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #888888 !important;
        box-shadow: 0 0 0 4px rgba(180,180,180,0.2) !important;
    }
    .stSlider p { color: #686868 !important; }

    /* ─── Expanders ─── */
    div[data-testid="stExpander"] {
        background: #161616 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stExpander"] summary { color: #686868 !important; }
    div[data-testid="stExpander"] summary:hover { color: #c0c0c0 !important; }
    div[data-testid="stExpander"] summary svg { fill: #686868 !important; }

    /* ─── Alerts ─── */
    .stAlert { border-radius: 12px !important; }

    /* ─── Dataframes ─── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    [data-testid="stDataFrame"] th {
        background-color: #1e1e1e !important;
        color: #686868 !important;
    }
    [data-testid="stDataFrame"] td {
        color: #a8a8a8 !important;
        background-color: #111111 !important;
    }

    /* ─── Advice Cards ─── */
    .advice-card {
        background: rgba(25,25,25,0.7);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border-left: 4px solid #606060;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
    }
    .advice-card.warn {
        border-left-color: #888840;
        background: rgba(100,90,30,0.08);
        border: 1px solid rgba(150,140,50,0.2);
        border-left: 4px solid #888840;
    }
    .advice-card.urgent {
        border-left-color: #804040;
        background: rgba(100,30,30,0.08);
        border: 1px solid rgba(150,60,60,0.2);
        border-left: 4px solid #804040;
    }
    .advice-card h3 { color: #d4d4d4 !important; margin: 0 0 8px 0 !important; }
    .advice-card p { color: #787878 !important; margin: 0 !important; line-height: 1.6 !important; }

    /* ─── Multiselect Tags ─── */
    span[data-baseweb="tag"] {
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] span { color: #c0c0c0 !important; }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #2e2e2e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555555; }

    /* ─── Divider ─── */
    hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Project root (portable across machines) ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent

API_URL = "http://127.0.0.1:8000"
VALID_AIRLINES = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
VALID_CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
DEP_TIMES = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
STOP_OPTIONS = {'Non-stop': 'zero', '1 Stop': 'one', '2+ Stops': 'two_or_more'}

@st.cache_data
def load_data():
    df = pd.read_csv(PROJECT_ROOT / "airlines_flights_data.csv")
    return df.drop(columns=["index", "flight"], errors="ignore").dropna()

df_original = load_data()
df = df_original.copy()
encoders = {}
categorical_cols = ["airline", "source_city", "departure_time", "arrival_time", "destination_city", "stops", "class"]
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# ── API helper functions ──────────────────────────────────────────────────
def _api_get(endpoint: str, timeout: int = 4):
    """GET request; returns parsed JSON or None on failure."""
    try:
        res = requests.get(f"{API_URL}/{endpoint}", timeout=timeout)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def predict_single(payload):
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def predict_range(payload):
    """Returns price + confidence interval (p10, p90, std_dev)."""
    try:
        res = requests.post(f"{API_URL}/predict-range", json=payload, timeout=5)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def predict_batch(flights_list):
    try:
        res = requests.post(f"{API_URL}/simulate", json={"flights": flights_list}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#181818')
    for sp in ax.spines.values():
        sp.set_color('#2e2e2e')
    ax.tick_params(colors='#686868')
    ax.xaxis.label.set_color('#686868')
    ax.yaxis.label.set_color('#686868')
    ax.grid(True, linestyle='--', alpha=0.15, color='#404040')
    return fig, ax

# ===================== HEADER =====================
def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_hero_path = PROJECT_ROOT / "assets" / "hero.png"
_hero_b64  = _img_to_b64(_hero_path) if _hero_path.exists() else ""

st.markdown(f"""
<div class="hero-wrapper">
    <img src="data:image/png;base64,{_hero_b64}" alt="Flight hero" />
    <div class="hero-overlay">
        <div class="hero-badge">✈&nbsp;&nbsp;ML-POWERED FARE INTELLIGENCE</div>
        <h1 class="hero-title"><span>SkyPrice</span> — Smart Flight Fare Intelligence</h1>
        <p class="hero-subtitle">Predict fares · Compare airlines · Find the cheapest day to fly</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── API Status Banner ─────────────────────────────────────────────────────
_health = _api_get("health")
if _health and _health.get("model_loaded"):
    _meta_quick = _health.get("meta") or {}
    _r2  = _meta_quick.get("metrics", {}).get("r2_score", "—")
    _mae = _meta_quick.get("metrics", {}).get("mae", "—")
    st.markdown(
        f'<div style="background:rgba(40,70,40,0.25);border:1px solid rgba(100,180,100,0.3);'
        f'border-radius:12px;padding:10px 18px;margin-bottom:1rem;display:flex;'
        f'align-items:center;gap:12px;font-size:0.88rem;color:#a8d8a8;">'
        f'<span style="font-size:1.1rem;">✅</span>'
        f'<span><strong>API Online</strong> — Model loaded &nbsp;·&nbsp; '
        f'R²&nbsp;{_r2}&nbsp;·&nbsp;MAE&nbsp;Rs.&nbsp;{_mae}</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="background:rgba(80,30,30,0.3);border:1px solid rgba(180,80,80,0.35);'
        'border-radius:12px;padding:10px 18px;margin-bottom:1rem;display:flex;'
        'align-items:center;gap:12px;font-size:0.88rem;color:#e8a0a0;">'
        '<span style="font-size:1.1rem;">❌</span>'
        '<span><strong>API Offline</strong> — Start the server: '
        '<code>uvicorn api:app --reload</code></span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

tabs = st.tabs(["✈️ Smart Price Checker", "📊 Simulation Dashboard", "📈 EDA & Insights", "🤖 ML Models", "🧩 Clustering"])

# ===================== TAB 1: SMART PRICE CHECKER =====================
with tabs[0]:
  try:
    st.header("Find the Best Fare for Your Journey")
    st.markdown("Enter your trip details below — our ML engine will predict prices and tell you **when to book**.")

    with st.form("price_checker_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            src = st.selectbox("🛫 Flying From", VALID_CITIES, index=0)
        with r1c2:
            dst_opts = [c for c in VALID_CITIES if c != src]
            dst = st.selectbox("🛬 Flying To", dst_opts, index=0)
        with r1c3:
            travel_date = st.date_input("📅 Travel Date", value=date.today() + timedelta(days=15),
                                        min_value=date.today() + timedelta(days=1),
                                        max_value=date.today() + timedelta(days=60))

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            cabin = st.selectbox("💺 Cabin Class", ["Economy", "Business"])
        with r2c2:
            stop_label = st.selectbox("🔁 Stops Preference", list(STOP_OPTIONS.keys()))
            stops_val = STOP_OPTIONS[stop_label]
        with r2c3:
            airline_pref = st.selectbox("✈️ Preferred Airline", ["Best Available"] + VALID_AIRLINES)

        submitted = st.form_submit_button("🔍 Find Best Price", use_container_width=True)

    if submitted:
        days_left = (travel_date - date.today()).days
        airlines_to_check = VALID_AIRLINES if airline_pref == "Best Available" else [airline_pref]

        with st.spinner("Running ML inference across all airlines & departure slots..."):
            all_results = []
            for airline in airlines_to_check:
                for dep_time in DEP_TIMES:
                    arr_time_guess = DEP_TIMES[(DEP_TIMES.index(dep_time) + 2) % len(DEP_TIMES)]
                    payload = {
                        "airline": airline, "source_city": src, "destination_city": dst,
                        "departure_time": dep_time, "arrival_time": arr_time_guess,
                        "duration": 2.5, "days_left": days_left,
                        "stops": stops_val, "flight_class": cabin
                    }
                    result = predict_single(payload)
                    if result:
                        all_results.append({
                            "airline": airline, "departure_time": dep_time,
                            "price": result["predicted_price"], "latency": result["latency_seconds"]
                        })

        if not all_results:
            st.markdown(
                '<div style="background:rgba(80,30,30,0.35);border:1px solid rgba(180,80,80,0.4);border-radius:14px;padding:18px 22px;margin:12px 0;">'
                '<p style="color:#e8a0a0;font-size:1rem;font-weight:700;margin:0 0 8px 0;">&#10060; Could not reach the ML API</p>'
                '<p style="color:#b07070;margin:0 0 10px 0;font-size:0.9rem;">The API server is not responding. You need <strong>two terminals running simultaneously</strong>.</p>'
                '<p style="color:#b07070;margin:0 0 6px 0;font-size:0.85rem;">Keep Streamlit open, then in a <strong>new terminal</strong> run:</p>'
                '<code style="background:#1a0a0a;color:#f0a0a0;padding:8px 14px;border-radius:8px;display:block;font-size:0.9rem;">python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload</code>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            df_results = pd.DataFrame(all_results)
            best = df_results.loc[df_results["price"].idxmin()]
            avg_price = df_results["price"].mean()
            cheapest_airline = df_results.groupby("airline")["price"].min().idxmin()
            cheapest_airline_price = df_results.groupby("airline")["price"].min().min()

            # KPIs
            st.markdown("### Your Trip Summary")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best Predicted Fare", f"Rs. {best['price']:,.0f}", delta=f"via {best['airline']}")
            k2.metric("Days Until Departure", str(days_left), delta=f"{travel_date.strftime('%d %b %Y')}")
            k3.metric("Avg Fare (All Airlines)", f"Rs. {avg_price:,.0f}")
            k4.metric("Cheapest Airline", cheapest_airline, delta=f"Rs. {cheapest_airline_price:,.0f}")

            # Booking Advice
            if days_left <= 5:
                advice_class, icon, title = "urgent", "🚨", "Book Immediately!"
                advice_body = f"Only **{days_left} days** until your flight — prices are at their peak and rising fast. Book right now to avoid paying significantly more."
            elif days_left <= 14:
                advice_class, icon, title = "warn", "⏰", "Book Soon — Prices Are Rising"
                advice_body = f"You have **{days_left} days** left. Our model shows prices increase sharply in the final 2 weeks. Locking in today could save you Rs. {int(avg_price * 0.15):,}+ vs waiting."
            elif days_left <= 30:
                advice_class, icon, title = "advice-card", "💡", "Good Time to Book"
                advice_body = f"**{days_left} days** out is a sweet spot. Fares are reasonable now. If flexibility allows, monitoring for another 3–5 days before booking is low-risk."
            else:
                advice_class, icon, title = "advice-card", "✅", "Prices Are Favourable — No Rush"
                advice_body = f"With **{days_left} days** to go, fares are near their lowest. You can wait a bit longer without much risk, but booking now locks in a good rate."

            st.markdown(f"""
            <div class="advice-card {advice_class}">
                <h3 style="color:#e2e8f0;margin:0 0 8px 0">{icon} {title}</h3>
                <p style="color:#94a3b8;margin:0">{advice_body}</p>
            </div>
            """, unsafe_allow_html=True)

            # Airline Comparison Chart
            st.markdown("### Airline Price Comparison")
            airline_summary = df_results.groupby("airline")["price"].min().sort_values()

            fig_bar, ax_bar = dark_fig(9, 4)
            bars = ax_bar.barh(airline_summary.index, airline_summary.values,
                               color=["#d4d4d4" if a == cheapest_airline else "#505050" for a in airline_summary.index],
                               height=0.6, edgecolor='none')
            for bar, val in zip(bars, airline_summary.values):
                ax_bar.text(val + 200, bar.get_y() + bar.get_height()/2,
                            f"Rs. {val:,.0f}", va='center', color='#d4d4d4', fontsize=10, fontweight='600')
            ax_bar.set_xlabel("Predicted Fare (Rs.)", color='#686868')
            ax_bar.set_title(f"Best Fare per Airline — {src} → {dst} | {cabin} | {stop_label}", color='#d4d4d4', weight='bold', pad=15)
            ax_bar.set_xlim(0, airline_summary.max() * 1.22)
            plt.tight_layout()
            st.pyplot(fig_bar)

            # 30-Day Price Calendar
            st.markdown("### 30-Day Price Calendar")
            st.caption(f"Predicted fare for {cheapest_airline} ({cabin}) departing from {src} → {dst} over the next 30 days")

            with st.spinner("Generating 30-day price calendar..."):
                calendar_payloads = []
                for offset in range(1, 31):
                    d_left = (travel_date - date.today()).days - offset + 15
                    d_left = max(1, d_left)
                    calendar_payloads.append({
                        "airline": cheapest_airline, "source_city": src, "destination_city": dst,
                        "departure_time": best["departure_time"], "arrival_time": "Afternoon",
                        "duration": 2.5, "days_left": d_left,
                        "stops": stops_val, "flight_class": cabin
                    })
                batch_cal = predict_batch(calendar_payloads)

            if batch_cal:
                cal_dates = [date.today() + timedelta(days=i) for i in range(1, 31)]
                cal_prices = batch_cal["predicted_prices"]
                min_idx = cal_prices.index(min(cal_prices))

                fig_cal, ax_cal = dark_fig(11, 4)
                ax_cal.fill_between(cal_dates, cal_prices, alpha=0.15, color='#818cf8')
                ax_cal.plot(cal_dates, cal_prices, color='#818cf8', linewidth=2.5, marker='o', markersize=4)
                ax_cal.scatter([cal_dates[min_idx]], [cal_prices[min_idx]], color='#34d399', s=120, zorder=5, label=f"Cheapest: Rs. {cal_prices[min_idx]:,.0f} on {cal_dates[min_idx].strftime('%d %b')}")
                ax_cal.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                ax_cal.xaxis.set_major_locator(mdates.DayLocator(interval=3))
                plt.xticks(rotation=45)
                ax_cal.set_ylabel("Predicted Fare (Rs.)", color='#94a3b8')
                ax_cal.set_title("Price Trend — Next 30 Days", color='#e2e8f0', weight='bold', pad=15)
                leg = ax_cal.legend(frameon=True, facecolor='#1e2d4a', edgecolor='#263858')
                for t in leg.get_texts(): t.set_color('#e2e8f0')
                plt.tight_layout()
                st.pyplot(fig_cal)
                st.success(f"Cheapest day to fly: **{cal_dates[min_idx].strftime('%A, %d %B %Y')}** at **Rs. {cal_prices[min_idx]:,.0f}**")

            # Confidence Range for best fare
            st.markdown("### Price Confidence Range")
            st.caption("How much could the price vary? Based on individual decision tree estimates.")
            _range_payload = {
                "airline": best["airline"], "source_city": src, "destination_city": dst,
                "departure_time": best["departure_time"], "arrival_time": "Afternoon",
                "duration": 2.5, "days_left": days_left,
                "stops": stops_val, "flight_class": cabin,
            }
            _ci = predict_range(_range_payload)
            if _ci:
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric("💰 Predicted Fare",    f"Rs. {_ci['predicted_price']:,.0f}")
                ci2.metric("⬇️ Low Estimate (P10)",  f"Rs. {_ci['p10']:,.0f}")
                ci3.metric("⬆️ High Estimate (P90)", f"Rs. {_ci['p90']:,.0f}")
                ci4.metric("📊 Std Dev",             f"Rs. {_ci['std_dev']:,.0f}")

            # Best Departure Time Breakdown
            with st.expander("View Price Breakdown by Departure Time"):
                dep_summary = df_results[df_results["airline"] == cheapest_airline].sort_values("price")
                dep_summary_display = dep_summary[["departure_time", "price"]].rename(columns={"departure_time": "Departure Slot", "price": "Predicted Fare (Rs.)"})
                dep_summary_display["Predicted Fare (Rs.)"] = dep_summary_display["Predicted Fare (Rs.)"].apply(lambda x: f"Rs. {x:,.0f}")
                st.dataframe(dep_summary_display, use_container_width=True, hide_index=True)
  except Exception as e:
      st.error(f"⚠️ Smart Price Checker encountered an error: {e}")

# ===================== TAB 2: SIMULATION DASHBOARD =====================
with tabs[1]:
  try:
    st.header("Multi-Scenario Market Simulation")
    st.markdown("Simulate **60 competitive scenarios** instantly to model pricing elasticity by class and booking window.")

    sim_col1, sim_col2 = st.columns([1, 3])
    with sim_col1:
        sim_airline = st.selectbox("Carrier", VALID_AIRLINES, key="sim_air")
        sim_src = st.selectbox("Origin", VALID_CITIES, key="sim_src")
        sim_dst = st.selectbox("Destination", [c for c in VALID_CITIES if c != sim_src], index=1, key="sim_dst")
        sim_dep_time = st.selectbox("Departure Time", DEP_TIMES, index=1, key="sim_dep")
        sim_duration = st.slider("Est. Duration (hrs)", 1.0, 15.0, 2.5)
        sim_trigger = st.button("Explore Price Trends", type="primary")

    with sim_col2:
        if sim_trigger:
            with st.spinner("Executing matrix inference via REST API..."):
                scenarios = [{"airline": sim_airline, "source_city": sim_src, "destination_city": sim_dst,
                               "departure_time": sim_dep_time, "arrival_time": "Afternoon", "duration": sim_duration,
                               "days_left": d, "stops": "zero", "flight_class": c}
                             for c in ["Economy", "Business"] for d in range(1, 31)]
                batch_res = predict_batch(scenarios)

            if batch_res:
                st.success(f"Simulated {batch_res['total_scenarios']} scenarios in {batch_res['latency_seconds']:.4f}s")
                df_sim = pd.DataFrame({
                    "Days Left": list(range(1, 31)) * 2,
                    "Class": ["Economy"] * 30 + ["Business"] * 30,
                    "Price (Rs.)": batch_res['predicted_prices']
                })
                fig_sim, ax_sim = dark_fig(10, 5)
                sns.lineplot(data=df_sim, x="Days Left", y="Price (Rs.)", hue="Class",
                             palette=["#34d399", "#818cf8"], linewidth=2.5, marker="o", ax=ax_sim, markersize=6)
                ax_sim.set_title(f"Pricing Elasticity by Booking Window — {sim_src} → {sim_dst}",
                                 color="#e2e8f0", size=14, pad=15, weight='bold')
                ax_sim.invert_xaxis()
                ax_sim.set_xlabel("Days Before Departure", color='#94a3b8')
                ax_sim.set_ylabel("Predicted Fare (Rs.)", color='#94a3b8')
                leg = ax_sim.legend(frameon=True, facecolor='#1e2d4a', edgecolor='#263858')
                for t in leg.get_texts(): t.set_color('#e2e8f0')
                plt.tight_layout()
                st.pyplot(fig_sim)

                eco = df_sim[df_sim["Class"] == "Economy"]
                biz = df_sim[df_sim["Class"] == "Business"]
                s1, s2, s3 = st.columns(3)
                s1.metric("Economy — Best Fare", f"Rs. {eco['Price (Rs.)'].min():,.0f}", delta=f"Book {eco['Price (Rs.)'].idxmin()+1} days out")
                s2.metric("Business — Best Fare", f"Rs. {biz['Price (Rs.)'].min():,.0f}")
                s3.metric("Premium Uplift", f"Rs. {(biz['Price (Rs.)'].mean() - eco['Price (Rs.)'].mean()):,.0f}", delta="avg difference")
  except Exception as e:
      st.error(f"⚠️ Simulation Dashboard encountered an error: {e}")

# ===================== TAB 3: EDA =====================
with tabs[2]:
  try:
    st.header("Exploratory Data Analysis")
    st.markdown("Visualizing dataset patterns from `airlines_flights_data.csv`.")

    sb1, sb2 = st.columns(2)
    with sb1:
        eda_airline = st.multiselect("Filter Airline", df_original['airline'].unique())
    with sb2:
        eda_class = st.multiselect("Filter Class", df_original['class'].unique())

    flt = df_original.copy()
    if eda_airline: flt = flt[flt['airline'].isin(eda_airline)]
    if eda_class: flt = flt[flt['class'].isin(eda_class)]

    if len(flt) < 10:
        st.warning("Not enough data — adjust filters.")
    else:
        e1, e2 = st.columns(2)
        with e1:
            fig1, ax1 = dark_fig(6, 4)
            sns.scatterplot(data=flt.sample(min(1000, len(flt))), x="days_left", y="price",
                            alpha=0.6, color="#818cf8", ax=ax1, edgecolor="#080f1f", linewidth=0.3)
            ax1.set_title("Price vs Days Left", color="#e2e8f0", weight='bold')
            plt.tight_layout(); st.pyplot(fig1)
        with e2:
            fig2, ax2 = dark_fig(6, 4)
            sns.scatterplot(data=flt.sample(min(1000, len(flt))), x="duration", y="price",
                            alpha=0.6, color="#34d399", ax=ax2, edgecolor="#080f1f", linewidth=0.3)
            ax2.set_title("Price vs Duration", color="#e2e8f0", weight='bold')
            plt.tight_layout(); st.pyplot(fig2)

        flt_enc = flt.copy()
        for col in categorical_cols:
            flt_enc[col] = encoders[col].transform(flt_enc[col])
        fig3, ax3 = dark_fig(10, 4)
        sns.heatmap(flt_enc[["price","duration","days_left","stops","class"]].corr(),
                    annot=True, cmap="plasma", ax=ax3, linewidths=.5, annot_kws={"color":"white"})
        ax3.set_title("Correlation Heatmap", color="#d4d4d4", pad=15, weight='bold')
        ax3.tick_params(colors="#686868")
        plt.tight_layout(); st.pyplot(fig3)
  except Exception as e:
      st.error(f"⚠️ EDA tab encountered an error: {e}")

# ===================== TAB 4: ML MODELS =====================
with tabs[3]:
  try:
    st.header("Classical ML Benchmarks & Model Intelligence")

    # ── Live Model Info Panel (from API) ─────────────────────────────────
    model_info = _api_get("model-info")
    if model_info and "metrics" in model_info:
        m = model_info["metrics"]
        trained_raw = model_info.get("trained_at", "")
        try:
            trained_fmt = datetime.fromisoformat(trained_raw).strftime("%d %b %Y, %H:%M UTC")
        except Exception:
            trained_fmt = trained_raw

        st.subheader("🤖 Deployed Random Forest — Live Metrics")
        mi1, mi2, mi3, mi4 = st.columns(4)
        mi1.metric("R² Score",          f"{m.get('r2_score', '—'):.4f}")
        mi2.metric("MAE",               f"Rs. {m.get('mae', '—'):,.0f}")
        mi3.metric("R² Improvement",    f"+{m.get('improvement_r2_pct', '—'):.1f}%",
                   delta="vs Linear baseline")
        mi4.metric("MAE Reduction",     f"-{m.get('improvement_mae_pct', '—'):.1f}%",
                   delta="vs Linear baseline")
        st.caption(
            f"Model: **{model_info.get('model', 'RandomForest')}** &nbsp;·&nbsp; "
            f"Trained on **{model_info.get('training_samples', '—'):,}** samples &nbsp;·&nbsp; "
            f"Last trained: **{trained_fmt}**"
        )
        st.markdown("---")
    elif _health and not _health.get("model_loaded"):
        st.warning("API is offline — start the server to see live model metrics.")

    # ── Feature Importance Chart (from API) ──────────────────────────────
    fi_data = _api_get("feature-importance")
    if fi_data and "feature_importance" in fi_data:
        st.subheader("📊 Feature Importance — Random Forest")
        fi = fi_data["feature_importance"]
        fi_labels = list(fi.keys())
        fi_values = list(fi.values())

        fig_fi, ax_fi = dark_fig(9, 4)
        colors_fi = ["#d4d4d4" if i == 0 else "#505050" for i in range(len(fi_labels))]
        bars_fi = ax_fi.barh(fi_labels[::-1], fi_values[::-1], color=colors_fi[::-1],
                             height=0.6, edgecolor="none")
        for bar, val in zip(bars_fi, fi_values[::-1]):
            ax_fi.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                       f"{val:.3f}", va="center", color="#a8a8a8", fontsize=9, fontweight="600")
        ax_fi.set_xlabel("Importance Score", color="#686868")
        ax_fi.set_title("Feature Importance — What Drives Flight Prices?",
                        color="#d4d4d4", weight="bold", pad=14)
        ax_fi.set_xlim(0, max(fi_values) * 1.2)
        plt.tight_layout()
        st.pyplot(fig_fi)
        st.markdown("---")

    # ── Baseline Benchmarks ───────────────────────────────────────────────
    st.subheader("📐 Baseline Benchmarks — Linear & Logistic Regression")
    st.markdown("Trained locally on the dataset for comparison against the deployed model.")

    flt_ml = df.copy()
    if len(flt_ml) > 50:
        st.markdown("#### Price Prediction — Linear Regression")
        X_reg = flt_ml[["duration","days_left","stops","class"]]
        y_reg = flt_ml["price"]
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        sc = StandardScaler()
        lr = LinearRegression().fit(sc.fit_transform(Xr_tr), yr_tr)
        yr_pred = lr.predict(sc.transform(Xr_te))
        c1, c2 = st.columns(2)
        c1.metric("Linear R² Score", round(r2_score(yr_te, yr_pred), 3))
        c2.metric("Mean Absolute Error", f"Rs. {int(mean_absolute_error(yr_te, yr_pred)):,}")

        st.markdown("#### Flight Class Classification — Logistic Regression")
        X_clf = flt_ml[["price","duration","days_left","stops"]]
        y_clf = flt_ml["class"]
        if len(y_clf.unique()) > 1:
            Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
            sc2 = StandardScaler()
            clf = LogisticRegression(max_iter=1000).fit(sc2.fit_transform(Xc_tr), yc_tr)
            yc_pred = clf.predict(sc2.transform(Xc_te))
            st.metric("Classification Accuracy", f"{accuracy_score(yc_te, yc_pred)*100:.2f}%")
            with st.expander("Show Classification Report"):
                st.code(classification_report(yc_te, yc_pred), language="text")
  except Exception as e:
      st.error(f"⚠️ ML Models tab encountered an error: {e}")

# ===================== TAB 5: CLUSTERING =====================
with tabs[4]:
  try:
    st.header("Flight Segmentation — K-Means Clustering")
    st.markdown("Discovering natural flight clusters based on Price & Duration.")

    if len(df) > 10:
        n_clusters = st.slider("Number of Clusters (K)", 2, 6, 3)
        X_km = StandardScaler().fit_transform(df[["price","duration"]])
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_km)

        fig4, ax4 = dark_fig(8, 5)
        sns.scatterplot(x=df["duration"], y=df["price"], hue=labels, palette="cool",
                        ax=ax4, alpha=0.7, edgecolor="#111111", linewidth=0.2)
        ax4.set_title("Flight Clusters: Price vs Duration", color="#d4d4d4", weight='bold')
        ax4.set_xlabel("Duration (hrs)", color='#686868')
        ax4.set_ylabel("Price (Rs.)", color='#686868')
        leg4 = ax4.legend(title="Cluster", frameon=True, facecolor='#1e1e1e', edgecolor='#2e2e2e')
        plt.setp(leg4.get_title(), color='#d4d4d4')
        for t in leg4.get_texts(): t.set_color('#a8a8a8')
        plt.tight_layout(); st.pyplot(fig4)
  except Exception as e:
      st.error(f"⚠️ Clustering tab encountered an error: {e}")
