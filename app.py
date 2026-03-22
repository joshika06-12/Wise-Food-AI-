"""
Wise Food AI — Streamlit app (forecast pipeline unchanged).
"""
import hashlib
import json
import os
import random
from datetime import date, timedelta

import io
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

_BASE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_BASE, "assets")
_LOGO = os.path.join(_ASSETS, "wise_food_ai_logo.png")
_ICON = os.path.join(_ASSETS, "home_wise_food_app.png")
_IMG_SPREAD = os.path.join(_ASSETS, "home_wise_food_spread.png")
_IMG_APP = os.path.join(_ASSETS, "home_wise_food_app.png")

_page_icon = _LOGO if os.path.isfile(_LOGO) else (_ICON if os.path.isfile(_ICON) else "🍱")

st.set_page_config(
    page_title="Wise Food AI",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

USERS_FILE = os.path.join(_BASE, "wise_food_users.json")
MODEL_PATH = os.path.join(_BASE, "model.pkl")

IMG_SUSTAIN = "https://images.unsplash.com/photo-1542838132-92c53300491e?w=1000&auto=format&fit=crop&q=80"
IMG_COMPOST = "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=1000&auto=format&fit=crop&q=80"
IMG_STORAGE = "https://images.unsplash.com/photo-1556911220-bff31c812dba?w=1000&auto=format&fit=crop&q=80"

# Baseline story numbers + session additions
HOME_BASE_FOOD_KG = 1240.0
HOME_BASE_COST_INR = 385000.0
HOME_BASE_CO2_PCT = 18

st.markdown(
    """
    <style>
        .stApp, .main, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
            font-size: 1.14rem !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
            border-right: 1px solid #334155 !important;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #f8fafc !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] hr { border-color: #475569 !important; margin: 1rem 0 !important; }
        div.block-container {
            padding-top: 1.1rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1120px !important;
        }
        .main .block-container p, .main .block-container li {
            color: #0f172a !important;
            line-height: 1.6 !important;
            font-size: 1.08rem !important;
        }
        h1 {
            color: #0f172a !important;
            font-size: 2.55rem !important;
            font-weight: 800 !important;
            line-height: 1.15 !important;
        }
        h2 {
            color: #0f766e !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
        }
        h3 {
            color: #115e59 !important;
            font-size: 1.28rem !important;
            font-weight: 700 !important;
        }
        h4 { color: #134e4a !important; font-size: 1.12rem !important; font-weight: 700 !important; }
        [data-theme="dark"] .stApp, [data-theme="dark"] .main,
        [data-theme="dark"] [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
        }
        [data-theme="dark"] .main .block-container p,
        [data-theme="dark"] .main .block-container li {
            color: #f1f5f9 !important;
        }
        [data-theme="dark"] h1 { color: #f8fafc !important; }
        [data-theme="dark"] h2 { color: #5eead4 !important; }
        [data-theme="dark"] h3 { color: #2dd4bf !important; }
        [data-theme="dark"] h4 { color: #99f6e4 !important; }
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 14px rgba(15,23,42,0.06) !important;
        }
        [data-theme="dark"] div[data-testid="stMetric"] {
            background: #1e293b !important;
            border-color: #334155 !important;
        }
        [data-theme="dark"] div[data-testid="stMetric"] label { color: #cbd5e1 !important; }
        [data-theme="dark"] div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
        .wf-top {
            background: linear-gradient(95deg, #ffffff 0%, #ecfdf5 55%, #ccfbf1 100%);
            border: 1px solid #99f6e4;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 24px;
            box-shadow: 0 6px 24px rgba(15,118,110,0.1);
        }
        .wf-top .brand { font-size: 1.55rem; font-weight: 900; color: #0f766e; letter-spacing: -0.02em; }
        .wf-top .tag { font-size: 1.08rem; color: #115e59; margin-top: 8px; font-weight: 700; }
        [data-theme="dark"] .wf-top {
            background: linear-gradient(95deg, #1e293b 0%, #134e4a 100%);
            border-color: #2dd4bf;
        }
        [data-theme="dark"] .wf-top .brand { color: #f0fdfa !important; }
        [data-theme="dark"] .wf-top .tag { color: #99f6e4 !important; }
        .wf-hero {
            background: linear-gradient(145deg, #ffffff, #f0fdfa);
            border: 2px solid #14b8a6;
            border-radius: 18px;
            padding: 28px;
            text-align: center;
            margin: 8px 0 16px 0;
        }
        .wf-hero .label { font-size: 1.05rem; color: #0f766e; font-weight: 800; }
        .wf-hero .num { font-size: 3.2rem; font-weight: 900; color: #0f172a; line-height: 1.05; }
        .wf-hero .sub { font-size: 1.12rem; color: #134e4a; margin-top: 8px; font-weight: 600; }
        [data-theme="dark"] .wf-hero {
            background: linear-gradient(145deg, #1e293b, #134e4a);
            border-color: #2dd4bf;
        }
        [data-theme="dark"] .wf-hero .num { color: #f8fafc !important; }
        [data-theme="dark"] .wf-hero .label, [data-theme="dark"] .wf-hero .sub { color: #ccfbf1 !important; }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0d9488, #14b8a6) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 800 !important;
            font-size: 1.12rem !important;
            padding: 0.6rem 1.35rem !important;
            border-radius: 12px !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: #f1f5f9 !important;
            color: #000000 !important;
            border: 2px solid #0f766e !important;
            font-weight: 900 !important;
            font-size: 1.05rem !important;
        }
        [data-theme="dark"] [data-testid="stSidebar"] .stButton > button {
            background: #f8fafc !important;
            color: #000000 !important;
            border: 2px solid #14b8a6 !important;
            font-weight: 900 !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #ecfdf5 !important;
            color: #0f766e !important;
        }
        .wf-highlight-result {
            border: 3px solid #14b8a6;
            border-radius: 16px;
            padding: 20px 24px;
            background: linear-gradient(135deg, #f0fdfa, #ecfeff);
            margin: 12px 0;
        }
        [data-theme="dark"] .wf-highlight-result {
            background: linear-gradient(135deg, #134e4a, #1e293b);
            border-color: #2dd4bf;
        }
        .wf-hl-label { font-size:1.05rem;font-weight:800;color:#0f766e; }
        .wf-hl-num { font-size:2.85rem;font-weight:900;color:#0f172a;line-height:1.1; }
        .wf-hl-sub { font-size:1.1rem;font-weight:600;color:#134e4a;margin-top:6px; }
        [data-theme="dark"] .wf-hl-label { color:#5eead4 !important; }
        [data-theme="dark"] .wf-hl-num { color:#f8fafc !important; }
        [data-theme="dark"] .wf-hl-sub { color:#e2e8f0 !important; }
        hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.85rem 0 !important; }
        [data-theme="dark"] hr { border-top-color: #334155 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV_PAGES = [
    "🏠 Home",
    "📈 Predict Demand",
    "🔍 Spoilage Detection",
    "📅 Expiry Tracker",
    "📦 Smart Inventory & Menu",
    "📊 Analytics",
    "🌱 Sustainability",
    "🤝 Donation",
    "❄️ Storage Assistant",
    "⭐ Feedback",
]


def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()


def load_users() -> dict:
    if not os.path.isfile(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(data: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def init_session():
    defaults = [
        ("logged_in", False),
        ("username", ""),
        ("pending_pin", None),
        ("prediction_bundle", None),
        ("feedback_log", []),
        ("demo_avg_rating", 4.6),
        ("_show_predict_results", False),
        ("cum_pred_food_kg", 0.0),
        ("cum_pred_cost_inr", 0.0),
        ("cum_pred_co2_kg", 0.0),
        ("cum_inv_food_kg", 0.0),
        ("cum_inv_cost_inr", 0.0),
        ("expiry_items", []),
        ("inventory_items", []),
        ("menu_weekly", ""),
        ("menu_daily", ""),
        ("compost_daily_income", []),
        ("pred_explain_text", ""),
        ("pred_explain_audio", None),
        ("expiry_alarm_fired", False),
    ]
    for k, v in defaults:
        if k not in st.session_state:
            st.session_state[k] = v


def home_image_paths():
    spread = _IMG_SPREAD if os.path.isfile(_IMG_SPREAD) else None
    app = _IMG_APP if os.path.isfile(_IMG_APP) else None
    return spread, app


def display_home_headline_metrics():
    """Home headline totals track forecast runs (updates after each estimate)."""
    food = HOME_BASE_FOOD_KG + st.session_state.cum_pred_food_kg
    cost = int(HOME_BASE_COST_INR + st.session_state.cum_pred_cost_inr)
    co2_pct = HOME_BASE_CO2_PCT + min(22, int(st.session_state.cum_pred_co2_kg // 12))
    co2_pct = min(42, co2_pct)
    return food, cost, co2_pct


def display_running_metrics():
    food = HOME_BASE_FOOD_KG + st.session_state.cum_pred_food_kg + st.session_state.cum_inv_food_kg
    cost = int(HOME_BASE_COST_INR + st.session_state.cum_pred_cost_inr + st.session_state.cum_inv_cost_inr)
    co2_pct = HOME_BASE_CO2_PCT + min(15, int(st.session_state.cum_pred_co2_kg // 20) + int(st.session_state.cum_inv_food_kg // 5))
    co2_pct = min(45, co2_pct)
    return food, cost, co2_pct


def render_top_header():
    if os.path.isfile(_LOGO):
        c_logo, c_txt = st.columns([1, 5])
        with c_logo:
            st.image(_LOGO, width=100)
        with c_txt:
            st.markdown(
                """
                <div class="wf-top">
                    <div class="brand">Wise Food AI</div>
                    <div class="tag">Predict • Prepare • Prevent Waste</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="wf-top">
                <div class="brand">Wise Food AI</div>
                <div class="tag">Predict • Prepare • Prevent Waste</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_title_block(title: str, desc: str):
    st.markdown(f"## {title}")
    st.markdown(desc)
    st.markdown("---")


def auth_screen():
    render_top_header()
    if os.path.isfile(_LOGO):
        st.image(_LOGO, width=140)
    st.markdown("### Wise Food AI")
    st.caption("Sign in to continue")
    tab_signup, tab_login = st.tabs(["Create account", "Sign in"])
    with tab_signup:
        st.markdown("Choose a username. You’ll receive a **4-digit PIN** — save it.")
        su_user = st.text_input("Username", key="su_user", placeholder="Your name")
        if st.button("Create account", type="primary", key="btn_signup"):
            if not su_user or not su_user.strip():
                st.error("Please enter a username.")
            else:
                users = load_users()
                uname = su_user.strip().lower()
                if uname in users:
                    st.error("That username is already taken.")
                else:
                    pin = f"{random.randint(0, 9999):04d}"
                    users[uname] = {"pin_hash": _hash_pin(pin)}
                    save_users(users)
                    st.session_state.pending_pin = pin
                    st.session_state["_show_pin"] = uname
                    st.success("You’re registered!")
        if st.session_state.get("_show_pin"):
            st.info(
                f"**Username:** `{st.session_state['_show_pin']}`  \n**PIN:** `{st.session_state.pending_pin}`"
            )
    with tab_login:
        lg_user = st.text_input("Username", key="lg_user")
        lg_pin = st.text_input("PIN", type="password", key="lg_pin", max_chars=4)
        if st.button("Sign in", type="primary", key="btn_login"):
            users = load_users()
            u = (lg_user or "").strip().lower()
            if not u or not lg_pin:
                st.error("Enter username and PIN.")
            elif u not in users:
                st.error("No account found.")
            elif users[u].get("pin_hash") != _hash_pin(lg_pin.strip()):
                st.error("Incorrect PIN.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error("Forecasts are unavailable right now. Please contact your administrator.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_features(price: float, date, orders_last_week: int, orders_earlier: int, day_type: str):
    checkout_price = float(price)
    base_price = checkout_price * 1.0667
    month = date.month
    weekofyear = date.isocalendar()[1]
    if day_type == "Weekend":
        dayofweek = 5
        is_weekend = 1
    else:
        dayofweek = 2
        is_weekend = 0
    lag_1 = orders_last_week
    lag_2 = orders_earlier
    rolling_mean = (lag_1 + lag_2) / 2
    return np.array(
        [
            [
                checkout_price,
                base_price,
                month,
                weekofyear,
                dayofweek,
                is_weekend,
                lag_1,
                lag_2,
                rolling_mean,
            ]
        ]
    )


def run_forecast(model, input_data):
    return float(model.predict(input_data)[0])


def pnl_from_prediction(prediction: float, price: float, cost_per_order: float = 85.0):
    orders = max(0.0, prediction)
    revenue = orders * price
    cost = orders * cost_per_order
    profit = revenue - cost
    return {
        "estimated_revenue": round(revenue, 2),
        "estimated_cost": round(cost, 2),
        "profit_or_loss": round(profit, 2),
        "cost_per_order": cost_per_order,
    }


def storage_recommendation(food_item: str):
    t = (food_item or "").lower()
    if any(k in t for k in ["leaf", "salad", "lettuce", "spinach", "herb"]):
        return "Leafy greens", "0–4°C in the crisper", "High (about 90–95%)"
    if any(k in t for k in ["milk", "dairy", "curd", "paneer", "cheese"]):
        return "Dairy", "2–5°C", "Moderate (about 80–85%)"
    if any(k in t for k in ["rice", "biryani", "grain", "pulse", "dal"]):
        return "Cooked grains & pulses", "Chill within 2 hours, then keep at 4°C or below", "Moderate (about 70–75%)"
    if any(k in t for k in ["fruit", "apple", "banana", "citrus"]):
        return "Fruit", "Cool, dry place or fridge depending on the fruit", "Varies (often 85–90%)"
    if any(k in t for k in ["meat", "chicken", "fish", "egg"]):
        return "Protein", "Keep at 0–4°C; use within safe dates", "Not too dry"
    return "Prepared food (general)", "Refrigerate at 4°C or below", "Moderate (about 75–80%)"


def manure_estimate_kg_waste(prediction: float, food_item: str) -> float:
    base = max(0, 120 - prediction) * 0.15
    if any(k in (food_item or "").lower() for k in ["veg", "fruit", "salad"]):
        base *= 1.2
    return round(base, 2)


def surplus_detected(prediction: float) -> bool:
    return prediction < 100


def waste_risk_label(prediction: float) -> str:
    if prediction < 100:
        return "high"
    if prediction <= 200:
        return "medium"
    return "low"


def smart_prep_text(pred: float, food_item: str) -> str:
    fi = food_item or "this dish"
    if pred < 100:
        return f"Try preparing closer to **{max(5, int(pred * 0.9))}** servings of **{fi}**."
    if pred < 200:
        return f"A good range is about **{int(pred * 0.95)}–{int(pred * 1.05)}** servings of **{fi}**."
    return f"Demand looks strong — plan for at least **{int(pred)}** orders of **{fi}**."


def friendly_why_sentence(price: float, day_type: str, prediction: float) -> str:
    return (
        f"We combined your **₹{price:.0f}** price, a **{day_type.lower()}**, and your **recent sales pattern** — "
        f"that suggests about **{int(prediction)}** orders."
    )


def build_prediction_explanation(b: dict) -> str:
    parts = []
    if b.get("day_type") == "Weekend":
        parts.append("Demand tends to be higher because weekends usually bring more guests.")
    else:
        parts.append("Weekdays often mean steadier, moderate traffic.")
    price = float(b.get("price", 0))
    if price <= 180:
        parts.append("Your price feels approachable, which often encourages more orders.")
    else:
        parts.append("A higher price can mean slightly fewer orders — keep portions tight.")
    tl = int(b.get("orders_last", 0))
    tp = int(b.get("orders_prev", 0))
    if tl > tp:
        parts.append("The last two weeks suggest demand is picking up.")
    elif tl < tp:
        parts.append("The last two weeks look a bit softer than before — a little less prep may help.")
    else:
        parts.append("Recent demand has been fairly steady.")
    parts.append(f"All together, planning for about **{int(b['prediction'])}** orders is a good fit.")
    return "\n\n".join(parts)


def synthesize_tts(text: str):
    plain = "".join(c for c in text if c.isalnum() or c in " .,+-")
    plain = (plain[:380] + "...") if len(plain) > 380 else plain
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=plain or "Here is your summary.", lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def play_expiry_alarm_5s():
    html = """
    <div style="height:0;width:0;overflow:hidden;">
    <script>
    (function(){
      try {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        const ctx = new AC();
        const stopAt = ctx.currentTime + 5;
        function ping(t) {
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.type = 'sine';
          o.frequency.value = 880;
          o.connect(g);
          g.connect(ctx.destination);
          g.gain.setValueAtTime(0.0001, t);
          g.gain.exponentialRampToValueAtTime(0.2, t + 0.02);
          g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
          o.start(t);
          o.stop(t + 0.13);
        }
        let t = ctx.currentTime;
        while (t < stopAt - 0.15) { ping(t); t += 0.28; }
        setTimeout(function(){ try { ctx.close(); } catch(e) {} }, 5200);
      } catch(e) {}
    })();
    </script>
    </div>
    """
    components.html(html, height=1, scrolling=False)


def compost_nursery_forecast() -> tuple[float, list]:
    hist = st.session_state.get("compost_daily_income") or []
    b = st.session_state.get("prediction_bundle")
    cur = float(b["manure_income"]) if b and b.get("manure_income") is not None else None
    if not hist and cur is None:
        return 0.0, hist
    recent = [float(x["income"]) for x in hist[-7:]]
    avg = sum(recent) / len(recent) if recent else (cur or 0.0)
    if cur is None:
        cur = avg
    predicted = round(0.5 * float(avg) + 0.5 * float(cur), 0)
    return predicted, hist


DUMMY_NGOS = [
    {"name": "Annapurna Community Kitchen", "lat": 12.978, "lon": 77.606, "area": "2.1 km"},
    {"name": "Hope Shelter Meals", "lat": 12.962, "lon": 77.582, "area": "3.4 km"},
    {"name": "Green Plate NGO", "lat": 12.955, "lon": 77.598, "area": "4.0 km"},
]


def donation_route_figure():
    kitchen_lat, kitchen_lon = 12.9716, 77.5946
    lats = [kitchen_lat] + [n["lat"] for n in DUMMY_NGOS] + [kitchen_lat]
    lons = [kitchen_lon] + [n["lon"] for n in DUMMY_NGOS] + [kitchen_lon]
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines+markers+text",
            line=dict(width=3, color="#ea580c"),
            marker=dict(size=10, color=["#0f766e"] + ["#f97316"] * len(DUMMY_NGOS) + ["#0f766e"]),
            text=["You"] + [n["name"][:12] + "…" for n in DUMMY_NGOS] + ["You"],
            textposition="top center",
        )
    )
    fig.update_geos(
        projection_type="mercator",
        showland=True,
        landcolor="#f1f5f9",
        showocean=True,
        oceancolor="#e0f2fe",
        lataxis_range=[12.92, 13.02],
        lonaxis_range=[77.55, 77.63],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="Sample pickup route (GPS-style path)",
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", size=13),
    )
    return fig


LEFTOVER_GUIDE = [
    ("Rice → Fried Rice", "https://www.youtube.com/results?search_query=leftover+rice+recipes"),
    ("Chapati → Kothu Parotta / Noodles", "https://www.youtube.com/results?search_query=chapati+leftover+recipes"),
    ("Bread → Bread Upma / Pizza", "https://www.youtube.com/results?search_query=bread+leftover+recipes"),
    ("Dal → Dal Paratha", "https://www.youtube.com/results?search_query=leftover+dal+paratha+recipe"),
    ("Vegetables → Cutlets", "https://www.youtube.com/results?search_query=leftover+vegetable+cutlets+recipe"),
]

COMPOST_LINKS = [
    ("How to compost kitchen waste", "https://www.youtube.com/results?search_query=how+to+make+compost+from+kitchen+waste"),
    ("Composting vegetable peels at home", "https://www.youtube.com/results?search_query=vegetable+peel+composting+at+home"),
    ("Easy organic compost", "https://www.youtube.com/results?search_query=organic+compost+making+easy"),
]


def waste_risk_simple(prediction: float):
    r = waste_risk_label(prediction)
    st.markdown("### Waste risk")
    if r == "high":
        st.error("**Higher chance of leftovers** — consider cooking a bit less.")
    elif r == "medium":
        st.warning("**Balanced** — watch portions during service.")
    else:
        st.success("**Stronger demand** — less risk of unsold food if you plan well.")


def demand_chart_simple(prediction: float):
    labels = ["Quieter", "Steady", "Busy"]
    colors = (
        ["#5eead4", "#cbd5e1", "#cbd5e1"]
        if prediction <= 100
        else ["#cbd5e1", "#fbbf24", "#cbd5e1"]
        if prediction <= 200
        else ["#cbd5e1", "#cbd5e1", "#fb7185"]
    )
    fig = go.Figure(
        data=[go.Bar(x=labels, y=[100, 200, 300], marker_color=colors, showlegend=False)]
    )
    fig.add_hline(y=prediction, line_dash="dash", line_color="#0f766e", annotation_text=f"You: ~{int(prediction)}")
    fig.update_layout(
        title="How busy things might feel",
        height=280,
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", size=13),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def sidebar_nav():
    if os.path.isfile(_LOGO):
        st.sidebar.image(_LOGO, use_container_width=True)
    st.sidebar.markdown("### Wise Food AI")
    st.sidebar.markdown(
        "<p style='font-size:1.02rem;line-height:1.55;font-weight:600;color:#f8fafc;'>"
        "Plan smarter. Waste less. Do good.</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Menu", NAV_PAGES, key="nav_radio", label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<style>.stButton > button { color: #0f766e !important; }</style>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Log out", key="logout_btn", use_container_width=True):
        keys_reset = [
            "prediction_bundle",
            "_show_predict_results",
            "cum_pred_food_kg",
            "cum_pred_cost_inr",
            "cum_pred_co2_kg",
            "cum_inv_food_kg",
            "cum_inv_cost_inr",
            "expiry_items",
            "inventory_items",
            "menu_weekly",
            "menu_daily",
            "feedback_log",
            "demo_avg_rating",
            "compost_daily_income",
            "pred_explain_text",
            "pred_explain_audio",
            "expiry_alarm_fired",
        ]
        st.session_state.logged_in = False
        st.session_state.username = ""
        for k in keys_reset:
            st.session_state.pop(k, None)
        st.rerun()
    st.sidebar.markdown(
        f"<p style='color:#e2e8f0;font-weight:600;'>Signed in: {st.session_state.username}</p>",
        unsafe_allow_html=True,
    )
    return page


def page_home():
    render_top_header()
    page_title_block(
        "Home",
        "**Our mission is to reduce food waste using smart, data-driven insights and create a sustainable future.**",
    )
    st.markdown("### Predict • Prepare • Prevent Waste")
    st.markdown(
        "When portions don’t match real demand, food and money are lost. "
        "**Wise Food AI** helps you plan ahead, store wisely, and share surplus with care."
    )
    st.markdown("---")
    fk, ck, co2p = display_home_headline_metrics()
    b = st.session_state.get("prediction_bundle")
    d_f = b.get("food_saved_kg") if b else None
    d_c = int(b.get("money_saved", 0)) if b else None
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Food saved (kg)", f"{fk:,.0f}", delta=f"+{d_f:.1f} from last estimate" if d_f is not None else None)
    with c2:
        st.metric("Cost saved (₹)", f"{ck:,}", delta=f"+{d_c:,}" if d_c is not None else None)
    with c3:
        st.metric(
            "CO₂ reduced (%)",
            f"{co2p}%",
            delta="Updated from latest estimate" if b else None,
        )
    st.markdown("---")
    spread, appimg = home_image_paths()
    ic1, ic2 = st.columns(2, gap="medium")
    with ic1:
        if spread:
            st.image(spread, use_container_width=True, caption="Beautiful food deserves smart planning — not waste.")
        else:
            st.info("Add your hero image to the **assets** folder.")
    with ic2:
        if appimg:
            st.image(appimg, use_container_width=True, caption="Manage kitchens from your phone — clear, simple, actionable.")
    st.success(f"Welcome **{st.session_state.username}** — start with **Predict Demand** or explore the menu.")


def page_predict(model):
    render_top_header()
    page_title_block(
        "Predict Demand",
        "Tell us about your dish. Complete the quick checklist, then get an estimate for how many orders to expect.",
    )

    with st.container():
        st.markdown("### Daily inventory checklist")
        st.caption("Complete all items before requesting an estimate.")
        c1, c2, c3 = st.columns(3)
        with c1:
            chk1 = st.checkbox("Used older stock first today", key="pred_chk_fifo")
        with c2:
            chk2 = st.checkbox("Checked expiry dates today", key="pred_chk_exp")
        with c3:
            chk3 = st.checkbox("Reviewed items nearing expiry", key="pred_chk_near")
        checklist_ok = chk1 and chk2 and chk3
        if not checklist_ok:
            st.warning("Please tick **all three** checklist items to unlock **Get estimate**.")

    st.markdown("---")

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            dish = st.text_input("Dish name", placeholder="e.g. Veg thali")
            price = st.number_input("Typical selling price (₹)", min_value=10.0, value=150.0)
        with c2:
            day = st.date_input("Which day are you planning for?", key="pd_date")
            day_type = st.radio("Weekday or weekend?", ["Weekday", "Weekend"], horizontal=True)

        st.markdown("##### Ingredients guests might skip (optional)")
        remove_ing = st.text_input(
            "Ingredients to offer without (e.g. onion, nuts)",
            placeholder="Reduces plate waste when guests opt out",
            key="ing_remove",
        )

        st.markdown("##### Recent sales")
        oc1, oc2 = st.columns(2)
        with oc1:
            orders_last = st.number_input("About how many orders last week?", min_value=0, value=100, step=1)
        with oc2:
            orders_prev = st.number_input("About how many the week before?", min_value=0, value=90, step=1)

    st.markdown("---")
    input_data = build_features(price, day, orders_last, orders_prev, day_type)

    if st.button("Get estimate", type="primary", use_container_width=True, disabled=not checklist_ok):
        with st.spinner("Analyzing demand using AI..."):
            prediction = run_forecast(model, input_data)

        pnl = pnl_from_prediction(prediction, price)
        food_saved = round((200 - prediction) / 10, 2)
        money_saved = round(food_saved * 150, 2)
        co2_kg = round(food_saved * 2.5, 2)
        waste_pct = min(35, max(5, int(25 - prediction / 20)))
        cat, temp, hum = storage_recommendation(dish)
        spoil_kg = manure_estimate_kg_waste(prediction, dish)
        manure_income = round(spoil_kg * 12, 2)

        st.session_state.cum_pred_food_kg += max(0, food_saved)
        st.session_state.cum_pred_cost_inr += max(0, money_saved)
        st.session_state.cum_pred_co2_kg += max(0, co2_kg)

        entry = {"day": date.today().isoformat(), "income": float(manure_income)}
        st.session_state.compost_daily_income = (st.session_state.compost_daily_income + [entry])[-60:]

        st.session_state.prediction_bundle = {
            "prediction": prediction,
            "food_item": dish,
            "price": price,
            "remove_ing": remove_ing,
            "people": 4,
            "pnl": pnl,
            "food_saved_kg": food_saved,
            "money_saved": money_saved,
            "co2_kg": co2_kg,
            "waste_pct": waste_pct,
            "storage": {"category": cat, "temp": temp, "humidity": hum},
            "spoil_kg": spoil_kg,
            "manure_income": manure_income,
            "surplus": surplus_detected(prediction),
            "waste_risk": waste_risk_label(prediction),
            "day_type": day_type,
            "orders_last": orders_last,
            "orders_prev": orders_prev,
        }
        st.session_state.pred_explain_text = ""
        st.session_state.pred_explain_audio = None
        st.session_state._show_predict_results = True
        st.rerun()

    if st.session_state.get("_show_predict_results") and st.session_state.prediction_bundle:
        b = st.session_state.prediction_bundle
        st.markdown("---")
        st.success("Here is your demand estimate — review the highlight below.")
        st.markdown("### Your estimate")
        with st.container():
            st.markdown(
                f"""
                <div class="wf-highlight-result">
                  <div style="text-align:center;">
                    <div class="wf-hl-label">Estimated food demand</div>
                    <div class="wf-hl-num">{int(b["prediction"])}</div>
                    <div class="wf-hl-sub">orders · {b.get("food_item") or "your dish"}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        waste_risk_simple(b["prediction"])
        st.markdown("### Suggestion")
        st.markdown(smart_prep_text(b["prediction"], b.get("food_item", "")))
        if b.get("remove_ing"):
            st.info(f"When guests skip **{b['remove_ing']}**, offer smaller portions so less is thrown away.")
        st.markdown("---")
        st.markdown("### Why this number?")
        st.markdown(friendly_why_sentence(b["price"], b.get("day_type", "Weekday"), b["prediction"]))
        st.markdown("---")
        demand_chart_simple(b["prediction"])

        st.markdown("### How this plan helps")
        st.markdown("A quick snapshot from this run:")
        p1, p2, p3 = st.columns(3)
        people_fed = max(50, int(b["prediction"] * 1.15))
        with p1:
            st.metric("👨‍👩‍👧 People fed (est.)", f"{people_fed}+")
        with p2:
            st.metric("🍱 Food saved (this run, kg)", f"{b['food_saved_kg']}")
        with p3:
            st.metric("🌍 Waste reduced (score)", f"{b['waste_pct']}%")

        st.markdown("---")
        if st.button("👉 Explain this prediction", key="btn_explain_pred"):
            ex = build_prediction_explanation(b)
            st.session_state.pred_explain_text = ex
            st.session_state.pred_explain_audio = synthesize_tts(ex.replace("\n", " "))
            st.rerun()

        if st.session_state.pred_explain_text:
            st.markdown("### In plain words")
            for para in st.session_state.pred_explain_text.split("\n\n"):
                st.markdown(para)
            if st.session_state.pred_explain_audio:
                st.caption("Short voice summary — press play:")
                st.audio(st.session_state.pred_explain_audio, format="audio/mp3")
            else:
                st.caption("For voice: install **gTTS** with `pip install gtts` (needs internet once).")
            if st.button("🔊 Play explanation again", key="btn_replay_tts"):
                st.session_state.pred_explain_audio = synthesize_tts(
                    st.session_state.pred_explain_text.replace("\n", " ")
                )
                st.rerun()

        st.markdown("---")
        st.progress(min(1.0, max(0.06, b["waste_pct"] / 100)))
        st.caption(f"Rough waste-avoidance score vs guessing: **~{b['waste_pct']}%** (illustrative).")


def page_spoilage():
    render_top_header()
    page_title_block(
        "Spoilage Detection",
        "Upload a photo or use your camera for a quick freshness check (demo — not a medical or safety certificate).",
    )

    with st.container():
        up = st.file_uploader("Upload a food photo", type=["png", "jpg", "jpeg", "webp"])
        cam = st.camera_input("Or use your camera", key="spoil_cam")

    if st.button("Check freshness", type="primary", use_container_width=True):
        data = None
        if up is not None:
            data = up.getvalue()
        elif cam is not None:
            data = cam.getvalue()
        if not data:
            st.warning("Please upload a photo or capture one with the camera first.")
        else:
            fresh = random.random() > 0.35
            conf = round(random.uniform(0.78, 0.97), 2)
            label = "Fresh" if fresh else "Spoiled"
            st.balloons()
            st.markdown(f"### Result: **{label}**")
            st.metric("Confidence", f"{int(conf * 100)}%")
            st.caption("This is a simulated result for your demo.")


def expiry_row_category(expiry_d: date):
    today = date.today()
    if expiry_d < today:
        return "red", (expiry_d - today).days
    days_left = (expiry_d - today).days
    if days_left <= 2:
        return "red", days_left
    if days_left <= 14:
        return "yellow", days_left
    return "green", days_left


def page_expiry():
    render_top_header()
    page_title_block("Expiry Tracker", "Add items and expiry dates. We’ll highlight what needs attention.")

    with st.form("add_expiry", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            ename = st.text_input("Item name")
        with c2:
            ed = st.date_input("Expiry date")
        if st.form_submit_button("Add item"):
            if ename and ename.strip():
                st.session_state.expiry_items.append({"name": ename.strip(), "expiry": ed})
                st.session_state.expiry_alarm_fired = False
                st.success("Item added.")
            else:
                st.error("Enter an item name.")

    st.markdown("---")
    if not st.session_state.expiry_items:
        st.info("No items yet — add your first one above.")
        st.session_state.expiry_alarm_fired = False
    else:
        any_red = False
        for row in st.session_state.expiry_items:
            cat, days_left = expiry_row_category(row["expiry"])
            if cat == "red":
                any_red = True
                break
        if not any_red:
            st.session_state.expiry_alarm_fired = False
        if any_red:
            st.error("⚠️ **Alert:** One or more items are expired or within 2 days of expiry — use or donate soon.")
        if any_red and not st.session_state.expiry_alarm_fired:
            play_expiry_alarm_5s()
            st.session_state.expiry_alarm_fired = True

        for row in st.session_state.expiry_items:
            cat, days_left = expiry_row_category(row["expiry"])
            if cat == "red":
                st.markdown(
                    f'<p style="color:#b91c1c;font-weight:800;font-size:1.08rem;">🔴 <b>{row["name"]}</b> — expiry '
                    f'<b>{row["expiry"]}</b> · {"EXPIRED" if days_left < 0 else "Near expiry — act today"}</p>',
                    unsafe_allow_html=True,
                )
            elif cat == "yellow":
                st.markdown(
                    f'<p style="color:#a16207;font-weight:700;font-size:1.05rem;">🟡 <b>{row["name"]}</b> — expiry '
                    f'<b>{row["expiry"]}</b> · {days_left} day(s) left — plan to use this week.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p style="color:#15803d;font-weight:700;font-size:1.05rem;">🟢 <b>{row["name"]}</b> — expiry '
                    f'<b>{row["expiry"]}</b> · Safe to use — more than two weeks of shelf time.</p>',
                    unsafe_allow_html=True,
                )

        to_drop = st.selectbox("Remove an item", ["—"] + [x["name"] for x in st.session_state.expiry_items])
        if to_drop != "—" and st.button("Remove selected item"):
            st.session_state.expiry_items = [x for x in st.session_state.expiry_items if x["name"] != to_drop]
            st.session_state.expiry_alarm_fired = False
            st.rerun()


def inv_stock_label(qty: float, low: float = 3) -> str:
    if qty <= 0:
        return "Out of stock"
    if qty < low:
        return "Low stock"
    return "Available"


def suggest_meals_from_inv(items):
    names = " ".join((i["name"] or "").lower() for i in items)
    ideas = []
    if "rice" in names or "biryani" in names:
        ideas.append("Fried rice or lemon rice using leftover rice")
    if "tomato" in names or "veg" in names:
        ideas.append("Mixed vegetable curry or soup")
    if "paneer" in names or "cheese" in names:
        ideas.append("Grilled wrap or light curry with paneer")
    if "dal" in names or "pulse" in names:
        ideas.append("Dal soup or stuffed paratha")
    if not ideas:
        ideas.append("Stir-fry bowl with whatever vegetables you have")
        ideas.append("One-pot khichdi for a simple crowd-pleaser")
    return ideas


def bump_inv_impact(delta_food: float, delta_cost: float):
    st.session_state.cum_inv_food_kg += delta_food
    st.session_state.cum_inv_cost_inr += delta_cost


def page_smart_inventory():
    render_top_header()
    page_title_block(
        "Smart Inventory & Menu",
        "Track what you have, mark items as used, and sketch a simple menu plan.",
    )

    with st.form("add_inv", clear_on_submit=True):
        a1, a2, a3 = st.columns([2, 1, 1])
        with a1:
            iname = st.text_input("Item name", placeholder="e.g. Tomatoes")
        with a2:
            iqty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.5)
        with a3:
            iunit = st.text_input("Unit", value="kg")
        if st.form_submit_button("Add to stock"):
            if iname and iname.strip():
                st.session_state.inventory_items.append(
                    {"name": iname.strip(), "qty": iqty, "unit": iunit or "units"}
                )
                bump_inv_impact(0.2, 50)
                st.success("Added.")
            else:
                st.error("Enter an item name.")

    st.markdown("---")
    st.markdown("### Stock")
    if not st.session_state.inventory_items:
        st.info("No stock lines yet.")
    else:
        for idx, it in enumerate(st.session_state.inventory_items):
            col_a, col_b, col_c = st.columns([3, 2, 2])
            with col_a:
                st.markdown(f"**{it['name']}** · {it['qty']} {it['unit']}")
            with col_b:
                st.markdown(f"**{inv_stock_label(it['qty'])}**")
            with col_c:
                use_amt = st.number_input("Use amount", min_value=0.0, value=0.0, key=f"use_{idx}")
                if st.button("Apply use", key=f"btn_use_{idx}"):
                    cur = st.session_state.inventory_items[idx]["qty"]
                    st.session_state.inventory_items[idx]["qty"] = max(0.0, cur - use_amt)
                    bump_inv_impact(0.15, 40)
                    st.rerun()

    st.markdown("---")
    st.markdown("### Menu planning")
    st.session_state.menu_daily = st.text_area("Today’s menu (notes)", value=st.session_state.menu_daily, height=80)
    st.session_state.menu_weekly = st.text_area("This week’s ideas", value=st.session_state.menu_weekly, height=80)

    st.markdown("---")
    st.markdown("### Meal ideas from your stock")
    for idea in suggest_meals_from_inv(st.session_state.inventory_items):
        st.markdown(f"- {idea}")


def page_analytics():
    render_top_header()
    page_title_block("Analytics", "Trends and money — plus your running impact totals.")

    fk, ck, co2p = display_running_metrics()
    m1, m2, m3 = st.columns(3)
    m1.metric("Running food saved (kg)", f"{fk:,.0f}")
    m2.metric("Running cost saved (₹)", f"{ck:,}")
    m3.metric("Running CO₂ reduced (%)", f"{co2p}%")
    st.markdown("---")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    demand_trend = [120, 132, 128, 140, 190, 210, 198]
    fig_line = px.line(x=days, y=demand_trend, markers=True, title="Order trend (sample week)")
    fig_line.update_traces(line=dict(color="#0d9488", width=3))
    fig_line.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(color="#0f172a"))

    items = ["Thali", "Biryani", "Salad", "Soup", "Dessert"]
    volumes = [180, 220, 90, 70, 150]
    fig_bar = px.bar(
        x=items,
        y=volumes,
        title="Popular dishes (sample)",
        color=volumes,
        color_continuous_scale="Teal",
    )
    fig_bar.update_layout(height=360, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.plotly_chart(fig_line, use_container_width=True)
    with g2:
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Money snapshot")
    b = st.session_state.prediction_bundle
    if b:
        p = b["pnl"]
        n1, n2, n3 = st.columns(3)
        n1.metric("Expected sales (₹)", f"{p['estimated_revenue']:,.0f}")
        n2.metric("Expected costs (₹)", f"{p['estimated_cost']:,.0f}")
        n3.metric("Expected profit (₹)", f"{p['profit_or_loss']:+,.0f}")
    else:
        st.info("Run **Predict Demand** to see numbers from your latest estimate.")


def page_sustainability():
    render_top_header()
    page_title_block("Sustainability", "Stretch food further and keep waste out of landfills.")
    b = st.session_state.prediction_bundle

    # Initialize session state for sustainability inputs
    if "leftover_food_input" not in st.session_state:
        st.session_state.leftover_food_input = ""
    if "waste_amount_grams" not in st.session_state:
        st.session_state.waste_amount_grams = 0.0
    if "waste_type" not in st.session_state:
        st.session_state.waste_type = "Vegetable peels"

    with st.container():
        st.markdown("### Leftover reuse")
        st.markdown("**What leftover food do you have?**")
        leftover_input = st.text_input("Enter leftover food (e.g., rice, chapati, dal)", key="leftover_food_input_field", placeholder="Type your leftover food...")
        
        if leftover_input and leftover_input.strip():
            leftover_lower = leftover_input.lower().strip()
            matched = False
            for title, url in LEFTOVER_GUIDE:
                if any(key in leftover_lower for key in ["rice", "fried rice"] if "rice" in title.lower()):
                    st.markdown(f"**✨ Matching recipe:** [{title}]({url})")
                    matched = True
                    break
                elif any(key in leftover_lower for key in ["chapati", "roti", "bread"] if "chapati" in title.lower() or "bread" in title.lower()):
                    st.markdown(f"**✨ Matching recipe:** [{title}]({url})")
                    matched = True
                    break
                elif any(key in leftover_lower for key in ["dal", "pulse"] if "dal" in title.lower()):
                    st.markdown(f"**✨ Matching recipe:** [{title}]({url})")
                    matched = True
                    break
                elif any(key in leftover_lower for key in ["veg", "vegetable"] if "vegetable" in title.lower()):
                    st.markdown(f"**✨ Matching recipe:** [{title}]({url})")
                    matched = True
                    break
            
            if not matched:
                st.markdown("**Other leftover ideas:**")
                for title, url in LEFTOVER_GUIDE:
                    st.markdown(f"- **{title}** — [YouTube ideas]({url})")
        else:
            st.markdown("**Available recipes:**")
            for title, url in LEFTOVER_GUIDE:
                st.markdown(f"- **{title}** — [YouTube ideas]({url})")

    st.markdown("---")
    with st.container():
        st.markdown("### Composting")
        st.markdown(
            "Peels and scraps can become **rich compost** — better soil, fewer emissions, "
            "and sometimes **extra income** selling to nurseries."
        )
        st.image(IMG_COMPOST, use_container_width=True, caption="From kitchen scraps to garden gold.")
        
        st.markdown("**Get your composting video guide:**")
        for label, url in COMPOST_LINKS:
            st.markdown(f"- [{label}]({url})")
        
        st.markdown("---")
        st.markdown("**Calculate your compost income — tell us about your waste:**")
        
        col_waste1, col_waste2 = st.columns(2)
        with col_waste1:
            waste_grams = st.number_input("Waste amount (grams)", min_value=0.0, value=500.0, step=50.0, key="waste_grams_input")
        with col_waste2:
            waste_type = st.selectbox(
                "Type of waste",
                ["Vegetable peels", "Onion peel", "Egg shell", "Fruit scraps", "Mixed organic", "Other"],
                key="waste_type_input"
            )
        
        waste_kg = waste_grams / 1000.0
        
        # Calculate income based on waste type and amount
        waste_type_multipliers = {
            "Vegetable peels": 1.2,
            "Onion peel": 1.0,
            "Egg shell": 0.8,
            "Fruit scraps": 1.3,
            "Mixed organic": 1.1,
            "Other": 1.0
        }
        multiplier = waste_type_multipliers.get(waste_type, 1.0)
        user_input_income = round(waste_kg * 12 * multiplier, 2)
        
        # Highlight relevant composting video
        st.markdown(f"**✨ Video for {waste_type.lower()}:** ")
        if "onion" in waste_type.lower():
            st.markdown("[How to compost kitchen waste](https://www.youtube.com/results?search_query=how+to+make+compost+from+kitchen+waste)")
        elif "egg" in waste_type.lower():
            st.markdown("[How to compost kitchen waste](https://www.youtube.com/results?search_query=how+to+make+compost+from+kitchen+waste)")
        elif "fruit" in waste_type.lower():
            st.markdown("[Easy organic compost](https://www.youtube.com/results?search_query=organic+compost+making+easy)")
        else:
            st.markdown("[Composting vegetable peels at home](https://www.youtube.com/results?search_query=vegetable+peel+composting+at+home)")
        
        # Show income calculation
        st.markdown(f"**Income from your waste:**")
        st.metric("Calculated compost income (₹)", f"{user_input_income:,.2f}", help=f"{waste_kg:.2f} kg × base rate × {waste_type} multiplier")
        
        # Update compost history and forecast with user input
        predicted_nursery, hist = compost_nursery_forecast()
        
        # Calculate new predicted nursery income including user input
        all_incomes = [h["income"] for h in hist] + [user_input_income]
        new_predicted_nursery = round(sum(all_incomes) / len(all_incomes), 0) if all_incomes else user_input_income
        
        st.markdown("---")
        st.markdown("#### Compost income history & forecast (this session)")
        if hist:
            days = [h["day"] for h in hist]
            vals = [h["income"] for h in hist]
            fig_c = px.bar(x=days, y=vals, labels={"x": "Day", "y": "₹ (illustrative)"}, title="Compost income history (demo)")
            fig_c.update_traces(marker_color="#0d9488")
            fig_c.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(color="#0f172a"))
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.caption("Run **Predict Demand** once to start your compost income history.")
        
        st.metric(
            "Predicted income from nearby nursery (₹)",
            f"{new_predicted_nursery:,.0f}",
            help="Updated based on your waste amount and type.",
        )

    st.markdown("---")
    with st.container():
        st.markdown("### Smart usage tips")
        st.image(IMG_SUSTAIN, use_container_width=True, caption="Small habits compound: label, date, and plan portions.")
        st.markdown(
            "- Match prep to expected demand  \n"
            "- Label and date refrigerated food  \n"
            "- Offer flexible portions where it helps"
        )


def page_donation():
    render_top_header()
    page_title_block("Donation", "Share surplus food before it spoils — here are sample partners and a route idea.")
    b = st.session_state.prediction_bundle

    if b and b.get("surplus"):
        st.error("**You may have more food than demand** — consider donating soon.")
    elif b:
        st.success("**Your last estimate looks balanced** — keep partners saved for busy days.")
    else:
        st.info("Run **Predict Demand** for surplus hints tailored to your estimate.")

    st.markdown("---")
    for n in DUMMY_NGOS:
        st.markdown(f"**{n['name']}** — ~{n['area']} away")

    st.markdown("---")
    try:
        st.plotly_chart(donation_route_figure(), use_container_width=True)
    except Exception:
        st.info("Route preview unavailable in this environment — imagine a path from your kitchen to each pin.")
    st.caption("Illustrative GPS-style path — not real navigation.")

    if st.button("Notify a nearby shelter", type="primary", use_container_width=True, key="don_cta"):
        st.balloons()
        st.success("Thanks — a full product would share your pickup window securely.")


def page_storage():
    render_top_header()
    page_title_block("Storage Assistant", "Enter what you’re storing — we’ll suggest temperature and humidity.")
    st.image(IMG_STORAGE, use_container_width=True, caption="Good storage protects every plate you serve.")

    item_name = st.text_input("Food or dish name", placeholder="e.g. paneer curry, basmati rice, chopped salad")

    if (item_name or "").strip():
        st.markdown("---")
        st.markdown("### Suggestions for your item")
        cat, temp, hum = storage_recommendation(item_name.strip())
        st.markdown(f"**{cat}**  \n**Best temperature:** {temp}  \n**Humidity:** {hum}")
        if any(k in item_name.lower() for k in ["hot", "curry", "cooked", "rice", "biryani"]):
            st.info("**Hot food:** cool within **two hours**, then refrigerate.")
        if any(k in item_name.lower() for k in ["grain", "rice", "dal", "pulse", "atta"]):
            st.info("**Dry grains:** airtight containers in a **cool, dry** cupboard.")
        if any(k in item_name.lower() for k in ["veg", "vegetable", "salad", "leaf"]):
            st.info("**Vegetables:** crisper drawer, avoid excess moisture on delicate greens.")
    else:
        st.caption("Type a food name above to see tailored storage advice.")


def page_feedback():
    render_top_header()
    page_title_block("Feedback", "Rate Wise Food AI and share ideas.")

    rating = st.select_slider(
        "Your rating",
        options=[1, 2, 3, 4, 5],
        value=5,
        format_func=lambda x: "★" * x,
    )
    st.markdown(f"**You selected:** {rating} out of 5")

    fb = st.text_area("Comments", placeholder="What should we improve?", height=140)
    if st.button("Submit feedback", type="primary"):
        st.session_state.feedback_log.append({"rating": rating, "text": fb or ""})
        n = len(st.session_state.feedback_log)
        avg = sum(x["rating"] for x in st.session_state.feedback_log) / n
        st.session_state.demo_avg_rating = round(0.7 * st.session_state.demo_avg_rating + 0.3 * avg, 2)
        st.success("Thank you — we’ve received your feedback.")
    st.markdown("---")
    st.metric("Average satisfaction (demo)", f"{st.session_state.demo_avg_rating} / 5")


def main_app():
    if st.session_state.get("nav_radio") not in NAV_PAGES:
        st.session_state.nav_radio = NAV_PAGES[0]
    model = load_model()
    page = sidebar_nav()

    routes = {
        NAV_PAGES[0]: page_home,
        NAV_PAGES[1]: lambda: page_predict(model),
        NAV_PAGES[2]: page_spoilage,
        NAV_PAGES[3]: page_expiry,
        NAV_PAGES[4]: page_smart_inventory,
        NAV_PAGES[5]: page_analytics,
        NAV_PAGES[6]: page_sustainability,
        NAV_PAGES[7]: page_donation,
        NAV_PAGES[8]: page_storage,
        NAV_PAGES[9]: page_feedback,
    }
    routes[page]()

    st.markdown("---")
    st.caption("**Wise Food AI** · Predict • Prepare • Prevent Waste")


init_session()

if not st.session_state.logged_in:
    auth_screen()
else:
    main_app()
