"""
app/app.py  —  F1 Pit Strategy AI Dashboard (Streamlit)
Run with:  streamlit run app/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# ── Resolve imports ────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from simulator import CIRCUITS, simulate_strategy, compare_strategies, HISTORICAL_METRICS_CACHE
from predict   import predict_pit, recommend_strategy, load_model
import live_data

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Strategy AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark F1 theme ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=Titillium+Web:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Titillium Web', sans-serif; }
  h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif !important; }

  /* Main Background */
  [data-testid="stAppViewContainer"] { 
    background-color: #0b0b0e !important;
    background-image: 
      radial-gradient(circle at 15% 50%, rgba(225, 6, 0, 0.12), transparent 30%),
      radial-gradient(circle at 85% 20%, rgba(0, 210, 255, 0.12), transparent 30%),
      radial-gradient(circle at 50% 90%, rgba(255, 65, 54, 0.08), transparent 40%),
      linear-gradient(135deg, rgba(15,15,20,1) 0%, rgba(10,10,12,1) 100%) !important;
    background-attachment: fixed !important;
  }
  
  [data-testid="stHeader"] {
    background: transparent !important;
  }

  [data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -20%; left: -10%; width: 50%; height: 50%;
    background: radial-gradient(circle, rgba(0, 210, 255, 0.15) 0%, transparent 70%);
    pointer-events: none; z-index: -1;
  }
  
  /* Staggered Animations */
  @keyframes slideInUp {
    0% { transform: translateY(30px) scale(0.98); opacity: 0; }
    100% { transform: translateY(0) scale(1); opacity: 1; }
  }

  [data-testid="stVerticalBlock"] > div > div {
    animation: slideInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) backwards;
  }
  [data-testid="stVerticalBlock"] > div:nth-child(1) > div { animation-delay: 0.05s; }
  [data-testid="stVerticalBlock"] > div:nth-child(2) > div { animation-delay: 0.15s; }
  [data-testid="stVerticalBlock"] > div:nth-child(3) > div { animation-delay: 0.25s; }
  [data-testid="stVerticalBlock"] > div:nth-child(4) > div { animation-delay: 0.35s; }
  [data-testid="stVerticalBlock"] > div:nth-child(5) > div { animation-delay: 0.45s; }
  [data-testid="stVerticalBlock"] > div:nth-child(6) > div { animation-delay: 0.55s; }

  /* Glass classes */
  .glass-panel, [data-testid="stVerticalBlock"] > div > div {
    background: rgba(25, 25, 35, 0.2) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
  }

  /* F1 Lights */
  .f1-light-box { display:flex; gap:16px; margin-top:16px; padding:10px 20px; background:#111; border-radius:12px; border:2px solid #222; width:fit-content; box-shadow:inset 0 4px 10px rgba(0,0,0,0.8); }
  .f1-light { width:24px; height:24px; border-radius:50%; background:#2a0000; border:2px solid #000; }
  .light-1 { animation: l1 4s infinite; }
  .light-2 { animation: l2 4s infinite; }
  .light-3 { animation: l3 4s infinite; }
  .light-4 { animation: l4 4s infinite; }
  .light-5 { animation: l5 4s infinite; }
  
  @keyframes l1 { 20%, 50% { background: #ff1a1a; box-shadow: 0 0 20px #ff0000; } 0%, 19.9%, 50.1%, 100% { background: #2a0000; box-shadow: none; } }
  @keyframes l2 { 26%, 50% { background: #ff1a1a; box-shadow: 0 0 20px #ff0000; } 0%, 25.9%, 50.1%, 100% { background: #2a0000; box-shadow: none; } }
  @keyframes l3 { 32%, 50% { background: #ff1a1a; box-shadow: 0 0 20px #ff0000; } 0%, 31.9%, 50.1%, 100% { background: #2a0000; box-shadow: none; } }
  @keyframes l4 { 38%, 50% { background: #ff1a1a; box-shadow: 0 0 20px #ff0000; } 0%, 37.9%, 50.1%, 100% { background: #2a0000; box-shadow: none; } }
  @keyframes l5 { 44%, 50% { background: #ff1a1a; box-shadow: 0 0 20px #ff0000; } 0%, 43.9%, 50.1%, 100% { background: #2a0000; box-shadow: none; } }


  /* Sidebar */
  section[data-testid="stSidebar"] { 
    background: rgba(15, 15, 20, 0.5) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
  }
  section[data-testid="stSidebar"] * { color:#e0e0e0 !important; }

  /* Metrics */
  div[data-testid="metric-container"] { 
    background: rgba(30, 30, 45, 0.35); 
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15); 
    border-radius: 12px; 
    padding: 16px; 
    box-shadow: inset 0 0 10px rgba(255,255,255,0.02);
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap:8px; background:transparent; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:4px; }
  .stTabs [data-baseweb="tab"] { 
    background: rgba(25, 25, 35, 0.4); 
    backdrop-filter: blur(8px);
    border-radius: 12px 12px 0 0; 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    border-bottom: none;
    color: #999; 
    padding: 10px 24px; 
    font-weight: 600; 
    transition: all 0.3s ease;
  }
  .stTabs [aria-selected="true"] { 
    background: rgba(255, 255, 255, 0.1) !important; 
    color: #fff !important; 
    border-color: rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 -4px 12px rgba(255,255,255,0.05);
  }

  /* Buttons */
  .stButton > button { 
    background: linear-gradient(135deg, rgba(225,6,0,0.8), rgba(255,65,54,0.6)); 
    backdrop-filter: blur(4px);
    color: white; 
    border: 1px solid rgba(255,255,255,0.2); 
    border-radius: 12px; 
    padding: 10px 28px; 
    font-weight: 700; 
    font-size: 15px; 
    letter-spacing: 0.5px; 
    transition: all 0.3s ease; 
    width: 100%; 
  }
  .stButton > button:hover { 
    background: linear-gradient(135deg, rgba(255,26,24,0.9), rgba(225,6,0,0.9)); 
    transform: translateY(-2px); 
    box-shadow: 0 6px 24px rgba(225,6,0,0.5); 
    border-color: rgba(255,255,255,0.4);
  }

  /* Inputs */
  .stSlider > div > div { accent-color: #00d2ff; }
  .stSelectbox div[data-baseweb="select"] > div { 
    background: rgba(30, 30, 45, 0.3); 
    backdrop-filter: blur(8px);
    border-color: rgba(255, 255, 255, 0.15); 
    border-radius: 8px;
  }
  
  /* Decision banners */
  .pit-now { 
    background: linear-gradient(135deg, rgba(225,6,0,0.6), rgba(255,65,54,0.4)); 
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,100,100,0.3);
    border-radius: 16px; padding: 28px 32px; text-align: center; 
    animation: pulse 2s ease-in-out infinite; 
    box-shadow: 0 8px 32px rgba(225,6,0,0.2);
  }
  .stay-out { 
    background: linear-gradient(135deg, rgba(26,122,46,0.6), rgba(34,168,60,0.4)); 
    backdrop-filter: blur(12px);
    border: 1px solid rgba(100,255,100,0.3);
    border-radius: 16px; padding: 28px 32px; text-align: center; 
    box-shadow: 0 8px 32px rgba(26,122,46,0.2);
  }
  @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(225,6,0,0.4)} 50%{box-shadow:0 0 0 16px rgba(225,6,0,0)} }

  /* Strategy cards */
  .strat-card { 
    background: rgba(30, 30, 40, 0.35); 
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1); 
    border-radius: 16px; 
    padding: 18px 20px; 
    margin-bottom: 10px; 
    transition: all 0.3s; 
  }
  .strat-card:hover { 
    border-color: rgba(0, 210, 255, 0.5); 
    box-shadow: 0 8px 24px rgba(0, 210, 255, 0.15);
    transform: translateY(-2px);
  }
  .rank-badge { 
    display: inline-block; 
    background: linear-gradient(135deg, #00d2ff, #3a7bd5); 
    color: white; 
    font-weight: 900; 
    font-size: 0.85rem; 
    padding: 4px 12px; 
    border-radius: 20px; 
    margin-right: 10px; 
    box-shadow: 0 4px 10px rgba(0,210,255,0.3);
  }

  h1 { color: #fff !important; font-weight: 900 !important; }
  h2, h3 { color: #f0f0f0 !important; font-weight: 700 !important; }
  .stMarkdown p { color: #b0b0c0; }
  hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:24px; margin-bottom:24px; padding:32px; background:rgba(20,20,30,0.4); border:1px solid rgba(255,255,255,0.05); border-radius:24px; backdrop-filter:blur(16px); box-shadow:0 12px 40px rgba(0,0,0,0.5);">
  <div style="background:linear-gradient(135deg, #e10600, #ff4136); border-radius:20px; padding:20px 28px; font-size:3rem; box-shadow:0 8px 24px rgba(225,6,0,0.4); transform:rotate(-5deg);">🏎️</div>
  <div>
    <h1 style="margin:0; font-size:3.6rem; text-transform:uppercase; font-style:italic; letter-spacing:2px; font-weight:900; background:linear-gradient(135deg, #ffffff, #aaaaaa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">F1 Pit Strategy AI</h1>
    <div class="f1-light-box">
      <div class="f1-light light-1"></div>
      <div class="f1-light light-2"></div>
      <div class="f1-light light-3"></div>
      <div class="f1-light light-4"></div>
      <div class="f1-light light-5"></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None

model_bundle = get_model()
model_ok     = model_bundle is not None

if not model_ok:
    st.error("⚠️ No trained model found. Run `python src/data_loader.py` then `python src/model.py` first.")
    st.stop()

# ── Rain effect ────────────────────────────────────────────────────────────────
def render_rain():
    drops_html = "".join([
        f'<div class="drop" style="left:{np.random.randint(0,100)}%;'
        f'animation-delay:{np.random.random()*0.5}s;'
        f'animation-duration:{0.4+np.random.random()*0.3}s;'
        f'opacity:{0.3+np.random.random()*0.5}"></div>'
        for _ in range(70)
    ])
    st.markdown(f'<div class="rain-container">{drops_html}</div>', unsafe_allow_html=True)

# ── Dropdowns ──────────────────────────────────────────────────────────────────
available_teams   = sorted(list(HISTORICAL_METRICS_CACHE.get("team_pace_offsets", {}).keys())) or ["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Aston Martin"]
available_drivers = sorted(list(HISTORICAL_METRICS_CACHE.get("driver_tyre_factors", {}).keys())) or ["VER", "HAM", "LEC", "NOR", "ALO"]

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🛞  Live Pit Decision",
    "🗺️  Strategy Recommender",
    "📈  Lap Simulator",
    "🏆  Live Scoreboard",
    "🏅  Previous Champions",
    "📍  Live Track Map",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PIT DECISION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Current Race Conditions")
    st.markdown("Enter the live race state and get an instant pit recommendation from the AI.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Driver & Circuit**")
        team_t1      = st.selectbox("Constructors", available_teams, key="team_1")
        driver_t1    = st.selectbox("Driver", available_drivers, key="drv_1")
        circuit_t1   = st.selectbox("Circuit", list(CIRCUITS.keys()), key="c_t1")
        st.markdown("---")
        st.markdown("**Race Position**")
        lap_number     = st.slider("Current Lap", 1, 80, 30)
        laps_remaining = st.slider("Laps Remaining", 0, 80, 25)
        position       = st.slider("Current Position", 1, 20, 8)
        is_sc          = st.toggle("Safety Car / VSC Active", value=False)
        st.markdown("---")
        pit_threshold = st.slider(
            "Pit Sensitivity (threshold)", 0.20, 0.50, 0.35, 0.01,
            help="Lower = more aggressive pitting. Default 0.35 corrects for class imbalance in training data."
        )

    with col_r:
        st.markdown("**Tyre Status**")
        compound       = st.selectbox("Compound on Car", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
        laps_since_pit = st.slider("Laps on Current Tyre", 1, 55, 14)
        stint_number   = st.slider("Stint Number", 1, 4, 1)
        lap_time_sec   = st.number_input("Latest Lap Time (s)", 60.0, 130.0, 92.5, 0.1)
        lap_time_delta = st.number_input("Lap Time Delta vs Rolling Avg (s)", -5.0, 10.0, 0.8, 0.1,
                                          help="Positive = getting slower (degradation)")

        st.markdown("**Weather**")
        wc1, wc2 = st.columns(2)
        with wc1:
            air_temp   = st.number_input("Air Temp (°C)",   -5.0, 50.0, 26.0, 0.5)
            track_temp = st.number_input("Track Temp (°C)", 10.0, 70.0, 38.0, 0.5)
            rainfall   = st.number_input("Rainfall (0-1)",   0.0,  1.0,  0.0, 0.1, key="rain_1")
        if rainfall > 0:
            render_rain()
        with wc2:
            humidity   = st.number_input("Humidity (%)",    0.0, 100.0, 55.0, 1.0)
            wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 30.0,  5.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍  Get Pit Recommendation", key="btn_t1")

    if run_btn:
        circuit_list  = list(CIRCUITS.keys())
        track_encoded = circuit_list.index(circuit_t1) if circuit_t1 in circuit_list else 0

        result = predict_pit(
            lap_number=lap_number, laps_since_pit=float(laps_since_pit),
            compound=compound, lap_time_seconds=lap_time_sec,
            lap_time_delta=lap_time_delta, stint_number=float(stint_number),
            laps_remaining=float(laps_remaining), is_safety_car=int(is_sc),
            position=float(position), air_temp=air_temp, track_temp=track_temp,
            rainfall=rainfall, humidity=humidity, wind_speed=wind_speed,
            track_encoded=track_encoded, team=team_t1, driver=driver_t1,
            pit_threshold=pit_threshold,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        decision = result["decision"]
        conf     = result["confidence"]
        pit_p    = result["pit_probability"]
        stay_p   = result["stay_probability"]

        if decision == "PIT NOW":
            st.markdown(f"""
            <div class="pit-now">
              <p class="decision-label">🔴 PIT NOW</p>
              <p class="decision-sub">Confidence: {conf:.1%} &nbsp;|&nbsp; Pit probability: {pit_p:.1%}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stay-out">
              <p class="decision-label">🟢 STAY OUT</p>
              <p class="decision-sub">Confidence: {conf:.1%} &nbsp;|&nbsp; Stay probability: {stay_p:.1%}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pit Probability",  f"{pit_p:.1%}")
        m2.metric("Stay Probability", f"{stay_p:.1%}")
        m3.metric("Confidence",       f"{conf:.1%}")
        m4.metric("Tyre Age",         f"{laps_since_pit} laps")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pit_p * 100,
            title={"text": "Pit Probability (%)", "font": {"color": "#f0f0f0", "size": 14}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#888"},
                "bar":     {"color": "#e10600"},
                "bgcolor": "#18181f",
                "steps": [
                    {"range": [0,  40], "color": "#1a7a2e"},
                    {"range": [40, 65], "color": "#8a6500"},
                    {"range": [65, 100],"color": "#7a1010"},
                ],
                "threshold": {"line": {"color": "#e10600", "width": 4}, "value": 50},
            },
            number={"suffix": "%", "font": {"color": "#f0f0f0"}},
        ))
        fig.update_layout(paper_bgcolor="#0d0d0f", plot_bgcolor="#0d0d0f",
                          height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STRATEGY RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Race Strategy Recommender")
    st.markdown("Select a circuit and weather conditions to get the AI-ranked optimal strategies.")

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns([2, 2, 1.5, 1.5, 1, 1])
    with sc1:  team_t2      = st.selectbox("Constructor", available_teams, key="team_2")
    with sc2:  driver_t2    = st.selectbox("Driver", available_drivers, key="drv_2")
    with sc3:  circuit_t2   = st.selectbox("Circuit", list(CIRCUITS.keys()), key="c_t2",
                                           index=list(CIRCUITS.keys()).index("Bahrain Grand Prix"))
    with sc4:  start_cpd_t2 = st.selectbox("Start Tyre", ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"], key="cpd_2")
    with sc5:  trk_temp_t2  = st.number_input("Track Temp", 10.0, 70.0, 38.0, 1.0, key="tt2")
    with sc6:  rain_t2      = st.number_input("Rain", 0.0, 1.0, 0.0, 0.1, key="rain_2")

    top_n     = st.selectbox("Show Top N Strategies", [3, 5, 7], key="topn")
    if rain_t2 > 0: render_rain()

    strat_btn = st.button("🗺️  Generate Strategies", key="btn_t2")

    if strat_btn:
        weather_t2 = {"track_temp": trk_temp_t2, "rainfall": rain_t2}
        with st.spinner("Simulating strategies…"):
            ranked = recommend_strategy(circuit_t2, weather=weather_t2, top_n=top_n,
                                        team=team_t2, driver=driver_t2,
                                        starting_compound=start_cpd_t2)

        if not ranked:
            st.error("No strategies returned. Try different circuit or weather settings.")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            best_time = ranked[0]["total_time"]
            for r in ranked:
                delta     = r["total_time"] - best_time
                delta_str = f"+{delta:.1f}s" if delta > 0 else "Fastest ⚡"
                pit_desc  = ", ".join([f"Lap {p['pit_lap']} → {p['compound'].capitalize()}"
                                        for p in r["strategy"]]) if r["strategy"] else "No pit stop"
                st.markdown(f"""
                <div class="strat-card">
                  <span class="rank-badge">#{r['rank']}</span>
                  <strong style="color:#f0f0f0; font-size:1.05rem;">{r['label']}</strong>
                  <span style="color:#888; font-size:0.9rem; margin-left:10px;">{delta_str}</span>
                  <br>
                  <span style="color:#aaa; font-size:0.88rem; margin-top:6px; display:block;">{pit_desc}</span>
                  <span style="color:#e10600; font-weight:700; font-size:1rem; float:right; margin-top:-24px;">{r['total_time']:,.1f}s</span>
                </div>
                """, unsafe_allow_html=True)

            labels     = [f"#{r['rank']} {r['label']}" for r in ranked]
            times      = [r["total_time"] for r in ranked]
            colors_bar = ["#e10600" if i == 0 else "#3a3a4a" for i in range(len(ranked))]
            fig2 = go.Figure(go.Bar(x=times, y=labels, orientation="h",
                                    marker_color=colors_bar,
                                    text=[f"{t:,.1f}s" for t in times],
                                    textposition="outside",
                                    textfont={"color": "#d0d0d8"}))
            fig2.update_layout(paper_bgcolor="#0d0d0f", plot_bgcolor="#18181f",
                               xaxis=dict(title="Total Race Time (s)", color="#888", gridcolor="#2a2a35"),
                               yaxis=dict(color="#d0d0d8", autorange="reversed"),
                               height=300 + len(ranked)*28, margin=dict(t=20,b=20,l=120,r=100),
                               font={"color": "#d0d0d8"})
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LAP SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Lap-by-Lap Tyre Degradation Simulator")
    st.markdown("Compare how different strategies play out lap by lap across the race distance.")

    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns([2, 2, 1.5, 1.5, 1, 1])
    with lc1:  team_t3      = st.selectbox("Constructor", available_teams, key="team_3")
    with lc2:  driver_t3    = st.selectbox("Driver", available_drivers, key="drv_3")
    with lc3:  circuit_t3   = st.selectbox("Circuit", list(CIRCUITS.keys()), key="c_t3",
                                           index=list(CIRCUITS.keys()).index("British Grand Prix"))
    with lc4:  start_cpd_t3 = st.selectbox("Start Tyre", ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"], key="cpd_3")
    with lc5:  trk_temp_t3  = st.number_input("Track Temp", 10.0, 70.0, 30.0, 1.0, key="tt3")
    with lc6:  rain_t3      = st.number_input("Rain", 0.0, 1.0, 0.0, 0.1, key="rain_3")

    if rain_t3 > 0: render_rain()
    sim_btn = st.button("📈  Run Simulation", key="btn_t3")

    if sim_btn:
        weather_t3 = {"track_temp": trk_temp_t3, "rainfall": rain_t3}
        n_laps     = CIRCUITS[circuit_t3]["laps"]

        # Compute pit laps safely (min lap 2, max n_laps-3)
        def _pit(frac: float) -> int:
            return max(2, min(n_laps - 3, round(n_laps * frac)))

        strategies_t3 = [
            {"label": "No-stop",        "strategy": []},
            {"label": "1-stop (early)",  "strategy": [{"pit_lap": _pit(0.35), "compound": "HARD"}]},
            {"label": "1-stop (late)",   "strategy": [{"pit_lap": _pit(0.50), "compound": "HARD"}]},
            {"label": "2-stop",          "strategy": [
                {"pit_lap": _pit(0.28), "compound": "MEDIUM"},
                {"pit_lap": _pit(0.58), "compound": "HARD"},
            ]},
        ]
        COLORS = ["#e10600", "#00d2ff", "#f5a623", "#7ed321"]
        fig3 = go.Figure()

        for idx, s in enumerate(strategies_t3):
            result_s = simulate_strategy(circuit_t3, s["strategy"], weather_t3,
                                          team=team_t3, driver=driver_t3,
                                          starting_compound=start_cpd_t3)
            laps_x  = [lr.lap  for lr in result_s.lap_records]
            times_y = [lr.time for lr in result_s.lap_records]
            fig3.add_trace(go.Scatter(x=laps_x, y=times_y, mode="lines", name=s["label"],
                                      line=dict(color=COLORS[idx], width=2.5),
                                      hovertemplate=f"Lap %{{x}}<br>Lap time: %{{y:.2f}}s<extra>{s['label']}</extra>"))
            for pit_lap in result_s.pit_laps:
                if 1 <= pit_lap - 1 < len(result_s.lap_records):
                    fig3.add_vline(x=pit_lap, line_dash="dot",
                                   line_color=COLORS[idx], opacity=0.4,
                                   annotation_text=f"Pit ({s['label'].split()[0]})",
                                   annotation_font_color=COLORS[idx],
                                   annotation_font_size=10)

        fig3.update_layout(paper_bgcolor="#0d0d0f", plot_bgcolor="#18181f",
                           xaxis=dict(title="Lap Number", color="#888", gridcolor="#2a2a35"),
                           yaxis=dict(title="Lap Time (seconds)", color="#888", gridcolor="#2a2a35"),
                           legend=dict(bgcolor="#18181f", bordercolor="#2a2a35", font={"color":"#d0d0d8"}),
                           hovermode="x unified", height=480,
                           margin=dict(t=20,b=40,l=60,r=20), font={"color":"#d0d0d8"})
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Strategy Comparison Table**")
        rows = []
        for s in strategies_t3:
            r = simulate_strategy(circuit_t3, s["strategy"], weather_t3,
                                   team=team_t3, driver=driver_t3,
                                   starting_compound=start_cpd_t3)
            rows.append({"Strategy": s["label"],
                          "Pit Laps": ", ".join(map(str, r.pit_laps)) or "—",
                          "Total Time": f"{r.total_time:,.1f}s"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE SCOREBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("F1 Live Scoreboard & Championship Standings")
    st.markdown("Live data from the [Jolpica (Ergast) API](https://api.jolpi.ca) and [OpenF1](https://openf1.org).")

    # Refresh control
    sb_cols = st.columns([5, 1])
    with sb_cols[1]:
        refresh = st.button("🔄  Refresh", key="btn_refresh")

    if "scoreboard_data" not in st.session_state or refresh:
        with st.spinner("Fetching live F1 data…"):
            def _safe_standings(fn):
                """Call fn(); handle both (list, str) and bare list returns."""
                result = fn()
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                if isinstance(result, list):
                    return result, "current"
                return [], "unknown"

            drv_rows, drv_season = _safe_standings(live_data.get_driver_standings)
            con_rows, con_season = _safe_standings(live_data.get_constructor_standings)
            st.session_state["scoreboard_data"] = {
                "driver_standings":             drv_rows,
                "driver_standings_season":      drv_season,
                "constructor_standings":        con_rows,
                "constructor_standings_season": con_season,
                "last_race":    live_data.get_last_race_results(),
                "schedule":     live_data.get_season_schedule(),
                "live_session": live_data.get_live_session(),
            }

    sb = st.session_state["scoreboard_data"]

    # ── Live session banner ────────────────────────────────────────────────────
    live_sess = sb.get("live_session")
    if live_sess:
        sess_key  = live_sess["session_key"]
        live_pos  = live_data.get_live_positions(sess_key)
        live_drvs = live_data.get_live_drivers(sess_key)

        if live_pos:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a0000,#3a0000); border:1px solid #e10600;
                        border-radius:12px; padding:16px 20px; margin-bottom:16px;">
              <p style="margin:0; font-size:0.75rem; text-transform:uppercase; letter-spacing:2px; color:#e10600; font-weight:700;">🔴 LIVE NOW</p>
              <p style="margin:4px 0 0; font-size:1.3rem; font-weight:900; color:#fff;">
                {live_sess['meeting_name']} &nbsp;·&nbsp; {live_sess['circuit']}, {live_sess['country']}
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🏁 Live Race Order")
            live_rows = []
            for entry in live_pos:
                drv_num  = entry.get("driver_number")
                drv_info = live_drvs.get(drv_num, {})
                live_rows.append({
                    "Pos":    entry.get("position", "—"),
                    "#":      drv_num,
                    "Code":   drv_info.get("code", str(drv_num)),
                    "Driver": drv_info.get("full_name", "—"),
                    "Team":   drv_info.get("team", "—"),
                })
            if live_rows:
                st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True,
                             column_config={"Pos": st.column_config.NumberColumn("Pos", format="%d", width="small")})
            st.caption("Live positions update on each Refresh.")
            st.markdown("---")

    # ── Championship standings ─────────────────────────────────────────────────
    schedule  = sb.get("schedule", [])
    next_race = next((r for r in schedule if r["status"] == "next"), None)
    if next_race:
        st.caption(f"Next race: **{next_race['race_name']}** — {next_race['date']}")

    drv_std  = sb.get("driver_standings", [])
    drv_szn   = sb.get("driver_standings_season", "current")
    con_std  = sb.get("constructor_standings", [])
    con_szn   = sb.get("constructor_standings_season", "current")

    tab_drv, tab_con = st.tabs(["\U0001f9d1\u200d\U0001f3ce\ufe0f  Drivers Championship", "\U0001f3ed  Constructors Championship"])

    with tab_drv:
        if not drv_std:
            st.warning("Could not fetch driver standings — check your internet connection.")
        else:
            if "final" in drv_szn:
                st.info(f"\U0001f4cc Showing **{drv_szn}** standings — 2026 season kicks off March 8.")
            # Podium cards
            podium       = drv_std[:3]
            medals       = ["🥇", "🥈", "🥉"]
            medal_bg     = ["#3d2e00", "#1a1a1a", "#2a1000"]
            medal_border = ["#f5c518", "#aaa", "#c86a2a"]
            pod_cols = st.columns(3)
            for i, (col, drv) in enumerate(zip(pod_cols, podium)):
                with col:
                    st.markdown(f"""
                    <div style="background:{medal_bg[i]}; border:1px solid {medal_border[i]};
                                border-radius:12px; padding:16px; text-align:center; margin-bottom:12px;">
                      <p style="font-size:1.8rem; margin:0;">{medals[i]}</p>
                      <p style="font-size:1.3rem; font-weight:900; color:#fff; margin:6px 0 2px;">{drv['driver_code']}</p>
                      <p style="font-size:0.8rem; color:#aaa; margin:0;">{drv['driver_name']}</p>
                      <p style="font-size:0.82rem; color:#888; margin:3px 0 0;">{drv['team']}</p>
                      <p style="font-size:1.5rem; font-weight:900; color:{medal_border[i]}; margin:8px 0 0;">{drv['points']:.0f} pts</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Bar chart
            df_drv = pd.DataFrame(drv_std)
            df_drv.columns = ["Pos", "Code", "Driver", "Nationality", "Team", "Points", "Wins"]
            fig_pts = go.Figure(go.Bar(
                x=df_drv["Points"], y=df_drv["Code"], orientation="h",
                marker_color=["#f5c518" if i==0 else ("#aaa" if i==1 else ("#c86a2a" if i==2 else "#e10600"))
                               for i in range(len(df_drv))],
                text=df_drv["Points"].apply(lambda p: f"{p:.0f}"),
                textposition="outside", textfont={"color":"#d0d0d8","size":11},
                hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
            ))
            fig_pts.update_layout(
                paper_bgcolor="#0d0d0f", plot_bgcolor="#18181f",
                xaxis=dict(title="Championship Points", color="#888", gridcolor="#2a2a35"),
                yaxis=dict(color="#d0d0d8", autorange="reversed", tickfont={"size":10}),
                height=max(300, len(df_drv)*26+60),
                margin=dict(t=10,b=30,l=50,r=70), font={"color":"#d0d0d8"},
            )
            st.plotly_chart(fig_pts, use_container_width=True)

            st.dataframe(df_drv, use_container_width=True, hide_index=True,
                         column_config={
                             "Pos":    st.column_config.NumberColumn("#",      format="%d",   width="small"),
                             "Points": st.column_config.NumberColumn("Points", format="%.0f", width="small"),
                             "Wins":   st.column_config.NumberColumn("Wins",   format="%d",   width="small"),
                         })

    with tab_con:
        if not con_std:
            st.warning("Could not fetch constructor standings — check your internet connection.")
        else:
            if "final" in con_szn:
                st.info(f"\U0001f4cc Showing **{con_szn}** standings.")
            df_con = pd.DataFrame(con_std)
            df_con.columns = ["Pos", "Team", "Nationality", "Points", "Wins"]
            fig_con = go.Figure(go.Bar(
                x=df_con["Points"], y=df_con["Team"], orientation="h",
                marker_color=["#f5c518" if i==0 else "#e10600" for i in range(len(df_con))],
                text=df_con["Points"].apply(lambda p: f"{p:.0f}"),
                textposition="outside", textfont={"color":"#d0d0d8"},
                hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
            ))
            fig_con.update_layout(
                paper_bgcolor="#0d0d0f", plot_bgcolor="#18181f",
                xaxis=dict(title="Points", color="#888", gridcolor="#2a2a35"),
                yaxis=dict(color="#d0d0d8", autorange="reversed"),
                height=max(200, len(df_con)*36+50),
                margin=dict(t=10,b=30,l=160,r=70), font={"color":"#d0d0d8"},
            )
            st.plotly_chart(fig_con, use_container_width=True)
            st.dataframe(df_con, use_container_width=True, hide_index=True,
                         column_config={
                             "Pos":    st.column_config.NumberColumn("#",      format="%d",   width="small"),
                             "Points": st.column_config.NumberColumn("Points", format="%.0f", width="small"),
                             "Wins":   st.column_config.NumberColumn("Wins",   format="%d",   width="small"),
                         })

    # ── Last Race Results ──────────────────────────────────────────────────────
    st.markdown("---")
    last = sb.get("last_race", {})
    if last:
        st.markdown(f"### 🏁 Last Race: {last.get('race_name', '—')}")
        st.caption(f"{last.get('circuit','—')} &nbsp;·&nbsp; {last.get('date','—')}")

        results = last.get("results", [])
        if results:
            top3      = [r for r in results if r["pos"] <= 3]
            top3_cols = st.columns(len(top3))
            for col, r in zip(top3_cols, top3):
                medal = ["🥇","🥈","🥉"][r["pos"]-1]
                with col:
                    st.markdown(f"""
                    <div style="background:#18181f; border:1px solid #2a2a35; border-radius:12px; padding:14px; text-align:center; margin-bottom:12px;">
                      <p style="font-size:1.6rem; margin:0;">{medal}</p>
                      <p style="font-size:1.2rem; font-weight:900; color:#fff; margin:4px 0 2px;">{r['driver_code']}</p>
                      <p style="font-size:0.78rem; color:#aaa; margin:0;">{r['driver_name']}</p>
                      <p style="font-size:0.78rem; color:#888; margin:2px 0 0;">{r['team']}</p>
                      <p style="font-size:1rem; font-weight:700; color:#e10600; margin:8px 0 0;">+{r['points']:.0f} pts</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            df_res = pd.DataFrame([{
                "Pos":         r["pos"],
                "Code":        r["driver_code"],
                "Driver":      r["driver_name"],
                "Team":        r["team"],
                "Grid":        r["grid"],
                "Laps":        r["laps"],
                "Status":      r["status"],
                "Points":      r["points"],
                "Fastest Lap": r["fastest_lap"] + (" 🟣" if r["fl_rank"]==1 else ""),
            } for r in results])
            st.dataframe(df_res, use_container_width=True, hide_index=True,
                         column_config={
                             "Pos":    st.column_config.NumberColumn("#",    format="%d",   width="small"),
                             "Grid":   st.column_config.NumberColumn("Grid", format="%d",   width="small"),
                             "Laps":   st.column_config.NumberColumn("Laps", format="%d",   width="small"),
                             "Points": st.column_config.NumberColumn("Pts",  format="%.0f", width="small"),
                         })
    else:
        st.info("Last race results not available — try refreshing.")


    # ── Season Calendar ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗓️ Season Calendar")
    if schedule:

        STATUS_ICON   = {"done": "✅", "next": "⏭️", "upcoming": "🔜"}
        STATUS_COLOR  = {"done": "#2a2a35", "next": "#3a1500", "upcoming": "#18181f"}
        STATUS_BORDER = {"done": "#3a3a4a", "next": "#e10600",  "upcoming": "#2a2a35"}
        for i in range(0, len(schedule), 4):
            chunk    = schedule[i:i+4]
            cal_cols = st.columns(len(chunk))
            for col, race in zip(cal_cols, chunk):
                icon = STATUS_ICON.get(race["status"], "")
                bg   = STATUS_COLOR.get(race["status"], "#18181f")
                brd  = STATUS_BORDER.get(race["status"], "#2a2a35")
                with col:
                    st.markdown(f"""
                    <div style="background:{bg}; border:1px solid {brd}; border-radius:10px;
                                padding:12px; margin-bottom:8px; text-align:center;">
                      <p style="margin:0; font-size:0.68rem; color:#888;">Rd {race['round']}</p>
                      <p style="margin:2px 0; font-size:0.85rem; font-weight:700; color:#f0f0f0;">{icon} {race['country']}</p>
                      <p style="margin:0; font-size:0.68rem; color:#888;">{race['date']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Season calendar not available — try refreshing.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREVIOUS CHAMPIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("F1 Previous Champions")
    st.markdown("Select a season to view the final World Drivers' and Constructors' Championship standings.")

    CHAMP_YEARS = list(range(2025, 1999, -1))  # 2025 → 2000

    champ_year = st.selectbox(
        "Season",
        options=CHAMP_YEARS,
        format_func=lambda y: f"🏁 {y} Season",
        index=0,
        key="champ_year",
    )

    load_champ_btn = st.button("📊  Load Standings", key="btn_champ")

    champ_cache_key = f"champ_{champ_year}"
    if load_champ_btn or champ_cache_key not in st.session_state:
        with st.spinner(f"Fetching {champ_year} championship standings…"):
            drv_data, _ = live_data.get_driver_standings(str(champ_year))
            con_data, _ = live_data.get_constructor_standings(str(champ_year))
            st.session_state[champ_cache_key] = {
                "drivers":      drv_data,
                "constructors": con_data,
            }

    champ_sb = st.session_state.get(champ_cache_key, {})
    drv_champ = champ_sb.get("drivers", [])
    con_champ = champ_sb.get("constructors", [])

    if drv_champ or con_champ:
        # ── Champion banner ──────────────────────────────────────────────────
        if drv_champ:
            wdc = drv_champ[0]
            wcc = con_champ[0] if con_champ else {}
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#2a1f00,#4a3600); border:2px solid #f5c518;
                        border-radius:16px; padding:20px 28px; margin:16px 0; display:flex;
                        justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
              <div>
                <p style="margin:0; font-size:0.75rem; text-transform:uppercase; letter-spacing:2px;
                          color:#f5c518; font-weight:700;">🏆 {champ_year} World Champion</p>
                <p style="margin:6px 0 2px; font-size:2rem; font-weight:900; color:#fff;">
                  {wdc.get('driver_name','—')}
                </p>
                <p style="margin:0; font-size:0.9rem; color:#aaa;">{wdc.get('team','—')}</p>
              </div>
              <div style="text-align:right;">
                <p style="margin:0; font-size:2.8rem; font-weight:900; color:#f5c518;">
                  {wdc.get('points',0):.0f} <span style="font-size:1rem; color:#aaa;">pts</span>
                </p>
                <p style="margin:4px 0 0; font-size:0.85rem; color:#aaa;">{wdc.get('wins',0)} wins</p>
              </div>
              {f'<div><p style="margin:0; font-size:0.75rem; color:#f5c518; letter-spacing:2px; text-transform:uppercase; font-weight:700;">🏭 WCC</p><p style="margin:4px 0 2px; font-size:1.2rem; font-weight:900; color:#fff;">{wcc.get("team","—")}</p><p style="font-size:0.85rem; color:#aaa; margin:0;">{wcc.get("points",0):.0f} pts</p></div>' if wcc else ''}
            </div>
            """, unsafe_allow_html=True)

        # ── Sub-tabs: Drivers | Constructors ─────────────────────────────────
        ct_drv, ct_con = st.tabs(["🧑\u200d🏎️  Drivers Championship", "🏭  Constructors Championship"])

        with ct_drv:
            if drv_champ:
                df_drv_c = pd.DataFrame([{
                    "Pos":    d["pos"],
                    "Driver": d["driver_name"],
                    "Code":   d["driver_code"],
                    "Team":   d["team"],
                    "Points": d["points"],
                    "Wins":   d["wins"],
                } for d in drv_champ])

                # Bar chart
                fig_dc = go.Figure(go.Bar(
                    x=df_drv_c["Points"],
                    y=df_drv_c["Code"],
                    orientation="h",
                    marker_color=["#f5c518" if i==0 else ("#aaa" if i==1 else ("#c86a2a" if i==2 else "#e10600"))
                                  for i in range(len(df_drv_c))],
                    text=df_drv_c["Points"].apply(lambda p: f"{p:.0f}"),
                    textposition="outside",
                    textfont={"color": "#d0d0d8", "size": 11},
                    hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
                ))
                fig_dc.update_layout(
                    paper_bgcolor="#0d0d0f", plot_bgcolor="#18181f",
                    xaxis=dict(title="Championship Points", color="#888", gridcolor="#2a2a35"),
                    yaxis=dict(color="#d0d0d8", autorange="reversed", tickfont={"size": 10}),
                    height=max(300, len(df_drv_c) * 26 + 60),
                    margin=dict(t=10, b=30, l=50, r=70),
                    font={"color": "#d0d0d8"},
                )
                st.plotly_chart(fig_dc, use_container_width=True, key=f"prev_champ_drv_chart_{champ_year}")

                st.dataframe(df_drv_c, use_container_width=True, hide_index=True,
                             column_config={
                                 "Pos":    st.column_config.NumberColumn("#",      format="%d",   width="small"),
                                 "Points": st.column_config.NumberColumn("Points", format="%.0f", width="small"),
                                 "Wins":   st.column_config.NumberColumn("Wins",   format="%d",   width="small"),
                             })
            else:
                st.warning(f"No driver standings found for {champ_year}.")

        with ct_con:
            if con_champ:
                df_con_c = pd.DataFrame([{
                    "Pos":    c["pos"],
                    "Team":   c["team"],
                    "Points": c["points"],
                    "Wins":   c["wins"],
                } for c in con_champ])

                fig_cc = go.Figure(go.Bar(
                    x=df_con_c["Points"],
                    y=df_con_c["Team"],
                    orientation="h",
                    marker_color=["#f5c518" if i==0 else "#e10600" for i in range(len(df_con_c))],
                    text=df_con_c["Points"].apply(lambda p: f"{p:.0f}"),
                    textposition="outside",
                    textfont={"color": "#d0d0d8"},
                    hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
                ))
                fig_cc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Points", color="#888", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(color="#d0d0d8", autorange="reversed"),
                    height=max(200, len(df_con_c) * 36 + 50),
                    margin=dict(t=10, b=30, l=160, r=70),
                    font={"color": "#d0d0d8"},
                )
                st.plotly_chart(fig_cc, use_container_width=True, key=f"prev_champ_con_chart_{champ_year}")

                st.dataframe(df_con_c, use_container_width=True, hide_index=True,
                             column_config={
                                 "Pos":    st.column_config.NumberColumn("#",      format="%d",   width="small"),
                                 "Points": st.column_config.NumberColumn("Points", format="%.0f", width="small"),
                                 "Wins":   st.column_config.NumberColumn("Wins",   format="%d",   width="small"),
                             })
            else:
                st.warning(f"No constructor standings found for {champ_year}.")
    else:
        st.info("Select a season above and click **Load Standings** to view.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LIVE TRACK MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📍 Live Track Map")
    st.markdown("Real-time car positions from the OpenF1 telemetry feed.")

    # Refresh toggle
    auto_refresh = st.toggle("Auto-Refresh (every 5s)", value=False)
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
    
    with st.spinner("Fetching live telemetry..."):
        sess = live_data.get_live_session()
        if not sess:
            st.info("No live session currently active (within the last 4 hours).")
            
            # Display countdown to next race
            import datetime
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            schedule = live_data.get_season_schedule()
            next_race = next((r for r in schedule if r["status"] == "next"), None)
            
            if next_race:
                try:
                    race_date = datetime.datetime.strptime(next_race["date"], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                    delta = race_date - now_utc
                    days = delta.days
                    if days > 0:
                        st.info(f"⏳ Next Race: **{next_race['race_name']}** ({next_race['country']}) in **{days} days**.")
                    else:
                        st.info(f"⏳ Next Race: **{next_race['race_name']}** is happening soon!")
                except Exception:
                    st.info(f"⏳ Next Race: **{next_race['race_name']}** on {next_race['date']}")
            else:
                st.info("No upcoming races found in the schedule.")
            
        else:
            session_key = sess["session_key"]
            st.markdown(f"**Session:** {sess['meeting_name']} - {sess['country']}")
            
            locations = live_data.get_live_locations(session_key)
            drivers = live_data.get_live_drivers(session_key)
            
            if not locations:
                st.warning("No location data available for this session yet.")
            else:
                x_vals = []
                y_vals = []
                colors = []
                texts = []
                
                for drv_num, loc in locations.items():
                    x_vals.append(loc["x"])
                    y_vals.append(loc["y"])
                    drv_info = drivers.get(drv_num, {})
                    colors.append(drv_info.get("team_colour", "#ffffff"))
                    code = drv_info.get("code", str(drv_num))
                    texts.append(code)
                
                fig_map = go.Figure(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers+text",
                    marker=dict(size=14, color=colors, line=dict(width=2, color="white")),
                    text=texts,
                    textposition="top center",
                    textfont=dict(color="white", size=10)
                ))
                
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    height=600,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False
                )
                
                fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig_map, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin:32px 0 12px;">
<p style="text-align:center; color:#555; font-size:0.8rem;">
  F1 Pit Strategy AI &nbsp;·&nbsp; Built with FastF1, LightGBM &amp; Streamlit
  &nbsp;·&nbsp; Data: 2021–2025 seasons
</p>
""", unsafe_allow_html=True)
