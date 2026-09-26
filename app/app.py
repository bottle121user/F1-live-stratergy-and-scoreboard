"""
app/app.py  —  F1 Pit Strategy AI Dashboard (Streamlit)
Run with:  streamlit run app/app.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# ----------------------------------------------------------------------------

SRC_DIR = Path(__file__).parent.parent / "src"

sys.path.insert(0, str(SRC_DIR))


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live_data
from predict import load_model, predict_pit, recommend_strategy
from simulator import (
    CIRCUITS,
    HISTORICAL_METRICS_CACHE,
    simulate_strategy,
)

# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Pit Strategy AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------

F1_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=Titillium+Web:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: #FFFFFF !important;
}

/* Text readability */
p, li, .stMarkdown, .stText {
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    color: #F5F7FA !important;
}

/* Labels */
label, .st-bh, .st-bb, [data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
}

/* Pitch Black Minimalist Background with subtle F1 red ambient glow */
[data-testid="stAppViewContainer"] {
    background-color: #0B0D12 !important;
    background-image:
        radial-gradient(circle at 50% 0%, rgba(225, 6, 0, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(255, 65, 54, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 15% 70%, rgba(225, 6, 0, 0.05) 0%, transparent 40%) !important;
    background-attachment: fixed !important;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 2rem !important;
    max-width: 1400px !important;
}

/* Columns: Clean layout containers (NO rigid rectangular boxes!) */
[data-testid="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.6rem !important;
}

/* ============================================================ */
/* PILL FORM CONTROLS (Selectboxes, Inputs, Step-Buttons)       */
/* ============================================================ */
[data-testid="stSelectbox"],
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div:first-child {
    border-radius: 9999px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div {
    border-radius: 9999px !important;
    background-color: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
    box-shadow: inset 0 2px 8px rgba(255, 255, 255, 0.05), 0 8px 20px rgba(0, 0, 0, 0.35) !important;
    min-height: 46px !important;
    padding-left: 10px !important;
    padding-right: 10px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"]:hover > div,
div[data-baseweb="select"]:hover > div {
    border-color: rgba(225, 6, 0, 0.6) !important;
    box-shadow: 0 0 20px rgba(225, 6, 0, 0.25), inset 0 2px 8px rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* BaseWeb Dropdown Menu / Popover */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background-color: #12151D !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(30px) !important;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.75) !important;
    padding: 6px !important;
}

li[role="option"] {
    border-radius: 9999px !important;
    margin: 4px 6px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    color: #E2E8F0 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: rgba(225, 6, 0, 0.25) !important;
    color: #FFFFFF !important;
}

/* PILL INPUTS */
[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input"] {
    border-radius: 9999px !important;
    background-color: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
    box-shadow: inset 0 2px 8px rgba(255, 255, 255, 0.05), 0 8px 20px rgba(0, 0, 0, 0.35) !important;
    overflow: hidden !important;
    transition: all 0.25s ease !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
div[data-baseweb="input"]:hover {
    border-color: rgba(225, 6, 0, 0.6) !important;
    box-shadow: 0 0 20px rgba(225, 6, 0, 0.25), inset 0 2px 8px rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="input"] input {
    background: transparent !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    padding: 0 1.2rem !important;
    border-radius: 9999px !important;
}

[data-testid="stNumberInput"] button {
    background: transparent !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
}

[data-testid="stNumberInput"] button:hover {
    background: rgba(225, 6, 0, 0.25) !important;
    color: #FF4B4B !important;
}

/* ============================================================ */
/* PILL SLIDERS & TOGGLES                                       */
/* ============================================================ */
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    border-radius: 9999px !important;
    background: rgba(225, 6, 0, 0.35) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(225, 6, 0, 0.7) !important;
    color: #FFF !important;
    font-weight: 700 !important;
    padding: 0.2rem 0.8rem !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: #E10600 !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 18px rgba(225, 6, 0, 0.85) !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
    background: linear-gradient(90deg, #E10600, #FF4B4B) !important;
}

[data-testid="stToggle"] label span[data-baseweb="switch"] {
    border-radius: 9999px !important;
    background-color: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

[data-testid="stToggle"] label span[aria-checked="true"] {
    background-color: #E10600 !important;
    box-shadow: 0 0 18px rgba(225, 6, 0, 0.7) !important;
}

/* ============================================================ */
/* GLOWING PILL BUTTONS                                         */
/* ============================================================ */
.stButton > button,
[data-testid="stButton"] > button {
    border-radius: 9999px !important;
    padding: 0.95rem 3rem !important;
    background: linear-gradient(135deg, rgba(225, 6, 0, 0.28), rgba(225, 6, 0, 0.12)) !important;
    border: 1px solid rgba(225, 6, 0, 0.65) !important;
    border-top: 1px solid rgba(255, 80, 80, 0.8) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(225, 6, 0, 0.25), inset 0 1px 8px rgba(255, 255, 255, 0.15) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer;
}

.stButton > button:hover,
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(225, 6, 0, 0.5), rgba(225, 6, 0, 0.3)) !important;
    border-color: #FF4B4B !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 40px rgba(225, 6, 0, 0.6), inset 0 2px 12px rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-2px) scale(1.02);
}

.stButton > button:active,
[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.98);
}

/* ============================================================ */
/* SEGMENTED PILL TABS                                          */
/* ============================================================ */
.stTabs, [data-testid="stTabs"] {
    background: transparent !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"],
.stTabs [data-baseweb="tab-list"],
div[data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border-radius: 9999px !important;
    padding: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4), inset 0 1px 6px rgba(255, 255, 255, 0.06) !important;
    border-bottom: none !important;
}

[data-testid="stTabs"] [data-baseweb="tab"],
.stTabs [data-baseweb="tab"],
button[data-baseweb="tab"] {
    height: 48px !important;
    border-radius: 9999px !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    padding: 0 24px !important;
    transition: all 0.25s ease !important;
    white-space: nowrap !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover,
button[data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(225, 6, 0, 0.35), rgba(225, 6, 0, 0.18)) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(225, 6, 0, 0.35), inset 0 1px 8px rgba(255, 255, 255, 0.2) !important;
    border: 1px solid rgba(225, 6, 0, 0.6) !important;
    border-radius: 9999px !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-highlight"],
div[data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-border"],
div[data-baseweb="tab-border"] {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    background: transparent !important;
    border: none !important;
    visibility: hidden !important;
}

/* ============================================================ */
/* PILL METRIC CARDS                                            */
/* ============================================================ */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 9999px !important;
    padding: 1.4rem 2.5rem !important;
    box-shadow: inset 0 2px 10px rgba(255, 255, 255, 0.06), 0 12px 35px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.28) !important;
    box-shadow: inset 0 2px 12px rgba(255, 255, 255, 0.1), 0 18px 40px rgba(0, 0, 0, 0.6) !important;
}

[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #FFF !important;
}

[data-testid="stMetricLabel"] {
    font-size: 1.05rem !important;
    color: #9298A3 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ============================================================ */
/* FROSTED GLASS PLOTLY CHARTS & DATAFRAMES                     */
/* ============================================================ */
[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(35px) !important;
    -webkit-backdrop-filter: blur(35px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 36px !important;
    padding: 1.5rem !important;
    box-shadow: 0 20px 45px rgba(0,0,0,0.5), inset 0 2px 15px rgba(255,255,255,0.04) !important;
}

[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 28px !important;
    padding: 0.75rem !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.45), inset 0 1px 8px rgba(255,255,255,0.04) !important;
    overflow: hidden !important;
}

/* ============================================================ */
/* PILL ALERTS & EXPANDERS                                      */
/* ============================================================ */
[data-testid="stAlert"] {
    border-radius: 9999px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    padding: 0.9rem 2.2rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
}

[data-testid="stAlert"] p {
    font-size: 1.1rem !important;
    margin: 0 !important;
}

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 28px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    overflow: hidden !important;
}

/* ============================================================ */
/* GIANT DECISION PILLS (Pit Now / Stay Out)                    */
/* ============================================================ */
.pit-now {
    background: rgba(225, 6, 0, 0.16) !important;
    backdrop-filter: blur(40px) !important;
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid rgba(225, 6, 0, 0.65) !important;
    border-top: 1px solid rgba(255, 80, 80, 0.8) !important;
    border-radius: 9999px !important;
    padding: 2.8rem 4rem !important;
    text-align: center;
    box-shadow: 0 0 50px rgba(225, 6, 0, 0.45), inset 0 2px 20px rgba(225, 6, 0, 0.25) !important;
    animation: pulse 2s ease-in-out infinite;
}

.stay-out {
    background: rgba(50, 213, 131, 0.12) !important;
    backdrop-filter: blur(40px) !important;
    -webkit-backdrop-filter: blur(40px) !important;
    border: 1px solid rgba(50, 213, 131, 0.5) !important;
    border-top: 1px solid rgba(80, 255, 160, 0.7) !important;
    border-radius: 9999px !important;
    padding: 2.8rem 4rem !important;
    text-align: center;
    box-shadow: 0 0 45px rgba(50, 213, 131, 0.25), inset 0 2px 20px rgba(50, 213, 131, 0.15) !important;
}

.decision-label {
    font-size: 3rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem !important;
}

.pit-now .decision-label { color: #FF4B4B !important; }
.stay-out .decision-label { color: #32D583 !important; }

.decision-sub {
    font-size: 1.3rem !important;
    color: #E2E8F0 !important;
    font-weight: 500 !important;
    margin: 0;
}

/* ============================================================ */
/* PILL STRATEGY CARDS & BADGES                                 */
/* ============================================================ */
.strat-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 9999px !important;
    padding: 1.2rem 2.5rem !important;
    margin-bottom: 0.9rem !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35), inset 0 1px 8px rgba(255, 255, 255, 0.06) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.strat-card:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(225, 6, 0, 0.6) !important;
    box-shadow: 0 15px 35px rgba(225, 6, 0, 0.2), inset 0 1px 10px rgba(255, 255, 255, 0.1) !important;
}

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    background: rgba(225, 6, 0, 0.25);
    border: 1px solid rgba(225, 6, 0, 0.65);
    color: #FF5A5A;
    font-weight: 800;
    font-size: 1rem;
    margin-right: 0.85rem;
}

/* ============================================================ */
/* PODIUM PILL CARDS                                            */
/* ============================================================ */
.podium-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 36px !important;
    padding: 1.8rem 1.4rem !important;
    text-align: center;
    margin-bottom: 1rem !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.3s ease !important;
}

.podium-card:hover {
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.08) !important;
}

/* ============================================================ */
/* PILL BANNERS & BADGES                                        */
/* ============================================================ */
.f1-live-banner {
    background: rgba(225, 6, 0, 0.18) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border: 1px solid rgba(225, 6, 0, 0.55) !important;
    border-radius: 9999px !important;
    padding: 1.2rem 2.8rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 10px 30px rgba(225, 6, 0, 0.3) !important;
}

.champion-banner {
    background: rgba(245, 197, 24, 0.09) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border: 1px solid rgba(245, 197, 24, 0.5) !important;
    border-radius: 9999px !important;
    padding: 1.5rem 3.2rem !important;
    margin: 1.5rem 0 !important;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5), inset 0 2px 15px rgba(245, 197, 24, 0.15) !important;
}

.calendar-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 9999px !important;
    padding: 0.85rem 1.6rem !important;
    margin-bottom: 0.8rem !important;
    text-align: center;
    transition: all 0.25s ease !important;
}

.calendar-card:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.09) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

/* Animations */
@keyframes pulse {
    0%   { box-shadow: 0 0 35px rgba(225,6,0,0.25); }
    50%  { box-shadow: 0 0 75px rgba(225,6,0,0.55); }
    100% { box-shadow: 0 0 35px rgba(225,6,0,0.25); }
}

.f1-light-box { display: flex; gap: 8px; }
.f1-light {
    width: 20px; height: 20px; border-radius: 50%;
    background: #333; border: 2px solid #555;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.8);
}

@keyframes lightSeq {
    0%, 10% { background: #333; box-shadow: none; border-color: #555; }
    20%, 90% { background: #e10600; box-shadow: 0 0 15px #e10600; border-color: #ff4136; }
    100% { background: #333; box-shadow: none; border-color: #555; }
}

.light-1 { animation: lightSeq 4s infinite 0.2s; }
.light-2 { animation: lightSeq 4s infinite 0.4s; }
.light-3 { animation: lightSeq 4s infinite 0.6s; }
.light-4 { animation: lightSeq 4s infinite 0.8s; }
.light-5 { animation: lightSeq 4s infinite 1.0s; }

/* Responsive Breakpoints */
@media (max-width: 1024px) {
    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        padding: 6px !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px !important;
        padding: 0 16px !important;
        font-size: 0.95rem !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }

    .pit-now, .stay-out {
        padding: 1.8rem 1.2rem !important;
        border-radius: 9999px !important;
    }
    .decision-label {
        font-size: clamp(2rem, 6vw, 2.5rem) !important;
        letter-spacing: 0.05em !important;
    }

    .stButton > button {
        width: 100% !important;
        padding: 0.85rem 1.5rem !important;
    }

    [data-testid="stMetric"] {
        border-radius: 9999px !important;
        padding: 1rem 1.5rem !important;
    }

    .f1-hero-header {
        flex-direction: column !important;
        gap: 16px !important;
        padding: 1.5rem 1.2rem !important;
        border-radius: 36px !important;
    }
}
</style>
"""

if hasattr(st, "html"):
    st.html(F1_CUSTOM_CSS)
else:
    st.markdown(F1_CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="f1-hero-header" style="display:flex; align-items:center; justify-content:center; gap:clamp(16px, 3vw, 32px); margin-bottom:clamp(1.5rem, 3vw, 3rem); padding:clamp(1.5rem, 3vw, 3rem); background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:clamp(24px, 5vw, 9999px); backdrop-filter:blur(30px); -webkit-backdrop-filter:blur(30px); box-shadow:inset 0 2px 10px rgba(255,255,255,0.02), 0 20px 40px rgba(0,0,0,0.4); transform: translateZ(0); flex-wrap:wrap; text-align:center;">
<div style="display:flex; align-items:center; justify-content:center; width:clamp(60px, 8vw, 80px); height:clamp(60px, 8vw, 80px); background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:50%; font-size:clamp(1.8rem, 3vw, 2.5rem); box-shadow:inset 0 2px 10px rgba(255,255,255,0.05);">🏎️</div>
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
<h1 style="margin:0; font-size:clamp(1.8rem, 4.5vw, 3.2rem); font-family:'Outfit', sans-serif; font-weight:800; letter-spacing:clamp(1px, 0.4vw, 4px); color:#FFFFFF; text-transform:uppercase; line-height:1.1;">F1 PIT STRATEGY AI</h1>
<div class="f1-light-box" style="margin-top:14px; transform:scale(clamp(0.75, 1vw, 0.95)); transform-origin:center; align-self:center;">
<div class="f1-light light-1"></div>
<div class="f1-light light-2"></div>
<div class="f1-light light-3"></div>
<div class="f1-light light-4"></div>
<div class="f1-light light-5"></div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------


@st.cache_resource
def get_model():

    try:
        return load_model()

    except FileNotFoundError:
        return None


model_bundle = get_model()

model_ok = model_bundle is not None


if not model_ok:
    st.error(
        "⚠️ No trained model found. Run `python src/data_loader.py` then `python src/model.py` first."
    )

    st.stop()


# ----------------------------------------------------------------------------


def render_rain():

    drops_html = "".join(
        [
            f'<div class="drop" style="left:{np.random.randint(0, 100)}%;'
            f"animation-delay:{np.random.random() * 0.5}s;"
            f"animation-duration:{0.4 + np.random.random() * 0.3}s;"
            f'opacity:{0.3 + np.random.random() * 0.5}"></div>'
            for _ in range(70)
        ]
    )

    st.markdown(
        f'<div class="rain-container">{drops_html}</div>', unsafe_allow_html=True
    )


# ----------------------------------------------------------------------------

available_teams = sorted(
    list(HISTORICAL_METRICS_CACHE.get("team_pace_offsets", {}).keys())
) or ["Red Bull Racing", "Mercedes", "Ferrari", "McLaren", "Aston Martin"]

available_drivers = sorted(
    list(HISTORICAL_METRICS_CACHE.get("driver_tyre_factors", {}).keys())
) or ["VER", "HAM", "LEC", "NOR", "ALO"]

DRIVER_NAMES = {
    "VER": "Max Verstappen",
    "HAM": "Lewis Hamilton",
    "LEC": "Charles Leclerc",
    "NOR": "Lando Norris",
    "ALO": "Fernando Alonso",
    "SAI": "Carlos Sainz",
    "RUS": "George Russell",
    "PIA": "Oscar Piastri",
    "PER": "Sergio Perez",
    "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",
    "OCO": "Esteban Ocon",
    "ALB": "Alexander Albon",
    "TSU": "Yuki Tsunoda",
    "BOT": "Valtteri Bottas",
    "ZHO": "Zhou Guanyu",
    "MAG": "Kevin Magnussen",
    "HUL": "Nico Hulkenberg",
    "RIC": "Daniel Ricciardo",
    "SAR": "Logan Sargeant",
    "BEA": "Oliver Bearman",
    "LAW": "Liam Lawson",
    "COL": "Franco Colapinto",
}


# ----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🏎️ Live Pit Decision",
        "📊 Strategy Recommender",
        "📈 Lap Simulator",
        "🏆 Live Scoreboard",
        "🏆 Previous Champions",
        "📍 Live Track Map",
    ]
)


# ----------------------------------------------------------------------------

# TAB 1 — LIVE PIT DECISION

# ----------------------------------------------------------------------------

with tab1:
    st.subheader("Current Race Conditions")

    st.markdown(
        "Enter the live race state and get an instant pit recommendation from the AI."
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Driver & Circuit**")

        team_t1 = st.selectbox("Constructors", available_teams, key="team_1")

        driver_t1 = st.selectbox(
            "Driver",
            available_drivers,
            format_func=lambda x: DRIVER_NAMES.get(x, x),
            key="drv_1",
        )

        circuit_t1 = st.selectbox("Circuit", list(CIRCUITS.keys()), key="c_t1")

        st.markdown("---")

        st.markdown("**Race Position**")

        lap_number = st.slider("Current Lap", 1, 80, 30)

        laps_remaining = st.slider("Laps Remaining", 0, 80, 25)

        position = st.slider("Current Position", 1, 20, 8)

        is_sc = st.toggle("Safety Car / VSC Active", value=False)

        st.markdown("---")

        pit_threshold = st.slider(
            "Pit Sensitivity (threshold)",
            0.20,
            0.50,
            0.35,
            0.01,
            help="Lower = more aggressive pitting. Default 0.35 corrects for class imbalance in training data.",
        )

    with col_r:
        st.markdown("**Tyre Status**")

        compound = st.selectbox(
            "Compound on Car", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
        )

        laps_since_pit = st.slider("Laps on Current Tyre", 1, 55, 14)

        stint_number = st.slider("Stint Number", 1, 4, 1)

        lap_time_sec = st.number_input("Latest Lap Time (s)", 60.0, 130.0, 92.5, 0.1)

        lap_time_delta = st.number_input(
            "Lap Time Delta vs Rolling Avg (s)",
            -5.0,
            10.0,
            0.8,
            0.1,
            help="Positive = getting slower (degradation)",
        )

        st.markdown("**Weather**")

        wc1, wc2 = st.columns(2)

        with wc1:
            air_temp = st.number_input("Air Temp (°C)", -5.0, 50.0, 26.0, 0.5)

            track_temp = st.number_input("Track Temp (°C)", 10.0, 70.0, 38.0, 0.5)

            rainfall = st.number_input(
                "Rainfall (0-1)", 0.0, 1.0, 0.0, 0.1, key="rain_1"
            )

        if rainfall > 0:
            render_rain()

        with wc2:
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 55.0, 1.0)

            wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button("🧠 Get Pit Recommendation", key="btn_t1")

    if run_btn:
        circuit_list = list(CIRCUITS.keys())

        track_encoded = (
            circuit_list.index(circuit_t1) if circuit_t1 in circuit_list else 0
        )

        result = predict_pit(
            lap_number=lap_number,
            laps_since_pit=float(laps_since_pit),
            compound=compound,
            lap_time_seconds=lap_time_sec,
            lap_time_delta=lap_time_delta,
            stint_number=float(stint_number),
            laps_remaining=float(laps_remaining),
            is_safety_car=int(is_sc),
            position=float(position),
            air_temp=air_temp,
            track_temp=track_temp,
            rainfall=rainfall,
            humidity=humidity,
            wind_speed=wind_speed,
            track_encoded=track_encoded,
            team=team_t1,
            driver=driver_t1,
            pit_threshold=pit_threshold,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        decision = result["decision"]

        conf = result["confidence"]

        pit_p = result["pit_probability"]

        stay_p = result["stay_probability"]

        if decision == "PIT NOW":
            st.markdown(
                f"""
<div class="pit-now">
<p class="decision-label">🔴 PIT NOW</p>
<p class="decision-sub">Confidence: {conf:.1%} &nbsp;|&nbsp; Pit probability: {pit_p:.1%}</p>
</div>""",
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                f"""
<div class="stay-out">
<p class="decision-label">🟢 STAY OUT</p>
<p class="decision-sub">Confidence: {conf:.1%} &nbsp;|&nbsp; Stay probability: {stay_p:.1%}</p>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Pit Probability", f"{pit_p:.1%}")

        m2.metric("Stay Probability", f"{stay_p:.1%}")

        m3.metric("Confidence", f"{conf:.1%}")

        m4.metric(
            "Tyre Age", f"{laps_since_pit} laps"
        )  # Custom Telemetry Probability Bar

        bar_color = (
            "#32D583" if pit_p < 0.4 else ("#F5C451" if pit_p < 0.65 else "#FF4B4B")
        )

        st.markdown(
            f"""
<div style="margin-top: 2rem; padding: 2.5rem; background: rgba(255,255,255,0.02); border-radius: 40px; border: 1px solid rgba(255,255,255,0.08); box-shadow: inset 0 2px 10px rgba(255,255,255,0.02);">
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 16px;">
<span style="color: #9298A3; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: 1.1rem;">Live AI Telemetry Analysis</span>
<span style="color: #FFF; font-weight: 900; font-size: 2rem;">{pit_p:.1%} <span style="font-size:1rem; color:#888;">PIT PROBABILITY</span></span>
</div>
<div style="width: 100%; height: 16px; background: rgba(0,0,0,0.6); border-radius: 9999px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
<div style="width: {pit_p * 100}%; height: 100%; background: {bar_color}; border-radius: 9999px; box-shadow: 0 0 20px {bar_color}; transition: width 1.5s cubic-bezier(0.16, 1, 0.3, 1);"></div>
</div>
</div>
        """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------

# TAB 2 — STRATEGY RECOMMENDER

# ----------------------------------------------------------------------------

with tab2:
    st.subheader("Race Strategy Recommender")

    st.markdown(
        "Select a circuit and weather conditions to get the AI-ranked optimal strategies."
    )

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns([2, 2, 1.5, 1.5, 1, 1])

    with sc1:
        team_t2 = st.selectbox("Constructor", available_teams, key="team_2")

    with sc2:
        driver_t2 = st.selectbox(
            "Driver",
            available_drivers,
            format_func=lambda x: DRIVER_NAMES.get(x, x),
            key="drv_2",
        )

    with sc3:
        circuit_t2 = st.selectbox(
            "Circuit",
            list(CIRCUITS.keys()),
            key="c_t2",
            index=list(CIRCUITS.keys()).index("Bahrain Grand Prix"),
        )

    with sc4:
        start_cpd_t2 = st.selectbox(
            "Start Tyre", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], key="cpd_2"
        )

    with sc5:
        trk_temp_t2 = st.number_input("Track Temp", 10.0, 70.0, 38.0, 1.0, key="tt2")

    with sc6:
        rain_t2 = st.number_input("Rain", 0.0, 1.0, 0.0, 0.1, key="rain_2")

    top_n = st.selectbox("Show Top N Strategies", [3, 5, 7], key="topn")

    if rain_t2 > 0:
        render_rain()

    strat_btn = st.button("⚙️ Generate Strategies", key="btn_t2")

    if strat_btn:
        weather_t2 = {"track_temp": trk_temp_t2, "rainfall": rain_t2}

        with st.spinner("Simulating strategies..."):
            ranked = recommend_strategy(
                circuit_t2,
                weather=weather_t2,
                top_n=top_n,
                team=team_t2,
                driver=driver_t2,
                starting_compound=start_cpd_t2,
            )

        if not ranked:
            st.error(
                "No strategies returned. Try different circuit or weather settings."
            )

        else:
            st.markdown("<br>", unsafe_allow_html=True)

            best_time = ranked[0]["total_time"]

            for r in ranked:
                delta = r["total_time"] - best_time

                delta_str = f"+{delta:.1f}s" if delta > 0 else "Fastest ⚡"

                pit_desc = (
                    ", ".join(
                        [
                            f"Lap {p['pit_lap']} → {p['compound'].capitalize()}"
                            for p in r["strategy"]
                        ]
                    )
                    if r["strategy"]
                    else "No pit stop"
                )

                st.markdown(
                    f"""
<div class="strat-card">
<span class="rank-badge">#{r["rank"]}</span>
<strong style="color:#f0f0f0; font-size:1.05rem;">{r["label"]}</strong>
<span style="color:#888; font-size:0.9rem; margin-left:10px;">{delta_str}</span>
<br>
<span style="color:#aaa; font-size:0.88rem; margin-top:6px; display:block;">{pit_desc}</span>
<span style="color:#e10600; font-weight:700; font-size:1rem; float:right; margin-top:-24px;">{r["total_time"]:,.1f}s</span>
</div>
                """,
                    unsafe_allow_html=True,
                )

            labels = [f"#{r['rank']} {r['label']}" for r in ranked]

            times = [r["total_time"] for r in ranked]

            colors_bar = [
                "#e10600" if i == 0 else "#3a3a4a" for i in range(len(ranked))
            ]

            fig2 = go.Figure(
                go.Bar(
                    x=times,
                    y=labels,
                    orientation="h",
                    marker_color=colors_bar,
                    text=[f"{t:,.1f}s" for t in times],
                    textposition="outside",
                    textfont={"color": "#d0d0d8"},
                )
            )

            fig2.update_layout(
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#18181f",
                xaxis=dict(
                    title="Total Race Time (s)", color="#888", gridcolor="#2a2a35"
                ),
                yaxis=dict(color="#d0d0d8", autorange="reversed"),
                height=300 + len(ranked) * 28,
                margin=dict(t=20, b=20, l=120, r=100),
                font={"color": "#d0d0d8"},
            )

            st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------------------------------

# TAB 3 — LAP SIMULATOR

# ----------------------------------------------------------------------------

with tab3:
    st.subheader("Lap-by-Lap Tyre Degradation Simulator")

    st.markdown(
        "Compare how different strategies play out lap by lap across the race distance."
    )

    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns([2, 2, 1.5, 1.5, 1, 1])

    with lc1:
        team_t3 = st.selectbox("Constructor", available_teams, key="team_3")

    with lc2:
        driver_t3 = st.selectbox(
            "Driver",
            available_drivers,
            format_func=lambda x: DRIVER_NAMES.get(x, x),
            key="drv_3",
        )

    with lc3:
        circuit_t3 = st.selectbox(
            "Circuit",
            list(CIRCUITS.keys()),
            key="c_t3",
            index=list(CIRCUITS.keys()).index("British Grand Prix"),
        )

    with lc4:
        start_cpd_t3 = st.selectbox(
            "Start Tyre", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], key="cpd_3"
        )

    with lc5:
        trk_temp_t3 = st.number_input("Track Temp", 10.0, 70.0, 30.0, 1.0, key="tt3")

    with lc6:
        rain_t3 = st.number_input("Rain", 0.0, 1.0, 0.0, 0.1, key="rain_3")

    if rain_t3 > 0:
        render_rain()

    sim_btn = st.button("🏎️ Run Simulation", key="btn_t3")

    if sim_btn:
        weather_t3 = {"track_temp": trk_temp_t3, "rainfall": rain_t3}

        n_laps = CIRCUITS[circuit_t3]["laps"]

        # Compute pit laps safely (min lap 2, max n_laps-3)

        def _pit(frac: float) -> int:

            return max(2, min(n_laps - 3, round(n_laps * frac)))

        strategies_t3 = [
            {"label": "No-stop", "strategy": []},
            {
                "label": "1-stop (early)",
                "strategy": [{"pit_lap": _pit(0.35), "compound": "HARD"}],
            },
            {
                "label": "1-stop (late)",
                "strategy": [{"pit_lap": _pit(0.50), "compound": "HARD"}],
            },
            {
                "label": "2-stop",
                "strategy": [
                    {"pit_lap": _pit(0.28), "compound": "MEDIUM"},
                    {"pit_lap": _pit(0.58), "compound": "HARD"},
                ],
            },
        ]

        COLORS = ["#e10600", "#00d2ff", "#f5a623", "#7ed321"]

        fig3 = go.Figure()

        for idx, s in enumerate(strategies_t3):
            result_s = simulate_strategy(
                circuit_t3,
                s["strategy"],
                weather_t3,
                team=team_t3,
                driver=driver_t3,
                starting_compound=start_cpd_t3,
            )

            laps_x = [lr.lap for lr in result_s.lap_records]

            times_y = [lr.time for lr in result_s.lap_records]

            fig3.add_trace(
                go.Scatter(
                    x=laps_x,
                    y=times_y,
                    mode="lines",
                    name=s["label"],
                    line=dict(color=COLORS[idx], width=2.5),
                    hovertemplate=f"Lap %{{x}}<br>Lap time: %{{y:.2f}}s<extra>{s['label']}</extra>",
                )
            )

            for pit_lap in result_s.pit_laps:
                if 1 <= pit_lap - 1 < len(result_s.lap_records):
                    fig3.add_vline(
                        x=pit_lap,
                        line_dash="dot",
                        line_color=COLORS[idx],
                        opacity=0.4,
                        annotation_text=f"Pit ({s['label'].split()[0]})",
                        annotation_font_color=COLORS[idx],
                        annotation_font_size=10,
                    )

        fig3.update_layout(
            paper_bgcolor="#0d0d0f",
            plot_bgcolor="#18181f",
            xaxis=dict(title="Lap Number", color="#888", gridcolor="#2a2a35"),
            yaxis=dict(title="Lap Time (seconds)", color="#888", gridcolor="#2a2a35"),
            legend=dict(
                bgcolor="#18181f", bordercolor="#2a2a35", font={"color": "#d0d0d8"}
            ),
            hovermode="x unified",
            height=480,
            margin=dict(t=20, b=40, l=60, r=20),
            font={"color": "#d0d0d8"},
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Strategy Comparison Table**")

        rows = []

        for s in strategies_t3:
            r = simulate_strategy(
                circuit_t3,
                s["strategy"],
                weather_t3,
                team=team_t3,
                driver=driver_t3,
                starting_compound=start_cpd_t3,
            )

            rows.append(
                {
                    "Strategy": s["label"],
                    "Pit Laps": ", ".join(map(str, r.pit_laps)) or "—",
                    "Total Time": f"{r.total_time:,.1f}s",
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------

# TAB 4 — LIVE SCOREBOARD

# ----------------------------------------------------------------------------

with tab4:
    st.subheader("F1 Live Scoreboard & Championship Standings")

    st.markdown(
        "Live data from the [Jolpica (Ergast) API](https://api.jolpi.ca) and [OpenF1](https://openf1.org)."
    )

    # Refresh control

    sb_cols = st.columns([5, 1])

    with sb_cols[1]:
        refresh = st.button("🔄 Refresh", key="btn_refresh")

    if "scoreboard_data" not in st.session_state or refresh:
        with st.spinner("Fetching live F1 data..."):

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
                "driver_standings": drv_rows,
                "driver_standings_season": drv_season,
                "constructor_standings": con_rows,
                "constructor_standings_season": con_season,
                "last_race": live_data.get_last_race_results(),
                "schedule": live_data.get_season_schedule(),
                "live_session": live_data.get_live_session(),
            }

    sb = st.session_state["scoreboard_data"]

# ----------------------------------------------------------------------------

    live_sess = sb.get("live_session")

    if live_sess:
        sess_key = live_sess["session_key"]

        live_pos = live_data.get_live_positions(sess_key)

        live_drvs = live_data.get_live_drivers(sess_key)

        if live_pos:
            st.markdown(
                f"""
<div class="f1-live-banner">
<p style="margin:0; font-size:0.8rem; text-transform:uppercase; letter-spacing:2px; color:#FF4B4B; font-weight:800;">🔴 LIVE NOW</p>
<p style="margin:4px 0 0; font-size:1.4rem; font-weight:900; color:#fff;">
{live_sess["meeting_name"]} &nbsp;·&nbsp; {live_sess["circuit"]}, {live_sess["country"]}
</p>
</div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 🏁 Live Race Order")

            live_rows = []

            for entry in live_pos:
                drv_num = entry.get("driver_number")

                drv_info = live_drvs.get(drv_num, {})

                live_rows.append(
                    {
                        "Pos": entry.get("position", "—"),
                        "#": drv_num,
                        "Code": drv_info.get("code", str(drv_num)),
                        "Driver": drv_info.get("full_name", "—"),
                        "Team": drv_info.get("team", "—"),
                    }
                )

            if live_rows:
                st.dataframe(
                    pd.DataFrame(live_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Pos": st.column_config.NumberColumn(
                            "Pos", format="%d", width="small"
                        )
                    },
                )

            st.caption("Live positions update on each Refresh.")

            st.markdown("---")

# ----------------------------------------------------------------------------

    schedule = sb.get("schedule", [])

    next_race = next((r for r in schedule if r["status"] == "next"), None)

    if next_race:
        st.caption(f"Next race: **{next_race['race_name']}** — {next_race['date']}")

    drv_std = sb.get("driver_standings", [])

    drv_szn = sb.get("driver_standings_season", "current")

    con_std = sb.get("constructor_standings", [])

    con_szn = sb.get("constructor_standings_season", "current")

    tab_drv, tab_con = st.tabs(
        [
            "\U0001f9d1\u200d\U0001f3ce\ufe0f  Drivers Championship",
            "\U0001f3ed  Constructors Championship",
        ]
    )

    with tab_drv:
        if not drv_std:
            st.warning(
                "Could not fetch driver standings — check your internet connection."
            )

        else:
            if "final" in drv_szn:
                st.info(
                    f"\U0001f4cc Showing **{drv_szn}** standings — 2026 season kicks off March 8."
                )

            # Podium cards

            podium = drv_std[:3]

            medals = ["🥇", "🥈", "🥉"]

            medal_bg = ["#3d2e00", "#1a1a1a", "#2a1000"]

            medal_border = ["#f5c518", "#aaa", "#c86a2a"]

            pod_cols = st.columns(3)

            for i, (col, drv) in enumerate(zip(pod_cols, podium)):
                with col:
                    st.markdown(
                        f"""
<div class="podium-card" style="border-color:{medal_border[i]}88; box-shadow:0 10px 30px rgba(0,0,0,0.5), inset 0 2px 10px {medal_border[i]}22;">
<p style="font-size:2rem; margin:0;">{medals[i]}</p>
<p style="font-size:1.4rem; font-weight:900; color:#fff; margin:6px 0 2px;">{drv["driver_code"]}</p>
<p style="font-size:0.9rem; color:#aaa; margin:0;">{drv["driver_name"]}</p>
<p style="font-size:0.85rem; color:#888; margin:4px 0 0;">{drv["team"]}</p>
<p style="font-size:1.6rem; font-weight:900; color:{medal_border[i]}; margin:8px 0 0;">{drv["points"]:.0f} pts</p>
</div>
                    """,
                        unsafe_allow_html=True,
                    )

            # Bar chart

            df_drv = pd.DataFrame(drv_std)

            df_drv.columns = [
                "Pos",
                "Code",
                "Driver",
                "Nationality",
                "Team",
                "Points",
                "Wins",
            ]

            fig_pts = go.Figure(
                go.Bar(
                    x=df_drv["Points"],
                    y=df_drv["Code"],
                    orientation="h",
                    marker_color=[
                        "#f5c518"
                        if i == 0
                        else (
                            "#aaa" if i == 1 else ("#c86a2a" if i == 2 else "#e10600")
                        )
                        for i in range(len(df_drv))
                    ],
                    text=df_drv["Points"].apply(lambda p: f"{p:.0f}"),
                    textposition="outside",
                    textfont={"color": "#d0d0d8", "size": 11},
                    hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
                )
            )

            fig_pts.update_layout(
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#18181f",
                xaxis=dict(
                    title="Championship Points", color="#888", gridcolor="#2a2a35"
                ),
                yaxis=dict(
                    color="#d0d0d8", autorange="reversed", tickfont={"size": 10}
                ),
                height=max(300, len(df_drv) * 26 + 60),
                margin=dict(t=10, b=30, l=50, r=70),
                font={"color": "#d0d0d8"},
            )

            st.plotly_chart(fig_pts, use_container_width=True)

            st.dataframe(
                df_drv,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pos": st.column_config.NumberColumn(
                        "#", format="%d", width="small"
                    ),
                    "Points": st.column_config.NumberColumn(
                        "Points", format="%.0f", width="small"
                    ),
                    "Wins": st.column_config.NumberColumn(
                        "Wins", format="%d", width="small"
                    ),
                },
            )

    with tab_con:
        if not con_std:
            st.warning(
                "Could not fetch constructor standings — check your internet connection."
            )

        else:
            if "final" in con_szn:
                st.info(f"\U0001f4cc Showing **{con_szn}** standings.")

            df_con = pd.DataFrame(con_std)

            df_con.columns = ["Pos", "Team", "Nationality", "Points", "Wins"]

            fig_con = go.Figure(
                go.Bar(
                    x=df_con["Points"],
                    y=df_con["Team"],
                    orientation="h",
                    marker_color=[
                        "#f5c518" if i == 0 else "#e10600" for i in range(len(df_con))
                    ],
                    text=df_con["Points"].apply(lambda p: f"{p:.0f}"),
                    textposition="outside",
                    textfont={"color": "#d0d0d8"},
                    hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
                )
            )

            fig_con.update_layout(
                paper_bgcolor="#0d0d0f",
                plot_bgcolor="#18181f",
                xaxis=dict(title="Points", color="#888", gridcolor="#2a2a35"),
                yaxis=dict(color="#d0d0d8", autorange="reversed"),
                height=max(200, len(df_con) * 36 + 50),
                margin=dict(t=10, b=30, l=160, r=70),
                font={"color": "#d0d0d8"},
            )

            st.plotly_chart(fig_con, use_container_width=True)

            st.dataframe(
                df_con,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pos": st.column_config.NumberColumn(
                        "#", format="%d", width="small"
                    ),
                    "Points": st.column_config.NumberColumn(
                        "Points", format="%.0f", width="small"
                    ),
                    "Wins": st.column_config.NumberColumn(
                        "Wins", format="%d", width="small"
                    ),
                },
            )

# ----------------------------------------------------------------------------

    st.markdown("---")

    last = sb.get("last_race", {})

    if last:
        st.markdown(f"### 🏁 Last Race: {last.get('race_name', '—')}")

        st.caption(f"{last.get('circuit', '—')} &nbsp;·&nbsp; {last.get('date', '—')}")

        results = last.get("results", [])

        if results:
            top3 = [r for r in results if r["pos"] <= 3]

            top3_cols = st.columns(len(top3))

            for col, r in zip(top3_cols, top3):
                medal = ["🥇", "🥈", "🥉"][r["pos"] - 1]

                with col:
                    st.markdown(
                        f"""
<div class="podium-card" style="padding:1.4rem 1rem;">
<p style="font-size:1.8rem; margin:0;">{medal}</p>
<p style="font-size:1.3rem; font-weight:900; color:#fff; margin:4px 0 2px;">{r["driver_code"]}</p>
<p style="font-size:0.85rem; color:#aaa; margin:0;">{r["driver_name"]}</p>
<p style="font-size:0.82rem; color:#888; margin:2px 0 0;">{r["team"]}</p>
<p style="font-size:1.2rem; font-weight:800; color:#FF4B4B; margin:8px 0 0;">+{r["points"]:.0f} pts</p>
</div>
                    """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            df_res = pd.DataFrame(
                [
                    {
                        "Pos": r["pos"],
                        "Code": r["driver_code"],
                        "Driver": r["driver_name"],
                        "Team": r["team"],
                        "Grid": r["grid"],
                        "Laps": r["laps"],
                        "Status": r["status"],
                        "Points": r["points"],
                        "Fastest Lap": r["fastest_lap"]
                        + (" 🟣" if r["fl_rank"] == 1 else ""),
                    }
                    for r in results
                ]
            )

            st.dataframe(
                df_res,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pos": st.column_config.NumberColumn(
                        "#", format="%d", width="small"
                    ),
                    "Grid": st.column_config.NumberColumn(
                        "Grid", format="%d", width="small"
                    ),
                    "Laps": st.column_config.NumberColumn(
                        "Laps", format="%d", width="small"
                    ),
                    "Points": st.column_config.NumberColumn(
                        "Pts", format="%.0f", width="small"
                    ),
                },
            )

    else:
        st.info("Last race results not available — try refreshing.")

# ----------------------------------------------------------------------------

    st.markdown("---")

    st.markdown("### 🗓️ Season Calendar")

    if schedule:
        STATUS_ICON = {"done": "✅", "next": "⚡ï¸", "upcoming": "🔜"}

        STATUS_COLOR = {"done": "#2a2a35", "next": "#3a1500", "upcoming": "#18181f"}

        STATUS_BORDER = {"done": "#3a3a4a", "next": "#e10600", "upcoming": "#2a2a35"}

        for i in range(0, len(schedule), 4):
            chunk = schedule[i : i + 4]

            cal_cols = st.columns(len(chunk))

            for col, race in zip(cal_cols, chunk):
                icon = STATUS_ICON.get(race["status"], "")

                bg = STATUS_COLOR.get(race["status"], "#18181f")

                brd = STATUS_BORDER.get(race["status"], "#2a2a35")

                with col:
                    st.markdown(
                        f"""
<div class="calendar-card" style="border-color:{brd};">
<p style="margin:0; font-size:0.75rem; color:#888; text-transform:uppercase; font-weight:600;">Rd {race["round"]}</p>
<p style="margin:2px 0; font-size:0.95rem; font-weight:800; color:#f0f0f0;">{icon} {race["country"]}</p>
<p style="margin:0; font-size:0.75rem; color:#aaa;">{race["date"]}</p>
</div>
                    """,
                        unsafe_allow_html=True,
                    )

    else:
        st.info("Season calendar not available — try refreshing.")


# ----------------------------------------------------------------------------

# TAB 5 — PREVIOUS CHAMPIONS

# ----------------------------------------------------------------------------

with tab5:
    st.subheader("F1 Previous Champions")

    st.markdown(
        "Select a season to view the final World Drivers' and Constructors' Championship standings."
    )

    CHAMP_YEARS = list(range(2025, 1999, -1))  # 2025 → 2000

    champ_year = st.selectbox(
        "Season",
        options=CHAMP_YEARS,
        format_func=lambda y: f"🏁 {y} Season",
        index=0,
        key="champ_year",
    )

    load_champ_btn = st.button("📊 Load Standings", key="btn_champ")

    champ_cache_key = f"champ_{champ_year}"

    if load_champ_btn or champ_cache_key not in st.session_state:
        with st.spinner(f"Fetching {champ_year} championship standings..."):
            drv_data, _ = live_data.get_driver_standings(str(champ_year))

            con_data, _ = live_data.get_constructor_standings(str(champ_year))

            st.session_state[champ_cache_key] = {
                "drivers": drv_data,
                "constructors": con_data,
            }

    champ_sb = st.session_state.get(champ_cache_key, {})

    drv_champ = champ_sb.get("drivers", [])

    con_champ = champ_sb.get("constructors", [])

    if drv_champ or con_champ:
# ----------------------------------------------------------------------------

        if drv_champ:
            wdc = drv_champ[0]

            wcc = con_champ[0] if con_champ else {}

            st.markdown(
                f"""
<div class="champion-banner">
<div>
<p style="margin:0; font-size:0.8rem; text-transform:uppercase; letter-spacing:2px;
color:#f5c518; font-weight:800;">🏆 {champ_year} World Champion</p>
<p style="margin:6px 0 2px; font-size:2.2rem; font-weight:900; color:#fff;">
{wdc.get("driver_name", "—")}
</p>
<p style="margin:0; font-size:1rem; color:#aaa; font-weight:600;">{wdc.get("team", "—")}</p>
</div>
<div style="text-align:right;">
<p style="margin:0; font-size:3rem; font-weight:900; color:#f5c518;">
{wdc.get("points", 0):.0f} <span style="font-size:1.1rem; color:#aaa;">pts</span>
</p>
<p style="margin:4px 0 0; font-size:0.9rem; color:#aaa; font-weight:600;">{wdc.get("wins", 0)} wins</p>
</div>
{f'<div><p style="margin:0; font-size:0.8rem; color:#f5c518; letter-spacing:2px; text-transform:uppercase; font-weight:800;">🏭 WCC</p><p style="margin:4px 0 2px; font-size:1.4rem; font-weight:900; color:#fff;">{wcc.get("team", "—")}</p><p style="font-size:0.95rem; color:#aaa; margin:0; font-weight:600;">{wcc.get("points", 0):.0f} pts</p></div>' if wcc else ""}
</div>
            """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------

        ct_drv, ct_con = st.tabs(
            ["🏎️  Drivers Championship", "🏭  Constructors Championship"]
        )

        with ct_drv:
            if drv_champ:
                df_drv_c = pd.DataFrame(
                    [
                        {
                            "Pos": d["pos"],
                            "Driver": d["driver_name"],
                            "Code": d["driver_code"],
                            "Team": d["team"],
                            "Points": d["points"],
                            "Wins": d["wins"],
                        }
                        for d in drv_champ
                    ]
                )

                # Bar chart

                fig_dc = go.Figure(
                    go.Bar(
                        x=df_drv_c["Points"],
                        y=df_drv_c["Code"],
                        orientation="h",
                        marker_color=[
                            "#f5c518"
                            if i == 0
                            else (
                                "#aaa"
                                if i == 1
                                else ("#c86a2a" if i == 2 else "#e10600")
                            )
                            for i in range(len(df_drv_c))
                        ],
                        text=df_drv_c["Points"].apply(lambda p: f"{p:.0f}"),
                        textposition="outside",
                        textfont={"color": "#d0d0d8", "size": 11},
                        hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
                    )
                )

                fig_dc.update_layout(
                    paper_bgcolor="#0d0d0f",
                    plot_bgcolor="#18181f",
                    xaxis=dict(
                        title="Championship Points", color="#888", gridcolor="#2a2a35"
                    ),
                    yaxis=dict(
                        color="#d0d0d8", autorange="reversed", tickfont={"size": 10}
                    ),
                    height=max(300, len(df_drv_c) * 26 + 60),
                    margin=dict(t=10, b=30, l=50, r=70),
                    font={"color": "#d0d0d8"},
                )

                st.plotly_chart(
                    fig_dc,
                    use_container_width=True,
                    key=f"prev_champ_drv_chart_{champ_year}",
                )

                st.dataframe(
                    df_drv_c,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Pos": st.column_config.NumberColumn(
                            "#", format="%d", width="small"
                        ),
                        "Points": st.column_config.NumberColumn(
                            "Points", format="%.0f", width="small"
                        ),
                        "Wins": st.column_config.NumberColumn(
                            "Wins", format="%d", width="small"
                        ),
                    },
                )

            else:
                st.warning(f"No driver standings found for {champ_year}.")

        with ct_con:
            if con_champ:
                df_con_c = pd.DataFrame(
                    [
                        {
                            "Pos": c["pos"],
                            "Team": c["team"],
                            "Points": c["points"],
                            "Wins": c["wins"],
                        }
                        for c in con_champ
                    ]
                )

                fig_cc = go.Figure(
                    go.Bar(
                        x=df_con_c["Points"],
                        y=df_con_c["Team"],
                        orientation="h",
                        marker_color=[
                            "#f5c518" if i == 0 else "#e10600"
                            for i in range(len(df_con_c))
                        ],
                        text=df_con_c["Points"].apply(lambda p: f"{p:.0f}"),
                        textposition="outside",
                        textfont={"color": "#d0d0d8"},
                        hovertemplate="%{y}<br>%{x:.0f} pts<extra></extra>",
                    )
                )

                fig_cc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        title="Points", color="#888", gridcolor="rgba(255,255,255,0.1)"
                    ),
                    yaxis=dict(color="#d0d0d8", autorange="reversed"),
                    height=max(200, len(df_con_c) * 36 + 50),
                    margin=dict(t=10, b=30, l=160, r=70),
                    font={"color": "#d0d0d8"},
                )

                st.plotly_chart(
                    fig_cc,
                    use_container_width=True,
                    key=f"prev_champ_con_chart_{champ_year}",
                )

                st.dataframe(
                    df_con_c,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Pos": st.column_config.NumberColumn(
                            "#", format="%d", width="small"
                        ),
                        "Points": st.column_config.NumberColumn(
                            "Points", format="%.0f", width="small"
                        ),
                        "Wins": st.column_config.NumberColumn(
                            "Wins", format="%d", width="small"
                        ),
                    },
                )

            else:
                st.warning(f"No constructor standings found for {champ_year}.")

    else:
        st.info("Select a season above and click **Load Standings** to view.")


# ----------------------------------------------------------------------------

# TAB 6 — LIVE TRACK MAP

# ----------------------------------------------------------------------------

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
                    race_date = datetime.datetime.strptime(
                        next_race["date"], "%Y-%m-%d"
                    ).replace(tzinfo=datetime.timezone.utc)

                    delta = race_date - now_utc

                    days = delta.days

                    if days > 0:
                        st.info(
                            f"⚡ Next Race: **{next_race['race_name']}** ({next_race['country']}) in **{days} days**."
                        )

                    else:
                        st.info(
                            f"⚡ Next Race: **{next_race['race_name']}** is happening soon!"
                        )

                except Exception:
                    st.info(
                        f"⚡ Next Race: **{next_race['race_name']}** on {next_race['date']}"
                    )

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

                fig_map = go.Figure(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers+text",
                        marker=dict(
                            size=14, color=colors, line=dict(width=2, color="white")
                        ),
                        text=texts,
                        textposition="top center",
                        textfont=dict(color="white", size=10),
                    )
                )

                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    height=600,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                )

                fig_map.update_yaxes(scaleanchor="x", scaleratio=1)

                st.plotly_chart(fig_map, use_container_width=True)


# ----------------------------------------------------------------------------

st.markdown(
    """
<hr style="margin:32px 0 12px;">
<p style="text-align:center; color:#555; font-size:0.8rem;">
F1 Pit Strategy AI · Built with FastF1, LightGBM &amp; Streamlit
&nbsp;·&nbsp; Data: 2021–2025 seasons
</p>
""",
    unsafe_allow_html=True,
)
