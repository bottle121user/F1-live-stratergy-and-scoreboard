# 🏎️ F1 Pit Strategy AI

An AI-powered F1 pit stop strategy advisor — predicts optimal pit windows, compares race strategies lap-by-lap, and shows a live championship scoreboard.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Features

| Tab | What it does |
|-----|-------------|
| ⚡ Live Pit Decision | Real-time AI pit recommendation (PIT NOW / STAY OUT) with confidence gauge |
| 📊 Strategy Recommender | Top-N ranked pit stop strategies with interactive comparison |
| 📈 Lap Simulator | Lap-by-lap tyre degradation curves and total race duration modeling |
| 🏆 Live Scoreboard | Live championship standings, latest Grand Prix results, and 2026 calendar |
| 📜 Previous Champions | Historical World Drivers' & Constructors' Championship archive |
| 📍 Live Track Map | Real-time car positions and telemetry via OpenF1 API |

---

## 🚀 Deploy to Streamlit Cloud (5 minutes)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "deploy: add streamlit config and slim requirements"
   git push
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**
   - Click **New app**
   - Select your repo
   - Set **Main file path** → `app/app.py`
   - Click **Deploy**

That's it — no secrets or API keys needed.

---

## Local Setup

```powershell
# 1. Clone and enter the project
cd z:\prosss\f1-strategy-ai

# 2. Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Re-training the model (optional — pre-trained model included)

```powershell
# Download 2021-2025 race data (~10 min first run)
python src/data_loader.py

# Train the LightGBM model
python src/model.py
```

### Run the dashboard

```powershell
streamlit run app/app.py
```

---

## Project Structure

```
f1-strategy-ai/
├── .streamlit/
│   └── config.toml           ← Streamlit Cloud theme config
├── app/
│   └── app.py                ← Streamlit dashboard (6 tabs)
├── src/
│   ├── data_loader.py        ← Fetches FastF1 race data (2021-2025)
│   ├── features.py           ← ML feature engineering
│   ├── model.py              ← Trains LightGBM pit predictor
│   ├── simulator.py          ← Lap-by-lap race simulator
│   ├── predict.py            ← Inference + strategy recommender
│   └── live_data.py          ← Live data from Jolpica & OpenF1 APIs
├── models/
│   └── pit_predictor.pkl     ← Pre-trained model (committed, 0.4 MB)
├── data/
│   └── processed/            ← Training CSVs + historical metrics JSON
├── tests/
│   └── test_all.py           ← Full test suite
├── MODEL_REPORT.md           ← Complete accuracy & stress limit testing report
├── requirements.txt
└── .gitignore
```

---

## 📊 Model Performance & Benchmark Results

### Architecture Comparison (Class-Weighted Pit Detection)

| Model | Accuracy | Precision | Recall (Pit Laps) | F1-Score | Status in Production |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM (Prod)** | **99.1%** | **1.000** | **1.000** | **1.000** | **Selected Core Model (`models/pit_predictor.pkl`)** |
| **RandomForest** | 97.7% | 0.625 | 0.833 | 0.714 | Reliable baseline; higher false alarm rate |
| **XGBoost (unweighted)** | 96.5% | 0.000 | 0.000 | 0.000 | Failed: fell into majority-class ("stay out") trap |

### Stress & Limit Testing Summary (5,000 Consecutive Telemetry Runs)

| Stress Benchmark | Value | Production Verdict |
| :--- | :---: | :--- |
| **Average Inference Latency** | **2.31 ms** | Sub-second real-time telemetry processing |
| **Throughput Capacity** | **432.7 predictions / sec** | Easily handles all 20 cars concurrently |
| **Circuit Coverage** | **24 / 24 Circuits** | 100% numerical stability across 72 simulated race distances |
| **Adversarial Edge Cases** | **100% Handled** | Weather mismatches, SC windows, and tyre cliffs properly categorized |

> 📖 **Full Technical Audit:** See [MODEL_REPORT.md](MODEL_REPORT.md) for the complete benchmark methodologies, confusion matrices, and adversarial scenarios.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | FastF1, Jolpica (Ergast) API, OpenF1 API |
| ML | scikit-learn, LightGBM |
| Simulation | Custom lap-time physics model |
| Dashboard | Streamlit, Plotly |
| Live Data | Jolpica API · OpenF1 API |
