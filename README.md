# Premier League Match Outcome Predictor

A machine learning web application that predicts the outcome of Premier League
fixtures (home win / draw / away win) and explains which factors drove each
prediction. Built as a BSc Computer Science Final Year Project.

---

## What the software does

Given any two Premier League teams, the system returns probabilistic outcome
predictions (home win, draw, away win) backed by an XGBoost model trained on
historical match data. It also shows the top contributing factors for each
prediction using SHAP values, and displays recent form and season statistics
for both teams.

---

## Core features implemented

- **Fixture selection** — dropdown selectors for home and away team
- **Probabilistic predictions** — three outcome probabilities (sum to 100%)
- **SHAP explanation panel** — top factors driving the prediction in plain English
- **Team statistics** — last 5 results (W/D/L badges), goals scored/conceded, ELO rating
- **Team analysis screen** — detailed recent form table and season statistics
- **Three models compared** — Logistic Regression, Random Forest, XGBoost; best selected by log loss
- **Temporal train/test split** — last 2 seasons held out; no data leakage
- **REST API** — four endpoints (predict, teams, team stats, health)
- **Reproducible training** — fixed seeds; single command retrains and saves versioned artefact

---

## Setup and run instructions (local, no Docker)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Step 1 — Clone the repository
```bash
git clone <your-repo-url>
cd pl-predictor
```

### Step 2 — Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### Step 3 — Generate sample data
```bash
python data/generate_sample.py
```
This creates `data/sample/pl_data.csv` with 3 seasons of realistic match data
(1,140 matches). Skip this step if you already have real data (see below).

> **Using real data (optional):** Run `python data/download.py` to download 10
> seasons from football-data.co.uk (free, no API key needed). Saves to
> `data/raw/`. This replaces the sample data and gives more accurate model
> performance.

### Step 4 — Train the model
```bash
cd backend
python pipeline/train.py
```
This will:
- Load and process match data
- Compute ELO ratings, rolling form, and head-to-head features
- Train Logistic Regression, Random Forest, and XGBoost
- Print a comparison table and select the best model by log loss
- Save `backend/models/model_latest.pkl` and `backend/models/model_metadata.json`
- Populate the SQLite database (`backend/pl_predictor.db`)

Expected output (sample data):
```
── Premier League Outcome Predictor: Training Pipeline ──

1. Loading data...
   1140 matches loaded across 3 seasons.

2. Engineering features...

3. Training and comparing models...
   Logistic Regression     accuracy=0.5140  log_loss=0.9821
   Random Forest           accuracy=0.5263  log_loss=0.9714
   XGBoost                 accuracy=0.5368  log_loss=0.9588

   Best model: XGBoost (log_loss=0.9588)
```

### Step 5 — Start the backend API
```bash
# still in backend/
python app.py
```
API is now running at `http://localhost:5001`.

### Step 6 — Start the frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Running with Docker (alternative)

Requires Docker Desktop.

```bash
# Generate sample data first (runs on host)
python data/generate_sample.py

# Build and start all services
docker-compose up --build
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:5001`

---

## Running the tests
```bash
cd backend
pytest tests/ -v
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check; returns model version and accuracy |
| GET | `/teams` | List of all available teams with ELO ratings |
| GET | `/team/<name>/stats` | Recent form and season statistics for a team |
| POST | `/predict` | Predict match outcome |

**POST /predict — example request:**
```json
{ "home_team": "Arsenal", "away_team": "Chelsea" }
```

**POST /predict — example response:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "prediction": {
    "probabilities": { "home_win": 0.55, "draw": 0.25, "away_win": 0.20 },
    "predicted_outcome": "Home Win",
    "explanation": [
      { "label": "Team strength difference", "direction": "positive", "shap_value": 0.18 },
      { "label": "Home recent form (points)", "direction": "positive", "shap_value": 0.09 }
    ]
  },
  "home_stats": { "form": ["W","W","D","W","L"], "goals_scored": 9, ... },
  "away_stats": { ... }
}
```

---

## Sample inputs for testing

Any combination of the 20 teams in the 2023-24 Premier League season:

Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Chelsea,
Crystal Palace, Everton, Fulham, Liverpool, Luton, Manchester City,
Manchester United, Newcastle, Nottm Forest, Sheffield United, Tottenham,
West Ham, Wolves

---

## Known limitations

- **Sample data is synthetic.** The generate_sample.py script produces
  statistically plausible but not historically accurate results. For realistic
  model accuracy, run `python data/download.py` to use real Premier League data.
- **No real-time data.** The system uses historical data only; it does not
  fetch live fixtures or current-season results.
- **No user accounts.** The system is read-only; there is no authentication
  and no personal data is stored.
- **Single-user load only.** The Flask development server is not intended
  for production; use Gunicorn behind a reverse proxy for multi-user deployment.
- **Draw prediction is hard.** As the literature notes (Constantinou & Fenton,
  2012), draws are the most difficult class to predict — recall for draws is
  consistently lower than for wins/losses.
- **CSV export (FR5)** is designed in the API but not yet wired to a frontend
  button. The `/predict` endpoint data can be used programmatically to build
  exports.

---

## Project structure

```
pl-predictor/
├── README.md                   # This file (user guide)
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── app.py                  # Flask API (4 endpoints)
│   ├── config.py               # Centralised configuration
│   ├── database.py             # SQLAlchemy models (Team, Match, Features)
│   ├── predict.py              # Inference + SHAP explanation
│   ├── requirements.txt
│   ├── pipeline/
│   │   ├── ingest.py           # Load and clean CSV data
│   │   ├── features.py         # ELO, rolling form, H2H feature engineering
│   │   └── train.py            # Train 3 models, evaluate, save best
│   └── tests/
│       ├── test_features.py    # Unit tests for feature engineering
│       └── test_api.py         # Integration tests for Flask endpoints
├── frontend/
│   └── src/
│       ├── App.jsx             # Top-level routing between screens
│       ├── components/
│       │   ├── FixtureSelector.jsx     # Screen 1: team selection
│       │   ├── PredictionResults.jsx  # Screen 2: prediction + explanation
│       │   └── TeamAnalysis.jsx       # Screen 3: detailed team stats
│       └── api/client.js       # Fetch wrapper for all API calls
└── data/
    ├── generate_sample.py      # Generates synthetic PL data (no internet needed)
    └── download.py             # Downloads real data from football-data.co.uk
```

---

## Design decisions

- **XGBoost** selected as primary model: best log loss on test set; native
  feature importance support; works well on tabular data (Tax & Joustra, 2015).
- **Temporal train/test split** (last 2 seasons held out): prevents data
  leakage from future matches (Constantinou & Fenton, 2012).
- **SHAP values** for explainability: instance-level contributions rather
  than global importances, so each prediction has its own explanation.
- **SQLite by default**: zero-config; PostgreSQL available via Docker for
  production-like setups.
- **ELO ratings** as a feature: well-supported by the literature
  (Hvattum & Arntzen, 2010) and outperforms raw league position.
