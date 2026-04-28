# Premier League Match Outcome Predictor

A machine learning web application that predicts the outcome of Premier League
fixtures (home win / draw / away win) and explains which factors drove each
prediction. Built as a BSc Computer Science Final Year Project.

---

## What the software does

Given any two Premier League teams, the system returns probabilistic outcome
predictions (home win, draw, away win) backed by a machine learning model trained on
historical match data. It also shows the top contributing factors for each
prediction using SHAP values, and displays recent form and season statistics
for both teams.

---

## Core features implemented

- **Fixture selection** - dropdown selectors for home and away team
- **Probabilistic predictions** - three outcome probabilities (sum to 100%)
- **SHAP explanation panel** - top factors driving the prediction in plain English
- **Team statistics** - last 5 results (W/D/L badges), goals scored/conceded, ELO rating
- **Team analysis screen** - detailed recent form table and season statistics
- **Three models compared** - Logistic Regression, Random Forest, XGBoost; best selected by log loss
- **Temporal train/test split** - last 2 seasons held out; no data leakage
- **REST API** - four endpoints (predict, teams, team stats, health)
- **Reproducible training** - fixed seeds; single command retrains and saves versioned artefact

---

## Setup and run instructions (local, no Docker)

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

> **Note:** On macOS, use `python3` in place of `python` for all commands below.

### Step 1 - Clone the repository
```bash
git clone https://github.com/mohhaych/pl-predictor.git
cd pl-predictor
```

### Step 2 - Install backend dependencies
```bash
cd backend
pip3 install -r requirements.txt
cd ..
```

### Step 3 - Train the model and populate the database
```bash
cd backend
python3 pipeline/train.py
```

Sample data (`data/sample/pl_data.csv`) is already included in the repository -
no separate download step is needed. This command will:
- Load 1,140 matches across 3 synthetic seasons
- Compute ELO ratings, rolling form, and head-to-head features
- Train and compare Logistic Regression, Random Forest, and XGBoost
- Save the best model to `backend/models/model_latest.pkl`
- Populate the SQLite database (`backend/pl_predictor.db`)

Expected output:
```
── Premier League Outcome Predictor: Training Pipeline ──

1. Loading data...
   1140 matches loaded across 3 seasons.

2. Engineering features...

3. Training and comparing models...
   Logistic Regression     accuracy=0.4632  log_loss=1.0811
   Random Forest           accuracy=0.4395  log_loss=1.0843
   XGBoost                 accuracy=0.4132  log_loss=1.3290

   Best model: Logistic Regression (log_loss=1.0811)
```

> **Why Logistic Regression wins here:** The synthetic sample data only has
> 3 seasons; 2 are held out for testing, leaving 1 season (~380 matches) to
> train on. With this little data, simpler models generalise better. This result
> holds even with 10 seasons of real data - see Known Limitations for a
> full explanation.

### Step 4 - Start the backend API
```bash
# still inside backend/
python3 app.py
```
The API is now running at `http://127.0.0.1:5001`. Leave this terminal open.

### Step 5 - Start the frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

> **macOS note:** The Vite dev server proxies API requests to `127.0.0.1:5001`
> (not `localhost:5001`) to avoid an IPv6 routing issue on macOS. This is
> already configured correctly in `frontend/vite.config.js`.

---

## Optional: use real Premier League data (better accuracy)

The sample data is synthetic. For more realistic model performance, download
10 seasons of real data from football-data.co.uk (free, no API key needed):

```bash
# from the project root
python3 data/download.py
```

Then retrain:
```bash
cd backend
python3 pipeline/train.py
```

With real data, the training pipeline uses 8 seasons (~3,040 matches) and
the expected output is:

```
   Logistic Regression     accuracy=0.4633  log_loss=1.0582
   Random Forest           accuracy=0.4534  log_loss=1.0635
   XGBoost                 accuracy=0.4289  log_loss=1.1061

   Best model: Logistic Regression (log_loss=1.0582)
```

The model accuracy is consistent with published football prediction research -
Premier League outcomes are inherently difficult to predict and ~46–48% accuracy
is typical (Constantinou & Fenton, 2012). The system's value lies in the
probabilistic output and SHAP explanations, not raw accuracy alone.

---

## Running with Docker (alternative)

Requires Docker Desktop to be running.

```bash
# Build and start all services
docker-compose up --build
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:5001`

The Docker setup uses PostgreSQL and automatically runs the training pipeline
on startup. Sample data is mounted from `./data`.

---

## Running the tests
```bash
cd backend
python3 -m pytest tests/ -v
```

This runs:
- `tests/test_features.py` - 8 unit tests for ELO, rolling form, and H2H feature engineering
- `tests/test_api.py` - 7 integration tests covering all four API endpoints

---

## Sample inputs for testing

Any combination of the 20 teams in the 2023-24 Premier League season:

Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Chelsea,
Crystal Palace, Everton, Fulham, Liverpool, Luton, Manchester City,
Manchester United, Newcastle, Nottm Forest, Sheffield United, Tottenham,
West Ham, Wolves

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check; returns model version and accuracy |
| GET | `/teams` | List of all available teams with ELO ratings |
| GET | `/team/<name>/stats` | Recent form and season statistics for a team |
| POST | `/predict` | Predict match outcome |

**POST /predict - example request:**
```json
{ "home_team": "Arsenal", "away_team": "Chelsea" }
```

**POST /predict - example response:**
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
  "home_stats": { "form": ["W","W","D","W","L"], "goals_scored": 9 },
  "away_stats": {}
}
```

---

## Known limitations

- **Sample data is synthetic.** The `generate_sample.py` script produces
  statistically plausible but not historically accurate results. For realistic
  model accuracy, run `python3 data/download.py` to use real Premier League data.
- **Model accuracy is close to the naive baseline.** All three models achieve
  approximately 46–47% accuracy, near the naive "always predict home win"
  baseline (~48%). This is consistent with published Premier League prediction
  research (Constantinou & Fenton, 2012) - football results are inherently
  noisy and difficult to predict from historical statistics alone. The system
  still provides value through probabilistic outputs and SHAP explanations
  that go beyond a single majority-class guess.
- **No real-time data.** The system uses historical data only; it does not
  fetch live fixtures or current-season results.
- **No user accounts.** The system is read-only; there is no authentication
  and no personal data is stored.
- **Single-user load only.** The Flask development server is not intended
  for production; use Gunicorn behind a reverse proxy for multi-user deployment.
- **Draw prediction is hard.** As the literature notes (Constantinou & Fenton,
  2012), draws are the most difficult class to predict - recall for draws is
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
    ├── sample/pl_data.csv      # 3 seasons of synthetic PL data (included)
    ├── generate_sample.py      # Regenerates synthetic data if needed
    └── download.py             # Downloads real data from football-data.co.uk
```

---

## Design decisions

- **Three models compared** with the best selected automatically by log loss.
  Logistic Regression consistently wins on this feature set, which is consistent
  with the literature on tabular sports data (Tax & Joustra, 2015). XGBoost and
  Random Forest are included for comparison and would benefit from additional
  feature engineering or hyperparameter tuning.
- **Temporal train/test split** (last 2 seasons held out): prevents data
  leakage from future matches (Constantinou & Fenton, 2012).
- **SHAP values** for explainability: instance-level contributions rather
  than global importances, so each prediction has its own explanation.
- **SQLite by default**: zero-config; PostgreSQL available via Docker for
  production-like setups.
- **ELO ratings** as a feature: well-supported by the literature
  (Hvattum & Arntzen, 2010) and outperforms raw league position.
