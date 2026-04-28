"""
Model training pipeline.

Trains three classifiers (Logistic Regression, Random Forest, XGBoost) on
historical Premier League data using a temporally ordered train/test split
to prevent data leakage. The best model by log loss is saved as the active
artefact alongside a JSON metrics file.

Usage:
    python backend/pipeline/train.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_DIR, DATA_DIR, RANDOM_SEED, ELO_K, ELO_INITIAL, FORM_WINDOW
from pipeline.ingest import load_data
from pipeline.features import compute_all_features, compute_current_team_state, FEATURE_COLUMNS
from database import init_db, get_session, Team, Match, Features, TeamCurrentForm, ModelMetadata

# Encode result labels: H=0, D=1, A=2
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
LABEL_NAMES = ["Home Win", "Draw", "Away Win"]


def run():
    print("── Premier League Outcome Predictor: Training Pipeline ──\n")

    # 1. Load and prepare data
    print("1. Loading data...")
    matches = load_data(DATA_DIR)
    print(f"   {len(matches)} matches loaded across {matches['season'].nunique()} seasons.")
    print(f"   Teams: {sorted(matches['home_team'].unique())}\n")

    # 2. Compute features (chronological, no leakage)
    print("2. Engineering features...")
    features = compute_all_features(matches, k=ELO_K, initial_elo=ELO_INITIAL, form_window=FORM_WINDOW)
    X = features[FEATURE_COLUMNS].values
    y = matches["result"].map(LABEL_MAP).values

    # 3. Temporal train/test split — last 2 seasons held out as test set
    seasons = matches["season"].unique()
    seasons_sorted = sorted(seasons)
    test_seasons = set(seasons_sorted[-2:])
    train_mask = ~matches["season"].isin(test_seasons)
    test_mask = matches["season"].isin(test_seasons)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"   Train: {X_train.shape[0]} matches ({', '.join(sorted(set(seasons_sorted[:-2])))})")
    print(f"   Test:  {X_test.shape[0]} matches ({', '.join(sorted(test_seasons))})\n")

    # 4. Train and compare models
    print("3. Training and comparing models...")
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, multi_class="multinomial")),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                           random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_SEED, eval_metric="mlogloss",
                                  use_label_encoder=False, verbosity=0)),
        ]),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        results[name] = {"pipeline": pipeline, "accuracy": acc, "log_loss": ll}
        print(f"   {name:22s}  accuracy={acc:.4f}  log_loss={ll:.4f}")

    # 5. Select best model (lowest log loss)
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    best = results[best_name]
    print(f"\n   Best model: {best_name} (log_loss={best['log_loss']:.4f})\n")

    # Classification report for best model
    best_preds = best["pipeline"].predict(X_test)
    print("4. Classification report (test set):")
    print(classification_report(y_test, best_preds, target_names=LABEL_NAMES))

    # Naive baseline comparison
    home_win_rate = np.mean(y_test == 0)
    print(f"   Naive baseline (always predict Home Win): accuracy={home_win_rate:.4f}\n")

    # 6. Persist model artefact
    os.makedirs(MODEL_DIR, exist_ok=True)
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(MODEL_DIR, "model_latest.pkl")
    joblib.dump(best["pipeline"], model_path)

    metadata = {
        "version": version,
        "algorithm": best_name,
        "trained_at": datetime.now().isoformat(),
        "accuracy": round(best["accuracy"], 6),
        "log_loss": round(best["log_loss"], 6),
        "naive_baseline_accuracy": round(home_win_rate, 6),
        "feature_names": FEATURE_COLUMNS,
        "train_seasons": sorted(set(seasons_sorted[:-2])),
        "test_seasons": sorted(test_seasons),
        "random_seed": RANDOM_SEED,
        "all_model_results": {
            n: {"accuracy": round(r["accuracy"], 6), "log_loss": round(r["log_loss"], 6)}
            for n, r in results.items()
        },
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"5. Artefacts saved:")
    print(f"   Model:    {model_path}")
    print(f"   Metadata: {meta_path}\n")

    # 7. Populate database
    print("6. Populating database...")
    _populate_db(matches, features, metadata)
    print("   Database populated.\n")

    # 8. Save current team state for future predictions
    print("7. Computing current team state for inference...")
    team_state = compute_current_team_state(matches, k=ELO_K, initial_elo=ELO_INITIAL, form_window=FORM_WINDOW)
    _save_team_state(team_state)
    print(f"   Saved state for {len(team_state)} teams.\n")

    print("── Training complete ──")


def _populate_db(matches: pd.DataFrame, features: pd.DataFrame, metadata: dict):
    init_db()
    session = get_session()
    try:
        # Clear existing data
        session.query(Features).delete()
        session.query(Match).delete()
        session.query(Team).delete()
        session.query(ModelMetadata).delete()
        session.commit()

        # Insert teams
        team_names = sorted(set(matches["home_team"].unique()) | set(matches["away_team"].unique()))
        team_map = {}
        for name in team_names:
            t = Team(name=name)
            session.add(t)
            session.flush()
            team_map[name] = t.id

        # Insert matches + features
        for idx, (_, row) in enumerate(matches.iterrows()):
            m = Match(
                date=row["date"].date(),
                season=row["season"],
                home_team_id=team_map[row["home_team"]],
                away_team_id=team_map[row["away_team"]],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                result=row["result"],
            )
            session.add(m)
            session.flush()

            feat_row = features.iloc[idx]
            f = Features(
                match_id=m.id,
                home_elo=float(feat_row["home_elo"]),
                away_elo=float(feat_row["away_elo"]),
                elo_diff=float(feat_row["elo_diff"]),
                home_form_pts=float(feat_row["home_form_pts"]),
                away_form_pts=float(feat_row["away_form_pts"]),
                home_form_gf=float(feat_row["home_form_gf"]),
                away_form_gf=float(feat_row["away_form_gf"]),
                home_form_ga=float(feat_row["home_form_ga"]),
                away_form_ga=float(feat_row["away_form_ga"]),
                h2h_home_win_rate=float(feat_row["h2h_home_win_rate"]),
                h2h_draw_rate=float(feat_row["h2h_draw_rate"]),
                h2h_away_win_rate=float(feat_row["h2h_away_win_rate"]),
            )
            session.add(f)

        # Insert model metadata
        meta_record = ModelMetadata(
            version=metadata["version"],
            trained_at=metadata["trained_at"],
            accuracy=metadata["accuracy"],
            log_loss=metadata["log_loss"],
            algorithm=metadata["algorithm"],
            feature_names=json.dumps(FEATURE_COLUMNS),
        )
        session.add(meta_record)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def _save_team_state(team_state: dict):
    session = get_session()
    try:
        session.query(TeamCurrentForm).delete()
        session.commit()

        # Update team ELO values
        for team_name, state in team_state.items():
            team = session.query(Team).filter_by(name=team_name).first()
            if team:
                team.current_elo = state["elo"]
                form = TeamCurrentForm(
                    team_id=team.id,
                    form_pts=state["pts"],
                    form_gf=state["gf"],
                    form_ga=state["ga"],
                )
                session.add(form)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    run()
