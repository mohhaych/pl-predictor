"""
Inference helper. Loads the trained model pipeline and computes predictions
with SHAP-based explanations for a given fixture.
"""
import os
import json
import joblib
import numpy as np

import shap

from config import MODEL_DIR
from pipeline.features import FEATURE_COLUMNS, build_prediction_features

# Plain-English labels shown to users in the explanation panel
FEATURE_LABELS = {
    "home_elo": "Home team strength (ELO)",
    "away_elo": "Away team strength (ELO)",
    "elo_diff": "Team strength difference",
    "home_form_pts": "Home recent form (points)",
    "away_form_pts": "Away recent form (points)",
    "home_form_gf": "Home goals scored (recent)",
    "away_form_gf": "Away goals scored (recent)",
    "home_form_ga": "Home goals conceded (recent)",
    "away_form_ga": "Away goals conceded (recent)",
    "h2h_home_win_rate": "Head-to-head home win rate",
    "h2h_draw_rate": "Head-to-head draw rate",
    "h2h_away_win_rate": "Head-to-head away win rate",
}

OUTCOME_LABELS = ["Home Win", "Draw", "Away Win"]

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        model_path = os.path.join(MODEL_DIR, "model_latest.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "No trained model found. Run `python backend/pipeline/train.py` first."
            )
        _model_cache = joblib.load(model_path)
    return _model_cache


def predict(home_team: str, away_team: str, team_state: dict, matches) -> dict:
    """
    Run inference for a fixture and return probabilities + SHAP explanation.

    Returns
    -------
    dict with keys:
        probabilities: {home_win, draw, away_win}
        predicted_outcome: str
        explanation: list of {feature, label, value, shap_value, direction}
        feature_values: dict of raw feature values
    """
    model = load_model()
    X = build_prediction_features(home_team, away_team, team_state, matches)

    # Probabilities
    probs = model.predict_proba(X)[0]
    predicted_idx = int(np.argmax(probs))

    # SHAP values — use TreeExplainer for XGBoost, otherwise KernelExplainer approximation
    explanation = _compute_shap(model, X)

    feature_values = {FEATURE_COLUMNS[i]: round(float(X[0, i]), 3) for i in range(len(FEATURE_COLUMNS))}

    return {
        "probabilities": {
            "home_win": round(float(probs[0]), 4),
            "draw": round(float(probs[1]), 4),
            "away_win": round(float(probs[2]), 4),
        },
        "predicted_outcome": OUTCOME_LABELS[predicted_idx],
        "explanation": explanation,
        "feature_values": feature_values,
    }


def _compute_shap(model, X: np.ndarray) -> list:
    """
    Compute SHAP values for the predicted class using the right explainer for
    each model type (TreeExplainer for XGBoost/RF, LinearExplainer for LR).
    Returns top features sorted by absolute contribution to the predicted class.
    """
    clf = model.named_steps.get("clf", model)
    scaler = model.named_steps.get("scaler", None)
    X_transformed = scaler.transform(X) if scaler is not None else X
    predicted_class = int(np.argmax(model.predict_proba(X)[0]))

    try:
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.ensemble import RandomForestClassifier as _RF
        from xgboost import XGBClassifier as _XGB

        if isinstance(clf, (_XGB, _RF)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
            # RandomForest returns list[n_classes]; XGBoost returns 3D array
            if isinstance(shap_values, list):
                sv = shap_values[predicted_class][0]
            else:
                sv = shap_values[0, :, predicted_class]

        elif isinstance(clf, _LR):
            # LinearExplainer with multinomial LR returns (n_samples, n_features, n_classes)
            background = np.zeros((1, X_transformed.shape[1]))
            explainer = shap.LinearExplainer(clf, background)
            shap_values = np.array(explainer.shap_values(X_transformed))
            if shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) → take slice for predicted class
                sv = shap_values[0, :, predicted_class]
            elif shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = shap_values.flatten()
        else:
            sv = np.zeros(len(FEATURE_COLUMNS))

    except Exception:
        sv = np.zeros(len(FEATURE_COLUMNS))

    explanation = []
    for i, col in enumerate(FEATURE_COLUMNS):
        explanation.append({
            "feature": col,
            "label": FEATURE_LABELS[col],
            "value": round(float(X[0, i]), 3),
            "shap_value": round(float(sv[i]), 4),
            "direction": "positive" if sv[i] > 0 else "negative",
        })

    # Sort by absolute SHAP value descending
    explanation.sort(key=lambda e: abs(e["shap_value"]), reverse=True)
    return explanation[:5]  # return top 5 for display
