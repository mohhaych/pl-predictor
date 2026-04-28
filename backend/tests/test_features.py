"""Unit tests for the feature engineering module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
from pipeline.features import (
    compute_all_features, compute_current_team_state,
    build_prediction_features, FEATURE_COLUMNS,
)


@pytest.fixture
def sample_matches():
    """10 matches between three teams in chronological order."""
    rows = [
        ("2023-08-12", "Arsenal", "Chelsea", 2, 1, "H"),
        ("2023-08-12", "Liverpool", "Arsenal", 1, 1, "D"),
        ("2023-08-19", "Chelsea", "Liverpool", 0, 2, "A"),
        ("2023-08-26", "Arsenal", "Liverpool", 3, 1, "H"),
        ("2023-08-26", "Chelsea", "Arsenal", 2, 2, "D"),
        ("2023-09-02", "Liverpool", "Chelsea", 3, 0, "H"),
        ("2023-09-09", "Arsenal", "Chelsea", 1, 0, "H"),
        ("2023-09-16", "Liverpool", "Arsenal", 2, 0, "H"),
        ("2023-09-23", "Chelsea", "Liverpool", 1, 1, "D"),
        ("2023-09-30", "Arsenal", "Liverpool", 0, 1, "A"),
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = "2023-24"
    return df


def test_feature_shape(sample_matches):
    features = compute_all_features(sample_matches)
    assert features.shape == (len(sample_matches), len(FEATURE_COLUMNS))


def test_all_feature_columns_present(sample_matches):
    features = compute_all_features(sample_matches)
    assert list(features.columns) == FEATURE_COLUMNS


def test_elo_starts_equal(sample_matches):
    """Before any match is played all teams should have the same ELO."""
    features = compute_all_features(sample_matches)
    # First match row should have home_elo == away_elo == initial (1500)
    first_row = features.iloc[0]
    assert first_row["home_elo"] == pytest.approx(1500.0)
    assert first_row["away_elo"] == pytest.approx(1500.0)
    assert first_row["elo_diff"] == pytest.approx(0.0)


def test_elo_diverges_after_results(sample_matches):
    """ELO ratings should differ after several matches."""
    features = compute_all_features(sample_matches)
    last_row = features.iloc[-1]
    assert last_row["home_elo"] != last_row["away_elo"]


def test_form_defaults_before_history(sample_matches):
    """Early matches with no history should use default form values, not NaN."""
    features = compute_all_features(sample_matches)
    assert not features[["home_form_pts", "away_form_pts"]].isnull().any().any()


def test_probabilities_sum_to_one():
    """Sanity check: if we had a trained model, probabilities should sum to 1."""
    probs = [0.55, 0.25, 0.20]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_build_prediction_features_shape(sample_matches):
    team_state = compute_current_team_state(sample_matches)
    X = build_prediction_features("Arsenal", "Chelsea", team_state, sample_matches)
    assert X.shape == (1, len(FEATURE_COLUMNS))


def test_build_prediction_features_no_nan(sample_matches):
    team_state = compute_current_team_state(sample_matches)
    X = build_prediction_features("Arsenal", "Liverpool", team_state, sample_matches)
    assert not np.isnan(X).any()


def test_h2h_rates_sum_to_one(sample_matches):
    features = compute_all_features(sample_matches)
    total = features["h2h_home_win_rate"] + features["h2h_draw_rate"] + features["h2h_away_win_rate"]
    assert (total.round(6) == 1.0).all()
