"""
Feature engineering for Premier League match outcome prediction.

Three groups of features are computed in chronological order to prevent
data leakage: ELO-style team ratings, rolling form, and head-to-head record.
All features for match i are derived exclusively from matches before match i.
"""
import pandas as pd
import numpy as np
from collections import defaultdict

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form_pts", "away_form_pts",
    "home_form_gf", "away_form_gf",
    "home_form_ga", "away_form_ga",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate",
]


def compute_all_features(matches: pd.DataFrame, k: int = 20, initial_elo: float = 1500.0,
                         form_window: int = 5, h2h_window: int = 5) -> pd.DataFrame:
    """
    Compute the full feature matrix for a chronologically sorted match DataFrame.

    Parameters
    ----------
    matches : DataFrame with columns [date, home_team, away_team, home_goals, away_goals, result]
              Must be sorted by date ascending before calling.
    k        : ELO K-factor (controls how quickly ratings update)
    initial_elo : Starting ELO for all teams
    form_window : Number of recent matches to use for rolling form
    h2h_window  : Number of recent H2H meetings to use

    Returns
    -------
    DataFrame with FEATURE_COLUMNS, aligned to the input index.
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    elo_rows = _compute_elo(matches, k, initial_elo)
    form_rows = _compute_rolling_form(matches, form_window)
    h2h_rows = _compute_h2h(matches, h2h_window)

    features = pd.concat([elo_rows, form_rows, h2h_rows], axis=1)
    features["elo_diff"] = features["home_elo"] - features["away_elo"]
    return features[FEATURE_COLUMNS]


def compute_current_team_state(matches: pd.DataFrame, k: int = 20, initial_elo: float = 1500.0,
                                form_window: int = 5) -> dict:
    """
    Return the latest ELO and form for every team after processing all historical matches.
    Used when predicting a future fixture.
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    elo = defaultdict(lambda: initial_elo)
    team_history: dict[str, list] = defaultdict(list)

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        result = row["result"]

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        act_home = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        act_away = 1.0 - act_home

        elo[home] = elo[home] + k * (act_home - exp_home)
        elo[away] = elo[away] + k * (act_away - exp_away)

        home_pts = 3 if result == "H" else (1 if result == "D" else 0)
        away_pts = 3 if result == "A" else (1 if result == "D" else 0)
        team_history[home].append({"gf": row["home_goals"], "ga": row["away_goals"], "pts": home_pts})
        team_history[away].append({"gf": row["away_goals"], "ga": row["home_goals"], "pts": away_pts})

    team_state = {}
    for team in set(list(elo.keys()) + list(team_history.keys())):
        hist = team_history[team][-form_window:]
        team_state[team] = {
            "elo": round(elo[team], 2),
            **_form_stats(hist),
        }
    return team_state


def build_prediction_features(home_team: str, away_team: str,
                               team_state: dict, matches: pd.DataFrame,
                               h2h_window: int = 5) -> np.ndarray:
    """
    Build a single feature vector for a new fixture using the latest team state.
    Returns a 1D numpy array matching FEATURE_COLUMNS order.
    """
    home_state = team_state.get(home_team, {"elo": 1500, "pts": 1.0, "gf": 1.2, "ga": 1.2})
    away_state = team_state.get(away_team, {"elo": 1500, "pts": 1.0, "gf": 1.2, "ga": 1.2})

    h2h = _latest_h2h(home_team, away_team, matches, h2h_window)

    return np.array([[
        home_state["elo"],
        away_state["elo"],
        home_state["elo"] - away_state["elo"],
        home_state["pts"],
        away_state["pts"],
        home_state["gf"],
        away_state["gf"],
        home_state["ga"],
        away_state["ga"],
        h2h["home_win_rate"],
        h2h["draw_rate"],
        h2h["away_win_rate"],
    ]])


# ── private helpers ──────────────────────────────────────────────────────────

def _compute_elo(matches: pd.DataFrame, k: float, initial_elo: float) -> pd.DataFrame:
    elo = defaultdict(lambda: initial_elo)
    rows = []
    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_elo, away_elo = elo[home], elo[away]
        rows.append({"home_elo": home_elo, "away_elo": away_elo})

        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        act_home = 1.0 if row["result"] == "H" else (0.5 if row["result"] == "D" else 0.0)

        elo[home] += k * (act_home - exp_home)
        elo[away] += k * ((1 - act_home) - (1 - exp_home))

    return pd.DataFrame(rows)


def _compute_rolling_form(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    team_history: dict[str, list] = defaultdict(list)
    rows = []

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]

        home_form = _form_stats(team_history[home][-window:])
        away_form = _form_stats(team_history[away][-window:])

        rows.append({
            "home_form_pts": home_form["pts"],
            "away_form_pts": away_form["pts"],
            "home_form_gf": home_form["gf"],
            "away_form_gf": away_form["gf"],
            "home_form_ga": home_form["ga"],
            "away_form_ga": away_form["ga"],
        })

        home_pts = 3 if row["result"] == "H" else (1 if row["result"] == "D" else 0)
        away_pts = 3 if row["result"] == "A" else (1 if row["result"] == "D" else 0)
        team_history[home].append({"gf": row["home_goals"], "ga": row["away_goals"], "pts": home_pts})
        team_history[away].append({"gf": row["away_goals"], "ga": row["home_goals"], "pts": away_pts})

    return pd.DataFrame(rows)


def _compute_h2h(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    # Store past meetings keyed by frozenset({home, away})
    h2h_history: dict = defaultdict(list)
    rows = []

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        key = (min(home, away), max(home, away))

        past = h2h_history[key][-window:]
        rows.append(_h2h_rates(past, home))

        h2h_history[key].append({"home": home, "away": away, "result": row["result"]})

    return pd.DataFrame(rows)


def _form_stats(history: list) -> dict:
    """Average pts/gf/ga from a list of recent match dicts. Defaults when no history."""
    if not history:
        return {"pts": 1.0, "gf": 1.2, "ga": 1.2}
    pts = np.mean([m["pts"] for m in history])
    gf = np.mean([m["gf"] for m in history])
    ga = np.mean([m["ga"] for m in history])
    return {"pts": round(pts, 3), "gf": round(gf, 3), "ga": round(ga, 3)}


def _h2h_rates(history: list, perspective_home: str) -> dict:
    """Compute H2H win/draw/loss rates from the perspective of perspective_home."""
    if not history:
        return {"h2h_home_win_rate": 0.33, "h2h_draw_rate": 0.33, "h2h_away_win_rate": 0.34}
    wins = draws = away_wins = 0
    for m in history:
        if m["home"] == perspective_home:
            if m["result"] == "H":
                wins += 1
            elif m["result"] == "D":
                draws += 1
            else:
                away_wins += 1
        else:
            # Roles reversed in this historical match
            if m["result"] == "A":
                wins += 1
            elif m["result"] == "D":
                draws += 1
            else:
                away_wins += 1
    n = len(history)
    return {
        "h2h_home_win_rate": wins / n,
        "h2h_draw_rate": draws / n,
        "h2h_away_win_rate": away_wins / n,
    }


def _latest_h2h(home: str, away: str, matches: pd.DataFrame, window: int) -> dict:
    key1 = matches[(matches["home_team"] == home) & (matches["away_team"] == away)]
    key2 = matches[(matches["home_team"] == away) & (matches["away_team"] == home)]
    combined = pd.concat([key1, key2]).sort_values("date").tail(window)

    history = []
    for _, row in combined.iterrows():
        history.append({"home": row["home_team"], "away": row["away_team"], "result": row["result"]})

    rates = _h2h_rates(history, home)
    return {
        "home_win_rate": rates["h2h_home_win_rate"],
        "draw_rate": rates["h2h_draw_rate"],
        "away_win_rate": rates["h2h_away_win_rate"],
    }
