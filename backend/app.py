"""
Flask REST API — Premier League Match Outcome Predictor

Endpoints:
  GET  /health              — liveness check
  GET  /teams               — list of available teams
  GET  /team/<name>/stats   — season stats + recent form for a team
  POST /predict             — match outcome probabilities + explanation
"""
import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import func, desc

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, MODEL_DIR
from database import init_db, get_session, Team, Match, TeamCurrentForm, ModelMetadata
from pipeline.ingest import load_data
from predict import predict

app = Flask(__name__)
CORS(app)

# Load historical matches into memory once at startup (needed for H2H lookup)
_matches_cache = None
_team_state_cache = None


def get_matches():
    global _matches_cache
    if _matches_cache is None:
        _matches_cache = load_data(DATA_DIR)
    return _matches_cache


def get_team_state():
    """Load current team state (ELO + form) from the database."""
    global _team_state_cache
    if _team_state_cache is not None:
        return _team_state_cache

    session = get_session()
    try:
        teams = session.query(Team).all()
        state = {}
        for team in teams:
            form = session.query(TeamCurrentForm).filter_by(team_id=team.id).first()
            state[team.name] = {
                "elo": team.current_elo or 1500.0,
                "pts": form.form_pts if form else 1.0,
                "gf": form.form_gf if form else 1.2,
                "ga": form.form_ga if form else 1.2,
            }
        _team_state_cache = state
        return state
    finally:
        session.close()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    session = get_session()
    try:
        team_count = session.query(func.count(Team.id)).scalar()
        match_count = session.query(func.count(Match.id)).scalar()
        meta = session.query(ModelMetadata).order_by(desc(ModelMetadata.id)).first()
        return jsonify({
            "status": "ok",
            "teams": team_count,
            "matches": match_count,
            "model_version": meta.version if meta else "none",
            "model_accuracy": meta.accuracy if meta else None,
            "model_log_loss": meta.log_loss if meta else None,
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500
    finally:
        session.close()


@app.get("/teams")
def list_teams():
    session = get_session()
    try:
        teams = session.query(Team).order_by(Team.name).all()
        if not teams:
            return jsonify({"error": "No teams found. Has the model been trained?"}), 404
        return jsonify([{"id": t.id, "name": t.name, "elo": round(t.current_elo or 1500, 1)} for t in teams])
    finally:
        session.close()


@app.get("/team/<path:team_name>/stats")
def team_stats(team_name: str):
    session = get_session()
    try:
        team = session.query(Team).filter_by(name=team_name).first()
        if not team:
            return jsonify({"error": f"Team '{team_name}' not found"}), 404

        # Last 5 matches (home or away)
        recent_matches = (
            session.query(Match)
            .filter((Match.home_team_id == team.id) | (Match.away_team_id == team.id))
            .order_by(desc(Match.date))
            .limit(10)
            .all()
        )

        form = []
        goals_scored = goals_conceded = wins = draws = losses = 0
        for m in recent_matches:
            is_home = m.home_team_id == team.id
            gf = m.home_goals if is_home else m.away_goals
            ga = m.away_goals if is_home else m.home_goals
            if is_home:
                outcome = m.result  # H=win, D=draw, A=loss
            else:
                outcome = {"H": "A", "D": "D", "A": "H"}.get(m.result, "D")

            result_label = "W" if outcome == "H" else ("D" if outcome == "D" else "L")
            form.append({
                "date": m.date.isoformat(),
                "opponent": m.away_team.name if is_home else m.home_team.name,
                "home_or_away": "H" if is_home else "A",
                "goals_for": gf,
                "goals_against": ga,
                "result": result_label,
            })
            goals_scored += gf or 0
            goals_conceded += ga or 0
            wins += 1 if result_label == "W" else 0
            draws += 1 if result_label == "D" else 0
            losses += 1 if result_label == "L" else 0

        n = len(recent_matches)
        form_record = session.query(TeamCurrentForm).filter_by(team_id=team.id).first()

        return jsonify({
            "team": team.name,
            "current_elo": round(team.current_elo or 1500, 1),
            "recent_form": form[:5],
            "season_stats": {
                "matches_played": n,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_scored": goals_scored,
                "goals_conceded": goals_conceded,
                "win_rate": round(wins / n, 2) if n > 0 else 0,
            },
            "form_pts_avg": round(form_record.form_pts, 2) if form_record else None,
        })
    finally:
        session.close()


@app.post("/predict")
def predict_match():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    home_team = data.get("home_team", "").strip()
    away_team = data.get("away_team", "").strip()

    if not home_team or not away_team:
        return jsonify({"error": "home_team and away_team are required"}), 400
    if home_team == away_team:
        return jsonify({"error": "home_team and away_team must be different"}), 400

    session = get_session()
    try:
        if not session.query(Team).filter_by(name=home_team).first():
            return jsonify({"error": f"Unknown team: {home_team}"}), 400
        if not session.query(Team).filter_by(name=away_team).first():
            return jsonify({"error": f"Unknown team: {away_team}"}), 400
    finally:
        session.close()

    try:
        team_state = get_team_state()
        matches = get_matches()
        result = predict(home_team, away_team, team_state, matches)

        # Augment response with team stats
        home_stats = _quick_stats(home_team)
        away_stats = _quick_stats(away_team)

        return jsonify({
            "home_team": home_team,
            "away_team": away_team,
            "prediction": result,
            "home_stats": home_stats,
            "away_stats": away_stats,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": "Internal prediction error", "detail": str(e)}), 500


def _quick_stats(team_name: str) -> dict:
    """Return a compact stats dict for inline display on the results page."""
    session = get_session()
    try:
        team = session.query(Team).filter_by(name=team_name).first()
        if not team:
            return {}
        recent = (
            session.query(Match)
            .filter((Match.home_team_id == team.id) | (Match.away_team_id == team.id))
            .order_by(desc(Match.date))
            .limit(5)
            .all()
        )
        form_str = []
        goals_for = goals_against = 0
        for m in recent:
            is_home = m.home_team_id == team.id
            gf = m.home_goals if is_home else m.away_goals
            ga = m.away_goals if is_home else m.home_goals
            res = m.result if is_home else {"H": "A", "D": "D", "A": "H"}.get(m.result, "D")
            form_str.append("W" if res == "H" else ("D" if res == "D" else "L"))
            goals_for += gf or 0
            goals_against += ga or 0
        n = len(recent)
        return {
            "form": form_str,
            "goals_scored": goals_for,
            "goals_conceded": goals_against,
            "elo": round(team.current_elo or 1500, 1),
            "matches": n,
        }
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001)
