"""Integration tests for the Flask API endpoints."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import json


@pytest.fixture
def client():
    """Create a test Flask client with a fresh in-memory SQLite database."""
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    import app as flask_app
    import database as db_module

    # Re-create tables on a fresh in-memory engine each time
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    fresh_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.engine = fresh_engine
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)

    # Also reset the team-state cache in app so it re-queries the fresh DB
    flask_app._team_state_cache = None

    from database import get_session, Team, TeamCurrentForm
    session = get_session()
    t1 = Team(name="Arsenal", current_elo=1520.0)
    t2 = Team(name="Chelsea", current_elo=1480.0)
    session.add_all([t1, t2])
    session.flush()
    session.add(TeamCurrentForm(team_id=t1.id, form_pts=2.0, form_gf=2.0, form_ga=0.8))
    session.add(TeamCurrentForm(team_id=t2.id, form_pts=1.2, form_gf=1.4, form_ga=1.6))
    session.commit()
    session.close()

    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_teams_returns_list(client):
    resp = client.get("/teams")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    names = [t["name"] for t in data]
    assert "Arsenal" in names
    assert "Chelsea" in names


def test_predict_missing_body(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_same_team(client):
    resp = client.post("/predict",
                       data=json.dumps({"home_team": "Arsenal", "away_team": "Arsenal"}),
                       content_type="application/json")
    assert resp.status_code == 400
    assert "different" in resp.get_json()["error"].lower()


def test_predict_unknown_team(client):
    resp = client.post("/predict",
                       data=json.dumps({"home_team": "Arsenal", "away_team": "NonExistentFC"}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_team_stats_not_found(client):
    resp = client.get("/team/NonExistentFC/stats")
    assert resp.status_code == 404


def test_team_stats_found(client):
    resp = client.get("/team/Arsenal/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["team"] == "Arsenal"
    assert "current_elo" in data
    assert "recent_form" in data
    assert "season_stats" in data
