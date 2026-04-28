from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    # Current ELO stored after the most recent match processed
    current_elo = Column(Float, default=1500.0)

    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    season = Column(String, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(String)  # H, D, A

    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    features = relationship("Features", back_populates="match", uselist=False)


class Features(Base):
    """Pre-computed feature vector for each historical match."""
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True, nullable=False)

    home_elo = Column(Float)
    away_elo = Column(Float)
    elo_diff = Column(Float)

    home_form_pts = Column(Float)   # avg points per game, last 5 matches
    away_form_pts = Column(Float)
    home_form_gf = Column(Float)    # avg goals scored, last 5 matches
    away_form_gf = Column(Float)
    home_form_ga = Column(Float)    # avg goals conceded, last 5 matches
    away_form_ga = Column(Float)

    h2h_home_win_rate = Column(Float)   # fraction of last 5 H2H won by home
    h2h_draw_rate = Column(Float)
    h2h_away_win_rate = Column(Float)

    match = relationship("Match", back_populates="features")


class TeamCurrentForm(Base):
    """Latest form snapshot per team — used for predicting future fixtures."""
    __tablename__ = "team_current_form"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), unique=True, nullable=False)
    form_pts = Column(Float, default=1.0)
    form_gf = Column(Float, default=1.2)
    form_ga = Column(Float, default=1.2)

    team = relationship("Team")


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    trained_at = Column(String)
    accuracy = Column(Float)
    log_loss = Column(Float)
    algorithm = Column(String)
    feature_names = Column(Text)  # JSON-encoded list


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()
