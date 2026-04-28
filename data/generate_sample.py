"""
Generates realistic synthetic Premier League match data in football-data.co.uk
CSV format and writes it to data/sample/pl_data.csv.

Produces 3 full seasons (1,140 matches), enough for meaningful ELO ratings,
rolling form features, and a proper temporal train/test split.

Usage:
    python data/generate_sample.py
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Liverpool", "Luton", "Manchester City", "Manchester United",
    "Newcastle", "Nottm Forest", "Sheffield United", "Tottenham",
    "West Ham", "Wolves",
]

# Rough relative team strengths (higher = stronger)
STRENGTH = {
    "Manchester City": 95, "Arsenal": 88, "Liverpool": 87, "Chelsea": 82,
    "Newcastle": 78, "Manchester United": 78, "Tottenham": 76, "Aston Villa": 75,
    "Brighton": 73, "West Ham": 70, "Crystal Palace": 67, "Brentford": 66,
    "Fulham": 65, "Wolves": 64, "Bournemouth": 63, "Nottm Forest": 62,
    "Everton": 61, "Luton": 58, "Burnley": 57, "Sheffield United": 55,
}

SEASONS = [
    ("2021-22", date(2021, 8, 13), date(2022, 5, 22)),
    ("2022-23", date(2022, 8, 5), date(2023, 5, 28)),
    ("2023-24", date(2023, 8, 11), date(2024, 5, 19)),
]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_result(home: str, away: str) -> tuple[int, int]:
    """Generate goals for a match using team strengths and home advantage."""
    home_str = STRENGTH.get(home, 65) + 5  # +5 home advantage
    away_str = STRENGTH.get(away, 65)
    diff = (home_str - away_str) / 100.0

    # Expected goals using Poisson distribution
    home_xg = max(0.5, 1.4 + diff * 2.5)
    away_xg = max(0.5, 1.1 - diff * 2.0)

    hg = np.random.poisson(home_xg)
    ag = np.random.poisson(away_xg)
    return int(hg), int(ag)


def generate_season_fixtures(season_name: str, start: date, end: date) -> list[dict]:
    """Generate a round-robin fixture list for all 20 teams across a season."""
    from itertools import permutations
    fixtures = list(permutations(TEAMS, 2))  # 380 unique home/away pairings
    random.shuffle(fixtures)

    # Spread matches across the season (roughly 10 matchdays per month)
    total_days = (end - start).days
    rows = []
    for i, (home, away) in enumerate(fixtures):
        day_offset = int(i * total_days / len(fixtures))
        match_date = start + timedelta(days=day_offset)
        # Cluster to weekends
        weekday = match_date.weekday()
        if weekday < 5:
            match_date += timedelta(days=(5 - weekday))

        hg, ag = generate_result(home, away)
        result = "H" if hg > ag else ("D" if hg == ag else "A")
        rows.append({
            "Div": "E0",
            "Date": match_date.strftime("%d/%m/%Y"),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": result,
        })

    return sorted(rows, key=lambda r: r["Date"])


def main():
    all_rows = []
    for season_name, start, end in SEASONS:
        rows = generate_season_fixtures(season_name, start, end)
        all_rows.extend(rows)
        print(f"  {season_name}: {len(rows)} matches generated")

    df = pd.DataFrame(all_rows)

    out_dir = os.path.join(os.path.dirname(__file__), "sample")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pl_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} total matches to {out_path}")

    # Quick sanity check
    results = df["FTR"].value_counts(normalize=True)
    print(f"\nResult distribution:")
    print(f"  Home wins: {results.get('H', 0):.1%}")
    print(f"  Draws:     {results.get('D', 0):.1%}")
    print(f"  Away wins: {results.get('A', 0):.1%}")


if __name__ == "__main__":
    main()
