"""
Data ingestion: loads Premier League CSV files (football-data.co.uk format),
cleans and normalises them into a standard DataFrame used by the pipeline.
"""
import os
import glob
import pandas as pd

# Column mapping from football-data.co.uk to internal names
FDC_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "Season": "season",
}

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from data_dir (and its sample/ subdirectory).
    Accepts football-data.co.uk format. Returns a cleaned, sorted DataFrame.
    """
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Run `python data/generate_sample.py` to create sample data, "
            "or `python data/download.py` to fetch real data."
        )

    frames = []
    for path in csv_files:
        df = _load_single_file(path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No valid match data found in the CSV files.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"])
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def _load_single_file(path: str) -> pd.DataFrame | None:
    try:
        raw = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            return None

    # Check required columns exist
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        # Try to infer season from file path and skip if columns missing
        print(f"  Warning: {path} missing columns {missing}, skipping.")
        return None

    df = raw[REQUIRED_COLS].copy()
    df = df.dropna(subset=REQUIRED_COLS)

    # Parse dates — football-data.co.uk uses DD/MM/YY or DD/MM/YYYY.
    # We create the parsed column under a temp name to avoid colliding with
    # the "Date" -> "date" rename that comes later via FDC_MAP.
    df["_date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_date_parsed"])

    df["season"] = df["_date_parsed"].apply(_date_to_season)
    df = df.drop(columns=["_date_parsed"])

    df = df.rename(columns=FDC_MAP)
    # Reparse the now-renamed "date" column as datetime
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["home_goals", "away_goals"])

    df["result"] = df["result"].str.upper().str.strip()
    df = df[df["result"].isin(["H", "D", "A"])]

    # Normalise team names
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    return df[["date", "season", "home_team", "away_team", "home_goals", "away_goals", "result"]]


def _date_to_season(d: pd.Timestamp) -> str:
    """Map a date to a football season string, e.g. 2023-01-15 -> '2022-23'."""
    year = d.year
    month = d.month
    if month >= 8:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"
