"""
Downloads historical Premier League CSV files from football-data.co.uk
for the past 10 seasons and saves them to data/raw/.

Usage:
    python data/download.py

No API key required. Data is freely available from football-data.co.uk.
"""
import os
import time
import requests

# Season codes used by football-data.co.uk (most recent first)
SEASONS = [
    ("2324", "2023-24"),
    ("2223", "2022-23"),
    ("2122", "2021-22"),
    ("2021", "2020-21"),
    ("1920", "2019-20"),
    ("1819", "2018-19"),
    ("1718", "2017-18"),
    ("1617", "2016-17"),
    ("1516", "2015-16"),
    ("1415", "2014-15"),
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {len(SEASONS)} seasons to {OUT_DIR}/\n")

    for code, label in SEASONS:
        url = BASE_URL.format(code=code)
        out_path = os.path.join(OUT_DIR, f"E0_{code}.csv")

        if os.path.exists(out_path):
            print(f"  {label}: already downloaded, skipping.")
            continue

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)
            rows = len(response.text.strip().splitlines()) - 1
            print(f"  {label}: {rows} matches saved → {out_path}")
        except requests.RequestException as e:
            print(f"  {label}: FAILED ({e})")

        time.sleep(0.5)  # be polite to the server

    print("\nDownload complete. Run `python backend/pipeline/train.py` next.")


if __name__ == "__main__":
    download()
