"""Load Premier League results from football-data.co.uk (cached) or a synthetic demo set."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://www.football-data.co.uk/mmz4281/{yy}/E0.csv"
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HST", "AST"]


def load_season(start_year: int) -> pd.DataFrame | None:
    """Return one season (start_year/start_year+1), downloading once and caching."""
    code = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
    path = RAW / f"E0_{code}.csv"
    try:
        if not path.exists():
            RAW.mkdir(parents=True, exist_ok=True)
            pd.read_csv(URL.format(yy=code), encoding="latin-1", on_bad_lines="skip").to_csv(path, index=False)
        df = pd.read_csv(path)
    except Exception as exc:  # network failure or season not published yet
        print(f"skip {start_year}/{start_year + 1}: {exc}")
        return None
    df = df.dropna(subset=["HomeTeam", "FTR"])
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    df = df[COLS].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["Season"] = start_year
    return df


def load_matches(first: int = 2015, last: int = 2026) -> pd.DataFrame:
    frames = [f for y in range(first, last + 1) if (f := load_season(y)) is not None]
    if not frames:
        raise RuntimeError("No data loaded. Check your connection or use --demo.")
    return pd.concat(frames).sort_values("Date").reset_index(drop=True)


def synthetic_matches(seasons: int = 8, seed: int = 7) -> pd.DataFrame:
    """Fake league with latent team strength. For smoke tests and CI only, never for real claims."""
    rng = np.random.default_rng(seed)
    teams = [f"Team {i:02d}" for i in range(20)]
    skill = dict(zip(teams, rng.normal(0, 0.35, 20)))
    rows, start = [], pd.Timestamp("2016-08-13")
    for s in range(seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                gh = rng.poisson(np.exp(0.25 + 0.3 + skill[h] - skill[a] * 0.6))
                ga = rng.poisson(np.exp(0.25 + skill[a] - skill[h] * 0.6))
                date = start + pd.Timedelta(days=365 * s + int(rng.integers(0, 280)))
                rows.append((date, h, a, gh, ga, "H" if gh > ga else "A" if ga > gh else "D",
                             rng.poisson(4), rng.poisson(3), 2016 + s))
    df = pd.DataFrame(rows, columns=COLS + ["Season"])
    return df.sort_values("Date").reset_index(drop=True)
