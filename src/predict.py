"""Predict an upcoming fixture: python -m src.predict "Arsenal" "Chelsea" [--demo]"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_matches, synthetic_matches
from .features import FeatureBuilder

ROOT = Path(__file__).resolve().parents[1]


def predict_fixture(home: str, away: str, matches: pd.DataFrame, artifact: dict,
                    date: str | None = None, inj_home: float = 0.0, inj_away: float = 0.0) -> dict:
    known = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    for t in (home, away):
        if t not in known:
            raise ValueError(f"Unknown team '{t}'. Use football-data.co.uk spellings, e.g. 'Man City'.")
    fb = FeatureBuilder()
    for m in matches.itertuples():
        fb.update(m.HomeTeam, m.AwayTeam, pd.Timestamp(m.Date), int(m.FTHG), int(m.FTAG), m.HST, m.AST)
    row = pd.DataFrame([fb.row(home, away, pd.Timestamp(date or pd.Timestamp.today()), inj_home, inj_away)])
    X = row[["elo_diff"]] if artifact["name"] == "elo_logreg" else row[artifact["features"]]
    p = artifact["model"].predict_proba(X)[0]
    return {"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--inj-home", type=float, default=0.0)
    ap.add_argument("--inj-away", type=float, default=0.0)
    a = ap.parse_args()
    art = joblib.load(ROOT / "models" / "best.joblib")
    hist = synthetic_matches() if a.demo else load_matches()
    for k, v in predict_fixture(a.home, a.away, hist, art, inj_home=a.inj_home, inj_away=a.inj_away).items():
        print(f"{k:>9}: {v:6.1%}")
