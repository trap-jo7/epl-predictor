"""Leakage-safe feature engineering. Every feature for a match uses only matches played before it."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

PRIOR_PPG, PRIOR_GOALS, PRIOR_SOT = 1.35, 1.35, 4.5
RESULT_TO_INT = {"H": 0, "D": 1, "A": 2}  # class order used everywhere


class FeatureBuilder:
    """Stateful walk through history: call row() to featurise a match, then update() with its result."""

    def __init__(self, k: float = 20.0, home_adv: float = 60.0):
        self.k, self.home_adv = k, home_adv
        self.elo = defaultdict(lambda: 1500.0)
        self.hist = defaultdict(list)   # team -> [dict(date, gf, ga, pts, sot, home)]
        self.h2h = defaultdict(list)    # frozenset({a, b}) -> [(home_team, points_for_that_home_team)]

    def _form(self, team: str, n: int, venue: bool | None = None) -> dict:
        games = [g for g in self.hist[team] if venue is None or g["home"] == venue][-n:]
        if not games:
            return dict(ppg=PRIOR_PPG, gf=PRIOR_GOALS, ga=PRIOR_GOALS, sot=PRIOR_SOT)
        m = len(games)
        return {k: sum(g[s] for g in games) / m for k, s in
                (("ppg", "pts"), ("gf", "gf"), ("ga", "ga"), ("sot", "sot"))}

    def _rest(self, team: str, date: pd.Timestamp) -> float:
        g = self.hist[team]
        return float(min((date - g[-1]["date"]).days, 14)) if g else 7.0

    def _h2h_ppg(self, home: str, away: str) -> float:
        """Points per game for `home` over the last 5 meetings, whichever ground they were played on."""
        pts = []
        for host, p in self.h2h[frozenset((home, away))][-5:]:
            pts.append(p if host == home else {3: 0, 0: 3, 1: 1}[p])
        return sum(pts) / len(pts) if pts else PRIOR_PPG

    def row(self, home: str, away: str, date: pd.Timestamp, inj_home: float = 0.0, inj_away: float = 0.0) -> dict:
        fh, fa = self._form(home, 5), self._form(away, 5)
        return {
            "elo_diff": self.elo[home] + self.home_adv - self.elo[away],
            "elo_home": self.elo[home], "elo_away": self.elo[away],
            "h_ppg5": fh["ppg"], "a_ppg5": fa["ppg"],
            "h_gf5": fh["gf"], "h_ga5": fh["ga"], "a_gf5": fa["gf"], "a_ga5": fa["ga"],
            "h_sot5": fh["sot"], "a_sot5": fa["sot"],
            "h_home_ppg": self._form(home, 8, True)["ppg"], "a_away_ppg": self._form(away, 8, False)["ppg"],
            "rest_h": self._rest(home, date), "rest_a": self._rest(away, date),
            "h2h_ppg": self._h2h_ppg(home, away),
            "inj_home": inj_home, "inj_away": inj_away,  # 0-1 share of squad quality unavailable
        }

    def update(self, home: str, away: str, date: pd.Timestamp, hg: int, ag: int, hsot=None, asot=None) -> None:
        exp_h = 1 / (1 + 10 ** (-(self.elo[home] + self.home_adv - self.elo[away]) / 400))
        score_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = self.k * (score_h - exp_h)
        self.elo[home] += delta
        self.elo[away] -= delta
        ph, pa = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        sh = PRIOR_SOT if pd.isna(hsot) else hsot
        sa = PRIOR_SOT if pd.isna(asot) else asot
        self.hist[home].append(dict(date=date, gf=hg, ga=ag, pts=ph, sot=sh, home=True))
        self.hist[away].append(dict(date=date, gf=ag, ga=hg, pts=pa, sot=sa, home=False))
        self.h2h[frozenset((home, away))].append((home, ph))


def build_features(matches: pd.DataFrame, injuries: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Featurise every match in date order. Optional `injuries` columns: date, team, impact (0-1)."""
    inj = {}
    if injuries is not None:
        inj = {(r.team, pd.Timestamp(r.date)): float(r.impact) for r in injuries.itertuples()}
    fb, rows = FeatureBuilder(), []
    for m in matches.itertuples():
        d = pd.Timestamp(m.Date)
        rows.append(fb.row(m.HomeTeam, m.AwayTeam, d, inj.get((m.HomeTeam, d), 0.0), inj.get((m.AwayTeam, d), 0.0)))
        fb.update(m.HomeTeam, m.AwayTeam, d, int(m.FTHG), int(m.FTAG), m.HST, m.AST)
    return pd.DataFrame(rows, index=matches.index), matches["FTR"].map(RESULT_TO_INT)
