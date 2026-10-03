"""Fetch live Premier League data for the website. Run daily by .github/workflows/data.yml.

Writes docs/data/{players,fixtures,formations,meta}.json.
- Players, injuries and fixtures come from the public Fantasy Premier League API (no key needed).
- Most-used formations come from API-Football if API_FOOTBALL_KEY is set; otherwise skipped
  and the site keeps its built-in formations.
"""
from __future__ import annotations

import difflib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "data"
FPL = "https://fantasy.premierleague.com/api/"
AF = "https://v3.football.api-sports.io/"
TIMEOUT, MAX_ATTEMPTS = 20, 3
SITE_NAMES = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nottingham Forest",
              "Leeds": "Leeds United", "Coventry": "Coventry City", "Hull": "Hull City", "Ipswich": "Ipswich Town"}
AF_ALIASES = {"Man United": "Manchester United", "Man City": "Manchester City", "Leeds United": "Leeds",
              "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich"}
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STATUS_OK = {"a"}


def get(url: str, headers: dict | None = None):
    """GET JSON with a timeout and a bounded number of retries."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pitchside portfolio)", **(headers or {})})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.load(r)
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"retry {attempt} for {url}: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)


def team_names(bootstrap: dict) -> dict[int, str]:
    return {t["id"]: SITE_NAMES.get(t["name"], t["name"]) for t in bootstrap["teams"]}


def build_players(bootstrap: dict) -> dict[str, list[dict]]:
    names, out = team_names(bootstrap), {}
    for p in bootstrap["elements"]:
        code = str(p.get("photo", "")).rsplit(".", 1)[0]
        out.setdefault(names[p["team"]], []).append({
            "id": p["id"], "name": p["web_name"], "full": f'{p["first_name"]} {p["second_name"]}',
            "pos": POS[p["element_type"]], "st": p["status"], "news": p.get("news", ""),
            "min": p["minutes"], "g": p["goals_scored"], "a": p["assists"], "form": float(p["form"] or 0),
            "photo": f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png" if code else None,
        })
    return out


def build_fixtures(fixtures: list[dict], names: dict[int, str], limit: int = 20) -> list[dict]:
    up = [f for f in fixtures if f.get("kickoff_time") and not f.get("finished")]
    up.sort(key=lambda f: f["kickoff_time"])
    return [{"kickoff": f["kickoff_time"], "home": names[f["team_h"]], "away": names[f["team_a"]]} for f in up[:limit]]


def fetch_formations(key: str, season: int, site_teams: list[str]) -> dict[str, list[str]]:
    headers, out = {"x-apisports-key": key}, {}
    teams = get(f"{AF}teams?league=39&season={season}", headers).get("response", [])
    by_name = {t["team"]["name"].lower(): t["team"]["id"] for t in teams}
    for site in site_teams:
        hit = difflib.get_close_matches(AF_ALIASES.get(site, site).lower(), list(by_name), n=1, cutoff=0.6)
        if not hit:
            print(f"no API-Football match for {site}", file=sys.stderr)
            continue
        try:
            stats = get(f"{AF}teams/statistics?league=39&season={season}&team={by_name[hit[0]]}", headers)
            lineups = sorted(stats["response"].get("lineups", []), key=lambda x: -x["played"])
            out[site] = [x["formation"] for x in lineups[:3]]
        except Exception as exc:
            print(f"formations failed for {site}: {exc}", file=sys.stderr)
        time.sleep(0.3)
    return out


def main() -> int:
    bs = get(FPL + "bootstrap-static/")
    names = team_names(bs)
    players, fixtures = build_players(bs), build_fixtures(get(FPL + "fixtures/"), names)
    if len(players) < 20 or not fixtures:
        print("unexpected FPL payload, refusing to overwrite data", file=sys.stderr)
        return 1
    formations, key = {}, os.environ.get("API_FOOTBALL_KEY")
    if key:
        season = datetime.fromisoformat(bs["events"][0]["deadline_time"].replace("Z", "+00:00")).year
        try:
            formations = fetch_formations(key, season, sorted(players))
        except Exception as exc:
            print(f"formation fetch skipped: {exc}", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in {"players": players, "fixtures": fixtures, "formations": formations,
                       "meta": {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}}.items():
        (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {len(players)} teams, {len(fixtures)} fixtures, {len(formations)} formation sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
