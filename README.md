# Pitchside: Premier League match predictor

A machine-learning pipeline that predicts Premier League results (home / draw / away probabilities),
plus an interactive tactics and squad site in `docs/` (host it free with GitHub Pages: Settings, Pages, `/docs`).

## What is inside
| Path | Purpose |
|---|---|
| `src/data.py` | Downloads and caches real results from football-data.co.uk. Synthetic generator for tests. |
| `src/features.py` | Leakage-safe features: Elo with home advantage, rolling form, home/away splits, head-to-head, rest days, shots on target, optional injury impact. |
| `src/train.py` | Random Forest, XGBoost and an Elo-only baseline. Chronological hold-out, `TimeSeriesSplit` CV, log loss, Brier, calibration and feature-importance plots. |
| `src/predict.py` | Predict any fixture from the saved model. |
| `scripts/fetch_data.py` | Pulls real players, photos, injury news and fixtures (Fantasy Premier League API, no key) and optional most-used formations (API-Football) into `docs/data/*.json`. |
| `.github/workflows/data.yml` | Refreshes that data daily and commits it. |
| `tests/` | Unit tests, including a check that features never see future matches. |
| `docs/index.html` | The website: predictor, tactics lab with formations, press / long-ball / counter / break-the-defence animations, squad cards. |

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train                       # downloads 2015/16 onward, trains, writes reports/
python -m src.predict "Arsenal" "Chelsea" # probabilities for a fixture
pytest -q
python -m src.train --demo                # offline smoke test on synthetic data
```

## Live data for the website
```bash
python scripts/fetch_data.py                       # players, injuries, fixtures
API_FOOTBALL_KEY=your_key python scripts/fetch_data.py   # also real formations (optional)
```
1. Run the script once, commit `docs/data/`, and the site shows real names, photos, injury news and fixtures.
2. For daily updates, push to GitHub, enable Actions, and (optional) add `API_FOOTBALL_KEY` under Settings, Secrets, Actions.
3. If the data files are missing, the site falls back to labelled sample data.

Known limits: the Fantasy API has no shirt numbers or formations, so numbers are hidden for real players and formations stay sample until you add the API-Football key (check that your plan includes the current season). Starting XIs are the most-played healthy players, not announced lineups. Player photos are hotlinked from premierleague.com, so keep the project non-commercial.

## Method notes (read before quoting numbers)
- Split is **chronological**: the most recent 380 matches are held out, never shuffled.
- Model choice uses cross-validation on the training period only, not the test set.
- Football is noisy. Judge the model on **log loss and calibration** against the Elo baseline and `always_home_accuracy`, not on accuracy alone. Bookmaker-quality models land around 52-55% accuracy, so treat anything far above that as a bug.
- Injuries: football-data.co.uk has no injury data. Pass a DataFrame (`date, team, impact`) to `build_features` to use the `inj_*` features; otherwise they are 0.
- Promoted teams start at Elo 1500, which is a known weakness.
- The website's ratings, squads and injuries are illustrative sample data and are not yet wired to the model.

## Next steps
Export model ratings to JSON for the site, add expected-goals data, calibrate probabilities (`CalibratedClassifierCV`), add an injury source.
# epl-predictor
