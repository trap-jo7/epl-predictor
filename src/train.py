"""Train and evaluate Random Forest and XGBoost on chronological splits.

python -m src.train            # real data from football-data.co.uk
python -m src.train --demo     # synthetic smoke test, no network
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from .data import load_matches, synthetic_matches
from .features import build_features

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Home", "Draw", "Away"]


def make_models() -> dict:
    models = {
        "elo_logreg": None,  # baseline: logistic regression on elo_diff alone
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=8, max_features="sqrt",
                                                random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                          colsample_bytree=0.8, objective="multi:softprob",
                                          eval_metric="mlogloss", random_state=42, n_jobs=-1)
    except ImportError:
        print("xgboost not installed: training Random Forest and baseline only")
    return models


def fit(name, model, X, y):
    if name == "elo_logreg":
        return LogisticRegression(max_iter=1000).fit(X[["elo_diff"]], y)
    return model.fit(X, y)


def proba(name, model, X):
    return model.predict_proba(X[["elo_diff"]] if name == "elo_logreg" else X)


def brier(p, y):
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--first", type=int, default=2015, help="first season start year")
    ap.add_argument("--test-matches", type=int, default=380, help="most recent N matches held out")
    args = ap.parse_args()

    matches = synthetic_matches() if args.demo else load_matches(args.first)
    X, y = build_features(matches)
    cut = len(X) - args.test_matches
    Xtr, ytr, Xte, yte = X.iloc[:cut], y.iloc[:cut].to_numpy(), X.iloc[cut:], y.iloc[cut:].to_numpy()
    print(f"{len(Xtr)} train matches, {len(Xte)} test matches (chronological split)")

    report, fitted = {}, {}
    for name, model in make_models().items():
        cv = []
        for a, b in TimeSeriesSplit(n_splits=5).split(Xtr):
            m = fit(name, model, Xtr.iloc[a], ytr[a])
            cv.append(log_loss(ytr[b], proba(name, m, Xtr.iloc[b]), labels=[0, 1, 2]))
        m = fit(name, model, Xtr, ytr)
        p = proba(name, m, Xte)
        report[name] = dict(cv_log_loss=float(np.mean(cv)), test_accuracy=float(accuracy_score(yte, p.argmax(1))),
                            test_log_loss=float(log_loss(yte, p, labels=[0, 1, 2])), test_brier=brier(p, yte))
        fitted[name] = (m, p)
        print(name, {k: round(v, 4) for k, v in report[name].items()})
    report["always_home_accuracy"] = float(np.mean(yte == 0))

    best = min(fitted, key=lambda n: report[n]["cv_log_loss"])  # chosen on CV only, never on the test set
    final = fit(best, make_models()[best], X, y.to_numpy())
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump({"model": final, "name": best, "features": list(X.columns)}, ROOT / "models" / "best.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps({"best": best, "demo": args.demo, **report}, indent=2))

    model, p = fitted[best]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    idx = np.digitize(p[:, 0], np.linspace(0, 1, 9)) - 1
    pts = [(p[idx == i, 0].mean(), (yte[idx == i] == 0).mean()) for i in range(8) if (idx == i).sum() >= 5]
    ax[0].plot([0, 1], [0, 1], "--", c="gray")
    if pts:
        ax[0].plot(*zip(*pts), "o-")
    ax[0].set(title="Calibration: home win", xlabel="predicted", ylabel="observed")
    ConfusionMatrixDisplay.from_predictions(yte, p.argmax(1), display_labels=LABELS, ax=ax[1], colorbar=False)
    ax[1].set_title("Confusion matrix (test)")
    if hasattr(model, "feature_importances_"):
        order = np.argsort(model.feature_importances_)[-10:]
        ax[2].barh(np.array(X.columns)[order], model.feature_importances_[order])
    ax[2].set_title(f"Top features ({best})")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "evaluation.png", dpi=150)
    print(f"best by CV log loss: {best}. Saved models/best.joblib and reports/.")


if __name__ == "__main__":
    main()
