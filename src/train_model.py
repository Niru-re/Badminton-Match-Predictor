"""Train and evaluate match winner prediction models."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_DISCIPLINE, MODELS_DIR, PROCESSED_DIR
from src.feature_engineering import FEATURE_COLS, run_feature_engineering


def load_features(discipline: str = DEFAULT_DISCIPLINE) -> pd.DataFrame:
    path = PROCESSED_DIR / f"features_{discipline.lower()}.csv"
    if not path.exists():
        return run_feature_engineering(discipline=discipline)
    return pd.read_csv(path, parse_dates=["match_date"])


def train_models(discipline: str = DEFAULT_DISCIPLINE) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features(discipline)

    X = df[FEATURE_COLS]
    y = df["target_player1_wins"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        ),
    }

    results = {}
    best_name, best_model, best_acc = "", None, 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred),
        }
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.2%}  |  ROC-AUC: {auc:.4f}")
        print(results[name]["classification_report"])

        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    # Feature importance (tree models)
    if hasattr(best_model, "feature_importances_"):
        importance = dict(zip(FEATURE_COLS, best_model.feature_importances_))
        sorted_imp = dict(sorted(importance.items(), key=lambda x: -x[1]))
        print(f"\nFeature importance ({best_name}):")
        for feat, val in sorted_imp.items():
            print(f"  {feat}: {val:.1%}")

    # Save best model
    model_path = MODELS_DIR / f"best_model_{discipline.lower()}.pkl"
    meta_path = MODELS_DIR / f"metrics_{discipline.lower()}.json"

    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": best_model, "features": FEATURE_COLS, "discipline": discipline},
            f,
        )

    meta = {
        "best_model": best_name,
        "discipline": discipline,
        "feature_columns": FEATURE_COLS,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "classification_report"} for k, v in results.items()},
        "best_accuracy": round(best_acc, 4),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved best model ({best_name}, {best_acc:.2%}) -> {model_path}")
    return results


if __name__ == "__main__":
    train_models()
