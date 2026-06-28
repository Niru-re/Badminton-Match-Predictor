"""Pure-Python inference — no sklearn needed at runtime.

Loads model_<discipline>.json and runs prediction using only
the Python standard library + math module.
"""
from __future__ import annotations
import json, math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@lru_cache(maxsize=5)
def _load(discipline: str) -> dict:
    path = MODELS_DIR / f"model_{discipline.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"No model JSON for {discipline}. Run: python scripts/export_model_json.py")
    with open(path) as f:
        return json.load(f)


def _predict_tree_rf(node: dict, x: list[float]) -> float:
    """Walk a RandomForest tree, return class-1 probability at leaf."""
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["prob"]


def _predict_tree_gb(node: dict, x: list[float]) -> float:
    """Walk a GradientBoosting tree, return leaf value."""
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["val"]


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def predict_proba(discipline: str, features: dict) -> float:
    """Return probability that player1 wins (float 0-1)."""
    bundle = _load(discipline)
    feat_cols: list[str] = bundle["features"]
    x = [float(features.get(c, 0.0)) for c in feat_cols]

    mtype: str = bundle["type"]
    trees: list[dict] = bundle["trees"]

    if mtype == "RandomForest":
        prob = sum(_predict_tree_rf(t, x) for t in trees) / len(trees)
        return prob

    if mtype == "GradientBoosting":
        lr: float = bundle["learning_rate"]
        log_odds: float = bundle["init_log_odds"]
        for t in trees:
            log_odds += lr * _predict_tree_gb(t, x)
        return _sigmoid(log_odds)

    raise ValueError(f"Unknown model type: {mtype}")
