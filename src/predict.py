"""Shared inference logic for Streamlit, Vercel API, and local use."""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR
from src.feature_engineering import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
DISCIPLINES = ["MS", "WS", "MD", "WD", "XD"]


@lru_cache(maxsize=5)
def _load_model_bundle(discipline: str) -> dict | None:
    path = MODELS_DIR / f"best_model_{discipline.lower()}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=5)
def _load_features(discipline: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"features_{discipline.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.feature_engineering"
        )
    return pd.read_csv(path, parse_dates=["match_date"])


def list_players(discipline: str) -> list[str]:
    """Return players that have feature history (can actually be predicted)."""
    feat_path = PROCESSED_DIR / f"features_{discipline.lower()}.csv"
    if feat_path.exists():
        df = pd.read_csv(feat_path, usecols=["player1", "player2"])
        players = set(df["player1"].unique()) | set(df["player2"].unique())
        return sorted(p for p in players if p)
    # Fallback to matches_clean if features not yet generated
    path = PROCESSED_DIR / "matches_clean.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, usecols=["player1", "player2", "discipline"])
    df = df[df["discipline"] == discipline]
    players = set(df["player1"].unique()) | set(df["player2"].unique())
    return sorted(p for p in players if p)


def compute_live_features(
    p1: str,
    p2: str,
    discipline: str,
    tier: int,
    round_imp: int,
) -> dict | None:
    try:
        df = _load_features(discipline)
    except FileNotFoundError:
        return None

    p1_hist = df[(df["player1"] == p1) | (df["player2"] == p1)].sort_values("match_date")
    p2_hist = df[(df["player1"] == p2) | (df["player2"] == p2)].sort_values("match_date")
    if p1_hist.empty or p2_hist.empty:
        return None

    def last_stats(hist: pd.DataFrame, player: str) -> tuple[float, float, int]:
        row = hist.iloc[-1]
        if row["player1"] == player:
            return row["player1_strength"], row["recent_form_p1"], int(row["fatigue_p1"])
        return row["player2_strength"], row["recent_form_p2"], int(row["fatigue_p2"])

    s1, f1, fat1 = last_stats(p1_hist, p1)
    s2, f2, fat2 = last_stats(p2_hist, p2)

    h2h = df[
        ((df["player1"] == p1) & (df["player2"] == p2))
        | ((df["player1"] == p2) & (df["player2"] == p1))
    ]
    h2h_diff = 0
    if not h2h.empty:
        p1_wins = int(
            ((h2h["player1"] == p1) & (h2h["target_player1_wins"] == 1)).sum()
            + ((h2h["player2"] == p1) & (h2h["target_player1_wins"] == 0)).sum()
        )
        p2_wins = int(
            ((h2h["player1"] == p2) & (h2h["target_player1_wins"] == 1)).sum()
            + ((h2h["player2"] == p2) & (h2h["target_player1_wins"] == 0)).sum()
        )
        h2h_diff = p1_wins - p2_wins

    return {
        "strength_diff": float(s1 - s2),
        "head_to_head_diff": h2h_diff,
        "recent_form_p1": float(f1),
        "recent_form_p2": float(f2),
        "fatigue_p1": fat1,
        "fatigue_p2": fat2,
        "tournament_tier": tier,
        "round_importance": round_imp,
    }


def predict_match(
    player1: str,
    player2: str,
    discipline: str = "MS",
    tournament_tier: int = 300,
    round_importance: int = 3,
) -> dict:
    if player1 == player2:
        raise ValueError("Players must be different.")

    bundle = _load_model_bundle(discipline)
    if bundle is None:
        raise FileNotFoundError(
            f"No model for {discipline}. Run: python -m src.train_model"
        )

    feats = compute_live_features(
        player1, player2, discipline, tournament_tier, round_importance
    )
    if feats is None:
        raise ValueError(
            "Could not compute features — one or both players have no match history."
        )

    X = pd.DataFrame([feats])[FEATURE_COLS]
    model = bundle["model"]
    prob_p1 = float(model.predict_proba(X)[0][1])
    prob_p2 = 1.0 - prob_p1
    winner = player1 if prob_p1 >= prob_p2 else player2

    return {
        "player1": player1,
        "player2": player2,
        "discipline": discipline,
        "predicted_winner": winner,
        "player1_win_probability": round(prob_p1, 4),
        "player2_win_probability": round(prob_p2, 4),
        "features": feats,
        "model": bundle.get("discipline", discipline),
    }
