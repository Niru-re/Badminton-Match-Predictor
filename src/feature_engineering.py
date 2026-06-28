"""Feature engineering — time-safe features computed from prior match history only."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.config import DEFAULT_DISCIPLINE, PROCESSED_DIR
from src.clean_data import run_cleaning


def build_features(df: pd.DataFrame, discipline: str | None = None) -> pd.DataFrame:
    """
    Compute pre-match features without data leakage.
    Uses rolling win-rate as strength proxy (source data has no BWF rankings).
    """
    if discipline:
        df = df[df["discipline"] == discipline].copy()
    df = df.sort_values("match_date").reset_index(drop=True)

    # Per-player stats updated AFTER each match
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    recent: dict[str, list[int]] = defaultdict(list)  # last 5 results
    match_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)
    h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    rows = []
    for _, row in df.iterrows():
        p1, p2 = row["player1"], row["player2"]
        date = row["match_date"]

        # --- features BEFORE this match ---
        p1_games = wins[p1] + losses[p1]
        p2_games = wins[p2] + losses[p2]
        p1_strength = wins[p1] / p1_games if p1_games else 0.5
        p2_strength = wins[p2] / p2_games if p2_games else 0.5

        pair_key = tuple(sorted([p1, p2]))
        h2h_diff = h2h[pair_key][p1] - h2h[pair_key][p2]

        r1 = recent[p1][-5:]
        r2 = recent[p2][-5:]
        form_p1 = sum(r1) / len(r1) if r1 else 0.5
        form_p2 = sum(r2) / len(r2) if r2 else 0.5

        fatigue_p1 = sum(1 for d in match_dates[p1] if 0 < (date - d).days <= 7)
        fatigue_p2 = sum(1 for d in match_dates[p2] if 0 < (date - d).days <= 7)

        rows.append(
            {
                "match_date": date,
                "tournament": row["tournament"],
                "discipline": row["discipline"],
                "round": row["round"],
                "player1": p1,
                "player2": p2,
                "player1_strength": round(p1_strength, 4),
                "player2_strength": round(p2_strength, 4),
                "strength_diff": round(p1_strength - p2_strength, 4),
                "head_to_head_diff": h2h_diff,
                "recent_form_p1": round(form_p1, 4),
                "recent_form_p2": round(form_p2, 4),
                "fatigue_p1": fatigue_p1,
                "fatigue_p2": fatigue_p2,
                "tournament_tier": row["tournament_tier"],
                "round_importance": row["round_importance"],
                "target_player1_wins": row["player1_wins"],
            }
        )

        # --- update stats AFTER match ---
        winner = p1 if row["player1_wins"] == 1 else p2
        loser = p2 if winner == p1 else p1
        wins[winner] += 1
        losses[loser] += 1
        recent[p1].append(row["player1_wins"])
        recent[p2].append(1 - row["player1_wins"])
        match_dates[p1].append(date)
        match_dates[p2].append(date)
        h2h[pair_key][winner] += 1

    features = pd.DataFrame(rows)
    # Drop early matches where players have no history (optional — keeps model stable)
    features = features[
        (features["player1_strength"] != 0.5)
        | (features["player2_strength"] != 0.5)
        | (features["recent_form_p1"] != 0.5)
    ].reset_index(drop=True)
    return features


FEATURE_COLS = [
    "strength_diff",
    "head_to_head_diff",
    "recent_form_p1",
    "recent_form_p2",
    "fatigue_p1",
    "fatigue_p2",
    "tournament_tier",
    "round_importance",
]


def run_feature_engineering(discipline: str = DEFAULT_DISCIPLINE) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED_DIR / "matches_clean.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path, parse_dates=["match_date"])
    else:
        df = run_cleaning()

    features = build_features(df, discipline=discipline)
    out_path = PROCESSED_DIR / f"features_{discipline.lower()}.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved {len(features):,} feature rows -> {out_path}")
    return features


if __name__ == "__main__":
    run_feature_engineering()
