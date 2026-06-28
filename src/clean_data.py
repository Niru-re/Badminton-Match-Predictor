"""Load, clean, and unify BWF match CSVs into a single processed dataset."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import DOUBLES_FILES, PROCESSED_DIR, RAW_DIR, SINGLES_FILES


def normalize_name(name: str) -> str:
    if pd.isna(name) or not str(name).strip():
        return ""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def parse_tournament_tier(tournament_type: str) -> int:
    """Extract numeric tier from strings like 'HSBC BWF World Tour Super 300'."""
    if pd.isna(tournament_type):
        return 0
    match = re.search(r"super\s*(\d+)", str(tournament_type), re.I)
    return int(match.group(1)) if match else 0


def encode_round(round_name: str) -> int:
    """Higher = deeper in tournament (more pressure)."""
    if pd.isna(round_name):
        return 0
    r = str(round_name).lower()
    if "final" in r and "semi" not in r and "quarter" not in r:
        return 5
    if "semi" in r:
        return 4
    if "quarter" in r:
        return 3
    if "round of 16" in r or "16" in r:
        return 2
    if "round of 32" in r or "32" in r:
        return 1
    return 0


def load_singles(path: Path, discipline: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["discipline"] = discipline
    df["player1"] = df["team_one_players"].map(normalize_name)
    df["player2"] = df["team_two_players"].map(normalize_name)
    df["country1"] = df["team_one_nationalities"]
    df["country2"] = df["team_two_nationalities"]
    return df


def load_doubles(path: Path, discipline: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["discipline"] = discipline
    df["player1"] = (
        df["team_one_player_one"].map(normalize_name)
        + " / "
        + df["team_one_player_two"].map(normalize_name)
    )
    df["player2"] = (
        df["team_two_player_one"].map(normalize_name)
        + " / "
        + df["team_two_player_two"].map(normalize_name)
    )
    df["country1"] = df["team_one_player_one_nationality"]
    df["country2"] = df["team_two_player_one_nationality"]
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "tournament",
        "city",
        "country",
        "date",
        "tournament_type",
        "discipline",
        "round",
        "winner",
        "nb_sets",
        "retired",
        "player1",
        "player2",
        "country1",
        "country2",
        "team_one_total_points",
        "team_two_total_points",
        "game_1_score",
        "game_2_score",
        "game_3_score",
    ]
    df = df[core_cols].copy()
    df = df.drop_duplicates()

    df["match_date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["match_date", "player1", "player2"])

    # Remove retired / invalid winners
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce")
    df = df[df["winner"].isin([1, 2])]
    df["retired"] = df["retired"].astype(bool)

    df["tournament_tier"] = df["tournament_type"].map(parse_tournament_tier)
    df["round_importance"] = df["round"].map(encode_round)
    df["player1_wins"] = (df["winner"] == 1).astype(int)

    df = df.sort_values("match_date").reset_index(drop=True)
    return df


def load_all_raw() -> pd.DataFrame:
    frames = []
    for disc, path in SINGLES_FILES.items():
        if path.exists():
            frames.append(load_singles(path, disc))
    for disc, path in DOUBLES_FILES.items():
        if path.exists():
            frames.append(load_doubles(path, disc))
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    return pd.concat(frames, ignore_index=True)


def run_cleaning() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_all_raw()
    cleaned = clean_matches(raw)

    out_path = PROCESSED_DIR / "matches_clean.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Saved {len(cleaned):,} matches -> {out_path}")
    print(f"Disciplines: {cleaned['discipline'].value_counts().to_dict()}")
    print(f"Date range: {cleaned['match_date'].min()} to {cleaned['match_date'].max()}")
    return cleaned


if __name__ == "__main__":
    run_cleaning()
