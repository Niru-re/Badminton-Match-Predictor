"""ETL: load cleaned matches and features into Supabase PostgreSQL."""
from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
from supabase import Client, create_client

from src.clean_data import run_cleaning
from src.config import SUPABASE_KEY, SUPABASE_URL
from src.feature_engineering import FEATURE_COLS, run_feature_engineering


def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            "Set SUPABASE_URL and SUPABASE_KEY in .env before running ETL."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _chunk(records: list[dict], size: int = 500):
    for i in range(0, len(records), size):
        yield records[i : i + size]


def upsert_players(client: Client, df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Return mapping (player_name, discipline) -> player_id.
    Fetches existing UUIDs first so foreign keys stay intact on re-runs."""
    # Fetch all existing players
    existing: dict[tuple[str, str], str] = {}
    page, size = 0, 1000
    while True:
        res = client.table("players").select("player_id,player_name,discipline").range(page * size, (page + 1) * size - 1).execute()
        for r in res.data:
            existing[(r["player_name"], r["discipline"])] = r["player_id"]
        if len(res.data) < size:
            break
        page += 1

    player_map: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str, str]] = set()
    records = []

    for _, row in df.iterrows():
        for col, country_col in [("player1", "country1"), ("player2", "country2")]:
            key = (row[col], row["discipline"])
            if key in player_map:
                continue
            # Reuse existing UUID if present
            uid = existing.get(key) or str(uuid.uuid4())
            player_map[key] = uid
            triple = (row[col], row["discipline"], row[country_col])
            if triple not in seen:
                seen.add(triple)
                records.append(
                    {
                        "player_id": uid,
                        "player_name": row[col],
                        "country": row[country_col],
                        "discipline": row["discipline"],
                    }
                )

    for batch in _chunk(records):
        client.table("players").upsert(batch, on_conflict="player_name,discipline").execute()
    print(f"Upserted {len(records):,} players")
    return player_map


def upsert_tournaments(client: Client, df: pd.DataFrame) -> dict[tuple, str]:
    # Fetch existing tournaments to preserve UUIDs
    existing: dict[tuple, str] = {}
    page, size = 0, 1000
    while True:
        res = client.table("tournaments").select("tournament_id,name,city,country").range(page * size, (page + 1) * size - 1).execute()
        for r in res.data:
            existing[(r["name"], r["city"], r["country"])] = r["tournament_id"]
        if len(res.data) < size:
            break
        page += 1

    tourney_map: dict[tuple, str] = {}
    records = []
    for _, row in df.drop_duplicates(["tournament", "city", "country"]).iterrows():
        key = (row["tournament"], row["city"], row["country"])
        if key in tourney_map:
            continue
        uid = existing.get(key) or str(uuid.uuid4())
        tourney_map[key] = uid
        records.append(
            {
                "tournament_id": uid,
                "name": row["tournament"],
                "city": row["city"],
                "country": row["country"],
                "category": row["tournament_type"],
                "category_tier": int(row["tournament_tier"]),
            }
        )

    for batch in _chunk(records):
        client.table("tournaments").upsert(
            batch, on_conflict="name,city,country"
        ).execute()
    print(f"Upserted {len(records):,} tournaments")
    return tourney_map


def _safe(val):
    """Convert NaN / Inf floats to None for JSON serialisation."""
    if val is None:
        return None
    try:
        import math
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
    except Exception:
        pass
    return val


def load_matches(client: Client, df: pd.DataFrame, player_map, tourney_map) -> dict[int, str]:
    """Load matches; return row index -> match_id."""
    match_ids: dict[int, str] = {}
    records = []

    for idx, row in df.iterrows():
        mid = str(uuid.uuid4())
        match_ids[idx] = mid
        t_key = (row["tournament"], row["city"], row["country"])
        p1_key = (row["player1"], row["discipline"])
        p2_key = (row["player2"], row["discipline"])
        winner_key = p1_key if row["player1_wins"] == 1 else p2_key

        records.append(
            {
                "match_id": mid,
                "tournament_id": tourney_map[t_key],
                "player1_id": player_map[p1_key],
                "player2_id": player_map[p2_key],
                "winner_id": player_map[winner_key],
                "discipline": row["discipline"],
                "round": row["round"],
                "match_date": row["match_date"].strftime("%Y-%m-%d"),
                "nb_sets": int(row["nb_sets"]) if pd.notna(row["nb_sets"]) else None,
                "retired": bool(row["retired"]),
                "team1_total_points": int(row["team_one_total_points"])
                if pd.notna(row["team_one_total_points"])
                else None,
                "team2_total_points": int(row["team_two_total_points"])
                if pd.notna(row["team_two_total_points"])
                else None,
                "game_1_score": _safe(row.get("game_1_score")),
                "game_2_score": _safe(row.get("game_2_score")),
                "game_3_score": _safe(row.get("game_3_score")),
            }
        )

    for batch in _chunk(records):
        client.table("matches").upsert(batch).execute()
    print(f"Loaded {len(records):,} matches")
    return match_ids


def load_features(
    client: Client,
    features: pd.DataFrame,
    matches_df: pd.DataFrame,
    match_ids: dict[int, str],
) -> None:
    merged = features.merge(
        matches_df.reset_index().rename(columns={"index": "orig_idx"}),
        on=["match_date", "player1", "player2", "discipline"],
        how="inner",
        suffixes=("", "_match"),
    )
    # After merge, colliding cols get suffix _match; use features-side values
    def _col(row, name):
        return row[name] if name in row.index else row[f"{name}_match"]

    records = []
    for _, row in merged.iterrows():
        mid = match_ids.get(row["orig_idx"])
        if not mid:
            continue
        records.append(
            {
                "feature_id": str(uuid.uuid4()),
                "match_id": mid,
                "player1_strength": float(row["player1_strength"]),
                "player2_strength": float(row["player2_strength"]),
                "strength_diff": float(row["strength_diff"]),
                "head_to_head_diff": int(row["head_to_head_diff"]),
                "recent_form_p1": float(row["recent_form_p1"]),
                "recent_form_p2": float(row["recent_form_p2"]),
                "fatigue_p1": int(row["fatigue_p1"]),
                "fatigue_p2": int(row["fatigue_p2"]),
                "tournament_tier": int(_col(row, "tournament_tier")),
                "round_importance": int(_col(row, "round_importance")),
                "target_player1_wins": int(row["target_player1_wins"]),
            }
        )

    for batch in _chunk(records):
        client.table("match_features").upsert(batch, on_conflict="match_id").execute()
    print(f"Loaded {len(records):,} feature rows")


def run_etl(discipline: str = "MS") -> None:
    client = get_client()
    matches_df = run_cleaning()
    matches_df = matches_df[matches_df["discipline"] == discipline].reset_index(drop=True)
    features = run_feature_engineering(discipline=discipline)

    player_map = upsert_players(client, matches_df)
    tourney_map = upsert_tournaments(client, matches_df)
    match_ids = load_matches(client, matches_df, player_map, tourney_map)
    load_features(client, features, matches_df, match_ids)
    print("ETL complete.")


if __name__ == "__main__":
    run_etl()
