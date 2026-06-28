"""Quick EDA summary — run after cleaning."""
from __future__ import annotations

import pandas as pd

from src.clean_data import run_cleaning


def main():
    df = run_cleaning()
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDiscipline counts:\n{df['discipline'].value_counts()}")
    print(f"\nTournament tiers:\n{df['tournament_tier'].value_counts().sort_index()}")
    print(f"\nPlayer 1 win rate: {df['player1_wins'].mean():.1%}")
    print(f"\nTop tournaments:\n{df['tournament'].value_counts().head(10)}")


if __name__ == "__main__":
    main()
