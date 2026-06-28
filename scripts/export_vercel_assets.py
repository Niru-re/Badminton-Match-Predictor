"""Export static JSON assets for the Vercel frontend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR
from src.feature_engineering import run_feature_engineering
from src.predict import DISCIPLINES, list_players
from src.train_model import train_models

DEPLOY_DATA = ROOT / "data" / "deploy"


def ensure_pipeline(discipline: str = "MS") -> None:
    clean = PROCESSED_DIR / "matches_clean.csv"
    features = PROCESSED_DIR / f"features_{discipline.lower()}.csv"
    model = MODELS_DIR / f"best_model_{discipline.lower()}.pkl"

    if not clean.exists():
        from src.clean_data import run_cleaning
        run_cleaning()
    if not features.exists():
        run_feature_engineering(discipline=discipline)
    if not model.exists():
        train_models(discipline=discipline)


def export_assets(discipline: str = "MS") -> None:
    ensure_pipeline(discipline)
    DEPLOY_DATA.mkdir(parents=True, exist_ok=True)

    players_by_discipline = {
        d: list_players(d) for d in DISCIPLINES
    }
    (DEPLOY_DATA / "players.json").write_text(
        json.dumps(players_by_discipline, indent=2), encoding="utf-8"
    )

    metrics_path = MODELS_DIR / f"metrics_{discipline.lower()}.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    manifest = {
        "default_discipline": discipline,
        "disciplines": DISCIPLINES,
        "model_accuracy": metrics.get("best_accuracy"),
        "best_model": metrics.get("best_model"),
    }
    (DEPLOY_DATA / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Exported -> {DEPLOY_DATA}")


if __name__ == "__main__":
    export_assets()
