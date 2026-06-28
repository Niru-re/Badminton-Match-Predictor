"""Project configuration — load from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

SINGLES_FILES = {
    "MS": RAW_DIR / "ms.csv",
    "WS": RAW_DIR / "ws.csv",
}
DOUBLES_FILES = {
    "MD": RAW_DIR / "md.csv",
    "WD": RAW_DIR / "wd.csv",
    "XD": RAW_DIR / "xd.csv",
}

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # service role key for ETL
DATABASE_URL = os.getenv("DATABASE_URL", "")    # optional direct Postgres connection

# Default discipline for MVP prediction (men's singles has most data)
DEFAULT_DISCIPLINE = os.getenv("DEFAULT_DISCIPLINE", "MS")
