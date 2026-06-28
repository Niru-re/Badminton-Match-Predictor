"""Vercel serverless — POST /api/predict
Uses pure-Python inference (no sklearn/pandas at runtime).
"""
from __future__ import annotations

import csv, json, sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer import predict_proba  # pure-Python, no sklearn


# ── lightweight feature loader (csv only, no pandas) ──────────────
_features_cache: dict[str, list[dict]] = {}

def _load_features(discipline: str) -> list[dict]:
    if discipline in _features_cache:
        return _features_cache[discipline]
    path = ROOT / "data" / "processed" / f"features_{discipline.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing features for {discipline}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _features_cache[discipline] = rows
    return rows


def _compute_features(p1: str, p2: str, discipline: str, tier: int, round_imp: int) -> dict:
    rows = _load_features(discipline)

    p1_rows = [r for r in rows if r["player1"] == p1 or r["player2"] == p1]
    p2_rows = [r for r in rows if r["player1"] == p2 or r["player2"] == p2]

    if not p1_rows or not p2_rows:
        raise ValueError("Could not compute features — one or both players have no match history.")

    def last_stats(hist: list[dict], player: str) -> tuple[float, float, int]:
        row = hist[-1]
        if row["player1"] == player:
            return float(row["player1_strength"]), float(row["recent_form_p1"]), int(float(row["fatigue_p1"]))
        return float(row["player2_strength"]), float(row["recent_form_p2"]), int(float(row["fatigue_p2"]))

    s1, f1, fat1 = last_stats(p1_rows, p1)
    s2, f2, fat2 = last_stats(p2_rows, p2)

    h2h_rows = [r for r in rows if
        (r["player1"] == p1 and r["player2"] == p2) or
        (r["player1"] == p2 and r["player2"] == p1)]

    h2h_diff = 0
    if h2h_rows:
        p1w = sum(1 for r in h2h_rows if
            (r["player1"] == p1 and int(float(r["target_player1_wins"])) == 1) or
            (r["player2"] == p1 and int(float(r["target_player1_wins"])) == 0))
        p2w = len(h2h_rows) - p1w
        h2h_diff = p1w - p2w

    return {
        "strength_diff": s1 - s2,
        "head_to_head_diff": h2h_diff,
        "recent_form_p1": f1,
        "recent_form_p2": f2,
        "fatigue_p1": fat1,
        "fatigue_p2": fat2,
        "tournament_tier": tier,
        "round_importance": round_imp,
    }


# ── CORS helper ────────────────────────────────────────────────────
def _cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


# ── Vercel handler ─────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")

            p1   = body["player1"]
            p2   = body["player2"]
            disc = body.get("discipline", "MS")
            tier = int(body.get("tournament_tier", 300))
            rnd  = int(body.get("round_importance", 3))

            if p1 == p2:
                raise ValueError("Players must be different.")

            feats = _compute_features(p1, p2, disc, tier, rnd)
            prob_p1 = predict_proba(disc, feats)
            prob_p2 = 1.0 - prob_p1
            winner = p1 if prob_p1 >= prob_p2 else p2

            result = {
                "player1": p1,
                "player2": p2,
                "discipline": disc,
                "predicted_winner": winner,
                "player1_win_probability": round(prob_p1, 4),
                "player2_win_probability": round(prob_p2, 4),
                "features": feats,
            }
            self._json(200, result)

        except KeyError as e:
            self._json(400, {"error": f"Missing field: {e}"})
        except (ValueError, FileNotFoundError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, code: int, data: dict) -> None:
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        _cors(self)
        self.end_headers()
        self.wfile.write(payload)
