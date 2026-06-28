"""Vercel serverless — POST /api/predict"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_match  # noqa: E402


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _cors_headers(self)
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            result = predict_match(
                player1=body["player1"],
                player2=body["player2"],
                discipline=body.get("discipline", "MS"),
                tournament_tier=int(body.get("tournament_tier", 300)),
                round_importance=int(body.get("round_importance", 3)),
            )
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            _cors_headers(self)
            self.end_headers()
            self.wfile.write(payload)
        except KeyError as e:
            self._error(400, f"Missing field: {e}")
        except (ValueError, FileNotFoundError) as e:
            self._error(400, str(e))
        except Exception as e:
            self._error(500, str(e))

    def _error(self, code: int, message: str) -> None:
        payload = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        _cors_headers(self)
        self.end_headers()
        self.wfile.write(payload)
