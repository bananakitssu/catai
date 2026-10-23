"""Private Vercel Python Function endpoint for CatAI inference."""
from __future__ import annotations

import hmac
import json
import logging
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

from runtime import generate, generate_stream, health

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 256 * 1024


class handler(BaseHTTPRequestHandler):
    def _json(self, body: dict, status: int = 200) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        # Avoid logging prompts, user content, or authorization headers.
        return

    def do_GET(self) -> None:
        if urlsplit(self.path).path.rstrip("/") not in ("/", "/api/generate", "/generate"):
            self._json({"error": "Not found"}, 404)
            return
        self._json(health())

    def do_POST(self) -> None:
        if urlsplit(self.path).path.rstrip("/") not in ("/", "/api/generate", "/generate"):
            self._json({"error": "Not found"}, 404)
            return
        expected = os.environ.get("CATAI_API_SECRET", "")
        supplied = self.headers.get("Authorization", "")
        if not expected:
            self._json({"error": "Inference authentication is not configured"}, 503)
            return
        if not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], expected):
            self._json({"error": "Unauthorized"}, 401)
            return
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self._json({"error": "Invalid Content-Length"}, 400)
            return
        if length < 2 or length > MAX_BODY_BYTES:
            self._json({"error": "Request body must be between 2 bytes and 256 KiB"}, 413)
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json({"error": "Invalid JSON body"}, 400)
            return
        if not isinstance(body, dict):
            self._json({"error": "JSON body must be an object"}, 400)
            return
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for event in generate_stream(
                    body.get("messages"),
                    max_new_tokens=body.get("max_new_tokens", 160),
                    temperature=body.get("temperature", 0.8),
                    top_k=body.get("top_k", 40),
                ):
                    payload = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
                    self.wfile.write(f"event: {event['event']}\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (TypeError, ValueError) as exc:
                payload = json.dumps({"error": str(exc)}, separators=(",", ":"), ensure_ascii=True)
                try:
                    self.wfile.write(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            except Exception:
                logger.exception("CatAI inference failed while streaming")
                payload = json.dumps({"error": "CatAI inference failed"}, separators=(",", ":"))
                try:
                    self.wfile.write(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        try:
            result = generate(
                body.get("messages"),
                max_new_tokens=body.get("max_new_tokens", 160),
                temperature=body.get("temperature", 0.8),
                top_k=body.get("top_k", 40),
            )
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        except Exception:
            # Keep details in server logs for diagnosis, never in the API response.
            logger.exception("CatAI inference failed while loading the model or generating")
            self._json({"error": "CatAI inference failed"}, 502)
            return
        self._json(result, 200)
