"""Vercel entry point: Vercel looks for an ASGI `app` in api/.

Vercel's Python runtime can hand the app its path in a different shape from
a local server (for example with "/api" moved into root_path, or stripped).
This wrapper puts every request back into the shape the app's routes use,
so the same routes work locally and on Vercel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crease_api.main import app as _app  # noqa: E402


def _normalise(scope: dict) -> dict:
    scope = dict(scope)
    path = scope.get("path") or "/"
    root = scope.get("root_path") or ""
    if root and not path.startswith(root):
        path = root.rstrip("/") + path
    if not (path.startswith("/api/") or path in ("/api", "/docs")):
        path = "/api" + (path if path.startswith("/") else "/" + path)
    scope["root_path"] = ""
    scope["path"] = path
    scope["raw_path"] = path.encode()
    return scope


async def app(scope, receive, send):
    if scope["type"] in ("http", "websocket"):
        scope = _normalise(scope)
    await _app(scope, receive, send)
