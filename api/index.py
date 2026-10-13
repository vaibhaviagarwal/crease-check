"""Vercel entry point: Vercel looks for an ASGI `app` in api/.

vercel.json rewrites every /api/... request to this one function. Vercel
then shows the function the rewritten path (/api/index), not the one that
was asked for, so the rewrite also passes the original path along as
?__path=... . This wrapper puts it back before the app routes the request,
so the same routes work locally and on Vercel.
"""

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crease_api.main import app as _app  # noqa: E402

KEY = "__path"


def _normalise(scope: dict) -> dict:
    scope = dict(scope)
    path = scope.get("path") or "/"
    root = scope.get("root_path") or ""
    if root and not path.startswith(root):
        path = root.rstrip("/") + path
    query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
    original = [v for k, v in query if k == KEY]
    if original and original[-1].startswith("/"):
        path = original[-1]
        query = [(k, v) for k, v in query if k != KEY]
        scope["query_string"] = urlencode(query).encode("latin-1")
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
