"""Vercel entry point: Vercel looks for an ASGI `app` in api/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crease_api.main import app  # noqa: E402,F401
