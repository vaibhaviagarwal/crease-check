"""HTTP API for the checker.

    uvicorn crease_api.main:app --reload       # then open http://localhost:8000

Endpoints (all under /api):

    GET  /health
    GET  /examples                list built-in patterns
    GET  /examples/{id}           one of them, checked
    POST /check                   JSON: a FOLD object, or {"cp": "<text>"}
    POST /check/file              upload a .fold or .cp file
    POST /photo                   upload a photo of a marker drawing
    POST /fix                     apply one suggested fix and re-check
    GET  /checks/recent           latest checks (needs DATABASE_URL)
    GET  /stats                   which rules fail most (needs DATABASE_URL)

Interactive docs at /docs.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crease.checks import Violation, apply_fix, check, describe_fix
from crease.examples import EXAMPLES
from crease.io import PatternError, parse, parse_cp, parse_fold, to_fold
from crease.model import Pattern
from crease.vision import PhotoError, read_photo

from . import store

MAX_UPLOAD = 8 * 1024 * 1024
MAX_EDGES = 20_000
PUBLIC = Path(__file__).resolve().parents[1] / "public"

app = FastAPI(
    title="Crease Check",
    description="Will your origami crease pattern fold flat?",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/api/openapi.json",
)


def _respond(pattern: Pattern, source: str, angle_tol: float, extra: dict | None = None) -> dict:
    if len(pattern.edges) > MAX_EDGES:
        raise HTTPException(413, f"Pattern too large: over {MAX_EDGES:,} lines.")
    report = check(pattern, angle_tol=angle_tol)
    body = report.to_dict()
    for v in body["violations"]:
        v["fix_text"] = describe_fix(v["fix"])
    fold = to_fold(report.pattern)
    body["pattern"] = fold
    body["check_id"] = store.save(source, body, fold)
    if extra:
        body.update(extra)
    return body


def _parse_or_400(fn, *args) -> Pattern:
    try:
        return fn(*args)
    except PatternError as e:
        raise HTTPException(400, str(e)) from None


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File too large (8 MB max).")
    if not data:
        raise HTTPException(400, "The file is empty.")
    return data


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "database": store.enabled()}


@app.get("/api/examples")
def examples() -> list[dict]:
    return [{"id": k, "label": label} for k, (label, _) in EXAMPLES.items()]


@app.get("/api/examples/{example_id}")
def example(example_id: str) -> dict:
    if example_id not in EXAMPLES:
        raise HTTPException(404, f"No example called {example_id!r}.")
    label, build = EXAMPLES[example_id]
    return _respond(build(), "example", 0.01, {"example": {"id": example_id, "label": label}})


@app.post("/api/check")
def check_json(body: dict, angle_tol: float = Query(0.01, gt=0, le=10)) -> dict:
    if "cp" in body:
        pattern = _parse_or_400(parse_cp, str(body["cp"]))
    else:
        pattern = _parse_or_400(parse_fold, body.get("fold", body))
    return _respond(pattern, "json", angle_tol)


@app.post("/api/check/file")
async def check_file(file: UploadFile = File(...), angle_tol: float = Form(0.01)) -> dict:
    data = await _read_upload(file)
    pattern = _parse_or_400(parse, data, file.filename or "")
    return _respond(pattern, "file", angle_tol)


@app.post("/api/photo")
async def check_photo(file: UploadFile = File(...), snap: bool = Form(True),
                      angle_tol: float = Form(1.0)) -> dict:
    data = await _read_upload(file)
    t0 = time.perf_counter()
    try:
        res = read_photo(data, snap=22.5 if snap else None)
    except PhotoError as e:
        raise HTTPException(422, str(e)) from None
    small = cv2.resize(res.sheet, (480, 480), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    sheet = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None
    return _respond(res.pattern, "photo", angle_tol,
                    {"photo": {**res.stats, "read_ms": round((time.perf_counter() - t0) * 1000),
                               "lines": res.lines, "sheet": sheet}})


class FixRequest(BaseModel):
    fold: dict = Field(..., description="The pattern from a previous response")
    vertex: int
    fix: dict
    angle_tol: float = 0.01


@app.post("/api/fix")
def fix(req: FixRequest) -> dict:
    pattern = _parse_or_400(parse_fold, req.fold)
    if not 0 <= req.vertex < len(pattern.vertices):
        raise HTTPException(400, "That vertex is not in the pattern.")
    try:
        fixed = apply_fix(pattern, Violation("", req.vertex, pattern.vertices[req.vertex], "", {}, req.fix))
    except (KeyError, ValueError, IndexError):
        raise HTTPException(400, "That fix does not match the pattern.") from None
    return _respond(fixed, "fix", req.angle_tol)


@app.get("/api/checks/recent")
def recent(limit: int = Query(10, ge=1, le=100)) -> list[dict]:
    return store.recent(limit)


@app.get("/api/stats")
def stats() -> dict:
    return store.stats()


# The web page. On Vercel the platform serves public/ itself; locally this does.
if PUBLIC.is_dir():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PUBLIC / "index.html")

    app.mount("/", StaticFiles(directory=PUBLIC), name="public")
