"""The HTTP API, end to end through FastAPI's test client."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from crease.examples import EXAMPLES
from crease.io import to_cp, to_fold
from crease.synth import render_photo
from crease_api import store
from crease_api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"]


def test_examples_list_and_fetch():
    ids = [e["id"] for e in client.get("/api/examples").json()]
    assert set(ids) == set(EXAMPLES)
    body = client.get("/api/examples/broken-twist").json()
    assert body["flat_foldable"] is False
    assert body["failing_vertices"] == 2
    assert all(v["fix_text"] for v in body["violations"] if v["fix"])
    assert client.get("/api/examples/nope").status_code == 404


def test_check_fold_json():
    fold = to_fold(EXAMPLES["waterbomb"][1]())
    r = client.post("/api/check", json=fold)
    assert r.status_code == 200
    body = r.json()
    assert body["flat_foldable"] is True and body["scope"] == "local"
    assert body["checked"] == {"vertices": 1, "creases": 6}


def test_check_cp_json():
    r = client.post("/api/check", json={"cp": to_cp(EXAMPLES["miura-ori"][1]())})
    assert r.json()["flat_foldable"] is True


def test_upload_file():
    data = json.dumps(to_fold(EXAMPLES["square-twist"][1]())).encode()
    r = client.post("/api/check/file", files={"file": ("twist.fold", data, "application/json")})
    assert r.status_code == 200 and r.json()["flat_foldable"] is True


def test_bad_file_is_a_400_with_a_reason():
    r = client.post("/api/check/file", files={"file": ("x.fold", b"{oops", "application/json")})
    assert r.status_code == 400
    assert "Not valid JSON" in r.json()["detail"]


def test_empty_upload():
    r = client.post("/api/check/file", files={"file": ("x.cp", b"", "text/plain")})
    assert r.status_code == 400


def test_photo_upload():
    img = render_photo(EXAMPLES["waterbomb"][1](), 1001, "easy")
    r = client.post("/api/photo", files={"file": ("photo.jpg", img, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["flat_foldable"] is True
    assert body["photo"]["paper_found"] is True
    assert len(body["photo"]["lines"]) == 3      # two diagonals and the midline


def test_photo_assessment_passes_a_real_pattern():
    img = render_photo(EXAMPLES["square-twist"][1](), 1001, "easy")
    a = client.post("/api/photo", files={"file": ("p.jpg", img, "image/jpeg")}).json()["photo"]["assessment"]
    assert a["looks_like_pattern"] is True and a["problems"] == []
    assert a["found"]["mountain"] > 0 and a["found"]["valley"] > 0


def test_photo_assessment_flags_random_scribbles():
    import cv2
    import numpy as np
    img = np.full((700, 900, 3), (70, 80, 90), np.uint8)
    cv2.rectangle(img, (150, 80), (750, 640), (228, 236, 242), -1)
    for x in (330, 420, 560):                                   # short strokes, touching nothing
        cv2.line(img, (x, 250), (x, 330), (190, 90, 40), 6)
        cv2.line(img, (x, 430), (x, 510), (190, 90, 40), 6)
    ok, buf = cv2.imencode(".jpg", img)
    a = client.post("/api/photo", files={"file": ("p.jpg", buf.tobytes(), "image/jpeg")}).json()["photo"]["assessment"]
    assert a["looks_like_pattern"] is False
    assert a["problems"]


def test_photo_that_is_not_an_image():
    r = client.post("/api/photo", files={"file": ("photo.jpg", b"nope", "image/jpeg")})
    assert r.status_code == 422


def test_fix_round_trip():
    """Apply each suggested fix through the API until the pattern folds."""
    body = client.get("/api/examples/broken-twist").json()
    for _ in range(4):
        if body["flat_foldable"]:
            break
        v = next(v for v in body["violations"] if v["fix"])
        body = client.post("/api/fix", json={"fold": body["pattern"], "vertex": v["vertex"],
                                             "fix": v["fix"]}).json()
    assert body["flat_foldable"] is True


def test_docs_are_served():
    assert client.get("/api/openapi.json").status_code == 200


# --- with a real database ---------------------------------------------------

needs_db = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.mark.db
@needs_db
def test_checks_are_saved_and_summarised():
    import psycopg
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute("drop table if exists checks")
    store._ready = False
    client.get("/api/examples/broken-twist")
    client.get("/api/examples/waterbomb")
    recent = client.get("/api/checks/recent").json()
    assert [r["source"] for r in recent] == ["example", "example"]
    assert recent[0]["flat_foldable"] is True          # newest first
    stats = client.get("/api/stats").json()
    assert stats["checks"] == 2 and stats["passed"] == 1
    rules = {r["rule"]: r["times"] for r in stats["rules_broken"]}
    assert rules["kawasaki"] == 1 and rules["maekawa"] == 1


def test_without_database_storage_is_a_no_op(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert store.save("json", {}, {}) is None
    assert client.get("/api/checks/recent").json() == []
