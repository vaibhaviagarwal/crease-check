"""Read and write crease pattern files.

Two formats:

* FOLD (.fold): the JSON standard from the computational origami
  community. Uses vertices_coords, edges_vertices, edges_assignment.
* ORIPA (.cp): plain text from the ORIPA design tool, one line per
  crease: "type x1 y1 x2 y2", type 1 = paper edge, 2 = mountain,
  3 = valley.

Coordinates are y-up (maths convention) throughout the project; the web
page flips them when drawing.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model import Pattern

_FOLD_MAP = {"M": "M", "V": "V", "B": "B", "F": "F", "U": "U", "C": "B", "J": "F"}
_CP_MAP = {1: "B", 2: "M", 3: "V", 4: "F"}


class PatternError(ValueError):
    """Raised for input that is not a usable crease pattern."""


def parse_fold(data: str | bytes | dict) -> Pattern:
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise PatternError(f"Not valid JSON: {e.msg} (line {e.lineno})") from None
    if not isinstance(data, dict):
        raise PatternError("A FOLD file must be a JSON object.")
    coords = data.get("vertices_coords")
    edges = data.get("edges_vertices")
    if not isinstance(coords, list) or not isinstance(edges, list):
        raise PatternError("A FOLD file needs vertices_coords and edges_vertices.")
    assigns = data.get("edges_assignment") or ["U"] * len(edges)
    if len(assigns) != len(edges):
        raise PatternError("edges_assignment must have one entry per edge.")
    try:
        vertices = [(float(c[0]), float(c[1])) for c in coords]
        pairs = [(int(e[0]), int(e[1])) for e in edges]
    except (TypeError, ValueError, IndexError):
        raise PatternError("Coordinates and edges must be lists of numbers.") from None
    kinds = []
    for a in assigns:
        key = str(a).upper()
        if key not in _FOLD_MAP:
            raise PatternError(f"Unknown edge assignment {a!r}.")
        kinds.append(_FOLD_MAP[key])
    try:
        return Pattern(vertices, pairs, kinds)
    except ValueError as e:
        raise PatternError(str(e)) from None


def parse_cp(text: str | bytes) -> Pattern:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    segs = []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise PatternError(f"Line {n}: expected 'type x1 y1 x2 y2'.")
        try:
            t = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])
        except ValueError:
            raise PatternError(f"Line {n}: not numbers.") from None
        # ORIPA's y axis points down; flip to y-up.
        segs.append(((x1, -y1), (x2, -y2), _CP_MAP.get(t, "U")))
    if not segs:
        raise PatternError("The .cp file has no lines.")
    return Pattern.from_segments(segs)


def parse(content: str | bytes, filename: str = "") -> Pattern:
    """Pick the format from the file name, falling back to sniffing."""
    name = filename.lower()
    if name.endswith(".cp"):
        return parse_cp(content)
    if name.endswith(".fold") or name.endswith(".json"):
        return parse_fold(content)
    head = content[:64].lstrip() if isinstance(content, (str, bytes)) else b""
    if head[:1] in ("{", b"{"):
        return parse_fold(content)
    return parse_cp(content)


def load(path: str | Path) -> Pattern:
    path = Path(path)
    return parse(path.read_bytes(), path.name)


def normalize(p: Pattern) -> Pattern:
    """Scale and shift so the drawing fits [0, 1] x [0, 1], keeping shape."""
    x0, y0, x1, y1 = p.bbox()
    s = max(x1 - x0, y1 - y0) or 1.0
    out = Pattern([((x - x0) / s, (y - y0) / s) for x, y in p.vertices],
                  list(p.edges), list(p.assignments))
    out.notes = list(p.notes)
    return out


def to_fold(p: Pattern, title: str = "") -> dict:
    return {
        "file_spec": 1.1,
        "file_creator": "crease-check",
        "file_title": title,
        "frame_classes": ["creasePattern"],
        "vertices_coords": [[round(x, 9), round(y, 9)] for x, y in p.vertices],
        "edges_vertices": [list(e) for e in p.edges],
        "edges_assignment": list(p.assignments),
    }


def to_cp(p: Pattern) -> str:
    code = {"B": 1, "M": 2, "V": 3, "F": 4, "U": 4}
    lines = []
    for (a, b), kind in zip(p.edges, p.assignments):
        (x1, y1), (x2, y2) = p.vertices[a], p.vertices[b]
        lines.append(f"{code[kind]} {x1:.9g} {-y1:.9g} {x2:.9g} {-y2:.9g}")
    return "\n".join(lines) + "\n"
