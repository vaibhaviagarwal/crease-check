"""Read a crease pattern from a phone photo of a marker drawing.

Red marker = mountain, blue marker = valley, on a square of light paper.
Classical computer vision, no neural network:

1. find the paper and straighten it (perspective warp to a flat square),
2. pick out red and blue ink by colour (HSV thresholds),
3. thin each stroke to a 1-pixel skeleton (Zhang-Suen),
4. find straight pieces in the skeleton (probabilistic Hough transform),
5. merge pieces that belong to one crease,
6. snap angles to the 22.5-degree grid most origami uses (optional),
7. make creases meet: endpoints move to nearby junctions and to the
   paper edge, overshoot stubs are trimmed, and each vertex is placed at
   the least-squares meeting point of its lines.

The output is an ordinary Pattern, checked like any uploaded file.
bench/vision_benchmark.py scores this against synthetic photos.
"""

from __future__ import annotations

import contextvars
import dataclasses
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .graph import cleanup
from .model import Pattern

@dataclass(frozen=True)
class Profile:
    """Tuning for one kind of input. All distances are in pixels of the
    straightened sheet (size x size)."""
    name: str
    size: int            # the paper is warped to size x size pixels
    join: float          # how far an endpoint may move to meet another line
    edge: float          # ... or to meet the paper edge
    extend: float        # a dangling end may grow this far to reach a line
    min_line: float      # shorter detections are noise
    junction: float      # line crossings this close are one vertex
    stub: float          # dangling bits shorter than this (fraction of side) go
    open_px: int         # speck removal before thinning (0 = none: keeps thin lines)
    widen: int           # tolerance band around ink, for support checks
    cover: int           # how wide a found line "explains" nearby ink
    gap: float           # collinear pieces this far apart are one line
    offset: float        # lines closer than this side by side are one
    hough: tuple[int, int, int]   # threshold, min length, max gap
    support: float = 0.8          # share of a line that must lie on ink


# Hand-drawn marker on paper, photographed: thick, wobbly strokes.
MARKER = Profile("marker", 800, join=30, edge=34, extend=90, min_line=36, junction=16, stub=0.05,
                 open_px=3, widen=7, cover=13, gap=60, offset=12, hough=(22, 18, 6))
# A screenshot or export from design software: thin, exact lines, often a
# dense grid, so every tolerance has to be far tighter than for marker.
DIGITAL = Profile("digital", 1600, join=10, edge=14, extend=24, min_line=14, junction=7, stub=0.012,
                  open_px=0, widen=5, cover=9, gap=10, offset=4, hough=(14, 12, 3))

_profile: contextvars.ContextVar[Profile] = contextvars.ContextVar("profile", default=MARKER)


def _p() -> Profile:
    return _profile.get()


class PhotoError(ValueError):
    pass


@dataclass
class PhotoResult:
    pattern: Pattern
    stats: dict = field(default_factory=dict)
    lines: list = field(default_factory=list)   # detected creases, unit coords, for the overlay
    sheet: np.ndarray | None = None              # the straightened photo


# --- 1. the paper ------------------------------------------------------------

def decode(data: bytes, keep_detail: bool = False) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PhotoError("That file is not an image I can read.")
    h, w = img.shape[:2]
    scale = (2400 if keep_detail else 1600) / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _quad_from_mask(mask: np.ndarray, min_share: float = 0.12):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_share * mask.shape[0] * mask.shape[1]:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        quad = cv2.approxPolyDP(hull, eps * peri, True)
        if len(quad) == 4:
            return order_corners(quad)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def _paper_mask(score: np.ndarray, img_shape) -> np.ndarray:
    score = cv2.GaussianBlur(score, (7, 7), 0)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(0.03 * max(img_shape[:2])))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # fill over ink
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def find_sheet(img: np.ndarray) -> np.ndarray | None:
    """Corners of the paper, or None.

    Paper is bright and unsaturated. Two readings of that: one global
    threshold, and one after dividing out the lighting (each pixel against
    its wide neighbourhood), which survives a shadow across the sheet. A
    shadow makes the first reading clip the paper, so the second wins when
    it finds clearly more paper.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    raw = np.clip(hsv[..., 2].astype(np.float32) - hsv[..., 1], 1, 255)
    small = cv2.resize(raw, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.2 * max(small.shape[:2]))
    field = cv2.resize(small, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
    flat = cv2.normalize(raw / np.maximum(field, 1.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    plain = _quad_from_mask(_paper_mask(raw.astype(np.uint8), img.shape))
    lit = _quad_from_mask(_paper_mask(flat, img.shape))
    if plain is None or lit is None:
        return plain if lit is None else lit
    area = lambda q: cv2.contourArea(q.astype(np.float32))
    # The plain reading has crisper corners; only distrust it when the
    # lighting-corrected one finds clearly more paper (a shadow clipped it).
    return lit if area(lit) > 1.08 * area(plain) else plain


def straighten(img: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    if corners is None:
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    dst = np.array([[0, 0], [_p().size, 0], [_p().size, _p().size], [0, _p().size]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (_p().size, _p().size), flags=cv2.INTER_LINEAR)


# --- 2-3. ink and skeleton ----------------------------------------------------

def ink_masks(sheet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # On a clean white background (digital) paler, anti-aliased colour still
    # means ink; on paper under room light it is more often shadow or tint.
    red_s, blue_s = (80, 60) if _p().name == "marker" else (40, 35)
    red = (((h <= 10) | (h >= 165)) & (s >= red_s) & (v >= 50)).astype(np.uint8) * 255
    blue = ((h >= 95) & (h <= 135) & (s >= blue_s) & (v >= 35)).astype(np.uint8) * 255
    band = 8 if _p().name == "marker" else 3   # ignore any sliver of desk left after the warp
    for m in (red, blue):
        m[:band], m[-band:], m[:, :band], m[:, -band:] = 0, 0, 0, 0
    if _p().open_px:   # marker: clear specks; thin digital lines would vanish too
        k = np.ones((_p().open_px, _p().open_px), np.uint8)
        red = cv2.morphologyEx(cv2.morphologyEx(red, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        blue = cv2.morphologyEx(cv2.morphologyEx(blue, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return red, blue


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, vectorised with numpy."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P2, P3, P4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            P5, P6, P7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            P8, P9 = img[1:-1, :-2], img[:-2, :-2]
            C = img[1:-1, 1:-1]
            ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(x.astype(np.int16) for x in ring[:8])
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = (P2 * P4 * P6) == 0, (P4 * P6 * P8) == 0
            else:
                c1, c2 = (P2 * P4 * P8) == 0, (P2 * P6 * P8) == 0
            kill = (C == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if kill.any():
                C[kill] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1] * 255


# --- 4-5. straight pieces and merging ----------------------------------------

def hough(skel: np.ndarray) -> list[tuple[float, float, float, float]]:
    thr, length, gap = _p().hough
    found = cv2.HoughLinesP(skel, 1, np.pi / 360, threshold=thr, minLineLength=length, maxLineGap=gap)
    # OpenCV 4 returns shape (n, 1, 4), OpenCV 5 (n, 4).
    return [] if found is None else [tuple(map(float, l)) for l in np.asarray(found).reshape(-1, 4)]


@dataclass
class Line:
    """An infinite line (point + unit direction) and the stretch t0..t1 inked."""
    px: float
    py: float
    ux: float
    uy: float
    t0: float
    t1: float
    kind: str
    border: bool = False

    def at(self, t: float) -> tuple[float, float]:
        return (self.px + t * self.ux, self.py + t * self.uy)

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.uy, self.ux)) % 180.0


def merge(pieces, kind: str, ang_tol: float = 4.0, dist_tol: float | None = None,
          gap: float | None = None, ink: np.ndarray | None = None) -> list[Line]:
    """Group pieces lying on one line, fit the line, and join pieces along it
    whose gap is short or is visibly inked (Hough can miss a wobbly stretch)."""
    dist_tol = dist_tol if dist_tol is not None else (7.0 if _p().name == "marker" else 3.0)
    gap = gap if gap is not None else _p().gap
    pieces = sorted(pieces, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
    groups: list[list] = []
    for s in pieces:
        x1, y1, x2, y2 = s
        th = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        placed = False
        for g in groups:
            gth, (ax, ay), (nx, ny) = g[0], g[1], g[2]
            dth = abs(th - gth)
            dth = min(dth, 180 - dth)
            if dth > ang_tol:
                continue
            if all(abs((x - ax) * nx + (y - ay) * ny) <= dist_tol for x, y in ((x1, y1), (x2, y2))):
                g[3].append(s)
                placed = True
                break
        if not placed:
            r = math.radians(th)
            groups.append([th, ((x1 + x2) / 2, (y1 + y2) / 2), (-math.sin(r), math.cos(r)), [s]])

    out: list[Line] = []
    for _, _, _, members in groups:
        pts, w = [], []
        for x1, y1, x2, y2 in members:
            L = math.hypot(x2 - x1, y2 - y1)
            pts += [(x1, y1), (x2, y2)]
            w += [L, L]
        P = np.array(pts)
        W = np.array(w)[:, None]
        c = (P * W).sum(0) / W.sum()
        cov = ((P - c) * W).T @ (P - c)
        evals, evecs = np.linalg.eigh(cov)
        ux, uy = evecs[:, -1]
        spans = sorted(
            tuple(sorted(((x1 - c[0]) * ux + (y1 - c[1]) * uy, (x2 - c[0]) * ux + (y2 - c[1]) * uy)))
            for x1, y1, x2, y2 in members)
        cur = list(spans[0])
        for a, b in spans[1:]:
            bridged = False
            if ink is not None and a - cur[1] <= 4 * gap:
                probe = Line(c[0], c[1], ux, uy, cur[1], a, kind)
                bridged = ink_support(probe, ink) >= 0.7
            if a <= cur[1] + gap or bridged:
                cur[1] = max(cur[1], b)
            else:
                out.append(Line(c[0], c[1], ux, uy, cur[0], cur[1], kind))
                cur = [a, b]
        out.append(Line(c[0], c[1], ux, uy, cur[0], cur[1], kind))
    return out


# --- 6-7. snapping and joining -------------------------------------------------

def snap_angles(lines: list[Line], grid: float = 22.5, within: float = 5.0) -> None:
    for l in lines:
        target = round(l.angle / grid) * grid
        if abs(target - l.angle) <= within:
            mx, my = l.at((l.t0 + l.t1) / 2)
            half = (l.t1 - l.t0) / 2
            r = math.radians(target)
            l.px, l.py, l.ux, l.uy = mx, my, math.cos(r), math.sin(r)
            l.t0, l.t1 = -half, half


def dedupe(lines: list[Line], offset_px: float | None = None, ang: float = 2.0) -> list[Line]:
    """Fold near-duplicate parallel lines of one colour into the longer one.
    A shaky stroke can come out as two lines a few pixels apart."""
    offset_px = offset_px if offset_px is not None else _p().offset
    lines = sorted(lines, key=lambda l: -(l.t1 - l.t0))
    kept: list[Line] = []
    for l in lines:
        absorbed = False
        for k in kept:
            if k.kind != l.kind:
                continue
            d = abs(l.angle - k.angle)
            if min(d, 180 - d) > ang:
                continue
            a, b = l.at(l.t0), l.at(l.t1)
            off = [abs((x - k.px) * -k.uy + (y - k.py) * k.ux) for x, y in (a, b)]
            if max(off) > offset_px:
                continue
            ta = sorted(((a[0] - k.px) * k.ux + (a[1] - k.py) * k.uy,
                         (b[0] - k.px) * k.ux + (b[1] - k.py) * k.uy))
            if ta[0] > k.t1 + _p().join or ta[1] < k.t0 - _p().join:
                continue
            k.t0, k.t1 = min(k.t0, ta[0]), max(k.t1, ta[1])
            absorbed = True
            break
        if not absorbed:
            kept.append(l)
    return kept


def _border_lines() -> list[Line]:
    S = float(_p().size)
    return [Line(0, 0, 1, 0, 0, S, "B", True), Line(S, 0, 0, 1, 0, S, "B", True),
            Line(S, S, -1, 0, 0, S, "B", True), Line(0, S, 0, -1, 0, S, "B", True)]


def _intersect(a: Line, b: Line):
    den = a.ux * b.uy - a.uy * b.ux
    if abs(den) < math.sin(math.radians(6)):
        return None
    dx, dy = b.px - a.px, b.py - a.py
    ta = (dx * b.uy - dy * b.ux) / den
    tb = (dx * a.uy - dy * a.ux) / den
    return ta, tb


def join(lines: list[Line]) -> None:
    """Make creases meet.

    Each endpoint moves to the nearest junction within the join distance (another
    line) or the edge distance (the paper edge, measured straight out to it). A
    crease cannot just stop in the middle of the paper, so an end with no
    junction that close is extended forward a little further to the first
    line it would hit.
    """
    allx = lines + _border_lines()
    for l in lines:
        near: dict[int, tuple[float, float]] = {}
        ahead: dict[int, tuple[float, float]] = {}
        for m in allx:
            if m is l:
                continue
            hit = _intersect(l, m)
            if hit is None:
                continue
            tl, tm = hit
            slack = _p().edge if m.border else _p().join
            if not (m.t0 - slack <= tm <= m.t1 + slack):
                continue
            sin = abs(l.ux * m.uy - l.uy * m.ux)
            for end, cur, out in ((0, l.t0, -1.0), (1, l.t1, 1.0)):
                d = abs(tl - cur)
                gap = d * sin if m.border else d       # perpendicular gap to the edge
                reach = _p().edge if m.border else _p().join
                if gap <= reach and (end not in near or d < near[end][0]):
                    near[end] = (d, tl)
                forward = (tl - cur) * out > 0
                if forward and d <= _p().extend and (end not in ahead or d < ahead[end][0]):
                    ahead[end] = (d, tl)
        for end in (0, 1):
            pick = near.get(end) or ahead.get(end)
            if pick:
                if end == 0:
                    l.t0 = pick[1]
                else:
                    l.t1 = pick[1]


def extend_along_ink(l: Line, mask: np.ndarray, step: float = 2.0, max_miss: int = 4) -> None:
    """Grow both ends while the ink continues. Hough often returns only
    part of a short wobbly stroke."""
    for sign in (1, -1):
        t = l.t1 if sign > 0 else l.t0
        last, miss = t, 0
        while miss <= max_miss:
            t += sign * step
            x, y = l.at(t)
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < _p().size and 0 <= yi < _p().size):
                break
            if mask[yi, xi]:
                last, miss = t, 0
            else:
                miss += 1
        if sign > 0:
            l.t1 = last
        else:
            l.t0 = last


def leftover_strokes(mask: np.ndarray, lines: list[Line], kind: str) -> list[Line]:
    """Ink that no line explains yet, if it is long and thin, is a stroke
    Hough missed (short wobbly ones especially). Fit a line to each."""
    covered = np.zeros_like(mask)
    for l in lines:
        a, b = l.at(l.t0), l.at(l.t1)
        cv2.line(covered, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 255, _p().cover)
    rest = cv2.bitwise_and(mask, cv2.bitwise_not(covered))
    if _p().open_px:
        rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < (80 if _p().name == "marker" else 25):
            continue
        ys, xs = np.nonzero(labels == k)
        P = np.column_stack([xs, ys]).astype(float)
        c = P.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((P - c).T))
        u = evecs[:, -1]
        proj = (P - c) @ u
        if proj.max() - proj.min() < _p().min_line or math.sqrt(max(evals[0], 0)) > (4.0 if _p().name == "marker" else 2.0):
            continue
        out.append(Line(c[0], c[1], u[0], u[1], float(proj.min()), float(proj.max()), kind))
    return out


def ink_support(l: Line, mask: np.ndarray) -> float:
    """Share of points along the line that sit on (slightly widened) ink."""
    n = max(8, int(l.t1 - l.t0) // 4)
    hits = 0
    for k in range(n):
        x, y = l.at(l.t0 + (l.t1 - l.t0) * (k + 0.5) / n)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < _p().size and 0 <= yi < _p().size and mask[yi, xi]:
            hits += 1
    return hits / n


def find_grid(lines: list[Line], tol_px: float = 3.0, need: float = 0.9) -> int | None:
    """Most software-designed patterns (box pleating especially) put every
    vertex on an N x N grid. Find the smallest N that nearly every line end
    sits on. A coarse grid that is not really there fits only a small
    share of points, so the first N to fit is the real one."""
    S = _p().size
    coords = np.array([c for l in lines for t_ in (l.t0, l.t1) for c in l.at(t_)])
    if coords.size < 40:
        return None
    for n in range(2, 129):
        unit = S / n
        tol = min(tol_px, 0.12 * unit)
        off = np.abs(coords / unit - np.round(coords / unit)) * unit
        if (off <= tol).mean() >= need:
            return n if n >= 4 else None
    return None


def snap_to_grid(lines: list[Line], n: int) -> list[Line]:
    S = _p().size
    unit = S / n
    out = []
    for l in lines:
        (ax, ay), (bx, by) = l.at(l.t0), l.at(l.t1)
        ax, ay, bx, by = (round(v / unit) * unit for v in (ax, ay, bx, by))
        L = math.hypot(bx - ax, by - ay)
        if L < 1e-6:
            continue
        out.append(Line(ax, ay, (bx - ax) / L, (by - ay) / L, 0.0, L, l.kind))
    return out


def read_grid(red: np.ndarray, blue: np.ndarray, n: int) -> tuple[Pattern, list[dict], float]:
    """Read a grid-based pattern by testing every step between neighbouring
    grid points (half-steps, so diagonals that cross mid-square are seen)
    in the four grid directions: is it inked red, blue, or not at all?

    Returns the pattern, the overlay lines, and the share of all ink the
    steps explain. A low share means the pattern has creases off the grid
    (22.5-degree designs, say), and the line-finding result should be kept.
    """
    S = _p().size
    m = 2 * n
    unit = S / m
    k = np.ones((3, 3), np.uint8)
    R, B = cv2.dilate(red, k) > 0, cv2.dilate(blue, k) > 0
    ts = np.linspace(0.2, 0.8, 7)
    steps: dict[tuple[int, int], dict[tuple[int, int], str]] = {}
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        found = {}
        i0, i1 = 0, m - dx
        j0, j1 = (0, m - dy) if dy >= 0 else (1, m)
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        xs = (I[..., None] + dx * ts) * unit
        ys = (J[..., None] + dy * ts) * unit
        xi = np.clip(np.round(xs).astype(int), 0, S - 1)
        yi = np.clip(np.round(ys).astype(int), 0, S - 1)
        r = R[yi, xi].mean(-1)
        b = B[yi, xi].mean(-1)
        for (a, c) in zip(*np.nonzero((r >= 0.85) | (b >= 0.85))):
            i, j = int(I[a, c]), int(J[a, c])
            found[(i, j)] = "M" if r[a, c] >= b[a, c] else "V"
        steps[(dx, dy)] = found

    # Join consecutive steps of one colour in one direction into creases.
    segs, overlay = [], []
    for (dx, dy), found in steps.items():
        seen = set()
        for (i, j), kind in found.items():
            if (i, j) in seen or found.get((i - dx, j - dy)) == kind:
                continue
            a, b2 = (i, j), (i, j)
            while found.get(b2) == kind:
                seen.add(b2)
                b2 = (b2[0] + dx, b2[1] + dy)
            pa = (a[0] / m, 1 - a[1] / m)
            pb = (b2[0] / m, 1 - b2[1] / m)
            segs.append((pa, pb, kind))
            overlay.append({"a": pa, "b": pb, "kind": kind})
    border = [((0.0, 0.0), (1.0, 0.0), "B"), ((1.0, 0.0), (1.0, 1.0), "B"),
              ((1.0, 1.0), (0.0, 1.0), "B"), ((0.0, 1.0), (0.0, 0.0), "B")]
    pat = cleanup(Pattern.from_segments(segs + border), tol=1e-9, add_border=False)

    # How much of the ink do these creases account for?
    drawn = np.zeros((S, S), np.uint8)
    for (x1, y1), (x2, y2), _ in segs:
        cv2.line(drawn, (int(x1 * S), int((1 - y1) * S)), (int(x2 * S), int((1 - y2) * S)), 255, _p().cover)
    ink = (red > 0) | (blue > 0)
    explained = float((ink & (drawn > 0)).sum() / max(1, ink.sum()))
    return pat, overlay, explained


def _to_unit(x: float, y: float) -> tuple[float, float]:
    return (min(max(x / _p().size, 0.0), 1.0), min(max(1.0 - y / _p().size, 0.0), 1.0))


def _prune_stubs(p: Pattern) -> Pattern:
    while True:
        deg = [0] * len(p.vertices)
        for (i, j), a in zip(p.edges, p.assignments):
            deg[i] += 1
            deg[j] += 1
        boundary = {v for (i, j), a in zip(p.edges, p.assignments) if a == "B" for v in (i, j)}
        keep = []
        for k, ((i, j), a) in enumerate(zip(p.edges, p.assignments)):
            (x1, y1), (x2, y2) = p.vertices[i], p.vertices[j]
            short = math.hypot(x2 - x1, y2 - y1) < _p().stub
            dangling = (deg[i] == 1 and i not in boundary) or (deg[j] == 1 and j not in boundary)
            keep.append(not (a != "B" and short and dangling))
        if all(keep):
            return p
        p = Pattern(p.vertices, [e for e, k in zip(p.edges, keep) if k],
                    [a for a, k in zip(p.assignments, keep) if k], p.notes)


def build_graph(lines: list[Line]) -> Pattern:
    """Turn detected lines into vertices and edges.

    1. Every place two lines meet (within their inked extent) is a
       junction candidate; candidates closer than the junction distance are one
       vertex (union-find).
    2. Hand-drawn lines meant to meet at one point miss each other by a
       pixel or two. Alternate: put each vertex at the least-squares
       meeting point of its lines, then shift each line sideways (keeping
       its snapped angle) towards its vertices. A few rounds make the
       drawing consistent, so angles at vertices are the snapped angles.
    3. Each line becomes edges between consecutive vertices along it.
    """
    L = lines + _border_lines()
    cand = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            hit = _intersect(L[i], L[j])
            if hit is None:
                continue
            ti, tj = hit
            s = _p().junction
            if L[i].t0 - s <= ti <= L[i].t1 + s and L[j].t0 - s <= tj <= L[j].t1 + s:
                x, y = L[i].at(ti)
                cand.append((x, y, i, j))

    parent = list(range(len(cand)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Only candidates in neighbouring grid cells can be close enough to merge.
    r = _p().junction
    cells: dict[tuple[int, int], list[int]] = {}
    for k, (x, y, _, _) in enumerate(cand):
        cells.setdefault((int(x // r), int(y // r)), []).append(k)
    for (cx, cy), members in cells.items():
        near = [m for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m in cells.get((cx + dx, cy + dy), [])]
        for a in members:
            for b in near:
                if b > a and math.hypot(cand[a][0] - cand[b][0], cand[a][1] - cand[b][1]) <= r:
                    parent[find(a)] = find(b)
    groups: dict[int, set[int]] = {}
    seed: dict[int, list] = {}
    for k, (x, y, i, j) in enumerate(cand):
        r = find(k)
        groups.setdefault(r, set()).update((i, j))
        seed.setdefault(r, []).append((x, y))
    roots = list(groups)
    normal = np.array([(-l.uy, l.ux) for l in L])
    offset = np.array([normal[k] @ (L[k].px, L[k].py) for k in range(len(L))])
    pos = {r: np.mean(seed[r], axis=0) for r in roots}

    for _ in range(12):
        for r in roots:
            ids = sorted(groups[r])
            A = normal[ids]
            if np.linalg.cond(A.T @ A) < 1e4:
                pos[r] = np.linalg.lstsq(A, offset[ids], rcond=None)[0]
        moved = 0.0
        for k, l in enumerate(L):
            if l.border:
                continue
            mine = [r for r in roots if k in groups[r]]
            if not mine:
                continue
            res = float(np.mean([normal[k] @ pos[r] - offset[k] for r in mine]))
            offset[k] += res
            moved = max(moved, abs(res))
        if moved < 1e-3:
            break

    vid: dict[int, int] = {}
    verts: list[tuple[float, float]] = []

    def vertex(xy) -> int:
        verts.append((float(xy[0]), float(xy[1])))
        return len(verts) - 1

    for r in roots:
        vid[r] = vertex(pos[r])
    edges, kinds = [], []
    for k, l in enumerate(L):
        nrm = normal[k]
        base = nrm * offset[k]           # a point on the (shifted) line
        u = np.array((l.ux, l.uy))
        stops = []
        for r in roots:
            if k in groups[r]:
                stops.append((float(u @ (pos[r] - base)), vid[r]))
        for t_end in (l.t0, l.t1):
            end = np.array(l.at(t_end))
            t = float(u @ (end - base))
            if not stops or min(abs(t - s) for s, _ in stops) > _p().junction:
                stops.append((t, vertex(base + u * t)))
        stops.sort()
        for (_, a), (_, b) in zip(stops, stops[1:]):
            if a != b:
                edges.append((a, b))
                kinds.append(l.kind)
    unit = [(min(max(x / _p().size, 0.0), 1.0), min(max(1.0 - y / _p().size, 0.0), 1.0)) for x, y in verts]
    return Pattern(unit, edges, kinds)


def stroke_width(mask: np.ndarray) -> float:
    """Typical stroke width in pixels: twice the distance from a stroke's
    centre line to its edge, measured along the skeleton."""
    if not mask.any():
        return 0.0
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    centre = dist[thin(mask) > 0]
    return float(2 * np.median(centre)) if centre.size else 0.0


def choose_profile(img: np.ndarray, corners) -> Profile:
    """Measured on an 800-px sheet, lines exported from design software
    come out about 3-4 px wide and marker strokes 9-12 px, so a cut at 6 px
    separates them with room to spare. Thin lines get the digital profile."""
    # Digital masks (no speck removal, so thin lines survive to be measured),
    # at the cheaper 800-px size.
    token = _profile.set(dataclasses.replace(DIGITAL, size=800))
    try:
        red, blue = ink_masks(straighten(img, corners))
    finally:
        _profile.reset(token)
    width = stroke_width(cv2.bitwise_or(red, blue))
    return DIGITAL if 0 < width <= 6 else MARKER


def read_photo(data: bytes, snap: float | None = 22.5, profile: Profile | None = None) -> PhotoResult:
    img = decode(data, keep_detail=True)
    corners = find_sheet(img)
    prof = profile or choose_profile(img, corners)
    token = _profile.set(prof)
    try:
        if prof is DIGITAL:
            corners = tighten_to_ink(img, corners)
        return _read(img, corners, snap)
    finally:
        _profile.reset(token)


def tighten_to_ink(img: np.ndarray, corners) -> np.ndarray:
    """Software draws the paper's edge as a frame a few pixels inside the
    image, and creases stop at that frame. Use the outer extent of the
    creases themselves as the paper's edge."""
    if corners is None:
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    sheet = straighten(img, corners)
    red, blue = ink_masks(sheet)
    ys, xs = np.nonzero(cv2.bitwise_or(red, blue))
    if xs.size < 100:
        return corners
    # Robust extent: ignore the odd stray pixel.
    x0, x1 = np.percentile(xs, [0.05, 99.95])
    y0, y1 = np.percentile(ys, [0.05, 99.95])
    S = _p().size
    if (x1 - x0) < 0.5 * S or (y1 - y0) < 0.5 * S:
        return corners
    dst = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    back = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
    box = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
    return cv2.perspectiveTransform(box, back)[0]


def _read(img: np.ndarray, corners, snap: float | None) -> PhotoResult:
    sheet = straighten(img, corners)
    red, blue = ink_masks(sheet)
    stats = {
        "paper_found": corners is not None,
        "kind": "photo of a drawing" if _p().name == "marker" else "digital crease pattern",
        "ink_pixels": {"mountain": int((red > 0).sum()), "valley": int((blue > 0).sum())},
    }
    lines: list[Line] = []
    raw = 0
    for mask, kind in ((red, "M"), (blue, "V")):
        pieces = hough(thin(mask))
        raw += len(pieces)
        wide = cv2.dilate(mask, np.ones((_p().widen, _p().widen), np.uint8))
        found = merge(pieces, kind, ink=wide)
        # A second pass: the first fit of a wobbly line can leave two
        # nearly identical lines side by side. Refit and merge again.
        for _ in range(3):
            if snap:
                snap_angles(found, snap)
            found = merge([(*l.at(l.t0), *l.at(l.t1)) for l in found], kind,
                          ang_tol=6.0, ink=wide)
        found += leftover_strokes(mask, found, kind)
        if snap:
            snap_angles(found, snap)
        found = merge([(*l.at(l.t0), *l.at(l.t1)) for l in found], kind, ang_tol=6.0, ink=wide)
        if snap:
            snap_angles(found, snap)
        for l in found:
            extend_along_ink(l, mask)
        found = [l for l in found if l.t1 - l.t0 >= _p().min_line and ink_support(l, wide) >= _p().support]
        lines += found
    stats["hough_pieces"] = raw
    if not lines:
        raise PhotoError("No red or blue lines found. Mountain folds should be red and valley folds blue.")
    if snap:
        snap_angles(lines, snap)
    lines = dedupe(lines)
    join(lines)
    # Joining can shrink a line to a stub across a corner where two strokes
    # of one colour meet (Hough "cutting the corner"). Real creases are
    # never that short in a hand drawing.
    lines = [l for l in lines if l.t1 - l.t0 >= _p().min_line]
    if _p() is DIGITAL:
        grid = find_grid(lines)
        if grid:
            lines = snap_to_grid(lines, 2 * grid)   # half steps: diagonals cross mid-square
            stats["grid"] = grid

    pat = _prune_stubs(cleanup(build_graph(lines), tol=1e-6, add_border=False))
    overlay = [{"a": _to_unit(*l.at(l.t0)), "b": _to_unit(*l.at(l.t1)), "kind": l.kind} for l in lines]
    if stats.get("grid"):
        gpat, goverlay, explained = read_grid(red, blue, stats["grid"])
        stats["grid_explains"] = round(explained, 3)
        if explained >= 0.97:
            pat, overlay = gpat, goverlay
            stats["method"] = "grid"
    pat.notes = []
    stats["creases"] = sum(1 for a in pat.assignments if a in ("M", "V"))
    stats["lines"] = len(overlay)
    stats.setdefault("method", "lines")
    return PhotoResult(pat, stats, overlay, sheet)
