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

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .graph import cleanup
from .model import Pattern

SIZE = 800          # the paper is warped to SIZE x SIZE pixels
JOIN_PX = 30        # how far an endpoint may move to meet another line
EDGE_PX = 34        # ... or to meet the paper edge
EXTEND_PX = 90      # a dangling end may grow this far to reach a line
MIN_LINE = 36       # shorter detections are noise (e.g. Hough cutting a corner)
SUPPORT = 0.8       # share of a line that must lie on ink
JUNCTION_PX = 16    # line crossings this close are one vertex
STUB = 0.05         # dangling bits shorter than this (fraction of side) go


class PhotoError(ValueError):
    pass


@dataclass
class PhotoResult:
    pattern: Pattern
    stats: dict = field(default_factory=dict)
    lines: list = field(default_factory=list)   # detected creases, unit coords, for the overlay
    sheet: np.ndarray | None = None              # the straightened photo


# --- 1. the paper ------------------------------------------------------------

def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PhotoError("That file is not an image I can read.")
    h, w = img.shape[:2]
    scale = 1600 / max(h, w)
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
    dst = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR)


# --- 2-3. ink and skeleton ----------------------------------------------------

def ink_masks(sheet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h <= 10) | (h >= 165)) & (s >= 80) & (v >= 50)).astype(np.uint8) * 255
    blue = ((h >= 95) & (h <= 135) & (s >= 60) & (v >= 35)).astype(np.uint8) * 255
    band = 8   # ignore any sliver of desk left at the edges after the warp
    for m in (red, blue):
        m[:band], m[-band:], m[:, :band], m[:, -band:] = 0, 0, 0, 0
    k = np.ones((3, 3), np.uint8)
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
    found = cv2.HoughLinesP(skel, 1, np.pi / 360, threshold=22, minLineLength=18, maxLineGap=6)
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


def merge(pieces, kind: str, ang_tol: float = 4.0, dist_tol: float = 7.0, gap: float = 60.0,
          ink: np.ndarray | None = None) -> list[Line]:
    """Group pieces lying on one line, fit the line, and join pieces along it
    whose gap is short or is visibly inked (Hough can miss a wobbly stretch)."""
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


def dedupe(lines: list[Line], offset_px: float = 12.0, ang: float = 2.0) -> list[Line]:
    """Fold near-duplicate parallel lines of one colour into the longer one.
    A shaky stroke can come out as two lines a few pixels apart."""
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
            if ta[0] > k.t1 + JOIN_PX or ta[1] < k.t0 - JOIN_PX:
                continue
            k.t0, k.t1 = min(k.t0, ta[0]), max(k.t1, ta[1])
            absorbed = True
            break
        if not absorbed:
            kept.append(l)
    return kept


def _border_lines() -> list[Line]:
    S = float(SIZE)
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

    Each endpoint moves to the nearest junction within JOIN_PX (another
    line) or EDGE_PX (the paper edge, measured straight out to it). A
    crease cannot just stop in the middle of the paper, so an end with no
    junction that close is extended forward up to EXTEND_PX to the first
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
            slack = EDGE_PX if m.border else JOIN_PX
            if not (m.t0 - slack <= tm <= m.t1 + slack):
                continue
            sin = abs(l.ux * m.uy - l.uy * m.ux)
            for end, cur, out in ((0, l.t0, -1.0), (1, l.t1, 1.0)):
                d = abs(tl - cur)
                gap = d * sin if m.border else d       # perpendicular gap to the edge
                reach = EDGE_PX if m.border else JOIN_PX
                if gap <= reach and (end not in near or d < near[end][0]):
                    near[end] = (d, tl)
                forward = (tl - cur) * out > 0
                if forward and d <= EXTEND_PX and (end not in ahead or d < ahead[end][0]):
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
            if not (0 <= xi < SIZE and 0 <= yi < SIZE):
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
        cv2.line(covered, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 255, 13)
    rest = cv2.bitwise_and(mask, cv2.bitwise_not(covered))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 80:
            continue
        ys, xs = np.nonzero(labels == k)
        P = np.column_stack([xs, ys]).astype(float)
        c = P.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((P - c).T))
        u = evecs[:, -1]
        proj = (P - c) @ u
        if proj.max() - proj.min() < MIN_LINE or math.sqrt(max(evals[0], 0)) > 4.0:
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
        if 0 <= xi < SIZE and 0 <= yi < SIZE and mask[yi, xi]:
            hits += 1
    return hits / n


def _to_unit(x: float, y: float) -> tuple[float, float]:
    return (min(max(x / SIZE, 0.0), 1.0), min(max(1.0 - y / SIZE, 0.0), 1.0))


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
            short = math.hypot(x2 - x1, y2 - y1) < STUB
            dangling = (deg[i] == 1 and i not in boundary) or (deg[j] == 1 and j not in boundary)
            keep.append(not (a != "B" and short and dangling))
        if all(keep):
            return p
        p = Pattern(p.vertices, [e for e, k in zip(p.edges, keep) if k],
                    [a for a, k in zip(p.assignments, keep) if k], p.notes)


def build_graph(lines: list[Line]) -> Pattern:
    """Turn detected lines into vertices and edges.

    1. Every place two lines meet (within their inked extent) is a
       junction candidate; candidates closer than JUNCTION_PX are one
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
            s = JUNCTION_PX
            if L[i].t0 - s <= ti <= L[i].t1 + s and L[j].t0 - s <= tj <= L[j].t1 + s:
                x, y = L[i].at(ti)
                cand.append((x, y, i, j))

    parent = list(range(len(cand)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            if math.hypot(cand[a][0] - cand[b][0], cand[a][1] - cand[b][1]) <= JUNCTION_PX:
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
            if not stops or min(abs(t - s) for s, _ in stops) > JUNCTION_PX:
                stops.append((t, vertex(base + u * t)))
        stops.sort()
        for (_, a), (_, b) in zip(stops, stops[1:]):
            if a != b:
                edges.append((a, b))
                kinds.append(l.kind)
    unit = [(min(max(x / SIZE, 0.0), 1.0), min(max(1.0 - y / SIZE, 0.0), 1.0)) for x, y in verts]
    return Pattern(unit, edges, kinds)


def read_photo(data: bytes, snap: float | None = 22.5) -> PhotoResult:
    img = decode(data)
    corners = find_sheet(img)
    sheet = straighten(img, corners)
    red, blue = ink_masks(sheet)
    stats = {
        "paper_found": corners is not None,
        "ink_pixels": {"mountain": int((red > 0).sum()), "valley": int((blue > 0).sum())},
    }
    lines: list[Line] = []
    raw = 0
    for mask, kind in ((red, "M"), (blue, "V")):
        pieces = hough(thin(mask))
        raw += len(pieces)
        wide = cv2.dilate(mask, np.ones((7, 7), np.uint8))
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
        found = [l for l in found if l.t1 - l.t0 >= MIN_LINE and ink_support(l, wide) >= SUPPORT]
        lines += found
    stats["hough_pieces"] = raw
    if not lines:
        raise PhotoError("No red or blue lines found. Draw mountains in red marker and valleys in blue.")
    if snap:
        snap_angles(lines, snap)
    lines = dedupe(lines)
    join(lines)
    # Joining can shrink a line to a stub across a corner where two strokes
    # of one colour meet (Hough "cutting the corner"). Real creases are
    # never that short in a hand drawing.
    lines = [l for l in lines if l.t1 - l.t0 >= MIN_LINE]

    pat = _prune_stubs(cleanup(build_graph(lines), tol=1e-6, add_border=False))
    pat.notes = []
    stats["creases"] = sum(1 for a in pat.assignments if a in ("M", "V"))
    stats["lines"] = len(lines)
    overlay = [{"a": _to_unit(*l.at(l.t0)), "b": _to_unit(*l.at(l.t1)), "kind": l.kind} for l in lines]
    return PhotoResult(pat, stats, overlay, sheet)
