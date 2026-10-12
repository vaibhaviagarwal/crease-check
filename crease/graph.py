"""Turn a messy drawing into a clean planar graph.

Files and photos both arrive as loose segments: lines cross without a
vertex where they cross, endpoints that should meet miss by a hair, the
same crease is drawn twice. The checks need a proper graph, where creases
only meet at shared vertices. cleanup() builds that in four steps:

1. find every pair of segments that touch or cross,
2. split each segment at those points,
3. merge vertices closer than a tolerance,
4. drop duplicate edges, resolving clashing assignments.

Step 1 is the expensive one. Comparing every segment with every other is
O(n^2); instead each segment is registered in the cells of a uniform grid
it passes through, and only segments sharing a cell are compared. A
pattern of 2,000 creases then needs thousands of comparisons rather than
two million. tests/test_graph.py checks the grid finds exactly what the
brute-force version finds.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .model import Pattern, Point

# Priority when the same edge appears twice with different assignments.
# A boundary edge stays a boundary; a real fold beats "flat"/"unassigned".
_RANK = {"B": 3, "M": 2, "V": 2, "U": 1, "F": 0}


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _grid_cells(p: Point, q: Point, h: float) -> set[tuple[int, int]]:
    """Grid cells the segment passes through (Amanatides-Woo traversal),
    each widened to its 3x3 neighbourhood so that near-misses within the
    tolerance are still compared."""
    x0, y0 = p[0] / h, p[1] / h
    x1, y1 = q[0] / h, q[1] / h
    cx, cy = math.floor(x0), math.floor(y0)
    ex, ey = math.floor(x1), math.floor(y1)
    dx, dy = x1 - x0, y1 - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    tdx = abs(1.0 / dx) if dx else math.inf
    tdy = abs(1.0 / dy) if dy else math.inf
    tmx = ((cx + (sx > 0)) - x0) / dx if dx else math.inf
    tmy = ((cy + (sy > 0)) - y0) / dy if dy else math.inf
    path = [(cx, cy)]
    guard = abs(ex - cx) + abs(ey - cy) + 2
    while (cx, cy) != (ex, ey) and guard > 0:
        if tmx < tmy:
            cx += sx
            tmx += tdx
        else:
            cy += sy
            tmy += tdy
        path.append((cx, cy))
        guard -= 1
    cells = set()
    for gx, gy in path:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cells.add((gx + ox, gy + oy))
    return cells


def candidate_pairs(segs: list[tuple[Point, Point]], use_grid: bool = True) -> set[tuple[int, int]]:
    n = len(segs)
    if not use_grid or n < 64:
        return {(i, j) for i in range(n) for j in range(i + 1, n)}
    xs = [c for p, q in segs for c in (p[0], q[0])]
    ys = [c for p, q in segs for c in (p[1], q[1])]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    h = span / max(4.0, math.sqrt(n))
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, (p, q) in enumerate(segs):
        for cell in _grid_cells(p, q, h):
            buckets[cell].append(k)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                pairs.add((i, j) if i < j else (j, i))
    return pairs


def _split_params(segs: list[tuple[Point, Point]], eps: float, use_grid: bool) -> list[list[float]]:
    """For each segment, the parameters t in [0, 1] where it must be split."""
    params = [[0.0, 1.0] for _ in segs]
    for i, j in candidate_pairs(segs, use_grid):
        (px, py), (p2x, p2y) = segs[i]
        (qx, qy), (q2x, q2y) = segs[j]
        rx, ry = p2x - px, p2y - py
        sx, sy = q2x - qx, q2y - qy
        lr, ls = math.hypot(rx, ry), math.hypot(sx, sy)
        rxs = _cross(rx, ry, sx, sy)
        qpx, qpy = qx - px, qy - py
        if abs(rxs) <= 1e-12 * lr * ls:
            # Parallel. Only matters when collinear and overlapping: then
            # each segment is split at the other's endpoints.
            if abs(_cross(qpx, qpy, rx, ry)) / lr > eps:
                continue
            for (ex, ey) in ((qx, qy), (q2x, q2y)):
                t = ((ex - px) * rx + (ey - py) * ry) / (lr * lr)
                if 0.0 < t < 1.0:
                    params[i].append(t)
            for (ex, ey) in ((px, py), (p2x, p2y)):
                u = ((ex - qx) * sx + (ey - qy) * sy) / (ls * ls)
                if 0.0 < u < 1.0:
                    params[j].append(u)
            continue
        t = _cross(qpx, qpy, sx, sy) / rxs
        u = _cross(qpx, qpy, rx, ry) / rxs
        di, dj = eps / lr, eps / ls
        if -di <= t <= 1 + di and -dj <= u <= 1 + dj:
            params[i].append(min(1.0, max(0.0, t)))
            params[j].append(min(1.0, max(0.0, u)))
    return params


class _VertexIndex:
    """Merges points closer than eps into one vertex (grid hash lookup)."""

    def __init__(self, eps: float) -> None:
        self.eps = eps
        self.h = eps * 4 if eps > 0 else 1e-12
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.points: list[Point] = []

    def add(self, p: Point) -> int:
        gx, gy = math.floor(p[0] / self.h), math.floor(p[1] / self.h)
        best, best_d = -1, self.eps
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for k in self.cells[(gx + ox, gy + oy)]:
                    q = self.points[k]
                    d = math.hypot(p[0] - q[0], p[1] - q[1])
                    if d <= best_d:
                        best, best_d = k, d
        if best >= 0:
            return best
        self.points.append(p)
        self.cells[(gx, gy)].append(len(self.points) - 1)
        return len(self.points) - 1


def add_bbox_border(pattern: Pattern) -> Pattern:
    """Add the bounding rectangle as the paper edge, for inputs that do not
    draw one. Creases ending on it then become boundary vertices."""
    x0, y0, x1, y1 = pattern.bbox()
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    segs = pattern.segments()
    for k in range(4):
        segs.append((corners[k], corners[(k + 1) % 4], "B"))
    out = Pattern.from_segments(segs)
    out.notes = pattern.notes + ["No paper edge in the input, so its bounding rectangle was used."]
    return out


def cleanup(pattern: Pattern, tol: float = 1e-7, add_border: bool = True,
            use_grid: bool = True) -> Pattern:
    """Return a planar version of the pattern.

    tol is relative to the drawing's diagonal: points closer than
    tol * diagonal are treated as the same point.
    """
    notes = list(pattern.notes)
    if add_border and "B" not in pattern.assignments and pattern.edges:
        pattern = add_bbox_border(pattern)
        notes = list(pattern.notes)
    eps = tol * (pattern.diagonal() or 1.0)

    segs: list[tuple[Point, Point]] = []
    kinds: list[str] = []
    dropped = 0
    for p, q, a in pattern.segments():
        if math.hypot(q[0] - p[0], q[1] - p[1]) <= eps:
            dropped += 1
            continue
        segs.append((p, q))
        kinds.append(a)

    params = _split_params(segs, eps, use_grid)

    index = _VertexIndex(eps)
    best: dict[tuple[int, int], str] = {}
    conflicts = 0
    for (p, q), a, ts in zip(segs, kinds, params):
        ts = sorted(set(ts))
        pts = [index.add((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))) for t in ts]
        for u, v in zip(pts, pts[1:]):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            old = best.get(key)
            if old is None:
                best[key] = a
            elif old != a:
                if {old, a} == {"M", "V"}:
                    best[key] = "U"
                    conflicts += 1
                elif _RANK[a] > _RANK[old]:
                    best[key] = a

    edges = sorted(best)
    out = Pattern(list(index.points), edges, [best[e] for e in edges])
    if dropped:
        notes.append(f"Dropped {dropped} zero-length line(s).")
    if conflicts:
        notes.append(f"{conflicts} crease(s) were drawn as both mountain and valley; marked unassigned.")
    out.notes = notes
    return out
