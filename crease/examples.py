"""Built-in patterns, plus a generator of random flat-foldable vertices.

The examples are constructed from geometry rather than typed in, so every
coordinate is exact. Their mountain/valley assignments are derived in the
comments and verified by tests/test_examples.py.
"""

from __future__ import annotations

import math
import random

from .model import Pattern, Point


def ray_to_box(p: Point, angle_deg: float, box=(0.0, 0.0, 1.0, 1.0)) -> Point:
    """Where a ray from p (inside the box) leaves the box."""
    x0, y0, x1, y1 = box
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    ts = []
    if dx > 1e-12:
        ts.append((x1 - p[0]) / dx)
    if dx < -1e-12:
        ts.append((x0 - p[0]) / dx)
    if dy > 1e-12:
        ts.append((y1 - p[1]) / dy)
    if dy < -1e-12:
        ts.append((y0 - p[1]) / dy)
    t = min(ts)
    x, y = p[0] + t * dx, p[1] + t * dy
    # Snap tiny float error onto the edge itself.
    x = min(max(x, x0), x1)
    y = min(max(y, y0), y1)
    return (x, y)


def _square_border() -> list[tuple[Point, Point, str]]:
    c = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    return [(c[k], c[(k + 1) % 4], "B") for k in range(4)]


def waterbomb() -> Pattern:
    """Both diagonals valley, horizontal midline mountain.

    The centre has six creases (4 V, 2 M: Maekawa holds) at 0, 45, 135,
    180, 225, 315 degrees, so the sectors are 45, 90, 45, 45, 90, 45 and
    alternate ones add to 180 (Kawasaki holds).
    """
    o = (0.5, 0.5)
    segs = _square_border()
    segs += [(o, (0.0, 0.0), "V"), (o, (1.0, 1.0), "V"),
             (o, (1.0, 0.0), "V"), (o, (0.0, 1.0), "V"),
             (o, (0.0, 0.5), "M"), (o, (1.0, 0.5), "M")]
    return Pattern.from_segments(segs)


def square_twist(h: float = 0.15) -> Pattern:
    """A central square with a pleat leaving each corner.

    At the bottom-left corner the creases point at 0 and 90 degrees (the
    square's sides) and 225 and 315 (pleats). Sectors 90, 135, 90, 45:
    Kawasaki holds. The 45-degree sector is a strict minimum, so its two
    creases (315 and 0) must disagree; crimping it leaves 135/135 with the
    creases at 90 and 225, which must agree. So: sides valley, the 225
    pleat valley, the 315 pleat mountain. The other corners are the same
    rotated by 90 degrees, and each pleat pairs a mountain with a parallel
    valley, as a real pleat does.
    """
    c = 0.5
    corners = [(c - h, c - h), (c + h, c - h), (c + h, c + h), (c - h, c + h)]
    segs = _square_border()
    for k in range(4):
        segs.append((corners[k], corners[(k + 1) % 4], "V"))
    for k, p in enumerate(corners):
        base = 90.0 * k
        segs.append((p, ray_to_box(p, base + 225.0), "V"))
        segs.append((p, ray_to_box(p, base + 315.0), "M"))
    return Pattern.from_segments(segs)


def miura_ori(m: int = 6, n: int = 6, lean: float | None = None) -> Pattern:
    """A Miura-ori grid on a square sheet.

    Horizontal creases are straight; the vertical ones zigzag, leaning
    right at even rows and left at odd rows. At each vertex the sectors are
    t, 180-t, 180-t, t, and the horizontal crease between the two small
    sectors is the odd one out (crimp it and the rest must agree). That
    makes zigzag column i mountain for even i and valley for odd i, and the
    horizontal segment between columns i and i+1 take column i+1's
    assignment on even rows and column i's on odd rows.
    """
    a, b = 1.0 / m, 1.0 / n
    if lean is None:
        # Zigzags at 67.5 degrees, on the 22.5-degree grid. Must stay below
        # a, or neighbouring zigzags would cross.
        lean = b * math.tan(math.radians(22.5))

    def col(i: int) -> str:
        return "M" if i % 2 == 0 else "V"

    def x_at(i: int, j: int) -> float:
        return i * a + (lean if j % 2 else 0.0)

    segs = _square_border()
    for i in range(1, m):
        for j in range(n):
            segs.append(((x_at(i, j), j * b), (x_at(i, j + 1), (j + 1) * b), col(i)))
    for j in range(1, n):
        xs = [0.0] + [x_at(i, j) for i in range(1, m)] + [1.0]
        for i in range(m):
            kind = col(i + 1) if j % 2 == 0 else col(i)
            segs.append(((xs[i], j * b), (xs[i + 1], j * b), kind))
    return Pattern.from_segments(segs)


def broken_twist() -> Pattern:
    """The square twist with two deliberate mistakes, for the demo.

    1. The bottom-left corner's mountain pleat is rotated 6 degrees, so
       Kawasaki fails there.
    2. The top-right corner's mountain pleat is flipped to valley, so
       Maekawa fails there (4 valleys, 0 mountains).
    """
    h, c = 0.15, 0.5
    corners = [(c - h, c - h), (c + h, c - h), (c + h, c + h), (c - h, c + h)]
    segs = _square_border()
    for k in range(4):
        segs.append((corners[k], corners[(k + 1) % 4], "V"))
    for k, p in enumerate(corners):
        base = 90.0 * k
        segs.append((p, ray_to_box(p, base + 225.0), "V"))
        tilt = -6.0 if k == 0 else 0.0
        kind = "V" if k == 2 else "M"
        segs.append((p, ray_to_box(p, base + 315.0 + tilt), kind))
    return Pattern.from_segments(segs)


EXAMPLES = {
    "broken-twist": ("Broken twist", broken_twist),
    "square-twist": ("Square twist", square_twist),
    "waterbomb": ("Waterbomb", waterbomb),
    "miura-ori": ("Miura-ori", miura_ori),
}


# ----------------------------------------------------------------------
# Random flat-foldable vertices, built by running the crimp test backwards.

def random_vertex(rng: random.Random, crimps: int, grid: float | None = None,
                  min_sector: float = 4.0, tries: int = 200) -> tuple[list[float], list[str]]:
    """Return (crease angles, assignments) of a vertex that folds flat.

    Start from a straight fold (two collinear creases that agree) and
    "un-crimp" repeatedly: replace a sector m by three sectors a, t, b with
    t <= a, t <= b and a - t + b = m, separated by two new creases that
    disagree. Crimping them back recovers the straight fold, so the result
    is flat-foldable by construction. With grid set, every angle is a
    multiple of it (e.g. 22.5 for classic origami).
    """
    for _ in range(tries):
        if grid:
            ts = [grid * rng.choice((1, 1, 2)) for _ in range(crimps)]
        else:
            ts = [rng.uniform(min_sector, 40.0) for _ in range(crimps)]
        start = 180.0 - sum(ts)
        if start < max(ts, default=0.0) + (grid or min_sector):
            continue
        first = rng.choice("MV")
        sectors, assigns = [start, start], [first, first]
        ok = True
        for t in ts:
            room = [i for i, s in enumerate(sectors) if s >= t + (grid or min_sector) - 1e-9]
            if not room:
                ok = False
                break
            i = rng.choice(room)
            m = sectors[i]
            if grid:
                steps = int(round((m - t) / grid))
                choices = [t + grid * k for k in range(steps + 1)]
                choices = [c for c in choices if c >= t and m - c + t >= t]
                a = rng.choice(choices)
            else:
                a = rng.uniform(t + 0.25 * min_sector, m - 0.25 * min_sector)
            b = m - a + t
            x = rng.choice("MV")
            y = "V" if x == "M" else "M"
            sectors[i:i + 1] = [a, t, b]
            assigns[i + 1:i + 1] = [x, y]
        if not ok or min(sectors) < (grid or min_sector) - 1e-9:
            continue
        offset = rng.choice([grid * k for k in range(int(360 / grid))]) if grid else rng.uniform(0, 360)
        angles, acc = [], offset
        for s in sectors:
            angles.append(acc % 360.0)
            acc += s
        order = sorted(range(len(angles)), key=lambda k: angles[k])
        return [angles[k] for k in order], [assigns[k] for k in order]
    raise RuntimeError("could not build a vertex with these settings")


def vertex_pattern(angles: list[float], assigns: list[str],
                   centre: Point = (0.5, 0.5)) -> Pattern:
    segs = _square_border()
    for ang, a in zip(angles, assigns):
        segs.append((centre, ray_to_box(centre, ang), a))
    return Pattern.from_segments(segs)
