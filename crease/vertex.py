"""Flat-foldability of a single vertex.

Everything here works on one vertex described by its creases in
counter-clockwise order: their assignments, and the sector angles between
them. Sector i is the wedge of paper from crease i to crease i + 1.

Three classic necessary conditions:

* Maekawa: #mountain - #valley = +-2.
* Kawasaki: alternate sectors add up to 180 degrees.
* Big-little-big: if a sector is strictly smaller than both neighbours,
  the two creases around it must be one mountain and one valley.

None of them is sufficient on its own. The exact test is the crimp
reduction (Hull, "The combinatorics of flat folds"): pick a sector no
bigger than its neighbours whose two creases disagree, fold it away
(a crimp), and repeat. The vertex folds flat exactly when this reaches a
single straight fold. tests/test_vertex.py checks it against a
brute-force search over every layer order.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


def angle_of(dx: float, dy: float) -> float:
    """Direction in degrees, 0 <= a < 360."""
    a = math.degrees(math.atan2(dy, dx))
    return a + 360.0 if a < 0 else a


def sectors_from_angles(angles: list[float]) -> list[float]:
    """Angles must be sorted ascending."""
    n = len(angles)
    return [((angles[(i + 1) % n] - angles[i]) % 360.0) or 360.0 for i in range(n)] if n > 1 else [360.0]


def kawasaki_error(sectors: list[float]) -> float:
    """Sum of the even-numbered sectors minus 180. Zero when Kawasaki holds."""
    return sum(sectors[0::2]) - 180.0


def maekawa_counts(assigns: list[str]) -> tuple[int, int]:
    return assigns.count("M"), assigns.count("V")


def maekawa_ok(assigns: list[str]) -> bool:
    m, v = maekawa_counts(assigns)
    return abs(m - v) == 2


def big_little_big(sectors: list[float], assigns: list[str], tol: float) -> list[int]:
    """Indices of strict local-minimum sectors whose two creases agree."""
    n = len(sectors)
    bad = []
    if n < 4:
        return bad
    for i in range(n):
        s = sectors[i]
        if s < sectors[i - 1] - tol and s < sectors[(i + 1) % n] - tol:
            if assigns[i] == assigns[(i + 1) % n]:
                bad.append(i)
    return bad


def crimp_foldable(sectors: list[float], assigns: list[str], tol: float = 1e-6) -> bool:
    """Exact local flat-foldability test (Kawasaki is checked here too)."""
    n = len(sectors)
    if n < 2 or n % 2:
        return False
    if abs(kawasaki_error(sectors)) > tol:
        return False
    s, a = list(sectors), list(assigns)
    while len(s) > 2:
        n = len(s)
        for i in sorted(range(n), key=lambda k: s[k]):
            if (s[i] <= s[i - 1] + tol and s[i] <= s[(i + 1) % n] + tol
                    and a[i] != a[(i + 1) % n]):
                # Rotate so the crimped sector sits at index 1, then fold it
                # away: creases 1 and 2 vanish, sectors 0..2 merge.
                k = (i - 1) % n
                s, a = s[k:] + s[:k], a[k:] + a[:k]
                s = [s[0] - s[1] + s[2]] + s[3:]
                a = [a[0]] + a[3:]
                break
        else:
            return False
    return abs(s[0] - s[1]) <= 2 * tol and a[0] == a[1]


@dataclass
class VertexVerdict:
    kawasaki_error: float
    mountains: int
    valleys: int
    blb_sectors: list[int]
    foldable: bool

    @property
    def kawasaki_ok(self) -> bool:
        return self.kawasaki_error == 0.0


def judge(sectors: list[float], assigns: list[str], tol: float) -> VertexVerdict:
    err = kawasaki_error(sectors)
    m, v = maekawa_counts(assigns)
    unknown = "U" in assigns
    blb = [] if unknown else big_little_big(sectors, assigns, tol)
    foldable = (not unknown) and len(sectors) % 2 == 0 and crimp_foldable(sectors, assigns, tol)
    return VertexVerdict(err if abs(err) > tol else 0.0, m, v, blb, foldable)


def min_flips(sectors: list[float], assigns: list[str], tol: float,
              cost=None, max_flips: int = 3) -> list[int] | None:
    """Smallest set of creases to flip (M<->V) that makes the vertex fold.

    cost(i) breaks ties between equally small sets, e.g. preferring flips
    that do not break the vertex at the crease's other end. Returns None if
    no set of up to max_flips flips works (or Kawasaki fails, in which case
    no flip can help).
    """
    if abs(kawasaki_error(sectors)) > tol or "U" in assigns:
        return None
    n = len(assigns)
    flip = {"M": "V", "V": "M"}
    for size in range(1, min(max_flips, n) + 1):
        found = []
        for combo in itertools.combinations(range(n), size):
            trial = list(assigns)
            for i in combo:
                trial[i] = flip[trial[i]]
            if crimp_foldable(sectors, trial, tol):
                found.append(combo)
        if found:
            if cost is not None:
                found.sort(key=lambda c: sum(cost(i) for i in c))
            return list(found[0])
    return None


def rotation_fix(sectors: list[float], err: float) -> list[tuple[int, float]]:
    """Ways to fix Kawasaki by rotating a single crease.

    Rotating crease k counter-clockwise by d grows sector k-1 and shrinks
    sector k by d. Those sectors have opposite parity, so the even-sector
    sum moves by +d or -d; choosing d = -err or +err cancels the error.
    Returns (crease index, signed degrees CCW) for every crease where the
    rotation keeps both neighbouring sectors positive.
    """
    n = len(sectors)
    out = []
    for k in range(n):
        d = -err if (k - 1) % 2 == 0 else err
        grow, shrink = sectors[k - 1] + d, sectors[k] - d
        if grow > 1e-9 and shrink > 1e-9:
            out.append((k, d))
    return out
