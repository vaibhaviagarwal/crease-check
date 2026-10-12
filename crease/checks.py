"""Check a whole crease pattern, vertex by vertex, and suggest fixes.

Every interior vertex (one not on the paper's edge) must fold flat on its
own. That is "local" flat-foldability. Passing every local check does not
prove the whole sheet folds flat: deciding that is NP-hard (Bern and
Hayes, 1996), because distant flaps can collide. So the report says
"locally flat-foldable" and means exactly that.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from . import vertex as vx
from .graph import cleanup
from .model import FOLDS, Pattern

NAMES = {"M": "mountain", "V": "valley"}


@dataclass
class Violation:
    rule: str
    vertex: int
    point: tuple[float, float]
    message: str
    details: dict = field(default_factory=dict)
    fix: dict | None = None


@dataclass
class Report:
    flat_foldable: bool
    vertices_checked: int
    creases: int
    violations: list[Violation]
    notes: list[str]
    elapsed_ms: float
    pattern: Pattern

    @property
    def failing_vertices(self) -> list[int]:
        return sorted({v.vertex for v in self.violations})

    def to_dict(self) -> dict:
        return {
            "flat_foldable": self.flat_foldable,
            "scope": "local",
            "checked": {"vertices": self.vertices_checked, "creases": self.creases},
            "failing_vertices": len(self.failing_vertices),
            "violations": [asdict(v) for v in self.violations],
            "notes": self.notes,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def crease_id(k: int) -> str:
    return f"C{k + 1}"


def _stars(p: Pattern) -> dict[int, list[tuple[float, int, int]]]:
    """For each vertex, its fold edges as (angle, edge index, other vertex),
    sorted counter-clockwise."""
    stars: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for k, ((i, j), a) in enumerate(zip(p.edges, p.assignments)):
        if a not in FOLDS:
            continue
        (xi, yi), (xj, yj) = p.vertices[i], p.vertices[j]
        stars[i].append((vx.angle_of(xj - xi, yj - yi), k, j))
        stars[j].append((vx.angle_of(xi - xj, yi - yj), k, i))
    for v in stars:
        stars[v].sort()
    return stars


def _sectors_and_assigns(star, assignments, override=None):
    sectors = vx.sectors_from_angles([s[0] for s in star])
    assigns = [override.get(s[1], assignments[s[1]]) if override else assignments[s[1]] for s in star]
    return sectors, assigns


def check(pattern: Pattern, angle_tol: float = 0.01, clean: bool = True,
          merge_tol: float = 1e-7) -> Report:
    """angle_tol is in degrees. Use ~0.01 for files, ~1 for photos."""
    t0 = time.perf_counter()
    p = cleanup(pattern, tol=merge_tol) if clean else pattern
    boundary = {v for (i, j), a in zip(p.edges, p.assignments) if a == "B" for v in (i, j)}
    stars = _stars(p)
    interior = sorted(v for v in stars if v not in boundary)

    def foldable_at(v: int, override: dict | None = None) -> bool:
        if v in boundary or v not in stars:
            return True
        s, a = _sectors_and_assigns(stars[v], p.assignments, override)
        return "U" not in a and vx.crimp_foldable(s, a, angle_tol)

    violations: list[Violation] = []
    for v in interior:
        star = stars[v]
        pt = p.vertices[v]
        sectors, assigns = _sectors_and_assigns(star, p.assignments)
        n = len(star)

        if n % 2:
            violations.append(Violation(
                "even_degree", v, pt,
                f"{n} creases meet here. A point that folds flat needs an even number.",
                {"degree": n}))
            continue

        verdict = vx.judge(sectors, assigns, angle_tol)

        if verdict.kawasaki_error:
            total = 180.0 + verdict.kawasaki_error
            options = vx.rotation_fix(sectors, verdict.kawasaki_error)
            fix = None
            if options:
                def rank(opt):
                    k, _ = opt
                    _, edge, other = star[k]
                    (x0, y0), (x1, y1) = p.vertices[v], p.vertices[other]
                    # Prefer creases whose far end is on the paper edge (it
                    # can slide along the edge), then the shortest crease.
                    return (other not in boundary, math.hypot(x1 - x0, y1 - y0))
                k, d = min(options, key=rank)
                edge = star[k][1]
                fix = {"rotate": crease_id(edge), "by_deg": round(d, 2)}
            violations.append(Violation(
                "kawasaki", v, pt,
                f"Every other angle should add up to 180°. These add up to {total:.1f}°.",
                {"alternating_sum_deg": round(total, 3), "sectors_deg": [round(s, 3) for s in sectors]},
                fix))

        if "U" in assigns:
            violations.append(Violation(
                "unassigned", v, pt,
                f"{assigns.count('U')} crease(s) here have no mountain/valley direction, so the folds cannot be checked.",
                {"unassigned": assigns.count("U")}))
            continue

        mv_problems: list[Violation] = []
        if not vx.maekawa_ok(assigns):
            mv_problems.append(Violation(
                "maekawa", v, pt,
                f"{verdict.mountains} mountain and {verdict.valleys} valley folds meet here. They need to differ by exactly 2.",
                {"mountain": verdict.mountains, "valley": verdict.valleys}))
        for si in verdict.blb_sectors:
            c1, c2 = star[si][1], star[(si + 1) % n][1]
            mv_problems.append(Violation(
                "big_little_big", v, pt,
                f"The smallest angle here ({sectors[si]:.1f}°) sits between two {NAMES[assigns[si]]} folds. "
                "Its two creases must be one mountain and one valley.",
                {"sector_deg": round(sectors[si], 3), "creases": [crease_id(c1), crease_id(c2)]}))
        if not verdict.kawasaki_error and not mv_problems and not verdict.foldable:
            mv_problems.append(Violation(
                "layers", v, pt,
                "Passes Maekawa, Kawasaki and big-little-big, but the layers would still have to pass through each other.",
                {}))

        if mv_problems and not verdict.kawasaki_error:
            def cost(i: int) -> int:
                _, edge, other = star[i]
                flipped = {edge: {"M": "V", "V": "M"}[p.assignments[edge]]}
                return int(foldable_at(other) and not foldable_at(other, flipped))
            flips = vx.min_flips(sectors, assigns, angle_tol, cost=cost)
            if flips:
                edges = [star[i][1] for i in flips]
                mv_problems[0].fix = {
                    "flip": [crease_id(e) for e in edges],
                    "to": [{"M": "V", "V": "M"}[p.assignments[e]] for e in edges],
                }
        violations.extend(mv_problems)

    folds = sum(1 for a in p.assignments if a in FOLDS)
    return Report(
        flat_foldable=not violations and folds > 0,
        vertices_checked=len(interior),
        creases=folds,
        violations=violations,
        notes=list(p.notes),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        pattern=p,
    )


def apply_fix(pattern: Pattern, violation: Violation) -> Pattern:
    """Return a copy of the (cleaned) pattern with the violation's fix applied.

    Flips change assignments. A rotation turns the crease about the failing
    vertex and re-extends it to the paper edge.
    """
    fix = violation.fix
    if not fix:
        return pattern
    if "flip" in fix:
        assigns = list(pattern.assignments)
        for cid, to in zip(fix["flip"], fix["to"]):
            assigns[int(cid[1:]) - 1] = to
        return Pattern(list(pattern.vertices), list(pattern.edges), assigns)
    if "rotate" in fix:
        k = int(fix["rotate"][1:]) - 1
        i, j = pattern.edges[k]
        pivot = violation.vertex
        other = j if i == pivot else i
        (px, py), (ox, oy) = pattern.vertices[pivot], pattern.vertices[other]
        ang = math.radians(vx.angle_of(ox - px, oy - py) + fix["by_deg"])
        dx, dy = math.cos(ang), math.sin(ang)
        best = None
        for (a, b), kind in zip(pattern.edges, pattern.assignments):
            if kind != "B":
                continue
            (ax, ay), (bx, by) = pattern.vertices[a], pattern.vertices[b]
            sx, sy = bx - ax, by - ay
            den = dx * sy - dy * sx
            if abs(den) < 1e-12:
                continue
            t = ((ax - px) * sy - (ay - py) * sx) / den
            u = ((ax - px) * dy - (ay - py) * dx) / den
            if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9 and (best is None or t < best):
                best = t
        if best is None:
            return pattern
        segs = pattern.segments()
        segs[k] = ((px, py), (px + best * dx, py + best * dy), pattern.assignments[k])
        return Pattern.from_segments(segs)
    return pattern


def describe_fix(fix: dict | None) -> str | None:
    if not fix:
        return None
    if "rotate" in fix:
        d = fix["by_deg"]
        way = "counter-clockwise" if d > 0 else "clockwise"
        return f"Rotate crease {fix['rotate']} by {abs(d):.1f}° {way}"
    if "flip" in fix:
        parts = [f"{c} to {NAMES[t]}" for c, t in zip(fix["flip"], fix["to"])]
        return "Flip " + ", ".join(parts)
    return None
