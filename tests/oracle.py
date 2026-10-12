"""A slow, independent answer to "does this vertex fold flat?".

Used only by the tests, to check vertex.crimp_foldable without trusting
the theory it is built on.

Folding a single vertex flat is a 1-D problem: walk around the vertex and
each sector lands on a line, alternating direction at every crease. What
remains is choosing a stacking order for the sectors (which is on top).
This tries every order and accepts the vertex if one of them

1. matches every crease's mountain/valley direction, and
2. never makes paper pass through paper. Two kinds of collision:
   * a sector sitting between the two halves of a fold whose crease it
     runs across (taco-tortilla), and
   * two folds at the same spot whose layers interleave (taco-taco).

That is (n)! orders for n sectors, so only small vertices are feasible.
"""

from __future__ import annotations

import itertools


def oracle_foldable(sectors: list[float], assigns: list[str], tol: float = 1e-6) -> bool:
    n = len(sectors)
    if n < 2 or n % 2:
        return False
    if abs(sum(sectors[0::2]) - 180.0) > tol:
        return False

    # Sector k occupies [lo_k, hi_k]; crease k (between sectors k-1 and k)
    # sits at position pos[k] with both sectors on side opens[k].
    pos = [0.0]
    for k in range(n - 1):
        pos.append(pos[-1] + (1 if k % 2 == 0 else -1) * sectors[k])
    orient = [1 if k % 2 == 0 else -1 for k in range(n)]
    lo = [min(pos[k], pos[k] + orient[k] * sectors[k]) for k in range(n)]
    hi = [max(pos[k], pos[k] + orient[k] * sectors[k]) for k in range(n)]
    opens = [orient[k] for k in range(n)]  # sector k leaves crease k in direction orient[k]

    def r(x: float) -> float:
        return round(x, 6)

    creases = list(range(n))  # crease k joins sectors (k-1) % n and k

    def ok(h: tuple[int, ...]) -> bool:
        # 1. Mountain/valley: a valley at crease k puts sector k on the
        # front face of sector k-1. Sector k-1's front faces up when its
        # orientation is +1.
        for k in creases:
            prev = (k - 1) % n
            above = h[k] > h[prev]
            front_up = orient[prev] == 1
            if above != ((assigns[k] == "V") == front_up):
                return False
        # 2a. Taco-tortilla.
        for k in creases:
            prev = (k - 1) % n
            x = pos[k]
            a, b = sorted((h[prev], h[k]))
            for m in range(n):
                if m in (prev, k):
                    continue
                if lo[m] + tol < x < hi[m] - tol and a < h[m] < b:
                    return False
        # 2b. Taco-taco.
        for k1, k2 in itertools.combinations(creases, 2):
            if r(pos[k1]) != r(pos[k2]) or opens[k1] != opens[k2]:
                continue
            a1, b1 = sorted((h[(k1 - 1) % n], h[k1]))
            a2, b2 = sorted((h[(k2 - 1) % n], h[k2]))
            if a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1:
                return False
        return True

    return any(ok(h) for h in itertools.permutations(range(n)))
