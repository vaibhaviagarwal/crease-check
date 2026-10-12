"""The one data structure everything else passes around.

A crease pattern is a planar drawing: points (vertices) joined by straight
lines (edges). Each edge has an assignment, using the letters of the FOLD
file format:

    M  mountain fold          V  valley fold
    B  boundary (paper edge)  F  flat (drawn but not folded)
    U  unassigned (a fold whose direction is unknown)

Coordinates are plain (x, y) floats. Nothing here assumes a unit square;
io.normalize() rescales when that is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ASSIGNMENTS = {"M", "V", "B", "F", "U"}
FOLDS = {"M", "V", "U"}  # edges that bend the paper at a vertex

Point = tuple[float, float]


@dataclass
class Pattern:
    vertices: list[Point]
    edges: list[tuple[int, int]]
    assignments: list[str]
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.assignments):
            raise ValueError(
                f"{len(self.edges)} edges but {len(self.assignments)} assignments"
            )
        n = len(self.vertices)
        for k, (i, j) in enumerate(self.edges):
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"edge {k} points at a vertex that does not exist")
        bad = set(self.assignments) - ASSIGNMENTS
        if bad:
            raise ValueError(f"unknown assignment(s): {sorted(bad)}")

    # ------------------------------------------------------------------
    @classmethod
    def from_segments(cls, segments: list[tuple[Point, Point, str]]) -> "Pattern":
        """Build from loose line segments. Shared endpoints are merged later
        by graph.cleanup(); here every endpoint is stored as given."""
        vertices: list[Point] = []
        edges: list[tuple[int, int]] = []
        assignments: list[str] = []
        for p, q, a in segments:
            vertices.append((float(p[0]), float(p[1])))
            vertices.append((float(q[0]), float(q[1])))
            edges.append((len(vertices) - 2, len(vertices) - 1))
            assignments.append(a)
        return cls(vertices, edges, assignments)

    def segments(self) -> list[tuple[Point, Point, str]]:
        return [
            (self.vertices[i], self.vertices[j], a)
            for (i, j), a in zip(self.edges, self.assignments)
        ]

    def bbox(self) -> tuple[float, float, float, float]:
        if not self.vertices:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    def diagonal(self) -> float:
        x0, y0, x1, y1 = self.bbox()
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def count(self, assignment: str) -> int:
        return sum(1 for a in self.assignments if a == assignment)
